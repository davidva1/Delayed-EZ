import os
import cv2
import gym
import torch
import random
import shutil
import logging

import numpy as np
from collections import deque, Counter

from scipy.stats import entropy
import core.ctree.cytree as cytree
from core.mcts import MCTS
from torch.cuda.amp import autocast as autocast

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def get_max_episode_steps(self):
        return self._max_episode_steps

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip
        self.max_frame = np.zeros(env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        self.max_frame = self._obs_buffer.max(axis=0)

        return self.max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode='human', **kwargs):
        img = self.max_frame
        img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA).astype(np.uint8)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs

class DelayWrapper(gym.Env):
    def __init__(self, env, max_delay_value, p=0.2):
        self.orig_env = env

        # Delay
        self.max_delay_value = max_delay_value
        self.current_delay_value = max_delay_value
        self.pending_actions = deque()
        self.action_delays = deque()
        self.underlying_delays = deque()
        self.p = p

        self.stored_init_state = None
        self.is_atari_env = True
        self.action_space = self.orig_env.action_space
        self.observation_space = self.orig_env.observation_space

    def find_executed_action(self, ix=0, delay_queue=None):
        """
        Returns the expected executed action ix steps forward.
        ----------
        :param ix: finds the action executed in ix steps according to the current action and delay queues:
        :param delay_queue: if given, each action has its delay written in delay queue. used for sampling different delay queues (robust criterion for unobserved delays)
        :return: pending actions in the environment engine
        """
        if not delay_queue:
            delay_queue = self.action_delays
        executed_action = self.pending_actions[0]
        for i, delay in enumerate(delay_queue):
            execution_t = delay + (i - self.max_delay_value)
            if execution_t <= ix:
                executed_action = self.pending_actions[i]  # Take the latest action which has excution time <=0
                # TODO: check if latest action is at self.pending_actions[0] or self.pending_actions[max_delay_value]

        return executed_action

    def step(self, action):
        if self.max_delay_value > 0:
            self.pending_actions.append(action)
            self.action_delays.append(self.current_delay_value)
            if len(self.pending_actions) - 1 >= self.max_delay_value:
                executed_action = self.find_executed_action() #self.pending_actions.popleft()
            else:
                executed_action = np.random.choice(self.action_space.n)
                #TODO: initialize the action queue if no use_forward

            self.pending_actions.popleft()
            self.action_delays.popleft()
            self.underlying_delays.popleft()
        else:
            executed_action = action
        observation, reward, done, info = self.orig_env.step(executed_action)
        info['executed_action'] = executed_action
        return observation, reward, done, info

    def reset(self):
        self.pending_actions.clear()
        self.action_delays.clear()
        self.underlying_delays.clear()
        self.current_delay_value    = self.max_delay_value
        return self.orig_env.reset()

    def set_pending_actions(self, model=None, stack_obs=None, config=None):
        if model is None:
            print('set_action_queue, None')
            self.pending_actions    = deque(np.random.randint(0, self.action_space.n, self.max_delay_value))
            self.action_delays      = deque([self.max_delay_value] * self.max_delay_value)
            self.underlying_delays  = deque([self.max_delay_value] * self.max_delay_value)
        else:
            assert stack_obs is not None and config is not None, "Providing model to find first actions but not stack_obs or config"
            # Use model to find m first actions
            model.eval()
            stack_obs = stack_obs.to(config.device).unsqueeze(0)
            if config.amp_type == 'torch_amp':
                with autocast():
                    network_output = model.initial_inference(stack_obs.float())
            else:
                network_output = model.initial_inference(stack_obs.float())
            action_i = search_action(model, config, network_output)
            action_i = np.asarray(action_i)
            # action_i = np.asarray([action_i for env_ix in range(env_nums)])
            self.pending_actions.append(action_i[0])
            self.action_delays.append(self.current_delay_value)
            self.underlying_delays.append(self.current_delay_value)

            for i in range(config.delay - 1): # a_0, ..., a_{m-1}
                # action_i = np.asarray([actions[env_ix][i] for env_ix in range(env_nums)])
                action_i = torch.from_numpy(action_i).to(config.device).unsqueeze(1).long()

                network_output = model_next_state(model, config, network_output, action_i)

                action_i = search_action(model, config, network_output)
                action_i = np.asarray(action_i)
                self.pending_actions.append(action_i[0])
                self.action_delays.append(config.delay)
                self.underlying_delays.append(config.delay)

    def initialize_pending_actions(self, model=None, stack_obs=None, config=None):
        """
        Sets the pending actions.
        ----------
        :param model:
        :param stack_obs:
        :param config:
        :return: pending actions in the environment engine
        """
        if len(self.pending_actions) == 0 and self.max_delay_value > 0:
            self.set_pending_actions(model, stack_obs, config)

    def get_pending_actions_for_agent(self, current_delay=None, delay_queue=None):
        """
        Returns the pending actions in the environment from the perspective of the agent until the current delay value.
        If the pending actions are empty, uses the model and the
        observation to infer the first m actions.
        If config.distributed_delay == True, the agent does not fully observe the delay process. In that case,
        the agent observes the sampled delay values and can only receive the expected pending actions
        ----------
        :return: pending actions in the environment engine from the agent's perspective
        """

        if current_delay is None:
            current_delay = self.underlying_delay_value

        effective_pending_actions = []
        for ix in range(current_delay):
            effective_pending_actions.append(self.find_executed_action(ix=ix, delay_queue=delay_queue))

        return effective_pending_actions

    def get_max_episode_steps(self):
        return self.orig_env.get_max_episode_steps()

    def set_current_delay(self, underlying_delay_value):
        self.current_delay_value = underlying_delay_value


def make_atari(env_id, skip=4, max_episode_steps=None):
    """Make Atari games
    Parameters
    ----------
    env_id: str
        name of environment
    skip: int
        frame skip
    max_episode_steps: int
        max moves for an episode
    delay: int
        delay value
    """
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=skip)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def set_seed(seed):
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def make_results_dir(exp_path, args):
    # make the result directory
    os.makedirs(exp_path, exist_ok=True)
    if args.opr == 'train' and os.path.exists(exp_path) and os.listdir(exp_path):
        if not args.force:
            raise FileExistsError('{} is not empty. Please use --force to overwrite it'.format(exp_path))
        else:
            print('Warning, path exists! Rewriting...')
            shutil.rmtree(exp_path)
            os.makedirs(exp_path)
    log_path = os.path.join(exp_path, 'logs')
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(os.path.join(exp_path, 'model'), exist_ok=True)
    return exp_path, log_path


def init_logger(base_path):
    # initialize the logger
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(filename)s>%(funcName)s] ==> %(message)s')
    for mode in ['train', 'test', 'train_test', 'root']:
        file_path = os.path.join(base_path, mode + '.log')
        logger = logging.getLogger(mode)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.FileHandler(file_path, mode='a')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)


def select_action(visit_counts, temperature=1, deterministic=True):
    """select action from the root visit counts.
    Parameters
    ----------
    temperature: float
        the temperature for the distribution
    deterministic: bool
        True -> select the argmax
        False -> sample from the distribution
    """
    action_probs = [visit_count_i ** (1 / temperature) for visit_count_i in visit_counts]
    total_count = sum(action_probs)
    action_probs = [x / total_count for x in action_probs]
    if deterministic:
        # best_actions = np.argwhere(visit_counts == np.amax(visit_counts)).flatten()
        # action_pos = np.random.choice(best_actions)
        action_pos = np.argmax([v for v in visit_counts])
    else:
        action_pos = np.random.choice(len(visit_counts), p=action_probs)

    count_entropy = entropy(action_probs, base=2)
    return action_pos, count_entropy

def search_action(model, config, network_output):
    env_nums = 1
    hidden_state_roots = network_output.hidden_state
    reward_hidden_roots = network_output.reward_hidden
    value_prefix_pool = network_output.value_prefix
    policy_logits_pool = network_output.policy_logits.tolist()

    if not isinstance(value_prefix_pool, list):
        value_prefix_pool = value_prefix_pool.reshape(-1).tolist()

    roots = cytree.Roots(env_nums, config.action_space_size, config.num_simulations)
    noises = [np.random.dirichlet([config.root_dirichlet_alpha] * config.action_space_size).astype(
        np.float32).tolist() for _ in range(env_nums)]
    roots.prepare(config.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool)
    # do MCTS for a policy
    MCTS(config).search(roots, model, hidden_state_roots, reward_hidden_roots)

    roots_distributions = roots.get_distributions()
    roots_values = roots.get_values()
    actions = []
    for i in range(env_nums):
        action, visit_entropy = select_action(roots_distributions[i])
        actions.append(action)

    return actions

def model_next_state(model, config, prev_network_output, action_i):
    hidden_state_roots = prev_network_output.hidden_state
    reward_hidden_roots = prev_network_output.reward_hidden

    hidden_states = torch.from_numpy(np.asarray(hidden_state_roots)).to(config.device).float()
    hidden_states_c_reward = torch.from_numpy(np.asarray(reward_hidden_roots[0])).to(config.device)
    hidden_states_h_reward = torch.from_numpy(np.asarray(reward_hidden_roots[1])).to(config.device)

    network_output = model.recurrent_inference(hidden_states, (hidden_states_c_reward, hidden_states_h_reward),
                                               action_i)
    return network_output

def update_envs_current_delay(envs, current_underlying_delay_value, min_delay_value, max_delay_value, p):
        # UNDERLYING DELAY
        p=0.2
        rn = np.random.rand()
        new_underlying_delay_value = current_underlying_delay_value
        if rn < p:
            new_underlying_delay_value -= 1
        elif rn > 1 - p:
            new_underlying_delay_value += 1
        new_underlying_delay_value = max(min(new_underlying_delay_value, max_delay_value), min_delay_value)

        for env in envs:
            env.set_current_delay(new_underlying_delay_value)
        return new_underlying_delay_value

def sample_delay_queue(delay_queue, max_delay_value, q):
    sampled = []
    for i in range(len(delay_queue)):
        rn = np.random.rand()
        sampled_delay_value = delay_queue[i]
        if rn < q:
            sampled_delay_value -= 1
        elif rn > 1 - q:
            sampled_delay_value += 1
        sampled_delay_value = max(min(sampled_delay_value, max_delay_value), 0)
        sampled.append(sampled_delay_value)
    return sampled

def get_env_action_from_samples(sampled_actions):
    # Return the most common action in the sampled actions
    # It could return the action that maximizes the worst predicted state.

    counter = Counter(sampled_actions)
    most_common = counter.most_common(1)
    if not most_common:
        print(f'In most_common: sampled_actions={sampled_actions}, counter={counter}')
    return most_common[0][0] if most_common else None

def prepare_observation_lst(observation_lst):
    """Prepare the observations to satisfy the input fomat of torch
    [B, S, W, H, C] -> [B, S x C, W, H]
    batch, stack num, width, height, channel
    """
    # B, S, W, H, C
    observation_lst = np.array(observation_lst, dtype=np.uint8)
    observation_lst = np.moveaxis(observation_lst, -1, 2)

    shape = observation_lst.shape
    observation_lst = observation_lst.reshape((shape[0], -1, shape[-2], shape[-1]))

    return observation_lst


def arr_to_str(arr):
    """To reduce memory usage, we choose to store the jpeg strings of image instead of the numpy array in the buffer.
    This function encodes the observation numpy arr to the jpeg strings
    """
    img_str = cv2.imencode('.jpg', arr)[1].tobytes()

    return img_str


def str_to_arr(s, gray_scale=False):
    """To reduce memory usage, we choose to store the jpeg strings of image instead of the numpy array in the buffer.
    This function decodes the observation numpy arr from the jpeg strings
    Parameters
    ----------
    s: string
        the inputs
    gray_scale: bool
        True -> the inputs observation is gray not RGB.
    """
    nparr = np.frombuffer(s, np.uint8)
    if gray_scale:
        arr = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        arr = np.expand_dims(arr, -1)
    else:
        arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return arr
