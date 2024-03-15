import matplotlib.pyplot as plt
import ray
import time
import torch

import numpy as np
import core.ctree.cytree as cytree

from torch.nn import L1Loss
from torch.cuda.amp import autocast as autocast
from core.mcts import MCTS
from core.game import GameHistory
from core.utils import select_action, prepare_observation_lst, update_envs_current_delay, sample_delay_queue, get_env_action_from_samples
from copy import deepcopy
from collections import deque


@ray.remote(num_gpus=0.125)
class DataWorker(object):
    def __init__(self, rank, replay_buffer, storage, config):
        """Data Worker for collecting data through self-play
        Parameters
        ----------
        rank: int
            id of the worker
        replay_buffer: Any
            Replay buffer
        storage: Any
            The model storage
        """
        self.rank = rank
        self.config = config
        self.storage = storage
        self.replay_buffer = replay_buffer
        # double buffering when data is sufficient
        self.trajectory_pool = []
        self.pool_size = 1
        self.device = self.config.device
        self.gap_step = self.config.num_unroll_steps + self.config.td_steps
        self.last_model_index = -1

        #stats
        self.obs_deque = deque()

        # Delay
        self.current_delay_value = self.config.delay


    def put(self, data):
        # put a game history into the pool
        game_history = data[0]
        #print('len of game after process: ', len(game_history))
        self.storage.set_delay_logs.remote(processed_eps_length=len(game_history))
        if len(game_history) == 0 or len(game_history.action_delays) == 0:
            print(f'In put(): len(game_history.actions) = {len(game_history.actions)}, '
                  f'len(game_history.action_delays) = {len(game_history.action_delays)}')

        self.trajectory_pool.append(data)
        return True

    def len_pool(self):
        # current pool size
        return len(self.trajectory_pool)

    def free(self):
        # save the game histories and clear the pool
        if self.len_pool() >= self.pool_size:
            self.replay_buffer.save_pools.remote(self.trajectory_pool, self.gap_step)
            del self.trajectory_pool[:]

    def put_last_trajectory(self, i, last_game_histories, last_game_priorities, game_histories):
        """put the last game history into the pool if the current game is finished
        Parameters
        ----------
        last_game_histories: list
            list of the last game histories
        last_game_priorities: list
            list of the last game priorities
        game_histories: list
            list of the current game histories
        """
        # pad over last block trajectory
        beg_index = self.config.stacked_observations
        end_index = beg_index + self.config.num_unroll_steps

        pad_obs_lst = game_histories[i].obs_history[beg_index:end_index]
        pad_child_visits_lst = game_histories[i].child_visits[beg_index:end_index]

        beg_index = 0
        end_index = beg_index + self.gap_step - 1

        pad_reward_lst = game_histories[i].rewards[beg_index:end_index]

        beg_index = 0
        end_index = beg_index + self.gap_step

        pad_root_values_lst = game_histories[i].root_values[beg_index:end_index]

        # pad over and save
        last_game_histories[i].pad_over(pad_obs_lst, pad_reward_lst, pad_root_values_lst, pad_child_visits_lst)
        last_game_histories[i].game_over()

        self.put((last_game_histories[i], last_game_priorities[i]))
        self.free()

        # reset last block
        last_game_histories[i] = None
        last_game_priorities[i] = None

    def get_priorities(self, i, pred_values_lst, search_values_lst):
        # obtain the priorities at index i
        if self.config.use_priority and not self.config.use_max_priority:
            pred_values = torch.from_numpy(np.array(pred_values_lst[i])).to(self.device).float()
            search_values = torch.from_numpy(np.array(search_values_lst[i])).to(self.device).float()
            priorities = L1Loss(reduction='none')(pred_values, search_values).detach().cpu().numpy() + self.config.prioritized_replay_eps
        else:
            # priorities is None -> use the max priority for all newly collected data
            priorities = None

        return priorities


    def run(self):
        # number of parallel mcts
        env_nums = self.config.p_mcts_num
        model = self.config.get_uniform_network()
        model.to(self.device)
        model.eval()
        start_training = False
        envs = [self.config.new_game(self.config.seed + (self.rank + 1) * i, delay_enable=True) for i in range(env_nums)]
        def _get_max_entropy(action_space):
            p = 1.0 / action_space
            ep = - action_space * p * np.log2(p)
            return ep
        max_visit_entropy = _get_max_entropy(self.config.action_space_size)
        # 100k benchmark
        self.total_transitions = 0
        # max transition to collect for this data worker
        max_transitions = self.config.total_transitions // self.config.num_actors
        with torch.no_grad():
            while True:
                trained_steps = ray.get(self.storage.get_counter.remote())
                # training finished
                if trained_steps >= self.config.training_steps + self.config.last_steps:
                    time.sleep(30)
                    break

                init_obses = [env.reset() for env in envs]
                dones = np.array([False for _ in range(env_nums)])
                game_histories = [GameHistory(envs[_].env.action_space, max_length=self.config.history_length,
                                              config=self.config) for _ in range(env_nums)]
                last_game_histories = [None for _ in range(env_nums)]
                last_game_priorities = [None for _ in range(env_nums)]

                # stack observation windows in boundary: s398, s399, s400, current s1 -> for not init trajectory
                stack_obs_windows = [[] for _ in range(env_nums)]

                for i in range(env_nums):
                    stack_obs_windows[i] = [init_obses[i] for _ in range(self.config.stacked_observations)]
                    game_histories[i].init(stack_obs_windows[i])

                # for priorities in self-play
                search_values_lst = [[] for _ in range(env_nums)]
                pred_values_lst = [[] for _ in range(env_nums)]

                # some logs
                eps_ori_reward_lst, eps_reward_lst, eps_steps_lst, visit_entropies_lst = np.zeros(env_nums), np.zeros(env_nums), np.zeros(env_nums), np.zeros(env_nums)
                step_counter = 0

                self_play_rewards = 0.
                self_play_ori_rewards = 0.
                self_play_moves = 0.
                self_play_episodes = 0.

                self_play_rewards_max = - np.inf
                self_play_moves_max = 0

                self_play_visit_entropy = []
                other_dist = {}

                # play games until max moves
                while not dones.all() and (step_counter <= self.config.max_moves):
                    if not start_training:
                        start_training = ray.get(self.storage.get_start_signal.remote())

                    # get model
                    trained_steps = ray.get(self.storage.get_counter.remote())
                    if trained_steps >= self.config.training_steps + self.config.last_steps:
                        # training is finished
                        time.sleep(30)
                        return
                    if start_training and (self.total_transitions / max_transitions) > (trained_steps / self.config.training_steps):
                        # self-play is faster than training speed or finished
                        time.sleep(1)
                        continue

                    # set temperature for distributions
                    _temperature = np.array(
                        [self.config.visit_softmax_temperature_fn(num_moves=0, trained_steps=trained_steps) for env in
                         envs])

                    # update the models in self-play every checkpoint_interval
                    new_model_index = trained_steps // self.config.checkpoint_interval
                    if new_model_index > self.last_model_index:
                        self.last_model_index = new_model_index
                        # update model
                        weights = ray.get(self.storage.get_weights.remote())
                        model.set_weights(weights)
                        model.to(self.device)
                        model.eval()

                        # log if more than 1 env in parallel because env will reset in this loop.
                        if env_nums > 1:
                            if len(self_play_visit_entropy) > 0:
                                visit_entropies = np.array(self_play_visit_entropy).mean()
                                visit_entropies /= max_visit_entropy
                            else:
                                visit_entropies = 0.

                            if self_play_episodes > 0:
                                log_self_play_moves = self_play_moves / self_play_episodes
                                log_self_play_rewards = self_play_rewards / self_play_episodes
                                log_self_play_ori_rewards = self_play_ori_rewards / self_play_episodes
                            else:
                                log_self_play_moves = 0
                                log_self_play_rewards = 0
                                log_self_play_ori_rewards = 0

                            self.storage.set_data_worker_logs.remote(log_self_play_moves, self_play_moves_max,
                                                                            log_self_play_ori_rewards, log_self_play_rewards,
                                                                            self_play_rewards_max, _temperature.mean(),
                                                                            visit_entropies, 0,
                                                                            other_dist)
                            self_play_rewards_max = - np.inf

                    step_counter += 1
                    for i in range(env_nums):
                        # reset env if finished
                        if dones[i]:

                            # pad over last block trajectory
                            if last_game_histories[i] is not None:
                                self.put_last_trajectory(i, last_game_histories, last_game_priorities, game_histories)

                            # store current block trajectory
                            priorities = self.get_priorities(i, pred_values_lst, search_values_lst)
                            game_histories[i].game_over()

                            self.put((game_histories[i], priorities))
                            self.free()

                            # reset the finished env and new a env
                            envs[i].close()
                            init_obs = envs[i].reset()

                            game_histories[i] = GameHistory(env.env.action_space, max_length=self.config.history_length,
                                                            config=self.config)
                            last_game_histories[i] = None
                            last_game_priorities[i] = None
                            stack_obs_windows[i] = [init_obs for _ in range(self.config.stacked_observations)]
                            game_histories[i].init(stack_obs_windows[i])

                            # log
                            self_play_rewards_max = max(self_play_rewards_max, eps_reward_lst[i])
                            self_play_moves_max = max(self_play_moves_max, eps_steps_lst[i])
                            self_play_rewards += eps_reward_lst[i]
                            self_play_ori_rewards += eps_ori_reward_lst[i]
                            self_play_visit_entropy.append(visit_entropies_lst[i] / eps_steps_lst[i])
                            self_play_moves += eps_steps_lst[i]
                            self_play_episodes += 1

                            pred_values_lst[i] = []
                            search_values_lst[i] = []
                            # end_tags[i] = False
                            eps_steps_lst[i] = 0
                            eps_reward_lst[i] = 0
                            eps_ori_reward_lst[i] = 0
                            visit_entropies_lst[i] = 0

                    # stack obs for model inference
                    stack_obs = [game_history.step_obs() for game_history in game_histories]
                    if self.config.image_based:
                        stack_obs = prepare_observation_lst(stack_obs)
                        stack_obs = torch.from_numpy(stack_obs).to(self.device).float() / 255.0
                    else:
                        stack_obs = [game_history.step_obs() for game_history in game_histories]
                        stack_obs = torch.from_numpy(np.array(stack_obs)).to(self.device)

                    if self.config.amp_type == 'torch_amp':
                        with autocast():
                            network_output = model.initial_inference(stack_obs.float())
                    else:
                        network_output = model.initial_inference(stack_obs.float())
                    hidden_state_roots = network_output.hidden_state
                    reward_hidden_roots = network_output.reward_hidden
                    value_prefix_pool = network_output.value_prefix
                    policy_logits_pool = network_output.policy_logits.tolist()
                    hidden_state_0 = hidden_state_roots


                    if self.config.test_use_forward:
                        [envs[i].initialize_pending_actions(model, stack_obs[i], self.config) for i in range(env_nums)]
                    else:
                        # If test_use_forward is False, we don't plan the first M actions from stack_obs
                        [envs[i].initialize_pending_actions() for i in range(env_nums)]

                    self.current_delay_value = update_envs_current_delay(envs, self.current_delay_value,
                                                                         self.config.min_delay,
                                                                         self.config.delay, p=0.2)

                    # Useful when several actions queues are possible and we want to predict action for each one.
                    action_queues = [[envs[i].get_pending_actions_for_agent() for i in range(env_nums)]]

                    policy_actions_per_env = [[] for _ in range(env_nums)]
                    for actions_ix, actions in enumerate(action_queues):
                        if self.config.delay and self.config.test_use_forward:
                            #TODO : check if need to use model (which predicts actions based on stack_obs )
                            # in the case of Oblivious EZ . We may want to use the random action queue if test_use_forward == False
                            for i in range(self.current_delay_value):
                                action_i = np.asarray([actions[env_ix][i] for env_ix in range(env_nums)])
                                action_i = torch.from_numpy(action_i).to(self.device).unsqueeze(1).long()

                                hidden_states = torch.from_numpy(np.asarray(hidden_state_roots)).to(self.device).squeeze(
                                    0).float()
                                hidden_states_c_reward = torch.from_numpy(np.asarray(reward_hidden_roots[0])).to(self.device)
                                hidden_states_h_reward = torch.from_numpy(np.asarray(reward_hidden_roots[1])).to(self.device)

                                network_output = model.recurrent_inference(hidden_states,
                                                                           (hidden_states_c_reward, hidden_states_h_reward),
                                                                           action_i)
                                hidden_state_roots = network_output.hidden_state
                                reward_hidden_roots = network_output.reward_hidden
                                # value_prefix_pool = network_output.value_prefix
                                policy_logits_pool = network_output.policy_logits.tolist()
                            value_prefix_pool = [0. for _ in range(env_nums)]

                        self.report_selfplay_stats(hidden_state_0, hidden_state_roots, game_histories, self.current_delay_value)

                        roots = cytree.Roots(env_nums, self.config.action_space_size, self.config.num_simulations)
                        noises = [np.random.dirichlet([self.config.root_dirichlet_alpha] * self.config.action_space_size).astype(np.float32).tolist() for _ in range(env_nums)]
                        roots.prepare(self.config.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool)
                        # do MCTS for a policy
                        MCTS(self.config).search(roots, model, hidden_state_roots, reward_hidden_roots)

                        roots_distributions = roots.get_distributions()
                        roots_values = roots.get_values()

                        for i in range(env_nums):
                            deterministic = False
                            if start_training:
                                distributions, value, temperature, env = roots_distributions[i], roots_values[i], _temperature[i], envs[i]
                            else:
                                # before starting training, use random policy
                                value, temperature, env = roots_values[i], _temperature[i], envs[i]
                                distributions = np.ones(self.config.action_space_size)

                            action, visit_entropy = select_action(distributions, temperature=temperature, deterministic=deterministic)
                            policy_actions_per_env[i].append(action)

                            if actions_ix == len(action_queues) - 1:
                                # There are sample_N actions queues in total, we add the stats for the last sample of the delay queues
                                # In EfficientZero there is only one delay queue, then it looks slightly different
                                game_histories[i].store_search_stats(distributions, value)
                                visit_entropies_lst[i] += visit_entropy

                                if self.config.use_priority and not self.config.use_max_priority and start_training:
                                    pred_values_lst[i].append(network_output.value[i].item())
                                    search_values_lst[i].append(roots_values[i])

                    for i in range(env_nums):
                        env = envs[i]
                        action = get_env_action_from_samples(policy_actions_per_env[i])
                        obs, ori_reward, done, info = env.step(action)
                        if self.config.test_use_forward:
                            executed_action = info['executed_action']
                        else:
                            executed_action = action
                        # clip the reward
                        if self.config.clip_reward:
                            clip_reward = np.sign(ori_reward)
                        else:
                            clip_reward = ori_reward

                        # store data
                        game_histories[i].append(action, executed_action, obs, clip_reward, self.config.delay)

                        eps_reward_lst[i] += clip_reward
                        eps_ori_reward_lst[i] += ori_reward
                        dones[i] = done

                        eps_steps_lst[i] += 1
                        self.total_transitions += 1


                        # fresh stack windows
                        del stack_obs_windows[i][0]
                        stack_obs_windows[i].append(obs)

                        # if game history is full;
                        # we will save a game history if it is the end of the game or the next game history is finished.
                        if game_histories[i].is_full():
                            # pad over last block trajectory
                            if last_game_histories[i] is not None:
                                self.put_last_trajectory(i, last_game_histories, last_game_priorities, game_histories)

                            # calculate priority
                            priorities = self.get_priorities(i, pred_values_lst, search_values_lst)

                            # save block trajectory
                            last_game_histories[i] = game_histories[i]
                            last_game_priorities[i] = priorities

                            # new block trajectory
                            game_histories[i] = GameHistory(envs[i].env.action_space, max_length=self.config.history_length,
                                                            config=self.config)
                            game_histories[i].init(stack_obs_windows[i])

                for i in range(env_nums):
                    env = envs[i]
                    env.close()

                    if dones[i]:
                        print('End of episode (selfplay_worker)')
                        # pad over last block trajectory
                        if last_game_histories[i] is not None:
                            self.put_last_trajectory(i, last_game_histories, last_game_priorities, game_histories)

                        # store current block trajectory
                        priorities = self.get_priorities(i, pred_values_lst, search_values_lst)
                        game_histories[i].game_over()

                        self.put((game_histories[i], priorities))
                        self.free()

                        self_play_rewards_max = max(self_play_rewards_max, eps_reward_lst[i])
                        self_play_moves_max = max(self_play_moves_max, eps_steps_lst[i])
                        self_play_rewards += eps_reward_lst[i]
                        self_play_ori_rewards += eps_ori_reward_lst[i]
                        self_play_visit_entropy.append(visit_entropies_lst[i] / eps_steps_lst[i])
                        self_play_moves += eps_steps_lst[i]
                        self_play_episodes += 1
                    else:
                        # if the final game history is not finished, we will not save this data.
                        self.total_transitions -= len(game_histories[i])

                # logs
                visit_entropies = np.array(self_play_visit_entropy).mean()
                visit_entropies /= max_visit_entropy

                if self_play_episodes > 0:
                    log_self_play_moves = self_play_moves / self_play_episodes
                    log_self_play_rewards = self_play_rewards / self_play_episodes
                    log_self_play_ori_rewards = self_play_ori_rewards / self_play_episodes
                else:
                    log_self_play_moves = 0
                    log_self_play_rewards = 0
                    log_self_play_ori_rewards = 0

                other_dist = {}
                # send logs
                self.storage.set_data_worker_logs.remote(log_self_play_moves, self_play_moves_max,
                                                                log_self_play_ori_rewards, log_self_play_rewards,
                                                                self_play_rewards_max, _temperature.mean(),
                                                                visit_entropies, 0,
                                                                other_dist)

    def report_selfplay_stats(self, hidden_state_0, hidden_state_m, game_histories, current_delay_value):
        self.obs_deque.append(hidden_state_m)
        if len(self.obs_deque) <= self.config.delay:
            return
        predicted_hidden_state_m = self.obs_deque.pop()

        error_sum = 0
        num_errors = 0
        for i in range(self.config.p_mcts_num):
            if len(game_histories[i]) >= self.config.delay:
                error_sum += np.sqrt(np.sum(np.square(predicted_hidden_state_m[i] - hidden_state_0[i])))
                num_errors += 1

        if num_errors > 0:
            mean_error = error_sum / num_errors
            self.storage.set_delay_logs.remote(predict_errors=mean_error, current_delay_value=current_delay_value)


