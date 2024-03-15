import numpy as np
from core.game import Game
from core.utils import arr_to_str


class AtariWrapper(Game):
    def __init__(self, env, discount: float, cvt_string=True):
        """Atari Wrapper
        Parameters
        ----------
        env: Any
            another env wrapper
        discount: float
            discount of env
        cvt_string: bool
            True -> convert the observation into string in the replay buffer
        """
        super().__init__(env, env.action_space.n, discount)
        self.cvt_string = cvt_string

    def legal_actions(self):
        return [_ for _ in range(self.env.action_space.n)]

    def get_max_episode_steps(self):
        return self.env.get_max_episode_steps()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = observation.astype(np.uint8)

        if self.cvt_string:
            observation = arr_to_str(observation)

        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = observation.astype(np.uint8)

        if self.cvt_string:
            observation = arr_to_str(observation)

        return observation

    def close(self):
        self.env.close()

    def initialize_pending_actions(self, model=None, stack_obs=None, config=None):
        return self.env.initialize_pending_actions(model, stack_obs, config)

    def get_pending_actions_for_agent(self, current_delay=None, delay_queue=None):
        return self.env.get_pending_actions_for_agent(current_delay, delay_queue)

    def set_current_delay(self, underlying_delay_value):
        self.env.set_current_delay(underlying_delay_value)
