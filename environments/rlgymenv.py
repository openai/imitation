import numpy as np
import policyopt

import gym
from gym import spaces, envs

gym.undo_logger_setup()
import logging; logging.getLogger('gym.core').addHandler(logging.NullHandler())


class RLGymSim(policyopt.Simulation):
    def __init__(self, env_name):
        self.env = envs.make(env_name)
        self.action_space = self.env.action_space
        self.curr_obs = self.env.reset()
        self.is_done = False

    def step(self, action):
        if isinstance(self.action_space, spaces.Discrete):
            # We encode actions in finite spaces as an integer inside a length-1 array
            # but Gym wants the integer itself
            assert action.ndim == 1 and action.size == 1 and action.dtype in (np.int32, np.int64)
            action = action[0]
        else:
            assert action.ndim == 1 and action.dtype == np.float64

        self.curr_obs, reward, self.is_done, _ = self.env.step(action)
        return reward

    @property
    def obs(self):
        return self.curr_obs.copy()

    @property
    def done(self):
        return self.is_done

    def draw(self, track_body_name='torso'):
        self.env.render()
        if track_body_name is not None and track_body_name in self.env.model.body_names:
            self.env.viewer.cam.trackbodyid = self.env.model.body_names.index(track_body_name)

    def __del__(self):
        if self.env.viewer:
            self.env.viewer.finish()

    def reset(self):
        self.curr_obs = self.env.reset()
        self.is_done = False

def _convert_space(space):
    '''Converts a rl-gym space to our own space representation'''
    if isinstance(space, spaces.Box):
        assert space.low.ndim == 1 and space.low.shape >= 1
        return policyopt.ContinuousSpace(dim=space.low.shape[0])
    elif isinstance(space, spaces.Discrete):
        return policyopt.FiniteSpace(size=space.n)
    raise NotImplementedError(space)


class RLGymMDP(policyopt.MDP):
    def __init__(self, env_name):
        print 'Gym version:', gym.version.VERSION
        self.env_name = env_name

        tmpsim = self.new_sim()
        self._obs_space = _convert_space(tmpsim.env.observation_space)
        self._action_space = _convert_space(tmpsim.env.action_space)
        self.env_spec = tmpsim.env.spec
        self.gym_env = tmpsim.env

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._action_space

    def new_sim(self, init_state=None):
        assert init_state is None
        return RLGymSim(self.env_name)
