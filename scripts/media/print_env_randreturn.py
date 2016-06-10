import numpy as np

import gym
from gym import spaces, envs

gym.undo_logger_setup()
import logging; logging.getLogger('gym.core').addHandler(logging.NullHandler())

num_trials = 50

print 'Name & Random policy performance'

names = ['CartPole-v0', 'Acrobot-v0', 'MountainCar-v0', 'Reacher-v1', 'HalfCheetah-v1', 'Hopper-v1', 'Walker2d-v1', 'Ant-v1', 'Humanoid-v1']
for env_name in names:
    env = envs.make(env_name)

    returns = []
    for _ in xrange(num_trials):
        env.reset()
        ret = 0.
        for _ in xrange(env.spec.timestep_limit):
            _, r, done, _ = env.step(env.action_space.sample())
            ret += r
            if done: break
        returns.append(ret)

    print '{} & {} \pm {}'.format(env_name, np.mean(returns), np.std(returns))
