import gym
from gym import spaces, envs

gym.undo_logger_setup()
import logging; logging.getLogger('gym.core').addHandler(logging.NullHandler())

names = ['CartPole-v0', 'Acrobot-v0', 'MountainCar-v0', 'Reacher-v1', 'HalfCheetah-v1', 'Hopper-v1', 'Walker2d-v1', 'Ant-v1', 'Humanoid-v1']
for n in names:
    env = envs.make(n)

    aspace = env.action_space
    if isinstance(aspace, spaces.Box):
        acont = True
        asize = aspace.low.shape[0]
    else:
        acont = False
        asize = aspace.n

    ospace = env.observation_space
    if isinstance(ospace, spaces.Box):
        ocont = True
        osize = ospace.low.shape[0]
    else:
        ocont = False
        osize = ospace.n

    print '{} & {} ({}) & {} ({}) \\\\'.format(n, osize, 'continuous' if ocont else 'discrete', asize, 'continuous' if acont else 'discrete')
