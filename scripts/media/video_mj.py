import argparse
import json
import h5py
import numpy as np

from environments import rlgymenv
import gym

import policyopt
from policyopt import SimConfig, rl, util, nn, tqdm
import os.path

import cv2

def main():
    np.set_printoptions(suppress=True, precision=5, linewidth=1000)

    parser = argparse.ArgumentParser()
    # MDP options
    parser.add_argument('policy', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--deterministic', default=1, type=int)
    parser.add_argument('--max_steps', type=int, required=True)
    parser.add_argument('--env_name', type=str, default=None)
    args = parser.parse_args()

    util.mkdir_p(args.output_dir)
    assert not os.listdir(args.output_dir), '%s is not empty' % args.output_dir
    print 'Writing to', args.output_dir

    # Load the saved state
    policy_file, policy_key = util.split_h5_name(args.policy)
    print 'Loading policy parameters from %s in %s' % (policy_key, policy_file)
    with h5py.File(policy_file, 'r') as f:
        train_args = json.loads(f.attrs['args'])
        dset = f[policy_key]
        import pprint
        pprint.pprint(dict(dset.attrs))

    # Initialize the MDP
    env_name = train_args['env_name'] if args.env_name is None else args.env_name
    print 'Loading environment', env_name
    mdp = rlgymenv.RLGymMDP(env_name)
    util.header('MDP observation space, action space sizes: %d, %d\n' % (mdp.obs_space.dim, mdp.action_space.storage_size))

    util.header('Max steps is {}'.format(args.max_steps))

    # Initialize the policy and load its parameters
    enable_obsnorm = bool(train_args['enable_obsnorm']) if 'enable_obsnorm' in train_args else train_args['obsnorm_mode'] != 'none'
    if isinstance(mdp.action_space, policyopt.ContinuousSpace):
        policy_cfg = rl.GaussianPolicyConfig(
            hidden_spec=train_args['policy_hidden_spec'],
            min_stdev=0.,
            init_logstdev=0.,
            enable_obsnorm=enable_obsnorm)
        policy = rl.GaussianPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'GaussianPolicy')
    else:
        policy_cfg = rl.GibbsPolicyConfig(
            hidden_spec=train_args['policy_hidden_spec'],
            enable_obsnorm=enable_obsnorm)
        policy = rl.GibbsPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'GibbsPolicy')
    policy.load_h5(policy_file, policy_key)

    # Animate
    sim = mdp.new_sim()
    steps = 0
    exit = False
    while not exit:
        sim.reset()
        while not sim.done:
            a = policy.sample_actions(sim.obs[None,:], bool(args.deterministic))[0][0,:]
            sim.step(a)
            sim.draw()
            viewer = sim.env.viewer
            data, w, h = viewer.get_image()
            image = np.fromstring(data, dtype='uint8').reshape(h, w, 3)[::-1,:,:]
            cv2.imwrite('%s/img_%08d.png' % (args.output_dir, steps), image[:,:,::-1])

            print steps
            steps += 1

            if steps >= args.max_steps:
                exit = True
                break

    #print 'Steps: %d, return: %.5f' % (steps, totalr)

if __name__ == '__main__':
    main()
