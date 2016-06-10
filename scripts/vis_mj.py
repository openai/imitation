import argparse
import json
import h5py
import numpy as np

from environments import rlgymenv
import gym

import policyopt
from policyopt import SimConfig, rl, util, nn, tqdm
import os.path

def main():
    np.set_printoptions(suppress=True, precision=5, linewidth=1000)

    parser = argparse.ArgumentParser()
    # MDP options
    parser.add_argument('policy', type=str)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--max_traj_len', type=int, default=None) # only used for saving
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--count', type=int, default=None)
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()

    # Load the saved state
    policy_file, policy_key = util.split_h5_name(args.policy)
    print 'Loading policy parameters from %s in %s' % (policy_key, policy_file)
    with h5py.File(policy_file, 'r') as f:
        train_args = json.loads(f.attrs['args'])
        dset = f[policy_key]
        import pprint
        pprint.pprint(dict(dset.attrs))

    # Initialize the MDP
    env_name = train_args['env_name']
    print 'Loading environment', env_name
    mdp = rlgymenv.RLGymMDP(env_name)
    util.header('MDP observation space, action space sizes: %d, %d\n' % (mdp.obs_space.dim, mdp.action_space.storage_size))

    if args.max_traj_len is None:
        args.max_traj_len = mdp.env_spec.timestep_limit
    util.header('Max traj len is {}'.format(args.max_traj_len))

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

    if args.eval_only:
        n = 50
        print 'Evaluating based on {} trajs'.format(n)

        if False:
            eval_trajbatch = mdp.sim_mp(
                policy_fn=lambda obs_B_Do: policy.sample_actions(obs_B_Do, args.deterministic),
                obsfeat_fn=lambda obs:obs,
                cfg=policyopt.SimConfig(
                    min_num_trajs=n, min_total_sa=-1,
                    batch_size=None, max_traj_len=args.max_traj_len))
            returns = eval_trajbatch.r.padded(fill=0.).sum(axis=1)
            avgr = eval_trajbatch.r.stacked.mean()
            lengths = np.array([len(traj) for traj in eval_trajbatch])
            ent = policy._compute_actiondist_entropy(eval_trajbatch.adist.stacked).mean()
            print 'ret: {} +/- {}'.format(returns.mean(), returns.std())
            print 'avgr: {}'.format(avgr)
            print 'len: {} +/- {}'.format(lengths.mean(), lengths.std())
            print 'ent: {}'.format(ent)
            print returns
        else:
            returns = []
            lengths = []
            sim = mdp.new_sim()
            for i_traj in xrange(n):
                print i_traj, n
                sim.reset()
                totalr = 0.
                l = 0
                while not sim.done:
                    a = policy.sample_actions(sim.obs[None,:], bool(args.deterministic))[0][0,:]
                    r = sim.step(a)
                    totalr += r
                    l += 1
                returns.append(totalr)
                lengths.append(l)
        import IPython; IPython.embed()

    elif args.out is not None:
        # Sample trajs and write to file
        print 'Saving traj samples to file: {}'.format(args.out)

        assert not os.path.exists(args.out)
        assert args.count > 0
        # Simulate to create a trajectory batch
        util.header('Sampling {} trajectories of maximum length {}'.format(args.count, args.max_traj_len))
        trajs = []
        for i in tqdm.trange(args.count):
            trajs.append(mdp.sim_single(
                lambda obs: policy.sample_actions(obs, args.deterministic),
                lambda obs: obs,
                args.max_traj_len))
        trajbatch = policyopt.TrajBatch.FromTrajs(trajs)

        print
        print 'Average return:', trajbatch.r.padded(fill=0.).sum(axis=1).mean()

        # Save the trajs to a file
        with h5py.File(args.out, 'w') as f:
            def write(name, a):
                # chunks of 128 trajs each
                f.create_dataset(name, data=a, chunks=(min(128, a.shape[0]),)+a.shape[1:], compression='gzip', compression_opts=9)

            # Right-padded trajectory data
            write('obs_B_T_Do', trajbatch.obs.padded(fill=0.))
            write('a_B_T_Da', trajbatch.a.padded(fill=0.))
            write('r_B_T', trajbatch.r.padded(fill=0.))
            # Trajectory lengths
            write('len_B', np.array([len(traj) for traj in trajbatch], dtype=np.int32))

            # Also save args to this script
            argstr = json.dumps(vars(args), separators=(',', ':'), indent=2)
            f.attrs['args'] = argstr

    else:
        # Animate
        sim = mdp.new_sim()
        raw_obs, normalized_obs = [], []
        while True:
            sim.reset()
            totalr = 0.
            steps = 0
            while not sim.done:
                raw_obs.append(sim.obs[None,:])
                normalized_obs.append(policy.compute_internal_normalized_obsfeat(sim.obs[None,:]))

                a = policy.sample_actions(sim.obs[None,:], args.deterministic)[0][0,:]
                r = sim.step(a)
                totalr += r
                steps += 1
                sim.draw()

                if steps % 1000 == 0:
                    tmpraw = np.concatenate(raw_obs, axis=0)
                    tmpnormed = np.concatenate(normalized_obs, axis=0)
                    print 'raw mean, raw std, normed mean, normed std'
                    print np.stack([tmpraw.mean(0), tmpraw.std(0), tmpnormed.mean(0), tmpnormed.std(0)])
            print 'Steps: %d, return: %.5f' % (steps, totalr)

if __name__ == '__main__':
    main()
