import argparse
import json

import numpy as np

from environments import rlgymenv
import policyopt
from policyopt import SimConfig, rl, util, nn

TINY_ARCHITECTURE = '[{"type": "fc", "n": 64}, {"type": "nonlin", "func": "tanh"}, {"type": "fc", "n": 64}, {"type": "nonlin", "func": "tanh"}]'
SIMPLE_ARCHITECTURE = '[{"type": "fc", "n": 100}, {"type": "nonlin", "func": "tanh"}, {"type": "fc", "n": 100}, {"type": "nonlin", "func": "tanh"}]'
def main():
    np.set_printoptions(suppress=True, precision=5, linewidth=1000)

    parser = argparse.ArgumentParser()
    # MDP options
    parser.add_argument('--discount', type=float, default=.995)
    parser.add_argument('--lam', type=float, default=.97)
    parser.add_argument('--max_traj_len', type=int, default=None)
    parser.add_argument('--env_name', type=str, required=True)
    # Policy architecture
    parser.add_argument('--policy_hidden_spec', type=str, default=SIMPLE_ARCHITECTURE)
    parser.add_argument('--enable_obsnorm', type=int, default=1)
    parser.add_argument('--tiny_policy', action='store_true')
    parser.add_argument('--use_tanh', type=int, default=0)
    # Optimizer
    parser.add_argument('--max_iter', type=int, default=1000000)
    parser.add_argument('--policy_max_kl', type=float, default=.01)
    parser.add_argument('--policy_cg_damping', type=float, default=.1)
    parser.add_argument('--vf_max_kl', type=float, default=.01)
    parser.add_argument('--vf_cg_damping', type=float, default=.1)
    # Sampling
    parser.add_argument('--sim_batch_size', type=int, default=None)
    parser.add_argument('--min_total_sa', type=int, default=100000)
    # Saving stuff
    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--log', type=str, required=False)

    args = parser.parse_args()

    if args.tiny_policy or args.use_tanh:
        assert args.policy_hidden_spec == SIMPLE_ARCHITECTURE, 'policy_hidden_spec must remain unspecified if --tiny_policy is set'
        args.policy_hidden_spec = TINY_ARCHITECTURE

        if args.use_tanh:
            arch = json.loads(args.policy_hidden_spec)
            for layer in arch:
                if layer['type'] == 'nonlin':
                    layer['func'] = 'tanh'
            args.policy_hidden_spec = json.dumps(arch)
        print 'Modified architecture:', args.policy_hidden_spec

    argstr = json.dumps(vars(args), separators=(',', ':'), indent=2)
    print(argstr)

    mdp = rlgymenv.RLGymMDP(args.env_name)
    util.header('MDP observation space, action space sizes: %d, %d\n' % (mdp.obs_space.dim, mdp.action_space.storage_size))

    if isinstance(mdp.action_space, policyopt.ContinuousSpace):
        policy_cfg = rl.GaussianPolicyConfig(
            hidden_spec=args.policy_hidden_spec,
            min_stdev=0.,
            init_logstdev=0.,
            enable_obsnorm=bool(args.enable_obsnorm))
        policy = rl.GaussianPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'GaussianPolicy')
    else:
        policy_cfg = rl.GibbsPolicyConfig(
            hidden_spec=args.policy_hidden_spec,
            enable_obsnorm=bool(args.enable_obsnorm))
        policy = rl.GibbsPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'GibbsPolicy')

    util.header('Policy architecture')
    policy.print_trainable_variables()

    vf = rl.ValueFunc(
        hidden_spec=args.policy_hidden_spec,
        obsfeat_space=mdp.obs_space,
        enable_obsnorm=bool(args.enable_obsnorm),
        enable_vnorm=True,
        max_kl=args.vf_max_kl,
        damping=args.vf_cg_damping,
        time_scale=1./mdp.env_spec.timestep_limit,
        varscope_name='ValueFunc')

    max_traj_len = args.max_traj_len if args.max_traj_len is not None else mdp.env_spec.timestep_limit
    print 'Max traj len:', max_traj_len
    opt = rl.SamplingPolicyOptimizer(
        mdp=mdp,
        discount=args.discount,
        lam=args.lam,
        policy=policy,
        sim_cfg=SimConfig(
            min_num_trajs=-1,
            min_total_sa=args.min_total_sa,
            batch_size=args.sim_batch_size,
            max_traj_len=max_traj_len),
        step_func=rl.TRPO(max_kl=args.policy_max_kl, damping=args.policy_cg_damping),
        value_func=vf,
        obsfeat_fn=lambda obs: obs,
    )

    log = nn.TrainingLog(args.log, [('args', argstr)])

    for i in xrange(args.max_iter):
        iter_info = opt.step()
        log.write(iter_info, print_header=i % 20 == 0)
        if args.save_freq != 0 and i % args.save_freq == 0 and args.log is not None:
            log.write_snapshot(policy, i)


if __name__ == '__main__':
    main()
