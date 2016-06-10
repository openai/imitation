import argparse
import json
import h5py
import numpy as np
import yaml
import os, os.path, shutil
from policyopt import util



# PBS
import subprocess, tempfile, datetime
def create_pbs_script(commands, outputfiles, jobname, queue, nodes, ppn):
    assert len(commands) == len(outputfiles)
    template = '''#!/bin/bash

#PBS -l walltime=72:00:00,nodes={nodes}:ppn={ppn},mem=10gb
#PBS -N {jobname}
#PBS -q {queue}
#PBS -o /dev/null
#PBS -e /dev/null

sleep $[ ( $RANDOM % 120 ) + 1 ]s

read -r -d '' COMMANDS << END
{cmds_str}
END
cmd=$(echo "$COMMANDS" | awk "NR == $PBS_ARRAYID")
echo $cmd

read -r -d '' OUTPUTFILES << END
{outputfiles_str}
END
outputfile=$PBS_O_WORKDIR/$(echo "$OUTPUTFILES" | awk "NR == $PBS_ARRAYID")
echo $outputfile
# Make sure output directory exists
mkdir -p "`dirname \"$outputfile\"`" 2>/dev/null

cd $PBS_O_WORKDIR

echo $cmd >$outputfile
eval $cmd >>$outputfile 2>&1
'''
    return template.format(
        jobname=jobname,
        queue=queue,
        cmds_str='\n'.join(commands),
        outputfiles_str='\n'.join(outputfiles),
        nodes=nodes,
        ppn=ppn)

def runpbs(cmd_templates, outputfilenames, argdicts, jobname, queue, nodes, ppn, job_range=None, outputfile_dir=None, qsub_script_copy=None):
    assert len(cmd_templates) == len(outputfilenames) == len(argdicts)
    num_cmds = len(cmd_templates)

    outputfile_dir = outputfile_dir if outputfile_dir is not None else 'logs_%s_%s' % (jobname, datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))

    cmds, outputfiles = [], []
    for i in range(num_cmds):
        cmds.append(cmd_templates[i].format(**argdicts[i]))
        outputfiles.append(os.path.join(outputfile_dir, '{:04d}_{}'.format(i+1, outputfilenames[i])))

    script = create_pbs_script(cmds, outputfiles, jobname, queue, nodes, ppn)
    print script
    with tempfile.NamedTemporaryFile(suffix='.sh') as f:
        f.write(script)
        f.flush()

        if job_range is not None:
            assert len(job_range.split('-')) == 2, 'Invalid job range'
            cmd = 'qsub -t %s %s' % (job_range, f.name)
        else:
            cmd = 'qsub -t %d-%d %s' % (1, len(cmds), f.name)

        print 'Running command:', cmd
        print 'ok ({} jobs)? y/n'.format(num_cmds)
        if raw_input() == 'y':
            # Write a copy of the script
            if qsub_script_copy is not None:
                assert not os.path.exists(qsub_script_copy)
                with open(qsub_script_copy, 'w') as fcopy:
                    fcopy.write(script)
                print 'qsub script written to {}'.format(qsub_script_copy)
            # Run qsub
            subprocess.check_call(cmd, shell=True)

        else:
            raise RuntimeError('Canceled.')



def load_trained_policy_and_mdp(env_name, policy_state_str):
    import gym
    import policyopt
    from policyopt import nn, rl
    from environments import rlgymenv

    # Load the saved state
    policy_file, policy_key = util.split_h5_name(policy_state_str)
    print 'Loading policy parameters from %s in %s' % (policy_key, policy_file)
    with h5py.File(policy_file, 'r') as f:
        train_args = json.loads(f.attrs['args'])

    # Initialize the MDP
    print 'Loading environment', env_name
    mdp = rlgymenv.RLGymMDP(env_name)
    print 'MDP observation space, action space sizes: %d, %d\n' % (mdp.obs_space.dim, mdp.action_space.storage_size)

    # Initialize the policy
    nn.reset_global_scope()
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

    # Load the policy parameters
    policy.load_h5(policy_file, policy_key)

    return mdp, policy, train_args


def gen_taskname2outfile(spec, assert_not_exists=False):
    '''
    Generate dataset filenames for each task. Phase 0 (sampling) writes to these files,
    phase 1 (training) reads from them.
    '''
    taskname2outfile = {}
    trajdir = os.path.join(spec['options']['storagedir'], spec['options']['traj_subdir'])
    util.mkdir_p(trajdir)
    for task in spec['tasks']:
        assert task['name'] not in taskname2outfile
        fname = os.path.join(trajdir, 'trajs_{}.h5'.format(task['name']))
        if assert_not_exists:
            assert not os.path.exists(fname), 'Traj destination {} already exists'.format(fname)
        taskname2outfile[task['name']] = fname
    return taskname2outfile



def exec_saved_policy(env_name, policystr, num_trajs, deterministic, max_traj_len=None):
    import policyopt
    from policyopt import SimConfig, rl, util, nn, tqdm
    from environments import rlgymenv
    import gym

    # Load MDP and policy
    mdp, policy, _ = load_trained_policy_and_mdp(env_name, policystr)
    max_traj_len = min(mdp.env_spec.timestep_limit, max_traj_len) if max_traj_len is not None else mdp.env_spec.timestep_limit

    print 'Sampling {} trajs (max len {}) from policy {} in {}'.format(num_trajs, max_traj_len, policystr, env_name)

    # Sample trajs
    trajbatch = mdp.sim_mp(
        policy_fn=lambda obs_B_Do: policy.sample_actions(obs_B_Do, deterministic),
        obsfeat_fn=lambda obs:obs,
        cfg=policyopt.SimConfig(
            min_num_trajs=num_trajs,
            min_total_sa=-1,
            batch_size=None,
            max_traj_len=max_traj_len))

    return trajbatch, policy, mdp


def eval_snapshot(env_name, checkptfile, snapshot_idx, num_trajs, deterministic):
    policystr = '{}/snapshots/iter{:07d}'.format(checkptfile, snapshot_idx)
    trajbatch, _, _ = exec_saved_policy(
        env_name,
        policystr,
        num_trajs,
        deterministic=deterministic,
        max_traj_len=None)
    returns = trajbatch.r.padded(fill=0.).sum(axis=1)
    lengths = np.array([len(traj) for traj in trajbatch])
    util.header('{} gets return {} +/- {}'.format(policystr, returns.mean(), returns.std()))
    return returns, lengths


def phase0_sampletrajs(spec, specfilename):
    util.header('=== Phase 0: Sampling trajs from expert policies ===')

    num_trajs = spec['training']['full_dataset_num_trajs']
    util.header('Sampling {} trajectories'.format(num_trajs))

    # Make filenames and check if they're valid first
    taskname2outfile = gen_taskname2outfile(spec, assert_not_exists=True)

    # Sample trajs for each task
    for task in spec['tasks']:
        # Execute the policy
        trajbatch, policy, _ = exec_saved_policy(
            task['env'], task['policy'], num_trajs,
            deterministic=spec['training']['deterministic_expert'],
            max_traj_len=None)

        # Quick evaluation
        returns = trajbatch.r.padded(fill=0.).sum(axis=1)
        avgr = trajbatch.r.stacked.mean()
        lengths = np.array([len(traj) for traj in trajbatch])
        ent = policy._compute_actiondist_entropy(trajbatch.adist.stacked).mean()
        print 'ret: {} +/- {}'.format(returns.mean(), returns.std())
        print 'avgr: {}'.format(avgr)
        print 'len: {} +/- {}'.format(lengths.mean(), lengths.std())
        print 'ent: {}'.format(ent)

        # Save the trajs to a file
        with h5py.File(taskname2outfile[task['name']], 'w') as f:
            def write(dsetname, a):
                f.create_dataset(dsetname, data=a, compression='gzip', compression_opts=9)
            # Right-padded trajectory data
            write('obs_B_T_Do', trajbatch.obs.padded(fill=0.))
            write('a_B_T_Da', trajbatch.a.padded(fill=0.))
            write('r_B_T', trajbatch.r.padded(fill=0.))
            # Trajectory lengths
            write('len_B', np.array([len(traj) for traj in trajbatch], dtype=np.int32))
            # # Also save args to this script
            # argstr = json.dumps(vars(args), separators=(',', ':'), indent=2)
            # f.attrs['args'] = argstr
        util.header('Wrote {}'.format(taskname2outfile[task['name']]))


def phase1_train(spec, specfilename):
    util.header('=== Phase 1: training ===')

    # Generate array job that trains all algorithms
    # over all tasks, for all dataset sizes (3 loops)

    taskname2dset = gen_taskname2outfile(spec)

    # Make checkpoint dir. All outputs go here
    checkptdir = os.path.join(spec['options']['storagedir'], spec['options']['checkpt_subdir'])
    util.mkdir_p(checkptdir)
    # Make sure checkpoint dir is empty
    assert not os.listdir(checkptdir), 'Checkpoint directory {} is not empty!'.format(checkptdir)

    # Assemble the commands to run on the cluster
    cmd_templates, outputfilenames, argdicts = [], [], []
    for alg in spec['training']['algorithms']:
        for task in spec['tasks']:
            for num_trajs in spec['training']['dataset_num_trajs']:
                assert num_trajs <= spec['training']['full_dataset_num_trajs']
                for run in range(spec['training']['runs']):
                    # A string identifier. Used in filenames for this run
                    strid = 'alg={},task={},num_trajs={},run={}'.format(alg['name'], task['name'], num_trajs, run)
                    cmd_templates.append(alg['cmd'].replace('\n', ' ').strip())
                    outputfilenames.append(strid + '.txt')
                    argdicts.append({
                        'env': task['env'],
                        'dataset': taskname2dset[task['name']],
                        'num_trajs': num_trajs,
                        'cuts_off_on_success': int(task['cuts_off_on_success']),
                        'data_subsamp_freq': task['data_subsamp_freq'],
                        'out': os.path.join(checkptdir, strid + '.h5'),
                    })

    pbsopts = spec['options']['pbs']
    runpbs(
        cmd_templates, outputfilenames, argdicts,
        jobname=pbsopts['jobname'], queue=pbsopts['queue'], nodes=1, ppn=pbsopts['ppn'],
        job_range=pbsopts['range'] if 'range' in pbsopts else None,
        qsub_script_copy=os.path.join(checkptdir, 'qsub_script.sh')
    )

    # Copy the pipeline yaml file to the output dir too
    shutil.copyfile(specfilename, os.path.join(checkptdir, 'pipeline.yaml'))

    # Keep git commit
    import subprocess
    git_hash = subprocess.check_output('git rev-parse HEAD', shell=True).strip()
    with open(os.path.join(checkptdir, 'git_hash.txt'), 'w') as f:
        f.write(git_hash + '\n')

def phase2_eval(spec, specfilename):
    util.header('=== Phase 2: evaluating trained models ===')
    import pandas as pd

    taskname2dset = gen_taskname2outfile(spec)

    # This is where model logs are stored.
    # We will also store the evaluation here.
    checkptdir = os.path.join(spec['options']['storagedir'], spec['options']['checkpt_subdir'])
    print 'Evaluating results in {}'.format(checkptdir)

    results_full_path = os.path.join(checkptdir, spec['options']['results_filename'])
    print 'Will store results in {}'.format(results_full_path)
    if os.path.exists(results_full_path):
        raise RuntimeError('Results file {} already exists'.format(results_full_path))

    # First, pre-determine which evaluations we have to do
    evals_to_do = []
    nonexistent_checkptfiles = []
    for task in spec['tasks']:
        # See how well the algorithms did...
        for alg in spec['training']['algorithms']:
            # ...on various dataset sizes
            for num_trajs in spec['training']['dataset_num_trajs']:
                # for each rerun, for mean / error bars later
                for run in range(spec['training']['runs']):
                    # Make sure the checkpoint file exists (maybe PBS dropped some jobs)
                    strid = 'alg={},task={},num_trajs={},run={}'.format(alg['name'], task['name'], num_trajs, run)
                    checkptfile = os.path.join(checkptdir, strid + '.h5')
                    if not os.path.exists(checkptfile):
                        nonexistent_checkptfiles.append(checkptfile)
                    evals_to_do.append((task, alg, num_trajs, run, checkptfile))

    if nonexistent_checkptfiles:
        print 'Cannot find checkpoint files:\n', '\n'.join(nonexistent_checkptfiles)
        raise RuntimeError

    # Walk through all saved checkpoints
    collected_results = []
    for i_eval, (task, alg, num_trajs, run, checkptfile) in enumerate(evals_to_do):
        util.header('Evaluating run {}/{}: alg={},task={},num_trajs={},run={}'.format(
            i_eval+1, len(evals_to_do), alg['name'], task['name'], num_trajs, run))

        # Load the task's traj dataset to see how well the expert does
        with h5py.File(taskname2dset[task['name']], 'r') as trajf:
            # Expert's true return and traj lengths
            ex_traj_returns = trajf['r_B_T'][...].sum(axis=1)
            ex_traj_lengths = trajf['len_B'][...]

        # Load the checkpoint file
        with pd.HDFStore(checkptfile, 'r') as f:
            log_df = f['log']
            log_df.set_index('iter', inplace=True)

            # Evaluate true return for the learned policy
            if alg['name'] == 'bclone':
                # Pick the policy with the best validation accuracy
                best_snapshot_idx = log_df['valacc'].argmax()
                alg_traj_returns, alg_traj_lengths = eval_snapshot(
                    task['env'], checkptfile, best_snapshot_idx,
                    spec['options']['eval_num_trajs'], deterministic=True)

            elif any(alg['name'].startswith(s) for s in ('ga', 'fem', 'simplex')):
                # Evaluate the last saved snapshot
                snapshot_names = f.root.snapshots._v_children.keys()
                assert all(name.startswith('iter') for name in snapshot_names)
                snapshot_inds = sorted([int(name[len('iter'):]) for name in snapshot_names])
                best_snapshot_idx = snapshot_inds[-1]
                alg_traj_returns, alg_traj_lengths = eval_snapshot(
                    task['env'], checkptfile, best_snapshot_idx,
                    spec['options']['eval_num_trajs'], deterministic=True)

            else:
                raise NotImplementedError('Analysis not implemented for {}'.format(alg['name']))

            collected_results.append({
                # Trial info
                'alg': alg['name'],
                'task': task['name'],
                'num_trajs': num_trajs,
                'run': run,
                # Expert performance
                'ex_traj_returns': ex_traj_returns,
                'ex_traj_lengths': ex_traj_lengths,
                # Learned policy performance
                'alg_traj_returns': alg_traj_returns,
                'alg_traj_lengths': alg_traj_lengths,
            })

    collected_results = pd.DataFrame(collected_results)
    with pd.HDFStore(results_full_path, 'w') as outf:
        outf['results'] = collected_results


def main():
    np.set_printoptions(suppress=True, precision=5, linewidth=1000)

    phases = {
        '0_sampletrajs': phase0_sampletrajs,
        '1_train': phase1_train,
        '2_eval': phase2_eval,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('spec', type=str)
    parser.add_argument('phase', choices=sorted(phases.keys()))
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = yaml.load(f)

    phases[args.phase](spec, args.spec)

if __name__ == '__main__':
    main()
