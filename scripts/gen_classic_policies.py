import os, os.path

outdir = 'classic_policies'
cmd_template = 'python scripts/run_rl_mj.py --env_name {env} --tiny_policy --min_total_sa 5000 --sim_batch_size 1 --max_iter 101 --log {out}'

try: os.mkdir(outdir)
except OSError: pass

for env in ['CartPole-v0', 'Acrobot-v0', 'MountainCar-v0', 'InvertedPendulum-v1']:
    cmd = cmd_template.format(env=env, out=os.path.join(outdir, env+'.h5'))
    print cmd
    os.system(cmd)
    print '\n\n\n'
