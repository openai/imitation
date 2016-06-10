'''
Check how many snapshots were saved in a set of log files
'''

import argparse
import h5py
from policyopt import util

parser = argparse.ArgumentParser()
parser.add_argument('logfiles', nargs='+', type=str)
args = parser.parse_args()

for filename in args.logfiles:
    if filename.endswith('.h5'):
        try:
            with h5py.File(filename, 'r') as f:
                snapshot_name = sorted(f['snapshots'].keys())[-1]
                assert snapshot_name.startswith('iter')
                last_snapshot_iter = int(snapshot_name[len('iter'):])
                #if last_snapshot_iter != 300:
                print filename, last_snapshot_iter
        except:
            util.warn('Error opening {}'.format(filename))
            continue
