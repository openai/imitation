'''
Show expert trajectory returns
'''

import argparse
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('trajfiles', nargs='+', type=str)
args = parser.parse_args()

for filename in args.trajfiles:
    print filename
    with h5py.File(filename, 'r') as f:
        rets = f['r_B_T'][...].sum(axis=1)
        lengths = f['len_B'][...]

        print 'return: {} +/- {}'.format(rets.mean(), rets.std())
        print rets

        print 'lengths: {} +/- {}'.format(lengths.mean(), lengths.std())
        print lengths

    print
