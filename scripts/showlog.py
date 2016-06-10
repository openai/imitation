import argparse
import pandas as pd
import h5py
import json
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logfiles', type=str, nargs='+')
    parser.add_argument('--fields', type=str, default='trueret,avglen,ent,kl,vf_r2,vf_kl,tdvf_r2,rloss,racc')
    parser.add_argument('--noplot', action='store_true')
    parser.add_argument('--plotfile', type=str, default=None)
    parser.add_argument('--range_end', type=int, default=None)
    args = parser.parse_args()

    assert len(set(args.logfiles)) == len(args.logfiles), 'Log files must be unique'

    fields = args.fields.split(',')

    # Load logs from all files
    fname2log = {}
    for fname in args.logfiles:
        with pd.HDFStore(fname, 'r') as f:
            assert fname not in fname2log
            df = f['log']
            df.set_index('iter', inplace=True)
            fname2log[fname] = df.loc[:args.range_end, fields]


    # Print stuff
    if not args.noplot or args.plotfile is not None:
        import matplotlib
        if args.plotfile is not None:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt; plt.style.use('ggplot')

        ax = None
        for fname, df in fname2log.items():
            with pd.option_context('display.max_rows', 9999):
                print fname
                print df[-1:]


            df['vf_r2'] = np.maximum(0,df['vf_r2'])

            if ax is None:
                ax = df.plot(subplots=True, title=fname)
            else:
                df.plot(subplots=True, title=fname, ax=ax, legend=False)
        if not args.noplot:
            plt.show()
        if args.plotfile is not None:
            plt.savefig(args.plotfile, bbox_inches='tight', dpi=200)


if __name__ == '__main__':
    main()
