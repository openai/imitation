import errno
import os
import numpy as np


def safezip(*ls):
    assert all(len(l) == len(ls[0]) for l in ls)
    return zip(*ls)

def flatten(lists):
    out = []
    for l in lists:
        out.extend(l)
    return out

def flatcat(arrays):
    return np.concatenate([a.ravel() for a in arrays])

def maxnorm(a):
    return np.abs(a).max()

def discount(r_N_T_D, gamma):
    '''
    Computes Q values from rewards.
    q_N_T_D[i,t,:] == r_N_T_D[i,t,:] + gamma*r_N_T_D[i,t+1,:] + gamma^2*r_N_T_D[i,t+2,:] + ...
    '''
    assert r_N_T_D.ndim == 2 or r_N_T_D.ndim == 3
    input_ndim = r_N_T_D.ndim
    if r_N_T_D.ndim == 2: r_N_T_D = r_N_T_D[...,None]

    discfactors_T = np.power(gamma, np.arange(r_N_T_D.shape[1]))
    discounted_N_T_D = r_N_T_D * discfactors_T[None,:,None]
    q_N_T_D = np.cumsum(discounted_N_T_D[:,::-1,:], axis=1)[:,::-1,:] # this is equal to gamma**t * (r_N_T_D[i,t,:] + gamma*r_N_T_D[i,t+1,:] + ...)
    q_N_T_D /= discfactors_T[None,:,None]

    # Sanity check: Q values at last timestep should equal original rewards
    assert np.allclose(q_N_T_D[:,-1,:], r_N_T_D[:,-1,:])

    if input_ndim == 2:
        assert q_N_T_D.shape[-1] == 1
        return q_N_T_D[:,:,0]
    return q_N_T_D

def test_discount():
    N, T, D = 5, 6, 7
    gamma = .81
    r_N_T_D = np.random.rand(N, T, D)
    q_N_T_D = discount(r_N_T_D, gamma)
    for i in xrange(N):
        for j in xrange(D):
            r_T = r_N_T_D[i,:,j]
            for t in xrange(T):
                assert np.allclose(q_N_T_D[i,t,j], (r_T[t:] * np.power(gamma, np.arange(T-t))).sum())

def standardized(a):
    out = a.copy()
    out -= a.mean()
    out /= a.std() + 1e-8
    return out

def gaussian_kl(means1_N_D, stdevs1_N_D, means2_N_D, stdevs2_N_D):
    '''
    KL divergences between Gaussians with diagonal covariances
    Covariances matrices are specified with square roots of the diagonal (standard deviations)
    '''
    assert means1_N_D.shape == stdevs1_N_D.shape == means2_N_D.shape == stdevs2_N_D.shape
    N, D = means1_N_D.shape

    return .5 * (
        np.square(stdevs1_N_D/stdevs2_N_D).sum(axis=1) +
        np.square((means2_N_D-means1_N_D)/stdevs2_N_D).sum(axis=1) +
        2.*(np.log(stdevs2_N_D).sum(axis=1) - np.log(stdevs1_N_D).sum(axis=1)) - D
    )


def gaussian_entropy(stdevs_N_D):
    d = stdevs_N_D.shape[1]
    return .5*d*(1. + np.log(2.*np.pi)) + np.log(stdevs_N_D).sum(axis=1)


def categorical_entropy(probs_N_K):
    tmp = -probs_N_K * np.log(probs_N_K + 1e-10)
    tmp[~np.isfinite(tmp)] = 0
    return tmp.sum(axis=1)

def sample_cats(probs_N_K):
    '''Sample from N categorical distributions, each over K outcomes'''
    N, K = probs_N_K.shape
    return np.array([np.random.choice(K, p=probs_N_K[i,:]) for i in xrange(N)])


def batched(lst, max_batch_size):
    num_batches = int(np.ceil(float(len(lst)) / max_batch_size))
    for i in xrange(num_batches):
        yield lst[i*max_batch_size : (i+1)*max_batch_size]

def objtranspose(lists):
    if not lists: return []
    num_outputs = len(lists[0])
    outputs = [[] for _ in range(num_outputs)]
    for l in lists:
        for o, x in safezip(outputs, l):
            o.append(x)
    return outputs

def angle(v1, v2):
    u1 = v1 / np.linalg.norm(v1)
    u2 = v2 / np.linalg.norm(v2)
    return np.arccos(u1.dot(u2))

import h5py
def split_h5_name(fullpath, sep='/'):
    '''
    From h5ls.c:
     * Example: ../dir1/foo/bar/baz
     *          \_________/\______/
     *             file       obj
     *
    '''
    sep_inds = [i for i, c in enumerate(fullpath) if c == sep]
    for sep_idx in sep_inds:
        filename, objname = fullpath[:sep_idx], fullpath[sep_idx:]
        if not filename: continue
        # Try to open the file. If it fails, try the next separation point.
        try: h5py.File(filename, 'r').close()
        except IOError: continue
        # It worked!
        return filename, objname
    raise IOError('Could not open HDF5 file/object %s' % fullpath)


import timeit
class Timer(object):
    def __enter__(self):
        self.t_start = timeit.default_timer()
        return self
    def __exit__(self, _1, _2, _3):
        self.t_end = timeit.default_timer()
        self.dt = self.t_end - self.t_start


class Colors(object):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
def header(s): print Colors.HEADER + s + Colors.ENDC
def warn(s): print Colors.WARNING + s + Colors.ENDC
def failure(s): print Colors.FAIL + s + Colors.ENDC


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

import scipy.io as sio
def loadmat(filename):
    '''Loads mat files with hierarchies as nested dictionaries'''
    def todict(d):
        if isinstance(d, sio.matlab.mio5_params.mat_struct):
            return {name: todict(d.__dict__[name]) for name in d._fieldnames}
        elif isinstance(d, np.ndarray) and d.dtype == np.object:
            return np.vectorize(todict)(d)
        return d

    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True, mat_dtype=True)
    return {k: todict(v) for k, v in data.iteritems()}
