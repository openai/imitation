from collections import namedtuple
from policyopt import util
import numpy as np
import multiprocessing
from time import sleep


# State/action spaces
class Space(object):
    @property
    def storage_size(self): raise NotImplementedError
    @property
    def storage_type(self): raise NotImplementedError


class FiniteSpace(Space):
    def __init__(self, size): self._size = size
    @property
    def storage_size(self): return 1
    @property
    def storage_type(self): return int
    @property
    def size(self): return self._size


class ContinuousSpace(Space):
    def __init__(self, dim): self._dim = dim
    @property
    def storage_size(self): return self._dim
    @property
    def storage_type(self): return float
    @property
    def dim(self): return self._dim


class Trajectory(object):
    __slots__ = ('obs_T_Do', 'obsfeat_T_Df', 'adist_T_Pa', 'a_T_Da', 'r_T')
    def __init__(self, obs_T_Do, obsfeat_T_Df, adist_T_Pa, a_T_Da, r_T):
        assert (
            obs_T_Do.ndim == 2 and obsfeat_T_Df.ndim == 2 and adist_T_Pa.ndim == 2 and a_T_Da.ndim == 2 and r_T.ndim == 1 and
            obs_T_Do.shape[0] == obsfeat_T_Df.shape[0] == adist_T_Pa.shape[0] == a_T_Da.shape[0] == r_T.shape[0]
        )
        self.obs_T_Do = obs_T_Do
        self.obsfeat_T_Df = obsfeat_T_Df
        self.adist_T_Pa = adist_T_Pa
        self.a_T_Da = a_T_Da
        self.r_T = r_T

    def __len__(self):
        return self.obs_T_Do.shape[0]

    # Saving/loading discards obsfeat
    def save_h5(self, grp, **kwargs):
        grp.create_dataset('obs_T_Do', data=self.obs_T_Do, **kwargs)
        grp.create_dataset('adist_T_Pa', data=self.adist_T_Pa, **kwargs)
        grp.create_dataset('a_T_Da', data=self.a_T_Da, **kwargs)
        grp.create_dataset('r_T', data=self.r_T, **kwargs)

    @classmethod
    def LoadH5(cls, grp, obsfeat_fn):
        '''
        obsfeat_fn: used to fill in observation features. if None, the raw observations will be copied over.
        '''
        obs_T_Do = grp['obs_T_Do'][...]
        obsfeat_T_Df = obsfeat_fn(obs_T_Do) if obsfeat_fn is not None else obs_T_Do.copy()
        return cls(obs_T_Do, obsfeat_T_Df, grp['adist_T_Pa'][...], grp['a_T_Da'][...], grp['r_T'][...])


# Utilities for dealing with batches of trajectories with different lengths


def raggedstack(arrays, fill=0., axis=0, raggedaxis=1):
    '''
    Stacks a list of arrays, like np.stack with axis=0.
    Arrays may have different length (along the raggedaxis), and will be padded on the right
    with the given fill value.
    '''
    assert axis == 0 and raggedaxis == 1, 'not implemented'
    arrays = [a[None,...] for a in arrays]
    assert all(a.ndim >= 2 for a in arrays)

    outshape = list(arrays[0].shape)
    outshape[0] = sum(a.shape[0] for a in arrays)
    outshape[1] = max(a.shape[1] for a in arrays) # take max along ragged axes
    outshape = tuple(outshape)

    out = np.full(outshape, fill, dtype=arrays[0].dtype)
    pos = 0
    for a in arrays:
        out[pos:pos+a.shape[0], :a.shape[1], ...] = a
        pos += a.shape[0]
    assert pos == out.shape[0]
    return out


class RaggedArray(object):
    def __init__(self, arrays, lengths=None):
        if lengths is None:
            # Without provided lengths, `arrays` is interpreted as a list of arrays
            # and self.lengths is set to the list of lengths for those arrays
            self.arrays = arrays
            self.stacked = np.concatenate(arrays, axis=0)
            self.lengths = np.array([len(a) for a in arrays])
        else:
            # With provided lengths, `arrays` is interpreted as concatenated data
            # and self.lengths is set to the provided lengths.
            self.arrays = np.split(arrays, np.cumsum(lengths)[:-1])
            self.stacked = arrays
            self.lengths = np.asarray(lengths, dtype=int)
        assert all(len(a) == l for a,l in util.safezip(self.arrays, self.lengths))
        self.boundaries = np.concatenate([[0], np.cumsum(self.lengths)])
        assert self.boundaries[-1] == len(self.stacked)
    def __len__(self):
        return len(self.lengths)
    def __getitem__(self, idx):
        return self.stacked[self.boundaries[idx]:self.boundaries[idx+1], ...]
    def padded(self, fill=0.):
        return raggedstack(self.arrays, fill=fill, axis=0, raggedaxis=1)


class TrajBatch(object):
    def __init__(self, trajs, obs, obsfeat, adist, a, r, time):
        self.trajs, self.obs, self.obsfeat, self.adist, self.a, self.r, self.time = trajs, obs, obsfeat, adist, a, r, time

    @classmethod
    def FromTrajs(cls, trajs):
        assert all(isinstance(traj, Trajectory) for traj in trajs)
        obs = RaggedArray([t.obs_T_Do for t in trajs])
        obsfeat = RaggedArray([t.obsfeat_T_Df for t in trajs])
        adist = RaggedArray([t.adist_T_Pa for t in trajs])
        a = RaggedArray([t.a_T_Da for t in trajs])
        r = RaggedArray([t.r_T for t in trajs])
        time = RaggedArray([np.arange(len(t), dtype=float) for t in trajs])
        return cls(trajs, obs, obsfeat, adist, a, r, time)

    def with_replaced_reward(self, new_r):
        new_trajs = [Trajectory(traj.obs_T_Do, traj.obsfeat_T_Df, traj.adist_T_Pa, traj.a_T_Da, traj_new_r) for traj, traj_new_r in util.safezip(self.trajs, new_r)]
        return TrajBatch(new_trajs, self.obs, self.obsfeat, self.adist, self.a, new_r, self.time)

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        return self.trajs[idx]

    def save_h5(self, f, starting_id=0, **kwargs):
        for i, traj in enumerate(self.trajs):
            traj.save_h5(f.require_group('%06d' % (i+starting_id)), **kwargs)

    @classmethod
    def LoadH5(cls, dset, obsfeat_fn):
        return cls.FromTrajs([Trajectory.LoadH5(v, obsfeat_fn) for k, v in dset.iteritems()])


# MDP stuff

class Simulation(object):
    def step(self, action):
        '''
        Returns: reward
        '''
        raise NotImplementedError

    @property
    def obs(self):
        '''
        Get current observation. The caller must not assume that the contents of
        this array will never change, so this should usually be followed by a copy.

        Returns:
            numpy array
        '''
        raise NotImplementedError

    @property
    def done(self):
        '''
        Is this simulation done?

        Returns:
            boolean
        '''
        raise NotImplementedError

    def draw(self):
        raise NotImplementedError


class BatchedSim(object):
    def __len__(self): raise NotImplementedError

    def reset_sim(self, idx): raise NotImplementedError

    def reset_all(self):
        for i in xrange(len(self)):
            self.reset_sim(i)

    def is_done(self, idx): raise NotImplementedError

    @property
    def batch_obs(self):
        '''
        Get current observations for the simulation batch.

        The caller must not assume that the contents of this array will never
        change, so this should usually be followed by a copy.

        Returns:
            numpy array of shape (batch_size, observation_dim)
        '''
        raise NotImplementedError

    def batch_step(self, actions_B_Da, num_threads): raise NotImplementedError


class SequentialBatchedSim(BatchedSim):
    '''
    A 'fake' batched simulator that runs single-threaded simulations sequentially.
    '''
    def __init__(self, mdp, batch_size):
        self.mdp = mdp
        self.sims = [mdp.new_sim() for _ in xrange(batch_size)] # current active simulations
    def __len__(self): return len(self.sims)
    def reset_sim(self, idx): self.sims[idx] = self.mdp.new_sim()
    def is_done(self, idx): return self.sims[idx].done
    @property
    def batch_obs(self): return np.stack([s.obs.copy() for s in self.sims])
    def batch_step(self, actions_B_Da, num_threads=None):
        assert actions_B_Da.shape[0] == len(self.sims)
        rewards_B = np.zeros(len(self.sims))
        for i_sim in xrange(len(self.sims)):
            rewards_B[i_sim] = self.sims[i_sim].step(actions_B_Da[i_sim,:])
        return rewards_B


SimConfig = namedtuple('SimConfig', 'min_num_trajs min_total_sa batch_size max_traj_len')

class MDP(object):
    '''General MDP'''

    @property
    def obs_space(self):
        '''Observation space'''
        raise NotImplementedError

    @property
    def action_space(self):
        '''Action space'''
        raise NotImplementedError

    def new_sim(self, init_state=None):
        raise NotImplementedError

    def new_batched_sim(self, batch_size):
        return SequentialBatchedSim(self, batch_size)

    def sim_single(self, policy_fn, obsfeat_fn, max_traj_len, init_state=None):
        '''Simulate a single trajectory'''
        sim = self.new_sim(init_state=init_state)
        obs, obsfeat, actions, actiondists, rewards = [], [], [], [], []
        for _ in xrange(max_traj_len):
            obs.append(sim.obs[None,...].copy())
            obsfeat.append(obsfeat_fn(obs[-1]))
            a, adist = policy_fn(obsfeat[-1])
            actions.append(a)
            actiondists.append(adist)
            rewards.append(sim.step(a[0,:]))
            if sim.done: break
        obs_T_Do = np.concatenate(obs); assert obs_T_Do.shape == (len(obs), self.obs_space.storage_size)
        obsfeat_T_Df = np.concatenate(obsfeat); assert obsfeat_T_Df.shape[0] == len(obs)
        adist_T_Pa = np.concatenate(actiondists); assert adist_T_Pa.ndim == 2 and adist_T_Pa.shape[0] == len(obs)
        a_T_Da = np.concatenate(actions); assert a_T_Da.shape == (len(obs), self.action_space.storage_size)
        r_T = np.asarray(rewards); assert r_T.shape == (len(obs),)
        return Trajectory(obs_T_Do, obsfeat_T_Df, adist_T_Pa, a_T_Da, r_T)

    # @profile
    def sim_multi(self, policy_fn, obsfeat_fn, cfg, num_threads=None, no_reward=False):
        '''
        Run many simulations, with policy evaluations batched together.

        Samples complete trajectories (stopping when Simulation.done is true,
        or when cfg.max_traj_len is reached) until both
            (1) at least cfg.min_num_trajs trajectories have been sampled, and
            (2) at least cfg.min_total_sa transitions have been sampled.
        '''
        util.warn('sim_multi is deprecated!')
        assert isinstance(cfg, SimConfig)
        Do, Da = self.obs_space.storage_size, self.action_space.storage_size

        if num_threads is None:
            num_threads = multiprocessing.cpu_count()

        # Completed trajectories
        num_sa = 0
        completed_translists = []

        # Simulations and their current trajectories
        simbatch = self.new_batched_sim(cfg.batch_size) # TODO: reuse this across runs
        sim_trans_B = [[] for _ in xrange(cfg.batch_size)] # list of (o,obsfeat,adist,a,r) transitions for each simulation

        # Keep running simulations until we fill up the quota of trajectories and transitions
        while True:
            # If a simulation is done, pull out and save its trajectory, and restart it.
            for i_sim in xrange(cfg.batch_size):
                if simbatch.is_done(i_sim) or len(sim_trans_B[i_sim]) >= cfg.max_traj_len:
                    # Save the trajectory
                    completed_translists.append(sim_trans_B[i_sim])
                    num_sa += len(sim_trans_B[i_sim])
                    # and restart the simulation
                    sim_trans_B[i_sim] = []
                    simbatch.reset_sim(i_sim)

            # Are both quotas filled? If so, we're done.
            if len(completed_translists) >= cfg.min_num_trajs and num_sa >= cfg.min_total_sa:
                break

            # Keep simulating otherwise. Pull together observations from all simulations
            obs_B_Do = simbatch.batch_obs.copy(); assert obs_B_Do.shape == (cfg.batch_size, Do)

            # Evaluate policy
            obsfeat_B_Df = obsfeat_fn(obs_B_Do)
            a_B_Da, adist_B_Pa = policy_fn(obsfeat_B_Df)
            assert a_B_Da.shape == (cfg.batch_size, Da)
            assert adist_B_Pa.shape[0] == cfg.batch_size and adist_B_Pa.ndim == 2

            # Step simulations
            r_B = simbatch.batch_step(a_B_Da, num_threads=num_threads)
            if no_reward: r_B[:] = np.nan

            # Save the transitions
            for i_sim in xrange(cfg.batch_size):
                sim_trans_B[i_sim].append((obs_B_Do[i_sim,:], obsfeat_B_Df[i_sim,:], adist_B_Pa[i_sim,:], a_B_Da[i_sim,:], r_B[i_sim]))

        assert sum(len(tlist) for tlist in completed_translists) == num_sa

        # Pack together each trajectory individually
        def translist_to_traj(tlist):
            obs_T_Do = np.stack([trans[0] for trans in tlist]);  assert obs_T_Do.shape == (len(tlist), self.obs_space.storage_size)
            obsfeat_T_Df = np.stack([trans[1] for trans in tlist]); assert obsfeat_T_Df.shape[0] == len(tlist)
            adist_T_Pa = np.stack([trans[2] for trans in tlist]); assert adist_T_Pa.ndim == 2 and adist_T_Pa.shape[0] == len(tlist)
            a_T_Da = np.stack([trans[3] for trans in tlist]); assert a_T_Da.shape == (len(tlist), self.action_space.storage_size)
            r_T = np.stack([trans[4] for trans in tlist]); assert r_T.shape == (len(tlist),)
            return Trajectory(obs_T_Do, obsfeat_T_Df, adist_T_Pa, a_T_Da, r_T)
        completed_trajs = [translist_to_traj(tlist) for tlist in completed_translists]
        assert len(completed_trajs) >= cfg.min_num_trajs and sum(len(traj) for traj in completed_trajs) >= cfg.min_total_sa
        return TrajBatch.FromTrajs(completed_trajs)

    def sim_mp(self, policy_fn, obsfeat_fn, cfg, maxtasksperchild=200):
        '''
        Multiprocessed simulation
        Not thread safe! But why would you want this to be thread safe anyway?
        '''
        num_processes = cfg.batch_size if cfg.batch_size is not None else multiprocessing.cpu_count()//2

        # Bypass multiprocessing if only using one process
        if num_processes == 1:
            trajs = []
            num_sa = 0
            while True:
                t = self.sim_single(policy_fn, obsfeat_fn, cfg.max_traj_len)
                trajs.append(t)
                num_sa += len(t)
                if len(trajs) >= cfg.min_num_trajs and num_sa >= cfg.min_total_sa:
                    break
            return TrajBatch.FromTrajs(trajs)

        global _global_sim_info
        _global_sim_info = (self, policy_fn, obsfeat_fn, cfg.max_traj_len)

        trajs = []
        num_sa = 0

        with set_mkl_threads(1):
            # Thanks John
            pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=maxtasksperchild)
            pending = []
            done = False
            while True:
                if len(pending) < num_processes and not done:
                    pending.append(pool.apply_async(_rollout))
                stillpending = []
                for job in pending:
                    if job.ready():
                        traj = job.get()
                        trajs.append(traj)
                        num_sa += len(traj)
                    else:
                        stillpending.append(job)
                pending = stillpending
                if len(trajs) >= cfg.min_num_trajs and num_sa >= cfg.min_total_sa:
                    done = True
                    if len(pending) == 0:
                        break
                sleep(.001)
            pool.close()

        assert len(trajs) >= cfg.min_num_trajs and sum(len(traj) for traj in trajs) >= cfg.min_total_sa
        return TrajBatch.FromTrajs(trajs)

_global_sim_info = None
def _rollout():
    try:
        import os, random; random.seed(os.urandom(4)); np.random.seed(int(os.urandom(4).encode('hex'), 16))
        global _global_sim_info
        mdp, policy_fn, obsfeat_fn, max_traj_len = _global_sim_info
        return mdp.sim_single(policy_fn, obsfeat_fn, max_traj_len)
    except KeyboardInterrupt:
        pass

# Stuff for temporarily disabling MKL threading during multiprocessing
# http://stackoverflow.com/a/28293128
import ctypes
mkl_rt = None
try:
    mkl_rt = ctypes.CDLL('libmkl_rt.so')
    mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
    mkl_get_max_threads = mkl_rt.MKL_Get_Max_Threads
except OSError: # library not found
    util.warn('MKL runtime not found. Will not attempt to disable multithreaded MKL for parallel rollouts.')
from contextlib import contextmanager
@contextmanager
def set_mkl_threads(n):
    if mkl_rt is not None:
        orig = mkl_get_max_threads()
        mkl_set_num_threads(n)
    yield
    if mkl_rt is not None:
        mkl_set_num_threads(orig)
