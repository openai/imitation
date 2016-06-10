from . import nn, util, thutil, optim, ContinuousSpace, FiniteSpace, RaggedArray
from collections import namedtuple
from contextlib import contextmanager
import environments
import numpy as np

import theano
from theano import tensor

from abc import abstractmethod

class Policy(nn.Model):
    def __init__(self, obsfeat_space, action_space, num_actiondist_params, enable_obsnorm, varscope_name):
        self.obsfeat_space, self.action_space, self._num_actiondist_params = obsfeat_space, action_space, num_actiondist_params

        with nn.variable_scope(varscope_name) as self.__varscope:
            # Action distribution for this current policy
            obsfeat_B_Df = tensor.matrix(name='obsfeat_B_Df')
            with nn.variable_scope('obsnorm'):
                self.obsnorm = (nn.Standardizer if enable_obsnorm else nn.NoOpStandardizer)(self.obsfeat_space.dim)
            normalized_obsfeat_B_Df = self.obsnorm.standardize_expr(obsfeat_B_Df)
            # Convert (normalized) observations to action distributions
            actiondist_B_Pa = self._make_actiondist_ops(normalized_obsfeat_B_Df) # Pa == parameters of action distribution
            self._compute_action_dist_params = thutil.function([obsfeat_B_Df], actiondist_B_Pa)

        # Only code above this line (i.e. _make_actiondist_ops) is allowed to make trainable variables.
        param_vars = self.get_trainable_variables()

        # Reinforcement learning
        input_actions_B_Da = tensor.matrix(name='input_actions_B_Da', dtype=theano.config.floatX if self.action_space.storage_type == float else 'int64')
        logprobs_B = self._make_actiondist_logprob_ops(actiondist_B_Pa, input_actions_B_Da)
        # Proposal distribution from old policy
        proposal_actiondist_B_Pa = tensor.matrix(name='proposal_actiondist_B_Pa')
        proposal_logprobs_B = self._make_actiondist_logprob_ops(proposal_actiondist_B_Pa, input_actions_B_Da)
        # Local RL objective
        advantage_B = tensor.vector(name='advantage_B')
        impweight_B = tensor.exp(logprobs_B - proposal_logprobs_B)
        obj = (impweight_B*advantage_B).mean()
        objgrad_P = thutil.flatgrad(obj, param_vars)
        # KL divergence from old policy
        kl_B = self._make_actiondist_kl_ops(proposal_actiondist_B_Pa, actiondist_B_Pa)
        kl = kl_B.mean()
        compute_obj_kl = thutil.function([obsfeat_B_Df, input_actions_B_Da, proposal_actiondist_B_Pa, advantage_B], [obj, kl])
        compute_obj_kl_with_grad = thutil.function([obsfeat_B_Df, input_actions_B_Da, proposal_actiondist_B_Pa, advantage_B], [obj, kl, objgrad_P])
        # KL Hessian-vector product
        v_P = tensor.vector(name='v')
        klgrad_P = thutil.flatgrad(kl, param_vars)
        hvpexpr = thutil.flatgrad((klgrad_P*v_P).sum(), param_vars)
        # hvpexpr = tensor.Rop(klgrad_P, param_vars, thutil.unflatten_into_tensors(v_P, [v.get_value().shape for v in param_vars]))
        hvp = thutil.function([obsfeat_B_Df, proposal_actiondist_B_Pa, v_P], hvpexpr)
        compute_hvp = lambda _obsfeat_B_Df, _input_actions_B_Da, _proposal_actiondist_B_Pa, _advantage_B, _v_P: hvp(_obsfeat_B_Df, _proposal_actiondist_B_Pa, _v_P)
        # TRPO step
        self._ngstep = optim.make_ngstep_func(self, compute_obj_kl, compute_obj_kl_with_grad, compute_hvp)

        ##### Publicly-exposed functions #####
        # for debugging
        self.compute_internal_normalized_obsfeat = thutil.function([obsfeat_B_Df], normalized_obsfeat_B_Df)

        # Supervised learning objective: log likelihood of actions given state
        bclone_loss = -logprobs_B.mean()
        bclone_lr = tensor.scalar(name='bclone_lr')
        self.step_bclone = thutil.function(
            [obsfeat_B_Df, input_actions_B_Da, bclone_lr],
            bclone_loss,
            updates=thutil.adam(bclone_loss, param_vars, lr=bclone_lr))
        self.compute_bclone_loss_and_grad = thutil.function(
            [obsfeat_B_Df, input_actions_B_Da],
            [bclone_loss, thutil.flatgrad(bclone_loss, param_vars)])
        self.compute_bclone_loss = thutil.function(
            [obsfeat_B_Df, input_actions_B_Da],
            bclone_loss)

        self.compute_action_logprobs = thutil.function(
            [obsfeat_B_Df, input_actions_B_Da],
            logprobs_B)


    @property
    def varscope(self): return self.__varscope

    def update_obsnorm(self, obs_B_Do):
        '''Update observation normalization using a moving average'''
        self.obsnorm.update(obs_B_Do)

    def sample_actions(self, obsfeat_B_Df, deterministic=False):
        '''Samples actions conditioned on states'''
        actiondist_B_Pa = self._compute_action_dist_params(obsfeat_B_Df)
        return self._sample_from_actiondist(actiondist_B_Pa, deterministic), actiondist_B_Pa

    # To be overridden
    @abstractmethod
    def _make_actiondist_ops(self, obsfeat_B_Df): pass
    @abstractmethod
    def _make_actiondist_logprob_ops(self, actiondist_B_Pa, input_actions_B_Da): pass
    @abstractmethod
    def _make_actiondist_kl_ops(self, proposal_actiondist_B_Pa, actiondist_B_Pa): pass
    @abstractmethod
    def _sample_from_actiondist(self, actiondist_B_Pa, deterministic): pass
    @abstractmethod
    def _compute_actiondist_entropy(self, actiondist_B_Pa): pass


GaussianPolicyConfig = namedtuple('GaussianPolicyConfig', 'hidden_spec, min_stdev, init_logstdev, enable_obsnorm')
class GaussianPolicy(Policy):
    def __init__(self, cfg, obsfeat_space, action_space, varscope_name):
        assert isinstance(cfg, GaussianPolicyConfig)
        assert isinstance(obsfeat_space, ContinuousSpace) and isinstance(action_space, ContinuousSpace)

        self.cfg = cfg
        Policy.__init__(
            self,
            obsfeat_space=obsfeat_space,
            action_space=action_space,
            num_actiondist_params=action_space.dim*2,
            enable_obsnorm=cfg.enable_obsnorm,
            varscope_name=varscope_name)


    def _make_actiondist_ops(self, obsfeat_B_Df):
        # Computes action distribution mean (of a Gaussian) using MLP
        with nn.variable_scope('hidden'):
            net = nn.FeedforwardNet(obsfeat_B_Df, (self.obsfeat_space.dim,), self.cfg.hidden_spec)
        with nn.variable_scope('out'):
            mean_layer = nn.AffineLayer(net.output, net.output_shape, (self.action_space.dim,), initializer=np.zeros((net.output_shape[0], self.action_space.dim)))
            assert mean_layer.output_shape == (self.action_space.dim,)
        means_B_Da = mean_layer.output

        # Action distribution log standard deviations are parameters themselves
        logstdevs_1_Da = nn.get_variable('logstdevs_1_Da', np.full((1, self.action_space.dim), self.cfg.init_logstdev), broadcastable=(True,False))
        stdevs_1_Da = self.cfg.min_stdev + tensor.exp(logstdevs_1_Da) # minimum stdev seems to make density / kl computations more stable
        stdevs_B_Da = tensor.ones_like(means_B_Da)*stdevs_1_Da # "broadcast" to (B,Da)

        actiondist_B_Pa = tensor.concatenate([means_B_Da, stdevs_B_Da], axis=1)
        return actiondist_B_Pa

    def _extract_actiondist_params(self, actiondist_B_Pa):
        means_B_Da = actiondist_B_Pa[:, :self.action_space.dim]
        stdevs_B_Da = actiondist_B_Pa[:, self.action_space.dim:]
        return means_B_Da, stdevs_B_Da

    def _make_actiondist_logprob_ops(self, actiondist_B_Pa, input_actions_B_Da):
        means_B_Da, stdevs_B_Da = self._extract_actiondist_params(actiondist_B_Pa)
        return thutil.gaussian_log_density(means_B_Da, stdevs_B_Da, input_actions_B_Da)

    def _make_actiondist_kl_ops(self, proposal_actiondist_B_Pa, actiondist_B_Pa):
        proposal_means_B_Da, proposal_stdevs_B_Da = self._extract_actiondist_params(proposal_actiondist_B_Pa)
        means_B_Da, stdevs_B_Da = self._extract_actiondist_params(actiondist_B_Pa)
        return thutil.gaussian_kl(proposal_means_B_Da, proposal_stdevs_B_Da, means_B_Da, stdevs_B_Da)

    def _sample_from_actiondist(self, actiondist_B_Pa, deterministic):
        adim = self.action_space.dim
        means_B_Da, stdevs_B_Da = actiondist_B_Pa[:,:adim], actiondist_B_Pa[:,adim:]
        if deterministic:
            return means_B_Da
        stdnormal_B_Da = np.random.randn(actiondist_B_Pa.shape[0], adim)
        assert stdnormal_B_Da.shape == means_B_Da.shape == stdevs_B_Da.shape
        return (stdnormal_B_Da*stdevs_B_Da) + means_B_Da

    def _compute_actiondist_entropy(self, actiondist_B_Pa):
        _, stdevs_B_Da = self._extract_actiondist_params(actiondist_B_Pa)
        return util.gaussian_entropy(stdevs_B_Da)

    def compute_actiondist_mean(self, obsfeat_B_Df):
        actiondist_B_Pa = self._compute_action_dist_params(obsfeat_B_Df)
        means_B_Da, _ = self._extract_actiondist_params(actiondist_B_Pa)
        return means_B_Da

GibbsPolicyConfig = namedtuple('GibbsPolicyConfig', 'hidden_spec, enable_obsnorm')
class GibbsPolicy(Policy):
    def __init__(self, cfg, obsfeat_space, action_space, varscope_name):
        assert isinstance(cfg, GibbsPolicyConfig)
        assert isinstance(obsfeat_space, ContinuousSpace) and isinstance(action_space, FiniteSpace)
        self.cfg = cfg
        Policy.__init__(
            self,
            obsfeat_space=obsfeat_space,
            action_space=action_space,
            num_actiondist_params=action_space.size,
            enable_obsnorm=cfg.enable_obsnorm,
            varscope_name=varscope_name)

    def _make_actiondist_ops(self, obsfeat_B_Df):
        # Computes action distribution using MLP
        # Actiondist consists of the log probabilities
        with nn.variable_scope('hidden'):
            net = nn.FeedforwardNet(obsfeat_B_Df, (self.obsfeat_space.dim,), self.cfg.hidden_spec)
        with nn.variable_scope('out'):
            out_layer = nn.AffineLayer(
                net.output, net.output_shape, (self.action_space.size,),
                initializer=np.zeros((net.output_shape[0], self.action_space.size)))
            assert out_layer.output_shape == (self.action_space.size,)
        scores_B_Pa = out_layer.output # unnormalized (unshifted) log probability
        actiondist_B_Pa = scores_B_Pa - thutil.logsumexp(scores_B_Pa, axis=1) # log probability
        return actiondist_B_Pa

    def _make_actiondist_logprob_ops(self, actiondist_B_Pa, input_actions_B_Da):
        return actiondist_B_Pa[tensor.arange(actiondist_B_Pa.shape[0]), input_actions_B_Da[:,0]]

    def _make_actiondist_kl_ops(self, proposal_actiondist_B_Pa, actiondist_B_Pa):
        return thutil.categorical_kl(proposal_actiondist_B_Pa, actiondist_B_Pa)

    def _sample_from_actiondist(self, actiondist_B_Pa, deterministic):
        probs_B_A = np.exp(actiondist_B_Pa); assert probs_B_A.shape[1] == self.action_space.size
        if deterministic:
            return probs_B_A.argmax(axis=1)[:,None]
        return util.sample_cats(probs_B_A)[:,None]

    def _compute_actiondist_entropy(self, actiondist_B_Pa):
        return util.categorical_entropy(np.exp(actiondist_B_Pa))


def compute_qvals(r, gamma):
    assert isinstance(r, RaggedArray)
    trajlengths = r.lengths
    # Zero-fill the rewards on the right, then compute Q values
    rewards_B_T = r.padded(fill=0.)
    qvals_zfilled_B_T = util.discount(rewards_B_T, gamma)
    assert qvals_zfilled_B_T.shape == (len(trajlengths), trajlengths.max())
    return RaggedArray([qvals_zfilled_B_T[i,:l] for i, l in enumerate(trajlengths)]), rewards_B_T

def compute_advantage(r, obsfeat, time, value_func, gamma, lam):
    assert isinstance(r, RaggedArray) and isinstance(obsfeat, RaggedArray) and isinstance(time, RaggedArray)
    trajlengths = r.lengths
    assert np.array_equal(obsfeat.lengths, trajlengths) and np.array_equal(time.lengths, trajlengths)
    B, maxT = len(trajlengths), trajlengths.max()

    # Compute Q values
    q, rewards_B_T = compute_qvals(r, gamma)
    q_B_T = q.padded(fill=np.nan); assert q_B_T.shape == (B, maxT) # q values, padded with nans at the end

    # Time-dependent baseline that cheats on the current batch
    simplev_B_T = np.tile(np.nanmean(q_B_T, axis=0, keepdims=True), (B, 1)); assert simplev_B_T.shape == (B, maxT)
    simplev = RaggedArray([simplev_B_T[i,:l] for i, l in enumerate(trajlengths)])

    # State-dependent baseline (value function)
    v_stacked = value_func.evaluate(obsfeat.stacked, time.stacked); assert v_stacked.ndim == 1
    v = RaggedArray(v_stacked, lengths=trajlengths)

    # Compare squared loss of value function to that of the time-dependent value function
    constfunc_prediction_loss = np.var(q.stacked)
    simplev_prediction_loss = np.var(q.stacked-simplev.stacked) #((q.stacked-simplev.stacked)**2).mean()
    simplev_r2 = 1. - simplev_prediction_loss/(constfunc_prediction_loss + 1e-8)
    vfunc_prediction_loss = np.var(q.stacked-v_stacked) #((q.stacked-v_stacked)**2).mean()
    vfunc_r2 = 1. - vfunc_prediction_loss/(constfunc_prediction_loss + 1e-8)

    # Compute advantage -- GAE(gamma, lam) estimator
    v_B_T = v.padded(fill=0.)
    # append 0 to the right
    v_B_Tp1 = np.concatenate([v_B_T, np.zeros((B,1))], axis=1); assert v_B_Tp1.shape == (B, maxT+1)
    delta_B_T = rewards_B_T + gamma*v_B_Tp1[:,1:] - v_B_Tp1[:,:-1]
    adv_B_T = util.discount(delta_B_T, gamma*lam); assert adv_B_T.shape == (B, maxT)
    adv = RaggedArray([adv_B_T[i,:l] for i, l in enumerate(trajlengths)])
    assert np.allclose(adv.padded(fill=0), adv_B_T)

    return adv, q, vfunc_r2, simplev_r2


class SamplingPolicyOptimizer(object):
    def __init__(self, mdp, discount, lam, policy, sim_cfg, step_func, value_func, obsfeat_fn):
        self.mdp, self.discount, self.lam, self.policy = mdp, discount, lam, policy
        self.sim_cfg = sim_cfg
        self.step_func = step_func
        self.value_func = value_func
        self.obsfeat_fn = obsfeat_fn

        self.total_num_sa = 0
        self.total_time = 0.
        self.curr_iter = 0

    def step(self):
        with util.Timer() as t_all:

            # Sample trajectories using current policy
            with util.Timer() as t_sample:
                # At the first iter, sample an extra batch to initialize standardization parameters
                if self.curr_iter == 0:
                    trajbatch0 = self.mdp.sim_mp(
                        policy_fn=lambda obsfeat_B_Df: self.policy.sample_actions(obsfeat_B_Df),
                        obsfeat_fn=self.obsfeat_fn,
                        cfg=self.sim_cfg)
                    self.policy.update_obsnorm(trajbatch0.obsfeat.stacked)
                    self.value_func.update_obsnorm(trajbatch0.obsfeat.stacked)

                trajbatch = self.mdp.sim_mp(
                    policy_fn=lambda obsfeat_B_Df: self.policy.sample_actions(obsfeat_B_Df),
                    obsfeat_fn=self.obsfeat_fn,
                    cfg=self.sim_cfg)
                # TODO: normalize rewards

            # Compute baseline / advantages
            with util.Timer() as t_adv:
                advantages, qvals, vfunc_r2, simplev_r2 = compute_advantage(
                    trajbatch.r, trajbatch.obsfeat, trajbatch.time,
                    self.value_func, self.discount, self.lam)

            # Take a step
            with util.Timer() as t_step:
                params0_P = self.policy.get_params()
                extra_print_fields = self.step_func(
                    self.policy, params0_P,
                    trajbatch.obsfeat.stacked, trajbatch.a.stacked, trajbatch.adist.stacked,
                    advantages.stacked)
                self.policy.update_obsnorm(trajbatch.obsfeat.stacked)

            # Fit value function for next iteration
            with util.Timer() as t_vf_fit:
                if self.value_func is not None:
                    extra_print_fields += self.value_func.fit(
                        trajbatch.obsfeat.stacked, trajbatch.time.stacked, qvals.stacked)

        # Log
        self.total_num_sa += sum(len(traj) for traj in trajbatch)
        self.total_time += t_all.dt
        fields = [
            ('iter', self.curr_iter, int),
            ('ret', trajbatch.r.padded(fill=0.).sum(axis=1).mean(), float), # average return for this batch of trajectories
            # ('discret', np.mean([q[0] for q in qvals]), float),
            # ('ravg', trajbatch.r.stacked.mean(), float), # average reward encountered
            ('avglen', int(np.mean([len(traj) for traj in trajbatch])), int), # average traj length
            ('nsa', self.total_num_sa, int), # total number of state-action pairs sampled over the course of training
            ('ent', self.policy._compute_actiondist_entropy(trajbatch.adist.stacked).mean(), float), # entropy of action distributions
            ('vf_r2', vfunc_r2, float),
            ('tdvf_r2', simplev_r2, float),
            ('dx', util.maxnorm(params0_P - self.policy.get_params()), float), # max parameter difference from last iteration
        ] + extra_print_fields + [
            ('tsamp', t_sample.dt, float), # time for sampling
            ('tadv', t_adv.dt + t_vf_fit.dt, float), # time for advantage computation
            ('tstep', t_step.dt, float), # time for step computation
            ('ttotal', self.total_time, float), # total time
        ]
        self.curr_iter += 1
        return fields


# Policy gradient update rules

def TRPO(max_kl, damping, subsample_hvp_frac=.1, grad_stop_tol=1e-6):

    def trpo_step(policy, params0_P, obsfeat, a, adist, adv):
        feed = (obsfeat, a, adist, util.standardized(adv))
        stepinfo = policy._ngstep(feed, max_kl=max_kl, damping=damping, subsample_hvp_frac=subsample_hvp_frac, grad_stop_tol=grad_stop_tol)
        return [
            ('dl', stepinfo.obj1 - stepinfo.obj0, float), # improvement of penalized objective
            ('kl', stepinfo.kl1, float), # kl cost of solution
            ('gnorm', stepinfo.gnorm, float), # gradient norm
            ('bt', stepinfo.bt, int), # number of backtracking steps
        ]

    return trpo_step


import scipy.linalg
class LinearValueFunc(object):
    def __init__(self, l2reg=1e-5):
        self.w_Df = None
        self.l2reg = l2reg

    def _feat(self, obs_B_Do, t_B):
        assert obs_B_Do.ndim == 2 and t_B.ndim == 1
        B = obs_B_Do.shape[0]
        return np.concatenate([
                obs_B_Do,
                t_B[:,None]/100.,
                (t_B[:,None]/100.)**2,
                np.ones((B,1))
            ], axis=1)

    def evaluate(self, obs_B_Do, t_B):
        feat_Df = self._feat(obs_B_Do, t_B)
        if self.w_Df is None:
            self.w_Df = np.zeros(feat_Df.shape[1], dtype=obs_B_Do.dtype)
        return feat_Df.dot(self.w_Df)

    def fit(self, obs_B_Do, t_B, y_B):
        assert y_B.shape == (obs_B_Do.shape[0],)
        feat_B_Df = self._feat(obs_B_Do, t_B)
        self.w_Df = scipy.linalg.solve(
            feat_B_Df.T.dot(feat_B_Df) + self.l2reg*np.eye(feat_B_Df.shape[1]),
            feat_B_Df.T.dot(y_B),
            sym_pos=True)


class ValueFunc(nn.Model):
    def __init__(self, hidden_spec, obsfeat_space, enable_obsnorm, enable_vnorm, varscope_name, max_kl, damping, time_scale):
        self.hidden_spec = hidden_spec
        self.obsfeat_space = obsfeat_space
        self.enable_obsnorm = enable_obsnorm
        self.enable_vnorm = enable_vnorm
        self.max_kl = max_kl
        self.damping = damping
        self.time_scale = time_scale

        with nn.variable_scope(varscope_name) as self.__varscope:
            # Standardizers. Used outside, not in the computation graph.
            with nn.variable_scope('obsnorm'):
                self.obsnorm = (nn.Standardizer if enable_obsnorm else nn.NoOpStandardizer)(self.obsfeat_space.dim)
            with nn.variable_scope('vnorm'):
                self.vnorm = (nn.Standardizer if enable_vnorm else nn.NoOpStandardizer)(1)

            # Input observations
            obsfeat_B_Df = tensor.matrix(name='obsfeat_B_Df')
            t_B = tensor.vector(name='t_B')
            scaled_t_B = t_B * self.time_scale
            net_input = tensor.concatenate([obsfeat_B_Df, scaled_t_B[:,None]], axis=1)
            # Compute (normalized) value of states using a feedforward network
            with nn.variable_scope('hidden'):
                net = nn.FeedforwardNet(net_input, (self.obsfeat_space.dim + 1,), self.hidden_spec)
            with nn.variable_scope('out'):
                out_layer = nn.AffineLayer(net.output, net.output_shape, (1,), initializer=np.zeros((net.output_shape[0], 1)))
                assert out_layer.output_shape == (1,)
            val_B = out_layer.output[:,0]
        # Only code above this line is allowed to make trainable variables.
        param_vars = self.get_trainable_variables()

        self._evaluate_raw = thutil.function([obsfeat_B_Df, t_B], val_B)

        # Squared loss for fitting the value function
        target_val_B = tensor.vector(name='target_val_B')
        obj = -tensor.square(val_B - target_val_B).mean()
        objgrad_P = thutil.flatgrad(obj, param_vars)
        # KL divergence (as Gaussian) and its gradient
        old_val_B = tensor.vector(name='old_val_B')
        kl = tensor.square(old_val_B - val_B).mean()
        compute_obj_kl = thutil.function([obsfeat_B_Df, t_B, target_val_B, old_val_B], [obj, kl])
        compute_obj_kl_with_grad = thutil.function([obsfeat_B_Df, t_B, target_val_B, old_val_B], [obj, kl, objgrad_P])
        # KL Hessian-vector product
        x_P = tensor.vector(name='x')
        klgrad_P = thutil.flatgrad(kl, param_vars)
        hvp = thutil.function([obsfeat_B_Df, t_B, old_val_B, x_P], thutil.flatgrad((klgrad_P*x_P).sum(), param_vars))
        compute_kl_hvp = lambda _obsfeat_B_Df, _t_B, _target_val_B, _old_val_B, _x_P: hvp(_obsfeat_B_Df, _t_B, _old_val_B, _x_P)
        self._ngstep = optim.make_ngstep_func(self, compute_obj_kl, compute_obj_kl_with_grad, compute_kl_hvp)

    @property
    def varscope(self): return self.__varscope

    def evaluate(self, obs_B_Do, t_B):
        # ignores the time
        assert obs_B_Do.shape[0] == t_B.shape[0]
        return self.vnorm.unstandardize(self._evaluate_raw(self.obsnorm.standardize(obs_B_Do), t_B)[:,None])[:,0]

    def fit(self, obs_B_Do, t_B, y_B):
        # ignores the time
        assert obs_B_Do.shape[0] == t_B.shape[0] == y_B.shape[0]

        # Update normalization
        self.obsnorm.update(obs_B_Do)
        self.vnorm.update(y_B[:,None])

        # Take step
        sobs_B_Do = self.obsnorm.standardize(obs_B_Do)
        feed = (sobs_B_Do, t_B, self.vnorm.standardize(y_B[:,None])[:,0], self._evaluate_raw(sobs_B_Do, t_B))
        stepinfo = self._ngstep(feed, max_kl=self.max_kl, damping=self.damping)
        return [
            ('vf_dl', stepinfo.obj1 - stepinfo.obj0, float), # improvement of penalized objective
            ('vf_kl', stepinfo.kl1, float), # kl cost of solution
            ('vf_gnorm', stepinfo.gnorm, float), # gradient norm
            ('vf_bt', stepinfo.bt, int), # number of backtracking steps
        ]

    def update_obsnorm(self, obs_B_Do):
        self.obsnorm.update(obs_B_Do)


class ConstantValueFunc(object):
    def __init__(self, max_timesteps):
        self.max_timesteps = max_timesteps
        self.v_T = np.zeros(max_timesteps)

    def evaluate(self, obs_B_Do, t_B):
        int_t_B = t_B.astype(int, copy=False)
        assert np.all(int_t_B == t_B) and np.all(0 <= t_B) and np.all(t_B < self.max_timesteps)
        return self.v_T[int_t_B].copy()

    def fit(self, obs_B_Do, t_B, y_B):
        int_t_B = t_B.astype(int, copy=False)
        assert np.all(int_t_B == t_B) and np.all(0 <= t_B) and np.all(t_B < self.max_timesteps)
        # Add up y values at various timesteps
        sum_T = np.zeros(self.max_timesteps)
        np.add.at(sum_T, int_t_B, y_B) # like sum_T[t_B] += y_B, but accumulates over duplicated time indices
        # Count number of values added at each timestep
        counts_T = np.zeros(self.max_timesteps)
        np.add.at(counts_T, int_t_B, 1)
        counts_T[counts_T < 1] = 1
        # Divide to get average
        self.v_T = sum_T / counts_T
        return []

    def update_obsnorm(self, obs_B_Do):
        pass
