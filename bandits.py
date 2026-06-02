"""
bandits.py
==========
All bandit algorithms for the MINTS comparison study.

Algorithms
----------
MINTS (this paper):
  MINTS_Unstructured   — MINTS with no structural assumption
  MINTS_Unimodal       — MINTS with unimodal constraint

Unstructured baselines:
  KLUCBGaussian        — KL-UCB for Gaussian rewards (Garivier & Cappé 2011)
  ThompsonSampling     — Gaussian Thompson Sampling with known sigma

Unimodal algorithms:
  OSUB                 — Optimal Sampling for Unimodal Bandits
                         (Combes & Proutière, ICML 2014)
  UTS                  — Unimodal Thompson Sampling
                         (Paladino et al., AAAI 2017)

General structured (applied to unimodal):
  OSSB                 — Optimal Sampling for Structured Bandits,
                         unimodal closed-form specialisation
                         (Combes, Magureanu & Proutière, NeurIPS 2017)

Shared interface
----------------
All algorithms expose:
    alg = AlgorithmClass(K, reward_std=1.0, seed=None, **kwargs)
    arm = alg.sample_decision()   -> int in {0, ..., K-1}
    alg.update_belief(reward)     -> None

All algorithms assume Gaussian rewards N(mu_i, sigma^2) with known sigma.
Arms are indexed 0, ..., K-1 and arranged on a line graph (for unimodal
algorithms, the ordering 0 < 1 < ... < K-1 defines the structure).

Registries
----------
ALL_ALGORITHMS    : dict {key: class}  — all seven algorithms
UNIMODAL_ALGORITHMS    : set of keys that exploit unimodal structure
UNSTRUCTURED_ALGORITHMS: set of keys that use no structural assumption
"""

import numpy as np
from scipy.optimize import minimize


# ======================================================================
# Shared utilities
# ======================================================================

def _klucb_gaussian_index(mu_hat, n, t, sigma2):
    """
    Closed-form KL-UCB index for Gaussian rewards with known variance:
        B_i(t) = mu_hat_i + sigma * sqrt(2 * log(t) / n_i)
    Returns inf for unobserved arms (n=0).
    """
    if n == 0:
        return np.inf
    return mu_hat + np.sqrt(sigma2 * 2.0 * np.log(t) / n)


# ======================================================================
# MINTS
# ======================================================================

class MINTS:
    """
    MINTS (MINimalist Thompson Sampling) for multi-armed bandits with mean constraints.
    The vector of mean rewards is known to belong to a constraint set named Theta.
    Rewards are Gaussian with unit variance.

    Maintains a posterior over the identity of the optimal arm j* using the profile log-likelihood

        log PL(j) = -residual(j) / (2 sigma^2)

    where

        residual(j) = min_{theta in Theta_j} sum_i n_i (theta_i - mu_hat_i)^2

    is the weighted least-squares residual after projecting mu_hat onto a constraint set

        Theta_j = {theta in Theta : theta_j >= theta_i  for all i}.

    Arms are sampled proportional to the posterior (Thompson-style).

    This program implements MINTS for unstructured and unimodal bandits.

    Parameters
    ----------
    K          : int    number of arms (indexed 0,...,K-1 on a line)
    reward_std : float  known reward standard deviation sigma
    unimodal   : bool
        False (default) — unstructured bandit:
            Theta_j = {theta : theta_i <= theta_j  for all i}
            Solved in O(K log K) via a closed-form pooling algorithm.
        True  — unimodal constraint on the line graph:
            Theta_j = {theta : theta_0 <= ... <= theta_j >= ... >= theta_{K-1}}
            Solved via SLSQP with warm-start caching (no correct closed form
            exists for fixed-j unimodal isotonic regression).
    prior      : array or None  (default: uniform)
    update_freq: int, default = 1. Update the posterior every update_freq rounds.
    seed       : int or None
    """

    def __init__(self, K, reward_std=1.0, unimodal=False, prior=None,
                 update_freq=1, seed=None):
        self.K           = K
        self.sigma2      = float(reward_std) ** 2
        self.unimodal    = unimodal
        self.update_freq = int(update_freq)
        self.rng         = np.random.default_rng(seed)
        self.prior       = (np.ones(K) / K if prior is None
                            else np.asarray(prior, float) / np.sum(prior))
        self.posterior   = self.prior.copy()
        self.counts      = np.zeros(K)
        self.rewards_sum = np.zeros(K)
        self._new_idx    = None
        self._round      = 0   # counts every update_belief call

        # Unimodal only: SLSQP warm-start cache and pre-built constraints
        if unimodal:
            self._warm_cache  = [None] * K
            self._constraints = self._build_unimodal_constraints()

    @property
    def name(self):
        return 'MINTS (unimodal)' if self.unimodal else 'MINTS (unstr.)'

    @property
    def key(self):
        return 'mints_m' if self.unimodal else 'mints_u'

    def _build_unimodal_constraints(self):
        """
        SLSQP constraint dicts for candidate j under the unimodal cone:
            theta_{i+1} - theta_i >= 0  for i < j   (increasing left of peak)
            theta_i - theta_{i+1} >= 0  for i >= j  (decreasing right of peak)
        """
        K = self.K
        result = {}
        for j in range(K):
            rows = []
            for i in range(j):
                r = np.zeros(K); r[i+1] = 1.0; r[i] = -1.0
                rows.append(r)
            for i in range(j, K-1):
                r = np.zeros(K); r[i] = 1.0; r[i+1] = -1.0
                rows.append(r)
            result[j] = [{'type': 'ineq',
                           'fun':  lambda th, r=r: float(r @ th),
                           'jac':  lambda th, r=r: r} for r in rows]
        return result

    def sample_decision(self):
        self._new_idx = int(self.rng.choice(self.K, p=self.posterior))
        return self._new_idx

    def update_belief(self, reward):
        if self._new_idx is None:
            raise RuntimeError(
                "update_belief() called before sample_decision(). "
                "Call sample_decision() first to select an arm.")
        i = self._new_idx
        self.counts[i]      += 1.0
        self.rewards_sum[i] += float(reward)
        self._round         += 1
        # Update posterior every update_freq rounds; sufficient statistics are always kept current 
        # Use the most recent posterior for arm selection between updates.
        if self._round % self.update_freq == 0:
            self._update_posterior(self._profile_likelihood())

    def _profile_likelihood(self):
        """Dispatch to the appropriate method based on unimodal flag."""
        return self._pl_unimodal() if self.unimodal else self._pl_unstructured()

    def _pl_unstructured(self):
        """
        Closed-form O(K log K) profile likelihood for the unstructured case.

        For candidate j, the constrained MLE theta*(j) satisfies:
            theta*_i(j) = mu_hat_i       for i outside the pool  (at their MLE)
            theta*_i(j) = pool_mean_j    for i inside the pool   (all equal)

        where the pool P(j) = {j} union {i : mu_hat_i > pool_mean_j}.
        The pool mean pool_mean_j is found by a single sorted pass.

        Profile log-likelihood (sum-of-squares form):
            log PL(j) = -residual(j) / (2 sigma^2)

            residual(j) = sum_i n_i (theta*_i(j) - mu_hat_i)^2
                        = sum_{i in P(j)} n_i (pool_mean_j - mu_hat_i)^2
                        = pool_ss - pool_S^2 / pool_n

        where pool_ss = sum_{i in P(j)} S_i^2/n_i.  The last equality
        follows by expanding and using pool_mean = pool_S/pool_n:

            sum_{i in P} n_i(bar_mu - mu_hat_i)^2
              = pool_n * bar_mu^2 - 2*bar_mu*pool_S + pool_ss
              = pool_S^2/pool_n - 2*pool_S^2/pool_n + pool_ss
              = pool_ss - pool_S^2/pool_n.

        Unobserved candidate (n[j]=0): arm j is unconstrained so all
        observed arms sit at their MLE, residual = 0, log PL(j) = 0.
        """
        n, S, K, s2 = self.counts, self.rewards_sum, self.K, self.sigma2
        obs    = n > 0
        mu_hat = np.where(obs, S / np.where(obs, n, 1.0), -np.inf)
        lpl    = np.full(K, -np.inf)

        for j in range(K):
            if n[j] == 0:
                lpl[j] = 0.0   # all arms free at MLE => residual = 0
                continue
            others     = np.where((np.arange(K) != j) & obs)[0]
            sorted_idx = others[np.argsort(-mu_hat[others])]
            pn, pS, pm = n[j], S[j], mu_hat[j]
            pss = S[j] ** 2 / n[j]     # sum_{i in pool} S_i^2/n_i
            for i in sorted_idx:
                if mu_hat[i] > pm:
                    pn  += n[i]; pS += S[i]
                    pm   = pS / pn
                    pss += S[i] ** 2 / n[i]
                else:
                    break               # safe: sorted descending, pm non-decreasing
            lpl[j] = -(pss - pS ** 2 / pn) / (2.0 * s2)
        return lpl

    def _pl_unimodal(self):
        """
        SLSQP profile likelihood for the unimodal constraint.

        For each candidate j, solves:
            theta*(j) = argmin_{theta in Theta_j} sum_i n_i (theta_i - mu_hat_i)^2
        then computes:
            log PL(j) = -sum_i n_i (theta*_i(j) - mu_hat_i)^2 / (2 sigma^2)

        Warm-start: theta*(j) from the previous round initialises the next
        solve. Falls back to the grand mean (always feasible) on first call.
        """
        n, S, K, s2 = self.counts, self.rewards_sum, self.K, self.sigma2
        obs    = n > 0
        mu_hat = np.where(obs, S / np.where(obs, n, 1.0), 0.0)
        gm     = float(S.sum() / n.sum()) if n.sum() > 0 else 0.0

        def obj(th):  return float(np.dot(n, (th - mu_hat) ** 2))
        def grad(th): return 2.0 * n * (th - mu_hat)

        lpl = np.full(K, -np.inf)
        for j in range(K):
            x0  = (self._warm_cache[j] if self._warm_cache[j] is not None
                   else np.full(K, gm))
            res = minimize(obj, x0, jac=grad, method='SLSQP',
                           constraints=self._constraints[j],
                           options={'ftol': 1e-12, 'maxiter': 1000})
            th = res.x
            if all(float(c['fun'](th)) >= -1e-8 for c in self._constraints[j]):
                self._warm_cache[j] = th.copy()
                lpl[j] = -float(np.dot(n, (th - mu_hat) ** 2)) / (2.0 * s2)
        return lpl

    def _update_posterior(self, log_pl):
        finite = np.isfinite(log_pl)
        if not np.any(finite):
            raise RuntimeError("All profile-likelihood computations failed.")
        log_pl[~finite] = -np.inf
        log_pl -= np.max(log_pl[finite])
        log_pl  = np.clip(log_pl, -30.0, np.inf)
        w = np.exp(log_pl) * self.prior
        self.posterior = w / w.sum()


# Back-compat aliases
MINTS_Unstructured = lambda *a, **kw: MINTS(*a, unimodal=False, **kw)
MINTS_Unimodal     = lambda *a, **kw: MINTS(*a, unimodal=True,  **kw)



# ======================================================================
# KL-UCB (Gaussian)
# ======================================================================

class KLUCBGaussian:
    """
    KL-UCB algorithm for Gaussian rewards with known variance.

    Index:  B_i(t) = mu_hat_i + sigma * sqrt(2 * log(t) / n_i)

    This is the closed-form Gaussian special case of KL-UCB
    (Garivier & Cappé, 2011). Asymptotically optimal for unstructured
    Gaussian bandits:
        lim inf R(T) / log(T) >= sum_{i != i*} 2 sigma^2 / Delta_i
    """
    name = 'KL-UCB'
    key  = 'klucb'

    def __init__(self, K, reward_std=1.0, seed=None):
        self.K      = K
        self.sigma2 = float(reward_std) ** 2
        self.rng    = np.random.default_rng(seed)
        self.counts = np.zeros(K)
        self.sums   = np.zeros(K)
        self.t      = 0
        self._new_idx = None

    def sample_decision(self):
        self.t += 1
        if self.t <= self.K:
            self._new_idx = self.t - 1
        else:
            mu_hat  = np.where(self.counts > 0,
                               self.sums / self.counts, 0.0)
            indices = np.array([
                _klucb_gaussian_index(mu_hat[i], self.counts[i],
                                      self.t, self.sigma2)
                for i in range(self.K)
            ])
            self._new_idx = int(np.argmax(indices))
        return self._new_idx

    def update_belief(self, reward):
        i = self._new_idx
        self.counts[i] += 1
        self.sums[i]   += reward


# ======================================================================
# Thompson Sampling (Gaussian, known sigma)
# ======================================================================

class ThompsonSampling:
    """
    Gaussian Thompson Sampling with known variance. 
    Jeffreys Prior is used, following the paper "Thompson sampling for 1-dimensional exponential family bandits" by Korda et al., NeurIPS 2013.

    Conjugate posterior (known sigma^2):
        mu_i | data ~ N(S_i / n_i,  sigma^2 / n_i)
    when n_i = 0, mu_i is drawn from a diffuse normal to ensure early exploration.
    """
    name = 'Thompson Sampling'
    key  = 'ts'

    def __init__(self, K, reward_std=1.0, seed=None):
        self.K      = K
        self.sigma  = float(reward_std)
        self.rng    = np.random.default_rng(seed)
        self.counts = np.zeros(K)
        self.sums   = np.zeros(K)
        self._new_idx = None

    def sample_decision(self):
        samples = np.array([
            self.rng.normal(0.0, 1e6) if self.counts[i] == 0
            else self.rng.normal(self.sums[i] / self.counts[i],
                                 self.sigma / np.sqrt(self.counts[i]))
            for i in range(self.K)
        ])
        self._new_idx = int(np.argmax(samples))
        return self._new_idx

    def update_belief(self, reward):
        i = self._new_idx
        self.counts[i] += 1
        self.sums[i]   += reward


# ======================================================================
# OSUB  (Combes & Proutière, ICML 2014)
# ======================================================================

class OSUB:
    """
    Optimal Sampling for Unimodal Bandits on a line-graph.

    Reference: Combes & Proutière,
    "Unimodal Bandits: Regret Lower Bounds and Optimal Algorithms",
    ICML 2014.

    Structure: arms 0,...,K-1 on a line.  N(i) = {i-1, i+1} ∩ {0,...,K-1}.
    Unique optimum i* with mu_0 < ... < mu_{i*} > ... > mu_{K-1}.

    Regret lower bound:
        lim inf R(T)/log(T) >= sum_{j in N(i*)} Delta_j / KL(mu_j, mu*)

    Algorithm each round t:
      1. l = argmax mu_hat_i  (leader)
      2. N+(l) = {l} ∪ N(l)
      3. gamma_l = degree of l on the line graph (1 for endpoints, 2 otherwise)
      4. If l has been pulled >= gamma_l * L_l times while leader:  pull l (exploit)
         Else: pull arm in N+(l) with highest KL-UCB index

    For Gaussian, KL(mu_a, mu_b) = (mu_b - mu_a)^2 / (2 sigma^2), giving
    the closed-form index mu_hat + sigma*sqrt(2 log t / n).
    """
    name = 'OSUB'
    key  = 'osub'

    def __init__(self, K, reward_std=1.0, seed=None):
        self.K      = K
        self.sigma2 = float(reward_std) ** 2
        self.rng    = np.random.default_rng(seed)
        self.counts = np.zeros(K)
        self.sums   = np.zeros(K)
        self.t      = 0
        self.L      = np.zeros(K)            # times arm i was leader
        self.pulls_as_leader = np.zeros(K)   # times leader was pulled while leading
        self._degree = np.full(K, 2)
        self._degree[0] = self._degree[K-1] = 1
        self._new_idx = None

    def _neighbours(self, i):
        nbrs = []
        if i > 0:        nbrs.append(i - 1)
        if i < self.K-1: nbrs.append(i + 1)
        return nbrs

    def sample_decision(self):
        self.t += 1
        if self.t <= self.K:
            self._new_idx = self.t - 1
            return self._new_idx

        mu_hat = self.sums / np.where(self.counts > 0, self.counts, 1.0)
        l      = int(np.argmax(mu_hat))
        self.L[l] += 1

        nplus   = [l] + self._neighbours(l)
        indices = {i: _klucb_gaussian_index(mu_hat[i], self.counts[i],
                                             self.t, self.sigma2)
                   for i in nplus}

        if self.pulls_as_leader[l] >= self._degree[l] * self.L[l]:
            arm = l
        else:
            arm = max(nplus, key=lambda i: indices[i])

        if arm == l:
            self.pulls_as_leader[l] += 1

        self._new_idx = arm
        return arm

    def update_belief(self, reward):
        i = self._new_idx
        self.counts[i] += 1
        self.sums[i]   += reward


# ======================================================================
# UTS  (Paladino et al., AAAI 2017)
# ======================================================================

class UTS:
    """
    Unimodal Thompson Sampling for a line-graph.

    Reference: Paladino, Trovò, Restelli & Gatti,
    "Unimodal Thompson Sampling for Graph-Structured Arms", AAAI 2017.

    Algorithm each round t:
      1. l = argmax mu_hat_i  (leader)
      2. N+(l) = {l} ∪ N(l)
      3. If L_l mod |N+(l)| == 0:  pull l  (forced leader pull)
         Else: draw Thompson samples for arms in N+(l), pull argmax

    Gaussian conjugate posterior (known sigma):
        mu_i | data ~ N(S_i/n_i, sigma^2/n_i)

    Asymptotic regret matches the unimodal lower bound.
    """
    name = 'UTS'
    key  = 'uts'

    def __init__(self, K, reward_std=1.0, seed=None):
        self.K      = K
        self.sigma  = float(reward_std)
        self.rng    = np.random.default_rng(seed)
        self.counts = np.zeros(K)
        self.sums   = np.zeros(K)
        self.t      = 0
        self.L      = np.zeros(K, dtype=int)
        self._new_idx = None

    def _neighbours(self, i):
        nbrs = []
        if i > 0:        nbrs.append(i - 1)
        if i < self.K-1: nbrs.append(i + 1)
        return nbrs

    def _sample(self, i):
        if self.counts[i] == 0:
            return self.rng.normal(0.0, 1e6)
        return self.rng.normal(self.sums[i] / self.counts[i],
                                self.sigma / np.sqrt(self.counts[i]))

    def sample_decision(self):
        self.t += 1
        if self.t <= self.K:
            self._new_idx = self.t - 1
            return self._new_idx

        mu_hat = self.sums / np.where(self.counts > 0, self.counts, 1.0)
        l      = int(np.argmax(mu_hat))
        self.L[l] += 1
        nplus  = [l] + self._neighbours(l)

        if self.L[l] % len(nplus) == 0:
            arm = l
        else:
            arm = max(nplus, key=lambda i: self._sample(i))

        self._new_idx = arm
        return arm

    def update_belief(self, reward):
        i = self._new_idx
        self.counts[i] += 1
        self.sums[i]   += reward


# ======================================================================
# OSSB  (Combes, Magureanu & Proutière, NeurIPS 2017)
# ======================================================================

class OSSB:
    """
    Optimal Sampling for Structured Bandits — unimodal closed-form.

    Reference: Combes, Magureanu & Proutière,
    "Minimal Exploration in Structured Stochastic Bandits", NeurIPS 2017.

    Graves-Lai LP solution for unimodal structure (closed form):
        c(x, theta) = 1{|x - x*| == 1} * 2*sigma^2 / (mu* - mu_x)^2

    Only the two immediate neighbours of the optimal arm require
    Omega(log t) pulls; all other arms have c = 0.

    Algorithm (epsilon, gamma):
      s(t) = non-exploitation round counter
      Each round:
        1. x* = argmax mu_hat
        2. Compute c(x, mu_hat) using closed form above
        3. If N(x,t) >= c(x, mu_hat)*(1+gamma)*log(t) for ALL x:
               pull x*  [exploitation]
        4. Else s += 1:
               X_bar   = arm with smallest N(x) / c(x)*log(t) ratio  [exploration target]
               X_tilde = least-pulled arm                              [estimation target]
               If N(X_tilde) <= epsilon * s:  pull X_tilde  [estimation]
               Else:                          pull X_bar    [exploration]

    Parameters
    ----------
    epsilon : float, default 0.0
        Estimation phase threshold.  With epsilon=0 (the paper's experimental
        setting, Section 7), the estimation phase only triggers for arms that
        have never been pulled (N(X_tilde)=0 <= 0).  Since the algorithm
        initialises with a round-robin that pulls every arm once, epsilon=0
        means estimation never fires after initialisation and all non-
        exploitation rounds go directly to exploration.  epsilon > 0 is used
        in the theoretical analysis to guarantee sufficient estimation but is
        not needed in practice when initialisation already covers all arms.
    gamma : float, default 0.0
        Exploitation trigger slack (0 = exploit as soon as targets met).
    """
    name = 'OSSB'
    key  = 'ossb'

    def __init__(self, K, reward_std=1.0, epsilon=0.0, gamma=0.0, seed=None):
        self.K       = K
        self.sigma2  = float(reward_std) ** 2
        self.epsilon = epsilon
        self.gamma   = gamma
        self.rng     = np.random.default_rng(seed)
        self.counts  = np.zeros(K)
        self.sums    = np.zeros(K)
        self.t       = 0
        self.s       = 0
        self._new_idx = None

    def _exploration_rates(self, mu_hat):
        """c(x, mu_hat) for unimodal line-graph: nonzero only at neighbours of x*."""
        xstar = int(np.argmax(mu_hat))
        c     = np.zeros(self.K)
        mu_star = mu_hat[xstar]
        for x in range(self.K):
            if abs(x - xstar) == 1:
                gap = mu_star - mu_hat[x]
                c[x] = 2.0 * self.sigma2 / gap ** 2 if gap > 1e-10 else np.inf
        return c, xstar

    def sample_decision(self):
        self.t += 1
        if self.t <= self.K:
            self._new_idx = self.t - 1
            return self._new_idx

        mu_hat       = self.sums / np.where(self.counts > 0, self.counts, 1.0)
        c, xstar     = self._exploration_rates(mu_hat)
        log_t        = np.log(self.t)
        target       = c * (1.0 + self.gamma) * log_t

        exploit = np.all((c == 0) | (self.counts >= target))

        if exploit:
            arm = xstar
        else:
            self.s += 1
            active  = c > 1e-12
            if np.any(active):
                ratio   = np.where(active,
                                   self.counts / (c * log_t + 1e-300),
                                   np.inf)
                X_bar   = int(np.argmin(ratio))
            else:
                X_bar   = xstar
            X_tilde = int(np.argmin(self.counts))
            arm     = X_tilde if self.counts[X_tilde] <= self.epsilon * self.s \
                      else X_bar

        self._new_idx = arm
        return arm

    def update_belief(self, reward):
        i = self._new_idx
        self.counts[i] += 1
        self.sums[i]   += reward


# ======================================================================
# Registries
# ======================================================================

#: All algorithms in canonical order: {key: class}
ALL_ALGORITHMS = {
    'mints_u': lambda *a, **kw: MINTS(*a, unimodal=False, **kw),
    'mints_m': lambda *a, **kw: MINTS(*a, unimodal=True,  **kw),
    'klucb':   KLUCBGaussian,
    'ts':      ThompsonSampling,
    'osub':    OSUB,
    'uts':     UTS,
    'ossb':    OSSB,
}

#: Algorithm keys that exploit the unimodal structure
UNIMODAL_ALGORITHMS = {'mints_m', 'osub', 'uts', 'ossb'}

#: Algorithm keys that use no structural assumption
UNSTRUCTURED_ALGORITHMS = {'mints_u', 'klucb', 'ts'}

