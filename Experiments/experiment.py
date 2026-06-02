"""
experiment.py
=============
Large-scale experiment infrastructure for comparing bandit algorithms.

Designed for distributed execution across multiple Google Colab sessions,
with incremental saving and resume support.

Typical workflow
----------------
Run 4 Colab sessions in parallel (100 seeds total, 25 per session):

    # Session 0  (change shard=0,1,2,3 in each session)
    from experiment import run_colab_shard
    run_colab_shard(shard=0)

After all sessions finish, merge and plot:

    from experiment import merge_and_plot_colab
    summary = merge_and_plot_colab()

File format (.npz)
------------------
    seeds                 int (n_done,)
    cum_regret_<key>      float (n_done, T)  for each algorithm key
    theta_true            float (K,)
    alg_names             str   comma-separated list of algorithm keys
    T, reward_std, K      scalars
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from bandits import ALL_ALGORITHMS, MINTS_Unstructured, MINTS_Unimodal


# ======================================================================
# Single-replicate runner
# ======================================================================

def run_one_seed(theta_true, T, reward_std, seed, algorithms=None):
    """
    Run one replicate for all algorithms on the SAME reward trajectory.

    All algorithms share a single environment seed (and thus the same
    (T, K) reward table), while each gets its own independent algorithm
    seed.  This eliminates environment variance from comparisons.

    Parameters
    ----------
    theta_true  : array-like (K,)
    T           : int
    reward_std  : float
    seed        : int
    algorithms  : dict {key: class} or None
        Algorithms to run.  Defaults to ALL_ALGORITHMS from bandits.py.
        Each class must accept (K, reward_std, seed) and expose
        sample_decision() / update_belief(reward).

    Returns
    -------
    dict {key: cumulative_regret_array (T,)}
    """
    if algorithms is None:
        algorithms = ALL_ALGORITHMS

    theta_true = np.asarray(theta_true, float)
    K          = len(theta_true)
    theta_max  = theta_true.max()

    # Spawn independent seeds: [env, alg_0, alg_1, ...]
    n_algs   = len(algorithms)
    spawned  = np.random.SeedSequence(seed).spawn(1 + n_algs)
    env_seed = spawned[0]
    alg_seeds = spawned[1:]

    rewards = np.random.default_rng(env_seed).normal(
        loc=theta_true, scale=reward_std, size=(T, K))

    # Instantiate
    algs = {key: cls(K=K, reward_std=reward_std, seed=alg_seeds[i])
            for i, (key, cls) in enumerate(algorithms.items())}

    # Storage
    reg = {key: np.empty(T) for key in algs}

    # Simulate
    for t in range(T):
        for key, alg in algs.items():
            a = alg.sample_decision()
            alg.update_belief(rewards[t, a])
            reg[key][t] = theta_max - theta_true[a]

    return {key: np.cumsum(arr) for key, arr in reg.items()}


# ======================================================================
# Shard runner  (one Colab session)
# ======================================================================

def run_shard(theta_true, T, reward_std, n_seeds, shard, n_shards,
              seed_base, out_path, algorithms=None,
              verbose=True, resume=True):
    """
    Run the seeds assigned to this shard and save results incrementally.

    Seed assignment: shard s receives seeds with index ≡ s (mod n_shards),
    i.e. [seed_base+s, seed_base+s+n_shards, seed_base+s+2*n_shards, ...].

    Results are saved to `out_path` after every completed seed (overwrite),
    so a session crash never loses more than one seed of work.

    Resume (resume=True): if `out_path` already exists, completed seeds are
    loaded back in and skipped.  Re-run the same call to resume seamlessly.

    Parameters
    ----------
    theta_true  : array-like (K,)
    T           : int
    reward_std  : float
    n_seeds     : int    total seeds across all shards
    shard       : int    0-based index of this shard
    n_shards    : int    total number of shards
    seed_base   : int    first seed value
    out_path    : str    path for .npz output file
    algorithms  : dict {key: class} or None  (defaults to ALL_ALGORITHMS)
    verbose     : bool
    resume      : bool
    """
    if algorithms is None:
        algorithms = ALL_ALGORITHMS

    theta_true = np.asarray(theta_true, float)
    K          = len(theta_true)
    alg_names  = list(algorithms.keys())

    # Assign seeds for this shard
    all_seeds = [seed_base + i for i in range(n_seeds)]
    my_seeds  = [s for i, s in enumerate(all_seeds) if i % n_shards == shard]
    n_mine    = len(my_seeds)

    # Allocate storage for all algorithms
    cum = {name: np.full((n_mine, T), np.nan) for name in alg_names}
    done_seeds = []

    # ------------------------------------------------------------------
    # Resume: reload any completed seeds from existing file
    # ------------------------------------------------------------------
    if resume and os.path.exists(out_path):
        try:
            prev       = np.load(out_path, allow_pickle=False)
            prev_seeds = list(prev['seeds'].astype(int))
            if (int(prev['T']) != T
                    or not np.allclose(prev['theta_true'], theta_true)
                    or float(prev['reward_std']) != reward_std):
                raise ValueError("Saved file has incompatible experiment config.")
            for prev_idx, s in enumerate(prev_seeds):
                if s in my_seeds:
                    slot = my_seeds.index(s)
                    for name in alg_names:
                        key = f'cum_regret_{name}'
                        if key in prev.files:
                            cum[name][slot] = prev[key][prev_idx]
                    done_seeds.append(s)
            if verbose and done_seeds:
                print(f"Resuming shard {shard}: "
                      f"{len(done_seeds)}/{n_mine} seeds already done, "
                      f"skipping {done_seeds}")
        except Exception as e:
            if verbose:
                print(f"Warning: could not load existing file ({e}). Starting fresh.")
            cum        = {name: np.full((n_mine, T), np.nan) for name in alg_names}
            done_seeds = []

    done_set = set(done_seeds)

    if verbose:
        remaining = [s for s in my_seeds if s not in done_set]
        print(f"Shard {shard}/{n_shards}: {n_mine} seeds total, "
              f"{len(remaining)} remaining  →  {out_path}")

    t_shard            = time.perf_counter()
    completed_this_run = 0

    for idx, seed in enumerate(my_seeds):
        if seed in done_set:
            continue

        t0     = time.perf_counter()
        result = run_one_seed(theta_true, T, reward_std, seed, algorithms)
        elapsed = time.perf_counter() - t0

        for name in alg_names:
            cum[name][idx] = result[name]
        done_seeds.append(seed)
        done_set.add(seed)
        completed_this_run += 1

        # Save incrementally (overwrite, only completed rows)
        finished_slots = [i for i, s in enumerate(my_seeds) if s in done_set]
        finished_seeds = [my_seeds[i] for i in finished_slots]
        save_dict = dict(
            seeds      = np.array(finished_seeds, dtype=int),
            theta_true = theta_true,
            T          = np.int64(T),
            reward_std = np.float64(reward_std),
            K          = np.int64(K),
            alg_names  = ','.join(alg_names),
        )
        for name in alg_names:
            save_dict[f'cum_regret_{name}'] = cum[name][finished_slots]
        np.savez(out_path, **save_dict)

        if verbose:
            n_done = len(done_seeds)
            eta    = ((time.perf_counter() - t_shard) / completed_this_run
                      * (n_mine - n_done)) if completed_this_run > 0 else 0.0
            print(f"  seed {seed:5d}  [{n_done:3d}/{n_mine}]  "
                  f"{elapsed:.1f}s  ETA {eta/60:.1f}min")

    total = time.perf_counter() - t_shard
    if verbose:
        if completed_this_run == 0:
            print(f"Shard {shard}: nothing to do (all seeds already complete).")
        else:
            print(f"Shard {shard} done in {total/60:.1f}min  →  {out_path}")

    return cum, np.array(done_seeds, dtype=int)


# ======================================================================
# Merge
# ======================================================================

def merge_shards(shard_paths, out_path=None):
    """
    Merge results from multiple shard files into a single .npz file.

    Parameters
    ----------
    shard_paths : list of str
    out_path    : str or None

    Returns
    -------
    dict with all arrays, sorted by seed value
    """
    shards    = [np.load(p, allow_pickle=False) for p in shard_paths]
    theta_ref = shards[0]['theta_true']
    T_ref     = int(shards[0]['T'])
    std_ref   = float(shards[0]['reward_std'])
    for i, s in enumerate(shards[1:], 1):
        assert np.allclose(s['theta_true'], theta_ref), \
            f"theta_true mismatch in shard {i}"
        assert int(s['T']) == T_ref,  f"T mismatch in shard {i}"
        assert float(s['reward_std']) == std_ref, \
            f"reward_std mismatch in shard {i}"

    names_str = str(shards[0]['alg_names'])
    alg_names = names_str.split(',')

    all_seeds = np.concatenate([s['seeds'] for s in shards])
    order     = np.argsort(all_seeds)
    all_seeds = all_seeds[order]

    result = dict(seeds=all_seeds, theta_true=theta_ref,
                  T=T_ref, reward_std=std_ref, K=int(shards[0]['K']),
                  alg_names=names_str)
    for name in alg_names:
        key = f'cum_regret_{name}'
        arr = np.concatenate([s[key] for s in shards], axis=0)
        result[key] = arr[order]

    if out_path is None:
        base     = shard_paths[0]
        out_path = base[:base.rfind('shard')] + 'merged.npz'

    np.savez(out_path, **result)
    print(f"Merged {len(all_seeds)} seeds from {len(shard_paths)} shards "
          f"→ {out_path}")
    return result


def load_results(path):
    """Load a .npz results file (shard or merged) into a plain dict."""
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


# ======================================================================
# Summary statistics
# ======================================================================

def summarize(results):
    """
    Compute mean and 95% CI half-widths for every algorithm.

    Parameters
    ----------
    results : dict (from load_results or merge_shards)

    Returns
    -------
    dict with keys:
        t_grid              (T,)
        alg_names           list[str]
        mean_<key>          (T,)   for each algorithm
        ci95_<key>          (T,)   95% CI half-width
        theory_u, theory_m  (T,)   asymptotic lower bounds
        n_seeds, T, reward_std, K, theta_true
    """
    T          = int(results['T'])
    alg_names  = str(results['alg_names']).split(',')
    first_key  = f'cum_regret_{alg_names[0]}'
    n_seeds    = results[first_key].shape[0]

    t_grid, th_u, th_m = theory_curves(
        results['theta_true'], T, float(results['reward_std']))

    out = dict(t_grid=t_grid, theory_u=th_u, theory_m=th_m,
               alg_names=alg_names, n_seeds=n_seeds, T=T,
               reward_std=float(results['reward_std']),
               K=int(results['K']), theta_true=results['theta_true'])

    for name in alg_names:
        arr  = results[f'cum_regret_{name}']
        m    = arr.mean(0)
        se   = arr.std(0, ddof=1) / np.sqrt(n_seeds)
        out[f'mean_{name}'] = m
        out[f'ci95_{name}'] = 1.96 * se

    return out


# ======================================================================
# Asymptotic lower bounds
# ======================================================================

def theory_curves(theta_true, T, reward_std=1.0):
    """
    Asymptotic Lai-Robbins lower bounds (× log t):

    Unstructured:  C_u = sum_{j != j*}  2 sigma^2 / Delta_j
    Unimodal:      C_m = sum_{j neighbour of j*}  2 sigma^2 / Delta_j

    Returns (t_grid, C_u * log(t_grid), C_m * log(t_grid)).
    """
    theta_true = np.asarray(theta_true, float)
    Delta      = theta_true.max() - theta_true
    idx_opt    = int(theta_true.argmax())
    s2         = reward_std ** 2
    t_grid     = np.arange(1, T + 1)
    logt       = np.log(t_grid)
    c_u = np.sum(2.0 * s2 / Delta[Delta > 0])
    c_m = sum(2.0 * s2 / Delta[j]
              for j in range(len(theta_true))
              if j != idx_opt and abs(j - idx_opt) == 1 and Delta[j] > 0)
    return t_grid, c_u * logt, c_m * logt


def graves_lai_lower_bound(theta_true, reward_std=1.0,
                           lipschitz_L=None, n_iter=60):
    """
    Compute the Graves-Lai asymptotic lower bound constant C(theta)
    by solving the semi-infinite LP numerically.

    The lower bound is C(theta) * log(T).

    For pure unimodal (lipschitz_L=None) uses the closed-form:
        C = sum_{j in N(i*)} Delta_j / KL(theta_j, theta_{i*})

    For unimodal + Lipschitz-L, solves the Graves-Lai LP iteratively:
        min_{eta >= 0}  sum_i eta_i * Delta_i
        s.t.  sum_i eta_i * KL(theta_i, lambda_i) >= 1
              for all lambda in Lambda_j(theta), each j != i*
    where Lambda_j = {lambda : unimodal at j, Lipschitz-L,
                               lambda_{i*} = theta_{i*}}.
    The inner problem (find worst lambda) is a convex QP solved by SLSQP.

    Parameters
    ----------
    theta_true  : array (K,)
    reward_std  : float
    lipschitz_L : float or None
    n_iter      : int   LP iterations

    Returns
    -------
    C : float   lower bound constant (R(T) >= C * log T asymptotically)
    """
    from scipy.optimize import minimize, linprog

    theta  = np.asarray(theta_true, float)
    K      = len(theta)
    sigma2 = float(reward_std) ** 2
    xstar  = int(np.argmax(theta))
    Delta  = theta[xstar] - theta

    # Closed form for pure unimodal (no Lipschitz):
    #   C = sum_{j in N(i*)} Delta_j / KL(theta_j, theta_{i*})
    #     = sum_{j in N(i*)} 2*sigma^2 / Delta_j
    if lipschitz_L is None:
        return sum(2.0 * sigma2 / Delta[j]
                   for j in range(K)
                   if j != xstar and abs(j-xstar) == 1 and Delta[j] > 0)

    L = float(lipschitz_L)
    candidates = [j for j in range(K) if j != xstar]

    def worst_lambda(j, eta):
        def obj(lam):  return float(np.dot(eta, (lam-theta)**2)) / (2.*sigma2)
        def grd(lam):  return eta*(lam-theta) / sigma2
        cs = []
        r = np.zeros(K); r[xstar] = 1.
        cs.append({'type':'eq','fun':lambda lam,v=theta[xstar],r=r: r@lam-v,
                   'jac':lambda lam,r=r:r})
        r2 = np.zeros(K); r2[j] = 1.
        cs.append({'type':'ineq','fun':lambda lam,r=r2: r@lam-theta[xstar],
                   'jac':lambda lam,r=r2:r})
        for i in range(j):
            r3=np.zeros(K);r3[i+1]=1.;r3[i]=-1.
            cs.append({'type':'ineq','fun':lambda lam,r=r3:r@lam,'jac':lambda lam,r=r3:r})
        for i in range(j,K-1):
            r3=np.zeros(K);r3[i]=1.;r3[i+1]=-1.
            cs.append({'type':'ineq','fun':lambda lam,r=r3:r@lam,'jac':lambda lam,r=r3:r})
        for i in range(K-1):
            r3=np.zeros(K);r3[i+1]=1.;r3[i]=-1.
            cs.append({'type':'ineq','fun':lambda lam,r=r3:L-r@lam,'jac':lambda lam,r=r3:-r})
            r4=np.zeros(K);r4[i]=1.;r4[i+1]=-1.
            cs.append({'type':'ineq','fun':lambda lam,r=r4:L-r@lam,'jac':lambda lam,r=r4:-r})
        res = minimize(obj, theta.copy(), jac=grd, method='SLSQP',
                       constraints=cs, options={'ftol':1e-13,'maxiter':2000})
        return res.x

    eta = np.ones(K) / K
    for _ in range(n_iter):
        kl_mat = np.zeros((len(candidates), K))
        for idx, j in enumerate(candidates):
            lam = worst_lambda(j, eta)
            kl_mat[idx] = (lam - theta)**2 / (2.*sigma2)
        res_lp = linprog(Delta, A_ub=-kl_mat, b_ub=-np.ones(len(candidates)),
                         bounds=[(0.,None)]*K, method='highs')
        if res_lp.success:
            eta = res_lp.x

    return float(res_lp.fun) if res_lp.success else np.nan


# ======================================================================
# Plotting
# ======================================================================

# Visual style for each algorithm key
# _ALG_STYLE defines colours and linestyles used in the regret curve plots
# (where all algorithms appear together and must be globally distinct) and
# in the empirical CDF plots (where algorithms appear in separate group panels).
#
# CDF color convention (per user's request):
#   Unstructured panel — MINTS: blue,  UCB: red,  Thompson Sampling: gold
#   Unimodal panel     — MINTS: blue,  OSUB: red,    UTS: gold,  OSSB: cyan
#
# Regret curve colours: globally distinct so all 7 curves are separable.
#   MINTS variants stay in the blue family (solid = unstr., dashed = unimodal).
#   Other algorithms use the CDF colors where possible; where two
#   algorithms share the same CDF colour (UCB and OSUB both red,
#   TS and UTS both gold), they are distinguished by linestyle.
_ALG_STYLE = {
    'mints_u':  ('blue',       '-',   'MINTS (unstructured)'),
    'mints_m':  ('blue',       '--',  'MINTS (unimodal)'),
    'mints_ul': ('blue',       ':',   'MINTS (uni+Lip)'),
    'klucb':    ('red',        '-',   'UCB'),
    'ts':       ('goldenrod',  '-',   'Thompson Sampling'),
    'osub':     ('red',        '--',  'OSUB'),
    'uts':      ('cyan',       '--',  'UTS'),
    'ossb':     ('gold',       '-',   'OSSB'),
    'ossb_ul':  ('gold',       '--',  'OSSB (uni+Lip)'),
}
_DEFAULT_STYLE = ('grey', '-', None)


def alg_style(key):
    """Return (colour, linestyle, label) for algorithm key."""
    c, ls, lbl = _ALG_STYLE.get(key, _DEFAULT_STYLE)
    return c, ls, lbl or key


def _strip_img_suffix(path):
    """Remove .pdf or .png suffix if present, otherwise return unchanged."""
    for ext in ('.pdf', '.png'):
        if path.endswith(ext):
            return path[:-len(ext)]
    return path


def print_summary_table(summary):
    """Print mean ± 95% CI for all algorithms at four checkpoints."""
    T     = summary['T']
    names = summary['alg_names']
    chk   = [T//4, T//2, T]
    col_w = 24

    _, _, labels = zip(*[alg_style(n) for n in names])
    width = 8 + col_w * len(names)

    print(f"\n{'':=<{width}}")
    print(f"  Summary — {summary['n_seeds']} seeds, T={T}, "
          f"K={summary['K']}, sigma={summary['reward_std']}")
    print(f"  theta = {summary['theta_true']}")
    print(f"{'':=<{width}}")
    print(f"\n  {'t':>6}  "
          + "  ".join(f"{lbl:>{col_w}}" for lbl in labels))
    print("  " + "-" * (width - 2))
    for t in chk:
        i   = t - 1
        row = f"  {t:>6}  "
        for name in names:
            m  = summary[f'mean_{name}'][i]
            ci = summary[f'ci95_{name}'][i]
            row += f"{m:>8.1f} ± {ci:<6.1f}{'':>{col_w-18}}  "
        print(row)
    print(f"{'':=<{width}}\n")


# ======================================================================
# CDFs of per-seed regret distributions
# ======================================================================

def plot_cdfs(results, summary, out_path=None):
    """
    Regret distribution plots at checkpoints.

    Layout: 3 rows (T/4, T/2, T) × 2 columns (unstructured group
    on the left, unimodal group on the right).  Within each panel all
    algorithms in that group are overlaid as ECDF curves.  Two vertical
    dashed lines mark the asymptotic lower bounds for the unstructured
    and unimodal cases.

    Grouping keeps the x-scales comparable within each panel:
      Unstructured group — algorithms that use no structural assumption
                           (MINTS-unstructured, UCB, Thompson Sampling)
      Unimodal group     — algorithms that exploit unimodal structure
                           (MINTS-unimodal, OSUB, UTS, OSSB, ...)

    Any algorithm key not found in the registry falls into the unimodal
    group by default.

    Parameters
    ----------
    results  : dict from load_results() or merge_shards()
    summary  : dict from summarize()
    out_path : str or None — if given, saves <out_path>_cdf.pdf/.png
    """
    from bandits import UNSTRUCTURED_ALGORITHMS   # set of unstructured keys

    T       = int(results['T'])
    names   = summary['alg_names']
    chk     = [T//4, T//2, T]
    chk_idx = [t - 1 for t in chk]
    th_u    = summary['theory_u']
    th_m    = summary['theory_m']

    # Partition algorithms into the two groups, preserving order
    unstruct = [n for n in names if n in UNSTRUCTURED_ALGORITHMS]
    unimodal = [n for n in names if n not in UNSTRUCTURED_ALGORITHMS]

    fig, axes = plt.subplots(nrows=len(chk), ncols=2, figsize=(11, 7.5), squeeze=False)

    groups = [
        (unstruct, 'Unstructured algorithms'),
        (unimodal, 'Unimodal algorithms'),
    ]

    # ── Pass 1: compute a unified x-limit per column (across all checkpoints).
    # This keeps the horizontal axis fixed across rows so the distributions are
    # directly comparable over time. The cap at 2 × LB_unstr.(T) prevents one
    # volatile algorithm from squashing the rest; the floor at 1.1 × LB_unstr.(T)
    # guarantees both lower-bound lines are always visible.
    lb_u_at_T = float(th_u[chk_idx[-1]])   # LB_unstr at the final checkpoint
    col_xmax = []
    for col, (group, _) in enumerate(groups):
        # Collect values across ALL checkpoints for this group
        all_vals = np.concatenate([
            results[f'cum_regret_{n}'][:, idx]
            for n in group for idx in chk_idx
        ])
        obs_max = float(all_vals.max())
        xmax    = min(obs_max, 2.0 * lb_u_at_T)
        xmax    = max(xmax, 1.1 * lb_u_at_T)   # floor: LB lines always visible
        col_xmax.append(xmax)

    # ── Pass 2: draw CDFs, LB lines, labels using the fixed x-limits.
    for row, (t, idx) in enumerate(zip(chk, chk_idx)):
        lb_u_val = float(th_u[idx])
        lb_m_val = float(th_m[idx])

        for col, (group, group_label) in enumerate(groups):
            ax   = axes[row, col]
            xmax = col_xmax[col]

            # CDF of each algorithm
            x_grid = np.linspace(0, xmax, 400)
            for name in group:
                color, _, label = alg_style(name)
                vals = np.sort(results[f'cum_regret_{name}'][:, idx].astype(float))
                ecdf = np.arange(1, len(vals) + 1) / len(vals)
                ax.step(vals, ecdf, color=color, lw=1.8, label=label, where='post')

            # Theoretical lower bound lines: black solid = unstructured LB,
            # black dashed = unimodal LB. Lines shift right across rows as t
            # increases, showing the growing lower bound. No value annotations.
            ax.axvline(lb_u_val, color='black', lw=1.4, ls='-',
                       #label='LB unstructured'
                       )
            ax.axvline(lb_m_val, color='black', lw=1.4, ls='--',
                       #label='LB unimodal'
                       )

            ax.set_xlim(0, xmax)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.25, lw=0.5)

            if col == 0:
                ax.set_ylabel(f't = {t}', fontsize=9)
            else:
                ax.set_ylabel('', fontsize=9)
            # ax.set_xlabel('Cumulative regret', fontsize=9)

            if row == 0:
                ax.set_title(group_label, fontsize=10, fontweight='bold')
                ax.legend(fontsize=7.5, framealpha=0.9, loc='upper right')

    n     = summary['n_seeds']
    theta = summary['theta_true']
    tsstr = ', '.join(f'{x:.2g}' for x in theta)
    plt.tight_layout()

    if out_path is not None:
        base = _strip_img_suffix(out_path)
        fig.savefig(base + '_cdf.pdf', dpi=150, bbox_inches='tight')
        fig.savefig(base + '_cdf.png', dpi=150, bbox_inches='tight')
        print(f"Saved {base}_cdf.pdf  and  {base}_cdf.png")

    return fig


# ======================================================================
# Filename helpers
# ======================================================================

def shard_filename(out_dir, shard, n_shards, tag=''):
    suffix = f'_{tag}' if tag else ''
    return os.path.join(out_dir,
                        f'results_shard{shard}_of{n_shards}{suffix}.npz')


def merged_filename(out_dir, n_shards, tag=''):
    suffix = f'_{tag}' if tag else ''
    return os.path.join(out_dir,
                        f'results_merged_{n_shards}shards{suffix}.npz')


# ======================================================================
# Colab convenience functions
# ======================================================================

def run_colab_shard(shard, n_shards=4, n_seeds=100, seed_base=0,
                    theta_true=None, T=2000, reward_std=1.0,
                    algorithms=None,
                    out_dir='/content/drive/MyDrive/mints_results',
                    tag='', resume=True):
    """
    One-liner for Google Colab.  Change only `shard` in each session.

    resume=True (default): re-running the same cell after a disconnect
    automatically skips already-completed seeds.

    Example:
        from experiment import run_colab_shard
        run_colab_shard(shard=0)   # Session 0 of 4
    """
    if theta_true is None:
        theta_true = [1.0, 1.5, 2.0, 1.2, 0.5]
    os.makedirs(out_dir, exist_ok=True)
    run_shard(
        theta_true = np.array(theta_true, float),
        T          = T,
        reward_std = reward_std,
        n_seeds    = n_seeds,
        shard      = shard,
        n_shards   = n_shards,
        seed_base  = seed_base,
        out_path   = shard_filename(out_dir, shard, n_shards, tag),
        algorithms = algorithms,
        verbose    = True,
        resume     = resume,
    )


def merge_and_plot_colab(n_shards=4,
                         out_dir='/content/drive/MyDrive/mints_results',
                         tag='', save_plot=True):
    """
    Merge all shard files and plot after all Colab sessions finish.

    Produces one figure:
        Empirical CDFs of per-seed regret at T/4, T/2, T,
        showing the full distributional shape across seeds.

    Example:
        from experiment import merge_and_plot_colab
        summary = merge_and_plot_colab()
    """
    paths = sorted([
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.startswith('results_shard') and f.endswith('.npz')
        and f'_of{n_shards}' in f and tag in f
    ])
    print(f"Found {len(paths)} shard files:")
    for p in paths:
        d = np.load(p)
        print(f"  {p}  ({len(d['seeds'])} seeds)")

    out_merged = merged_filename(out_dir, n_shards, tag)
    results    = merge_shards(paths, out_path=out_merged)
    summary    = summarize(results)
    print_summary_table(summary)

    out_plot = out_merged.replace('.npz', '_plot') if save_plot else None
    fig_cdf = plot_cdfs(results, summary, out_path=out_plot)
    plt.show()
    return summary

