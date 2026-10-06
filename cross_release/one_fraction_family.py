#!/usr/bin/env python3
"""One noise fraction common to all bins: the model-based DR1+DR2 combination of Section 4.1
and Appendix B.2, and the checks around it -> results/numbers.json.

Everything is computed with the analysis code in code/ (eprocess_joint.py, evalue_analysis.py,
eprocess_hierarchical_mc.py); nothing here re-implements the e-value kernels.

Sections written
  anchors              per-release values, the equal-weight average, the alpha_1 = 1/3 member and
                       reference points of the family; the script stops if any anchor fails
  null_test            the DR2 - DR1 null test on the 11 matched quantities; variance ratios
  alpha_max            the largest admissible fraction (smallest generalized eigenvalue)
  family_scan          a fine scan of M_joint(alpha) over [0, alpha_max) and its continuum worst case
  calibration_budget   the coherent-offset budget (alpha = 1/3 and family worst case, Table 3) and
                       the data-allowed offset region of shift_bound.py (run via runpy)
  simulated_fpr        simulated false-positive rates under calibration offsets, read from the
                       outputs of record of simulations/fpr_calibration_offsets.py and fpr_crossing.py
  simulations_context  false-positive rates in four cross-release worlds and power under the DR2
                       MLE (outputs of record of the other scripts in simulations/)
  series               arrays used by Figure 4 and the budget curves
  figures              the four figures of the paper and the numbers each one reads
  meta

Run (from the repository root, about 20 s): python cross_release/one_fraction_family.py
Reads cross_release/inputs/ (simulation outputs of record) and runs cross_release/shift_bound.py.
"""
import contextlib
import io
import json
import re
import runpy
import sys
import time
from pathlib import Path

import numpy as np
from scipy import linalg, optimize, stats
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import (ROOT, CROSS_RELEASE, INPUTS, OUT_DIR, NUMBERS, add_code_path,  # noqa: E402
                    load_numbers, save_numbers)
add_code_path()
REPO = ROOT                  # repository root: code/ and data/
LOGS = INPUTS / 'logs'       # logs of record of the simulation scripts
OUT = NUMBERS                # results/numbers.json
THIS = 'cross_release/one_fraction_family.py'
OUT_DIR.mkdir(parents=True, exist_ok=True)

from data_loader import load_desi_data                          # noqa: E402
from evalue_analysis import precompute_kernels, mixture_log_e_from_residuals  # noqa: E402
from eprocess_hierarchical_mc import build_bin_matching, nearest_psd, is_psd  # noqa: E402
import eprocess_joint as EJ                                     # noqa: E402

t_start = time.time()
THR = 20.0
LOG_THR = np.log(THR)
NUM = {}


def rec(section, key, value, definition='(text entry)', script=THIS, **extra):
    d = dict(value=value, definition=definition, script=script)
    d.update(extra)
    NUM.setdefault(section, {})[key] = d


def to_py(x):
    if isinstance(x, dict):
        return {k: to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_py(v) for v in x]
    if isinstance(x, np.ndarray):
        return [to_py(v) for v in x.tolist()]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None if np.isnan(x) else ('inf' if x > 0 else '-inf')
    return x


# ---------------------------------------------------------------- data
ds1 = load_desi_data(REPO / 'data' / 'dr1', 'DR1')
ds2 = load_desi_data(REPO / 'data' / 'dr2', 'DR2')
K1 = precompute_kernels(ds1.z_eff, ds1.quantities, ds1.cov, EJ.W0_GRID, EJ.WA_GRID)
K2 = precompute_kernels(ds2.z_eff, ds2.quantities, ds2.cov, EJ.W0_GRID, EJ.WA_GRID)
_, delta1 = EJ.build_delta_matrix(ds1.z_eff, ds1.quantities, EJ.W0_GRID, EJ.WA_GRID)
_, delta2 = EJ.build_delta_matrix(ds2.z_eff, ds2.quantities, EJ.W0_GRID, EJ.WA_GRID)
dr2_to_dr1, _ = build_bin_matching(ds1, ds2)
pairs = [(i, j) for j, i in enumerate(dr2_to_dr1) if i >= 0]
m1 = np.array([i for i, _ in pairs], dtype=int)
m2 = np.array([j for _, j in pairs], dtype=int)
u2 = np.array([j for j in range(len(ds2.data)) if dr2_to_dr1[j] < 0], dtype=int)
n_match, n_u = len(pairs), len(u2)
C1M = ds1.cov[np.ix_(m1, m1)]
C2M = ds2.cov[np.ix_(m2, m2)]
sig2 = np.sqrt(np.diag(C2M))                       # published DR2 sigmas, matched
cov_u = nearest_psd(ds2.cov[np.ix_(u2, u2)])[0] + 1e-14 * np.eye(n_u)
eps1_obs = ds1.data - K1['mu_null']
eps2_obs = ds2.data - K2['mu_null']
G = len(EJ.W0_GRID) * len(EJ.WA_GRID)

TRACER = {0.295: 'BGS', 0.510: 'LRG1', 0.706: 'LRG2', 0.934: 'LRG3+ELG1',
          1.321: 'ELG2', 1.484: 'QSO', 2.330: 'Lya'}
QNAME = {'DV_over_rs': 'DV', 'DM_over_rs': 'DM', 'DH_over_rs': 'DH'}
labels = [f"{TRACER[round(float(ds2.z_eff[j]), 3)]} {QNAME[ds2.quantities[j]]}"
          for j in m2]
zlab = [round(float(ds2.z_eff[j]), 3) for j in m2]


def construct(a):
    """Exactly the kernel path of EJ.run_construction at weight a."""
    cov_y23, min_eig, psd = EJ.decomposition_for_alpha(C1M, C2M, a)
    if not psd:
        return None
    cov_y23 = cov_y23 + 1e-14 * np.eye(n_match)
    Sy = (1.0 - a) ** 2 * cov_y23
    kern = EJ.joint_kernel(delta1, delta2, m1, m2, u2, ds1.cov, Sy, cov_u, a)
    y_obs = eps2_obs[m2] - a * eps1_obs[m1]
    logM = float(EJ.joint_log_e(eps1_obs[:, None], y_obs[:, None],
                                eps2_obs[u2][:, None], kern)[0])
    return dict(alpha=a, kern=kern, Sy=Sy, logM=logM, min_eig=min_eig)


def logM_at(a):
    r = construct(a)
    return np.inf if r is None else r['logM']


# ================================================================ 1. anchors
log_M1 = float(mixture_log_e_from_residuals(eps1_obs[:, None], K1)[0])
log_M2 = float(mixture_log_e_from_residuals(eps2_obs[:, None], K2)[0])
M1, M2 = np.exp(log_M1), np.exp(log_M2)
res13 = construct(1.0 / 3.0)
MJ13 = np.exp(res13['logM'])

# MC under the alpha=1/3 model (same call and seed as code/eprocess_joint.py)
mc13 = EJ.run_construction(ds1, ds2, 1.0 / 3.0, K1, K2, delta1, delta2,
                           n_mc=100000, seed=20260711)
p_sup20 = mc13['mc']['sup_tail']['20']['p_emp']
n_sup20 = mc13['mc']['sup_tail']['20']['n_exceed']


def mc_components(a=1.0 / 3.0, n_mc=100000, seed=20260711, batch=5000):
    """Replicates run_construction's H0 draws (same seed/order) and returns
    per-draw log M_DR1 and log M_joint so single-statistic rates can be read off."""
    r = construct(a)
    cov_y23 = r['Sy'] / (1 - a) ** 2
    rng = np.random.default_rng(seed)
    L1 = K1['L_chol']; Ly = np.linalg.cholesky(cov_y23); Lu = np.linalg.cholesky(cov_u)
    lm1, lmj = np.empty(n_mc), np.empty(n_mc)
    done = 0
    while done < n_mc:
        bs = min(batch, n_mc - done)
        e1 = L1 @ rng.standard_normal(size=(len(ds1.data), bs))
        e23 = Ly @ rng.standard_normal(size=(n_match, bs))
        eu = Lu @ rng.standard_normal(size=(n_u, bs))
        lm1[done:done + bs] = mixture_log_e_from_residuals(e1, K1)
        lmj[done:done + bs] = EJ.joint_log_e(e1, (1 - a) * e23, eu, r['kern'])
        done += bs
    return lm1, lmj


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


lm1_mc, lmj_mc = mc_components()
k_sup = int((np.maximum(lm1_mc, lmj_mc) >= LOG_THR).sum())
k_j = int((lmj_mc >= LOG_THR).sum())
k_1 = int((lm1_mc >= LOG_THR).sum())
assert k_sup == n_sup20, (k_sup, n_sup20)

SCAN_POINTS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 1 / 3, 0.36, 0.40,
             0.42, 0.44, 0.45, 0.46]
SCAN_EXPECTED = [734.05, 466.92, 246.16, 147.06, 98.59, 74.45, 65.01, 66.71,
             75.82, 131.42, 253.59, 1142.67, 7137, 1.7e6]
scan_points = {f"{a:.4f}": float(np.exp(logM_at(a))) for a in SCAN_POINTS}

d_obs = ds2.data[m2] - ds1.data[m1]


def null_chi2(a):
    S = (1 - 2 * a) * C1M + C2M
    c2 = float(d_obs @ np.linalg.solve(S, d_obs))
    return c2, float(stats.chi2.sf(c2, n_match))


S13 = C1M / 3.0 + C2M
pulls = d_obs / np.sqrt(np.diag(S13))
ratios = np.diag(C2M) / np.diag(C1M)
chi2_13, p_13 = null_chi2(1 / 3)
chi2_25, p_25 = null_chi2(0.25)
chi2_40, p_40 = null_chi2(0.40)
chi2_ind, p_ind = null_chi2(0.0)                    # Cov = 0 -> Var = C1 + C2
chi2_06 = float(d_obs @ np.linalg.solve(C1M + C2M - 1.2 * C1M, d_obs))
p_06 = float(stats.chi2.sf(chi2_06, n_match))

checks = [
    ('M_DR1', M1, 1.0535, 5e-4), ('M_DR2', M2, 33.97, 5e-3),
    ('M_joint(1/3)', MJ13, 66.71, 5e-3),
    ('E_2|1', MJ13 / M1, 63.3, 0.05),
    ('P(sup>=20) MC', p_sup20, 1.2e-3, 5e-5),
    ('chi2(1/3)', chi2_13, 10.83, 5e-3), ('p(1/3)', p_13, 0.457, 5e-4),
    ('chi2(0.25)', chi2_25, 8.73, 5e-3), ('p(0.25)', p_25, 0.647, 5e-4),
    ('chi2(0.40)', chi2_40, 13.43, 5e-3), ('p(0.40)', p_40, 0.266, 5e-4),
    ('chi2(indep)', chi2_ind, 5.53, 5e-3), ('p(indep)', p_ind, 0.903, 5e-4),
    ('chi2(0.6C1)', chi2_06, 54.6, 0.05),
    ('median C2/C1', float(np.median(ratios)), 0.318, 5e-4),
    ('min C2/C1', float(ratios.min()), 0.221, 5e-4),
    ('max C2/C1', float(ratios.max()), 0.493, 5e-4),
]
for a, v in zip(SCAN_POINTS, SCAN_EXPECTED):
    tol = 0.005 if v < 1e3 else (0.5 if v < 1e4 else 0.05e6)
    checks.append((f'M_joint({a:.3f})', scan_points[f"{a:.4f}"], v, tol))
PULLS_EXPECTED = [+0.14, -0.15, +1.58, +1.96, -1.30, -0.58, -0.83, -0.36, +1.06, +0.77, -0.95]
for L, pv, sv in zip(labels, pulls, PULLS_EXPECTED):
    checks.append((f'pull {L}', float(pv), sv, 0.005))
fails = [(n, v, ref) for n, v, ref, tol in checks if abs(v - ref) > tol]
print("ANCHORS:")
for n, v, ref, tol in checks:
    print(f"  {n:24s} {v:14.6g}  (expected {ref})  {'OK' if abs(v - ref) <= tol else 'FAIL'}")
if fails:
    print("ANCHOR FAILURE -> stopping:", fails)
    sys.exit(2)

rec('anchors', 'M_DR1', M1, 'Default-prior (30x30 uniform grid) mixture e-value of DR1 BAO alone vs LCDM (published DR1 covariance).')
rec('anchors', 'M_DR2', M2, 'Same mixture e-value for DR2 BAO alone (published DR2 covariance).')
rec('anchors', 'p_DR2_fixed_release', 1 / M2, 'Markov bound 1/M_DR2 for DR2 analysed as a fixed release.')
rec('anchors', 'average_M', 0.5 * (M1 + M2), '(M_DR1 + M_DR2)/2: equal-weight average, an e-value under any dependence between releases.')
rec('anchors', 'p_average', 2 / (M1 + M2), 'Markov bound for the average: 1/((M1+M2)/2), printed as 0.0571 (a bound is rounded up).')
rec('anchors', 'product_M1_M2', M1 * M2, 'M_DR1 * M_DR2 from unrounded values (independence benchmark).')
rec('anchors', 'p_product', 1 / (M1 * M2), '1/(M_DR1 M_DR2).')
rec('anchors', 'M_joint_alpha_1_3', MJ13, 'Exact joint construction at Cov(X1,X2) = C1/3 on matched bins (eq. (B.3)), value at DR2.')
rec('anchors', 'p_joint_alpha_1_3', 1 / MJ13, '1/M_joint(1/3) (sup over DR1, DR2 equals M_joint since M_DR1 < M_joint).')
rec('anchors', 'E_incremental_2_given_1', MJ13 / M1, 'Incremental factor E_{2|1} = M_joint(1/3)/M_DR1.')
rec('anchors', 'MC_P_sup_ge_20_alpha_1_3', p_sup20, f'Monte Carlo P(max(M_DR1, M_joint) >= 20) under the alpha=1/3 model itself, N=100000, seed 20260711 ({n_sup20} exceedances); code/eprocess_joint.py run_construction.',
    n_exceed=n_sup20, N=100000)
rec('anchors', 'MC_P_Mjoint_ge_20_alpha_1_3', k_j / 1e5, 'Same 100000 draws (replicated exactly; sup count matches): P(M_joint(1/3) >= 20) for the DR2-time statistic alone, with Wilson 95% interval.', n_exceed=k_j, N=100000, wilson95=wilson(k_j, 100000))
rec('anchors', 'MC_P_MDR1_ge_20', k_1 / 1e5, 'Same draws: P(M_DR1 >= 20).', n_exceed=k_1, N=100000, wilson95=wilson(k_1, 100000))
rec('anchors', 'family_scan_points', scan_points, 'M_joint(alpha) at fixed reference points of the family (keys = alpha to 4 d.p.).')
rec('anchors', 'M_joint_alpha_0', float(np.exp(logM_at(0.0))), 'Joint construction at alpha = 0 (cross-release covariance zero, coherent mixture over theta). Differs from the incoherent product M1*M2.')

rec('null_test', 'n_matched', n_match, 'Matched DR1/DR2 quantities: BGS D_V; LRG1, LRG2, LRG3+ELG1, ELG2, Lya each D_M and D_H (QSO unmatched).')
rec('null_test', 'labels', labels, 'Order of the matched quantities (DR2 data-vector order).')
rec('null_test', 'z_eff', zlab, 'DR2 effective redshifts of the matched quantities.')
rec('null_test', 'chi2_alpha_1_3', chi2_13, 'd = DR2 - DR1 on matched quantities, chi2 = d^T [(1-2a)C1 + C2]^{-1} d at a = 1/3.')
rec('null_test', 'p_alpha_1_3', p_13, 'chi2 survival probability, 11 dof.')
rec('null_test', 'chi2_alpha_0_25', chi2_25, 'as above at a = 0.25'); rec('null_test', 'p_alpha_0_25', p_25, '11 dof')
rec('null_test', 'chi2_alpha_0_40', chi2_40, 'as above at a = 0.40'); rec('null_test', 'p_alpha_0_40', p_40, '11 dof')
rec('null_test', 'chi2_independence', chi2_ind, 'Var(d) = C1 + C2 (Cov = 0).'); rec('null_test', 'p_independence', p_ind, '11 dof')
rec('null_test', 'chi2_cov_0p6_C1', chi2_06, 'Var(d) = C1 + C2 - 1.2 C1 (Cov = 0.6 C1).'); rec('null_test', 'p_cov_0p6_C1', p_06, '11 dof')
rec('null_test', 'pulls_alpha_1_3', pulls, 'Per-quantity (DR2 - DR1)/sqrt(diag[C1/3 + C2]).')
rec('null_test', 'variance_ratio_C2_over_C1', ratios, 'diag(C2)/diag(C1) per matched quantity (published).')
rec('null_test', 'variance_ratio_min_median_max', [float(ratios.min()), float(np.median(ratios)), float(ratios.max())], 'min / median / max of diag(C2)/diag(C1).')

# null-test sensitivity to a coherent offset delta = s * sigma_DR2 (same sign all bins)
nc_unit = float(sig2 @ np.linalg.solve(S13, sig2))   # noncentrality per s^2
crit = stats.chi2.ppf(0.95, n_match)


def power_at(s):
    return float(stats.ncx2.sf(crit, n_match, nc_unit * s * s))


s_pow50 = optimize.brentq(lambda s: power_at(s) - 0.5, 1e-3, 5)
s_pow80 = optimize.brentq(lambda s: power_at(s) - 0.8, 1e-3, 5)
rec('null_test', 'offset_for_50pct_power', s_pow50, 'Coherent offset (every matched DR2 quantity shifted by s x its own DR2 sigma, same sign) at which the 5%-level null test (alpha=1/3 covariance) has 50% power; noncentral chi2, 11 dof.')
rec('null_test', 'offset_for_80pct_power', s_pow80, 'Same, 80% power.')
rec('null_test', 'power_at_offset_1sigma', power_at(1.0), 'Power of the 5%-level null test for a coherent offset of 1 DR2 sigma per quantity.')

# ================================================================ 2. alpha_max + scan
gev = linalg.eigh(C2M, C1M, eigvals_only=True)
alpha_max = float(np.sqrt(gev.min()))
alpha_max_code = EJ.largest_admissible_alpha(C1M, C2M)   # repo bisection, tol -1e-10
rec('alpha_max', 'alpha_max_exact', alpha_max, 'sqrt(smallest generalized eigenvalue of (C2, C1)) on the 11x11 matched blocks, as built by the construction (C1M = DR1 cov, C2M = DR2 cov in matched order): the largest alpha with C2 - alpha^2 C1 positive semi-definite. At alpha_max the matrix is singular, so the family is [0, alpha_max).')
rec('alpha_max', 'alpha_max_4dp', round(alpha_max, 4), 'alpha_max to 4 decimals.')
rec('alpha_max', 'alpha_max_code_bisection', alpha_max_code, 'code/eprocess_joint.largest_admissible_alpha (bisection with is_psd tolerance -1e-10); agrees with the exact value to ~1e-10.')
rec('alpha_max', 'generalized_eigenvalues_C2_vs_C1', np.sort(gev), 'All generalized eigenvalues lambda of C2 v = lambda C1 v (matched blocks).')

# scan grid: 0.005 coarse on [0, 0.46], 0.001 on [0.20, 0.42], and log-spaced toward alpha_max
coarse = np.round(np.arange(0.0, 0.4601, 0.005), 6)
fine = np.round(np.arange(0.20, 0.4201, 0.001), 6)
edge = alpha_max - np.logspace(np.log10(alpha_max - 0.4605), -7, 60)
a_grid = np.unique(np.concatenate([coarse, fine, edge, [1 / 3]]))
a_grid = a_grid[a_grid < alpha_max]
logM_grid = np.empty(len(a_grid))
c_grid = np.empty((len(a_grid), G))          # Ay(alpha) @ sigma_DR2
for k, a in enumerate(a_grid):
    r = construct(a)
    assert r is not None, a
    logM_grid[k] = r['logM']
    c_grid[k] = r['kern']['Ay'] @ sig2
k_min = int(np.argmin(logM_grid))
opt = optimize.minimize_scalar(logM_at, bounds=(a_grid[k_min] - 0.002, a_grid[k_min] + 0.002),
                               method='bounded', options=dict(xatol=1e-8))
a_star, logM_star = float(opt.x), float(opt.fun)
M_star = float(np.exp(logM_star))
rec('family_scan', 'worst_case_alpha', a_star, 'Continuum minimiser of M_joint(alpha) over [0, alpha_max) (bounded Brent on the construction, xatol 1e-8, started from the 0.001 grid minimum).')
rec('family_scan', 'worst_case_M', M_star, 'min over alpha in [0, alpha_max) of M_joint(alpha) (continuum).')
rec('family_scan', 'worst_case_p', 1 / M_star, '1/worst_case_M (report rounded up: never round a bound down).')
rec('family_scan', 'grid_min_alpha_0p001', float(a_grid[k_min]), 'Minimiser on the scan grid (0.001 steps near the minimum).')
rec('family_scan', 'grid_min_M_0p001', float(np.exp(logM_grid[k_min])), 'Grid minimum of M_joint.')
rec('family_scan', 'M_at_0p30', float(np.exp(logM_at(0.30))), 'M_joint(0.30).')

# smoothness near the minimum: (i) 0.001 grid within +/-0.03, (ii) 1e-4 grid within +/-0.005
aa = np.round(np.arange(round(a_star, 3) - 0.03, round(a_star, 3) + 0.0301, 0.001), 6)
ll = np.array([logM_at(a) for a in aa])
d2 = np.diff(ll, 2) / 0.001 ** 2
qfit = np.polyfit(aa - a_star, ll, 4)
resid = ll - np.polyval(qfit, aa - a_star)
curv = float(2 * np.polyfit(aa - a_star, ll, 2)[0])
aa2 = a_star + np.arange(-0.005, 0.00501, 0.0001)
ll2 = np.array([logM_at(a) for a in aa2])
d2b = np.diff(ll2, 2) / 0.0001 ** 2
pfit2 = np.polyfit(aa2 - a_star, ll2, 2)
resid2 = ll2 - np.polyval(pfit2, aa2 - a_star)
rec('family_scan', 'smoothness_check', dict(
    alpha_window=[float(aa[0]), float(aa[-1])], step=0.001,
    second_difference_logM_min=float(d2.min()), second_difference_logM_max=float(d2.max()),
    all_second_differences_positive=bool((d2 > 0).all()),
    quartic_fit_max_abs_residual_logM=float(np.abs(resid).max()),
    curvature_d2logM_dalpha2_at_min=float(2 * pfit2[0]),
    fine_window=[float(aa2[0]), float(aa2[-1])], fine_step=1e-4,
    fine_second_difference_min=float(d2b.min()), fine_second_difference_max=float(d2b.max()),
    fine_quadratic_fit_max_abs_residual_logM=float(np.abs(resid2).max())),
    'log M_joint on a 0.001 grid within +/-0.03 of the minimiser and on a 1e-4 grid within +/-0.005: all second differences positive (locally convex, no numerical jitter); a quartic (0.001 grid) and a quadratic (1e-4 grid) fit leave tiny residuals in log M. Smooth interior minimum, log-curvature ~98 per unit alpha^2.')
wide = (a_grid >= 0.20) & (a_grid <= 0.42)
aw, Mw = a_grid[wide], np.exp(logM_grid[wide])
M_width = {f'{t}': [float(aw[Mw <= t * M_star].min()), float(aw[Mw <= t * M_star].max())] for t in (1.01, 1.05, 1.10, 1.5)}
rec('family_scan', 'alpha_range_within_1_5_10_50pct_of_min', M_width, 'alpha interval (0.001 scan grid) where M_joint <= (1.01, 1.05, 1.10, 1.5) x worst case: the minimum is broad.')
rec('family_scan', 'M_joint_at_alpha_max_minus', {f'{x:g}': float(np.exp(min(logM_at(alpha_max - x), 700))) for x in (0.02, 0.01, 0.005)}, 'M_joint at alpha_max - 0.02, -0.01, -0.005 (capped at e^700).')

# divergence toward alpha_max
edge_rows = [(float(alpha_max - a), float(logM_grid[i] / np.log(10)))
             for i, a in enumerate(a_grid) if a > 0.46]
rec('family_scan', 'log10M_near_alpha_max', edge_rows[::6], '(alpha_max - alpha, log10 M_joint) pairs: M_joint diverges as alpha -> alpha_max because the DR2 innovation covariance C2 - alpha^2 C1 becomes singular.')

# ================================================================ 3. calibration budget
def log_defl(c, s):
    """log max over the two signs of E[M] under a coherent offset s*sigma_DR2 (exact MGF)."""
    return max(logsumexp(s * c), logsumexp(-s * c)) - np.log(G)


def D13(s):
    return res13['logM'] - log_defl(res13['kern']['Ay'] @ sig2, s)


c13 = res13['kern']['Ay'] @ sig2
E_plus = {p: float(np.exp(logsumexp(p / 100 * c13) - np.log(G))) for p in (1, 2, 5)}
E_minus = {p: float(np.exp(logsumexp(-p / 100 * c13) - np.log(G))) for p in (1, 2, 5)}
table13 = {f'{p}%': float(np.exp(D13(p / 100))) for p in (0, 0.5, 1, 1.5, 2, 3, 4, 5, 6)}


def Dfam_grid(s):
    vals = logM_grid - np.array([log_defl(c, s) for c in c_grid])
    return vals


def Dfam(s, return_alpha=False):
    vals = Dfam_grid(s)
    k = int(np.argmin(vals))
    lo = a_grid[max(k - 2, 0)]
    hi = a_grid[min(k + 2, len(a_grid) - 1)]

    def f(a):
        r = construct(a)
        return np.inf if r is None else r['logM'] - log_defl(r['kern']['Ay'] @ sig2, s)
    if hi > lo:
        o = optimize.minimize_scalar(f, bounds=(lo, hi), method='bounded',
                                     options=dict(xatol=1e-9))
        best = (float(o.fun), float(o.x)) if o.fun < vals[k] else (float(vals[k]), float(a_grid[k]))
    else:
        best = (float(vals[k]), float(a_grid[k]))
    return best if return_alpha else best[0]


tablefam = {}
for p in (0, 0.5, 1, 1.5, 2, 3, 4, 5, 6):
    v, a_arg = Dfam(p / 100, return_alpha=True)
    tablefam[f'{p}%'] = dict(value=float(np.exp(v)), alpha_argmin=a_arg)

b13 = optimize.brentq(lambda s: D13(s) - LOG_THR, 1e-4, 0.05, xtol=1e-10)
bfam = optimize.brentq(lambda s: Dfam(s) - LOG_THR, 1e-4, 0.05, xtol=1e-9)
_, a_at_bfam = Dfam(bfam, return_alpha=True)

# asymptotic critical offset above which the family worst case is exactly 0
lam_all, V_all = linalg.eigh(C2M, C1M)
v0 = V_all[:, 0]                                  # C1-orthonormal, lambda_min
Dm_edge = delta2[:, m2] - alpha_max * delta1[:, m1]
y_edge = eps2_obs[m2] - alpha_max * eps1_obs[m1]
proj = Dm_edge @ v0
f_top = float(np.max(proj * (y_edge @ v0) - 0.5 * proj ** 2))
s_crit = f_top / (abs(float(sig2 @ v0)) * float(np.abs(proj).max()))
# numeric confirmation very close to the boundary
a_close = alpha_max - 1e-7
r_close = construct(a_close)
c_close = r_close['kern']['Ay'] @ sig2
conf = {f'{s:.4f}': float(r_close['logM'] - log_defl(c_close, s))
        for s in (s_crit * 0.98, s_crit * 1.02)}

rec('calibration_budget', 'definition',
    'Offset of b% of sigma: every one of the 11 matched DR2 measurements is displaced, relative to DR1, by b/100 times its own published DR2 standard deviation, all in the same direction; the budget entry is the observed combined statistic divided by the larger (over the two signs) of its exact expectations under LCDM with that offset (Gaussian moment-generating function, E[M] = mean_g exp(+/- s Ay_g . sigma_DR2)), i.e. the largest constant multiple of the statistic that is still an e-value when an offset of that size and pattern (either sign) may be present.',
    'Technical definition of the relative-calibration budget.')
rec('calibration_budget', 'definition_one_sentence',
    'A coherent offset of b% of sigma means every matched DR2 measurement is shifted relative to DR1 by b% of its own quoted DR2 uncertainty in the same direction, and the tabulated value is the combined statistic divided by its largest possible mean under LCDM with such an offset, so that it remains a valid e-value.')
rec('calibration_budget', 'method',
    'Exact expectation under a coherent offset: E[M_shifted] = mean_g exp(Ay_g . delta), delta = +/- s sigma_DR2 on the matched bins, value = 66.71 / max_{+/-} E. simulations/mean_shift_robustness.py computes the direction-free Cauchy-Schwarz bound (b* = 0.0200 in the Sigma_y^-1 metric) and shift_bound.py the exact breakdown radius over all directions; the tables here are the exact computation for the coherent pattern.')
rec('calibration_budget', 'alpha_1_3_table', table13, 'alpha = 1/3 construction: budget value at coherent offsets 0-6% of sigma.')
rec('calibration_budget', 'alpha_1_3_expectation_plus', E_plus, 'E[M_joint(1/3)] under a +s sigma_DR2 offset (s = 1, 2, 5%).')
rec('calibration_budget', 'alpha_1_3_expectation_minus', E_minus, 'E[M_joint(1/3)] under a -s sigma_DR2 offset (the damaging sign at 2% and 5%).')
rec('calibration_budget', 'alpha_1_3_tolerance_b_star', b13, 'Coherent offset (fraction of DR2 sigma) at which the alpha=1/3 budget value equals 20.')
rec('calibration_budget', 'family_worst_case_table', tablefam, 'Family worst case: min over alpha in [0, alpha_max) of M_joint(alpha)/max_{+/-}E_alpha[M_joint(alpha)] at each offset, with the minimising alpha.')
rec('calibration_budget', 'family_worst_case_tolerance_b_star', bfam, 'Coherent offset at which the family worst case equals 20.')
rec('calibration_budget', 'family_worst_case_alpha_at_b_star', a_at_bfam, 'Minimising alpha at the family tolerance.')
rec('calibration_budget', 'family_offset_collapse_threshold', s_crit,
    'Above this coherent offset (fraction of DR2 sigma) the family worst case is exactly 0: as alpha -> alpha_max the DR2 innovation becomes noiseless along one direction and the offset-deflated statistic tends to 0. Asymptotic formula max_g[(Dm_g.v)(y.v) - (Dm_g.v)^2/2] / (|sigma_DR2.v| max_g|Dm_g.v|), v = generalized eigenvector of lambda_min; outside the 0-6% range of the tables above.',
    numeric_check_log_value_at_alpha_max_minus_1e7=conf)

# ---- shift_bound.py (data-allowed region, exact breakdown radius)
buf = io.StringIO()
t0 = time.time()
with contextlib.redirect_stdout(buf):
    SB = runpy.run_path(str(CROSS_RELEASE / 'shift_bound.py'), run_name='__main__')
(OUT_DIR / 'shift_bound.log').write_text(buf.getvalue())
sb_s = time.time() - t0
rec('calibration_budget', 'shift_bound_observed_norm_S', float(SB['norm_d_S']), '||d||_{S^-1}, d = DR2 - DR1 observed, S = C1/3 + C2 (sqrt of the null-test chi2).', script='cross_release/shift_bound.py')
rec('calibration_budget', 'shift_bound_r95', float(SB['r95']), 'sqrt(chi2_{11,0.95}): radius of the 95% region for the offset in the S^-1 metric.', script='cross_release/shift_bound.py')
rec('calibration_budget', 'allowed_region_max_norm_Sigma_y', float(SB['ub_Sy_exact']), 'Largest offset norm ||delta||_{Sigma_y^-1} (Sigma_y = C2 - C1/9) inside the 95% data-allowed region {(delta-d)^T S^-1 (delta-d) <= r95^2} (exact ellipsoid maximum).', script='cross_release/shift_bound.py')
rec('calibration_budget', 'exact_breakdown_radius_Sigma_y', float(SB['b_exact']), 'Smallest ||delta||_{Sigma_y^-1} (any direction) that drives M_joint(1/3) below 20 (convex minimisation + bisection).', script='cross_release/shift_bound.py')
rec('calibration_budget', 'conservative_budget_Cauchy_Schwarz', float(SB['b_star']), 'ln(M/20)/K with K = sqrt(max_g consty_g): direction-free lower bound on the breakdown radius.', script='cross_release/shift_bound.py')
rec('calibration_budget', 'allowed_over_breakdown_ratio', float(SB['ub_Sy_exact'] / SB['b_exact']), 'allowed_region_max_norm / exact_breakdown_radius; a different quantity from null_test_sensitivity_over_tolerance.', script='cross_release/shift_bound.py')
rec('calibration_budget', 'min_M_over_allowed_region', float(SB['M_min_region']), 'Minimum of M_joint(1/3) over offsets inside the 95% data-allowed region (convex minimisation, multi-start SLSQP).', script='cross_release/shift_bound.py')
rec('calibration_budget', 'M_at_best_estimate_offset', float(SB['M_at_d']), 'M_joint(1/3) recomputed after removing the best-estimate offset delta = d (the observed DR2 - DR1): the data\'s own offset direction raises the statistic.', script='cross_release/shift_bound.py')
rec('calibration_budget', 'uniform_1pct_norm_Sigma_y', float(np.sqrt((0.01 * sig2) @ np.linalg.solve(C2M - C1M / 9, 0.01 * sig2))), '||delta||_{Sigma_y^-1} of a coherent 1%-of-sigma offset, to convert between conventions.')
rec('calibration_budget', 'null_test_sensitivity_over_tolerance', s_pow50 / b13, 'offset_for_50pct_power / alpha_1_3_tolerance_b_star: how much larger an offset the null test needs to see than the budget tolerates.')

# ---- direction-free bounds from simulations/mean_shift_robustness.py (log of record)
msr_log = (LOGS / 'mean_shift_robustness.log')
if msr_log.exists():
    t = msr_log.read_text()
    mK = re.search(r'K = sqrt\(max_g consty_g\) = ([0-9.]+)', t)
    mb = re.search(r'b\* = ln\(M/20\)/K = ([0-9.]+)', t)
    ms = re.search(r'b\*_sharp = ([0-9.]+), s\*_sharp_inn = ([0-9.]+), s\*_sharp_DR2 = ([0-9.]+)', t)
    rec('calibration_budget', 'direction_free_bounds', dict(
        K=float(mK.group(1)), b_star_K=float(mb.group(1)),
        b_star_sharp=float(ms.group(1)), b_star_sharp_as_uniform_DR2_sigma_fraction=float(ms.group(3))),
        'Budgets valid for an offset of given size in ANY direction (norm ||delta||_{Sigma_y^-1}): Cauchy-Schwarz K-bound b* = ln(M/20)/K and the sharper mean-deflator b*_sharp; the latter corresponds to 1.17% of each DR2 sigma if spread uniformly. Context only; the tabulated budget is the coherent-pattern exact computation.',
        script='cross_release/simulations/mean_shift_robustness.py (log cross_release/inputs/logs/mean_shift_robustness.log)')

# ================================================================ 4. simulated FPR
kc_log = (LOGS / 'fpr_calibration_offsets.log').read_text()
cr_log = (LOGS / 'fpr_crossing.log').read_text()
kc = json.load(open(INPUTS / 'fpr_calibration_offsets_results.json'))
N_B = kc['N']
unif = {float(k.split('=')[1]): v['unif']['JOINT13'] for k, v in kc['grid'].items() if k.startswith('DR2|')}
advb = {float(k.split('=')[1]): v['adv_best']['JOINT13'] for k, v in kc['grid'].items() if k.startswith('DR2|')}
dr1_unif = {float(k.split('=')[1]): v['unif']['JOINT13'] for k, v in kc['grid'].items() if k.startswith('DR1|')}
for m in re.finditer(r'UNIF s=([0-9.]+)\s+JOINT13 FPR = (\d+)/(\d+)', cr_log):
    unif[float(m.group(1))] = int(m.group(2)) / int(m.group(3))
adv_unit = {}
for m in re.finditer(r'ADVunit s=([0-9.]+)\s+JOINT13 FPR = (\d+)/(\d+)', cr_log):
    adv_unit[float(m.group(1))] = int(m.group(2)) / int(m.group(3))
adv_unit.update({s: kc['grid'][f'DR2|s={s}']['adv_each']['FPR50']['JOINT13'] for s in (0.5, 1.0, 2.0)})
su = sorted(unif)
fu = [unif[s] for s in su]
k5 = next(i for i, f in enumerate(fu) if f >= 0.05)
s_cross_unif = su[k5 - 1] + (0.05 - fu[k5 - 1]) * (su[k5] - su[k5 - 1]) / (fu[k5] - fu[k5 - 1])
sa = sorted(adv_unit)
fa = [adv_unit[s] for s in sa]
ka = next(i for i, f in enumerate(fa) if f >= 0.05)
s_cross_adv = sa[ka - 1] + (0.05 - fa[ka - 1]) * (sa[ka] - sa[ka - 1]) / (fa[ka] - fa[ka - 1])
v_adv = np.array(kc['adv_dirs']['DR2']['FPR50'])
base_fpr = kc['baseline']['JOINT13'] / N_B
Sy13 = C2M - C1M / 9.0
v_adv_n = v_adv / np.linalg.norm(v_adv)
delta_adv = s_cross_adv * sig2 * v_adv_n
norm_adv_Sy = float(np.sqrt(delta_adv @ np.linalg.solve(Sy13, delta_adv)))
norm_unif_Sy = float(s_cross_unif * np.sqrt(sig2 @ np.linalg.solve(Sy13, sig2)))
norm_b13_Sy = float(b13 * np.sqrt(sig2 @ np.linalg.solve(Sy13, sig2)))
rec('simulated_fpr', 'source', 'cross_release/simulations/fpr_calibration_offsets.py (N=4000 paired LCDM draws per cell, seed 20260831, alpha_true = 1/3; log cross_release/inputs/logs/fpr_calibration_offsets.log, grid cross_release/inputs/fpr_calibration_offsets_results.json) and cross_release/simulations/fpr_crossing.py (log cross_release/inputs/logs/fpr_crossing.log).', script='cross_release/simulations/fpr_calibration_offsets.py + cross_release/simulations/fpr_crossing.py')
rec('simulated_fpr', 'baseline_fpr_joint13_N100k', k_j / 1e5, 'Better-determined baseline: P(M_joint(1/3) >= 20) under LCDM from 100000 draws of the alpha=1/3 model (see anchors.MC_P_Mjoint_ge_20_alpha_1_3); the 4000-draw value baseline_fpr_joint13 has only 3 events.', wilson95=wilson(k_j, 100000))
rec('simulated_fpr', 'baseline_fpr_joint13', base_fpr, 'Fraction of LCDM draws (no offset) with M_joint(1/3) >= 20 in the 4000-draw design of fpr_calibration_offsets.py.', script='cross_release/simulations/fpr_calibration_offsets.py', counts=f"{kc['baseline']['JOINT13']}/{N_B}", wilson95=wilson(kc['baseline']['JOINT13'], N_B))
rec('simulated_fpr', 'unif_dr2_fpr_vs_offset', {f'{s}': unif[s] for s in su}, 'False-positive rate at threshold 20 of M_joint(1/3) when every matched DR2 measurement is shifted by s x its DR2 sigma (same sign).', script='cross_release/simulations/fpr_calibration_offsets.py + cross_release/simulations/fpr_crossing.py')
rec('simulated_fpr', 'unif_dr2_5pct_crossing', s_cross_unif, 'Linear interpolation of the coherent per-bin offset at which the simulated rate reaches 5% (between s=0.50: 0.0365 and s=0.55: 0.0532).', script='cross_release/simulations/fpr_crossing.py')
rec('simulated_fpr', 'adv_unit_fpr_vs_total_norm', {f'{s}': adv_unit[s] for s in sa}, 'Same, offset along the most damaging direction found (FPR-targeted ascent, unit vector in sigma units); s is the TOTAL offset norm over the 11 quantities in units of sigma.', script='cross_release/simulations/fpr_crossing.py')
rec('simulated_fpr', 'adv_unit_5pct_crossing_total_norm', s_cross_adv, 'Interpolated total offset norm (sigma units, quadrature sum over 11 quantities) at which the rate reaches 5% along the adversarial direction.', script='cross_release/simulations/fpr_crossing.py')
rec('simulated_fpr', 'adv_unit_5pct_crossing_per_bin_rms', s_cross_adv / np.sqrt(n_match), 'The adversarial crossing expressed per quantity (rms over the 11 quantities).', script='cross_release/simulations/fpr_crossing.py')
rec('simulated_fpr', 'adv_unit_5pct_crossing_per_bin_max', s_cross_adv * float(np.abs(v_adv).max()), 'The adversarial crossing, largest single-quantity offset.', script='cross_release/simulations/fpr_crossing.py')
rec('simulated_fpr', 'ratio_unif_crossing_to_tolerance', s_cross_unif / b13, 'Coherent per-bin crossing / alpha=1/3 budget tolerance b*.')
rec('simulated_fpr', 'like_for_like_ratios', dict(
    coherent_pattern_crossing_over_coherent_tolerance=s_cross_unif / b13,
    coherent_crossing_norm_Sigma_y=norm_unif_Sy, coherent_tolerance_norm_Sigma_y=norm_b13_Sy,
    adversarial_crossing_norm_Sigma_y=norm_adv_Sy,
    adversarial_crossing_over_exact_breakdown_radius=norm_adv_Sy / float(SB['b_exact']),
    adversarial_crossing_over_coherent_tolerance_norm=norm_adv_Sy / norm_b13_Sy,
    coherent_crossing_over_exact_breakdown_radius=norm_unif_Sy / float(SB['b_exact'])),
    'Like-for-like comparisons in the budget metric ||delta||_{Sigma_y^-1}, Sigma_y = C2 - C1/9: coherent pattern (0.54 sigma per bin vs tolerance b* per bin), adversarial direction vs the all-direction exact breakdown radius (0.361), and coherent crossing vs the breakdown radius.')
rec('simulated_fpr', 'dr1_side_unif_fpr_vs_offset', {f'{s}': dr1_unif[s] for s in sorted(dr1_unif)}, 'Rate for M_joint(1/3) when the offset is placed on the matched DR1 measurements instead (DR2 untouched): flat at baseline.', script='cross_release/simulations/fpr_calibration_offsets.py')

# ---- cross-release worlds, power under the DR2 MLE, and the full-family power check
EA = json.load(open(INPUTS / 'fpr_cross_release_worlds_results.json'))
EC = json.load(open(INPUTS / 'power_dr2_mle_results.json'))
PF = json.load(open(INPUTS / 'power_full_family_results.json'))
rec('simulations_context', 'worlds_average_closed_form_mean_log10', {w: EA['closed_form_log10_means'][w]['AVG'] for w in ('W1', 'W2', 'W3', 'W4')},
    'log10 of the exact (closed-form Gaussian) mean of (M_DR1+M_DR2)/2 under LCDM in four worlds: W1 alpha=1/3 law, W2 alpha=0.25, W3 independent releases, W4 out-of-family (diagonal cross-covariance 0.35 sqrt(C1_kk C2_kk)). All 0 to 1e-11: mean exactly 1.',
    script='cross_release/simulations/fpr_cross_release_worlds.py')
rec('simulations_context', 'worlds_fpr_at_20', {w: {st: EA['mc'][w][st]['fpr'] for st in ('M1', 'M2', 'AVG', 'PROD', 'JOINT13', 'FAMINF')} for w in ('W1', 'W2', 'W3', 'W4')},
    'Simulated false-positive rate at threshold 20, N=100000 LCDM draws per world (FAMINF here = min over alpha in {0.05,0.15,0.25,0.30,1/3,0.40,0.45}).',
    script='cross_release/simulations/fpr_cross_release_worlds.py')
rec('simulations_context', 'worlds_JOINT13_closed_form_mean_log10', {w: EA['closed_form_log10_means'][w]['JOINT13'] for w in ('W1', 'W2', 'W3', 'W4')},
    'log10 exact mean of M_joint(1/3) under each world: 0 in its own world, 2.2 (alpha=0.25), 13.4 (independent), 19.7 (out-of-family).', script='cross_release/simulations/fpr_cross_release_worlds.py')
rec('simulations_context', 'power_7alpha_family', {k: EC['alternative'][k]['power'] for k in EC['alternative']},
    'Power at threshold 20, DR2-MLE alternative in both releases, alpha_true=1/3 noise, N=20000 (seed 20260831). FAMINF here is the minimum over the 7-alpha family {0.25,...,0.40} only; JOINT13 and FAMINF are max with M_DR1.',
    script='cross_release/simulations/power_dr2_mle.py')
rec('simulations_context', 'power_full_family', {k: v['rate'] for k, v in PF['alternative'].items() if isinstance(v, dict)},
    'Same draws as power_7alpha_family, adding the family worst case over the FULL admissible range [0, alpha_max) (dense grid + boundary asymptotics; draws whose statistic collapses to 0 at alpha_max counted as non-rejections). FAMINF_full is the power of the family worst case over [0, alpha_max) (Appendix B.2).',
    script='cross_release/simulations/power_full_family.py', wilson95={k: v['wilson95'] for k, v in PF['alternative'].items() if isinstance(v, dict)},
    median={k: v['median'] for k, v in PF['alternative'].items() if isinstance(v, dict)})
rec('simulations_context', 'fpr_full_family_H0', {k: v['rate'] for k, v in PF['H0_alpha_1_3'].items() if isinstance(v, dict)},
    'LCDM (alpha=1/3 law) false-positive rates at threshold 20 for the same statistics, N=20000 (seed 20260832).', script='cross_release/simulations/power_full_family.py')
rec('simulations_context', 'full_family_collapse_fraction', dict(alternative=PF['alternative']['fraction_collapsing_to_0_at_alpha_max'], H0=PF['H0_alpha_1_3']['fraction_collapsing_to_0_at_alpha_max']),
    'Fraction of simulated draws for which the full-family worst case is exactly 0 (statistic -> 0 as alpha -> alpha_max). On the real data it diverges instead (f_top > 0), so the observed worst case is interior.', script='cross_release/simulations/power_full_family.py')

# ================================================================ series for the figures
budget_s = np.round(np.arange(0.0, 0.0601, 0.0005), 6)
b13_curve = np.array([np.exp(D13(s)) for s in budget_s])
bfam_curve = np.array([np.exp(Dfam(s)) for s in budget_s])
series = dict(
    alpha=a_grid, M_joint=np.exp(np.minimum(logM_grid, 700)), log10_M_joint=logM_grid / np.log(10),
    budget_offset_fraction=budget_s, budget_alpha_1_3=b13_curve, budget_family_worst=bfam_curve,
    pulls=pulls, pull_labels=labels, pull_z=zlab)
NUM['series'] = dict(value=series, definition='Arrays: the family scan (alpha, M_joint, log10_M_joint; Figure 4a), the budget values as functions of the coherent offset (budget_offset_fraction in units of sigma_DR2, budget_alpha_1_3, budget_family_worst; the curves through the entries of Table 3) and the null-test pulls with their labels and redshifts.', script=THIS)

NUM['figures'] = dict(
    figure1_bao_data=dict(
        pdf='figure1_bao_data.pdf', script='figures/figure1_bao_data.py', paper='Figure 1, Section 3',
        description='(a) DR2 BAO residuals from the Planck 2018 LCDM prediction with the w0waCDM prediction at the DR2 BAO-only MLE; (b) the (w0, wa) plane with the Default box, the DR2 Fisher ellipses, the thawing band and the freezing box.',
        reads='data/dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt; the DR2 mean vector and the fixed background are written in the script'),
    figure2_combination_bounds=dict(
        pdf='figure2_combination_bounds.pdf', script='figures/figure2_combination_bounds.py', paper='Figure 2, Section 4.1',
        description='One row per statement about the combined DR1+DR2 evidence, in the order of Section 4.1, each placed at its e-value; filled markers for the statements that need no model of the relation between the releases, hollow markers for the model-based ones.',
        reads='anchors.M_DR2; anchors.average_M; weights_rule.information_proportional; family_scan.worst_case_M; structured_families.M_joint_desi_rule; structured_families.per_bin_worst_case'),
    figure3_loo_bins=dict(
        pdf='figure3_loo_bins.pdf', script='figures/figure3_loo_bins.py', paper='Figure 3, Section 4.2',
        description='(a) The leave-one-out e-value of each DR2 redshift bin; (b) the aggregate e-values with and without LRG2.',
        reads='results/loo.json, results/lrg2_drop.json, results/per_bin_regrow.json (or their copies of record in cross_release/inputs/results/)'),
    figure4_cross_release_checks=dict(
        pdf='figure4_cross_release_checks.pdf', script='figures/figure4_cross_release_checks.py', paper='Figure 4, Appendix B.2',
        description='(a) The model-based combined statistic across the one-fraction family with its minimum, alpha_1 = 1/3 and the per-bin worst case marked; (b) the DR2 - DR1 pulls on the 11 matched quantities at alpha_1 = 1/3.',
        reads='series.alpha; series.M_joint; series.pull_labels; alpha_max.alpha_max_exact; family_scan.worst_case_alpha and worst_case_M; anchors.M_joint_alpha_1_3; structured_families.per_bin_worst_case; structured_families.null_test_zeff_corrected'),
    notation_note='The figures, like the paper, write alpha_1 for the cross-release noise fraction, keeping alpha = 0.05 for the test level.')

NUM['meta'] = dict(
    generated=time.strftime('%Y-%m-%d %H:%M:%S'),
    runtime_seconds=round(time.time() - t_start, 1),
    shift_bound_runtime_seconds=round(sb_s, 1),
    grid='30x30 uniform (w0 in [-1.5,-0.5], wa in [-2,1]), Default prior',
    threshold=THR,
    python=sys.version.split()[0])
ALL = load_numbers()                      # keep the sections written by the other scripts
ALL.update(to_py(NUM))
save_numbers(ALL)

# ---------------------------------------------------------------- summary
print(f"\nalpha_max exact = {alpha_max:.10f} ({alpha_max:.4f}); code bisection {alpha_max_code:.10f}")
print(f"worst case: alpha* = {a_star:.5f}, M* = {M_star:.4f}, p = {1 / M_star:.5f}")
print(f"grid min: alpha = {a_grid[k_min]:.3f}, M = {np.exp(logM_grid[k_min]):.4f}")
print(f"smoothness: d2 min/max {d2.min():.2f}/{d2.max():.2f}, quartic resid {np.abs(resid).max():.2e}")
print(f"M(0) = {np.exp(logM_at(0.0)):.2f}")
print("budget a=1/3:", {k: round(v, 5) for k, v in table13.items()}, f" b* = {b13:.5f}")
print("budget fam  :", {k: (round(v['value'], 5), round(v['alpha_argmin'], 4)) for k, v in tablefam.items()}, f" b* = {bfam:.5f} at alpha {a_at_bfam:.4f}")
print(f"family collapse threshold s_crit = {s_crit:.5f}; check {conf}")
print(f"null test: 50% power at s = {s_pow50:.3f}, 80% at {s_pow80:.3f}; ratio to b* = {s_pow50 / b13:.1f}")
print(f"shift_bound: allowed {SB['ub_Sy_exact']:.3f}, breakdown {SB['b_exact']:.4f}, ratio {SB['ub_Sy_exact'] / SB['b_exact']:.1f}, "
      f"min M {SB['M_min_region']:.4g}, M(d) {SB['M_at_d']:.4g}")
print(f"sim FPR: baseline {base_fpr:.5f}; unif crossing {s_cross_unif:.4f}; adv total {s_cross_adv:.3f} "
      f"(rms/bin {s_cross_adv / np.sqrt(n_match):.3f}, max/bin {s_cross_adv * np.abs(v_adv).max():.3f})")
print(f"wrote {OUT}  [{time.time() - t_start:.0f}s]")
