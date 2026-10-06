#!/usr/bin/env python3
"""Structured cross-release models and related checks -> results/numbers.json["structured_families"].

Uses the analysis kernels of code/ (eprocess_joint.py, evalue_analysis.py,
eprocess_hierarchical_mc.py).

Sections
  1. validation: the scalar-family construction reproduces 66.71 (alpha=1/3) and 64.73 (0.3097)
  2. the correlations of DESI's DR1-DR2 consistency check (DR2 Results II, Sec. III.3.1 and
     footnote 12): rho = sqrt(N_DR1/N_DR2) (galaxies, quasars), 0.61 (Lya); implied
     alpha_i = rho_i s2/s1; combined statistic under three completions of these correlations
     to a full cross-covariance, and under the nested rule alpha_i = C2/C1
  3. one fraction per bin: per-bin PSD limits and worst cases over several ranges
  4. QSO: scalar construction without the DR2 QSO (D_M, D_H) term
  5. calibration tolerance of other statistics (DR2 alone; Narrow sub-box of the grid)
  6. null test with d corrected for the effective-redshift shifts (residuals about each
     release's own LCDM prediction), with the offset-region quantities of shift_bound.py
  7. combination weights (the conventions of the Section 4.1 footnote, DR1-free bound)
  8. exact-tail Bonferroni (results/sigma_mc_results.json), background worst case
     (results/background_sensitivity.json), DR2 with the Table IV errors

Run (from the repository root, about 15 s): python cross_release/structured_families.py
Reads the numbers written by one_fraction_family.py (results/numbers.json, or the copy of
record) and the two results files above (or their copies of record in cross_release/inputs/results/).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import linalg, optimize, stats
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import ROOT, NUMBERS, add_code_path, load_numbers, save_numbers, result_json  # noqa: E402
add_code_path()
REPO = ROOT
OUT = NUMBERS
THIS = 'cross_release/structured_families.py'

from data_loader import load_desi_data                                       # noqa: E402
from evalue_analysis import precompute_kernels, mixture_log_e_from_residuals  # noqa: E402
from eprocess_hierarchical_mc import build_bin_matching, nearest_psd          # noqa: E402
import eprocess_joint as EJ                                                   # noqa: E402

t0 = time.time()
THR = 20.0
LTHR = np.log(THR)
R = {}


def rec(key, value, definition):
    R[key] = dict(value=value, definition=definition, script=THIS)


def py(x):
    if isinstance(x, dict):
        return {k: py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [py(v) for v in x]
    if isinstance(x, np.ndarray):
        return py(x.tolist())
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


# ------------------------------------------------------------------ data
ds1 = load_desi_data(REPO / 'data' / 'dr1', 'DR1')
ds2 = load_desi_data(REPO / 'data' / 'dr2', 'DR2')
K1 = precompute_kernels(ds1.z_eff, ds1.quantities, ds1.cov, EJ.W0_GRID, EJ.WA_GRID)
K2 = precompute_kernels(ds2.z_eff, ds2.quantities, ds2.cov, EJ.W0_GRID, EJ.WA_GRID)
_, D1 = EJ.build_delta_matrix(ds1.z_eff, ds1.quantities, EJ.W0_GRID, EJ.WA_GRID)
_, D2 = EJ.build_delta_matrix(ds2.z_eff, ds2.quantities, EJ.W0_GRID, EJ.WA_GRID)
dr2_to_dr1, _ = build_bin_matching(ds1, ds2)
pairs = [(i, j) for j, i in enumerate(dr2_to_dr1) if i >= 0]
m1 = np.array([i for i, _ in pairs]); m2 = np.array([j for _, j in pairs])
u2 = np.array([j for j in range(len(ds2.data)) if dr2_to_dr1[j] < 0])
n = len(pairs)
C1 = ds1.cov
C1M = C1[np.ix_(m1, m1)]
C2M = ds2.cov[np.ix_(m2, m2)]
covu = nearest_psd(ds2.cov[np.ix_(u2, u2)])[0] + 1e-14 * np.eye(len(u2))
e1 = ds1.data - K1['mu_null']
e2 = ds2.data - K2['mu_null']
s1 = np.sqrt(np.diag(C1M)); s2 = np.sqrt(np.diag(C2M))
TRACER = {0.295: 'BGS', 0.51: 'LRG1', 0.706: 'LRG2', 0.934: 'LRG3+ELG1', 1.321: 'ELG2', 2.33: 'Lya'}
QN = {'DV_over_rs': 'DV', 'DM_over_rs': 'DM', 'DH_over_rs': 'DH'}
zz = [round(float(ds2.z_eff[j]), 3) for j in m2]
labels = [f"{TRACER[z]} {QN[ds2.quantities[j]]}" for z, j in zip(zz, m2)]
bins = sorted(set(zz))
idx = [np.array([k for k, z in enumerate(zz) if z == b]) for b in bins]
bin_names = [TRACER[b] for b in bins]
W0, WA = np.meshgrid(EJ.W0_GRID, EJ.WA_GRID, indexing='ij')
W0 = W0.ravel(); WA = WA.ravel()
G = len(W0)

C1inv = np.linalg.inv(C1)
A1 = D1 @ C1inv
c1 = np.einsum('gi,ij,gj->g', D1, C1inv, D1)
Qu = np.linalg.inv(covu)
Du = D2[:, u2]
Au = Du @ Qu
cu = np.einsum('gi,ij,gj->g', Du, Qu, Du)


def lme(L, mask=None):
    if mask is not None:
        L = L[mask]
    return float(logsumexp(L) - np.log(len(L)))


def terms_regression(avec, e1v=e1, e2v=e2, include_u=True):
    """Per-grid log-LR for the model X2_m = mu2 + A (X1_m - mu1) + noise, A = diag(avec)
    (cross-covariance C1M A, i.e. alpha_i C1 column-wise). Scalar avec reproduces the paper."""
    A = np.diag(avec)
    Sy = C2M - A @ C1M @ A.T
    if np.linalg.eigvalsh(Sy).min() <= 0:
        return None
    Qy = np.linalg.inv(Sy)
    Dm = D2[:, m2] - D1[:, m1] @ A.T
    y = e2v[m2] - A @ e1v[m1]
    Ay = Dm @ Qy
    L = A1 @ e1v + Ay @ y - 0.5 * (c1 + np.einsum('gi,ij,gj->g', Dm, Qy, Dm))
    if include_u:
        L = L + Au @ e2v[u2] - 0.5 * cu
    return dict(L=L, Ay=Ay, Sy=Sy)


def logM(avec, **kw):
    t = terms_regression(avec, **kw)
    return np.inf if t is None else lme(t['L'])


def terms_general(X, e1v=e1, e2v=e2):
    """Per-grid log-LR for a general cross-covariance X = Cov(X1_m, X2_m) (11x11)."""
    S12 = np.zeros((len(ds1.data), n)); S12[m1, :] = X
    B = S12.T @ C1inv
    Sc = C2M - S12.T @ C1inv @ S12
    if np.linalg.eigvalsh(Sc).min() <= 0:
        return None
    Qc = np.linalg.inv(Sc)
    r = e2v[m2] - B @ e1v
    Dc = D2[:, m2] - D1 @ B.T
    L = A1 @ e1v - 0.5 * c1 + Dc @ Qc @ r - 0.5 * np.einsum('gi,ij,gj->g', Dc, Qc, Dc) \
        + Au @ e2v[u2] - 0.5 * cu
    return dict(L=L)


# ------------------------------------------------------------------ 1. validation
v13 = np.exp(logM(np.full(n, 1 / 3)))
v31 = np.exp(logM(np.full(n, 0.30965149996910124)))
vg13 = np.exp(lme(terms_general(C1M / 3)['L']))
assert abs(v13 - 66.7089) < 2e-3 and abs(v31 - 64.7349) < 2e-3 and abs(vg13 - v13) < 1e-6, (v13, v31, vg13)
rec('validation', dict(M_joint_1_3=v13, M_joint_0p3097=v31, general_form_at_1_3=vg13),
    'Both constructions used here reproduce numbers.json anchors (66.709 at alpha=1/3, 64.735 at the worst case 0.3097).')

# ------------------------------------------------------------------ 2. DESI's consistency-check correlations
NTR = {0.295: (300017, 1188526), 0.51: (506905, 1052151), 0.706: (771875, 1613562),
       0.934: (1876164, 4540343), 1.321: (1415687, 3797271)}
rho = np.array([np.sqrt(NTR[z][0] / NTR[z][1]) if z in NTR else 0.61 for z in zz])
aD = rho * s2 / s1
aN = s2 ** 2 / s1 ** 2
rec('desi_rule_rho', dict(zip(labels, rho)), 'DESI DR2 Results II Sec. III.3.1 and footnote 12: rho = sqrt(N_DR1/N_DR2) (tracer counts) for galaxies/quasars, 0.61 for Lya.')
rec('desi_rule_alpha_i', dict(zip(labels, aD)), 'Implied per-quantity noise fraction alpha_i = rho_i sigma_DR2,i/sigma_DR1,i (Cov = rho s1 s2 = alpha_i s1^2), reading the printed -2C sigma_DR1 as -2C sigma_DR1 sigma_DR2.')
rec('desi_rule_alpha_min_median_max', [aD.min(), float(np.median(aD)), aD.max()], 'min/median/max of alpha_i under the DESI consistency-check correlations (paper: 0.25-0.49, median 0.37).')
M_desi_reg = np.exp(logM(aD))
M_desi_sym = np.exp(lme(terms_general(np.diag(np.sqrt(aD)) @ C1M @ np.diag(np.sqrt(aD)))['L']))
Xs = np.zeros((n, n))
for ii in idx:
    a1 = np.real(linalg.sqrtm(C1M[np.ix_(ii, ii)])); a2 = np.real(linalg.sqrtm(C2M[np.ix_(ii, ii)]))
    Mx = rho[ii[0]] * a1 @ a2
    Xs[np.ix_(ii, ii)] = 0.5 * (Mx + Mx.T)
M_desi_sqrt = np.exp(lme(terms_general(Xs)['L']))
M_nested = np.exp(logM(aN))
rec('M_joint_desi_rule', dict(regression_form=M_desi_reg, symmetric_alpha_i_C1=M_desi_sym,
                              per_bin_matrix_sqrt=M_desi_sqrt,
                              min=min(M_desi_reg, M_desi_sym, M_desi_sqrt),
                              max=max(M_desi_reg, M_desi_sym, M_desi_sqrt),
                              p_bound_at_min=1 / min(M_desi_reg, M_desi_sym, M_desi_sqrt)),
    "Combined statistic (eq. B.3 generalised) under the correlations of DESI's DR1-DR2 consistency check. These fix only each quantity's own DR1-DR2 correlation; three completions to a full cross-covariance: (i) regression form X2 = A X1 + noise, A = diag(alpha_i) (Cov = C1 A); (ii) symmetric D^1/2 C1 D^1/2; (iii) per bin rho C1^1/2 C2^1/2 (symmetrised). The paper quotes the range 28.9 to 33.7.")
rec('M_joint_nested_rule', M_nested, 'Combined statistic with alpha_i = C2_ii/C1_ii (strictly nested data), regression form.')

# ------------------------------------------------------------------ 3. per-bin family
amax_b = []
for ii in idx:
    w = linalg.eigh(C2M[np.ix_(ii, ii)], C1M[np.ix_(ii, ii)], eigvals_only=True)
    amax_b.append(np.sqrt(w.min()))
amax_b = np.array(amax_b)
capN = np.array([aN[ii].min() for ii in idx])
aDb = np.array([aD[ii].mean() for ii in idx])


def vec(ab):
    a = np.zeros(n)
    for b, ii in zip(ab, idx):
        a[ii] = b
    return a


rng = np.random.default_rng(20260925)


def pmin(bnds, nstart=40):
    best = None
    starts = [np.array([(l + h) / 2 for l, h in bnds])] + \
        [np.array([rng.uniform(l, h) for l, h in bnds]) for _ in range(nstart)]
    for x0 in starts:
        r = optimize.minimize(lambda x: logM(vec(x)), x0, method='L-BFGS-B', bounds=bnds)
        if best is None or r.fun < best.fun:
            best = r
    return float(np.exp(best.fun)), dict(zip(bin_names, np.round(best.x, 4)))


pb = {}
pb['nested_cap'] = pmin([(0.0, c) for c in capN])
pb['desi_span_0p25_0p49'] = pmin([(0.25, min(0.49, 0.98 * m)) for m in amax_b])
pb['desi_pm_0p1'] = pmin([(max(0, a - 0.1), min(a + 0.1, 0.99 * m)) for a, m in zip(aDb, amax_b)])
pb['full_psd_0p95'] = pmin([(0.0, 0.95 * m) for m in amax_b])
rec('per_bin_alpha_max', dict(zip(bin_names, amax_b)), 'Per-bin PSD limit sqrt(min generalised eigenvalue of the bin blocks of C2 vs C1).')
rec('per_bin_nested_cap', dict(zip(bin_names, capN)), 'Per-bin cap: min over the bin quantities of C2/C1 (cross-correlation no stronger than strictly nested data).')
rec('per_bin_desi_mean_alpha', dict(zip(bin_names, aDb)), 'Per-bin mean of the alpha_i implied by the DESI consistency-check correlations.')
rec('per_bin_worst_case', pb, 'Minimum of the combined statistic over one noise fraction per bin (Cov block = alpha_b C1 block): (a) 0 <= alpha_b <= nested cap; (b) alpha_b in [0.25, 0.49] (range of the DESI consistency-check alpha_i), capped at 0.98 alpha_max_b; (c) within +/-0.1 of the DESI consistency-check per-bin mean; (d) full per-bin PSD range capped at 0.95 alpha_max_b. Values: (M_min, argmin).')

# ------------------------------------------------------------------ 4. QSO
rec('M_joint_1_3_without_DR2_QSO', np.exp(logM(np.full(n, 1 / 3), include_u=False)),
    'Scalar alpha=1/3 construction with the unmatched DR2 QSO (D_M, D_H) term dropped (DR1 QSO D_V still scored): removes the zero-cross-covariance assumption for QSO.')
rec('M_joint_desi_regression_without_DR2_QSO', np.exp(logM(aD, include_u=False)), 'Same, DESI consistency-check correlations (regression form).')

# ------------------------------------------------------------------ 5. tolerance for other statistics
def tol_from(Lvec, Avec, sig, mask=None):
    """b* where log M - log max_{+/-} mean_g exp(+/- b A_g.sig) = log 20 (coherent offset b sigma)."""
    if mask is None:
        mask = np.ones(len(Lvec), bool)
    lm = lme(Lvec, mask)
    proj = (Avec @ sig)[mask]

    def defl(b):
        return lm - max(logsumexp(b * proj), logsumexp(-b * proj)) + np.log(mask.sum())
    if defl(0) < LTHR:
        return dict(M=np.exp(lm), b_star=None, max_abs_proj=float(np.abs(proj).max()))
    b = optimize.brentq(lambda x: defl(x) - LTHR, 0, 1, xtol=1e-10)
    k = int(np.argmax(np.abs(proj)))
    return dict(M=float(np.exp(lm)), b_star=float(b), max_abs_proj=float(np.abs(proj).max()),
                argmax_w0_wa=[float(W0[mask][k]), float(WA[mask][k])],
                deflated_at={f'{p}%': float(np.exp(defl(p / 100))) for p in (0.5, 1, 1.5, 2, 3, 5)})


narrow = (W0 >= -1.2 - 1e-9) & (W0 <= -0.8 + 1e-9) & (WA >= -1.0 - 1e-9) & (WA <= 0.5 + 1e-9)
t13 = terms_regression(np.full(n, 1 / 3))
tol = {}
tol['M_joint_1_3_default'] = tol_from(t13['L'], t13['Ay'], s2)
tol['M_joint_1_3_narrow_subbox'] = tol_from(t13['L'], t13['Ay'], s2, narrow)
LDR2 = K2['A'] @ e2 - 0.5 * K2['const']
sig2_all = np.sqrt(np.diag(ds2.cov))
sig_m_only = np.zeros(len(ds2.data)); sig_m_only[m2] = s2
tol['M_DR2_default_11_matched'] = tol_from(LDR2, K2['A'], sig_m_only)
tol['M_DR2_default_all_13'] = tol_from(LDR2, K2['A'], sig2_all)
tol['M_DR2_narrow_subbox_11_matched'] = tol_from(LDR2, K2['A'], sig_m_only, narrow)
rec('tolerance_by_statistic', tol, 'Coherent-offset tolerance b* (offset b sigma_DR2 on every quantity listed, same sign; statistic divided by its largest mean under LCDM with that offset, either sign) for the alpha=1/3 combined statistic and for DR2 alone (absolute offset of DR2), on the Default 30x30 grid and on its Narrow-box sub-grid ([-1.2,-0.8]x[-1.0,0.5]). max_abs_proj = max_g |A_g . sigma| (the deflator is dominated by the far prior corner).')
rec('narrow_subbox_npoints', int(narrow.sum()), 'Grid points of the Default grid inside the Narrow box.')
offs = np.array(load_numbers()['series']['value']['budget_offset_fraction'])
projDR2 = K2['A'] @ sig_m_only
lmDR2 = lme(LDR2)
rec('series_budget_DR2_alone', dict(
    offset_fraction=offs,
    value=[float(np.exp(lmDR2 - max(logsumexp(b * projDR2), logsumexp(-b * projDR2)) + np.log(G))) for b in offs]),
    'DR2-alone statistic divided by its largest mean under a coherent offset b sigma_DR2 of its 11 matched measurements (same offset grid as series.budget_offset_fraction).')

# ------------------------------------------------------------------ 6. z_eff-corrected null test
d_raw = ds2.data[m2] - ds1.data[m1]
d_c = e2[m2] - e1[m1]                       # residuals about each release's own LCDM prediction
zshift = d_raw - d_c                        # = mu_null(z2) - mu_null(z1)


def chi2(d, a=None, V=None):
    V = (1 - 2 * a) * C1M + C2M if V is None else V
    c = float(d @ np.linalg.solve(V, d))
    return c, float(stats.chi2.sf(c, n))


nt = {}
for key, a in (('1_3', 1 / 3), ('0_25', 0.25), ('0_40', 0.40), ('independence', 0.0)):
    nt[f'chi2_{key}'] = chi2(d_c, a)
nt['chi2_cov_0p6_C1'] = chi2(d_c, V=C1M + C2M - 1.2 * C1M)
nt['chi2_1_3_raw'] = chi2(d_raw, 1 / 3)
S13 = C1M / 3 + C2M
pulls_c = d_c / np.sqrt(np.diag(S13))
nt['pulls_1_3'] = dict(zip(labels, pulls_c))
nt['largest_pull'] = [labels[int(np.argmax(np.abs(pulls_c)))], float(pulls_c[np.argmax(np.abs(pulls_c))])]
nt['zshift_in_sigma_DR2'] = dict(zip(labels, zshift / s2))
nt['zshift_max_abs_sigma_DR2'] = float(np.abs(zshift / s2).max())
bins_chi = {}
for b, ii in zip(bin_names, idx):
    c = float(d_c[ii] @ np.linalg.solve(S13[np.ix_(ii, ii)], d_c[ii]))
    bins_chi[b] = [c, len(ii), float(stats.chi2.sf(c, len(ii)))]
nt['bin_chi2_1_3'] = bins_chi
Vd = np.diag(s1 ** 2 + s2 ** 2 - 2 * rho * s1 * s2)
nt['pulls_desi_rule_diag'] = dict(zip(labels, d_c / np.sqrt(np.diag(Vd))))
rec('null_test_zeff_corrected', nt, 'Null test with d = (X2 - mu_null(z_eff,DR2)) - (X1 - mu_null(z_eff,DR1)), i.e. corrected for the effective-redshift shifts (LRG3+ELG1 0.930->0.934, ELG2 1.317->1.321) using the paper\'s own fixed-background LCDM predictions. Values are (chi2, p) with 11 dof unless noted; bin_chi2 = (chi2, dof, p).')

# free-pattern offset region, recomputed with the corrected d (shift_bound.py logic)
a = 1 / 3
Sig_y = C2M - a ** 2 * C1M
Sy_inv = np.linalg.inv(Sig_y)
Lc = np.linalg.cholesky(S13)
r95 = float(np.sqrt(stats.chi2.ppf(0.95, n)))
Amat = Lc.T @ Sy_inv @ Lc
lam, V = np.linalg.eigh(Amat)
bb = V.T @ (Lc.T @ (Sy_inv @ d_c))


def wnorm(mu):
    return np.sqrt(np.sum((bb / (mu - lam)) ** 2))


lo, hi = lam.max() * (1 + 1e-12) + 1e-12, lam.max() + 1e6
while wnorm(lo) < r95:
    lo = lam.max() + (lo - lam.max()) / 10
while wnorm(hi) > r95:
    hi *= 10
for _ in range(200):
    mid = 0.5 * (lo + hi)
    if wnorm(mid) > r95:
        lo = mid
    else:
        hi = mid
wv = bb / (mid - lam)
max_norm = float(np.sqrt(d_c @ Sy_inv @ d_c + 2 * bb @ wv + wv @ (lam * wv)))
NUM0 = load_numbers()
breakdown = NUM0['calibration_budget']['exact_breakdown_radius_Sigma_y']['value']
base = t13['L']
Ay13 = t13['Ay']
Aw = Ay13 @ Lc
shift_d = Ay13 @ d_c


def f2(w):
    return logsumexp(base - shift_d - Aw @ w) - np.log(G)


def g2(w):
    z = base - shift_d - Aw @ w
    return -Aw.T @ np.exp(z - logsumexp(z))


con2 = [{'type': 'ineq', 'fun': lambda w: r95 ** 2 - w @ w, 'jac': lambda w: -2 * w}]
best2 = None
for s in range(8):
    rr = np.random.default_rng(100 + s)
    w0 = np.zeros(n) if s == 0 else rr.standard_normal(n) * r95 / np.sqrt(n)
    r = optimize.minimize(f2, w0, jac=g2, constraints=con2, method='SLSQP',
                          options=dict(maxiter=800, ftol=1e-14))
    if best2 is None or r.fun < best2.fun:
        best2 = r
rec('offset_region_zeff_corrected', dict(
    allowed_region_max_norm_Sigma_y=max_norm, exact_breakdown_radius_Sigma_y=breakdown,
    ratio=max_norm / breakdown, min_M_over_region=float(np.exp(best2.fun)),
    M_after_removing_best_estimate=float(np.exp(logsumexp(base - shift_d) - np.log(G)))),
    'shift_bound.py quantities (alpha=1/3) with the z_eff-corrected d: 95% data-allowed region centred on d_c; ratio of its largest Sigma_y-norm to the breakdown radius; min of M_joint(1/3) over it; M_joint(1/3) after removing delta = d_c. The values with the uncorrected d are in calibration_budget (allowed_over_breakdown_ratio, min_M_over_allowed_region, M_at_best_estimate_offset).')

# ------------------------------------------------------------------ 7. weights
M1 = float(np.exp(mixture_log_e_from_residuals(e1[:, None], K1)[0]))
M2 = float(np.exp(mixture_log_e_from_residuals(e2[:, None], K2)[0]))
w2_star = (THR - M1) / (M2 - M1)
wts = dict(
    equal=(0.5, 0.5), third_each_three_releases=(1 / 3, 1 / 3), one_third_two_thirds=(1 / 3, 2 / 3),
    precision_quarter_three_quarters=(0.25, 0.75), dr1_free_half=(0.0, 0.5))
wv_ = {k: dict(w=list(v), S=v[0] * M1 + v[1] * M2, p=1 / (v[0] * M1 + v[1] * M2)) for k, v in wts.items()}
rec('weights_at_DR2', wv_, 'S = w1 M_DR1 + w2 M_DR2 and its Markov bound 1/S under several conventions. dr1_free_half = M_DR2/2 (needs no assumption about DR1); precision weights ~ 1:3 (median C2/C1 = 0.32).')
rec('weight_on_DR2_for_S_20', w2_star, 'With w1 + w2 = 1, S >= 20 iff w2 >= this value.')

# ------------------------------------------------------------------ 8. results of code/
sig = result_json('sigma_mc_results.json')
p2 = sig['mc']['p_emp']
rec('exact_tail_bonferroni', dict(p_DR2_exact_tail=p2, ci95=sig['mc']['p_emp_ci'], N=sig['mc']['N'],
                                  bonferroni_two_looks=2 * p2, bonferroni_ci95_upper=2 * sig['mc']['p_emp_ci'][1]),
    'Monte Carlo null tail P(M_DR2 >= 33.97 | LCDM) under the published DR2 Gaussian likelihood (results/sigma_mc_results.json, the sigma_emp = 3.69 run); a Bonferroni correction for two looks, valid under any dependence between releases, doubles it.')
bg = result_json('background_sensitivity.json')
mins = min(r['M_DR2'] for r in bg)
rec('background_worst_case', dict(min_M_DR2=mins, p=1 / mins, per_column={r['label']: r['M_DR2'] for r in bg}),
    'Minimum of M_DR2 over the four fixed Planck backgrounds of Table 4 (results/background_sensitivity.json).')

# DR2 with the Table IV errors/correlations (means identical to the file)
T4 = {0.295: ('DV', 0.075), 0.51: (0.167, 0.425, -0.459), 0.706: (0.177, 0.330, -0.404),
      0.934: (0.152, 0.193, -0.416), 1.321: (0.318, 0.221, -0.434), 1.484: (0.760, 0.516, -0.500),
      2.33: (0.531, 0.101, -0.431)}
CT = np.zeros_like(ds2.cov)
zq = [(round(float(z), 3), QN[q]) for z, q in zip(ds2.z_eff, ds2.quantities)]
for i, (z, q) in enumerate(zq):
    t = T4[z]
    if q == 'DV':
        CT[i, i] = t[1] ** 2
    else:
        CT[i, i] = (t[0] if q == 'DM' else t[1]) ** 2
for i, (z, q) in enumerate(zq):
    for j, (z2_, q2) in enumerate(zq):
        if i != j and z == z2_:
            t = T4[z]
            CT[i, j] = t[2] * t[0] * t[1]
KT = precompute_kernels(ds2.z_eff, ds2.quantities, CT, EJ.W0_GRID, EJ.WA_GRID)
MT = float(np.exp(mixture_log_e_from_residuals((ds2.data - KT['mu_null'])[:, None], KT)[0]))
CL = ds2.cov.copy()
iL3 = [i for i, (z, q) in enumerate(zq) if z == 0.934]
CL[np.ix_(iL3, iL3)] = CT[np.ix_(iL3, iL3)]
KL = precompute_kernels(ds2.z_eff, ds2.quantities, CL, EJ.W0_GRID, EJ.WA_GRID)
M_L3only = float(np.exp(mixture_log_e_from_residuals((ds2.data - KL['mu_null'])[:, None], KL)[0]))
rel = {f'{TRACER.get(z, "QSO")} {q}': float(np.sqrt(ds2.cov[i, i]) / np.sqrt(CT[i, i]) - 1) for i, (z, q) in enumerate(zq)}
r_file = {TRACER.get(z, 'QSO'): float(ds2.cov[i, i + 1] / np.sqrt(ds2.cov[i, i] * ds2.cov[i + 1, i + 1]))
          for i, (z, q) in enumerate(zq) if i + 1 < len(zq) and zq[i + 1][0] == z}
r_tab = {TRACER.get(z, 'QSO'): T4[z][2] for z in T4 if z != 0.295}
rec('DR2_table_IV_covariance', dict(M_DR2=MT, p=1 / MT, M_DR2_only_LRG3ELG1_block_from_tableIV=M_L3only, file_sigma_over_tableIV_minus_1=rel,
                                    r_DM_DH_file=r_file, r_DM_DH_tableIV=r_tab),
    'M_DR2 recomputed with the errors and D_M-D_H correlations printed in DR2 Results II Table IV (means as in the CobayaSampler file, which match Table IV) instead of the CobayaSampler desi_gaussian_bao_ALL_GCcomb covariance.')

rec('external_quotes', dict(
    desi_dr1_dr2_largest_differences={'LRG3+ELG1 alpha_iso': 1.56, 'LRG2 alpha_AP': 1.93},
    desi_dr1_dr2_KS_p=0.40, desi_KS_statistic=0.25, desi_lya_correlation=0.61,
    desi_printed_joint_sigma='sqrt(sigma_DR2^2 + sigma_DR1^2 - 2 C sigma_DR1) [typo; read -2 C sigma_DR1 sigma_DR2]',
    desi_main_text='As DR1 is a subset of DR2, and the data reduction and analysis pipelines are very similar, we conservatively assume a perfect correlation',
    tableIV_LRG3ELG1=dict(DM=(21.576, 0.152), DH=(17.641, 0.193), r=-0.416)),
    'Quoted, not computed: DESI DR2 Results II (arXiv:2503.14738, PRD 112, 083515) Sec. III.3.1, footnote 12 and Table IV, verified against the arXiv LaTeX source of 2503.14738v3.')
R['meta'] = dict(value=dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), runtime_seconds=round(time.time() - t0, 1)),
                 definition='(meta)', script=THIS)
NUM0['structured_families'] = py(R)
save_numbers(NUM0)
for k, v in R.items():
    print(k, json.dumps(py(v['value']))[:600])
print('wrote', OUT)
