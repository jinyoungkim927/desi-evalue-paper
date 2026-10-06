#!/usr/bin/env python3
"""Data-allowed shift bound vs robustness budget for the joint e-value (a=1/3).

Model: DR1 = mu + eps1, DR2 = mu + delta + a*eps1 + (1-a)*eps23  (delta = relative
bias of the DR2 reprocessing on the 11 matched bins). Then d = DR2 - DR1 has
mean delta and cov S = (1-2a)C1 + C2.  The construction's innovation
y = eps2M - a*eps1M has mean delta and cov Sigma_y = C2 - a^2 C1 (code's
cov_y23_scaled).  Note ||delta_y||_{C_y23^-1} with delta_y = delta/(1-a),
C_y23 = Sigma_y/(1-a)^2 is IDENTICAL to ||delta||_{Sigma_y^-1}: the (1-a)
factors cancel, so the budget metric is delta' (C2 - a^2 C1)^{-1} delta.

Run (from the repository root, about 5 s): python cross_release/shift_bound.py
Also run by one_fraction_family.py, which records its quantities under calibration_budget.
"""
import sys
import numpy as np
from pathlib import Path
from scipy import stats, linalg, optimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import ROOT, add_code_path  # noqa: E402
add_code_path()
REPO = ROOT
from data_loader import load_desi_data
from evalue_analysis import precompute_kernels
from eprocess_hierarchical_mc import build_bin_matching, nearest_psd
import eprocess_joint as EJ

a = 1.0 / 3.0
ds1 = load_desi_data(REPO / 'data' / 'dr1', 'DR1')
ds2 = load_desi_data(REPO / 'data' / 'dr2', 'DR2')
K1 = precompute_kernels(ds1.z_eff, ds1.quantities, ds1.cov, EJ.W0_GRID, EJ.WA_GRID)
K2 = precompute_kernels(ds2.z_eff, ds2.quantities, ds2.cov, EJ.W0_GRID, EJ.WA_GRID)
mu1, d1 = EJ.build_delta_matrix(ds1.z_eff, ds1.quantities, EJ.W0_GRID, EJ.WA_GRID)
mu2, d2 = EJ.build_delta_matrix(ds2.z_eff, ds2.quantities, EJ.W0_GRID, EJ.WA_GRID)

dr2_to_dr1, _ = build_bin_matching(ds1, ds2)
pairs = [(i, j) for j, i in enumerate(dr2_to_dr1) if i >= 0]
m1 = np.array([i for i, _ in pairs]); m2 = np.array([j for _, j in pairs])
u2 = np.array([j for j in range(len(ds2.data)) if dr2_to_dr1[j] < 0], dtype=int)
n = len(pairs)

C1 = ds1.cov[np.ix_(m1, m1)]
C2 = ds2.cov[np.ix_(m2, m2)]
d = ds2.data[m2] - ds1.data[m1]                      # observed DR2 - DR1
S = (1 - 2 * a) * C1 + C2                           # cov of d under the model
Sigma_y = C2 - a**2 * C1                            # cov of y = eps2M - a eps1M
Sy_inv = np.linalg.inv(Sigma_y)
S_inv = np.linalg.inv(S)

# ---------- (1) observed difference in the S metric ----------
chi2_obs = float(d @ S_inv @ d)
norm_d_S = np.sqrt(chi2_obs)
print(f"n matched = {n}")
print(f"(1) chi2 = ||d||^2_(S^-1) = {chi2_obs:.3f}   ||d||_S = {norm_d_S:.3f}"
      f"   (p = {stats.chi2.sf(chi2_obs, n):.3f})")

# ---------- (2) 95% upper bound on ||delta||_{S^-1} ----------
r95 = float(np.sqrt(stats.chi2.ppf(0.95, n)))
ub_S = norm_d_S + r95
print(f"(2) r95 = sqrt(chi2_11,0.95) = sqrt({stats.chi2.ppf(0.95, n):.3f}) = {r95:.3f}")
print(f"    95% bound: ||delta||_S <= ||d||_S + r95 = {ub_S:.3f}")

# ---------- (3) metric conversion S^-1 -> Sigma_y^-1 ----------
# factor(delta) = ||delta||_{Sigma_y^-1} / ||delta||_{S^-1};
# extremes = sqrt(gen eigs of (S, Sigma_y)).
gev = linalg.eigh(S, Sigma_y, eigvals_only=True)
c_max, c_min = float(np.sqrt(gev.max())), float(np.sqrt(gev.min()))
norm_d_Sy = float(np.sqrt(d @ Sy_inv @ d))
print(f"(3) conversion factor sqrt(eig(S, Sigma_y)): min = {c_min:.3f}, "
      f"max (worst case) = {c_max:.3f}")
print(f"    observed direction: ||d||_Sigma_y = {norm_d_Sy:.3f} "
      f"(factor {norm_d_Sy / norm_d_S:.3f})")

# exact max of ||delta||_{Sigma_y^-1} over the 95% ellipsoid
# {(delta-d)' S^-1 (delta-d) <= r95^2}: delta = d + L w, L = chol(S), ||w||<=r.
L = np.linalg.cholesky(S)
A = L.T @ Sy_inv @ L
lam, V = np.linalg.eigh(A)
b = V.T @ (L.T @ (Sy_inv @ d))
def wnorm(mu):  # ||w(mu)||, w_i = b_i/(mu - lam_i), mu > lam_max
    return np.sqrt(np.sum((b / (mu - lam))**2))
lo, hi = lam.max() * (1 + 1e-12) + 1e-12, lam.max() + 1e6
while wnorm(lo) < r95: lo = lam.max() + (lo - lam.max()) / 10
while wnorm(hi) > r95: hi *= 10
for _ in range(200):
    mid = 0.5 * (lo + hi)
    if wnorm(mid) > r95: lo = mid
    else: hi = mid
w = b / (mid - lam)
max_sq = float(d @ Sy_inv @ d + 2 * b @ w + w @ (lam * w))
ub_Sy_exact = np.sqrt(max_sq)
ub_Sy_triangle = norm_d_Sy + c_max * r95
print(f"    95%-allowed worst-case ||delta||_Sigma_y: exact = {ub_Sy_exact:.3f} "
      f"(triangle bound {ub_Sy_triangle:.3f})")

# ---------- (4) robustness budget from the kernel ----------
cov_y23, min_eig, psd = EJ.decomposition_for_alpha(C1, C2, a)
assert psd
cov_y23 = cov_y23 + 1e-14 * np.eye(n)
cov_y23_scaled = (1 - a)**2 * cov_y23
cov_u = nearest_psd(ds2.cov[np.ix_(u2, u2)])[0] + 1e-14 * np.eye(len(u2))
kern = EJ.joint_kernel(d1, d2, m1, m2, u2, ds1.cov, cov_y23_scaled, cov_u, a)

Kmax = float(np.sqrt(kern['consty'].max()))
eps1_obs = ds1.data - K1['mu_null']
eps2_obs = ds2.data - K2['mu_null']
y_obs = eps2_obs[m2] - a * eps1_obs[m1]
e2u_obs = eps2_obs[u2]
base = (kern['A1'] @ eps1_obs + kern['Ay'] @ y_obs + kern['Au'] @ e2u_obs
        - 0.5 * kern['const'])
G = len(base)
logM = float(linalg.logsumexp(base) - np.log(G)) if hasattr(linalg, 'logsumexp') \
    else float(np.log(np.mean(np.exp(base - base.max()))) + base.max())
from scipy.special import logsumexp
logM = float(logsumexp(base) - np.log(G))
M_joint = float(np.exp(logM))
b_star = float(np.log(M_joint / 20.0) / Kmax)
# K at the DR2 MLE grid point for context
gg = int(np.argmin([(w0 - EJ.DR2_MLE[0])**2 + (wa - EJ.DR2_MLE[1])**2
                    for w0 in EJ.W0_GRID for wa in EJ.WA_GRID]))
K_mle = float(np.sqrt(kern['consty'][gg]))
print(f"(4) M_joint(a=1/3) = {M_joint:.2f}  (check vs 66.7)")
print(f"    K = sqrt(max consty) = {Kmax:.3f}   (at DR2 MLE gridpoint: {K_mle:.3f})")
print(f"    conservative budget b* = ln(M/20)/K = {b_star:.4f}  [units of ||delta||_Sigma_y]")

# exact breakdown radius: smallest ||delta||_{Sigma_y^-1} that pushes M below 20.
Ly = np.linalg.cholesky(Sigma_y)          # delta = Ly v, ||v|| = ||delta||_Sigma_y
Av = kern['Ay'] @ Ly                      # (G, n): base_g - Av_g . v after y -> y - delta

def min_logM_at_radius(t, x0=None):
    def f(v): return logsumexp(base - Av @ v) - np.log(G)
    def gr(v):
        z = base - Av @ v
        wgt = np.exp(z - logsumexp(z))
        return -Av.T @ wgt
    con = [{'type': 'ineq', 'fun': lambda v: t**2 - v @ v,
            'jac': lambda v: -2 * v}]
    best = None
    for s in range(4):
        rng = np.random.default_rng(s)
        v0 = x0 if (x0 is not None and s == 0) else rng.standard_normal(n) * t / np.sqrt(n)
        r = optimize.minimize(f, v0, jac=gr, constraints=con, method='SLSQP',
                              options=dict(maxiter=500, ftol=1e-12))
        if best is None or r.fun < best.fun: best = r
    return best

lo_t, hi_t = 0.0, 1.0
while min_logM_at_radius(hi_t).fun > np.log(20): hi_t *= 2
for _ in range(40):
    mid_t = 0.5 * (lo_t + hi_t)
    if min_logM_at_radius(mid_t).fun > np.log(20): lo_t = mid_t
    else: hi_t = mid_t
b_exact = 0.5 * (lo_t + hi_t)
print(f"    EXACT breakdown radius (min ||delta||_Sigma_y with M<20) = {b_exact:.4f}")

# ---------- (5) the direct question: min M over the 95% data-allowed region ----
# delta = d + L w, ||w|| <= r95 (S-metric ball); shift y -> y - delta.
Aw = kern['Ay'] @ L
shift_d = kern['Ay'] @ d
def f2(w): return logsumexp(base - shift_d - Aw @ w) - np.log(G)
def g2(w):
    z = base - shift_d - Aw @ w
    wgt = np.exp(z - logsumexp(z))
    return -Aw.T @ wgt
con2 = [{'type': 'ineq', 'fun': lambda w: r95**2 - w @ w, 'jac': lambda w: -2 * w}]
best2 = None
for s in range(6):
    rng = np.random.default_rng(100 + s)
    w0 = np.zeros(n) if s == 0 else rng.standard_normal(n) * r95 / np.sqrt(n)
    r = optimize.minimize(f2, w0, jac=g2, constraints=con2, method='SLSQP',
                          options=dict(maxiter=800, ftol=1e-14))
    if best2 is None or r.fun < best2.fun: best2 = r
M_min_region = float(np.exp(best2.fun))
delta_at_min = d + L @ best2.x
norm_at_min_Sy = float(np.sqrt(delta_at_min @ Sy_inv @ delta_at_min))
norm_at_min_S = float(np.sqrt(delta_at_min @ S_inv @ delta_at_min))

# M at the point-estimate bias delta = d (center of the confidence region)
M_at_d = float(np.exp(logsumexp(base - shift_d) - np.log(G)))
print(f"(5) M at delta = d (point estimate of bias):        M = {M_at_d:.4g}")
print(f"    min M over 95% region (exact):                  M = {M_min_region:.4g}")
print(f"    (attained at ||delta||_Sigma_y = {norm_at_min_Sy:.3f}, "
      f"||delta||_S = {norm_at_min_S:.3f})")

# smallest ||delta - d||_{S^-1} (distance from the point estimate, i.e. how
# deep into the confidence region) needed to bring M below 20
def min_logM_ball_around_d(t):
    conb = [{'type': 'ineq', 'fun': lambda w: t**2 - w @ w, 'jac': lambda w: -2 * w}]
    best = None
    for s in range(4):
        rng = np.random.default_rng(200 + s)
        w0 = np.zeros(n) if s == 0 else rng.standard_normal(n) * t / np.sqrt(n)
        r = optimize.minimize(f2, w0, jac=g2, constraints=conb, method='SLSQP',
                              options=dict(maxiter=500, ftol=1e-12))
        if best is None or r.fun < best.fun: best = r
    return best.fun

if f2(np.zeros(n)) < np.log(20):
    print("    delta = d ALONE already gives M < 20 -> required depth: 0 "
          "(the center of the confidence region kills the rejection)")
    # also: smallest ||delta||_S from ZERO that kills it, for scale
lo_t, hi_t = 0.0, r95
fn0 = None
# smallest ||delta||_{S^-1} from the origin with M < 20 (no data constraint):
Aw0 = kern['Ay'] @ L
def f0(w): return logsumexp(base - Aw0 @ w) - np.log(G)
def g0(w):
    z = base - Aw0 @ w
    wgt = np.exp(z - logsumexp(z)); return -Aw0.T @ wgt
def minf0(t):
    conb = [{'type': 'ineq', 'fun': lambda w: t**2 - w @ w, 'jac': lambda w: -2 * w}]
    best = None
    for s in range(3):
        rng = np.random.default_rng(300 + s)
        w0 = np.zeros(n) if s == 0 else rng.standard_normal(n) * t / np.sqrt(n)
        r = optimize.minimize(f0, w0, jac=g0, constraints=conb, method='SLSQP',
                              options=dict(maxiter=500, ftol=1e-12))
        if best is None or r.fun < best.fun: best = r
    return best.fun
lo_t, hi_t = 0.0, 2.0
while minf0(hi_t) > np.log(20): hi_t *= 2
for _ in range(30):
    mid_t = 0.5 * (lo_t + hi_t)
    if minf0(mid_t) > np.log(20): lo_t = mid_t
    else: hi_t = mid_t
kill_S = 0.5 * (lo_t + hi_t)
print(f"    smallest ||delta||_S (from 0, worst direction) with M < 20: {kill_S:.4f}"
      f"  (vs 95% allowance {ub_S:.2f})")

print()
print("=== SUMMARY ===")
print(f"budget metric = ||delta||_(Sigma_y^-1), Sigma_y = C2 - C1/9 (== ||delta_y||_(C_y23^-1))")
print(f"conservative breakdown budget b*      = {b_star:.4f}")
print(f"exact breakdown radius                = {b_exact:.4f}")
print(f"observed ||d|| in budget metric       = {norm_d_Sy:.3f}")
print(f"95% worst-case allowed ||delta||      = {ub_Sy_exact:.3f}")
print(f"ratio allowed/budget (conservative)   = {ub_Sy_exact / b_star:.0f}x")
print(f"ratio allowed/budget (exact)          = {ub_Sy_exact / b_exact:.0f}x")
print(f"min M over 95% data-allowed region    = {M_min_region:.3g}")
print(f"M at the point-estimate bias d        = {M_at_d:.3g}")
