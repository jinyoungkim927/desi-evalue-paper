#!/usr/bin/env python3
"""Mean-shift robustness of the joint (DR1, DR2) e-value construction.

Under a shifted null the matched-bin innovation y = eps2M - alpha*eps1M picks
up a relative bias delta_y (DR2 pipeline bias on matched bins; DR1 and the
unmatched DR2 block assumed unbiased). Then

  E[M_shifted] = mean_g exp(Ay_g . delta_y)          (exact, Gaussian MGF)
              <= mean_g exp(sqrt(consty_g) * b)      (Cauchy-Schwarz, sharper)
              <= exp(K * b),  K = sqrt(max_g consty_g),

for all ||delta_y||_{Sigma_y^-1} <= b, using Ay_g Sigma_y Ay_g' = consty_g.
So M * exp(-K b) is a valid e-value on the shifted-null class.

Computes K(alpha), breakdown budgets b* = ln(M/20)/K, sigma-translations,
family-level minima, and the deflated family worst case; verifies the mixture bound
exactly (closed form) and by Monte Carlo.

The output of record of this script is cross_release/inputs/logs/mean_shift_robustness.log (printed log), read by
cross_release/one_fraction_family.py. Running the script again writes a fresh copy to
results/simulations/ and prints the log to stdout.
Run from the repository root: python cross_release/simulations/mean_shift_robustness.py
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, OUT_DIR, INPUTS, add_code_path  # noqa: E402
add_code_path()
REPO = ROOT
SIM_OUT = OUT_DIR / 'simulations'   # results/simulations/

from data_loader import load_desi_data
from evalue_analysis import precompute_kernels, mixture_log_e_from_residuals
from eprocess_hierarchical_mc import build_bin_matching, nearest_psd
from eprocess_joint import (W0_GRID, WA_GRID, build_delta_matrix, joint_kernel,
                            joint_log_e, decomposition_for_alpha)

ds1 = load_desi_data(REPO / 'data' / 'dr1', 'DR1')
ds2 = load_desi_data(REPO / 'data' / 'dr2', 'DR2')
K1 = precompute_kernels(ds1.z_eff, ds1.quantities, ds1.cov, W0_GRID, WA_GRID)
K2 = precompute_kernels(ds2.z_eff, ds2.quantities, ds2.cov, W0_GRID, WA_GRID)
_, delta1 = build_delta_matrix(ds1.z_eff, ds1.quantities, W0_GRID, WA_GRID)
_, delta2 = build_delta_matrix(ds2.z_eff, ds2.quantities, W0_GRID, WA_GRID)

dr2_to_dr1, _ = build_bin_matching(ds1, ds2)
pairs = [(i, j) for j, i in enumerate(dr2_to_dr1) if i >= 0]
m1_idx = np.array([i for i, j in pairs], dtype=int)
m2_idx = np.array([j for i, j in pairs], dtype=int)
u2_idx = np.array([j for j in range(len(ds2.data)) if dr2_to_dr1[j] < 0], dtype=int)
n_match, n_u = len(pairs), len(u2_idx)
C1M = ds1.cov[np.ix_(m1_idx, m1_idx)]
C2M = ds2.cov[np.ix_(m2_idx, m2_idx)]
eps1_obs = ds1.data - K1['mu_null']
eps2_obs = ds2.data - K2['mu_null']
print(f"n_matched = {n_match}, n_unmatched_dr2 = {n_u}")


def build_at_alpha(a):
    """Replicates run_construction's kernel path; returns kern, Sigma_y, log M values."""
    cov_y23, min_eig, psd = decomposition_for_alpha(C1M, C2M, a)
    if not psd:
        return None
    cov_y23 = cov_y23 + 1e-14 * np.eye(n_match)
    Sigma_y = (1.0 - a) ** 2 * cov_y23
    if n_u:
        cov_u, _ = nearest_psd(ds2.cov[np.ix_(u2_idx, u2_idx)])
        cov_u = cov_u + 1e-14 * np.eye(n_u)
    else:
        cov_u = None
    kern = joint_kernel(delta1, delta2, m1_idx, m2_idx, u2_idx,
                        ds1.cov, Sigma_y, cov_u, a)
    y_obs = eps2_obs[m2_idx] - a * eps1_obs[m1_idx]
    e2u_obs = eps2_obs[u2_idx]
    logM = float(joint_log_e(eps1_obs[:, None], y_obs[:, None],
                             e2u_obs[:, None], kern)[0])
    return dict(kern=kern, Sigma_y=Sigma_y, cov_u=cov_u, logM=logM, alpha=a,
                cov_y23=cov_y23)


def robustness_numbers(res, thr=20.0):
    kern, Sy = res['kern'], res['Sigma_y']
    consty = kern['consty']
    # verify identity Ay Sigma_y Ay' = consty (basis of the whole bound)
    idmax = np.abs(np.einsum('gi,ij,gj->g', kern['Ay'], Sy, kern['Ay'])
                   - consty).max()
    K = float(np.sqrt(consty.max()))
    g_star = int(np.argmax(consty))
    M = np.exp(res['logM'])
    b_star = float((res['logM'] - np.log(thr)) / K)
    # uniform-shift translations
    Sy_inv = np.linalg.inv(Sy)
    sig_inn = np.sqrt(np.diag(Sy))                 # innovation sigmas
    sig_dr2 = np.sqrt(np.diag(C2M))                # published DR2 sigmas
    c_unif = float(np.sqrt(sig_inn @ Sy_inv @ sig_inn))
    c_dr2 = float(np.sqrt(sig_dr2 @ Sy_inv @ sig_dr2))
    return dict(alpha=res['alpha'], M=M, logM=res['logM'], K=K,
                b_star=b_star, c_unif=c_unif, c_dr2=c_dr2,
                s_star_inn=b_star / c_unif, s_star_dr2=b_star / c_dr2,
                g_star=g_star, identity_max_abs=float(idmax),
                consty=consty, Sy_inv=Sy_inv)


def sharper_deflator(consty, b):
    """mean_g exp(sqrt(consty_g) b) -- the sharper valid deflator."""
    r = np.sqrt(consty) * b
    m = r.max()
    return float(np.exp(m) * np.mean(np.exp(r - m)))


# ---------------- headline alpha = 1/3 ----------------
res3 = build_at_alpha(1.0 / 3.0)
r3 = robustness_numbers(res3)
consty = r3['consty']
print("\n=== alpha = 1/3 (headline) ===")
print(f"identity max |Ay Sy Ay' - consty| = {r3['identity_max_abs']:.2e}")
print(f"M_joint = {r3['M']:.2f}")
print(f"K = sqrt(max_g consty_g) = {r3['K']:.4f}")
q = np.sqrt(consty)
print(f"sqrt(consty_g) over 900 grid pts: min {q.min():.3f}, median {np.median(q):.3f}, "
      f"90% {np.quantile(q, .9):.3f}, max {q.max():.3f}")
gs = r3['g_star']
print(f"argmax grid point: w0 = {W0_GRID[gs // len(WA_GRID)]:.3f}, "
      f"wa = {WA_GRID[gs % len(WA_GRID)]:.3f}")
print(f"b* = ln(M/20)/K = {r3['b_star']:.4f}  (Sigma_y^-1 metric)")
print(f"c_unif (innovation sigmas) = {r3['c_unif']:.3f} -> s*_innovation = {r3['s_star_inn']:.4f}")
print(f"c_dr2 (published DR2 sigmas) = {r3['c_dr2']:.3f} -> s*_DR2sigma = {r3['s_star_dr2']:.4f}")

# sharper deflator comparison at a = 1/3
for b in (0.05, 0.1, r3['b_star'], 0.5, 1.0):
    Dk = np.exp(r3['K'] * b)
    Ds = sharper_deflator(consty, b)
    print(f"  b = {b:.4f}: K-deflator e^Kb = {Dk:.4g}, sharper mean-deflator = {Ds:.4g}, "
          f"M/e^Kb = {r3['M']/Dk:.4g}, M/mean = {r3['M']/Ds:.4g}")
# sharper breakdown budget: solve M / D(b) = 20 by bisection
from scipy.optimize import brentq
b_sharp = brentq(lambda b: r3['M'] / sharper_deflator(consty, b) - 20.0, 1e-6, 5.0)
print(f"sharper breakdown budget (mean-deflator) at a=1/3: b*_sharp = {b_sharp:.4f}, "
      f"s*_sharp_inn = {b_sharp / r3['c_unif']:.4f}, s*_sharp_DR2 = {b_sharp / r3['c_dr2']:.4f}")

# ---------------- exact + MC verification of the mixture bound ----------------
print("\n=== verification of the mixture bound (alpha = 1/3) ===")
rng = np.random.default_rng(20260831)
kern, Sy = res3['kern'], res3['Sigma_y']
b_test = 0.5
# worst-case direction for each of several grid points + random directions
Ls = np.linalg.cholesky(Sy)
worst = -np.inf
for g in [gs, int(np.argmax(consty)) // 2, 0, 450, 899]:
    d = b_test * (Sy @ kern['Ay'][g]) / np.sqrt(consty[g])
    nrm = np.sqrt(d @ r3['Sy_inv'] @ d)
    Em = np.mean(np.exp(np.minimum(kern['Ay'] @ d, 700)))
    worst = max(worst, Em)
    tag = " (argmax g*)" if g == gs else ""
    print(f"  delta along g={g}{tag}: ||delta|| = {nrm:.4f}, "
          f"exact E[M_shifted] = {Em:.4g} vs bounds: mean-deflator "
          f"{sharper_deflator(consty, b_test):.4g}, e^Kb = {np.exp(r3['K']*b_test):.4g}")
for _ in range(2000):
    u = rng.standard_normal(n_match)
    d = Ls @ u
    d *= b_test / np.sqrt(d @ r3['Sy_inv'] @ d)
    worst = max(worst, np.mean(np.exp(np.minimum(kern['Ay'] @ d, 700))))
print(f"  max exact E[M_shifted] over worst-case + 2000 random directions "
      f"(b = {b_test}): {worst:.4g} <= e^Kb = {np.exp(r3['K']*b_test):.4g}")

# Monte Carlo confirmation at the worst-case direction (heavy-tailed: expect
# the MC mean to sit at or below the exact value; this checks the sign/shape,
# the exact closed form above is the real proof).
d_star = b_test * (Sy @ kern['Ay'][gs]) / np.sqrt(consty[gs])
nmc = 400000
L1 = K1['L_chol']
Lu = np.linalg.cholesky(res3['cov_u'])
acc = 0.0
exact = float(np.mean(np.exp(kern['Ay'] @ d_star)))
for _ in range(4):
    e1 = L1 @ rng.standard_normal((len(ds1.data), nmc // 4))
    yv = Ls @ rng.standard_normal((n_match, nmc // 4)) + d_star[:, None]
    eu = Lu @ rng.standard_normal((n_u, nmc // 4))
    acc += np.exp(joint_log_e(e1, yv, eu, kern)).sum()
mc_mean = acc / nmc
print(f"  MC E[M_shifted] at worst-case delta (N = {nmc}): {mc_mean:.4g} "
      f"(exact {exact:.4g}, bound e^Kb = {np.exp(r3['K']*b_test):.4g})")

# unmatched block NOT covered: its own K would be
constu = res3['kern']['constu']
print(f"  unmatched block (NOT covered by the bound): K_u = sqrt(max constu) = "
      f"{np.sqrt(constu.max()):.4f}")

# ---------------- family sweep ----------------
print("\n=== family sweep: K(a), b*(a), deflated values ===")
alphas = sorted(set(list(np.round(np.arange(0.02, 0.47, 0.02), 3))
                    + [0.25, 0.30, 1.0 / 3.0, 0.36, 0.40, 0.44, 0.46]))
rows = []
print(f"{'alpha':>7} {'M(a)':>10} {'K(a)':>8} {'b*(a)':>8} {'s*_inn':>7} "
      f"{'M e^-0.5K':>10} {'M e^-1.0K':>10}")
for a in alphas:
    res = build_at_alpha(a)
    if res is None:
        print(f"{a:>7.3f}  inadmissible (C_y23 not PSD)")
        continue
    r = robustness_numbers(res)
    d05 = r['M'] * np.exp(-r['K'] * 0.5)
    d10 = r['M'] * np.exp(-r['K'] * 1.0)
    rows.append((a, r['M'], r['K'], r['b_star'], r['s_star_inn'], d05, d10))
    print(f"{a:>7.3f} {r['M']:>10.4g} {r['K']:>8.3f} {r['b_star']:>8.4f} "
          f"{r['s_star_inn']:>7.4f} {d05:>10.4g} {d10:>10.4g}")

arr = np.array(rows)
full = arr
capped = arr[arr[:, 0] <= 0.401]  # the scanned range alpha <= 0.40 of Appendix B.2
for name, A in (("full admissible grid", full), ("capped a <= 0.40", capped)):
    i = int(np.argmin(A[:, 3]))
    print(f"family-level breakdown ({name}): min_a b*(a) = {A[i,3]:.4f} at a = {A[i,0]:.3f}")
    for bb, col in ((0.5, 5), (1.0, 6)):
        j = int(np.argmin(A[:, col]))
        print(f"  family worst case deflated at b = {bb}: inf_a M(a) e^-K(a)b = {A[j,col]:.4g} "
              f"at a = {A[j,0]:.3f}")

# selected values
print("\nselected alphas:")
for a in (0.25, 0.30, 1.0 / 3.0, 0.40):
    r = robustness_numbers(build_at_alpha(a))
    print(f"  a = {a:.4f}: M = {r['M']:.3f}, K = {r['K']:.4f}, b* = {r['b_star']:.4f}, "
          f"s*_inn = {r['s_star_inn']:.4f}, s*_DR2 = {r['s_star_dr2']:.4f}")
