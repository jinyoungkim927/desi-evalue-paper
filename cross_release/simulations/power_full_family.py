#!/usr/bin/env python3
"""Power and false-positive rate of the FULL-family worst case, inf over alpha in [0, alpha_max).

power_dr2_mle.py takes the family worst case over 7 alphas in [0.25, 0.40]; the family of
Appendix B.2 is the whole admissible range [0, alpha_max), so this script recomputes, on the SAME draws (seed 20260831, batch 1000,
N = 20000, DR2-MLE alternative in both releases, alpha_true = 1/3 noise), the
statistic max(M_DR1, inf_{alpha in [0, alpha_max)} M_joint(alpha)).

Near alpha_max the innovation covariance C2 - alpha^2 C1 becomes singular along
the generalized eigenvector v of lambda_min; there
    log LR_g ~ [(Dm_g.v)(y.v) - (Dm_g.v)^2/2] / eps,   eps -> 0,
so M_joint -> +inf if f_top = max_g[(Dm_g.v)(y.v) - (Dm_g.v)^2/2] > 0 and -> 0
if f_top < 0 (the 30x30 grid does not contain LCDM exactly, so Dm_g.v != 0).
The inf over the open interval is therefore 0 whenever f_top < 0; otherwise it
is attained in the interior and is evaluated on a dense alpha grid
(0.0025 steps plus 30 log-spaced points toward alpha_max), which gives an upper
bound on the inf (so the power quoted is an upper bound; the grid error in
log M is < 1e-3 given the curvature).

Writes results/simulations/power_full_family_results.json and prints a summary.

The output of record of this script is cross_release/inputs/power_full_family_results.json, read by
cross_release/one_fraction_family.py. Running the script again writes a fresh copy to
results/simulations/ and prints the log to stdout.
Run from the repository root: python cross_release/simulations/power_full_family.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import linalg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, OUT_DIR, INPUTS, add_code_path  # noqa: E402
add_code_path()
REPO = ROOT
SIM_OUT = OUT_DIR / 'simulations'   # results/simulations/
from data_loader import load_desi_data                      # noqa: E402
from cosmology import CosmologyParams, LCDM, compute_bao_predictions  # noqa: E402
from evalue_analysis import (precompute_kernels, mixture_log_e_from_residuals,  # noqa: E402
                             _build_theory_vector)
from eprocess_joint import (W0_GRID, WA_GRID, DR2_MLE, build_delta_matrix,  # noqa: E402
                            joint_kernel, decomposition_for_alpha)
from eprocess_hierarchical_mc import build_bin_matching, nearest_psd  # noqa: E402

LOG20 = np.log(20.0)
A_TRUE = 1.0 / 3.0
FAM7 = [0.25, 0.275, 0.30, 1.0 / 3.0, 0.35, 0.375, 0.40]
N_ALT, N_H0 = 20000, 20000
SEED_ALT, SEED_H0 = 20260831, 20260832


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


t0 = time.time()
ds1 = load_desi_data(REPO / 'data' / 'dr1', 'DR1')
ds2 = load_desi_data(REPO / 'data' / 'dr2', 'DR2')
n1, n2 = len(ds1.data), len(ds2.data)
K1 = precompute_kernels(ds1.z_eff, ds1.quantities, ds1.cov, W0_GRID, WA_GRID)
K2 = precompute_kernels(ds2.z_eff, ds2.quantities, ds2.cov, W0_GRID, WA_GRID)
_, d1 = build_delta_matrix(ds1.z_eff, ds1.quantities, W0_GRID, WA_GRID)
_, d2 = build_delta_matrix(ds2.z_eff, ds2.quantities, W0_GRID, WA_GRID)
dr2_to_dr1, _ = build_bin_matching(ds1, ds2)
pairs = [(i, j) for j, i in enumerate(dr2_to_dr1) if i >= 0]
m1 = np.array([i for i, j in pairs], dtype=int)
m2 = np.array([j for i, j in pairs], dtype=int)
u2 = np.array([j for j in range(n2) if dr2_to_dr1[j] < 0], dtype=int)
n_match, n_u = len(pairs), len(u2)
C1M = ds1.cov[np.ix_(m1, m1)]
C2M = ds2.cov[np.ix_(m2, m2)]
cov_u = nearest_psd(ds2.cov[np.ix_(u2, u2)])[0] + 1e-14 * np.eye(n_u)

lam, V = linalg.eigh(C2M, C1M)
alpha_max = float(np.sqrt(lam.min()))
v0 = V[:, 0]

a_grid = np.unique(np.concatenate([
    np.round(np.arange(0.0, 0.4601, 0.0025), 6),
    alpha_max - np.logspace(np.log10(alpha_max - 0.4605), -7, 30),
    FAM7]))
a_grid = a_grid[a_grid < alpha_max]
kerns = []
for a in a_grid:
    cy, _, psd = decomposition_for_alpha(C1M, C2M, a)
    assert psd, a
    cy = cy + 1e-14 * np.eye(n_match)
    kerns.append(joint_kernel(d1, d2, m1, m2, u2, ds1.cov, (1 - a) ** 2 * cy, cov_u, a))
A1, Au = kerns[0]['A1'], kerns[0]['Au']              # alpha-independent
c1u = kerns[0]['const1'] + kerns[0]['constu']
Ay = np.stack([k['Ay'] for k in kerns])              # (nA, G, n_match)
cy_all = np.stack([k['consty'] for k in kerns])      # (nA, G)
i13 = int(np.argmin(np.abs(a_grid - 1 / 3)))
i7 = [int(np.argmin(np.abs(a_grid - a))) for a in FAM7]
Dm_edge = d2[:, m2] - alpha_max * d1[:, m1]
proj = Dm_edge @ v0                                   # (G,)

alt = CosmologyParams(w0=DR2_MLE[0], wa=DR2_MLE[1])
off1 = (_build_theory_vector(compute_bao_predictions(ds1.z_eff, alt), ds1.z_eff, ds1.quantities)
        - _build_theory_vector(compute_bao_predictions(ds1.z_eff, LCDM), ds1.z_eff, ds1.quantities))
off2 = (_build_theory_vector(compute_bao_predictions(ds2.z_eff, alt), ds2.z_eff, ds2.quantities)
        - _build_theory_vector(compute_bao_predictions(ds2.z_eff, LCDM), ds2.z_eff, ds2.quantities))
cy_t, _, _ = decomposition_for_alpha(C1M, C2M, A_TRUE)
cy_t = cy_t + 1e-14 * np.eye(n_match)
L1 = K1['L_chol']; L23 = np.linalg.cholesky(cy_t); Lu = np.linalg.cholesky(cov_u)


def lme(X):
    m = X.max(axis=0)
    return m + np.log(np.mean(np.exp(X - m), axis=0))


def simulate(n_draws, seed, with_signal, batch=1000):
    rng = np.random.default_rng(seed)
    out = dict(M1=[], M2=[], J13=[], J7min=[], Jgridmin=[], Jgrid_argmin=[], ftop=[])
    done = 0
    while done < n_draws:
        b = min(batch, n_draws - done)
        z1 = L1 @ rng.standard_normal((n1, b))
        e23 = L23 @ rng.standard_normal((n_match, b))
        eu = Lu @ rng.standard_normal((n_u, b))
        eps2 = np.zeros((n2, b))
        eps2[m2] = A_TRUE * z1[m1] + (1.0 - A_TRUE) * e23
        eps2[u2] = eu
        eps1 = z1
        if with_signal:
            eps1 = eps1 + off1[:, None]
            eps2 = eps2 + off2[:, None]
        out['M1'].append(mixture_log_e_from_residuals(eps1, K1))
        out['M2'].append(mixture_log_e_from_residuals(eps2, K2))
        base = A1 @ eps1 + Au @ eps2[u2] - 0.5 * c1u[:, None]
        lJ = np.empty((len(a_grid), b))
        for k, a in enumerate(a_grid):
            y = eps2[m2] - a * eps1[m1]
            lJ[k] = lme(base + Ay[k] @ y - 0.5 * cy_all[k][:, None])
        out['J13'].append(lJ[i13])
        out['J7min'].append(lJ[i7].min(axis=0))
        out['Jgridmin'].append(lJ.min(axis=0))
        out['Jgrid_argmin'].append(a_grid[lJ.argmin(axis=0)])
        y_edge = eps2[m2] - alpha_max * eps1[m1]
        yv = v0 @ y_edge                                  # (b,)
        f = proj[:, None] * yv[None, :] - 0.5 * proj[:, None] ** 2
        out['ftop'].append(f.max(axis=0))
        done += b
    return {k: np.concatenate(v) for k, v in out.items()}


def summarize(S, n):
    lM1 = S['M1']
    collapse = S['ftop'] < 0
    l_full = np.where(collapse, -np.inf, S['Jgridmin'])
    stats = dict(
        JOINT13=np.maximum(lM1, S['J13']),
        FAMINF_7alpha=np.maximum(lM1, S['J7min']),
        FAMINF_full=np.maximum(lM1, l_full),
        FAMINF_full_interior_only=np.maximum(lM1, S['Jgridmin']),
        M2=S['M2'],
        AVG=np.logaddexp(lM1, S['M2']) - np.log(2.0),
        PROD=lM1 + S['M2'])
    res = {}
    for name, ls in stats.items():
        k = int(np.sum(ls >= LOG20))
        res[name] = dict(rate=k / n, k=k, n=n, wilson95=wilson(k, n),
                         median=float(np.exp(np.median(ls))))
    res['fraction_collapsing_to_0_at_alpha_max'] = float(collapse.mean())
    res['fraction_interior_min_at_grid_edge'] = float(np.mean(S['Jgrid_argmin'] >= a_grid[-1] - 1e-12))
    res['median_argmin_alpha'] = float(np.median(S['Jgrid_argmin']))
    return res


# observed data check
e1o = (ds1.data - K1['mu_null'])[:, None]
e2o = (ds2.data - K2['mu_null'])[:, None]
base_o = A1 @ e1o + Au @ e2o[u2] - 0.5 * c1u[:, None]
lJo = np.array([lme(base_o + Ay[k] @ (e2o[m2] - a * e1o[m1]) - 0.5 * cy_all[k][:, None])[0]
                for k, a in enumerate(a_grid)])
yv_o = float(v0 @ (e2o[m2, 0] - alpha_max * e1o[m1, 0]))
ftop_o = float(np.max(proj * yv_o - 0.5 * proj ** 2))
print(f"observed: M_joint(1/3) = {np.exp(lJo[i13]):.2f}; grid min = {np.exp(lJo.min()):.3f} at "
      f"alpha = {a_grid[lJo.argmin()]:.4f}; f_top = {ftop_o:.4g} (>0: diverges at alpha_max)")

alt_S = simulate(N_ALT, SEED_ALT, True)
h0_S = simulate(N_H0, SEED_H0, False)
res = dict(alpha_max=alpha_max, n_alpha_grid=int(len(a_grid)),
           alternative=summarize(alt_S, N_ALT), H0_alpha_1_3=summarize(h0_S, N_H0),
           settings=dict(N_alt=N_ALT, N_H0=N_H0, seeds=[SEED_ALT, SEED_H0],
                         alternative='DR2 BAO MLE (w0, wa) = (-0.856, -0.430) in both releases',
                         noise='alpha_true = 1/3 construction law', threshold=20),
           observed=dict(grid_min=float(np.exp(lJo.min())), grid_argmin=float(a_grid[lJo.argmin()]),
                         f_top=ftop_o))
SIM_OUT.mkdir(parents=True, exist_ok=True)
(SIM_OUT / 'power_full_family_results.json').write_text(json.dumps(res, indent=1))
for lab in ('alternative', 'H0_alpha_1_3'):
    print(f"== {lab}")
    for k, v in res[lab].items():
        if isinstance(v, dict):
            print(f"  {k:28s} rate = {v['rate']:.4f} ({v['k']}/{v['n']}) Wilson95 [{v['wilson95'][0]:.4f}, {v['wilson95'][1]:.4f}]  median {v['median']:.4g}")
        else:
            print(f"  {k:40s} {v:.4f}")
print(f"[done] {time.time() - t0:.0f}s")
