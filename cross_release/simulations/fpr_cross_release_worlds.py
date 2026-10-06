#!/usr/bin/env python3
"""False-positive rates of six statistics in four cross-release worlds: each guarantee holds
where its assumptions hold and fails where they do not.

Four H0 (LCDM-true) cross-release worlds; six statistics per draw; FPR at the
paper's threshold 20 (alpha=0.05) with Wilson 95% CIs, plus MC means and
closed-form Gaussian means (MC means of heavy-tailed e-values are biased low,
so the closed forms are the reliable mean diagnostics).

Worlds (all preserve the published marginals C1, C2 exactly up to 1e-14 jitter):
  W1 one-fraction model, a_true = 1/3
  W2 one-fraction model, a_true = 0.25
  W3 independent releases
  W4 OUT-OF-FAMILY: joint Gaussian, marginals C1/C2, cross-cov diagonal-only
     Lambda_kk = 0.35*sqrt(C1[m1_k,m1_k]*C2[m2_k,m2_k]) on the 11 matched
     pairs (zero elsewhere).  NOT of the form a*C1M (C1M has off-diagonals);
     full 25-dim joint covariance verified PSD below.

Statistics: M1, M2 (snapshot mixtures), AVG=(M1+M2)/2, PROD=M1*M2,
JOINT13 = M_joint(alpha=1/3), FAMINF = min_a M_joint(a) over
a in {0.05,0.15,0.25,0.30,1/3,0.40,0.45} (all natively PSD-admissible).

The output of record of this script is cross_release/inputs/fpr_cross_release_worlds_results.json, read by
cross_release/one_fraction_family.py. Running the script again writes a fresh copy to
results/simulations/ and prints the log to stdout.
Run from the repository root: python cross_release/simulations/fpr_cross_release_worlds.py
"""
import sys, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, OUT_DIR, INPUTS, add_code_path  # noqa: E402
add_code_path()
REPO = ROOT
SIM_OUT = OUT_DIR / 'simulations'   # results/simulations/

from data_loader import load_desi_data
from evalue_analysis import precompute_kernels, mixture_log_e_from_residuals
from eprocess_joint import (W0_GRID, WA_GRID, build_delta_matrix, joint_kernel,
                            joint_log_e, decomposition_for_alpha)
from eprocess_hierarchical_mc import build_bin_matching, nearest_psd

N_DRAWS = 100000
BATCH = 5000
THR = 20.0
ALPHAS = [0.05, 0.15, 0.25, 0.30, 1.0 / 3.0, 0.40, 0.45]
A13 = ALPHAS.index(1.0 / 3.0)
LAM_COEF = 0.35
BASE_SEED = 20260831

# ---------------------------------------------------------------- setup
ds1 = load_desi_data(REPO / 'data' / 'dr1', 'DR1')
ds2 = load_desi_data(REPO / 'data' / 'dr2', 'DR2')
n1, n2 = len(ds1.data), len(ds2.data)
K1 = precompute_kernels(ds1.z_eff, ds1.quantities, ds1.cov, W0_GRID, WA_GRID)
K2 = precompute_kernels(ds2.z_eff, ds2.quantities, ds2.cov, W0_GRID, WA_GRID)
_, delta1 = build_delta_matrix(ds1.z_eff, ds1.quantities, W0_GRID, WA_GRID)
_, delta2 = build_delta_matrix(ds2.z_eff, ds2.quantities, W0_GRID, WA_GRID)

d21, _ = build_bin_matching(ds1, ds2)
pairs = [(i, j) for j, i in enumerate(d21) if i >= 0]
m1 = np.array([i for i, j in pairs], dtype=int)
m2 = np.array([j for i, j in pairs], dtype=int)
u2 = np.array([j for j in range(n2) if d21[j] < 0], dtype=int)
nm, nu = len(m1), len(u2)
C1, C2 = ds1.cov, ds2.cov
C1M = C1[np.ix_(m1, m1)]
C2M = C2[np.ix_(m2, m2)]
cov_u, _ = nearest_psd(C2[np.ix_(u2, u2)])
cov_u = cov_u + 1e-14 * np.eye(nu)

# 7 joint kernels, built exactly the way run_construction does
kerns, cov_y23s = [], []
for a in ALPHAS:
    cy, min_eig, psd = decomposition_for_alpha(C1M, C2M, a)
    assert psd, f"alpha={a} not natively PSD"
    cy = cy + 1e-14 * np.eye(nm)
    cy_scaled = (1.0 - a) ** 2 * cy
    kerns.append(joint_kernel(delta1, delta2, m1, m2, u2, C1, cy_scaled, cov_u, a))
    cov_y23s.append(cy)

# ------------------------------------------------------ sanity: observed data
eps1_obs = ds1.data - K1['mu_null']
eps2_obs = ds2.data - K2['mu_null']
M1_obs = float(np.exp(mixture_log_e_from_residuals(eps1_obs[:, None], K1)[0]))
M2_obs = float(np.exp(mixture_log_e_from_residuals(eps2_obs[:, None], K2)[0]))
logJ_obs = []
for a, kern in zip(ALPHAS, kerns):
    y = (eps2_obs[m2] - a * eps1_obs[m1])[:, None]
    logJ_obs.append(float(joint_log_e(eps1_obs[:, None], y, eps2_obs[u2][:, None], kern)[0]))
J13_obs = float(np.exp(logJ_obs[A13]))
FAM_obs = float(np.exp(min(logJ_obs)))
print(f"SANITY  M1={M1_obs:.4f} (want 1.05)  M2={M2_obs:.4f} (want 33.97)  "
      f"JOINT13={J13_obs:.4f} (want 66.71)")
print(f"        AVG={(M1_obs+M2_obs)/2:.4f}  PROD={M1_obs*M2_obs:.4f}  FAMINF={FAM_obs:.4f}")
assert abs(M1_obs - 1.05) < 0.02 and abs(M2_obs - 33.97) < 0.2 and abs(J13_obs - 66.71) < 0.3, \
    "sanity reproduction failed"

# --------------------------------------------------- world joint covariances
def world_cov(world):
    """Full 25-dim joint covariance of (eps1, eps2) for each world's truth."""
    S = np.zeros((n1 + n2, n1 + n2))
    S[:n1, :n1] = C1
    if world in ('W1', 'W2'):
        a = 1.0 / 3.0 if world == 'W1' else 0.25
        cy = cov_y23s[ALPHAS.index(a)]
        S22 = np.zeros((n2, n2))
        S22[np.ix_(m2, m2)] = a ** 2 * C1M + (1 - a) ** 2 * cy
        S22[np.ix_(u2, u2)] = cov_u
        S[n1:, n1:] = S22
        cross = np.zeros((n1, n2))
        cross[np.ix_(m1, m2)] = a * C1M
        S[:n1, n1:] = cross
        S[n1:, :n1] = cross.T
    elif world == 'W3':
        S[n1:, n1:] = C2
    elif world == 'W4':
        S[n1:, n1:] = C2
        cross = np.zeros((n1, n2))
        for k in range(nm):
            cross[m1[k], m2[k]] = LAM_COEF * np.sqrt(C1[m1[k], m1[k]] * C2[m2[k], m2[k]])
        S[:n1, n1:] = cross
        S[n1:, :n1] = cross.T
    return S

S_true = {w: world_cov(w) for w in ('W1', 'W2', 'W3', 'W4')}
w4_eigs = np.linalg.eigvalsh(0.5 * (S_true['W4'] + S_true['W4'].T))
print(f"W4 joint-cov min eigenvalue = {w4_eigs.min():.6e} (PSD: {w4_eigs.min() > 0})")
assert w4_eigs.min() > 0, "W4 joint covariance not PSD; reduce LAM_COEF"

# ------------------------------------------------ closed-form Gaussian means
LOG10E = np.log10(np.e)

def cf_mixture_log10mean(B, const, S):
    """log10 E[mixture] under x ~ N(0, S): E = mean_g exp(0.5 B_g'S B_g - const_g/2)."""
    q = np.einsum('gi,ij,gj->g', B, S, B)
    lg = 0.5 * q - 0.5 * const
    mm = lg.max()
    return float((mm + np.log(np.mean(np.exp(lg - mm)))) * LOG10E)

# Build full-space coefficient matrices for each statistic once.
B_M1 = np.zeros((K1['A'].shape[0], n1 + n2)); B_M1[:, :n1] = K1['A']
B_M2 = np.zeros((K2['A'].shape[0], n1 + n2)); B_M2[:, n1:] = K2['A']
B_J = []
for a, kern in zip(ALPHAS, kerns):
    G = kern['A1'].shape[0]
    B = np.zeros((G, n1 + n2))
    B[:, :n1] = kern['A1']
    B[:, m1] -= a * kern['Ay']
    B[:, n1 + m2] = kern['Ay']
    B[:, n1 + u2] = kern['Au']
    B_J.append(B)

def cf_prod_log10mean(S):
    """log10 E[M1*M2] = log10 mean_{g,h} exp(A1_g' S12 A2_h) (marginal terms cancel)."""
    U = K1['A'] @ S[:n1, n1:] @ K2['A'].T
    mm = U.max()
    return float((mm + np.log(np.mean(np.exp(U - mm)))) * LOG10E)

closed_form = {}  # log10 of the exact Gaussian expectation of each statistic
for w, S in S_true.items():
    l10_m1 = cf_mixture_log10mean(B_M1, K1['const'], S)
    l10_m2 = cf_mixture_log10mean(B_M2, K2['const'], S)
    closed_form[w] = dict(
        M1=l10_m1, M2=l10_m2,
        AVG=float(np.log10(0.5 * (10 ** l10_m1 + 10 ** l10_m2))),
        JOINT13=cf_mixture_log10mean(B_J[A13], kerns[A13]['const'], S),
        PROD=cf_prod_log10mean(S),
        E_Mjoint_by_alpha={f"{a:.3f}": cf_mixture_log10mean(B, k['const'], S)
                           for a, B, k in zip(ALPHAS, B_J, kerns)},
    )

# ---------------------------------------------------------------- samplers
L1 = K1['L_chol']
L2 = K2['L_chol']
L_u = np.linalg.cholesky(cov_u)
L_y23 = {a: np.linalg.cholesky(cov_y23s[ALPHAS.index(a)]) for a in (1.0 / 3.0, 0.25)}
L_W4 = np.linalg.cholesky(S_true['W4'] + 1e-12 * np.eye(n1 + n2))

def draw(world, rng, bs):
    if world in ('W1', 'W2'):
        a = 1.0 / 3.0 if world == 'W1' else 0.25
        e1 = L1 @ rng.standard_normal((n1, bs))
        e23 = L_y23[a] @ rng.standard_normal((nm, bs))
        eu = L_u @ rng.standard_normal((nu, bs))
        e2 = np.zeros((n2, bs))
        e2[m2] = a * e1[m1] + (1 - a) * e23
        e2[u2] = eu
        return e1, e2
    if world == 'W3':
        return (L1 @ rng.standard_normal((n1, bs)),
                L2 @ rng.standard_normal((n2, bs)))
    x = L_W4 @ rng.standard_normal((n1 + n2, bs))
    return x[:n1], x[n1:]

# ------------------------------------------------------------------ MC loop
def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)

STATS = ['M1', 'M2', 'AVG', 'PROD', 'JOINT13', 'FAMINF']
results = {}
for wi, world in enumerate(('W1', 'W2', 'W3', 'W4')):
    rng = np.random.default_rng(BASE_SEED + 17 * wi)
    vals = {s: np.empty(N_DRAWS) for s in STATS}
    logs = {'M1': np.empty(N_DRAWS), 'M2': np.empty(N_DRAWS)}
    t0 = time.time()
    done = 0
    while done < N_DRAWS:
        bs = min(BATCH, N_DRAWS - done)
        e1, e2 = draw(world, rng, bs)
        lm1 = mixture_log_e_from_residuals(e1, K1)
        lm2 = mixture_log_e_from_residuals(e2, K2)
        lJ = np.empty((len(ALPHAS), bs))
        for ai, (a, kern) in enumerate(zip(ALPHAS, kerns)):
            y = e2[m2] - a * e1[m1]
            lJ[ai] = joint_log_e(e1, y, e2[u2], kern)
        sl = slice(done, done + bs)
        logs['M1'][sl] = lm1
        logs['M2'][sl] = lm2
        vals['M1'][sl] = np.exp(np.minimum(lm1, 700))
        vals['M2'][sl] = np.exp(np.minimum(lm2, 700))
        vals['AVG'][sl] = 0.5 * (vals['M1'][sl] + vals['M2'][sl])
        vals['PROD'][sl] = np.exp(np.minimum(lm1 + lm2, 700))
        vals['JOINT13'][sl] = np.exp(np.minimum(lJ[A13], 700))
        vals['FAMINF'][sl] = np.exp(np.minimum(lJ.min(axis=0), 700))
        done += bs
    res = {}
    for s in STATS:
        k = int(np.sum(vals[s] >= THR))
        lo, hi = wilson(k, N_DRAWS)
        res[s] = dict(fpr=k / N_DRAWS, k=k, N=N_DRAWS, wilson95=(float(lo), float(hi)),
                      mean=float(vals[s].mean()), median=float(np.median(vals[s])),
                      q99=float(np.quantile(vals[s], 0.99)), max=float(vals[s].max()))
    res['corr_logM1_logM2'] = float(np.corrcoef(logs['M1'], logs['M2'])[0, 1])
    results[world] = res
    print(f"\n[{world}]  ({time.time()-t0:.1f}s, N={N_DRAWS})  "
          f"corr(logM1,logM2)={res['corr_logM1_logM2']:+.3f}")
    for s in STATS:
        r = res[s]
        cf = closed_form[world].get(s)
        cfs = f"  closed-form log10 E={cf:9.3f}" if cf is not None else " " * 26
        print(f"  {s:8s} FPR(>=20)={r['fpr']:.5f} [{r['wilson95'][0]:.5f},{r['wilson95'][1]:.5f}] "
              f"k={r['k']:5d}  MC mean={r['mean']:8.4f}{cfs}  max={r['max']:.3g}")

out = dict(observed=dict(M1=M1_obs, M2=M2_obs, AVG=(M1_obs + M2_obs) / 2,
                         PROD=M1_obs * M2_obs, JOINT13=J13_obs, FAMINF=FAM_obs,
                         Mjoint_by_alpha={f"{a:.3f}": float(np.exp(l))
                                          for a, l in zip(ALPHAS, logJ_obs)}),
           w4_min_eig=float(w4_eigs.min()), lam_coef=LAM_COEF,
           n_draws=N_DRAWS, threshold=THR, alphas=[float(a) for a in ALPHAS],
           closed_form_log10_means=closed_form, mc=results)
SIM_OUT.mkdir(parents=True, exist_ok=True)
outp = SIM_OUT / 'fpr_cross_release_worlds_results.json'
outp.write_text(json.dumps(out, indent=2))
print(f"\nwrote {outp}")
print("\nClosed-form log10 E[Mjoint(a)] by world (family-inf context):")
for w in ('W1', 'W2', 'W3', 'W4'):
    print(f"  {w}: " + ", ".join(f"a={k}:{v:+.2f}"
          for k, v in closed_form[w]['E_Mjoint_by_alpha'].items()))
