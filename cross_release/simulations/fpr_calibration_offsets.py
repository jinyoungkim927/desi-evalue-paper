#!/usr/bin/env python3
"""False-positive rate of the combined statistics under relative calibration offsets.

LCDM-true paired DR1/DR2 draws under the alpha = 1/3 law (N = 4000 per cell); every draw is
scored with and without an offset delta = s * sigma_DR2 * v on the 11 matched quantities,
placed on DR2 or on DR1, for:
  - adversarial unit directions: the exact E-inflation maximizer (EINF), directions that
    maximise the false-positive rate itself (pilot-MC mean-log gradient ascent at s_ref = 0.02
    and 0.5: FPR02, FPR50), and an MLE-displacement mimic (MLE);
  - 20 random unit directions;
  - coherent per-bin patterns (UNIF: every matched bin +s*sigma; SIGN: every bin s*sigma with
    the adversarial sign pattern), which have sigma-norm s*sqrt(11), i.e. each measurement
    offset by s of its own sigma;
  - magnitudes s in {0.01, 0.02, 0.05, 0.10, 0.25} and, to locate the 5% crossing, the
    extension {0.5, 1.0, 2.0} (marked EXT in the log).

ADVbest = max FPR over the unit-norm candidates {EINF, FPR02, FPR50, MLE} at each (allocation, s).

The output of record of this script is cross_release/inputs/fpr_calibration_offsets_results.json (JSON) and cross_release/inputs/logs/fpr_calibration_offsets.log (printed log), read by
cross_release/one_fraction_family.py. Running the script again writes a fresh copy to
results/simulations/ and prints the log to stdout.
Run from the repository root: python cross_release/simulations/fpr_calibration_offsets.py
"""
import sys, json, time
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
                            joint_log_e, decomposition_for_alpha, DR2_MLE)

t0 = time.time()
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
n1, n2 = len(ds1.data), len(ds2.data)
u2_idx = np.array([j for j in range(n2) if dr2_to_dr1[j] < 0], dtype=int)
n_match, n_u = len(pairs), len(u2_idx)
C1M = ds1.cov[np.ix_(m1_idx, m1_idx)]
C2M = ds2.cov[np.ix_(m2_idx, m2_idx)]
sig2m = np.sqrt(np.diag(C2M))

A_TRUE = 1.0 / 3.0
ALPHA_FAM = [0.05, 0.10, 0.15, 0.20, 0.25, 1.0/3.0, 0.40]
I13 = ALPHA_FAM.index(1.0/3.0)

def build_kernel(a):
    cov_y23, min_eig, psd = decomposition_for_alpha(C1M, C2M, a)
    assert psd, f"alpha={a} inadmissible"
    cov_y23 = cov_y23 + 1e-14 * np.eye(n_match)
    Sy = (1.0 - a) ** 2 * cov_y23
    cov_u, _ = nearest_psd(ds2.cov[np.ix_(u2_idx, u2_idx)])
    cov_u = cov_u + 1e-14 * np.eye(n_u)
    kern = joint_kernel(delta1, delta2, m1_idx, m2_idx, u2_idx,
                        ds1.cov, Sy, cov_u, a)
    return dict(kern=kern, cov_y23=cov_y23, cov_u=cov_u, alpha=a)

KERNS = [build_kernel(a) for a in ALPHA_FAM]
k13 = KERNS[I13]

eps1_obs = ds1.data - K1['mu_null']
eps2_obs = ds2.data - K2['mu_null']
M1_obs = float(np.exp(mixture_log_e_from_residuals(eps1_obs[:, None], K1)[0]))
M2_obs = float(np.exp(mixture_log_e_from_residuals(eps2_obs[:, None], K2)[0]))
y_obs = eps2_obs[m2_idx] - A_TRUE * eps1_obs[m1_idx]
Mj_obs = float(np.exp(joint_log_e(eps1_obs[:, None], y_obs[:, None],
                                  eps2_obs[u2_idx][:, None], k13['kern'])[0]))
print(f"SANITY: M1={M1_obs:.2f} M2={M2_obs:.2f} Mjoint13={Mj_obs:.2f}")

def coef_matrix(kobj, alloc):
    kern, a = kobj['kern'], kobj['alpha']
    if alloc == 'DR2':
        return kern['Ay']
    return kern['A1'][:, m1_idx] - a * kern['Ay']

def joint_offset(kobj, delta, alloc):
    return coef_matrix(kobj, alloc) @ delta

# ---------------- adversarial candidates ----------------
def einf_direction(B, s_ref=0.02, n_iter=400, n_restarts=8, seed=7):
    rng = np.random.default_rng(seed)
    inits = [B[np.argmax(np.linalg.norm(B, axis=1))], B.mean(axis=0)]
    inits += [rng.standard_normal(n_match) for _ in range(n_restarts)]
    best_v, best_f = None, -np.inf
    for v0 in inits:
        for sgn in (+1.0, -1.0):
            v = sgn * v0 / np.linalg.norm(v0)
            for _ in range(n_iter):
                z = s_ref * (B @ v)
                w = np.exp(z - z.max()); w /= w.sum()
                v_new = B.T @ w; nv = np.linalg.norm(v_new)
                if nv < 1e-300: break
                v_new /= nv
                if np.linalg.norm(v_new - v) < 1e-12: v = v_new; break
                v = v_new
            z = s_ref * (B @ v)
            f = z.max() + np.log(np.mean(np.exp(z - z.max())))
            if f > best_f: best_f, best_v = f, v.copy()
    return best_v

# pilot draws for FPR-targeted ascent
rng_p = np.random.default_rng(99)
L1 = np.linalg.cholesky(ds1.cov)
L23 = np.linalg.cholesky(k13['cov_y23'])
L_u = np.linalg.cholesky(k13['cov_u'])
NP_ = 800
e1p = L1 @ rng_p.standard_normal((n1, NP_))
e23p = L23 @ rng_p.standard_normal((n_match, NP_))
eup = L_u @ rng_p.standard_normal((n_u, NP_))
eps2Mp = A_TRUE * e1p[m1_idx] + (1 - A_TRUE) * e23p
kern13 = k13['kern']
y13p = eps2Mp - A_TRUE * e1p[m1_idx]
base13p = (kern13['A1'] @ e1p + kern13['Ay'] @ y13p + kern13['Au'] @ eup
           - 0.5 * kern13['const'][:, None])          # G x NP

def fpr_direction(B, s_ref, n_iter=150, seed=11):
    """Maximize mean over pilot draws of logmeanexp_g(base + s B v), ||v||=1."""
    rng = np.random.default_rng(seed)
    inits = [einf_direction(B), B.mean(axis=0)]
    inits += [rng.standard_normal(n_match) for _ in range(3)]
    best_v, best_f = None, -np.inf

    def obj_and_grad(v):
        X = base13p + s_ref * (B @ v)[:, None]
        m = X.max(axis=0)
        W = np.exp(X - m); Z = W.sum(axis=0)
        f = float(np.mean(m + np.log(Z / X.shape[0])))
        w = W / Z                                     # G x NP softmax per draw
        g = s_ref * (B.T @ w.mean(axis=1))
        return f, g

    for v0 in inits:
        for sgn in (+1.0, -1.0):
            v = sgn * v0 / np.linalg.norm(v0)
            for _ in range(n_iter):
                f, g = obj_and_grad(v)
                gn = np.linalg.norm(g)
                if gn < 1e-14: break
                v_new = v + 0.5 * g / gn
                v_new /= np.linalg.norm(v_new)
                if np.linalg.norm(v_new - v) < 1e-10: v = v_new; break
                v = v_new
            f, _ = obj_and_grad(v)
            if f > best_f: best_f, best_v = f, v.copy()
    return best_v

# MLE-displacement mimic: bias mimics Dm at the grid point nearest the DR2 MLE
gg = int(np.argmin([(w0 - DR2_MLE[0])**2 + (wa - DR2_MLE[1])**2
                    for w0 in W0_GRID for wa in WA_GRID]))
Dm_g = delta2[gg, m2_idx] - A_TRUE * delta1[gg, m1_idx]

ADV = {}
for alloc in ('DR2', 'DR1'):
    B = coef_matrix(k13, alloc) * sig2m[None, :]
    cands = dict(
        EINF=einf_direction(B),
        FPR02=fpr_direction(B, 0.02),
        FPR50=fpr_direction(B, 0.5),
        MLE=(Dm_g / sig2m) / np.linalg.norm(Dm_g / sig2m),
    )
    if alloc == 'DR1':
        # for DR1 allocation the innovation picks up -a*delta: flip mimic sign
        cands['MLE'] = -cands['MLE']
    ADV[alloc] = cands
    for k, v in cands.items():
        z = 0.02 * (B @ v)
        print(f"{alloc} {k}: logE-infl(s=0.02)={z.max()+np.log(np.mean(np.exp(z-z.max()))):.3f}")

rng_dir = np.random.default_rng(20260831)
N_DIR = 20
V_rand = rng_dir.standard_normal((N_DIR, n_match))
V_rand /= np.linalg.norm(V_rand, axis=1, keepdims=True)

S_GRID = [0.01, 0.02, 0.05, 0.10, 0.25]
S_EXT = [0.5, 1.0, 2.0]
S_ALL = S_GRID + S_EXT

def lme(base, offset):
    X = base + offset[:, None]
    m = X.max(axis=0)
    return m + np.log(np.mean(np.exp(X - m), axis=0))

# build cells
cells = []
for alloc in ('DR2', 'DR1'):
    dirs = [(k, v, 'unit') for k, v in ADV[alloc].items()]
    dirs += [(f"R{k}", V_rand[k], 'unit') for k in range(N_DIR)]
    dirs += [('UNIF', np.ones(n_match), 'perbin'),
             ('SIGN', np.sign(ADV[alloc]['FPR50']), 'perbin')]
    for dkey, v, norm in dirs:
        for s in S_ALL:
            delta = s * sig2m * v          # perbin rows: |delta_i| = s*sigma_i
            cells.append(dict(
                alloc=alloc, dkey=dkey, s=s, norm=norm,
                offs_joint=[joint_offset(K, delta, alloc) for K in KERNS],
                off_k1=(K1['A'][:, m1_idx] @ delta) if alloc == 'DR1' else None,
                off_k2=(K2['A'][:, m2_idx] @ delta) if alloc == 'DR2' else None,
                cnt=np.zeros(5, dtype=int)))
base_cnt = np.zeros(5, dtype=int)

N_MC, BATCH = 4000, 500
LOG_THR = np.log(20.0)
rng = np.random.default_rng(20260831)
done = 0
while done < N_MC:
    bs = min(BATCH, N_MC - done)
    e1 = L1 @ rng.standard_normal((n1, bs))
    e23 = L23 @ rng.standard_normal((n_match, bs))
    eu = L_u @ rng.standard_normal((n_u, bs))
    eps2M = A_TRUE * e1[m1_idx] + (1 - A_TRUE) * e23
    eps2 = np.zeros((n2, bs)); eps2[m2_idx] = eps2M; eps2[u2_idx] = eu

    base_joint = []
    for K in KERNS:
        a, kern = K['alpha'], K['kern']
        y_a = eps2M - a * e1[m1_idx]
        base_joint.append(kern['A1'] @ e1 + kern['Ay'] @ y_a + kern['Au'] @ eu
                          - 0.5 * kern['const'][:, None])
    base_k1 = K1['A'] @ e1 - 0.5 * K1['const'][:, None]
    base_k2 = K2['A'] @ eps2 - 0.5 * K2['const'][:, None]

    logM1_0 = lme(base_k1, np.zeros(base_k1.shape[0]))
    logM2_0 = lme(base_k2, np.zeros(base_k2.shape[0]))
    logJ_0 = np.stack([lme(bj, np.zeros(bj.shape[0])) for bj in base_joint])

    def counts(logJ, logM1, logM2):
        avg = 0.5 * (np.exp(np.minimum(logM1, 700)) + np.exp(np.minimum(logM2, 700)))
        return np.array([(logJ[I13] >= LOG_THR).sum(),
                         (logJ.min(axis=0) >= LOG_THR).sum(),
                         (avg >= 20.0).sum(),
                         (logM2 >= LOG_THR).sum(),
                         (logM1 >= LOG_THR).sum()], dtype=int)

    base_cnt += counts(logJ_0, logM1_0, logM2_0)
    for cell in cells:
        logJ = np.stack([lme(bj, off) for bj, off in
                         zip(base_joint, cell['offs_joint'])])
        lM1 = lme(base_k1, cell['off_k1']) if cell['off_k1'] is not None else logM1_0
        lM2 = lme(base_k2, cell['off_k2']) if cell['off_k2'] is not None else logM2_0
        cell['cnt'] += counts(logJ, lM1, lM2)
    done += bs
    print(f"  {done}/{N_MC}  [{time.time()-t0:.0f}s]", flush=True)

# ---------------- report ----------------
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return c - h, c + h

STATS = ['JOINT13', 'FAMINF', 'AVG', 'M2', 'M1']
ADV_KEYS = ['EINF', 'FPR02', 'FPR50', 'MLE']
out = dict(alpha_family=ALPHA_FAM, N=N_MC,
           sanity=dict(M1=M1_obs, M2=M2_obs, Mjoint13=Mj_obs),
           adv_dirs={a: {k: v.tolist() for k, v in ADV[a].items()} for a in ADV},
           baseline={st: int(base_cnt[i]) for i, st in enumerate(STATS)},
           grid={})

def get(alloc, dkey, s):
    return next(c for c in cells if c['alloc']==alloc and c['dkey']==dkey and c['s']==s)

print(f"\nBASELINE s=0: " + "  ".join(
    f"{st}={base_cnt[i]}/{N_MC}={base_cnt[i]/N_MC:.4f}" for i, st in enumerate(STATS)))
for alloc in ('DR2', 'DR1'):
    print(f"\n================ allocation: bias in {alloc} ================")
    hdr = f"{'s':>5} {'row':>10} | " + " ".join(f"{st:>9}" for st in STATS)
    print(hdr)
    for s in S_ALL:
        tag = " (EXT)" if s in S_EXT else ""
        # adversarial: max FPR over unit-norm candidates, per statistic
        advmat = np.stack([get(alloc, k, s)['cnt'] for k in ADV_KEYS])
        adv_best = advmat.max(axis=0)
        which = [ADV_KEYS[i] for i in advmat[:, 0].argmax(keepdims=True)]
        rnd = np.stack([get(alloc, f"R{k}", s)['cnt'] for k in range(N_DIR)])
        rows = [("ADVbest"+tag, adv_best), ("rand_med", np.median(rnd, axis=0)),
                ("rand_max", rnd.max(axis=0)),
                ("UNIF*", get(alloc, 'UNIF', s)['cnt']),
                ("SIGN*", get(alloc, 'SIGN', s)['cnt'])]
        for name, cnt in rows:
            print(f"{s:>5.2f} {name:>10} | " +
                  " ".join(f"{c/N_MC:>9.4f}" for c in cnt))
        out['grid'][f"{alloc}|s={s}"] = dict(
            adv_best={st: float(adv_best[i])/N_MC for i, st in enumerate(STATS)},
            adv_each={k: {st: int(get(alloc,k,s)['cnt'][i])/N_MC
                          for i, st in enumerate(STATS)} for k in ADV_KEYS},
            rand_median={st: float(np.median(rnd[:, i]))/N_MC for i, st in enumerate(STATS)},
            rand_max={st: int(rnd[:, i].max())/N_MC for i, st in enumerate(STATS)},
            unif={st: int(get(alloc,'UNIF',s)['cnt'][i])/N_MC for i, st in enumerate(STATS)},
            sign={st: int(get(alloc,'SIGN',s)['cnt'][i])/N_MC for i, st in enumerate(STATS)})
print("\n(* UNIF/SIGN rows: every matched bin shifted by s of its own DR2 sigma —")
print("   sigma-norm s*sqrt(11) = 3.32s; unit-norm rows have total sigma-norm s.)")

SIM_OUT.mkdir(parents=True, exist_ok=True)
with open(SIM_OUT / 'fpr_calibration_offsets_results.json', 'w') as f:
    json.dump(out, f, indent=1)
print(f"\ntotal {time.time()-t0:.0f}s")
