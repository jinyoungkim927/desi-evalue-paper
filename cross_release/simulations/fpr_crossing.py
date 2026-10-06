#!/usr/bin/env python3
"""The 5% false-positive crossing of M_joint(1/3) under DR2-side offsets (N = 4000 per cell):
coherent per-bin offsets (UNIF) and offsets along the FPR50 direction of
fpr_calibration_offsets.py (ADVunit, total sigma-norm s).

The output of record of this script is cross_release/inputs/logs/fpr_crossing.log (printed log), read by
cross_release/one_fraction_family.py. Running the script again writes a fresh copy to
results/simulations/ and prints the log to stdout.
Run from the repository root: python cross_release/simulations/fpr_crossing.py
"""
import sys, json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, OUT_DIR, INPUTS, add_code_path  # noqa: E402
add_code_path()
REPO = ROOT
SIM_OUT = OUT_DIR / 'simulations'   # results/simulations/
from data_loader import load_desi_data
from evalue_analysis import precompute_kernels
from eprocess_hierarchical_mc import build_bin_matching, nearest_psd
from eprocess_joint import (W0_GRID, WA_GRID, build_delta_matrix, joint_kernel,
                            decomposition_for_alpha)

ds1 = load_desi_data(REPO / 'data' / 'dr1', 'DR1')
ds2 = load_desi_data(REPO / 'data' / 'dr2', 'DR2')
K1 = precompute_kernels(ds1.z_eff, ds1.quantities, ds1.cov, W0_GRID, WA_GRID)
K2 = precompute_kernels(ds2.z_eff, ds2.quantities, ds2.cov, W0_GRID, WA_GRID)
_, delta1 = build_delta_matrix(ds1.z_eff, ds1.quantities, W0_GRID, WA_GRID)
_, delta2 = build_delta_matrix(ds2.z_eff, ds2.quantities, W0_GRID, WA_GRID)
dr2_to_dr1, _ = build_bin_matching(ds1, ds2)
pairs = [(i, j) for j, i in enumerate(dr2_to_dr1) if i >= 0]
m1_idx = np.array([i for i, j in pairs]); m2_idx = np.array([j for i, j in pairs])
n1, n2 = len(ds1.data), len(ds2.data)
u2_idx = np.array([j for j in range(n2) if dr2_to_dr1[j] < 0])
n_match, n_u = len(pairs), len(u2_idx)
C1M = ds1.cov[np.ix_(m1_idx, m1_idx)]; C2M = ds2.cov[np.ix_(m2_idx, m2_idx)]
sig2m = np.sqrt(np.diag(C2M))
A = 1.0/3.0
cov_y23, _, psd = decomposition_for_alpha(C1M, C2M, A); assert psd
cov_y23 += 1e-14*np.eye(n_match)
Sy = (1-A)**2 * cov_y23
cov_u, _ = nearest_psd(ds2.cov[np.ix_(u2_idx, u2_idx)]); cov_u += 1e-14*np.eye(n_u)
kern = joint_kernel(delta1, delta2, m1_idx, m2_idx, u2_idx, ds1.cov, Sy, cov_u, A)

_p = SIM_OUT / 'fpr_calibration_offsets_results.json'   # fresh fpr_calibration_offsets.py output if present, else the copy of record
adv = json.load(open(_p if _p.exists() else INPUTS / 'fpr_calibration_offsets_results.json'))['adv_dirs']['DR2']['FPR50']
v_adv = np.array(adv)
v_unif = np.ones(n_match)

cells = [('UNIF', v_unif, s) for s in (0.55, 0.60, 0.65, 0.70, 0.80)] \
      + [('ADVunit', v_adv, s) for s in (1.25, 1.5, 1.75, 2.25, 2.5)]
offs = [kern['Ay'] @ (s * sig2m * v) for name, v, s in cells]

N_MC, BATCH = 4000, 500
LOG_THR = np.log(20.0)
rng = np.random.default_rng(20260831)
L1 = np.linalg.cholesky(ds1.cov); L23 = np.linalg.cholesky(cov_y23)
L_u = np.linalg.cholesky(cov_u)
cnt = np.zeros(len(cells), dtype=int)
done = 0
while done < N_MC:
    bs = min(BATCH, N_MC - done)
    e1 = L1 @ rng.standard_normal((n1, bs))
    e23 = L23 @ rng.standard_normal((n_match, bs))
    eu = L_u @ rng.standard_normal((n_u, bs))
    y = (1-A) * e23
    base = (kern['A1'] @ e1 + kern['Ay'] @ y + kern['Au'] @ eu
            - 0.5*kern['const'][:, None])
    for i, off in enumerate(offs):
        X = base + off[:, None]; m = X.max(axis=0)
        logM = m + np.log(np.mean(np.exp(X - m), axis=0))
        cnt[i] += int((logM >= LOG_THR).sum())
    done += bs
for (name, v, s), k in zip(cells, cnt):
    print(f"{name:>8} s={s:<5} JOINT13 FPR = {k}/{N_MC} = {k/N_MC:.4f}")
