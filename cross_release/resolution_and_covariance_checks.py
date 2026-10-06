#!/usr/bin/env python3
"""Grid resolution of the quintessence priors, the released DR1 covariance, and the DR1-side
calibration tolerance.

  1. The quintessence priors of Section 4.3: the freezing box w0 in [-0.95, -0.75], wa in [0, 0.3]
     is small and its integrand peaks near one corner, so its uniform mixture depends on the grid.
     The box and the Caldwell-Linder thawing band are evaluated at several grid resolutions with
     the kernels of code/literature_priors.py (quintessence_resolution.py continues to
     160 x 160 and 320 x 320, where the freezing value is 14.8).
  2. DESI DR1 Table 1 (arXiv:2404.03002) versus the released DR1 CobayaSampler covariance: the same
     comparison as for DR2 Table IV in Section 3.
  3. DR1-side calibration tolerance of the alpha = 1/3 combined statistic (the DR2-side value is
     0.016 sigma): the coherent DR1 offset at which the statistic divided by its largest null mean
     falls to 20.

Writes results/numbers.json["resolution_and_covariance_checks"] and
results/resolution_and_covariance_checks.log.
Run (from the repository root, about 30 s): python cross_release/resolution_and_covariance_checks.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import optimize
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import ROOT, OUT_DIR, NUMBERS, add_code_path, load_numbers, save_numbers  # noqa: E402
add_code_path()
REPO = ROOT
OUT = NUMBERS
THIS = 'cross_release/resolution_and_covariance_checks.py'

from data_loader import load_desi_data                                        # noqa: E402
from evalue_analysis import precompute_kernels, mixture_log_e_from_residuals  # noqa: E402
from eprocess_hierarchical_mc import build_bin_matching                       # noqa: E402
import eprocess_joint as EJ                                                   # noqa: E402
from literature_priors import mixture_log_e_grid, flat_box_grid        # noqa: E402

t_start = time.time()
RES, LOG = {}, []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    LOG.append(s)


def rec(key, value, definition):
    RES[key] = dict(value=value, definition=definition, script=THIS)


def py(x):
    if isinstance(x, dict):
        return {str(k): py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [py(v) for v in x]
    if isinstance(x, np.ndarray):
        return py(x.tolist())
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


ds1 = load_desi_data(REPO / 'data' / 'dr1', 'DR1')
ds2 = load_desi_data(REPO / 'data' / 'dr2', 'DR2')

# ------------------------------------------------------------------ 1. quintessence priors
say("=== 1. quintessence priors (Section 4.3) ===")
quint = {}
for nres in (20, 40, 80):
    g = flat_box_grid(-0.95, -0.75, 0.0, 0.3, n_w0=nres, n_wa=nres)
    M2 = float(np.exp(mixture_log_e_grid(g, ds2)))
    M1 = float(np.exp(mixture_log_e_grid(g, ds1)))
    quint[f'freezing_box_{nres}x{nres}'] = dict(M_DR1=M1, M_DR2=M2)
    say(f"  freezing box [-0.95,-0.75]x[0,0.3], {nres}x{nres}: M_DR1 = {M1:.3f}, M_DR2 = {M2:.3f}")
for (nw, na) in ((40, 25), (80, 50)):
    pts = []
    for w0 in np.linspace(-1.0, -0.85, nw):
        lo, hi = -3 * (1 + w0), -(1 + w0)
        for wa in np.linspace(lo, hi, na):
            pts.append((w0, wa))
    g = np.array(pts)
    M2 = float(np.exp(mixture_log_e_grid(g, ds2)))
    M1 = float(np.exp(mixture_log_e_grid(g, ds1)))
    quint[f'thawing_CL_band_{nw}x{na}'] = dict(M_DR1=M1, M_DR2=M2)
    say(f"  thawing Caldwell-Linder band, {nw}x{na}: M_DR1 = {M1:.3f}, M_DR2 = {M2:.1f}")
# thawing band with several lower w0 edges
for w0lo in (-1.0, -0.98, -0.95, -0.9):
    pts = []
    for w0 in np.linspace(w0lo, -0.85, 40):
        lo, hi = -3 * (1 + w0), -(1 + w0)
        for wa in np.linspace(lo, hi, 25):
            pts.append((w0, wa))
    M2 = float(np.exp(mixture_log_e_grid(np.array(pts), ds2)))
    quint[f'thawing_w0_from_{w0lo}'] = M2
    say(f"  thawing band with w0 from {w0lo}: M_DR2 = {M2:.1f}")
rec('quintessence_priors', quint,
    'Section 4.3 priors evaluated with code/literature_priors.py (flat_box_grid / mixture_log_e_grid, fixed Planck background, Default DR1/DR2 likelihoods) at several grid resolutions. The freezing-box value falls with resolution, from about 21.5 at 20 x 20 to 14.8 at 320 x 320, the grid of Section 4.3 (quintessence_resolution continues to 160 x 160 and 320 x 320). The thawing values are checked at two resolutions and for several lower w0 edges.')

# ------------------------------------------------------------------ 2. DR1 Table 1 versus the released DR1 covariance
say("\n=== 2. DESI DR1 Table 1 versus the released DR1 covariance ===")
T1 = {0.295: ('DV', 0.15), 0.51: (0.25, 0.61, -0.445), 0.706: (0.32, 0.60, -0.420), 0.93: (0.28, 0.35, -0.389),
      1.317: (0.69, 0.42, -0.444), 1.491: ('DV', 0.67), 2.33: (0.94, 0.17, -0.477)}
QN = {'DV_over_rs': 'DV', 'DM_over_rs': 'DM', 'DH_over_rs': 'DH'}
TR = {0.295: 'BGS', 0.51: 'LRG1', 0.706: 'LRG2', 0.93: 'LRG3+ELG1', 1.317: 'ELG2', 1.491: 'QSO', 2.33: 'Lya'}
zq = [(round(float(z), 3), QN[q]) for z, q in zip(ds1.z_eff, ds1.quantities)]
sig_file = np.sqrt(np.diag(ds1.cov))
sig_t1 = np.array([T1[z][1] if q == 'DV' else (T1[z][0] if q == 'DM' else T1[z][1]) for z, q in zq])
corr_file = ds1.cov / np.outer(sig_file, sig_file)
ratio = {f'{TR[z]} {q}': float(sig_file[i] / sig_t1[i] - 1) for i, (z, q) in enumerate(zq)}
rfile = {TR[z]: float(corr_file[i, i + 1]) for i, (z, q) in enumerate(zq) if i + 1 < len(zq) and zq[i + 1][0] == z}
rt1 = {TR[z]: T1[z][2] for z in T1 if T1[z][0] != 'DV'}
for k, v in ratio.items():
    say(f"  DR1 sigma_file/sigma_Table1 - 1: {k:14s} {100 * v:6.2f}%   (Table 1 errors are printed to 2 decimals)")
say("  DR1 r(DM,DH) file:", {k: round(v, 3) for k, v in rfile.items()}, " Table 1:", rt1)
CT = np.zeros_like(ds1.cov)
for i, (z, q) in enumerate(zq):
    CT[i, i] = sig_t1[i] ** 2
for i, (z, q) in enumerate(zq):
    for j, (z2, q2) in enumerate(zq):
        if i != j and z == z2:
            CT[i, j] = T1[z][2] * sig_t1[i] * sig_t1[j]
KT = precompute_kernels(ds1.z_eff, ds1.quantities, CT, EJ.W0_GRID, EJ.WA_GRID)
M1_T1 = float(np.exp(mixture_log_e_from_residuals((ds1.data - KT['mu_null'])[:, None], KT)[0]))
K1 = precompute_kernels(ds1.z_eff, ds1.quantities, ds1.cov, EJ.W0_GRID, EJ.WA_GRID)
M1_file = float(np.exp(mixture_log_e_from_residuals((ds1.data - K1['mu_null'])[:, None], K1)[0]))
say(f"  M_DR1 released covariance = {M1_file:.4f}; with Table 1 errors/correlations = {M1_T1:.4f}")
rec('dr1_table1_vs_released', dict(sigma_ratio_minus_1=ratio, r_file=rfile, r_table1=rt1, M_DR1_released=M1_file, M_DR1_table1=M1_T1,
                                   note='Table 1 of arXiv:2404.03002 prints errors to two decimals, so differences below about 2% are within the printed precision'),
    'DESI DR1 BAO Table 1 (errors and D_M-D_H correlations as printed) against the released DR1 CobayaSampler covariance, and the Default-prior M_DR1 under each.')

# ------------------------------------------------------------------ 3. DR1-side tolerance
say("\n=== 3. DR1-side calibration tolerance of the alpha=1/3 combined statistic ===")
K2 = precompute_kernels(ds2.z_eff, ds2.quantities, ds2.cov, EJ.W0_GRID, EJ.WA_GRID)
_, D1 = EJ.build_delta_matrix(ds1.z_eff, ds1.quantities, EJ.W0_GRID, EJ.WA_GRID)
_, D2 = EJ.build_delta_matrix(ds2.z_eff, ds2.quantities, EJ.W0_GRID, EJ.WA_GRID)
dr2_to_dr1, _ = build_bin_matching(ds1, ds2)
pairs = [(i, j) for j, i in enumerate(dr2_to_dr1) if i >= 0]
m1 = np.array([i for i, _ in pairs]); m2 = np.array([j for _, j in pairs])
u2 = np.array([j for j in range(len(ds2.data)) if dr2_to_dr1[j] < 0])
C1M = ds1.cov[np.ix_(m1, m1)]; C2M = ds2.cov[np.ix_(m2, m2)]
a = 1 / 3
Sy = C2M - a * a * C1M
Qy = np.linalg.inv(Sy)
C1inv = np.linalg.inv(ds1.cov)
A1 = D1 @ C1inv
c1 = np.einsum('gi,ij,gj->g', D1, C1inv, D1)
Dm = D2[:, m2] - a * D1[:, m1]
Ay = Dm @ Qy
cy = np.einsum('gi,ij,gj->g', Dm, Qy, Dm)
covu = ds2.cov[np.ix_(u2, u2)]
Qu = np.linalg.inv(covu)
Du = D2[:, u2]
Au = Du @ Qu
cu = np.einsum('gi,ij,gj->g', Du, Qu, Du)
e1 = ds1.data - K1['mu_null']; e2 = ds2.data - K2['mu_null']
y = e2[m2] - a * e1[m1]
L = A1 @ e1 - 0.5 * c1 + Ay @ y - 0.5 * cy + Au @ e2[u2] - 0.5 * cu
G = len(L)
logM13 = float(logsumexp(L) - np.log(G))
s1 = np.sqrt(np.diag(C1M)); s2 = np.sqrt(np.diag(C2M))
c_dr1 = A1[:, m1] @ s1 - a * (Ay @ s1)
c_dr2 = Ay @ s2


def deflated(b, c):
    return logM13 - (max(logsumexp(b * c), logsumexp(-b * c)) - np.log(G))


b_dr2 = optimize.brentq(lambda b: deflated(b, c_dr2) - np.log(20), 1e-5, 0.1)
b_dr1 = optimize.brentq(lambda b: deflated(b, c_dr1) - np.log(20), 1e-3, 10)
say(f"  M_joint(1/3) = {np.exp(logM13):.4f}; DR2-side tolerance b* = {b_dr2:.5f} sigma_DR2 (paper 0.0160); DR1-side tolerance b* = {b_dr1:.3f} sigma_DR1")
say(f"  ratio of DR1-side to DR2-side tolerance = {b_dr1 / b_dr2:.1f}")
rec('dr1_side_tolerance', dict(M_joint_1_3=float(np.exp(logM13)), b_star_DR2_side=b_dr2, b_star_DR1_side=b_dr1, ratio=b_dr1 / b_dr2,
                               deflated_DR1_side_at={f'{s}': float(np.exp(deflated(s, c_dr1))) for s in (0.5, 1.0, 1.5, 2.0)}),
    'Coherent-offset calibration tolerance of the alpha=1/3 combined statistic when the offset is on the matched DR1 measurements (b sigma_DR1 each, same sign) instead of DR2: the offset at which the statistic divided by its largest null mean (either sign) falls to 20. Because a DR1 offset enters the DR1 factor and the innovation y = X2 - X1/3 with opposite signs, the coefficients nearly cancel and the DR1-side tolerance is about a hundred times the DR2-side value.')

RES['meta'] = dict(value=dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), runtime_seconds=round(time.time() - t_start, 1)),
                   definition='(meta)', script=THIS)
ALL = load_numbers()
qr = ALL.get('resolution_and_covariance_checks', {}).get('quintessence_resolution')   # written by quintessence_resolution.py
ALL['resolution_and_covariance_checks'] = py(RES)
if qr is not None:
    ALL['resolution_and_covariance_checks']['quintessence_resolution'] = qr
save_numbers(ALL)
(OUT_DIR / 'resolution_and_covariance_checks.log').write_text('\n'.join(LOG) + '\n')
say(f"\nwrote {OUT} [resolution_and_covariance_checks]  ({time.time() - t_start:.0f}s)")
