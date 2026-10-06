#!/usr/bin/env python3
"""The two forecast statements of Section 4.4.

(1) Two new high-redshift bins at z = 1.7 and 2.5 barely move the e-value, because there LCDM
    and the DR2 MLE predict distances differing by less than 0.5 sigma at the projected
    precision: D_M/r_d and D_H/r_d predicted by LCDM and by the DR2 BAO-only MLE at z = 1.7 and
    2.5, divided by the projected DR3 errors used in code/dr3_power_analysis.py
    (sigma_{DM/rd} = 0.7, 1.5; sigma_{DH/rd} = 0.4, 0.4).
(2) Halving the LRG2 error sends the median e-value to about 1.7e4 under the DR2 MLE and leaves
    it near zero under LCDM: synthetic DR2-like vectors (13 quantities, published covariance,
    LRG2 block with its errors halved, i.e. its covariance rows and columns scaled by 1/2) drawn
    under the DR2 MLE and under LCDM; the Default-prior mixture e-value distribution (median,
    10th and 90th percentiles, P(M >= 20), P(M < 1)), with the unhalved covariance as the
    baseline. 20 000 draws per case, fixed seeds.

Both use the code in code/ at the fixed Planck background of Section 3 and the Default prior
grid, and write results/numbers.json["dr3_forecast"] and results/dr3_forecast.log.

Run (from the repository root, about 5 s): python cross_release/dr3_forecast.py
Reads results/dr3_forecast.json (code/dr3_power_analysis.py) or its copy of record in
cross_release/inputs/results/.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import ROOT, OUT_DIR, NUMBERS, add_code_path, load_numbers, save_numbers, result_json  # noqa: E402
add_code_path()
REPO = ROOT
OUT = NUMBERS
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOGF = OUT_DIR / 'dr3_forecast.log'
THIS = 'cross_release/dr3_forecast.py'

from data_loader import load_desi_data                                        # noqa: E402
from cosmology import CosmologyParams, LCDM, compute_bao_predictions, log_likelihood  # noqa: E402
from evalue_analysis import (precompute_kernels, mixture_log_e_from_residuals,  # noqa: E402
                             _build_theory_vector, mixture_log_e)

t0 = time.time()
RES, LOG = {}, []


def say(s):
    print(s)
    LOG.append(s)


def py(o):
    if isinstance(o, dict):
        return {str(k): py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [py(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def rec(key, value, definition):
    RES[key] = dict(value=py(value), definition=definition, script=THIS)


W0_GRID = np.linspace(-1.5, -0.5, 30)   # Default flat prior on (w0, wa), as in code/
WA_GRID = np.linspace(-2.0, 1.0, 30)
DR3_NEW_BINS = [(1.7, 0.7, 0.4), (2.5, 1.5, 0.4)]   # (z, sigma_DM/rd, sigma_DH/rd) from code/dr3_power_analysis.py
N_MC = 20000
BATCH = 2000

ds = load_desi_data(REPO / 'data' / 'dr2', 'DR2')
bg = CosmologyParams()
say(f"fixed background: h = {bg.h}, Omega_m = {getattr(bg, 'omega_m', getattr(bg, 'Om', None))}, r_d = {bg.rd} Mpc")
logM_obs = mixture_log_e(ds.data, ds.cov, ds.z_eff, ds.quantities, W0_GRID, WA_GRID)[0]
say(f"observed M_DR2 (Default prior) = {np.exp(logM_obs):.4f}  (paper: 33.97)")


def fit_dr2_mle():
    def neg_ll(p):
        pred = compute_bao_predictions(ds.z_eff, CosmologyParams(w0=p[0], wa=p[1]))
        return -log_likelihood(ds.data, _build_theory_vector(pred, ds.z_eff, ds.quantities), ds.cov)
    return minimize(neg_ll, x0=[-0.86, -0.43], method='Nelder-Mead',
                    options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000}).x


mle = fit_dr2_mle()
say(f"DR2 BAO-only MLE at the fixed background = ({mle[0]:.4f}, {mle[1]:.4f})  (paper: (-0.856, -0.430))")
repo_forecast = result_json('dr3_forecast.json')
say(f"results/dr3_forecast.json w0wa_truth = {repo_forecast['w0wa_truth']}")

# ------------------------------------------------------------------ (1) new high-z bins
say("\n(1) LCDM vs DR2-MLE distances at the projected new DR3 bins, in projected-error units")
z_new = np.array([b[0] for b in DR3_NEW_BINS])
p_lcdm = compute_bao_predictions(z_new, LCDM)
p_mle = compute_bao_predictions(z_new, CosmologyParams(w0=mle[0], wa=mle[1]))
sep = {}
for i, (z, s_dm, s_dh) in enumerate(DR3_NEW_BINS):
    d_dm = (p_mle['DM_over_rd'][i] - p_lcdm['DM_over_rd'][i])
    d_dh = (p_mle['DH_over_rd'][i] - p_lcdm['DH_over_rd'][i])
    sep[f'z={z}'] = dict(DM_over_rd_LCDM=p_lcdm['DM_over_rd'][i], DM_over_rd_MLE=p_mle['DM_over_rd'][i],
                         DH_over_rd_LCDM=p_lcdm['DH_over_rd'][i], DH_over_rd_MLE=p_mle['DH_over_rd'][i],
                         sigma_DM=s_dm, sigma_DH=s_dh,
                         DM_separation_sigma=d_dm / s_dm, DH_separation_sigma=d_dh / s_dh,
                         DM_separation_percent=100 * d_dm / p_lcdm['DM_over_rd'][i],
                         DH_separation_percent=100 * d_dh / p_lcdm['DH_over_rd'][i])
    say(f"  z = {z}: dDM/rd = {d_dm:+.3f} ({d_dm / s_dm:+.2f} sigma, {100 * d_dm / p_lcdm['DM_over_rd'][i]:+.2f}%), "
        f"dDH/rd = {d_dh:+.3f} ({d_dh / s_dh:+.2f} sigma, {100 * d_dh / p_lcdm['DH_over_rd'][i]:+.2f}%)")
max_sep = max(abs(v['DM_separation_sigma']) for v in sep.values()) if True else None
max_sep = max(max_sep, max(abs(v['DH_separation_sigma']) for v in sep.values()))
say(f"  largest separation over the four new quantities: {max_sep:.2f} sigma (paper: less than 0.5 sigma)")
rec('new_bin_separation', dict(bins=sep, max_abs_separation_sigma=max_sep, mle=mle.tolist()),
    'Distances predicted by LCDM and by the DR2 BAO-only MLE (fixed Planck background) at the two '
    'projected DR3 high-z bins of code/dr3_power_analysis.py, divided by the projected errors '
    '(sigma_DM/rd = 0.7 and 1.5, sigma_DH/rd = 0.4 and 0.4 at z = 1.7 and 2.5). Section 4.4: the two new '
    'bins separate LCDM from the DR2 MLE by less than 0.5 sigma.')

# ------------------------------------------------------------------ (2) halving the LRG2 error
say("\n(2) Default-prior mixture e-value of synthetic DR2-like vectors, LRG2 errors halved vs published")
idx_lrg2 = [i for i, t in enumerate(ds.tracers) if t == 'LRG2']
say(f"  LRG2 indices in the 13-vector: {idx_lrg2} (quantities {[ds.quantities[i] for i in idx_lrg2]})")
S = np.ones(len(ds.data))
S[idx_lrg2] = 0.5
cov_half = (S[:, None] * ds.cov) * S[None, :]
truths = {'w0wa_DR2_MLE': (mle[0], mle[1]), 'LCDM': (-1.0, 0.0)}
covs = {'published_covariance': ds.cov, 'LRG2_errors_halved': cov_half}
halving = {}
seed = 20261005
for cname, cov in covs.items():
    K = precompute_kernels(ds.z_eff, ds.quantities, cov, W0_GRID, WA_GRID)
    for tname, (w0, wa) in truths.items():
        pred_truth = compute_bao_predictions(ds.z_eff, CosmologyParams(w0=w0, wa=wa))
        shift = _build_theory_vector(pred_truth, ds.z_eff, ds.quantities) - K['mu_null']
        rng = np.random.default_rng(seed)
        seed += 1
        logM = np.empty(N_MC)
        for b in range(0, N_MC, BATCH):
            eps = K['L_chol'] @ rng.standard_normal(size=(K['n'], min(BATCH, N_MC - b)))
            logM[b:b + eps.shape[1]] = mixture_log_e_from_residuals(eps + shift[:, None], K)
        M = np.exp(logM)
        out = dict(median=float(np.median(M)), p10=float(np.percentile(M, 10)), p90=float(np.percentile(M, 90)),
                   P_M_ge_20=float((M >= 20).mean()), P_M_lt_1=float((M < 1).mean()),
                   mean_log10_M=float(np.mean(logM) / np.log(10)), n_draws=N_MC)
        halving[f'{cname}|{tname}'] = out
        say(f"  {cname:22s} truth {tname:12s}: median M = {out['median']:9.3f}, 10-90% = [{out['p10']:.3f}, {out['p90']:.2f}], "
            f"P(M >= 20) = {out['P_M_ge_20']:.3f}, P(M < 1) = {out['P_M_lt_1']:.3f}")
rec('lrg2_halving', dict(results=halving, lrg2_indices=idx_lrg2, mle=mle.tolist(), n_draws=N_MC,
                         design='DR2 13-vector, published covariance; "LRG2 errors halved" scales the LRG2 rows and '
                                'columns of the covariance by 1/2 (variances by 1/4); data drawn around the truth '
                                'prediction at the fixed Planck background; Default 30x30 prior grid'),
    'Monte Carlo of the Default-prior mixture e-value for DR2-like data with the LRG2 errors halved, under the '
    'DR2 BAO-only MLE and under LCDM, against the published covariance as baseline. Section 4.4: halving the '
    'LRG2 error sends the median e-value to about 1.7e4 if w0waCDM is true and leaves it near zero if LCDM is true.')

RES['meta'] = dict(value=dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), runtime_seconds=round(time.time() - t0, 1)),
                   definition='(meta)', script=THIS)
ALL = load_numbers()
ALL['dr3_forecast'] = RES
save_numbers(ALL)
LOGF.write_text('\n'.join(LOG) + '\n')
say(f"\nwrote {OUT} [dr3_forecast] and {LOGF}  ({time.time() - t0:.0f}s)")
