#!/usr/bin/env python3
"""Calibration benchmark for the leave-one-out concentration statistic.

Section 4.2 reads the concentration of the LOO evidence in LRG2 (78.6% of
sum_k E_k^LOO) as pointing to a localised feature rather than a smooth w(z).
This script measures how concentrated the LOO evidence is when the signal is
smooth, by parametric bootstrap under each hypothesis (Table 5, Appendix B.4):

  truth = w0waCDM at the DR2 BAO MLE (-0.856, -0.430)  [500 draws], and
  truth = LCDM                                         [200 draws],

both at the paper's fixed background, with Gaussian noise drawn from the
published 13x13 DR2 covariance (Cholesky). Every synthetic data vector is
passed through the paper's exact LOO pipeline (evalue_analysis.loocv_evalue:
hold out one bin, fit (w0, wa) to the other six with 5 multistarts, score the
held-out bin) and we record each bin's share of sum_k E_k^LOO.

Outputs results/loo_benchmark.json with the raw draws and the summary
statistics quoted in the paper:
  - median / 68% / 95% intervals of the max-bin share under each truth,
  - P(some bin carries >= 75% of the LOO sum),
  - P(max-bin share >= the observed LRG2 share),
  - P(LRG2 specifically >= the observed share),
  - dominant-bin frequencies.

Seeds are fixed (BASE_SEED with per-draw spawn keys), so the run is exactly
reproducible.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'code'))

from data_loader import load_desi_data
from cosmology import CosmologyParams, LCDM, compute_bao_predictions
from evalue_analysis import loocv_evalue, _build_theory_vector

BASE_SEED = 20260710
MLE = CosmologyParams(w0=-0.856, wa=-0.430)
N_ALT = 500
N_NULL = 200


def summarize(draws, observed_share, lrg2_label='LRG2'):
    max_share = np.array([d['max_share'] for d in draws])
    lrg2_share = np.array([d['lrg2_share'] for d in draws])
    bins, counts = np.unique([d['max_bin'] for d in draws], return_counts=True)
    return dict(
        n_draws=len(draws),
        max_share_median=float(np.median(max_share)),
        max_share_ci68=[float(np.percentile(max_share, 16)),
                        float(np.percentile(max_share, 84))],
        max_share_ci95=[float(np.percentile(max_share, 2.5)),
                        float(np.percentile(max_share, 97.5))],
        p_some_bin_ge_75pct=float(np.mean(max_share >= 0.75)),
        p_max_share_ge_observed=float(np.mean(max_share >= observed_share)),
        p_lrg2_ge_observed=float(np.mean(lrg2_share >= observed_share)),
        observed_lrg2_share=float(observed_share),
        dominant_bin_freq={str(b): float(c / len(draws))
                           for b, c in zip(bins, counts)},
    )


def main():
    ds = load_desi_data(REPO / 'data' / 'dr2', 'DR2')
    z, quantities, cov = ds.z_eff, ds.quantities, ds.cov
    L = np.linalg.cholesky(cov)
    n = len(ds.data)

    mu_alt = _build_theory_vector(compute_bao_predictions(z, MLE), z, quantities)
    mu_null = _build_theory_vector(compute_bao_predictions(z, LCDM), z, quantities)

    zkey_to_tracer = {round(float(zi), 4): tr for zi, tr in zip(z, ds.tracers)}
    lrg2_key = [k for k, v in zkey_to_tracer.items() if v == 'LRG2'][0]

    # Observed concentration, recomputed from the real data for provenance.
    obs = loocv_evalue(ds.data, cov, z, quantities, verbose=False)
    obs_total = sum(obs.per_bin_e.values())
    observed_share = obs.per_bin_e[lrg2_key] / obs_total

    def one_draw(mu, seed):
        rng = np.random.default_rng([BASE_SEED, seed])
        data = mu + L @ rng.standard_normal(n)
        res = loocv_evalue(data, cov, z, quantities, verbose=False)
        e = res.per_bin_e
        total = sum(e.values())
        max_key = max(e, key=e.get)
        return dict(seed=int(seed),
                    lrg2_share=float(e[lrg2_key] / total),
                    max_share=float(e[max_key] / total),
                    max_bin=zkey_to_tracer[max_key],
                    avg_e=float(res.e_average))

    t0 = time.time()
    draws_alt, draws_null = [], []
    for i in range(N_ALT):
        draws_alt.append(one_draw(mu_alt, i))
        if (i + 1) % 50 == 0:
            print(f'[alt] {i+1}/{N_ALT} ({time.time()-t0:.0f}s)', flush=True)
    for i in range(N_NULL):
        draws_null.append(one_draw(mu_null, 100000 + i))
        if (i + 1) % 50 == 0:
            print(f'[null] {i+1}/{N_NULL} ({time.time()-t0:.0f}s)', flush=True)

    out = dict(
        description=('Parametric-bootstrap calibration of the LOO concentration '
                     'statistic under smooth w0waCDM truth (DR2 MLE) and LCDM '
                     'truth; paper background; published DR2 covariance; exact '
                     'Section 4.2 LOO pipeline.'),
        observed_lrg2_share=float(observed_share),
        smooth_truth=summarize(draws_alt, observed_share),
        null_truth=summarize(draws_null, observed_share),
        draws=dict(smooth_truth=draws_alt, null_truth=draws_null),
        seconds=time.time() - t0,
    )
    (REPO / 'results').mkdir(exist_ok=True)
    with open(REPO / 'results' / 'loo_benchmark.json', 'w') as f:
        json.dump(out, f, indent=1)

    s, h = out['smooth_truth'], out['null_truth']
    print(f"observed LRG2 share = {observed_share:.4f}")
    print(f"[smooth truth, N={s['n_draws']}] median max share = "
          f"{s['max_share_median']:.3f}; P(some bin >= 75%) = "
          f"{s['p_some_bin_ge_75pct']:.3f}; P(max >= obs) = "
          f"{s['p_max_share_ge_observed']:.3f}; P(LRG2 >= obs) = "
          f"{s['p_lrg2_ge_observed']:.3f}")
    print(f"[smooth truth] dominant bins: {s['dominant_bin_freq']}")
    print(f"[LCDM truth, N={h['n_draws']}] median max share = "
          f"{h['max_share_median']:.3f}; P(some bin >= 75%) = "
          f"{h['p_some_bin_ge_75pct']:.3f}")
    print(f"wrote results/loo_benchmark.json")


if __name__ == '__main__':
    main()
