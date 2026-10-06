#!/usr/bin/env python3
"""Posterior-mean cross-prediction E values (Appendix B.8).

For each SN+CMB compilation (Pantheon+, DES-Y5, Union3) we draw N=200,000 samples
of (w0, wa) from a Gaussian approximation to the published joint posterior
(built from the published MAP, marginal sigmas, and w0-wa correlation), compute
the cross-prediction E on DESI DR2 BAO at each draw, and average over draws.

The posterior-mean E is itself a valid e-value by linearity of expectation: each
draw is a pre-specified point alternative independent of the DESI BAO data.

Reproduces Appendix B.8. Point E (deterministic, at each published MAP):
Pantheon+ 9.6, DES-Y5 47, Union3 44. The posterior-mean E is heavy-tailed, so it
needs a large N to stabilise; at N=200,000 it converges (seed-independent to
<1%) to Pantheon+ 342, DES-Y5 240, Union3 128.

Writes results/cross_prediction.json.  Run: python posterior_mean_cross_prediction.py
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'code'))

from cosmology import (
    CosmologyParams, LCDM, compute_bao_predictions, chi_squared
)
from data_loader import load_desi_data
from evalue_analysis import _build_theory_vector
from extended_analysis import PANTHEON_PLUS, DESY5, UNION3


def cross_prediction_E_at_point(bao_data, w0, wa):
    """Cross-prediction e-value at a single (w0, wa) point.

    E = exp((chi2_lcdm - chi2_alt) / 2) on the BAO data.
    """
    cosmo_alt = CosmologyParams(w0=w0, wa=wa)
    pred_alt = compute_bao_predictions(bao_data.z_eff, cosmo_alt)
    theory_alt = _build_theory_vector(pred_alt, bao_data.z_eff, bao_data.quantities)

    pred_lcdm = compute_bao_predictions(bao_data.z_eff, LCDM)
    theory_lcdm = _build_theory_vector(pred_lcdm, bao_data.z_eff, bao_data.quantities)

    chi2_lcdm = chi_squared(bao_data.data, theory_lcdm, bao_data.cov)
    chi2_alt = chi_squared(bao_data.data, theory_alt, bao_data.cov)

    return np.exp((chi2_lcdm - chi2_alt) / 2)


def sample_posterior_w0wa(sn, n_samples=5000, seed=None):
    """Draw (w0, wa) from a Gaussian approximation to the SN+CMB joint posterior,
    using the published MAP, marginal sigmas, and the w0-wa correlation.

    Returns array of shape (n_samples, 2).
    """
    if seed is not None:
        np.random.seed(seed)

    mean = np.array([sn.w0, sn.wa])
    sigma = np.array([sn.sigma_w0, sn.sigma_wa])
    rho = sn.corr_w0_wa
    cov = np.array([
        [sigma[0]**2,               rho * sigma[0] * sigma[1]],
        [rho * sigma[0] * sigma[1], sigma[1]**2],
    ])
    return np.random.multivariate_normal(mean, cov, size=n_samples)


def posterior_mean_E(bao_data, sn, n_samples=200_000, seed=42):
    """Posterior-mean cross-prediction E for a SN+CMB compilation.

    E_post = (1/N) sum_i E(w0_i, wa_i) over N draws from the SN+CMB posterior.
    Valid as an e-value by linearity of expectation under independence of the
    SN+CMB posterior and the DESI BAO data. The LCDM null is constant across
    draws, so its chi^2 is computed once.

    Returns (E_mean, E_values).
    """
    samples = sample_posterior_w0wa(sn, n_samples=n_samples, seed=seed)
    theory_lcdm = _build_theory_vector(compute_bao_predictions(bao_data.z_eff, LCDM),
                                       bao_data.z_eff, bao_data.quantities)
    chi2_lcdm = chi_squared(bao_data.data, theory_lcdm, bao_data.cov)
    E_values = np.empty(len(samples))
    for i, (w0, wa) in enumerate(samples):
        theory = _build_theory_vector(
            compute_bao_predictions(bao_data.z_eff, CosmologyParams(w0=w0, wa=wa)),
            bao_data.z_eff, bao_data.quantities)
        E_values[i] = np.exp((chi2_lcdm - chi_squared(bao_data.data, theory, bao_data.cov)) / 2)
    return E_values.mean(), E_values


def main():
    dr2 = load_desi_data(REPO / 'data' / 'dr2', 'DR2')

    results = {}
    for sn in [PANTHEON_PLUS, DESY5, UNION3]:
        E_point = cross_prediction_E_at_point(dr2, sn.w0, sn.wa)
        E_post, _ = posterior_mean_E(dr2, sn, seed=42)
        results[sn.name] = {'E_point': float(E_point), 'E_post_mean': float(E_post)}
        print(f"{sn.name:12s}  point E = {E_point:6.2f}   post-mean E = {E_post:7.1f}")

    results_dir = REPO / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / 'cross_prediction.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
