#!/usr/bin/env python3
"""Monte Carlo power calibration for the data-split e-value procedure (Appendix B.6).

Produces the data-split power claim: under w0waCDM truth (DESI DR2 best-fit) with
the split at z=1, the median E_split is only ~1.2 over 10^4 Monte Carlo
realisations -- barely above 1 even when w0waCDM is the truth -- confirming the
split test is underpowered by construction, so the observed E_split = 1.4 is
uninformative about the signal. With 10^4 sims the MC standard error on any
rejection-rate (power) estimate is at most 0.5/sqrt(n) ~ 0.5%.

For context the same machinery is run under H0 (LCDM true) for calibration and at
alternative split points. Writes results/power_calibration.json and prints the
headline median and rejection rate.

References:
- Shafer (2021): Testing by betting
- DESI DR2: arXiv:2503.14738
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

from cosmology import (
    CosmologyParams, LCDM, DESI_DR2_BEST_FIT,
    compute_bao_predictions, chi_squared, log_likelihood
)
from data_loader import load_desi_data
from evalue_analysis import _build_theory_vector

REPO = Path(__file__).resolve().parents[1]


def generate_simulated_data(z_eff, quantities, cov, true_cosmo):
    """Draw one simulated BAO data vector: true theory plus Gaussian noise from cov."""
    pred = compute_bao_predictions(z_eff, true_cosmo)
    theory = _build_theory_vector(pred, z_eff, quantities)
    noise = np.random.multivariate_normal(np.zeros(len(theory)), cov)
    return theory + noise


def run_split_evalue_on_sim(sim_data, z_eff, quantities, cov, split_z=1.0):
    """Data-split e-value on one simulated dataset.

    Fits (w0, wa) on low-z (z < split_z), tests on high-z. Mirrors split_evalue in
    evalue_analysis.py but on a provided data vector. Returns None if either side
    has < 2 bins, else a dict with e_split / e_train and test chi^2 diagnostics.
    """
    low_z_idx = np.where(z_eff < split_z)[0]
    high_z_idx = np.where(z_eff >= split_z)[0]

    if len(low_z_idx) < 2 or len(high_z_idx) < 2:
        return None

    data_low = sim_data[low_z_idx]
    data_high = sim_data[high_z_idx]
    cov_low = cov[np.ix_(low_z_idx, low_z_idx)]
    cov_high = cov[np.ix_(high_z_idx, high_z_idx)]
    z_low = z_eff[low_z_idx]
    z_high = z_eff[high_z_idx]
    q_low = [quantities[i] for i in low_z_idx]
    q_high = [quantities[i] for i in high_z_idx]

    def neg_log_like_train(params):
        w0, wa = params
        cosmo = CosmologyParams(w0=w0, wa=wa)
        pred = compute_bao_predictions(z_low, cosmo)
        theory = _build_theory_vector(pred, z_low, q_low)
        return -log_likelihood(data_low, theory, cov_low)

    result = minimize(
        neg_log_like_train,
        x0=[-0.9, -0.5],
        bounds=[(-2.0, 0.0), (-3.0, 2.0)],
        method='L-BFGS-B'
    )
    w0_fit, wa_fit = result.x
    alt_cosmo = CosmologyParams(w0=w0_fit, wa=wa_fit)

    pred_null_high = compute_bao_predictions(z_high, LCDM)
    pred_alt_high = compute_bao_predictions(z_high, alt_cosmo)
    theory_null_high = _build_theory_vector(pred_null_high, z_high, q_high)
    theory_alt_high = _build_theory_vector(pred_alt_high, z_high, q_high)

    log_e_test = (log_likelihood(data_high, theory_alt_high, cov_high)
                  - log_likelihood(data_high, theory_null_high, cov_high))

    pred_null_low = compute_bao_predictions(z_low, LCDM)
    pred_alt_low = compute_bao_predictions(z_low, alt_cosmo)
    theory_null_low = _build_theory_vector(pred_null_low, z_low, q_low)
    theory_alt_low = _build_theory_vector(pred_alt_low, z_low, q_low)
    log_e_train = (log_likelihood(data_low, theory_alt_low, cov_low)
                   - log_likelihood(data_low, theory_null_low, cov_low))

    return {
        'e_split': np.exp(log_e_test),
        'log_e_split': log_e_test,
        'w0_fit': w0_fit,
        'wa_fit': wa_fit,
        'e_train': np.exp(log_e_train),
        'log_e_train': log_e_train,
        'chi2_null_test': chi_squared(data_high, theory_null_high, cov_high),
        'chi2_alt_test': chi_squared(data_high, theory_alt_high, cov_high),
    }


def run_full_evalue_on_sim(sim_data, z_eff, quantities, cov):
    """Full (non-split) likelihood-ratio e-value: fits (w0, wa) to all bins and
    tests on the same bins. This is the biased version used only for the ratio."""
    def neg_log_like(params):
        w0, wa = params
        cosmo = CosmologyParams(w0=w0, wa=wa)
        pred = compute_bao_predictions(z_eff, cosmo)
        theory = _build_theory_vector(pred, z_eff, quantities)
        return -log_likelihood(sim_data, theory, cov)

    result = minimize(
        neg_log_like,
        x0=[-0.9, -0.5],
        bounds=[(-2.0, 0.0), (-3.0, 2.0)],
        method='L-BFGS-B'
    )
    w0_fit, wa_fit = result.x
    alt_cosmo = CosmologyParams(w0=w0_fit, wa=wa_fit)

    pred_null = compute_bao_predictions(z_eff, LCDM)
    pred_alt = compute_bao_predictions(z_eff, alt_cosmo)
    theory_null = _build_theory_vector(pred_null, z_eff, quantities)
    theory_alt = _build_theory_vector(pred_alt, z_eff, quantities)

    log_e = (log_likelihood(sim_data, theory_alt, cov)
             - log_likelihood(sim_data, theory_null, cov))

    return {
        'e_full': np.exp(log_e),
        'log_e_full': log_e,
        'w0_fit': w0_fit,
        'wa_fit': wa_fit,
    }


def run_monte_carlo(z_eff, quantities, cov, true_cosmo, n_sims=500,
                    split_z=1.0, compute_full=False):
    """Run n_sims realisations of the split e-value under true_cosmo.

    Returns a dict of result arrays (e_split, e_train, fitted params, and e_full
    if compute_full) plus success/failure counts.
    """
    e_splits, log_e_splits = [], []
    w0_fits, wa_fits, e_trains = [], [], []
    e_fulls, log_e_fulls = [], []
    failed = 0

    for _ in range(n_sims):
        sim_data = generate_simulated_data(z_eff, quantities, cov, true_cosmo)
        res = run_split_evalue_on_sim(sim_data, z_eff, quantities, cov, split_z=split_z)
        if res is None:
            failed += 1
            continue

        e_splits.append(res['e_split'])
        log_e_splits.append(res['log_e_split'])
        w0_fits.append(res['w0_fit'])
        wa_fits.append(res['wa_fit'])
        e_trains.append(res['e_train'])

        if compute_full:
            res_full = run_full_evalue_on_sim(sim_data, z_eff, quantities, cov)
            e_fulls.append(res_full['e_full'])
            log_e_fulls.append(res_full['log_e_full'])

    results = {
        'e_split': np.array(e_splits),
        'log_e_split': np.array(log_e_splits),
        'w0_fit': np.array(w0_fits),
        'wa_fit': np.array(wa_fits),
        'e_train': np.array(e_trains),
        'n_success': len(e_splits),
        'n_failed': failed,
    }
    if compute_full:
        results['e_full'] = np.array(e_fulls)
        results['log_e_full'] = np.array(log_e_fulls)
    return results


def summarize(results):
    """Headline statistics for one MC run: the median E_split (robust to the heavy
    upper tail, which makes the mean uninformative) and the rejection rate
    P(E_split > 1.4) with its binomial MC standard error -- with n = 10^4 sims this
    proportion SE is at most 0.5%, the precision the paper's power claim rests on."""
    e = results['e_split']
    n = len(e)
    p_exceed = float(np.mean(e > 1.4))
    summary = {
        'n_success': n,
        'median_e_split': float(np.median(e)),
        'mean_e_split': float(np.mean(e)),
        'p_exceed_1.4': p_exceed,
        'p_exceed_1.4_se': float(np.sqrt(p_exceed * (1.0 - p_exceed) / n)),
    }
    if 'e_full' in results:
        ratio = results['e_full'] / np.maximum(e, 1e-10)
        summary['median_ratio_full_split'] = float(np.median(ratio))
    return summary


def main():
    dr2 = load_desi_data(REPO / 'data' / 'dr2', 'DR2')

    N_SIMS = 10000
    np.random.seed(42)

    results_h0 = run_monte_carlo(
        dr2.z_eff, dr2.quantities, dr2.cov,
        true_cosmo=LCDM, n_sims=N_SIMS, split_z=1.0, compute_full=False,
    )
    results_h1 = run_monte_carlo(
        dr2.z_eff, dr2.quantities, dr2.cov,
        true_cosmo=DESI_DR2_BEST_FIT, n_sims=N_SIMS, split_z=1.0, compute_full=True,
    )

    split_sensitivity = {}
    for sz in [0.8, 1.0, 1.2]:
        n_low = int(np.sum(dr2.z_eff < sz))
        n_high = int(np.sum(dr2.z_eff >= sz))
        if n_low < 2 or n_high < 2:
            continue
        res_sz = run_monte_carlo(
            dr2.z_eff, dr2.quantities, dr2.cov,
            true_cosmo=DESI_DR2_BEST_FIT, n_sims=N_SIMS, split_z=sz, compute_full=False,
        )
        split_sensitivity[f'{sz}'] = summarize(res_sz)

    # P(E_split > thresh) under H0 (size) and H1 (power) across thresholds.
    e_h0 = results_h0['e_split']
    e_h1 = results_h1['e_split']
    power_table = {
        f'{thresh}': {
            'size_h0': float(np.mean(e_h0 > thresh)),
            'power_h1': float(np.mean(e_h1 > thresh)),
        }
        for thresh in [1.0, 1.4, 3.0, 5.0, 10.0, 20.0]
    }

    out = {
        'n_sims': N_SIMS,
        'seed': 42,
        'split_z': 1.0,
        'w0waCDM_truth': {'w0': DESI_DR2_BEST_FIT.w0, 'wa': DESI_DR2_BEST_FIT.wa},
        'H0_lcdm_true': summarize(results_h0),
        'H1_w0waCDM_true': summarize(results_h1),
        'split_sensitivity_H1': split_sensitivity,
        'power_vs_threshold': power_table,
    }

    h1 = out['H1_w0waCDM_true']
    print(f"Median E_split under w0waCDM truth (split z=1.0): {h1['median_e_split']:.3f}")
    print(f"  P(E_split > 1.4) = {h1['p_exceed_1.4']:.3f} +/- {h1['p_exceed_1.4_se']:.4f} ({N_SIMS} sims)")
    print(f"Median E_split under LCDM truth: {out['H0_lcdm_true']['median_e_split']:.3f}")

    out_dir = REPO / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'power_calibration.json', 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
