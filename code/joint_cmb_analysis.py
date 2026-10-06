#!/usr/bin/env python3
"""
Joint DESI-DR2-BAO + compressed-Planck-CMB analysis (Appendix B.7).

Reproduces the joint BAO+CMB numbers reported in the paper:
  - joint MLE (w0, wa) = (-0.818, -0.695), Delta chi^2 = 14.78, Wilks sigma = 3.42
  - LOO e-value     E_LOO^BAO+CMB        = 7.07
  - mixture e-value E_mix,Default^BAO+CMB = 2.19
  - prior-width sensitivity table (Narrow 7.2 / Default 2.19 / Wide 0.91 / Ong 0.091)

Constructions (matching the paper):
  - The CMB block is the Chen et al. 2019 compressed Planck 2018 likelihood
    [R, l_A, omega_b h^2] with r_s(z*) = 144.65 Mpc (see extended_analysis).
  - E_mix: uniform mixture of the JOINT likelihood ratio over a 30x30 grid
    (the same grid convention as grow_evalue in evalue_analysis.py).
  - E_LOO: 7 folds, one per DESI redshift bin. The CMB block is ALWAYS in the
    training set and is NEVER held out; each fold fits (w0, wa) on the joint
    {6 remaining BAO bins + CMB} likelihood and evaluates the likelihood-ratio
    e-value on the held-out BAO bin only. The reported value is the average
    over folds (valid by linearity of expectation, as in loocv_evalue).

Writes results/joint_cmb.json and checks hard verification gates against the
paper values; exits nonzero if any gate fails.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2_dist
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))

from cosmology import (
    CosmologyParams, LCDM, compute_bao_predictions, chi_squared,
    log_likelihood,
)
from data_loader import load_desi_data
from evalue_analysis import _build_theory_vector, _identify_redshift_bins
from extended_analysis import CMBCompressed, compute_cmb_predictions

REPO = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO / 'results' / 'joint_cmb.json'

# Prior ranges for the mixture e-value sensitivity table (paper appendix).
PRIOR_RANGES = {
    'narrow': ((-1.0, -0.5), (-1.5, 0.0)),
    'default': ((-1.5, -0.5), (-2.0, 1.0)),
    'wide': ((-2.0, 0.0), (-3.0, 2.0)),
    'ong': ((-3.0, 1.0), (-3.0, 2.0)),
}

# Paper values and gate tolerances (any failure is reported and the script exits nonzero).
GATES = {
    'smoke_chi2_lcdm': (0.62, 3.0, 'abs_max'),     # LCDM vs Chen vector, 3 dof
    'w0_mle': (-0.818, 0.01, 'abs'),
    'wa_mle': (-0.695, 0.01, 'abs'),
    'delta_chi2': (14.78, 0.05, 'abs'),
    'wilks_sigma_two_sided': (3.42, 0.03, 'abs'),
    'E_LOO': (7.07, 0.02, 'rel'),
    'E_mix_narrow': (7.2, 0.03, 'rel'),
    'E_mix_default': (2.19, 0.03, 'rel'),
    'E_mix_wide': (0.91, 0.03, 'rel'),
    'E_mix_ong': (0.091, 0.03, 'rel'),
}


def bao_chi2_and_loglike(dataset, cosmo):
    """Returns: (float chi^2, float log-likelihood) of the full BAO vector."""
    pred = compute_bao_predictions(dataset.z_eff, cosmo)
    theory = _build_theory_vector(pred, dataset.z_eff, dataset.quantities)
    return (chi_squared(dataset.data, theory, dataset.cov),
            log_likelihood(dataset.data, theory, dataset.cov))


def cmb_chi2(cmb, cosmo):
    """Returns: float chi^2 of the compressed CMB block."""
    pred = compute_cmb_predictions(cosmo, cmb)
    return chi_squared(cmb.data, pred, cmb.cov)


def joint_chi2(w0, wa, dataset, cmb):
    """Returns: float chi^2_BAO + chi^2_CMB at (w0, wa), fixed background."""
    cosmo = CosmologyParams(w0=w0, wa=wa)
    chi2_bao, _ = bao_chi2_and_loglike(dataset, cosmo)
    return chi2_bao + cmb_chi2(cmb, cosmo)


def joint_mle(dataset, cmb):
    """Joint (w0, wa) MLE via Nelder-Mead with a coarse-grid fallback.

    Returns: dict with MLE, chi^2 values, Delta chi^2, and Wilks sigmas.
    """
    objective = lambda p: joint_chi2(p[0], p[1], dataset, cmb)

    result = minimize(objective, x0=[-0.85, -0.5], method='Nelder-Mead',
                      options={'xatol': 1e-4, 'fatol': 1e-6})

    # Coarse-grid fallback in case Nelder-Mead wandered to a local minimum
    w0_coarse = np.linspace(-1.5, -0.5, 21)
    wa_coarse = np.linspace(-2.0, 1.0, 21)
    grid_best = min(((objective([w0, wa]), w0, wa)
                     for w0 in w0_coarse for wa in wa_coarse))
    if grid_best[0] < result.fun:
        result = minimize(objective, x0=[grid_best[1], grid_best[2]],
                          method='Nelder-Mead',
                          options={'xatol': 1e-4, 'fatol': 1e-6})

    chi2_lcdm = joint_chi2(-1.0, 0.0, dataset, cmb)
    delta = chi2_lcdm - result.fun
    p_two = chi2_dist.sf(delta, 2)
    return {
        'w0': float(result.x[0]),
        'wa': float(result.x[1]),
        'chi2_mle': float(result.fun),
        'chi2_lcdm': float(chi2_lcdm),
        'delta_chi2': float(delta),
        'wilks_sigma_two_sided': float(norm.isf(p_two / 2)),
        'wilks_sigma_one_sided': float(norm.isf(p_two)),
    }


def joint_emix(dataset, cmb, w0_range, wa_range, n=30):
    """Uniform mixture e-value of the joint likelihood ratio on an n x n grid.

    Same construction as grow_evalue (log-sum-exp over a uniform grid), with
    the joint BAO+CMB chi^2 replacing the BAO-only chi^2.

    Returns: float mixture e-value.
    """
    chi2_null = joint_chi2(-1.0, 0.0, dataset, cmb)
    log_ratios = []
    for w0 in np.linspace(w0_range[0], w0_range[1], n):
        for wa in np.linspace(wa_range[0], wa_range[1], n):
            log_ratios.append(-(joint_chi2(w0, wa, dataset, cmb) - chi2_null) / 2.0)
    log_ratios = np.array(log_ratios)
    m = np.max(log_ratios)
    return float(np.exp(m + np.log(np.mean(np.exp(log_ratios - m)))))


def joint_loocv(dataset, cmb):
    """LOO e-value over the 7 BAO redshift bins with the CMB always in training.

    Mirrors loocv_evalue (same folds, same multi-start L-BFGS-B, same
    average-not-product), with the per-fold training objective extended by
    the CMB Gaussian log-likelihood. The held-out e-value is computed on the
    BAO bin alone; the CMB block is never held out.

    Returns: (float average e-value, dict per-bin e-values).
    """
    data, cov = dataset.data, dataset.cov
    z_values, quantities = dataset.z_eff, dataset.quantities
    bins = _identify_redshift_bins(z_values)

    per_bin_e = {}
    for z_held_out, held_out_idx in sorted(bins.items()):
        held_out_idx = np.array(held_out_idx)
        train_idx = np.array([i for i in range(len(data)) if i not in held_out_idx])

        data_train = data[train_idx]
        cov_train = cov[np.ix_(train_idx, train_idx)]
        z_train = z_values[train_idx]
        q_train = [quantities[i] for i in train_idx]

        data_test = data[held_out_idx]
        cov_test = cov[np.ix_(held_out_idx, held_out_idx)]
        z_test = z_values[held_out_idx]
        q_test = [quantities[i] for i in held_out_idx]

        def neg_log_likelihood_train(params):
            w0, wa = params
            cosmo = CosmologyParams(w0=w0, wa=wa)
            pred = compute_bao_predictions(z_train, cosmo)
            theory = _build_theory_vector(pred, z_train, q_train)
            nll = -log_likelihood(data_train, theory, cov_train)
            # CMB block: always in the training likelihood (Gaussian)
            nll += 0.5 * cmb_chi2(cmb, cosmo)
            return nll

        best_result = None
        best_nll = np.inf
        starts = [
            [-0.9, -0.5],
            [-1.0, 0.0],
            [-0.75, -1.0],
            [-0.8, -0.8],
            [-1.1, 0.5],
        ]
        for x0 in starts:
            try:
                result = minimize(
                    neg_log_likelihood_train,
                    x0=x0,
                    bounds=[(-2.0, 0.0), (-4.0, 3.0)],
                    method='L-BFGS-B'
                )
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_result = result
            except Exception:
                continue

        alt_cosmo = CosmologyParams(w0=best_result.x[0], wa=best_result.x[1])

        pred_null_test = compute_bao_predictions(z_test, LCDM)
        theory_null_test = _build_theory_vector(pred_null_test, z_test, q_test)
        pred_alt_test = compute_bao_predictions(z_test, alt_cosmo)
        theory_alt_test = _build_theory_vector(pred_alt_test, z_test, q_test)

        log_ek = (log_likelihood(data_test, theory_alt_test, cov_test)
                  - log_likelihood(data_test, theory_null_test, cov_test))
        per_bin_e[float(z_held_out)] = float(np.exp(log_ek))

    e_average = float(np.mean(list(per_bin_e.values())))
    return e_average, per_bin_e


def check_gates(values):
    """Compare computed values against the paper's numbers.

    Returns: list of (name, target, got, passed) tuples.
    """
    report = []
    for name, (target, tol, kind) in GATES.items():
        got = values[name]
        if kind == 'abs':
            passed = abs(got - target) <= tol
        elif kind == 'rel':
            passed = abs(got / target - 1.0) <= tol
        else:  # abs_max: value must simply be below tol
            passed = got < tol
        report.append((name, target, got, passed))
    return report


def main():
    data_dir = REPO / 'data' / 'dr2'
    dataset = load_desi_data(data_dir, 'DR2')
    cmb = CMBCompressed()

    # Smoke test: LCDM against the Chen 2019 measured vector (3 dof).
    # Catches any convention error (D_M vs D_A, r_s vs r_d, omega_r) at once.
    smoke = cmb_chi2(cmb, LCDM)

    mle = joint_mle(dataset, cmb)
    e_loo, per_bin = joint_loocv(dataset, cmb)
    e_mix = {name: joint_emix(dataset, cmb, w0r, war)
             for name, (w0r, war) in PRIOR_RANGES.items()}

    values = {
        'smoke_chi2_lcdm': smoke,
        'w0_mle': mle['w0'],
        'wa_mle': mle['wa'],
        'delta_chi2': mle['delta_chi2'],
        'wilks_sigma_two_sided': mle['wilks_sigma_two_sided'],
        'E_LOO': e_loo,
        'E_mix_narrow': e_mix['narrow'],
        'E_mix_default': e_mix['default'],
        'E_mix_wide': e_mix['wide'],
        'E_mix_ong': e_mix['ong'],
    }
    report = check_gates(values)

    out = {
        'mle': {'w0': mle['w0'], 'wa': mle['wa']},
        'chi2_lcdm': mle['chi2_lcdm'],
        'chi2_mle': mle['chi2_mle'],
        'delta_chi2': mle['delta_chi2'],
        'wilks_sigma_two_sided': mle['wilks_sigma_two_sided'],
        'wilks_sigma_one_sided': mle['wilks_sigma_one_sided'],
        'E_LOO_BAO_CMB': e_loo,
        'per_bin_E_LOO': {str(z): e for z, e in sorted(per_bin.items())},
        'E_mix': e_mix,
        'smoke_test_chi2_lcdm_3dof': smoke,
        'conventions': {
            'rs_star_mpc': 144.65, 'z_star': cmb.z_star,
            'rd_mpc': 147.05, 'omega_r': 9e-5,
            'h': 0.6766, 'omega_m': 0.3111,
        },
        'gates': {name: {'target': t, 'value': v, 'pass': bool(p)}
                  for name, t, v, p in report},
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(out, f, indent=2)

    all_pass = all(p for _, _, _, p in report)
    for name, target, got, passed in report:
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {got:.4g} (paper: {target})")
    print(f"\n{'ALL GATES PASS' if all_pass else 'GATE FAILURE'} -> {RESULTS_PATH}")
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
