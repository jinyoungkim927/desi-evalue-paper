#!/usr/bin/env python3
"""
Minimum prior concentration δ* for DESI DR2 to reject ΛCDM at α=0.05.

Produces Appendix B.5: δ*_lower = 0.87, δ*_upper = 13.5, δ_max = 2.7,
M_max = 1.7e2, and the local-Gaussian Δχ² = 16.5 vs exact Δχ² = 16.86.

For Gaussian likelihood + Gaussian prior N(θ_0, δ² F^{-1}), there is a closed
form for the mixture e-value:
  M(δ) = exp(½ Δχ² · δ²/(1+δ²)) / (1+δ²)^(k/2)
where Δχ² = (θ̂_MLE - θ_0)^T F (θ̂_MLE - θ_0), k = number of free alternative
parameters (2 for w0, wa), and δ measures prior width in Fisher-σ units.

Pipeline:
  1. Compute Fisher F, θ̂_MLE, Δχ² numerically from DR2.
  2. Verify the closed form against direct grid-integration of the mixture.
  3. Solve M(δ) = 20 (α=0.05 threshold) for the rejection band [δ*_lower, δ*_upper].
  4. Locate the Bayes-factor-maximizing δ_max and M_max = M(δ_max).

Outputs:
- results/delta_star_results.json
"""
import sys
import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize, brentq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))

from data_loader import load_desi_data
from cosmology import CosmologyParams, compute_bao_predictions, chi_squared
from evalue_analysis import _build_theory_vector

RESULTS_PATH = ROOT / 'results' / 'delta_star_results.json'


def chi2_at(theta, dataset):
    """Chi^2 at parameter point (w0, wa)."""
    w0, wa = theta
    cosmo = CosmologyParams(w0=w0, wa=wa)
    pred = compute_bao_predictions(dataset.z_eff, cosmo)
    mu = _build_theory_vector(pred, dataset.z_eff, dataset.quantities)
    return chi_squared(dataset.data, mu, dataset.cov)


def fit_mle(dataset, x0=(-0.85, -0.5)):
    """Find the MLE (w0, wa); returns (theta_mle, chi2_mle)."""
    res = minimize(lambda p: chi2_at(p, dataset), x0=x0, method='Nelder-Mead',
                   options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 5000})
    return res.x, res.fun


def fisher_matrix(theta_center, dataset, h_step=1e-3):
    """
    Numerical 2x2 Fisher information at theta_center for (w0, wa).

    For a Gaussian likelihood with fixed covariance,
      F_ij = (∂μ/∂θ_i)^T C^{-1} (∂μ/∂θ_j),
    with the derivatives taken by central finite difference on the mean vector.
    """
    w0_c, wa_c = theta_center

    pred_w0p = compute_bao_predictions(
        dataset.z_eff, CosmologyParams(w0=w0_c + h_step, wa=wa_c))
    mu_w0p = _build_theory_vector(pred_w0p, dataset.z_eff, dataset.quantities)
    pred_w0m = compute_bao_predictions(
        dataset.z_eff, CosmologyParams(w0=w0_c - h_step, wa=wa_c))
    mu_w0m = _build_theory_vector(pred_w0m, dataset.z_eff, dataset.quantities)
    dmu_dw0 = (mu_w0p - mu_w0m) / (2 * h_step)

    pred_wap = compute_bao_predictions(
        dataset.z_eff, CosmologyParams(w0=w0_c, wa=wa_c + h_step))
    mu_wap = _build_theory_vector(pred_wap, dataset.z_eff, dataset.quantities)
    pred_wam = compute_bao_predictions(
        dataset.z_eff, CosmologyParams(w0=w0_c, wa=wa_c - h_step))
    mu_wam = _build_theory_vector(pred_wam, dataset.z_eff, dataset.quantities)
    dmu_dwa = (mu_wap - mu_wam) / (2 * h_step)

    Cinv = np.linalg.inv(dataset.cov)
    F = np.array([
        [dmu_dw0 @ Cinv @ dmu_dw0, dmu_dw0 @ Cinv @ dmu_dwa],
        [dmu_dw0 @ Cinv @ dmu_dwa, dmu_dwa @ Cinv @ dmu_dwa]
    ])
    return F


def mixture_e_gaussian_prior(dataset, theta_0, F, delta, n_grid=80, n_sigma=5):
    """
    Numerical mixture e-value M(δ) = ∫ L(θ) π(θ) dθ / L(θ_0) with prior
    π = N(theta_0, δ² F^{-1}).

    Integrated on a grid in the eigenvector basis of F so the grid axes align
    with the Fisher (and hence prior) principal axes.
    """
    eigvals, eigvecs = np.linalg.eigh(F)
    grid_sigmas = delta / np.sqrt(eigvals)  # prior std per eigen-direction
    grid_bounds = n_sigma * grid_sigmas

    g1 = np.linspace(-grid_bounds[0], grid_bounds[0], n_grid)
    g2 = np.linspace(-grid_bounds[1], grid_bounds[1], n_grid)
    G1, G2 = np.meshgrid(g1, g2, indexing='ij')

    # eigen-coords z -> (w0, wa): theta = theta_0 + V z, V columns are eigvecs
    Z = np.stack([G1.ravel(), G2.ravel()], axis=0)
    Theta = theta_0[:, None] + eigvecs @ Z

    # log L = -0.5 chi^2 + const; the const cancels against the null in the ratio
    log_L = np.array([-0.5 * chi2_at(Theta[:, i], dataset)
                      for i in range(Theta.shape[1])])
    log_LR = log_L - (-0.5 * chi2_at(theta_0, dataset))

    log_prior = (-0.5 * ((G1.ravel() / grid_sigmas[0])**2
                         + (G2.ravel() / grid_sigmas[1])**2)
                 - np.log(2 * np.pi)
                 - np.log(grid_sigmas[0]) - np.log(grid_sigmas[1]))

    cell = (g1[1] - g1[0]) * (g2[1] - g2[0])

    # log-sum-exp on M = Σ exp(log_LR + log_prior) * cell
    log_integrand = log_LR + log_prior
    m_max = log_integrand.max()
    return cell * np.exp(m_max) * np.sum(np.exp(log_integrand - m_max))


def analytical_M(delta, delta_chi2, k=2):
    """Closed-form mixture e-value with a Gaussian-Fisher prior, k-dim alternative."""
    return np.exp(0.5 * delta_chi2 * delta**2 / (1 + delta**2)) / (1 + delta**2)**(k / 2)


def main():
    dr2 = load_desi_data(ROOT / 'data' / 'dr2', 'DR2')

    # 1. MLE, Fisher, and Δχ² (exact vs local-Gaussian)
    theta_mle, chi2_mle = fit_mle(dr2)
    chi2_null = chi2_at((-1.0, 0.0), dr2)
    delta_chi2 = chi2_null - chi2_mle

    F = fisher_matrix((-1.0, 0.0), dr2)
    eigvals, eigvecs = np.linalg.eigh(F)

    delta_theta = np.array(theta_mle) - np.array([-1.0, 0.0])
    delta_chi2_Fisher = delta_theta @ F @ delta_theta

    # 2. Verify the closed form against grid integration of the mixture
    verify_ratios = {}
    for d in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        M_an = analytical_M(d, delta_chi2_Fisher, k=2)
        M_num = mixture_e_gaussian_prior(dr2, np.array([-1.0, 0.0]), F, d,
                                         n_grid=60, n_sigma=5)
        verify_ratios[d] = M_num / M_an
    max_verify_err = max(abs(r - 1.0) for r in verify_ratios.values())

    # 3. Rejection band: solve M(δ) = 20 for its lower and upper roots
    def f_analytical(d):
        return analytical_M(d, delta_chi2_Fisher, k=2) - 20

    deltas_scan = np.linspace(0.05, 20, 200)
    M_scan = analytical_M(deltas_scan, delta_chi2_Fisher, k=2)

    if any(M_scan > 20):
        idx_lower = np.where(M_scan >= 20)[0][0]
        delta_star_lower = brentq(f_analytical, deltas_scan[idx_lower - 1],
                                  deltas_scan[idx_lower], xtol=1e-4)
        idx_upper = np.where(M_scan >= 20)[0][-1]
        if idx_upper < len(M_scan) - 1:
            delta_star_upper = brentq(f_analytical, deltas_scan[idx_upper],
                                      deltas_scan[idx_upper + 1], xtol=1e-4)
        else:
            delta_star_upper = np.inf

        # max of log M = ½ Δχ² δ²/(1+δ²) - (k/2) log(1+δ²) gives δ² = Δχ²/k - 1
        delta_max = np.sqrt(max(delta_chi2_Fisher / 2 - 1, 0))
        M_max = analytical_M(delta_max, delta_chi2_Fisher, k=2)

        M_num_at_lower = mixture_e_gaussian_prior(
            dr2, np.array([-1.0, 0.0]), F, delta_star_lower, n_grid=80, n_sigma=6)
        M_num_at_upper = mixture_e_gaussian_prior(
            dr2, np.array([-1.0, 0.0]), F, delta_star_upper, n_grid=80, n_sigma=6)
    else:
        delta_star_lower = delta_star_upper = delta_max = M_max = None
        M_num_at_lower = M_num_at_upper = None

    F_inv = np.linalg.inv(F)
    sigma_w0_F = np.sqrt(F_inv[0, 0])
    sigma_wa_F = np.sqrt(F_inv[1, 1])

    results = {
        'DR2_MLE': {'w0': float(theta_mle[0]), 'wa': float(theta_mle[1]),
                    'chi2_LCDM': float(chi2_null),
                    'chi2_MLE': float(chi2_mle),
                    'delta_chi2_exact': float(delta_chi2),
                    'delta_chi2_local_gaussian': float(delta_chi2_Fisher)},
        'Fisher_F': F.tolist(),
        'Fisher_eigvals': eigvals.tolist(),
        'Fisher_sigma_w0': float(sigma_w0_F),
        'Fisher_sigma_wa': float(sigma_wa_F),
        'closed_form_max_rel_error': float(max_verify_err),
        'delta_star_lower': float(delta_star_lower) if delta_star_lower is not None else None,
        'delta_star_upper': float(delta_star_upper) if delta_star_upper is not None else None,
        'delta_max': float(delta_max) if delta_max is not None else None,
        'M_max': float(M_max) if M_max is not None else None,
        'M_numerical_at_delta_star_lower': float(M_num_at_lower) if M_num_at_lower is not None else None,
        'M_numerical_at_delta_star_upper': float(M_num_at_upper) if M_num_at_upper is not None else None,
        'paper_M_DR2': 33.97,
        'narrative': (f"DESI DR2 evidence requires a test specification concentrated "
                      f"within δ* ≈ {delta_star_lower:.2f}-{delta_star_upper:.2f} "
                      f"Fisher-σ around ΛCDM to reject at the anytime-valid α=0.05 level.")
                     if delta_star_lower else 'no rejection regime',
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Delta chi^2: exact={delta_chi2:.2f}, local-Gaussian={delta_chi2_Fisher:.2f}")
    if delta_star_lower is not None:
        print(f"delta*_lower={delta_star_lower:.2f}, delta*_upper={delta_star_upper:.2f}, "
              f"delta_max={delta_max:.2f}, M_max={M_max:.3g}")
    print(f"Wrote {RESULTS_PATH}")


if __name__ == '__main__':
    main()
