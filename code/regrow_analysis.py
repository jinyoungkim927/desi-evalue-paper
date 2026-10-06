#!/usr/bin/env python3
"""REGROW-style minimum-effect-size prior for the DESI w0wa e-value analysis.

Produces Table 1: REGROW Fisher-ellipse priors (delta = 1, 2, 3), with flat-prior
comparison, written to results/regrow_results.json.

Implements an approximate REGROW prior in the sense of Gruenwald, de Heide &
Koolen (2024, JRSS-B), Sections 3-4, for the mixture e-value M_t that averages
the likelihood ratio L(theta)/L(LCDM) over (w0, wa):
  1. Model log L as Gaussian in (w0, wa) about the LCDM null, with curvature
     given by the Fisher matrix I_F = -d^2 log L / d theta d theta' at
     theta_0 = (w0=-1, wa=0).
  2. For Theta_1(delta) = {theta : (theta-theta_0)' I_F (theta-theta_0) >= delta^2},
     place a uniform prior on the boundary ellipse ||theta - theta_0||_F = delta.
  3. Compute the mixture e-value with the FULL non-Gaussian likelihood, so M is a
     proper e-value under H_0 = LCDM. The Gaussian approximation only defines the
     prior support.

compute_fisher_at_null, ellipse_uniform_grid, and mixture_log_e_from_points are
imported by loo_per_bin.py and literature_priors.py.
"""
import sys
import json
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'code'))

from data_loader import load_desi_data
from cosmology import (CosmologyParams, compute_bao_predictions,
                       log_likelihood)
from evalue_analysis import _build_theory_vector


def neg_log_L(w0, wa, ds):
    """Negative log-likelihood at (w0, wa) for dataset ds."""
    cosmo = CosmologyParams(w0=w0, wa=wa)
    pred = compute_bao_predictions(ds.z_eff, cosmo)
    theory = _build_theory_vector(pred, ds.z_eff, ds.quantities)
    return -log_likelihood(ds.data, theory, ds.cov)


def log_LR(w0, wa, ds, log_L_null):
    """log L(w0, wa) - log L(LCDM)."""
    cosmo = CosmologyParams(w0=w0, wa=wa)
    pred = compute_bao_predictions(ds.z_eff, cosmo)
    theory = _build_theory_vector(pred, ds.z_eff, ds.quantities)
    return log_likelihood(ds.data, theory, ds.cov) - log_L_null


def expected_chi2_at(theta, ds, theta0=(-1.0, 0.0)):
    """Fisher-metric squared distance (mu(theta)-mu(theta0))' Cinv (mu(theta)-mu(theta0)).

    Equals E_theta0[chi^2(theta0) - chi^2(theta)] in the Gaussian likelihood;
    its Hessian at theta0 gives 2 * I_Fisher.
    """
    cosmo_alt = CosmologyParams(w0=theta[0], wa=theta[1])
    mu_alt = _build_theory_vector(compute_bao_predictions(ds.z_eff, cosmo_alt),
                                  ds.z_eff, ds.quantities)
    cosmo_null = CosmologyParams(w0=theta0[0], wa=theta0[1])
    mu_null = _build_theory_vector(compute_bao_predictions(ds.z_eff, cosmo_null),
                                   ds.z_eff, ds.quantities)
    delta_mu = mu_alt - mu_null
    return float(delta_mu @ np.linalg.inv(ds.cov) @ delta_mu)


def compute_fisher_at_null(ds, h=1e-2):
    """Fisher information I_F = -d^2 log L / d theta d theta' at theta_0 = (-1, 0).

    Uses the Hessian of the EXPECTED chi^2 (the Fisher-metric squared distance,
    centred at LCDM) rather than the observed chi^2; its Hessian at LCDM is
    2 J' Cinv J = 2 I_Fisher, and I_F = (1/2) Hessian since chi^2 = -2 log L + const.

    Returns: (2, 2) np.ndarray Fisher information matrix.
    """
    w0_0, wa_0 = -1.0, 0.0

    def E(w0, wa):
        return expected_chi2_at((w0, wa), ds)

    f00 = (E(w0_0 + h, wa_0) - 2 * E(w0_0, wa_0) + E(w0_0 - h, wa_0)) / h**2
    f11 = (E(w0_0, wa_0 + h) - 2 * E(w0_0, wa_0) + E(w0_0, wa_0 - h)) / h**2
    f01 = (
        E(w0_0 + h, wa_0 + h) - E(w0_0 + h, wa_0 - h)
        - E(w0_0 - h, wa_0 + h) + E(w0_0 - h, wa_0 - h)
    ) / (4 * h**2)

    H = np.array([[f00, f01], [f01, f11]])
    return 0.5 * H


def ellipse_uniform_grid(I_F, delta, n_points=120, theta0=(-1.0, 0.0)):
    """Uniform sample on the Fisher-norm ellipse
        {theta : (theta - theta0)' I_F (theta - theta0) = delta^2}.

    Sampled uniformly in the eigenangle phi (uniform in the Mahalanobis-circular
    sense, not arc length). In the Gaussian approximation any boundary prior
    achieves the same leading-order growth, so the parameterisation is moot.

    Returns: (pts, eigvals, eigvecs, half_axes), pts of shape (n_points, 2).
    """
    eigvals, eigvecs = np.linalg.eigh(I_F)
    if np.min(eigvals) <= 0:
        raise ValueError(f"Fisher matrix not positive definite: eigvals = {eigvals}")
    half_axes = delta / np.sqrt(eigvals)

    phis = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pts_eigen = np.stack([half_axes[0] * np.cos(phis),
                          half_axes[1] * np.sin(phis)], axis=1)
    pts = pts_eigen @ eigvecs.T
    pts[:, 0] += theta0[0]
    pts[:, 1] += theta0[1]
    return pts, eigvals, eigvecs, half_axes


def mixture_log_e_from_points(theta_grid, ds):
    """log E = log mean over theta_grid of L(theta)/L(LCDM).

    Returns: (log_e, log_ratios), log_ratios of shape (n_pts,).
    """
    cosmo_null = CosmologyParams(w0=-1.0, wa=0.0)
    theory_null = _build_theory_vector(compute_bao_predictions(ds.z_eff, cosmo_null),
                                       ds.z_eff, ds.quantities)
    log_L_null = log_likelihood(ds.data, theory_null, ds.cov)

    log_ratios = np.empty(len(theta_grid))
    for i, (w0, wa) in enumerate(theta_grid):
        log_ratios[i] = log_LR(w0, wa, ds, log_L_null)
    m = np.max(log_ratios)
    log_e = m + np.log(np.mean(np.exp(log_ratios - m)))
    return log_e, log_ratios


def mixture_log_e_flat(w0_grid, wa_grid, ds):
    """Flat 2D-grid mixture log E over the (w0, wa) grid."""
    cosmo_null = CosmologyParams(w0=-1.0, wa=0.0)
    theory_null = _build_theory_vector(compute_bao_predictions(ds.z_eff, cosmo_null),
                                       ds.z_eff, ds.quantities)
    log_L_null = log_likelihood(ds.data, theory_null, ds.cov)

    log_ratios = []
    for w0 in w0_grid:
        for wa in wa_grid:
            log_ratios.append(log_LR(w0, wa, ds, log_L_null))
    log_ratios = np.array(log_ratios)
    m = np.max(log_ratios)
    return float(m + np.log(np.mean(np.exp(log_ratios - m))))


def main():
    ds_dr1 = load_desi_data(REPO / 'data' / 'dr1', 'DR1')
    ds_dr2 = load_desi_data(REPO / 'data' / 'dr2', 'DR2')

    # Fisher information at LCDM (DR2 is the richer constraint).
    I_F_dr2 = compute_fisher_at_null(ds_dr2, h=5e-3)
    I_F_dr1 = compute_fisher_at_null(ds_dr1, h=5e-3)
    eigvals_dr2, eigvecs_dr2 = np.linalg.eigh(I_F_dr2)
    one_sigma_axes = 1.0 / np.sqrt(eigvals_dr2)

    # Flat priors for comparison.
    flat_priors = [
        ('Narrow',  np.linspace(-1.2, -0.8, 30), np.linspace(-1.0, 0.5, 30)),
        ('Default', np.linspace(-1.5, -0.5, 30), np.linspace(-2.0, 1.0, 30)),
        ('Wide',    np.linspace(-2.0,  0.0, 30), np.linspace(-3.0, 2.0, 30)),
        ('Ong',     np.linspace(-3.0,  1.0, 30), np.linspace(-3.0, 2.0, 30)),
    ]
    flat_results = []
    for name, w0g, wag in flat_priors:
        M1 = float(np.exp(mixture_log_e_flat(w0g, wag, ds_dr1)))
        M2 = float(np.exp(mixture_log_e_flat(w0g, wag, ds_dr2)))
        sup_t = max(M1, M2)
        flat_results.append({
            'name': name,
            'w0_range': [float(w0g.min()), float(w0g.max())],
            'wa_range': [float(wag.min()), float(wag.max())],
            'M_DR1': M1, 'M_DR2': M2, 'sup_t': float(sup_t),
            'decision': "REJECT" if sup_t >= 20 else "do not reject",
        })

    # REGROW-style prior: uniform on the DR2 Fisher ellipse for delta in {1, 2, 3}.
    deltas = [1.0, 2.0, 3.0]
    n_points = 120
    regrow_results = []
    for delta in deltas:
        pts_dr2, eigvals, eigvecs, half_axes = ellipse_uniform_grid(
            I_F_dr2, delta, n_points=n_points
        )

        theta0 = np.array([-1.0, 0.0])
        Mdist_sq = np.array([(p - theta0) @ I_F_dr2 @ (p - theta0) for p in pts_dr2])
        assert np.allclose(Mdist_sq, delta**2, atol=1e-6), \
            f"Ellipse points not on Fisher-shell at delta={delta}"

        # Prior fixed by the DR2 Fisher; applying it to DR1 is the consistency check.
        log_E_dr1, _ = mixture_log_e_from_points(pts_dr2, ds_dr1)
        log_E_dr2, _ = mixture_log_e_from_points(pts_dr2, ds_dr2)
        M1, M2 = float(np.exp(log_E_dr1)), float(np.exp(log_E_dr2))
        sup_t = max(M1, M2)

        regrow_results.append({
            'delta': delta,
            'GRO_value_at_boundary_Wilks': float(0.5 * delta**2),
            'half_axis_w0_eigen': float(half_axes[0]),
            'half_axis_wa_eigen': float(half_axes[1]),
            'eigvals_I_F_DR2': eigvals.tolist(),
            'M_DR1': M1,
            'M_DR2': M2,
            'sup_t': float(sup_t),
            'decision': "REJECT" if sup_t >= 20 else "do not reject",
            'log_M_DR2': float(log_E_dr2),
            'n_boundary_points': n_points,
        })

    # DR2 MLE and its Fisher distance from LCDM.
    from scipy.optimize import minimize
    res = minimize(lambda p: neg_log_L(p[0], p[1], ds_dr2),
                   x0=[-0.85, -0.5], method='Nelder-Mead')
    w0_mle, wa_mle = res.x
    delta_mle = np.array([w0_mle - (-1.0), wa_mle - 0.0])
    mle_mahalanobis = float(np.sqrt(delta_mle @ I_F_dr2 @ delta_mle))

    output = {
        'meta': {
            'description': 'REGROW-approximate prior (uniform on Fisher ellipse) for DESI BAO w0wa e-value',
            'reference': 'Gruenwald, de Heide & Koolen (2024, JRSS-B), Sections 3-4',
            'approximation_notes': [
                'Fisher metric defines Theta_1(delta) = {theta : ||theta-theta_0||_F >= delta}',
                'Under the Wilks/Gaussian approximation, the GROW-optimal prior on Theta_1(delta) concentrates on the boundary ellipse',
                'We use a uniform-in-eigenangle prior on the boundary; rotation-invariant in the Mahalanobis metric',
                'The mixture E uses the FULL non-Gaussian likelihood, so it is a proper e-value under H0 regardless of Gaussian-approximation accuracy',
                'Strict REGROW (Theorem 1, eq 25-27) requires a minimax over W1 and W0; we use the boundary-restricted W1*-GROW form',
            ],
        },
        'fisher_information_DR2': {
            'I_F': I_F_dr2.tolist(),
            'eigenvalues': np.linalg.eigvalsh(I_F_dr2).tolist(),
            'eigenvectors_columns': eigvecs_dr2.tolist(),
            'half_axes_1sigma': one_sigma_axes.tolist(),
            'note': 'I_F = -d^2 log L / d(w0,wa)^2 at LCDM, via expected chi^2 Hessian',
        },
        'fisher_information_DR1': {
            'I_F': I_F_dr1.tolist(),
            'eigenvalues': np.linalg.eigvalsh(I_F_dr1).tolist(),
        },
        'DR2_MLE': {
            'w0': float(w0_mle), 'wa': float(wa_mle),
            'fisher_distance_from_LCDM': mle_mahalanobis,
        },
        'flat_priors': flat_results,
        'regrow_priors': regrow_results,
    }

    results_dir = REPO / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / 'regrow_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Table 1: REGROW Fisher-ellipse priors -> {out_path}")
    print(f"DR2 MLE Fisher-distance from LCDM: delta_MLE = {mle_mahalanobis:.2f} sigma_F")
    for r in regrow_results:
        print(f"  delta={r['delta']:.0f}: M_DR1={r['M_DR1']:.2f}  M_DR2={r['M_DR2']:.2f}  "
              f"sup_t={r['sup_t']:.2f}  [{r['decision']}]")


if __name__ == '__main__':
    main()
