#!/usr/bin/env python3
"""Numerical M_DR2 for literature and physics-motivated priors (Table 1, Section 4.3).

Produces Table 1 mixture e-values:
  Narrow 143.84/2.60, Default 33.97/1.05, Wide 10.69/0.32, Ong 4.81/0.16,
  REGROW delta=1 7.6/1.50, delta=2 72.5/3.02, delta=3 297/5.25 (M_DR2/M_DR1).
And the Section 4.3 quintessence priors: freezing M_DR2 = 14.8, the freezing box evaluated
on a 320 x 320 grid (the box is small and its integrand peaks near one corner, so it needs
a fine grid), thawing M_DR2 ~ 1.1e3 (1129 on the 40 x 25 band grid)
and thawing M_DR1 ~ 15. The script prints both quintessence values and exits with an
error if the freezing value is not within 1% of the paper's 14.8 or the thawing value
within 1% of 1129. Runtime about two minutes (the 320 x 320 box dominates).

Writes results/literature_priors.json.

Exports mixture_log_e_grid and flat_box_grid, imported by compute_lrg2_drop.py.
"""
import json
import numpy as np
from pathlib import Path

from data_loader import load_desi_data
from cosmology import (CosmologyParams, LCDM, compute_bao_predictions,
                       log_likelihood)
from evalue_analysis import _build_theory_vector
from regrow_analysis import (compute_fisher_at_null, ellipse_uniform_grid,
                             mixture_log_e_from_points)

ROOT = Path(__file__).resolve().parents[1]
FREEZING_GRID_N = 320          # points per axis for the freezing box (Section 4.3 names 320 x 320)
PAPER_FREEZING_M_DR2 = 14.8    # Section 4.3
PAPER_THAWING_M_DR2 = 1129.0   # Section 4.3 prints 1.1e3; the 40 x 25 band grid below gives 1129


def mixture_log_e_grid(theta_grid, ds, log_L_null=None):
    """Mixture log E from a list of (w0, wa) points (uniform weights)."""
    if log_L_null is None:
        pred_null = compute_bao_predictions(ds.z_eff, LCDM)
        theory_null = _build_theory_vector(pred_null, ds.z_eff, ds.quantities)
        log_L_null = log_likelihood(ds.data, theory_null, ds.cov)
    log_ratios = np.empty(len(theta_grid))
    for i, (w0, wa) in enumerate(theta_grid):
        cosmo = CosmologyParams(w0=w0, wa=wa)
        pred = compute_bao_predictions(ds.z_eff, cosmo)
        theory = _build_theory_vector(pred, ds.z_eff, ds.quantities)
        log_ratios[i] = log_likelihood(ds.data, theory, ds.cov) - log_L_null
    m = log_ratios.max()
    return float(m + np.log(np.mean(np.exp(log_ratios - m))))


def flat_box_grid(w0_min, w0_max, wa_min, wa_max, n_w0=30, n_wa=30):
    """Uniform grid in a flat box."""
    w0 = np.linspace(w0_min, w0_max, n_w0)
    wa = np.linspace(wa_min, wa_max, n_wa)
    W0, WA = np.meshgrid(w0, wa, indexing='ij')
    return np.stack([W0.ravel(), WA.ravel()], axis=1)


def gaussian_grid(mu, cov, n_sample=500, seed=42):
    """Sample n points from a Gaussian."""
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mu, cov, size=n_sample)


def main():
    dr1 = load_desi_data(ROOT / 'data' / 'dr1', 'DR1')
    dr2 = load_desi_data(ROOT / 'data' / 'dr2', 'DR2')

    # Table 1 flat boxes: (name, (w0_min, w0_max), (wa_min, wa_max))
    flat_priors = [
        ('Narrow',  (-1.2, -0.8), (-1.0, 0.5)),
        ('Default', (-1.5, -0.5), (-2.0, 1.0)),
        ('Wide',    (-2.0,  0.0), (-3.0, 2.0)),
        ('Ong',     (-3.0,  1.0), (-3.0, 2.0)),
    ]
    flat_results = []
    for name, w0_rng, wa_rng in flat_priors:
        grid = flat_box_grid(*w0_rng, *wa_rng, n_w0=30, n_wa=30)
        M1 = float(np.exp(mixture_log_e_grid(grid, dr1)))
        M2 = float(np.exp(mixture_log_e_grid(grid, dr2)))
        flat_results.append({'name': name, 'M_DR1': M1, 'M_DR2': M2,
                             'sup_t': max(M1, M2),
                             'decision': 'REJECT' if max(M1, M2) >= 20 else 'no'})

    # REGROW Fisher-ellipse priors (Table 1). DR2 Fisher sets the DR2 ellipse,
    # DR1 Fisher the DR1 ellipse (paper's convention).
    I_F_dr2 = compute_fisher_at_null(dr2, h=5e-3)
    I_F_dr1 = compute_fisher_at_null(dr1, h=5e-3)

    n_pts_ellipse = 240
    regrow_results = []
    for delta in (1.0, 2.0, 3.0):
        pts_dr2, _, _, _ = ellipse_uniform_grid(I_F_dr2, delta, n_pts_ellipse)
        log_E2, _ = mixture_log_e_from_points(pts_dr2, dr2)
        pts_dr1, _, _, _ = ellipse_uniform_grid(I_F_dr1, delta, n_pts_ellipse)
        log_E1, _ = mixture_log_e_from_points(pts_dr1, dr1)
        M1, M2 = float(np.exp(log_E1)), float(np.exp(log_E2))
        regrow_results.append({'name': f'REGROW_delta_{delta}', 'M_DR1': M1, 'M_DR2': M2,
                               'sup_t': max(M1, M2),
                               'decision': 'REJECT' if max(M1, M2) >= 20 else 'no'})

    # Section 4.3: theory-motivated priors, centered on predicted regions not LCDM.

    # Quintessence freezing (tracker, SUGRA): w0 in [-0.95, -0.75], wa in [0, 0.3].
    # The box is small and the integrand peaks near its (-0.75, 0) corner, so the uniform
    # mixture is evaluated on a 320 x 320 grid: 14.8, the value Section 4.3 quotes
    # (cross_release/quintessence_resolution.py tabulates the grid-resolution study).
    freezing_grid = flat_box_grid(-0.95, -0.75, 0.0, 0.3, n_w0=FREEZING_GRID_N, n_wa=FREEZING_GRID_N)
    M_freeze = float(np.exp(mixture_log_e_grid(freezing_grid, dr2)))

    # Quintessence thawing (PNGB, linear): full Caldwell-Linder band
    # wa in [-3(1+w0), -(1+w0)] for w0 in [-1, -0.85].
    thaw_pts = []
    for w0 in np.linspace(-1.0, -0.85, 40):
        lo, hi = -3.0 * (1.0 + w0), -1.0 * (1.0 + w0)
        for wa in np.linspace(lo, hi, 25):
            thaw_pts.append((w0, wa))
    thawing_grid = np.array(thaw_pts)
    M_thaw = float(np.exp(mixture_log_e_grid(thawing_grid, dr2)))
    M_thaw_dr1 = float(np.exp(mixture_log_e_grid(thawing_grid, dr1)))

    # Round (isotropic) priors around LCDM, NOT Fisher-aligned.
    round_01_grid = flat_box_grid(-1.1, -0.9, -0.1, 0.1, n_w0=20, n_wa=20)
    M_round_01 = float(np.exp(mixture_log_e_grid(round_01_grid, dr2)))
    round_03_grid = flat_box_grid(-1.3, -0.7, -0.3, 0.3, n_w0=20, n_wa=20)
    M_round_03 = float(np.exp(mixture_log_e_grid(round_03_grid, dr2)))

    # DR2-MLE-centered Fisher prior: oracle / upper bound on the mixture e-value.
    F_inv = np.linalg.inv(I_F_dr2)
    mle = np.array([-0.8556, -0.4301])
    posterior_grid = gaussian_grid(mle, F_inv, n_sample=500, seed=42)
    M_post = float(np.exp(mixture_log_e_grid(posterior_grid, dr2)))

    all_results = list(flat_results) + list(regrow_results)
    all_results.extend([
        {'name': 'Quintessence_freezing', 'M_DR1': None, 'M_DR2': M_freeze,
         'sup_t': M_freeze,
         'decision': 'REJECT' if M_freeze >= 20 else 'no',
         'notes': f'centered (-0.85, +0.15); wa sign opposite to DR2 MLE; {FREEZING_GRID_N} x {FREEZING_GRID_N} grid'},
        {'name': 'Quintessence_thawing_CL_band', 'M_DR1': M_thaw_dr1, 'M_DR2': M_thaw,
         'sup_t': M_thaw,
         'decision': 'REJECT' if M_thaw >= 20 else 'no',
         'notes': 'wa in [-3(1+w0), -(1+w0)], w0 in [-1, -0.85]; full Caldwell-Linder band'},
        {'name': 'Round_iso_pm01', 'M_DR1': None, 'M_DR2': M_round_01,
         'sup_t': M_round_01,
         'decision': 'REJECT' if M_round_01 >= 20 else 'no',
         'notes': 'NOT Fisher-shaped; isotropic +/-0.1 around LCDM'},
        {'name': 'Round_iso_pm03', 'M_DR1': None, 'M_DR2': M_round_03,
         'sup_t': M_round_03,
         'decision': 'REJECT' if M_round_03 >= 20 else 'no',
         'notes': 'NOT Fisher-shaped; isotropic +/-0.3 around LCDM'},
        {'name': 'MLE_centered_Fisher', 'M_DR1': None, 'M_DR2': M_post,
         'sup_t': M_post,
         'decision': 'REJECT' if M_post >= 20 else 'no',
         'notes': 'Oracle prior centered at MLE; upper bound on mixture E'},
    ])

    for r in all_results:
        print(f"{r['name']:30s}  M_DR2={r['M_DR2']:10.2f}  sup_t={r['sup_t']:10.2f}  {r['decision']}")

    results_dir = ROOT / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / 'literature_priors.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSection 4.3 quintessence priors: freezing box M_DR2 = {M_freeze:.3f} on a "
          f"{FREEZING_GRID_N} x {FREEZING_GRID_N} grid (paper 14.8); thawing band M_DR2 = {M_thaw:.1f} "
          f"(paper 1.1e3), M_DR1 = {M_thaw_dr1:.2f} (paper ~15)")
    for name, val, ref in (('freezing', M_freeze, PAPER_FREEZING_M_DR2),
                           ('thawing', M_thaw, PAPER_THAWING_M_DR2)):
        if abs(val / ref - 1) > 0.01:
            raise SystemExit(f"{name} quintessence e-value {val:.3f} differs from the paper's {ref} by more than 1%")
    print("both Section 4.3 quintessence values within 1% of the paper")


if __name__ == '__main__':
    main()
