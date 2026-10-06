#!/usr/bin/env python3
"""Table 2 per-bin diagnostics: leave-one-out and REGROW-product columns.

Two per-bin views of the DR2 BAO evidence that together make up Table 2:

  LOO column    -- hold out each redshift bin in turn, fit (w0, wa) to the
                   remaining six, and score the held-out bin (Proposition 4).
                   Average 10.17, dominated by LRG2 (z=0.706) at 55.98; dropping
                   LRG2 takes the average to 2.54.

  REGROW column -- per-bin mixture e-value M_k under the REGROW delta=2
                   Fisher-ellipse prior, with the bin-incoherent product
                   prod_k M_k = 4.46 and arithmetic mean 1.52 (joint M_DR2 = 72.5).

Writes results/loo.json and results/per_bin_regrow.json.
"""
import json
import numpy as np
from pathlib import Path

from data_loader import load_desi_data
from cosmology import CosmologyParams, LCDM, compute_bao_predictions, log_likelihood
from evalue_analysis import _build_theory_vector, loocv_evalue
from regrow_analysis import (compute_fisher_at_null, ellipse_uniform_grid,
                             mixture_log_e_from_points)
from bin_level_eprocess import group_bins

REPO = Path(__file__).resolve().parents[1]


def loo_column(ds):
    """Table 2 LOO column: per-bin leave-one-out e-values and their average."""
    r = loocv_evalue(ds.data, ds.cov, ds.z_eff, ds.quantities, verbose=False)

    print("held-out bin z_eff    E_k^LOO")
    for z in sorted(r.per_bin_e):
        print(f"  {z:>6.3f}           {r.per_bin_e[z]:8.2f}")
    print(f"LOO average            {r.e_average:8.2f}")

    out = {
        "per_bin_e": {f"{z:.3f}": r.per_bin_e[z] for z in sorted(r.per_bin_e)},
        "loo_average": r.e_average,
    }
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results" / "loo.json").write_text(json.dumps(out, indent=2))


def mixture_log_e_from_subset(theta_pts, ds, idx_subset):
    """Mixture log E over the measurement subset idx_subset of ds.

    Prior is a uniform mixture over theta_pts. Returns the float log mixture
    e-value for the subset.
    """
    data_sub = ds.data[idx_subset]
    z_sub = ds.z_eff[idx_subset]
    q_sub = [ds.quantities[i] for i in idx_subset]
    cov_sub = ds.cov[np.ix_(idx_subset, idx_subset)]

    pred_null = compute_bao_predictions(z_sub, LCDM)
    theory_null = _build_theory_vector(pred_null, z_sub, q_sub)
    log_L_null = log_likelihood(data_sub, theory_null, cov_sub)

    log_ratios = np.empty(len(theta_pts))
    for i, (w0, wa) in enumerate(theta_pts):
        cosmo = CosmologyParams(w0=w0, wa=wa)
        pred = compute_bao_predictions(z_sub, cosmo)
        theory = _build_theory_vector(pred, z_sub, q_sub)
        log_ratios[i] = log_likelihood(data_sub, theory, cov_sub) - log_L_null
    m = log_ratios.max()
    return float(m + np.log(np.mean(np.exp(log_ratios - m))))


def regrow_column(dr2):
    """Table 2 REGROW column: per-bin M_k under the REGROW delta=2 prior, the
    bin-incoherent product prod_k M_k, and the arithmetic mean."""
    # REGROW delta=2 prior from the DR2 Fisher matrix.
    I_F_dr2 = compute_fisher_at_null(dr2, h=5e-3)
    delta = 2.0
    n_pts = 240
    regrow_pts = ellipse_uniform_grid(I_F_dr2, delta, n_pts)[0]

    M_dists = np.array([(p - np.array([-1.0, 0.0])) @ I_F_dr2 @
                        (p - np.array([-1.0, 0.0])) for p in regrow_pts])
    assert np.allclose(M_dists, delta**2, atol=1e-6), "Ellipse construction failed!"

    log_M_full, _ = mixture_log_e_from_points(regrow_pts, dr2)
    M_full = float(np.exp(log_M_full))
    assert abs(M_full - 72.5) / 72.5 < 0.05, \
        f"Mismatch with paper: got {M_full}, target 72.5"

    # Per-bin M_k under REGROW delta=2.
    bin_groups = group_bins(dr2)
    bin_labels = []
    for g in bin_groups:
        ts = [dr2.tracers[i] for i in g] if dr2.tracers else [''] * len(g)
        bin_labels.append(f"{ts[0]} (z={dr2.z_eff[g[0]]:.3f})")

    log_M_k_regrow = np.zeros(len(bin_groups))
    for k, g in enumerate(bin_groups):
        log_M_k_regrow[k] = mixture_log_e_from_subset(regrow_pts, dr2, g)
    M_k_regrow = np.exp(log_M_k_regrow)

    # Bin-incoherent product allows per-bin theta; the joint mixture shares one
    # theta across bins. Under independence M_joint = int prod_k L_k(theta) pi dtheta
    # while prod_k M_k = prod_k int L_k(theta) pi dtheta, so the two differ.
    prod = float(np.exp(np.sum(log_M_k_regrow)))
    arith_mean = float(np.mean(M_k_regrow))

    out = {
        'per_bin_M_k_regrow_delta2': M_k_regrow.tolist(),
        'per_bin_labels': bin_labels,
        'per_bin_product': prod,
        'per_bin_arith_mean': arith_mean,
        'joint_M_DR2_regrow_delta2': M_full,
        'sanity_target_72p5': 72.5,
    }
    print(f"Joint M_DR2 (REGROW delta=2) = {M_full:.2f}  (target 72.5)")
    for label, mk in zip(bin_labels, M_k_regrow):
        print(f"  M_k {label:24s} = {mk:.3f}")
    print(f"product prod_k M_k = {prod:.3f}")
    print(f"arith mean (1/K) sum_k M_k = {arith_mean:.3f}")

    results_dir = REPO / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / 'per_bin_regrow.json', 'w') as f:
        json.dump(out, f, indent=2)


def main():
    dr2 = load_desi_data(REPO / 'data' / 'dr2', 'DR2')
    loo_column(dr2)
    regrow_column(dr2)


if __name__ == '__main__':
    main()
