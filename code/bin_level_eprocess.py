#!/usr/bin/env python3
"""Per-redshift-bin mixture e-value decomposition (Appendix B.4).

The DESI DR2 covariance is block-diagonal by redshift bin, so the 7 bins are
independent under H_0. By Vovk-Wang (2021) the product of independent e-values
is itself an e-value, so

    M_per-bin(pi) = prod_{k=1..7} M_k(pi),   M_k(pi) = mixture e-value on bin k

is a valid anytime-valid e-value testing the bin-incoherent alternative (each
bin may have its OWN (w0, wa)). Comparing it with the joint headline
M_joint = 33.97 and with the LRG2-only / Bonferroni values shows whether the
DESI signal is coherent across bins or concentrated in LRG2.

Produces (Appendix B.4):
  - results/bin_level_results.json : per-bin M_k, the product M_per-bin (= 0.0018,
    far below the joint M_DR2 = 33.97 -- the signal is not bin-coherent),
    M_joint, and the LRG2-only / LRG2-Bonferroni comparison values.
"""
import sys
import json
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'code'))

from data_loader import load_desi_data
from evalue_analysis import mixture_log_e


def group_bins(dataset, z_tol=0.05):
    """Group DESI measurements by redshift bin (block-diagonal in cov).

    Returns: list of index arrays, one per redshift bin.
    """
    groups = []
    used = np.zeros(len(dataset.z_eff), dtype=bool)
    for i, z in enumerate(dataset.z_eff):
        if used[i]:
            continue
        idx = np.where(np.abs(dataset.z_eff - z) < z_tol)[0]
        idx = idx[~used[idx]]
        groups.append(idx)
        used[idx] = True
    return groups


def main():
    dr2 = load_desi_data(REPO / 'data' / 'dr2', 'DR2')
    bin_groups = group_bins(dr2)

    bin_labels = []
    for g in bin_groups:
        z_str = f"{dr2.z_eff[g[0]]:.3f}"
        tracer = dr2.tracers[g[0]] if dr2.tracers else ''
        bin_labels.append(f"{tracer} (z={z_str})")

    # Default flat prior on (w0, wa); grid resolution fixes the published numbers.
    w0_grid = np.linspace(-1.5, -0.5, 30)
    wa_grid = np.linspace(-2.0, 1.0, 30)

    log_M_k = np.zeros(len(bin_groups))
    for k, g in enumerate(bin_groups):
        log_M_k[k], _ = mixture_log_e(
            dr2.data[g], dr2.cov[np.ix_(g, g)],
            dr2.z_eff[g], [dr2.quantities[i] for i in g],
            w0_grid, wa_grid)
    M_k = np.exp(log_M_k)

    # Product over independent bins is a valid e-value (Vovk-Wang 2021).
    log_M_per_bin = float(np.sum(log_M_k))
    M_per_bin = float(np.exp(log_M_per_bin))

    log_M_joint, _ = mixture_log_e(dr2.data, dr2.cov, dr2.z_eff, dr2.quantities,
                                   w0_grid, wa_grid)
    M_joint = float(np.exp(log_M_joint))

    lrg2_k = next((k for k, lbl in enumerate(bin_labels) if 'LRG2' in lbl), None)
    if lrg2_k is None:
        lrg2_k = next((k for k, g in enumerate(bin_groups)
                       if abs(dr2.z_eff[g[0]] - 0.706) < 0.01), None)
    M_lrg2 = float(M_k[lrg2_k]) if lrg2_k is not None else None
    M_lrg2_bonferroni = M_lrg2 / len(bin_groups) if lrg2_k is not None else None

    out = {
        'bin_labels': bin_labels,
        'M_k': M_k.tolist(),
        'log_M_k': log_M_k.tolist(),
        'M_per_bin_product': M_per_bin,
        'log_M_per_bin_product': log_M_per_bin,
        'M_joint_DR2': M_joint,
        'log_M_joint_DR2': float(log_M_joint),
        'LRG2_only_M': M_lrg2,
        'LRG2_bonferroni': M_lrg2_bonferroni,
    }
    results_dir = REPO / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / 'bin_level_results.json', 'w') as f:
        json.dump(out, f, indent=2)

    print(f"M_per-bin product = {M_per_bin:.4f}  (joint M_DR2 = {M_joint:.4f})")
    if M_lrg2 is not None:
        print(f"LRG2-only M = {M_lrg2:.4f}, LRG2/Bonferroni = {M_lrg2_bonferroni:.4f}")


if __name__ == '__main__':
    main()
