#!/usr/bin/env python3
"""Data-split e-values under different bin partitions (Appendix B.6).

The canonical z=1 split gives E_split = 1.43, with LRG2 (z=0.706) in the
training set. The value is governed almost entirely by which side LRG2 lands
on: an alternating partition gives E = 3.0 with LRG2 in training versus 1.9e2
with LRG2 in the test set -- a ~2-order swing -- and random 50/50 splits have
median E = 3.0. A further view of the LRG2 localisation of Section 4.2, not an
independent underpowered test.
"""
import json
import numpy as np
from pathlib import Path

from data_loader import load_desi_data
from evalue_analysis import split_evalue, split_evalue_by_indices

ROOT = Path(__file__).resolve().parent.parent


def bin_groups(z_eff, tol=1e-3):
    """Group measurement indices into redshift bins by shared z_eff.

    Returns a list of index arrays, one per bin, ordered by redshift
    (LRG2 is the third group, z_eff = 0.706).
    """
    z = np.asarray(z_eff)
    return [np.where(np.abs(z - zv) <= tol)[0] for zv in np.unique(np.round(z, 3))]


def split_variant_evalues(n_random=20):
    """Data-split test e-values for the partitions reported in Appendix B.6.

    Returns a dict with the z=1 split, the alternating-bin split with LRG2 in
    the training set and in the test set, and the median random 50/50 split E.
    """
    dr2 = load_desi_data(ROOT / 'data' / 'dr2', 'DR2')
    d, c = np.asarray(dr2.data), np.asarray(dr2.cov)
    z, q = np.asarray(dr2.z_eff), list(dr2.quantities)
    bins = bin_groups(z)
    n_bins = len(bins)

    # z=1 low/high split (LRG2 at z=0.706 falls in the low-z training set)
    _, e_z1 = split_evalue(d, c, z, q, split_z=1.0)

    def bin_split_E(train_bins):
        train = np.concatenate([bins[i] for i in train_bins])
        test = np.concatenate([bins[i] for i in range(n_bins) if i not in train_bins])
        return split_evalue_by_indices(d, c, z, q, train, test)[1].e_value

    e_alt_train = bin_split_E([0, 2, 4, 6])   # LRG2 (bin 2) in the training set
    e_alt_test = bin_split_E([1, 3, 5])       # LRG2 in the test set

    rand = []
    for seed in range(n_random):
        order = np.random.default_rng(seed).permutation(n_bins)
        rand.append(bin_split_E(sorted(order[:n_bins // 2 + 1])))

    return {
        'z1_LRG2_in_train': float(e_z1.e_value),
        'alternating_LRG2_in_train': float(e_alt_train),
        'alternating_LRG2_in_test': float(e_alt_test),
        'random_50_50_median': float(np.median(rand)),
    }


if __name__ == '__main__':
    res = split_variant_evalues()
    print(f"z=1 split E_split          = {res['z1_LRG2_in_train']:.2f}  (Appendix B.6)")
    print(f"alternating, LRG2 in train = {res['alternating_LRG2_in_train']:.2f}")
    print(f"alternating, LRG2 in test  = {res['alternating_LRG2_in_test']:.1f}")
    print(f"random 50/50 median        = {res['random_50_50_median']:.2f}")
    out_dir = ROOT / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'split_variants.json', 'w') as f:
        json.dump(res, f, indent=2)
