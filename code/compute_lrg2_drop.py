#!/usr/bin/env python3
"""How much of the DR2 rejection rests on the single LRG2 bin.

Recomputes the Default-prior running mixture e-value on the full 7-bin DR2 BAO
vector and on the 6-bin vector with LRG2 removed. The headline mixture falls
from M_DR2 = 33.97 (all bins) to ~0.49 (no LRG2) -- below unity -- so the
alpha = 0.05 rejection rests entirely on LRG2 (Section 4.2, Figure 3).
"""
import copy
import json
import numpy as np
from pathlib import Path

from data_loader import load_desi_data
from literature_priors import mixture_log_e_grid, flat_box_grid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOX = (-1.5, -0.5, -2.0, 1.0)   # Default flat prior on (w0, wa)
LRG2_ZEFF = 0.706


def drop_bin(ds, z_drop=LRG2_ZEFF, tol=1e-3):
    """Return a copy of dataset ds with all measurements at z_eff == z_drop removed."""
    z = np.asarray(ds.z_eff)
    keep = np.abs(z - z_drop) > tol
    sub = copy.copy(ds)
    sub.z_eff = z[keep]
    sub.data = np.asarray(ds.data)[keep]
    sub.cov = np.asarray(ds.cov)[np.ix_(keep, keep)]
    sub.quantities = [q for q, k in zip(ds.quantities, keep) if k]
    sub.tracers = [t for t, k in zip(ds.tracers, keep) if k]
    return sub


def lrg2_drop_summary(box=DEFAULT_BOX):
    """Default-prior mixture e-value for DR2 with and without the LRG2 bin.

    Returns a dict {'M_DR2_all_bins', 'M_DR2_no_LRG2'} of mixture e-values.
    """
    dr2 = load_desi_data(ROOT / 'data' / 'dr2', 'DR2')
    grid = flat_box_grid(*box, n_w0=30, n_wa=30)
    M_all = float(np.exp(mixture_log_e_grid(grid, dr2)))
    M_no_lrg2 = float(np.exp(mixture_log_e_grid(grid, drop_bin(dr2))))
    return {'M_DR2_all_bins': M_all, 'M_DR2_no_LRG2': M_no_lrg2}


if __name__ == '__main__':
    res = lrg2_drop_summary()
    print(f"M_DR2 all bins  = {res['M_DR2_all_bins']:.2f}")
    print(f"M_DR2 no LRG2   = {res['M_DR2_no_LRG2']:.2f}")
    out_dir = ROOT / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'lrg2_drop.json', 'w') as f:
        json.dump(res, f, indent=2)
