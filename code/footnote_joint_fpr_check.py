#!/usr/bin/env python3
"""Check for the Introduction footnote on Bonferroni: the joint false-positive
rate of per-look Wilks testing (alpha = 0.05 each) at DR1 and DR2.

DR1 is nested in DR2, so the two looks are positively correlated and the
joint FPR sits below both the independent-looks value 1-(1-.05)^2 = 0.0975
and Bonferroni's worst case 0.10. This script computes it under the
one-fraction model of Appendix B.2 (alpha_1 = 1/3): MC over the joint H0,
per-release Delta chi^2 approximated by the grid maximum of 2 log LR over a
41x41 (w0, wa) grid (slightly conservative per look).

Result (N = 40000, seed 7): P(reject at DR1 or DR2) = 0.085 +/- 0.001.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'code'))

from data_loader import load_desi_data
from evalue_analysis import precompute_kernels
from eprocess_joint import decomposition_for_alpha
from eprocess_hierarchical_mc import build_bin_matching, nearest_psd


def main(n_mc=40000, seed=7):
    w0g = np.linspace(-1.6, -0.4, 41)
    wag = np.linspace(-2.5, 1.5, 41)
    ds1 = load_desi_data(REPO / 'data' / 'dr1', 'DR1')
    ds2 = load_desi_data(REPO / 'data' / 'dr2', 'DR2')
    K1 = precompute_kernels(ds1.z_eff, ds1.quantities, ds1.cov, w0g, wag)
    K2 = precompute_kernels(ds2.z_eff, ds2.quantities, ds2.cov, w0g, wag)
    dr2_to_dr1, _ = build_bin_matching(ds1, ds2)
    pairs = [(i, j) for j, i in enumerate(dr2_to_dr1) if i >= 0]
    m1 = np.array([i for i, _ in pairs])
    m2 = np.array([j for _, j in pairs])
    u2 = np.array([j for j in range(len(ds2.data)) if dr2_to_dr1[j] < 0])
    a = 1.0 / 3.0
    cy23, _, _ = decomposition_for_alpha(
        ds1.cov[np.ix_(m1, m1)], ds2.cov[np.ix_(m2, m2)], a)
    Ly23 = np.linalg.cholesky(cy23 + 1e-14 * np.eye(len(m1)))
    cu, _ = nearest_psd(ds2.cov[np.ix_(u2, u2)])
    Lu = np.linalg.cholesky(cu + 1e-14 * np.eye(len(u2)))

    rng = np.random.default_rng(seed)
    thr = 5.991  # chi2(2), alpha = 0.05
    rej1 = np.zeros(n_mc, bool)
    rej2 = np.zeros(n_mc, bool)
    d, B = 0, 4000
    while d < n_mc:
        bs = min(B, n_mc - d)
        e1 = K1['L_chol'] @ rng.standard_normal((len(ds1.data), bs))
        ey23 = Ly23 @ rng.standard_normal((len(m1), bs))
        e2 = np.zeros((len(ds2.data), bs))
        e2[m2] = a * e1[m1] + (1 - a) * ey23
        e2[u2] = Lu @ rng.standard_normal((len(u2), bs))
        dchi1 = 2 * (K1['A'] @ e1 - 0.5 * K1['const'][:, None]).max(axis=0)
        dchi2 = 2 * (K2['A'] @ e2 - 0.5 * K2['const'][:, None]).max(axis=0)
        rej1[d:d + bs] = dchi1 > thr
        rej2[d:d + bs] = dchi2 > thr
        d += bs

    pj = (rej1 | rej2).mean()
    se = np.sqrt(pj * (1 - pj) / n_mc)
    print(f"P(reject DR1) = {rej1.mean():.4f}, P(reject DR2) = {rej2.mean():.4f}")
    print(f"P(reject at DR1 OR DR2) = {pj:.4f} +/- {se:.4f} "
          f"(indep {1 - 0.95 ** 2:.4f}, Bonferroni 0.1000)")


if __name__ == '__main__':
    main()
