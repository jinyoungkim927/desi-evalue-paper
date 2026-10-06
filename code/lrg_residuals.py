#!/usr/bin/env python3
"""
LRG2 vs LRG3+ELG1 residuals (the D_H/r_d residuals shown in Figure 1).

Compares DH/r_d residuals from Planck-LCDM at z=0.706 (LRG2) and
z=0.934 (LRG3+ELG1) in sigma units to check whether the deviation is
bin-localized.

Residuals:
  LRG2 DH/r_d  = -2.1 sigma,  LRG3+ELG1 DH/r_d = +0.3 sigma   (under Planck-LCDM)
  LRG2 DH/r_d  = -1.4 sigma,  LRG3+ELG1 DH/r_d = +1.0 sigma   (under DR2 w0wa MLE)

Writes results/lrg_residuals.json.
"""
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))

from data_loader import load_desi_data
from cosmology import (CosmologyParams, LCDM, compute_bao_predictions,
                       chi_squared)
from evalue_analysis import _build_theory_vector


def main():
    dr2 = load_desi_data(ROOT / 'data' / 'dr2', 'DR2')

    # LCDM predictions at Planck cosmology, residuals in sigma units.
    pred = compute_bao_predictions(dr2.z_eff, LCDM)
    theory_lcdm = _build_theory_vector(pred, dr2.z_eff, dr2.quantities)
    sigma_diag = np.sqrt(np.diag(dr2.cov))
    resid_raw = dr2.data - theory_lcdm
    resid_sigma = resid_raw / sigma_diag

    # Whitened residuals: xi = L^{-1} r are independent N(0,1) under H0.
    # sum(xi^2) must equal the directly-computed chi^2.
    L_chol = np.linalg.cholesky(dr2.cov)
    xi = np.linalg.solve(L_chol, resid_raw)
    chi2_via_xi = float(np.sum(xi**2))
    chi2_direct = chi_squared(dr2.data, theory_lcdm, dr2.cov)
    assert abs(chi2_via_xi - chi2_direct) < 1e-6, "Whitening sanity failed!"

    # Residuals under the DR2 w0wa MLE for comparison.
    DR2_MLE = CosmologyParams(w0=-0.8556, wa=-0.4301)
    pred_mle = compute_bao_predictions(dr2.z_eff, DR2_MLE)
    theory_mle = _build_theory_vector(pred_mle, dr2.z_eff, dr2.quantities)
    resid_mle = dr2.data - theory_mle
    resid_mle_sigma = resid_mle / sigma_diag

    lrg2_idx = np.where(np.isclose(dr2.z_eff, 0.706))[0]
    lrg3_idx = np.where(np.isclose(dr2.z_eff, 0.934))[0]

    def dh_idx(idxs):
        return [i for i in idxs if dr2.quantities[i] == 'DH_over_rs'][0]

    lrg2_dh = dh_idx(lrg2_idx)
    lrg3_dh = dh_idx(lrg3_idx)

    results = {
        'lcdm': {
            'LRG2_DH_over_rs_sigma': float(resid_sigma[lrg2_dh]),
            'LRG3+ELG1_DH_over_rs_sigma': float(resid_sigma[lrg3_dh]),
            'chi2': chi2_direct,
        },
        'dr2_mle': {
            'w0': DR2_MLE.w0,
            'wa': DR2_MLE.wa,
            'LRG2_DH_over_rs_sigma': float(resid_mle_sigma[lrg2_dh]),
            'LRG3+ELG1_DH_over_rs_sigma': float(resid_mle_sigma[lrg3_dh]),
            'chi2': chi_squared(dr2.data, theory_mle, dr2.cov),
        },
    }

    print("DH/r_d residuals (data - theory) / sigma:")
    print(f"  Planck-LCDM:  LRG2 = {resid_sigma[lrg2_dh]:+.1f}, "
          f"LRG3+ELG1 = {resid_sigma[lrg3_dh]:+.1f}")
    print(f"  DR2 w0wa MLE: LRG2 = {resid_mle_sigma[lrg2_dh]:+.1f}, "
          f"LRG3+ELG1 = {resid_mle_sigma[lrg3_dh]:+.1f}")

    results_dir = ROOT / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / 'lrg_residuals.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
