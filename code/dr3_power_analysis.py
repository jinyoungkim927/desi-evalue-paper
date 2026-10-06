"""DR3 forecast of Section 4.4: what two new high-redshift bins add.

Synthesises the two new high-z BAO measurements DR3 is expected to add -- at
z=1.7 (Y4-Y5 ELG/QSO) and z=2.5 (Y4-Y5 Lyman-alpha), at projected DESI
precisions sigma_{DM/rd} = 0.7, 1.5 and sigma_{DH/rd} = 0.4, 0.4 -- and
Monte-Carlos the incremental e-value E_{3|2} under the two truths the data must
separate: the DR2 (w0, wa) MLE and LCDM. With M_DR2 = 34 fixed and
M_3 = M_DR2 * E_{3|2}, it gives

    E_{3|2} median 0.30, 90th percentile 0.93;  M_3 median 10.0
    P(M_3 >= 20) = 0.18 under w0wa-truth, 0.17 under LCDM-truth

The two probabilities are nearly equal: at the projected precision the new bins
separate LCDM from the DR2-MLE w0wa by about 1% in D_M/r_d, so they barely move
the e-value under either hypothesis (Section 4.4). The decisive improvement is
in the precision of the existing bins (cross_release/dr3_forecast.py).

Default prior on (w0, wa); the incremental DR3 data is treated as independent
of DR2, so the mixture e-value multiplies (M_3 = M_DR2 * E_{3|2}).
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from data_loader import load_desi_data
from cosmology import CosmologyParams, compute_bao_predictions, log_likelihood
from evalue_analysis import (_build_theory_vector, mixture_log_e,
                             precompute_kernels, mixture_log_e_from_residuals)

REPO = Path(__file__).resolve().parents[1]

W0_GRID = np.linspace(-1.5, -0.5, 30)   # Default flat prior on (w0, wa)
WA_GRID = np.linspace(-2.0, 1.0, 30)
LCDM_TRUTH = (-1.0, 0.0)
N_MC = 10000

# DR3's two projected new high-z BAO measurements: (z, sigma_DM/rd, sigma_DH/rd).
DR3_NEW_BINS = [(1.7, 0.7, 0.4), (2.5, 1.5, 0.4)]


def fit_dr2_mle(ds):
    """(w0, wa) maximising the DR2 BAO likelihood -- the w0waCDM truth to forecast under."""
    def neg_ll(p):
        pred = compute_bao_predictions(ds.z_eff, CosmologyParams(w0=p[0], wa=p[1]))
        theory = _build_theory_vector(pred, ds.z_eff, ds.quantities)
        return -log_likelihood(ds.data, theory, ds.cov)
    return minimize(neg_ll, x0=[-0.86, -0.43], method='Nelder-Mead',
                    options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 5000}).x


def new_bin_block(bins):
    """(z, quantities, diagonal cov) for new independent (DM, DH) pairs at each z."""
    z, q, sigma = [], [], []
    for z_eff, s_dm, s_dh in bins:
        z += [z_eff, z_eff]
        q += ['DM_over_rs', 'DH_over_rs']
        sigma += [s_dm, s_dh]
    return np.array(z), q, np.diag(np.array(sigma) ** 2)


def simulate_incremental_e(z_new, q_new, cov_new, truth, seed):
    """Monte-Carlo the incremental e-value E_{3|2} for the new data under a truth.

    Mixture e-value of the new BAO residuals against the Default-prior grid, with
    residuals drawn around the truth cosmology. Returns the E_{3|2} sample.
    """
    K = precompute_kernels(z_new, q_new, cov_new, W0_GRID, WA_GRID)
    pred_truth = compute_bao_predictions(np.array(sorted(set(z_new))),
                                         CosmologyParams(w0=truth[0], wa=truth[1]))
    shift = _build_theory_vector(pred_truth, z_new, q_new) - K['mu_null']
    rng = np.random.default_rng(seed)
    E_inc = np.empty(N_MC)
    for b in range(0, N_MC, 2000):
        eps = K['L_chol'] @ rng.standard_normal(size=(K['n'], min(2000, N_MC - b)))
        E_inc[b:b + eps.shape[1]] = np.exp(
            mixture_log_e_from_residuals(eps + shift[:, None], K))
    return E_inc


def main():
    dr2 = load_desi_data(REPO / 'data' / 'dr2', 'DR2')
    M_DR2 = np.exp(mixture_log_e(dr2.data, dr2.cov, dr2.z_eff, dr2.quantities,
                                 W0_GRID, WA_GRID)[0])
    w0wa_truth = fit_dr2_mle(dr2)
    z_new, q_new, cov_new = new_bin_block(DR3_NEW_BINS)

    E_alt = simulate_incremental_e(z_new, q_new, cov_new, w0wa_truth, seed=2026)
    E_null = simulate_incremental_e(z_new, q_new, cov_new, LCDM_TRUTH, seed=2027)
    M3_alt, M3_null = M_DR2 * E_alt, M_DR2 * E_null

    print(f"M_DR2 (Default prior)         = {M_DR2:.2f}")
    print(f"w0waCDM truth (DR2 MLE)       = ({w0wa_truth[0]:.3f}, {w0wa_truth[1]:.3f})")
    print(f"E_3|2 median (w0wa truth)     = {np.median(E_alt):.3f}")
    print(f"E_3|2 90th percentile         = {np.percentile(E_alt, 90):.3f}")
    print(f"M_3 median (w0wa truth)       = {np.median(M3_alt):.2f}")
    print(f"P(M_3 >= 20 | w0wa truth)     = {(M3_alt >= 20).mean():.3f}")
    print(f"P(M_3 >= 20 | LCDM truth)     = {(M3_null >= 20).mean():.3f}")

    out = {
        'M_DR2': float(M_DR2),
        'w0wa_truth': w0wa_truth.tolist(),
        'new_bins': DR3_NEW_BINS,
        'n_mc': N_MC,
        'E_inc_median_w0wa': float(np.median(E_alt)),
        'E_inc_p90_w0wa': float(np.percentile(E_alt, 90)),
        'M3_median_w0wa': float(np.median(M3_alt)),
        'P_M3_ge_20_w0wa': float((M3_alt >= 20).mean()),
        'P_M3_ge_20_lcdm': float((M3_null >= 20).mean()),
    }
    (REPO / 'results').mkdir(exist_ok=True)
    (REPO / 'results' / 'dr3_forecast.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
