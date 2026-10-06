#!/usr/bin/env python3
"""
Anytime-valid e-process demonstration for DESI BAO releases.

Constructs an e-process {M_t} = running mixture e-value (Bayes factor with a
flat 30x30 grid prior on (w0,wa)) over data releases t = DR1, DR2, (forecast DR3),
and its prior sensitivity (Table 1).

Produces (paper results):
- M_DR1 = 1.05, M_DR2 = 33.97 (observed running mixture e-values)
- DR3 forecast with two new high-redshift bins (Section 4.4): M3 median ~10 under
  either truth, P(M3 >= 20) = 0.18 (w0waCDM truth) / 0.17 (LCDM truth)
- Numerical results -> results/eprocess_results.json
"""
import sys
import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'code'))

from data_loader import load_desi_data
from cosmology import (CosmologyParams, LCDM, compute_bao_predictions,
                       log_likelihood)
from evalue_analysis import (_build_theory_vector, mixture_log_e,
                             precompute_kernels, mixture_log_e_from_residuals)


def sigma_sweep_forecast(mu_lcdm_new, mu_w0wa_new, cov_new, z_new, quant_new,
                         w0_grid, wa_grid, M_DR2, sigmas):
    """Deterministic DR3 forecast parametrised by traditional Wilks significance.

    A hypothetical DR3 result is placed at the LCDM prediction plus a deviation
    along the DR2-MLE (w0wa) direction, scaled so its Wilks chi^2 against LCDM on
    the new bins equals sigma^2 (sign = direction: + toward w0wa, - toward LCDM).
    The incremental mixture e-value E_{3|2} (Default prior) and the running
    M_3 = M_DR2 * E_{3|2}, sup = max(M_DR2, M_3) follow with no Monte Carlo.
    """
    direction = mu_w0wa_new - mu_lcdm_new
    Cinv = np.linalg.inv(cov_new)
    nat_chi2 = float(direction @ Cinv @ direction)   # chi^2 if DR3 lands at the w0wa MLE
    nat_sigma = float(np.sqrt(nat_chi2))
    rows = []
    for s in sigmas:
        data = mu_lcdm_new + (s / nat_sigma) * direction
        log_E, _ = mixture_log_e(data, cov_new, z_new, quant_new, w0_grid, wa_grid)
        E = float(np.exp(log_E))
        M3 = M_DR2 * E
        rows.append(dict(sigma=float(s), E_inc=E, M3=float(M3),
                         sup=float(max(M_DR2, M3))))
    return dict(sigmas=[float(s) for s in sigmas], nat_sigma=nat_sigma, rows=rows)


def existing_bins_forecast(ds_dr2, w0_mle, wa_mle, w0_grid, wa_grid, factors):
    """Forecast: re-measure existing bins (esp. LRG2) at improved precision.

    Unlike two new high-z bins, tightening the LRG2 (z~0.706) covariance block
    separates the two truths: data at the w0wa-MLE mean give a steadily larger
    mixture e-value as precision improves, while data at the LCDM mean stay flat.
    Deterministic (data placed at each truth's mean); no Monte Carlo.
    """
    mle = CosmologyParams(w0=w0_mle, wa=wa_mle)
    mu_lcdm = _build_theory_vector(compute_bao_predictions(ds_dr2.z_eff, LCDM),
                                   ds_dr2.z_eff, ds_dr2.quantities)
    mu_w0wa = _build_theory_vector(compute_bao_predictions(ds_dr2.z_eff, mle),
                                   ds_dr2.z_eff, ds_dr2.quantities)
    lrg2 = np.where(np.abs(ds_dr2.z_eff - 0.706) < 0.01)[0]

    def evalue_at(mu, cov):
        log_E, _ = mixture_log_e(mu, cov, ds_dr2.z_eff, ds_dr2.quantities,
                                 w0_grid, wa_grid)
        return float(np.exp(log_E))

    rows = []
    for f in factors:
        cov_lrg2 = ds_dr2.cov.copy()
        cov_lrg2[np.ix_(lrg2, lrg2)] /= f ** 2
        rows.append(dict(factor=float(f), bins='LRG2',
                         E_lcdm_truth=evalue_at(mu_lcdm, cov_lrg2),
                         E_w0wa_truth=evalue_at(mu_w0wa, cov_lrg2)))
    cov_all = ds_dr2.cov / (2.0 ** 2)   # all bins tightened by 2x, for contrast
    rows.append(dict(factor=2.0, bins='all',
                     E_lcdm_truth=evalue_at(mu_lcdm, cov_all),
                     E_w0wa_truth=evalue_at(mu_w0wa, cov_all)))
    return dict(lrg2_indices=[int(i) for i in lrg2], rows=rows)


def main():
    ds_dr1 = load_desi_data(REPO / 'data' / 'dr1', 'DR1')
    ds_dr2 = load_desi_data(REPO / 'data' / 'dr2', 'DR2')

    # The 30x30 default prior on (w0, wa)
    w0_grid = np.linspace(-1.5, -0.5, 30)
    wa_grid = np.linspace(-2.0, 1.0, 30)

    # M_DR1 and M_DR2 with the SAME prior
    log_E_dr1, _ = mixture_log_e(ds_dr1.data, ds_dr1.cov, ds_dr1.z_eff,
                                 ds_dr1.quantities, w0_grid, wa_grid)
    log_E_dr2, _ = mixture_log_e(ds_dr2.data, ds_dr2.cov, ds_dr2.z_eff,
                                 ds_dr2.quantities, w0_grid, wa_grid)
    M_DR1 = float(np.exp(log_E_dr1))
    M_DR2 = float(np.exp(log_E_dr2))

    # Verify supermartingale property under H0 (joint MC). The MC treats DR1 and
    # DR2 noise as independent under H0, which is conservative for the sup_t bound.
    K1 = precompute_kernels(ds_dr1.z_eff, ds_dr1.quantities, ds_dr1.cov, w0_grid, wa_grid)
    K2 = precompute_kernels(ds_dr2.z_eff, ds_dr2.quantities, ds_dr2.cov, w0_grid, wa_grid)

    N_MC = 50000
    rng = np.random.default_rng(20260515)
    batch = 5000
    n_batches = (N_MC + batch - 1) // batch
    M1_all = np.empty(N_MC)
    M2_all = np.empty(N_MC)
    done = 0
    for b in range(n_batches):
        bs = min(batch, N_MC - done)
        z1 = rng.standard_normal(size=(K1['n'], bs))
        z2 = rng.standard_normal(size=(K2['n'], bs))
        eps1 = K1['L_chol'] @ z1  # (n_dr1, bs) ~ N(0, cov_dr1)
        eps2 = K2['L_chol'] @ z2  # (n_dr2, bs) ~ N(0, cov_dr2)
        M1_all[done:done+bs] = np.exp(mixture_log_e_from_residuals(eps1, K1))
        M2_all[done:done+bs] = np.exp(mixture_log_e_from_residuals(eps2, K2))
        done += bs

    sup_M = np.maximum(M1_all, M2_all)
    sup_obs = max(M_DR1, M_DR2)

    # Forecast DR3: ADDITIONAL INDEPENDENT data appended to DR2.
    z_new_full = np.array([1.7, 1.7, 2.5, 2.5])
    quant_new = ['DM_over_rs', 'DH_over_rs', 'DM_over_rs', 'DH_over_rs']

    # Forecasted Y5 precision for two new bins:
    # z=1.7 (ELG3 extension) ~ DR2 ELG2 precision; z=2.5 (Lya) cross-correlation.
    sigma_DM_17 = 0.7
    sigma_DH_17 = 0.4
    sigma_DM_25 = 1.5
    sigma_DH_25 = 0.4
    cov_new = np.diag([sigma_DM_17**2, sigma_DH_17**2, sigma_DM_25**2, sigma_DH_25**2])

    def neg_ll_dr2(p):
        w0, wa = p
        cosmo = CosmologyParams(w0=w0, wa=wa)
        pred = compute_bao_predictions(ds_dr2.z_eff, cosmo)
        theory = _build_theory_vector(pred, ds_dr2.z_eff, ds_dr2.quantities)
        return -log_likelihood(ds_dr2.data, theory, ds_dr2.cov)
    res = minimize(neg_ll_dr2, x0=[-0.85, -0.5], method='Nelder-Mead')
    w0_dr2_mle, wa_dr2_mle = res.x

    truth_w0wa = CosmologyParams(w0=w0_dr2_mle, wa=wa_dr2_mle)
    pred_lcdm_new = compute_bao_predictions(np.array([1.7, 2.5]), LCDM)
    mu_lcdm_new = _build_theory_vector(pred_lcdm_new, z_new_full, quant_new)
    pred_w0wa_new = compute_bao_predictions(np.array([1.7, 2.5]), truth_w0wa)
    mu_w0wa_new = _build_theory_vector(pred_w0wa_new, z_new_full, quant_new)

    # Incremental e-value E_3|2 = mixture E on the new data only, same flat prior.
    # Independent of DR2 by assumption, so the product E_2 * E_3|2 is a valid e-value.
    K3 = precompute_kernels(z_new_full, quant_new, cov_new, w0_grid, wa_grid)

    N_FORECAST = 20000
    forecast_results = {}
    for label, mu_truth in [('LCDM_truth', mu_lcdm_new), ('w0waCDM_truth', mu_w0wa_new)]:
        rng = np.random.default_rng(2026051501 + (0 if 'LCDM' in label else 1))
        M_increments = np.empty(N_FORECAST)
        for b in range(0, N_FORECAST, 1000):
            bs = min(1000, N_FORECAST - b)
            z = rng.standard_normal(size=(K3['n'], bs))
            eps = K3['L_chol'] @ z
            # Shift centred residuals to the truth's deviation from H0.
            eps_shifted = eps + (mu_truth - K3['mu_null'])[:, None]
            log_E_inc = mixture_log_e_from_residuals(eps_shifted, K3)
            M_increments[b:b+bs] = np.exp(log_E_inc)
        # Multiplicative running e-value M_3 = M_2 * E_3|2
        M3_dist = M_DR2 * M_increments
        forecast_results[label] = dict(
            E_inc_mean=float(np.mean(M_increments)),
            E_inc_median=float(np.median(M_increments)),
            E_inc_p90=float(np.percentile(M_increments, 90)),
            M3_median=float(np.median(M3_dist)),
            M3_p90=float(np.percentile(M3_dist, 90)),
            P_M3_ge_20=float(np.mean(M3_dist >= 20)),
            P_M3_ge_100=float(np.mean(M3_dist >= 100)),
        )

    # Deterministic DR3 forecast parametrised by traditional Wilks sigma, and the
    # existing-bins (LRG2) precision-improvement forecast that separates the truths.
    sigma_sweep = sigma_sweep_forecast(
        mu_lcdm_new, mu_w0wa_new, cov_new, z_new_full, quant_new,
        w0_grid, wa_grid, M_DR2,
        sigmas=[round(float(s), 3) for s in np.linspace(-2, 4, 61)])
    existing_bins = existing_bins_forecast(
        ds_dr2, w0_dr2_mle, wa_dr2_mle, w0_grid, wa_grid, factors=[1, 1.5, 2, 3])

    # Prior sensitivity: same e-process under different priors.
    prior_specs = [
        ('Narrow',  np.linspace(-1.2, -0.8, 30), np.linspace(-1.0, 0.5, 30)),
        ('Default', np.linspace(-1.5, -0.5, 30), np.linspace(-2.0, 1.0, 30)),
        ('Wide',    np.linspace(-2.0,  0.0, 30), np.linspace(-3.0, 2.0, 30)),
        ('Ong',     np.linspace(-3.0,  1.0, 30), np.linspace(-3.0, 2.0, 30)),
    ]
    prior_M1 = []
    prior_M2 = []
    prior_decisions = []
    for name, w0g, wag in prior_specs:
        log_M1, _ = mixture_log_e(ds_dr1.data, ds_dr1.cov, ds_dr1.z_eff,
                                  ds_dr1.quantities, w0g, wag)
        log_M2, _ = mixture_log_e(ds_dr2.data, ds_dr2.cov, ds_dr2.z_eff,
                                  ds_dr2.quantities, w0g, wag)
        M1 = np.exp(log_M1); M2 = np.exp(log_M2)
        decision = "REJECT" if max(M1, M2) >= 20 else "do not reject"
        prior_M1.append(M1); prior_M2.append(M2); prior_decisions.append(decision)

    results = dict(
        observed=dict(
            M_DR1=M_DR1,
            log_M_DR1=float(log_E_dr1),
            M_DR2=M_DR2,
            log_M_DR2=float(log_E_dr2),
            sup=float(sup_obs),
            anytime_valid_p=float(1 / sup_obs),
        ),
        mc_check=dict(
            N=N_MC,
            E_M1=float(np.mean(M1_all)),
            E_M2=float(np.mean(M2_all)),
            E_diff=float(np.mean(M2_all - M1_all)),
            Var_M1=float(np.var(M1_all)),
            Var_M2=float(np.var(M2_all)),
            P_sup_ge_20=float(np.mean(sup_M >= 20)),
            P_sup_ge_34=float(np.mean(sup_M >= 34)),
        ),
        forecast=forecast_results,
        sigma_sweep=sigma_sweep,
        existing_bins_forecast=existing_bins,
        DR2_MLE=dict(w0=float(w0_dr2_mle), wa=float(wa_dr2_mle)),
        prior_sensitivity=[
            dict(name=name, M_DR1=float(m1), M_DR2=float(m2),
                 sup=float(max(m1, m2)), decision=dec)
            for (name, _, _), m1, m2, dec in zip(prior_specs, prior_M1, prior_M2, prior_decisions)
        ],
    )
    results_dir = REPO / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / 'eprocess_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"M_DR1 = {M_DR1:.2f}  M_DR2 = {M_DR2:.2f}  sup = {sup_obs:.2f}  "
          f"anytime-valid p = {1 / sup_obs:.4f}")
    for label, fr in forecast_results.items():
        print(f"DR3 forecast [{label}]: M3 median = {fr['M3_median']:.1f}  "
              f"P(M3>=20) = {fr['P_M3_ge_20']:.2f}")

    print("sigma-sweep DR3 (Default prior):  sigma -> E_inc, M3, sup")
    for r in sigma_sweep['rows']:
        if abs(r['sigma'] - round(r['sigma'])) < 1e-6:
            print(f"  sigma={r['sigma']:+.0f}: E_inc={r['E_inc']:.3f}  "
                  f"M3={r['M3']:.2f}  sup={r['sup']:.2f}")
    print(f"  (w0wa-MLE prediction on the new bins = {sigma_sweep['nat_sigma']:.2f} sigma)")
    print("existing-bins forecast:  factor, bins -> E(LCDM truth), E(w0wa truth)")
    for r in existing_bins['rows']:
        print(f"  f={r['factor']:.1f} {r['bins']:>4}: "
              f"E_lcdm={r['E_lcdm_truth']:.3f}  E_w0wa={r['E_w0wa_truth']:.3f}")


if __name__ == '__main__':
    main()
