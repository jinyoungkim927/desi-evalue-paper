"""Monte Carlo check that the DR2 mixture-e-value sigma is Wilks-valid (Appendix B.1).

Simulates data under H0 (LCDM + Gaussian noise from the published DR2 covariance)
and confirms the empirical tail sigma_emp matches the two-sided Wilks sigma from
the observed Delta chi^2, with E[E|H0] ~ 1 and the strict-Markov bound as a
reference. At the default N_SIMS=200000 this gives sigma_emp = 3.69 ~
sigma_Wilks = 3.67, reproducing the paper's anytime-valid sigma ~ 3.70 (the exact
Delta chi^2 = 16.86 is produced by min_concentration_delta_star.py; the coarse
optimum found here is ~16.6). Set the simulation count via the N_SIMS env var.

Writes results/sigma_mc_results.json.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
from scipy.stats import norm, chi2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'code'))

from data_loader import load_desi_data
from cosmology import CosmologyParams, LCDM, compute_bao_predictions
from evalue_analysis import _build_theory_vector


def main():
    ds = load_desi_data(REPO / 'data' / 'dr2', 'DR2')
    n = len(ds.data)
    Cinv = np.linalg.inv(ds.cov)
    L = np.linalg.cholesky(ds.cov)  # noise = L @ z, z ~ N(0, I), L L^T = C

    w0_grid = np.linspace(-1.5, -0.5, 30)
    wa_grid = np.linspace(-2.0, 1.0, 30)
    G = len(w0_grid) * len(wa_grid)

    mu_null = _build_theory_vector(
        compute_bao_predictions(ds.z_eff, LCDM), ds.z_eff, ds.quantities)
    T_alt = np.zeros((G, n))
    g = 0
    for w0 in w0_grid:
        for wa in wa_grid:
            pred = compute_bao_predictions(ds.z_eff, CosmologyParams(w0=w0, wa=wa))
            T_alt[g] = _build_theory_vector(pred, ds.z_eff, ds.quantities)
            g += 1

    # Observed mixture E.
    chi2_obs_null = float((ds.data - mu_null) @ Cinv @ (ds.data - mu_null))
    r = ds.data[None, :] - T_alt
    chi2_obs_alt_arr = np.einsum('gi,ij,gj->g', r, Cinv, r)
    log_ratios_obs = -0.5 * (chi2_obs_alt_arr - chi2_obs_null)
    m_obs = np.max(log_ratios_obs)
    log_E_obs = m_obs + np.log(np.mean(np.exp(log_ratios_obs - m_obs)))
    E_obs = float(np.exp(log_E_obs))
    chi2_min = float(chi2_obs_alt_arr.min())
    delta_chi2 = chi2_obs_null - chi2_min
    sigma_wilks = norm.isf(0.5 * chi2.sf(delta_chi2, df=2))

    N_SIMS = int(os.environ.get('N_SIMS', 200_000))
    SEED = int(os.environ.get('SEED', 20260512))

    rng = np.random.default_rng(SEED)
    batch = 5_000
    n_batches = (N_SIMS + batch - 1) // batch
    done = 0
    exceed = 0
    sum_E = 0.0
    sum_E2 = 0.0
    max_E = 0.0
    sumlog2_E = 0.0
    sumlog_E = 0.0

    # Score decomposition for the vectorised MC:
    #   delta_g = mu_alt - mu_null;
    #   log L_alt - log L_null = delta_g^T Cinv eps - 0.5 delta_g^T Cinv delta_g
    delta = T_alt - mu_null[None, :]                      # (G, n)
    A = delta @ Cinv                                      # (G, n)
    const = np.einsum('gi,ij,gj->g', delta, Cinv, delta)  # (G,)

    for _ in range(n_batches):
        bs = min(batch, N_SIMS - done)
        z = rng.standard_normal(size=(n, bs))
        eps = L @ z                                  # (n, bs) noise ~ N(0, C)
        log_LR = (A @ eps) - 0.5 * const[:, None]    # (G, bs)
        m = log_LR.max(axis=0)
        log_E = m + np.log(np.mean(np.exp(log_LR - m), axis=0))
        E = np.exp(log_E)
        exceed += int(np.sum(E >= E_obs))
        sum_E += float(E.sum())
        sum_E2 += float((E**2).sum())
        sumlog_E += float(log_E.sum())
        sumlog2_E += float((log_E**2).sum())
        max_E = max(max_E, float(E.max()))
        done += bs

    p_emp = exceed / done
    mean_E = sum_E / done
    var_E = sum_E2 / done - mean_E**2
    mean_logE = sumlog_E / done
    var_logE = sumlog2_E / done - mean_logE**2
    se_p = np.sqrt(p_emp * (1 - p_emp) / done)
    ci_lo = max(0.0, p_emp - 1.96 * se_p)
    ci_hi = p_emp + 1.96 * se_p
    sigma_emp = norm.isf(0.5 * p_emp) if p_emp > 0 else float('inf')

    out = {
        "obs": {"E_mix": E_obs, "log_E": log_E_obs, "delta_chi2": delta_chi2,
                "chi2_null": chi2_obs_null, "chi2_alt_min": chi2_min},
        "mc": {"N": done, "exceed": exceed, "p_emp": p_emp, "p_emp_ci": [ci_lo, ci_hi],
               "sigma_emp_two_tail": sigma_emp, "EE_under_H0": mean_E, "VarE_under_H0": var_E,
               "ElogE": mean_logE, "VarlogE": var_logE, "max_E": max_E},
        "wilks": {"p": float(chi2.sf(delta_chi2, df=2)), "sigma_two_tail": sigma_wilks},
        "markov_bound_sigma": float(norm.isf(0.5 / E_obs)),
    }

    out_dir = REPO / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'sigma_mc_results.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"Delta chi^2 = {delta_chi2:.2f}, sigma_Wilks (two-sided) = {sigma_wilks:.2f}")
    print(f"MC N={done}: sigma_emp (two-sided) = {sigma_emp:.2f}, E[E|H0] = {mean_E:.3f}")
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
