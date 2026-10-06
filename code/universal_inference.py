#!/usr/bin/env python3
"""
Prior-free e-value via universal inference (Wasserman, Ramdas, Balakrishnan 2020).

Produces the prior-free check of Section 4.1: E_UI ~ 78 for within-DR2 random
bin-splits and ~ 110 for the DR1 -> DR2 cross-prediction, with
P(E_UI >= 20 | H_0) = 0 from the synthetic-H_0 calibration. Compare against the
mixture e-value M_DR2 = 33.97, which integrates against a flat 30x30 (w0, wa)
prior; UI gives an anytime-valid e-value without specifying such a prior.

Construction:
  Split data D into (D_train, D_eval).
  Fit MLE theta_hat = argmax_theta L(D_train; theta) under the alternative.
  E_UI = L(D_eval; theta_hat) / L(D_eval; theta_null).

Validity: E[E_UI | H_0] <= 1 because for ANY fixed theta (including the
data-dependent theta_hat(D_train) when conditioned on D_train),
  E_theta_null [L(D_eval; theta_hat) / L(D_eval; theta_null) | D_train]
  = integral L(D_eval; theta_hat(D_train)) dD_eval = 1.

Two forms are used:
1. WITHIN-DR2: random bin-splits of DR2 with a conditional Gaussian likelihood
   that accounts for covariance between train/eval bins.
2. CROSS-RELEASE: fit MLE theta_hat on DR1, evaluate on (3/2)*DR2 - (1/2)*DR1,
   which is independent of DR1 under the hierarchical noise model.

Writes results/universal_inference_results.json.
"""
import sys
import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))

from data_loader import load_desi_data
from cosmology import CosmologyParams, LCDM, compute_bao_predictions
from evalue_analysis import _build_theory_vector


def gaussian_log_lik(y, mu, Sigma):
    """Multivariate Gaussian log-likelihood (-inf if Sigma is singular)."""
    n = len(y)
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        return -np.inf
    r = y - mu
    return -0.5 * (n * np.log(2 * np.pi) + logdet + r @ np.linalg.solve(Sigma, r))


def conditional_gaussian(y_train, mu_train, mu_eval, Sigma):
    """Return (mu_cond, Sigma_cond) for y_eval | y_train under N(mu, Sigma)."""
    n_train = len(y_train)
    Sigma_TT = Sigma[:n_train, :n_train]
    Sigma_EE = Sigma[n_train:, n_train:]
    Sigma_ET = Sigma[n_train:, :n_train]
    mu_cond = mu_eval + Sigma_ET @ np.linalg.solve(Sigma_TT, y_train - mu_train)
    Sigma_cond = Sigma_EE - Sigma_ET @ np.linalg.solve(Sigma_TT, Sigma_ET.T)
    return mu_cond, Sigma_cond


def split_evalue_one(idx_train, idx_eval, data, cov, z_eff, quantities,
                      bounds=((-2.5, 0.5), (-3.0, 2.0))):
    """Single-split UI: MLE theta_hat on idx_train, conditional L on idx_eval.

    Returns (log_E, (w0_mle, wa_mle)).
    """
    idx = np.concatenate([idx_train, idx_eval])
    y = data[idx]
    z = z_eff[idx]
    q = [quantities[i] for i in idx]
    Sigma = cov[np.ix_(idx, idx)]
    n_t = len(idx_train)

    y_train = y[:n_t]
    z_train = z[:n_t]
    q_train = q[:n_t]
    Sigma_train = Sigma[:n_t, :n_t]

    def neg_log_lik_train(params):
        w0, wa = params
        cosmo = CosmologyParams(w0=w0, wa=wa)
        pred = compute_bao_predictions(z_train, cosmo)
        mu = _build_theory_vector(pred, z_train, q_train)
        return -gaussian_log_lik(y_train, mu, Sigma_train)

    try:
        res = minimize(neg_log_lik_train, x0=[-1.0, 0.0],
                       method='Nelder-Mead',
                       options={'xatol': 1e-3, 'fatol': 1e-3, 'maxiter': 2000})
        w0_mle, wa_mle = res.x
        # MLE can wander wildly with few training points; clip to physical bounds.
        w0_mle = float(np.clip(w0_mle, bounds[0][0], bounds[0][1]))
        wa_mle = float(np.clip(wa_mle, bounds[1][0], bounds[1][1]))
    except Exception:
        w0_mle, wa_mle = -1.0, 0.0

    cosmo_mle = CosmologyParams(w0=w0_mle, wa=wa_mle)
    pred_mle = compute_bao_predictions(z, cosmo_mle)
    mu_mle = _build_theory_vector(pred_mle, z, q)
    pred_null = compute_bao_predictions(z, LCDM)
    mu_null = _build_theory_vector(pred_null, z, q)

    mu_cond_mle, Sigma_cond_mle = conditional_gaussian(
        y[:n_t], mu_mle[:n_t], mu_mle[n_t:], Sigma)
    mu_cond_null, Sigma_cond_null = conditional_gaussian(
        y[:n_t], mu_null[:n_t], mu_null[n_t:], Sigma)

    log_L_mle = gaussian_log_lik(y[n_t:], mu_cond_mle, Sigma_cond_mle)
    log_L_null = gaussian_log_lik(y[n_t:], mu_cond_null, Sigma_cond_null)
    return log_L_mle - log_L_null, (w0_mle, wa_mle)


def crossfit_evalue(data, cov, z_eff, quantities, n_splits=200,
                     train_frac=0.5, seed=20260527):
    """Cross-fit UI: average single-split e-values over many random splits.

    An average of e-values is an e-value (Vovk-Wang 2021), so this is valid.
    Returns dict with average, median, distribution, and per-split MLEs.
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    n_train = max(2, int(np.round(train_frac * n)))
    E_list = []
    log_E_list = []
    mle_list = []
    for s in range(n_splits):
        idx_train = np.sort(rng.choice(n, size=n_train, replace=False))
        idx_eval = np.array([i for i in range(n) if i not in idx_train])
        if len(idx_eval) < 2:
            continue
        try:
            log_E, mle = split_evalue_one(idx_train, idx_eval, data, cov,
                                           z_eff, quantities)
            if not np.isfinite(log_E):
                continue
            E_list.append(np.exp(log_E))
            log_E_list.append(log_E)
            mle_list.append(mle)
        except Exception:
            continue
    E_arr = np.array(E_list)
    log_E_arr = np.array(log_E_list)
    mle_arr = np.array(mle_list)
    return dict(
        E_mean=float(np.mean(E_arr)),
        E_median=float(np.median(E_arr)),
        log_E_median=float(np.median(log_E_arr)),
        E_p90=float(np.percentile(E_arr, 90)),
        E_max=float(np.max(E_arr)),
        N_splits_used=len(E_arr),
        E_dist=E_arr,
        log_E_dist=log_E_arr,
        mle_w0=mle_arr[:, 0],
        mle_wa=mle_arr[:, 1],
    )


def synthetic_h0_check(data, cov, z_eff, quantities, n_trials=300,
                       n_splits=50, seed=42):
    """Sanity check: under synthetic LCDM data, E[E_UI] should be <= 1.

    Returns the per-trial mean cross-fit UI e-values, shape (n_trials,).
    """
    rng = np.random.default_rng(seed)
    pred_null = compute_bao_predictions(z_eff, LCDM)
    mu_null = _build_theory_vector(pred_null, z_eff, quantities)
    L_chol = np.linalg.cholesky(cov)
    n = len(data)
    E_means = []
    for t in range(n_trials):
        d_synth = mu_null + L_chol @ rng.standard_normal(n)
        res = crossfit_evalue(d_synth, cov, z_eff, quantities,
                              n_splits=n_splits, seed=t)
        E_means.append(res['E_mean'])
    return np.array(E_means)


def cross_release_evalue(dr1, dr2):
    """Cross-release UI: fit MLE on DR1, evaluate on the DR2 part independent of DR1.

    Hierarchical model:
      DR1 ~ mu + eps_1,  Var(eps_1) = Sigma_DR1
      DR2 ~ mu + (1/3) eps_1 + (2/3) eps_23
    Hence u := (3/2) DR2 - (1/2) DR1 ~ mu + eps_23 is INDEPENDENT of DR1, with
    Var(u) = (9/4) Sigma_DR2 - (1/4) Sigma_DR1 in the shared-bin subspace.
    DR2 measurements without a DR1 counterpart are independent of DR1 already.

    Returns dict with log_E, E, and the DR1 MLE.
    """
    z_tol = 0.01
    matched = []  # (i_dr1, i_dr2) pairs
    dr2_unmatched = []
    for i2, (z2, q2) in enumerate(zip(dr2.z_eff, dr2.quantities)):
        found = False
        for i1, (z1, q1) in enumerate(zip(dr1.z_eff, dr1.quantities)):
            if abs(z1 - z2) < z_tol and q1 == q2:
                matched.append((i1, i2))
                found = True
                break
        if not found:
            dr2_unmatched.append(i2)

    def neg_ll_dr1(params):
        w0, wa = params
        cosmo = CosmologyParams(w0=w0, wa=wa)
        pred = compute_bao_predictions(dr1.z_eff, cosmo)
        mu = _build_theory_vector(pred, dr1.z_eff, dr1.quantities)
        return -gaussian_log_lik(dr1.data, mu, dr1.cov)

    res = minimize(neg_ll_dr1, x0=[-1.0, 0.0], method='Nelder-Mead',
                   options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 2000})
    w0_mle, wa_mle = res.x

    n_matched = len(matched)
    n_unmatched = len(dr2_unmatched)
    n_u = n_matched + n_unmatched
    u = np.zeros(n_u)
    u_z = np.zeros(n_u)
    u_q = []
    Sigma_DR1_match = np.zeros((n_matched, n_matched))
    Sigma_DR2_match = np.zeros((n_matched, n_matched))
    Sigma_DR2_unmatched = np.zeros((n_unmatched, n_unmatched))

    for k, (i1, i2) in enumerate(matched):
        u[k] = 1.5 * dr2.data[i2] - 0.5 * dr1.data[i1]
        u_z[k] = dr2.z_eff[i2]
        u_q.append(dr2.quantities[i2])

    if n_matched > 0:
        i1_idx = np.array([m[0] for m in matched])
        i2_idx = np.array([m[1] for m in matched])
        Sigma_DR1_match = dr1.cov[np.ix_(i1_idx, i1_idx)]
        Sigma_DR2_match = dr2.cov[np.ix_(i2_idx, i2_idx)]

    Sigma_u_match = (9.0 / 4.0) * Sigma_DR2_match - (1.0 / 4.0) * Sigma_DR1_match
    eigvals_match = np.linalg.eigvalsh(Sigma_u_match)
    if eigvals_match.min() < -1e-6:
        eigvals_match[eigvals_match < 1e-8] = 1e-8
        evecs = np.linalg.eigh(Sigma_u_match)[1]
        Sigma_u_match = evecs @ np.diag(eigvals_match) @ evecs.T

    for k_local, i2 in enumerate(dr2_unmatched):
        k = n_matched + k_local
        u[k] = dr2.data[i2]
        u_z[k] = dr2.z_eff[i2]
        u_q.append(dr2.quantities[i2])

    if n_unmatched > 0:
        i2u_idx = np.array(dr2_unmatched)
        Sigma_DR2_unmatched = dr2.cov[np.ix_(i2u_idx, i2u_idx)]
        # cov(u_match, DR2_unmatched) = (3/2) cov(DR2_match, DR2_unmatched),
        # since DR2_unmatched is independent of DR1_match.
        if n_matched > 0:
            cross = 1.5 * dr2.cov[np.ix_(i2_idx, i2u_idx)]
        else:
            cross = np.zeros((0, n_unmatched))
    else:
        cross = np.zeros((n_matched, 0))

    Sigma_u = np.zeros((n_u, n_u))
    Sigma_u[:n_matched, :n_matched] = Sigma_u_match
    Sigma_u[n_matched:, n_matched:] = Sigma_DR2_unmatched
    Sigma_u[:n_matched, n_matched:] = cross
    Sigma_u[n_matched:, :n_matched] = cross.T

    Sigma_u = 0.5 * (Sigma_u + Sigma_u.T)
    eigs = np.linalg.eigvalsh(Sigma_u)
    if eigs.min() < -1e-6:
        eigs[eigs < 1e-8] = 1e-8
        evecs = np.linalg.eigh(Sigma_u)[1]
        Sigma_u = evecs @ np.diag(eigs) @ evecs.T

    cosmo_mle = CosmologyParams(w0=w0_mle, wa=wa_mle)
    pred_mle = compute_bao_predictions(u_z, cosmo_mle)
    mu_mle_u = _build_theory_vector(pred_mle, u_z, u_q)
    pred_null = compute_bao_predictions(u_z, LCDM)
    mu_null_u = _build_theory_vector(pred_null, u_z, u_q)

    log_L_mle = gaussian_log_lik(u, mu_mle_u, Sigma_u)
    log_L_null = gaussian_log_lik(u, mu_null_u, Sigma_u)
    log_E = log_L_mle - log_L_null
    return dict(
        log_E=float(log_E),
        E=float(np.exp(log_E)),
        w0_mle=float(w0_mle),
        wa_mle=float(wa_mle),
        n_matched=n_matched,
        n_unmatched=n_unmatched,
        log_L_mle=float(log_L_mle),
        log_L_null=float(log_L_null),
    )


def main():
    dr1 = load_desi_data(ROOT / 'data' / 'dr1', 'DR1')
    dr2 = load_desi_data(ROOT / 'data' / 'dr2', 'DR2')

    # Within-DR2 cross-fit UI
    ui_dr2 = crossfit_evalue(dr2.data, dr2.cov, dr2.z_eff, dr2.quantities,
                              n_splits=400, train_frac=0.5)

    train_results = {}
    for tf in [0.3, 0.4, 0.5, 0.6, 0.7]:
        train_results[tf] = crossfit_evalue(
            dr2.data, dr2.cov, dr2.z_eff, dr2.quantities,
            n_splits=200, train_frac=tf)

    # Cross-release UI (DR1 -> DR2)
    cross_res = cross_release_evalue(dr1, dr2)

    # Synthetic-H0 calibration on DR2 structure
    h0_E_means = synthetic_h0_check(dr2.data, dr2.cov, dr2.z_eff, dr2.quantities,
                                     n_trials=200, n_splits=40, seed=2026)

    out = dict(
        within_dr2=dict(
            E_mean=ui_dr2['E_mean'],
            E_median=ui_dr2['E_median'],
            E_p90=ui_dr2['E_p90'],
            E_max=ui_dr2['E_max'],
            n_splits=ui_dr2['N_splits_used'],
        ),
        cross_release=cross_res,
        train_frac_sensitivity={
            f'{tf:.1f}': dict(E_mean=r['E_mean'], E_median=r['E_median'])
            for tf, r in train_results.items()
        },
        h0_calibration=dict(
            mean=float(h0_E_means.mean()),
            median=float(np.median(h0_E_means)),
            P_ge_20=float((h0_E_means >= 20).mean()),
            P_ge_observed=float((h0_E_means >= ui_dr2['E_mean']).mean()),
        ),
        mixture_E_comparison=33.97,
    )

    results_dir = ROOT / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / 'universal_inference_results.json', 'w') as f:
        json.dump(out, f, indent=2)

    print(f"Within-DR2 UI:   E_mean = {ui_dr2['E_mean']:.1f}  "
          f"(median {ui_dr2['E_median']:.1f}, {ui_dr2['N_splits_used']} splits)")
    print(f"Cross-release UI: E = {cross_res['E']:.1f}  "
          f"(n_matched={cross_res['n_matched']}, n_unmatched={cross_res['n_unmatched']})")
    print(f"H0 calibration:  P(E_UI >= 20 | H0) = {out['h0_calibration']['P_ge_20']:.3f}  "
          f"(sim mean {out['h0_calibration']['mean']:.3f})")


if __name__ == '__main__':
    main()
