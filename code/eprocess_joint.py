#!/usr/bin/env python3
"""Joint (DR1, DR2) likelihood-ratio e-process for DESI BAO (exact martingale).

The headline running mixture in eprocess_demo.py evaluates M_DR2 as the
DR2-MARGINAL mixture ratio,

    M_t^marg = [ integral L(D_t | theta) dpi ] / L(D_t | H0),

which is a sharp e-value at each fixed t (E[M_t | H0] = 1) but is NOT a
(super)martingale on the release filtration F_1 = sigma(DR1),
F_2 = sigma(DR1, DR2): the marginal ratio equals the joint Radon-Nikodym
derivative only when the cumulative mean is sufficient for the pair, which
under the one-fraction model requires C_DR2M = alpha * C_DR1M (variance
ratio 1/3 at the survey-design alpha = 1/3, NOT alpha^2 = 1/9). The
published matched-bin ratios straddle that value bin-by-bin (0.22-0.49,
median 0.318) without satisfying the matrix identity, so the martingale
property fails, mildly and in both directions. This script shows the failure
analytically (closed-form conditional mean, no MC):
E[M_2^marg | F_1] = 0.75 vs M_1 = 1.05 at the observed DR1, and the
conditional mean exceeds M_1 on a positive fraction (~59%) of H0 draws,
so the marginal process is neither a super- nor a submartingale.

This script computes the construction Ville's inequality wants directly:
the joint mixture likelihood ratio on the one-fraction model of App. B.2,

    M_2^joint = integral [ L(DR1 | theta) L(DR2 | DR1, theta) ] dpi
                / [ L(DR1 | H0) L(DR2 | DR1, H0) ],

with, per matched bin i (noise fraction alpha = 1/3, from the survey durations):

    DR1_i  = mu_1i(theta) + eps_y1[i],            eps_y1 ~ N(0, C_DR1)
    DR2_i  = mu_2i(theta) + alpha eps_y1[i] + (1 - alpha) eps_y23[i],
             eps_y23 ~ N(0, C_y23) independent of eps_y1,
    C_y23  = (C_DR2M - alpha^2 C_DR1M) / (1 - alpha)^2   (PSD-checked),

and unmatched DR2 bins independent of DR1 with their published covariance.
Writing y = eps2M_obs - alpha eps1M_obs (the innovation residual, equal to
(1 - alpha) eps_y23 under the model) the theta-vs-H0 joint log-ratio is

    logLR(theta) = d1' C1^-1 e1 - d1' C1^-1 d1 / 2                 (DR1)
                 + Dm' Qy y   - Dm' Qy Dm / 2                      (matched)
                 + du' Qu e2U - du' Qu du / 2                      (unmatched)

with d1 = mu_1(theta) - mu_1(H0), Dm = d2M - alpha d1M,
Qy = [(1-alpha)^2 C_y23]^-1, Qu = C_U^-1.

Exactness. For each theta the conditional density ratio has unit conditional
expectation (Gaussian identity: E[exp(a'y)] = exp(a' Sigma a / 2) cancels the
-Dm'QyDm/2 term when Qy = Sigma_y^-1), so E[M_2^joint | F_1] = M_1 exactly:
(M_1, M_2^joint) is a nonnegative mean-one martingale under the one-fraction
H0 and Ville's inequality applies exactly, given the model (DESI publishes
no joint (DR1, DR2) covariance; the model is the App. B.2 approximation).
The script verifies the identity algebraically per grid point (max |c_g|
~ 1e-13) rather than leaning on conditional MC means: mixture e-values are
so heavy-tailed that finite-sample conditional means sit far below their
expectations (the same estimator bias affects the marginal construction's
decile table). A single-theta control martingale - exact by construction -
is run through the same decile diagnostic to calibrate that bias.

Produces:
  - results/eprocess_joint.json
  - results/eprocess_joint_summary.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'code'))

from data_loader import load_desi_data
from cosmology import CosmologyParams, LCDM, compute_bao_predictions
from evalue_analysis import (precompute_kernels, mixture_log_e_from_residuals,
                             _build_theory_vector)
from eprocess_hierarchical_mc import build_bin_matching, nearest_psd, is_psd

W0_GRID = np.linspace(-1.5, -0.5, 30)
WA_GRID = np.linspace(-2.0, 1.0, 30)
DR2_MLE = (-0.856, -0.430)  # BAO-only MLE at fixed background (paper Sec. 3)


def build_delta_matrix(z_values, quantities, w0_grid, wa_grid):
    """Per-grid-point theory offsets from the null: delta[g] = mu(theta_g) - mu(H0)."""
    mu_null = _build_theory_vector(
        compute_bao_predictions(z_values, LCDM), z_values, quantities)
    delta = np.zeros((len(w0_grid) * len(wa_grid), len(z_values)))
    g = 0
    for w0 in w0_grid:
        for wa in wa_grid:
            cosmo = CosmologyParams(w0=w0, wa=wa)
            delta[g] = _build_theory_vector(
                compute_bao_predictions(z_values, cosmo), z_values, quantities) - mu_null
            g += 1
    return mu_null, delta


def joint_kernel(delta1, delta2, matched_dr1_idx, matched_dr2_idx, unmatched_dr2,
                 cov_dr1, cov_y23_scaled, cov_u, alpha):
    """Affine kernel for the joint log-LR: logLR = A1 e1 + Ay y + Au e2U - const/2.

    cov_y23_scaled is (1 - alpha)^2 C_y23, the conditional covariance of the
    matched DR2 block given DR1.
    """
    C1inv = np.linalg.inv(cov_dr1)
    Qy = np.linalg.inv(cov_y23_scaled)
    A1 = delta1 @ C1inv
    const1 = np.einsum('gi,ij,gj->g', delta1, C1inv, delta1)

    Dm = delta2[:, matched_dr2_idx] - alpha * delta1[:, matched_dr1_idx]
    Ay = Dm @ Qy
    consty = np.einsum('gi,ij,gj->g', Dm, Qy, Dm)

    if len(unmatched_dr2) > 0:
        Qu = np.linalg.inv(cov_u)
        du = delta2[:, unmatched_dr2]
        Au = du @ Qu
        constu = np.einsum('gi,ij,gj->g', du, Qu, du)
    else:
        Au = np.zeros((delta1.shape[0], 0))
        constu = np.zeros(delta1.shape[0])

    return dict(A1=A1, Ay=Ay, Au=Au, const1=const1, consty=consty,
                constu=constu, const=const1 + consty + constu)


def joint_log_e(eps1, y, e2u, kern):
    """Vectorised joint mixture log e-value; inputs shaped (n, B) -> (B,)."""
    log_LR = (kern['A1'] @ eps1 + kern['Ay'] @ y
              + (kern['Au'] @ e2u if kern['Au'].shape[1] else 0.0)
              - 0.5 * kern['const'][:, None])
    m = log_LR.max(axis=0)
    return m + np.log(np.mean(np.exp(log_LR - m), axis=0))


def decomposition_for_alpha(cov_dr1_m, cov_dr2_m, alpha):
    """C_y23 for a noise fraction alpha, its smallest eigenvalue and whether it is PSD.

    The returned matrix is projected onto the PSD cone, which leaves a PSD C_y23
    unchanged up to rounding; run_construction stops when C_y23 is not PSD.
    """
    cov_y23 = (cov_dr2_m - alpha ** 2 * cov_dr1_m) / max((1.0 - alpha) ** 2, 1e-12)
    min_eig, psd = is_psd(cov_y23)
    cov_y23_psd, _ = nearest_psd(cov_y23)
    return cov_y23_psd, float(min_eig), bool(psd)


def largest_admissible_alpha(cov_dr1_m, cov_dr2_m, lo=1.0 / 3.0, hi=0.75):
    """Bisect for the largest year-1 share keeping C_y23 PSD."""
    _, _, psd_lo = decomposition_for_alpha(cov_dr1_m, cov_dr2_m, lo)
    if not psd_lo:
        return float('nan')
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        _, _, ok = decomposition_for_alpha(cov_dr1_m, cov_dr2_m, mid)
        if ok:
            lo = mid
        else:
            hi = mid
    return float(lo)


def analytic_verifications(ds_dr1, ds_dr2, K1, K2, delta1, delta2, alpha,
                           cov_y23, m1_idx, m2_idx, u2_idx, cov_u, kern,
                           log_M2_joint, n_draws=20000, seed=20260712):
    """Three closed-form checks (no MC noise):

    (1) full-joint reproduction: evaluate the mixture LR on the assembled
        (n1+n2)-dim joint Gaussian without the conditional factorization;
        must reproduce log M_2^joint to numerical precision. This is the
        main correctness check (the c_g identity is a kernel-vs-sampler
        consistency invariant).
    (2) marginal conditional mean: E[M_2^marg | F_1] in closed form, at the
        observed DR1 and over H0 draws - the analytic demonstration that
        the marginal construction is neither a super- nor a submartingale.
    (3) misspecification sensitivity: E[M_2^joint | H0] when the TRUE
        year-1 share alpha_t differs from the assumed 1/3 (both published
        marginals preserved). The joint construction's fixed-time validity
        is model-conditional; the marginal's is model-free.
    """
    n1, n2 = len(ds_dr1.data), len(ds_dr2.data)
    C1M = ds_dr1.cov[np.ix_(m1_idx, m1_idx)]
    eps1_obs = ds_dr1.data - K1['mu_null']
    eps2_obs = ds_dr2.data - K2['mu_null']

    def joint_cov(alpha_t, independent=False):
        S = np.zeros((n1 + n2, n1 + n2))
        S[:n1, :n1] = ds_dr1.cov
        if independent:
            S[n1:, n1:] = ds_dr2.cov
            return S
        cy_t, _, _ = decomposition_for_alpha(
            C1M, ds_dr2.cov[np.ix_(m2_idx, m2_idx)], alpha_t)
        S22 = np.zeros((n2, n2))
        S22[np.ix_(m2_idx, m2_idx)] = alpha_t ** 2 * C1M + (1 - alpha_t) ** 2 * cy_t
        if len(u2_idx):
            S22[np.ix_(u2_idx, u2_idx)] = cov_u
        S[n1:, n1:] = S22
        cross = np.zeros((n1, n2))
        cross[np.ix_(m1_idx, m2_idx)] = alpha_t * C1M
        S[:n1, n1:] = cross
        S[n1:, :n1] = cross.T
        return S

    # --- (1) full-joint reproduction, no factorization -------------------
    S = joint_cov(alpha)
    Sinv = np.linalg.inv(S)
    D = np.hstack([delta1, delta2])
    x = np.concatenate([eps1_obs, eps2_obs])
    logLR = D @ (Sinv @ x) - 0.5 * np.einsum('gi,ij,gj->g', D, Sinv, D)
    mm = logLR.max()
    log_M_full = float(mm + np.log(np.mean(np.exp(logLR - mm))))
    full_joint = dict(log_M_full=log_M_full,
                      abs_diff_vs_factorized=float(abs(log_M_full - log_M2_joint)))

    # --- (2) closed-form E[M_2^marg | F_1] --------------------------------
    A2, const2 = K2['A'], K2['const']
    Sc = np.zeros((n2, n2))
    Sc[np.ix_(m2_idx, m2_idx)] = (1 - alpha) ** 2 * cov_y23
    if len(u2_idx):
        Sc[np.ix_(u2_idx, u2_idx)] = cov_u
    c0 = 0.5 * np.einsum('gi,ij,gj->g', A2, Sc, A2) - 0.5 * const2
    proj = alpha * A2[:, m2_idx]                     # acts on eps1[m1_idx]
    lg_obs = proj @ eps1_obs[m1_idx] + c0
    mo = lg_obs.max()
    cond_obs = float(np.exp(mo + np.log(np.mean(np.exp(lg_obs - mo)))))
    M1_obs = float(np.exp(mixture_log_e_from_residuals(eps1_obs[:, None], K1)[0]))
    rng = np.random.default_rng(seed)
    e1 = K1['L_chol'] @ rng.standard_normal((n1, n_draws))
    LG = proj @ e1[m1_idx] + c0[:, None]
    mv = LG.max(axis=0)
    cond_draws = np.exp(mv + np.log(np.mean(np.exp(LG - mv), axis=0)))
    M1_draws = np.exp(mixture_log_e_from_residuals(e1, K1))
    marginal_conditional = dict(
        E_M2marg_given_obs_DR1=cond_obs, M1_obs=M1_obs,
        frac_draws_cond_exceeds_M1=float(np.mean(cond_draws > M1_draws)),
        max_ratio=float(np.max(cond_draws / M1_draws)),
        n_draws=int(n_draws))

    # --- (3) E[M_2^joint(assumed alpha) | H0 under true alpha_t] ---------
    G = delta1.shape[0]
    B1 = kern['A1'].copy()
    B1[:, m1_idx] -= alpha * kern['Ay']
    B2 = np.zeros((G, n2))
    B2[:, m2_idx] = kern['Ay']
    if len(u2_idx):
        B2[:, u2_idx] = kern['Au']
    Bfull = np.hstack([B1, B2])

    def E_joint_under(S_t):
        q = np.einsum('gi,ij,gj->g', Bfull, S_t, Bfull)
        lg = 0.5 * q - 0.5 * kern['const']
        m = lg.max()
        return float(np.exp(min(m + np.log(np.mean(np.exp(lg - m))), 700.0)))

    misspec = {}
    for a_t in (0.15, 0.25, 1.0 / 3.0, 0.40):
        misspec[f'alpha_true={a_t:.3f}'] = E_joint_under(joint_cov(a_t))
    misspec['DR1_DR2_independent'] = E_joint_under(joint_cov(0, independent=True))

    return dict(full_joint_reproduction=full_joint,
                marginal_conditional_mean=marginal_conditional,
                E_M2joint_under_misspecified_truth=misspec)


def decile_table(M1_all, M2_all):
    """Conditional-mean table by M1 decile, with the MC standard error of
    E[M2 | bin] (heavy-tailed: |z| is the violation measure)."""
    deciles = np.quantile(M1_all, np.linspace(0, 1, 11))
    rows = []
    for k in range(10):
        lo, hi = deciles[k], deciles[k + 1]
        mask = (M1_all >= lo) & (M1_all <= hi if k == 9 else M1_all < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        m1m = float(M1_all[mask].mean())
        m2m = float(M2_all[mask].mean())
        se = float(M2_all[mask].std(ddof=1) / np.sqrt(n))
        z = (m2m - m1m) / se if se > 0 else float('nan')
        rows.append(dict(decile=k, E_M1_in_bin=m1m, E_M2_in_bin=m2m,
                         ratio=m2m / m1m, se_M2=se, z=float(z), count=n))
    return rows


def run_construction(ds_dr1, ds_dr2, alpha, K1, K2, delta1, delta2,
                     n_mc=100000, seed=20260711, analytic=False):
    n1, n2 = len(ds_dr1.data), len(ds_dr2.data)
    dr2_to_dr1, _ = build_bin_matching(ds_dr1, ds_dr2)
    matched_pairs = [(i, j) for j, i in enumerate(dr2_to_dr1) if i >= 0]
    m1_idx = np.array([i for i, j in matched_pairs], dtype=int)
    m2_idx = np.array([j for i, j in matched_pairs], dtype=int)
    u2_idx = np.array([j for j in range(n2) if dr2_to_dr1[j] < 0], dtype=int)
    n_match, n_u = len(matched_pairs), len(u2_idx)

    cov_dr1_m = ds_dr1.cov[np.ix_(m1_idx, m1_idx)]
    cov_dr2_m = ds_dr2.cov[np.ix_(m2_idx, m2_idx)]
    cov_y23, min_eig, psd_native = decomposition_for_alpha(cov_dr1_m, cov_dr2_m, alpha)
    base = dict(alpha=float(alpha), cov_y23_min_eig=min_eig,
                cov_y23_psd_native=psd_native,
                bin_matching=dict(n_matched=int(n_match), n_dr2_unmatched=int(n_u)))
    if not psd_native:
        # The one-fraction decomposition does not exist at this alpha, so the
        # construction is inadmissible (a clipped C_y23 would change the model
        # and produce near-singular conditionals).
        base['inadmissible'] = True
        return base

    # One jitter, applied before both the sampler Cholesky and the kernel
    # inverse, so Qy is exactly the inverse of the sampled covariance.
    cov_y23 = cov_y23 + 1e-14 * np.eye(n_match)
    cov_y23_scaled = (1.0 - alpha) ** 2 * cov_y23
    if n_u:
        cov_u_psd, _ = nearest_psd(ds_dr2.cov[np.ix_(u2_idx, u2_idx)])
        cov_u_psd = cov_u_psd + 1e-14 * np.eye(n_u)
    else:
        cov_u_psd = None
    kern = joint_kernel(delta1, delta2, m1_idx, m2_idx, u2_idx,
                        ds_dr1.cov, cov_y23_scaled, cov_u_psd, alpha)

    # Algebraic martingale verification: per grid point g,
    # log E[exp(Ay_g y - consty_g/2) | y ~ N(0, Sigma_y)]
    #   = (Ay_g Sigma_y Ay_g' - consty_g)/2 =: c_g, which must vanish.
    c_match = 0.5 * (np.einsum('gi,ij,gj->g', kern['Ay'], cov_y23_scaled, kern['Ay'])
                     - kern['consty'])
    if n_u:
        c_unm = 0.5 * (np.einsum('gi,ij,gj->g', kern['Au'], cov_u_psd, kern['Au'])
                       - kern['constu'])
    else:
        c_unm = np.zeros(1)
    identity_check = dict(max_abs_c_matched=float(np.abs(c_match).max()),
                          max_abs_c_unmatched=float(np.abs(c_unm).max()))
    base['identity_check'] = identity_check

    # Observed values.
    eps1_obs = ds_dr1.data - K1['mu_null']
    eps2_obs = ds_dr2.data - K2['mu_null']
    y_obs = eps2_obs[m2_idx] - alpha * eps1_obs[m1_idx]
    e2u_obs = eps2_obs[u2_idx]
    log_M1 = float(mixture_log_e_from_residuals(eps1_obs[:, None], K1)[0])
    log_M2_marg = float(mixture_log_e_from_residuals(eps2_obs[:, None], K2)[0])
    log_M2_joint = float(joint_log_e(eps1_obs[:, None], y_obs[:, None],
                                     e2u_obs[:, None], kern)[0])
    sup_log = max(log_M1, log_M2_joint)

    base['observed'] = dict(
        M_DR1=float(np.exp(log_M1)),
        log_M_DR2_joint=log_M2_joint,
        M_DR2_joint=float(np.exp(min(log_M2_joint, 700.0))),
        M_DR2_marginal=float(np.exp(log_M2_marg)),
        E_incremental=float(np.exp(min(log_M2_joint - log_M1, 700.0))),
        sup_joint=float(np.exp(min(sup_log, 700.0))),
        anytime_valid_p_joint=float(min(1.0, np.exp(-sup_log))),
    )
    if analytic:
        base['analytic'] = analytic_verifications(
            ds_dr1, ds_dr2, K1, K2, delta1, delta2, alpha, cov_y23,
            m1_idx, m2_idx, u2_idx, cov_u_psd, kern, log_M2_joint)
    if n_mc == 0:
        return base

    # Monte Carlo under the hierarchical H0.
    rng = np.random.default_rng(seed)
    L1 = K1['L_chol']
    L_y23 = np.linalg.cholesky(cov_y23)
    L_u = (np.linalg.cholesky(cov_u_psd) if n_u else None)

    # Single-theta control martingale (exact by construction) on the same
    # draws: the grid point nearest the DR2 MLE. Its decile table calibrates
    # the heavy-tail bias of the conditional-mean diagnostic.
    gg = np.argmin([(w0 - DR2_MLE[0]) ** 2 + (wa - DR2_MLE[1]) ** 2
                    for w0 in W0_GRID for wa in WA_GRID])

    batch = 5000
    M1_all = np.empty(n_mc)
    M2_all = np.empty(n_mc)
    M1c_all = np.empty(n_mc)
    M2c_all = np.empty(n_mc)
    done = 0
    t0 = time.time()
    while done < n_mc:
        bs = min(batch, n_mc - done)
        e_y1 = L1 @ rng.standard_normal(size=(n1, bs))
        e_y23 = L_y23 @ rng.standard_normal(size=(n_match, bs))
        e_u = (L_u @ rng.standard_normal(size=(n_u, bs)) if n_u
               else np.zeros((0, bs)))
        y = (1.0 - alpha) * e_y23  # y = eps2M - alpha eps1M under the model
        M1_all[done:done + bs] = np.exp(mixture_log_e_from_residuals(e_y1, K1))
        M2_all[done:done + bs] = np.exp(joint_log_e(e_y1, y, e_u, kern))
        log_m1c = kern['A1'][gg] @ e_y1 - 0.5 * kern['const1'][gg]
        log_inc = (kern['Ay'][gg] @ y - 0.5 * kern['consty'][gg]
                   + (kern['Au'][gg] @ e_u - 0.5 * kern['constu'][gg]
                      if n_u else 0.0))
        M1c_all[done:done + bs] = np.exp(log_m1c)
        M2c_all[done:done + bs] = np.exp(log_m1c + log_inc)
        done += bs
    mc_seconds = time.time() - t0

    sup_M = np.maximum(M1_all, M2_all)
    tail = {}
    for thr in [10, 20, 34, 50, 100]:
        p_emp = float(np.mean(sup_M >= thr))
        tail[str(thr)] = dict(p_emp=p_emp,
                              p_se=float(np.sqrt(p_emp * (1 - p_emp) / n_mc)),
                              ville=1.0 / thr, n_exceed=int(np.sum(sup_M >= thr)))

    base.update(
        mc=dict(N=int(n_mc), seconds=mc_seconds,
                E_M1=float(M1_all.mean()), E_M2_joint=float(M2_all.mean()),
                corr_M1_M2=float(np.corrcoef(M1_all, M2_all)[0, 1]),
                conditional_by_decile=decile_table(M1_all, M2_all),
                control_martingale_by_decile=decile_table(M1c_all, M2c_all),
                control_theta=dict(w0=float(W0_GRID[gg // len(WA_GRID)]),
                                   wa=float(WA_GRID[gg % len(WA_GRID)])),
                sup_tail=tail),
    )
    return base


def main():
    ds_dr1 = load_desi_data(REPO / 'data' / 'dr1', 'DR1')
    ds_dr2 = load_desi_data(REPO / 'data' / 'dr2', 'DR2')

    # Shared (alpha-independent) kernels and theory offsets.
    K1 = precompute_kernels(ds_dr1.z_eff, ds_dr1.quantities, ds_dr1.cov,
                            W0_GRID, WA_GRID)
    K2 = precompute_kernels(ds_dr2.z_eff, ds_dr2.quantities, ds_dr2.cov,
                            W0_GRID, WA_GRID)
    _, delta1 = build_delta_matrix(ds_dr1.z_eff, ds_dr1.quantities, W0_GRID, WA_GRID)
    _, delta2 = build_delta_matrix(ds_dr2.z_eff, ds_dr2.quantities, W0_GRID, WA_GRID)

    main_run = run_construction(ds_dr1, ds_dr2, 1.0 / 3.0, K1, K2, delta1, delta2,
                                n_mc=100000, analytic=True)
    robustness = [run_construction(ds_dr1, ds_dr2, a, K1, K2, delta1, delta2,
                                   n_mc=50000, seed=20260711 ^ int(100 * a))
                  for a in (0.25, 0.30, 0.40, 0.50)]
    # Observed-only runs near alpha_max: the construction degenerates as the
    # conditional covariance approaches singularity (disclosed, not headline).
    degeneration = [run_construction(ds_dr1, ds_dr2, a, K1, K2, delta1, delta2,
                                     n_mc=0)
                    for a in (0.44, 0.46)]

    dr2_to_dr1, _ = build_bin_matching(ds_dr1, ds_dr2)
    pairs = [(i, j) for j, i in enumerate(dr2_to_dr1) if i >= 0]
    m1_idx = np.array([i for i, j in pairs])
    m2_idx = np.array([j for i, j in pairs])
    alpha_max = largest_admissible_alpha(
        ds_dr1.cov[np.ix_(m1_idx, m1_idx)], ds_dr2.cov[np.ix_(m2_idx, m2_idx)])

    out = dict(
        description=('Joint (DR1, DR2) mixture likelihood-ratio e-process under '
                     'the App. B.2 one-fraction model; exact martingale (verified '
                     'algebraically per grid point), Ville applies exactly given '
                     'the model. Default 30x30 prior on [-1.5,-0.5]x[-2,1].'),
        largest_admissible_alpha=alpha_max,
        main=main_run,
        year_weight_robustness=robustness,
        near_alpha_max_degeneration=degeneration,
    )
    results_dir = REPO / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / 'eprocess_joint.json', 'w') as f:
        json.dump(out, f, indent=2)

    obs = main_run['observed']
    mc = main_run['mc']
    idc = main_run['identity_check']
    with open(results_dir / 'eprocess_joint_summary.md', 'w') as f:
        f.write('# Joint (DR1, DR2) likelihood-ratio e-process (exact martingale)\n\n')
        f.write('Joint mixture LR on the App. B.2 one-fraction model '
                '(noise fraction alpha = 1/3), Default 30x30 prior. '
                'Derivation in code/eprocess_joint.py docstring.\n\n')
        f.write('## Observed values\n\n')
        f.write(f'- M_DR1 = {obs["M_DR1"]:.4f} (joint = marginal at t = 1).\n')
        f.write(f'- **M_DR2^joint = {obs["M_DR2_joint"]:.2f}** '
                f'(marginal construction: {obs["M_DR2_marginal"]:.2f}). '
                'The innovation (years 2-3) carries the DR2 signal on a smaller '
                'covariance than the full DR2 average, so the exact construction '
                'strengthens the rejection.\n')
        f.write(f'- Incremental e-value E_2|1 = {obs["E_incremental"]:.2f} '
                '(cf. the plug-in MLE across-release UI value ~110: mixture < '
                'plug-in, as expected from the Occam average).\n')
        f.write(f'- sup_t M_t = {obs["sup_joint"]:.2f}; anytime-valid '
                f'p = {obs["anytime_valid_p_joint"]:.4f} '
                '(the marginal fixed-look counterpart 1/33.97 = 0.029 is not '
                'itself certified anytime-valid).\n\n')
        an = main_run['analytic']
        f.write('## Exactness (closed form, not MC)\n\n')
        f.write('Kernel-vs-sampler consistency invariant (per grid point, '
                'log E[conditional factor | F_1] must vanish):\n')
        f.write(f'- matched block: max |c_g| = {idc["max_abs_c_matched"]:.2e}; '
                f'unmatched: {idc["max_abs_c_unmatched"]:.2e}\n')
        fj = an['full_joint_reproduction']
        f.write('- direct evaluation on the assembled '
                f'{len(ds_dr1.data)+len(ds_dr2.data)}-dim joint Gaussian, without '
                'the factorization: log M agrees with the factorized computation to '
                f'|diff| = {fj["abs_diff_vs_factorized"]:.1e}.\n')
        f.write('\nSo E[M_2^joint | F_1] = M_1 exactly under the one-fraction H0: '
                'the pair is a mean-one martingale and Ville holds with no slack '
                'argument needed.\n\n')
        mcnd = an['marginal_conditional_mean']
        f.write('## Why the marginal construction is not a martingale\n\n')
        f.write('Sufficiency of the cumulative mean requires C_DR2M = alpha * '
                'C_DR1M (variance ratio 1/3 at alpha = 1/3); the published '
                'matched-bin ratios straddle it (0.22-0.49, median 0.318) without '
                'satisfying the matrix identity. Closed-form conditional mean:\n')
        f.write(f'- at the OBSERVED DR1: E[M_2^marg | F_1] = '
                f'{mcnd["E_M2marg_given_obs_DR1"]:.4f} vs M_1 = '
                f'{mcnd["M1_obs"]:.4f} (a martingale would give equality);\n')
        f.write(f'- over {mcnd["n_draws"]} H0 draws the conditional mean exceeds '
                f'M_1 on {100*mcnd["frac_draws_cond_exceeds_M1"]:.0f}% of draws '
                f'(max ratio {mcnd["max_ratio"]:.1f}): neither a super- nor a '
                'submartingale. Conditional-mean MC decile tables cannot show '
                'this either way (see control-martingale calibration below).\n\n')
        f.write('## Model-conditionality of the joint construction\n\n')
        f.write('The marginal M_DR2 = 33.97 is a '
                'valid e-value at fixed t under any joint law with the published '
                'DR2 marginal (model-free); the joint construction is exact on '
                'the filtration but conditional on the one-fraction joint model. '
                'If the true year-1 share alpha_t differs from the assumed 1/3, '
                'E[M_2^joint | H0] can exceed 1 (closed form, both published '
                'marginals preserved):\n\n')
        f.write('| true alpha_t | E[M_2^joint | H0] |\n|---|---|\n')
        for k, v in an['E_M2joint_under_misspecified_truth'].items():
            f.write(f'| {k} | {v:.3g} |\n')
        f.write('\nThe model-free marginal 33.97 is the fixed-look number; the '
                'joint construction is the anytime-valid statistic under the '
                'one-fraction model, and this table gives its null mean when the '
                'model is misspecified.\n\n')
        f.write('## Why conditional-mean MC tables mislead here\n\n')
        f.write('The conditional factor is lognormal-like with large log-variance '
                'in high-M_1 bins; finite-sample means underestimate badly. The '
                'same diagnostic applied to a single-theta control martingale '
                f'(theta = ({mc["control_theta"]["w0"]:.3f}, '
                f'{mc["control_theta"]["wa"]:.3f}), exact by construction) '
                'shows the same decay - the pattern is estimator bias, not a '
                'supermartingale violation:\n\n')
        f.write('| Decile | joint ratio (z) | control ratio (z) |\n|---|---|---|\n')
        for cj, cc in zip(mc['conditional_by_decile'],
                          mc['control_martingale_by_decile']):
            f.write(f'| {cj["decile"]} | {cj["ratio"]:.3f} ({cj["z"]:+.1f}) | '
                    f'{cc["ratio"]:.3f} ({cc["z"]:+.1f}) |\n')
        f.write('\n(z = (E[M2|bin] - E[M1|bin]) / SE; heavy-tail SEs are '
                'themselves underestimates, so |z| <~ a few is consistent '
                'with the exact identity.)\n\n')
        f.write('## Ville tail (N = %d)\n\n' % mc['N'])
        f.write('| threshold k | P(sup M >= k) | Ville 1/k |\n|---|---|---|\n')
        for k, row in mc['sup_tail'].items():
            f.write(f'| {k} | {row["p_emp"]:.5f} | {row["ville"]:.5f} |\n')
        f.write('\n## Sensitivity to the noise fraction\n\n')
        f.write('The joint construction depends on the assumed noise fraction '
                'alpha; the decomposition exists (C_y23 PSD) only for alpha '
                f'<= {alpha_max:.3f}:\n\n')
        f.write('| alpha | admissible | M_DR2^joint | E_2|1 | P(sup >= 20) |\n')
        f.write('|---|---|---|---|---|\n')
        for r in [main_run] + robustness + degeneration:
            if r.get('inadmissible'):
                f.write(f'| {r["alpha"]:.2f} | no (min eig '
                        f'{r["cov_y23_min_eig"]:+.2e}) | - | - | - |\n')
            else:
                p20 = (f'{r["mc"]["sup_tail"]["20"]["p_emp"]:.5f}'
                       if 'mc' in r else '(observed only)')
                f.write(f'| {r["alpha"]:.3f} | yes | '
                        f'{r["observed"]["M_DR2_joint"]:.3g} | '
                        f'{r["observed"]["E_incremental"]:.3g} | {p20} |\n')
        f.write('\nEvery fraction in [0.25, 0.40] keeps the rejection, with M '
                'in [65, 131]; alpha = 1/3 sits near the family minimum, which '
                'cross_release/one_fraction_family.py computes over the whole '
                'admissible range. The observed value diverges as alpha '
                'approaches alpha_max (see the 0.44/0.46 rows), where the '
                'conditional covariance becomes singular. Above alpha_max '
                '(alpha = 0.5 here) C_y23 is not PSD and the decomposition '
                'does not exist.\n')
        f.write('\n## Caveats\n\n')
        f.write('- Exactness is conditional on the one-fraction joint model '
                '(DESI publishes no joint (DR1, DR2) covariance); same '
                'approximation as App. B.2.\n')
        f.write('- The unmatched DR2 QSO DM/DH bins are treated as independent '
                'of DR1 only because DR1 published a DV value that cannot be '
                'matched quantity-for-quantity (the underlying photons overlap). '
                'The approximation is conservative at the observed data: the QSO '
                'block lowers the observed joint log-LR.\n')
        f.write('- The Lya z=2.330 block is matched to DR1 although it also '
                'contains later-year photons (inherited from App. B.2).\n')

    print(f"[joint] identity check: max|c_g| = "
          f"{idc['max_abs_c_matched']:.1e} (matched), "
          f"{idc['max_abs_c_unmatched']:.1e} (unmatched)")
    print(f"[joint] full-joint (no factorization) reproduction: |dlog| = "
          f"{an['full_joint_reproduction']['abs_diff_vs_factorized']:.1e}")
    print(f"[joint] marginal NOT a martingale: E[M2marg|F1_obs] = "
          f"{an['marginal_conditional_mean']['E_M2marg_given_obs_DR1']:.4f} "
          f"vs M1 = {an['marginal_conditional_mean']['M1_obs']:.4f}; "
          f"cond > M1 on "
          f"{100*an['marginal_conditional_mean']['frac_draws_cond_exceeds_M1']:.0f}% of H0 draws")
    print("[joint] E[M2joint|H0] under misspecified true alpha_t: "
          + ", ".join(f"{k}={v:.3g}"
                      for k, v in an['E_M2joint_under_misspecified_truth'].items()))
    print(f"[joint] M_DR1 = {obs['M_DR1']:.4f}")
    print(f"[joint] M_DR2^joint = {obs['M_DR2_joint']:.3f} "
          f"(marginal {obs['M_DR2_marginal']:.3f}); E_2|1 = {obs['E_incremental']:.3f}")
    print(f"[joint] anytime-valid p = {obs['anytime_valid_p_joint']:.4f}")
    print(f"[joint] P(sup >= 20 | H0) = {mc['sup_tail']['20']['p_emp']:.2e} "
          f"(Ville 5.0e-2); largest admissible alpha = {alpha_max:.3f}")
    print(f"[joint] wrote results/eprocess_joint.json and _summary.md")


if __name__ == '__main__':
    main()
