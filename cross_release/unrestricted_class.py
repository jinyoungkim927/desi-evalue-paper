#!/usr/bin/env python3
"""The unrestricted class of joint laws: worst and best case of the exact combined DR1+DR2
statistic over the Frechet class of the two published covariances (Section 4.1, Appendix B.2).

Question. Over ALL joint Gaussian laws for (X1, X2) that are
consistent with the published DR1 and DR2 marginals, i.e. any cross-covariance
K = Cov(X1_matched, X2_matched) (11 x 11) such that [[C1, K], [K^T, C2]] is positive
semi-definite, what are the infimum and the supremum of the exact combined statistic

    M(K) = mean_g  L(X1 | theta_g) L(y | theta_g) / [ L(X1 | H0) L(y | H0) ],
    y    = X2_m - K^T C1^{-1} X1_m          (the DR2 innovation given DR1),

eq. (B.3) of the paper with K = alpha C1 replaced by an arbitrary admissible K?  As in the
paper's construction (code/eprocess_joint.py, cross_release/structured_families.py), the unmatched
DR2 QSO (D_M, D_H) term is independent of DR1 and the DR1 QSO D_V is scored in the DR1 factor.

Parametrisation. Write C1 = L1^2, C2 = L2^2 (symmetric square roots; block-diagonal by bin
because both covariances are). Every admissible K is K = L1 R L2 with ||R||_op <= 1, and the
singular values of R are the canonical correlations between the DR1 and the DR2 vectors. In
whitened coordinates e~1 = L1^{-1} e1_m, e~2 = L2^{-1} e2_m the innovation is
y~ = e~2 - R^T e~1 with covariance I - R^T R, and the mean shift of grid point g is
d~_g = D~2_g - D~1_g R.  The log-likelihood-ratio of grid point g is

    L_g(R) = const_g + d~_g (I - R^T R)^{-1} y~ - d~_g (I - R^T R)^{-1} d~_g^T / 2,

const_g holding the DR1 factor (all 12 DR1 quantities) and the unmatched DR2 term.

Computed here (set-up and functions in unrestricted_class_core.py)
  1. validation: alpha = 1/3 -> 66.709, alpha = 0.3097 -> 64.735, DESI consistency-check
     correlations in regression form -> 28.894 (anchors and structured_families), and a check
     of the analytic gradient against finite differences;
  2. the PSD boundary: an explicit rank-one R inside one bin with y~.v = 0 drives M to 0, and
     an explicit rank-one R drives M to +infinity; both live inside a single bin, so inf = 0 and
     sup = +infinity hold for the full class and for the block-diagonal-by-bin class;
  3. the capped classes ||R||_op <= rho (no linear combination of DR1 quantities correlated with
     any linear combination of DR2 quantities beyond rho): inf and sup by projected gradient
     descent with analytic gradients, Barzilai-Borwein steps, Armijo backtracking and multiple
     starts, full and block-diagonal, for a ladder of caps; the scalar family's own cap is
     alpha/alpha_max, so alpha = 1/3 corresponds to rho = 0.709;
  4. two further structured points: the matrix nested rule K = C2 (Cov(X1, X2) = Var(X2), the
     efficient-estimator identity for strictly nested data) and the per-quantity diagonal
     family alpha_i in [0, C2_ii/C1_ii] (regression form, 11 parameters).

Writes results/numbers.json["unrestricted_class"] (value / definition / script) and
results/unrestricted_class.log.
Run (from the repository root, 5 to 12 minutes): python cross_release/unrestricted_class.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unrestricted_class_core import *   # noqa: E402,F401,F403

THIS = 'cross_release/unrestricted_class.py'
t_start = time.time()
rng = np.random.default_rng(20261005)
LOG = []
RES = {}


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    LOG.append(s)


def rec(key, value, definition):
    RES[key] = dict(value=value, definition=definition, script=THIS)


# ------------------------------------------------------------------ 1. validation
v13, vws, vdesi = np.exp(logM(R13)), np.exp(logM(Rws)), np.exp(logM(R_desi))
say(f"validation: M(1/3) = {v13:.4f} (66.7089), M(0.3097) = {vws:.4f} (64.7349), M(DESI consistency check, regression form) = {vdesi:.4f} (28.8942)")
assert abs(v13 - 66.7089) < 2e-3 and abs(vws - 64.7349) < 2e-3 and abs(vdesi - 28.8942) < 2e-3
Rt = 0.15 * rng.standard_normal((n, n))
f0, g0 = f_and_g(Rt)
gfd = np.zeros_like(Rt)
h = 1e-6
for i in range(n):
    for j in range(n):
        Rp = Rt.copy(); Rp[i, j] += h
        Rm = Rt.copy(); Rm[i, j] -= h
        gfd[i, j] = (logM(Rp) - logM(Rm)) / (2 * h)
gerr = float(np.abs(g0 - gfd).max() / np.abs(gfd).max())
say(f"gradient check: max relative error vs central differences = {gerr:.2e}")
assert gerr < 1e-5
rec('validation', dict(M_alpha_1_3=v13, M_alpha_0p3097=vws, M_desi_regression_rule=vdesi,
                       gradient_max_rel_err=gerr, alpha_max=alpha_max,
                       canonical_correlation_cap_of_alpha_1_3=float(1 / 3 / alpha_max),
                       canonical_correlations_alpha_1_3=np.linalg.svd(R13, compute_uv=False),
                       canonical_correlations_nested_matrix_rule=np.linalg.svd(R_nest, compute_uv=False)),
    'The whitened general-K statistic reproduces 66.709 at alpha=1/3, 64.735 at the scalar worst case and 28.894 under the DESI consistency-check correlations in regression form; analytic gradient agrees with central differences. The singular values of R = L1^-1 K L2^-1 are the canonical correlations between the DR1 and DR2 vectors; for the scalar family they equal alpha/alpha_max, so alpha=1/3 has largest canonical correlation 0.709.')

# ------------------------------------------------------------------ 2. the PSD boundary: collapse and divergence
say("\n=== 2. explicit boundary constructions (rank-one R inside one bin) ===")
collapse = {}
for b, ii in zip(bin_names, idx):
    if len(ii) != 2:
        continue
    e1b, e2b = et1[ii], et2[ii]
    v = np.array([-e2b[1], e2b[0]]); v /= np.linalg.norm(v)      # e~2 . v = 0
    u = np.array([-e1b[1], e1b[0]]); u /= np.linalg.norm(u)      # e~1 . u = 0, hence y~ . v = 0 for every s
    dv = Dt2[:, ii] @ v
    du = Dt1[:, ii] @ u
    rows = {}
    for s in (0.5, 0.9, 0.99, 0.999, 1 - 1e-4, 1 - 1e-5, 1 - 1e-6):
        R = np.zeros((n, n)); R[np.ix_(ii, ii)] = s * np.outer(u, v)
        rows[f'{s:.6f}'] = logM(R)
    proj = dv - 1.0 * du                                         # d~_g . v in the limit s -> 1
    collapse[b] = dict(log_M_vs_s=rows, min_abs_projection_over_grid=float(np.abs(proj).min()),
                       predicted_slope_log_M_times_1_minus_s2=float(-0.5 * np.abs(proj).min() ** 2),
                       v_whitened_DR2=v, u_whitened_DR1=u)
    say(f"  collapse in {b:10s}: log M at s = 0.9, 0.99, 0.999, 1-1e-6 -> "
        + ', '.join(f"{rows[k]:.2f}" for k in ('0.900000', '0.990000', '0.999000', '0.999999'))
        + f";  min_g |d~_g.v| = {np.abs(proj).min():.4f}")
rec('collapse_construction', collapse,
    'Explicit admissible laws at the PSD boundary that drive the exact combined statistic to 0. In bin b take v (unit, whitened DR2 coordinates) perpendicular to the observed whitened DR2 residual and u (unit, whitened DR1) perpendicular to the observed whitened DR1 residual, R = s u v^T (K = L1 R L2, block-diagonal, rank one). Then the innovation along v is exactly 0 while its variance 1-s^2 -> 0, so for every grid point the log-LR gains -(d~_g.v)^2 / (2(1-s^2)) and log M -> -infinity; log_M_vs_s tabulates log M for s -> 1. At s = 1 the law is degenerate but consistent with both published marginals and with the observed data, and M = 0 exactly.')

div = {}
angles = np.linspace(0, np.pi, 181)[:-1]
for b, ii in zip(bin_names, idx):
    if len(ii) != 2:
        continue
    best = (-np.inf, None, None)
    e1b, e2b = et1[ii], et2[ii]
    for tu in angles:
        u = np.array([np.cos(tu), np.sin(tu)])
        for tv in angles:
            v = np.array([np.cos(tv), np.sin(tv)])
            yv = e2b @ v - u @ e1b                                 # y~ . v at s = 1
            dvg = Dt2[:, ii] @ v - Dt1[:, ii] @ u                  # d~_g . v at s = 1
            f = float(np.max(dvg * yv - 0.5 * dvg ** 2))
            if f > best[0]:
                best = (f, u, v)
    f, u, v = best
    rows = {}
    for s in (0.9, 0.99, 0.999, 1 - 1e-4, 1 - 1e-5, 1 - 1e-6):
        R = np.zeros((n, n)); R[np.ix_(ii, ii)] = s * np.outer(u, v)
        rows[f'{s:.6f}'] = logM(R)
    div[b] = dict(max_f_top=f, log_M_vs_s=rows, u_whitened_DR1=u, v_whitened_DR2=v)
    say(f"  divergence in {b:10s}: f_top = {f:.3f}; log M at s = 0.9, 0.99, 0.999, 1-1e-6 -> "
        + ', '.join(f"{rows[k]:.2f}" for k in ('0.900000', '0.990000', '0.999000', '0.999999')))
rec('divergence_construction', div,
    'Explicit admissible laws at the PSD boundary that drive the statistic to +infinity: rank-one R = s u v^T inside one bin with (u, v) chosen on a 1-degree grid to maximise f_top = max_g [(d~_g.v)(y~.v) - (d~_g.v)^2/2]; when f_top > 0, log M ~ f_top/(1-s^2) -> +infinity as s -> 1.')

# ------------------------------------------------------------------ 3. capped classes by projected gradient
say("\n=== 3. capped classes ||R||_op <= rho (largest canonical correlation at most rho) ===")
caps = [0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999]
table = {}
for variant, proj in (('full', proj_full), ('block_diagonal', proj_block)):
    warm_inf = None; warm_sup = None
    for rho in caps:
        tc = time.time()
        best_inf = None
        for R0 in make_starts(rng, rho, warm_inf):
            R, f, it, resid = pgd(R0, rho, proj, sign=+1.0)
            if best_inf is None or f < best_inf[1]:
                best_inf = (R, f, it, resid)
        best_sup = None
        for R0 in make_starts(rng, rho, warm_sup)[:6] + ([warm_sup] if warm_sup is not None else []):
            R, f, it, resid = pgd(R0, rho, proj, sign=-1.0)
            if best_sup is None or f > best_sup[1]:
                best_sup = (R, f, it, resid)
        warm_inf, warm_sup = best_inf[0], best_sup[0]
        a_cap = min(rho * alpha_max, alpha_max * (1 - 1e-9))
        a_grid = np.linspace(0, a_cap, 400)
        sc = [logM(R_from_K(C1M * a)) for a in a_grid]
        k = int(np.argmin(sc))
        scalar_min = float(np.exp(optimize.minimize_scalar(lambda a: logM(R_from_K(C1M * a)),
                                                           bounds=(max(0, a_grid[max(k - 1, 0)]), min(a_cap, a_grid[min(k + 1, len(a_grid) - 1)])),
                                                           method='bounded').fun)) if 0 < k < len(a_grid) - 1 else float(np.exp(sc[k]))
        table.setdefault(variant, {})[f'{rho}'] = dict(
            inf_log_M=best_inf[1], inf_M=float(np.exp(best_inf[1])), inf_iters=best_inf[2], inf_pg_residual=best_inf[3],
            inf_structure=describe_R(best_inf[0]),
            sup_log_M=best_sup[1], sup_M=float(np.exp(min(best_sup[1], 700))), sup_iters=best_sup[2], sup_pg_residual=best_sup[3],
            sup_structure=dict(singular_values=np.linalg.svd(best_sup[0], compute_uv=False)),
            scalar_family_min_with_alpha_le_rho_alpha_max=scalar_min, alpha_cap=a_cap)
        say(f"  {variant:14s} rho = {rho:<6} inf M = {np.exp(best_inf[1]):10.4g}  (sv top {np.linalg.svd(best_inf[0], compute_uv=False)[0]:.3f}, "
            f"resid {best_inf[3]:.1e}, {best_inf[2]} it)   sup M = {np.exp(min(best_sup[1], 700)):10.4g}   scalar-family min(alpha<= {a_cap:.3f}) = {scalar_min:8.3f}   [{time.time() - tc:.0f}s]")
rec('capped_class_table', table,
    'Worst (inf) and best (sup) case of the exact combined statistic over all admissible cross-covariances whose canonical correlations are at most rho (||R||_op <= rho, R = L1^-1 K L2^-1), for the full 11x11 class and the block-diagonal-by-bin class, by projected gradient descent (analytic gradient, Barzilai-Borwein steps, Armijo backtracking, up to 4000 iterations, 11 starts for the inf incl. the zero matrix, the alpha=1/3 member, the nested matrix rule, the DESI consistency-check correlations, random dense and random rank-one matrices and the previous cap\'s optimum; 6-7 starts for the sup). Each inf is an upper bound on the true infimum and each sup a lower bound on the true supremum; inf_structure gives the optimum\'s canonical correlations, the bin composition of its leading singular vectors, and the per-quantity correlations and noise fractions it implies. scalar_family_min is the one-fraction family\'s minimum restricted to alpha <= rho alpha_max for comparison.')

cross = {}
for variant in table:
    rh = np.array([float(k) for k in table[variant]])
    lm = np.array([table[variant][k]['inf_log_M'] for k in table[variant]])
    o = np.argsort(rh); rh, lm = rh[o], lm[o]
    for name, lev in (('rho_where_inf_crosses_20', np.log(THR)), ('rho_where_inf_crosses_1', 0.0)):
        if lm.min() < lev < lm.max():
            k = int(np.argmax(lm < lev))
            cross[f'{variant}:{name}'] = float(rh[k - 1] + (lev - lm[k - 1]) * (rh[k] - rh[k - 1]) / (lm[k] - lm[k - 1]))
        else:
            cross[f'{variant}:{name}'] = None
rec('inf_threshold_crossings', cross, 'Cap rho at which the worst case over the capped class falls below the threshold 20 and below 1 (linear interpolation of log M between tabulated caps); None if not bracketed (see unrestricted_class_small_caps for caps below 0.3).')

# ------------------------------------------------------------------ 4. structured points
say("\n=== 4. structured single points and the per-quantity diagonal family ===")
M_nest = float(np.exp(logM(R_nest)))
say(f"  matrix nested rule K = C2 (Cov(X1,X2) = Var(X2)): M = {M_nest:.3f}, canonical correlations {np.round(np.linalg.svd(R_nest, compute_uv=False), 3)}")
rec('matrix_nested_rule', dict(M=M_nest, p_bound=1 / M_nest, canonical_correlations=np.linalg.svd(R_nest, compute_uv=False),
                               admissible=bool(gev.max() < 1), max_generalised_eigenvalue_C2_vs_C1=float(gev.max())),
    'Exact combined statistic under the matrix nested rule K = Cov(X1, X2) = C2 (the efficient-estimator identity Cov(theta_1, theta_2) = Var(theta_2) that holds when DR2 is the efficient estimate from a superset of the DR1 data with a frozen pipeline); admissible because C1 - C2 is positive definite (all generalised eigenvalues of C2 vs C1 below 1). The paper\'s scalar "nested rule" (26.0) uses alpha_i = C2_ii/C1_ii per quantity in regression form.')

capN = s2 ** 2 / s1 ** 2


def logM_diag(a):
    v = logM(R_from_K(C1M @ np.diag(a)))
    return 60.0 if not np.isfinite(v) else v


best = None
for k in range(40):
    x0 = capN / 2 if k == 0 else rng.uniform(0, capN)
    r = optimize.minimize(logM_diag, x0, method='L-BFGS-B', bounds=[(0.0, c) for c in capN])
    if best is None or r.fun < best.fun:
        best = r
M_diag = float(np.exp(best.fun))
say(f"  per-quantity diagonal family alpha_i in [0, C2_ii/C1_ii]: worst case M = {M_diag:.3f} at {dict(zip(labels, np.round(best.x, 3)))}")
rec('per_quantity_nested_cap_worst_case', dict(M=M_diag, p_bound=1 / M_diag, argmin_alpha_i=dict(zip(labels, best.x)),
                                                 caps=dict(zip(labels, capN)), at_cap=dict(zip(labels, np.isclose(best.x, capN, atol=1e-4))),
                                                 at_zero=dict(zip(labels, best.x < 1e-4))),
    'Minimum of the exact combined statistic over one noise fraction per matched quantity (regression form Cov(X1, X2) = C1 diag(alpha_i)^T), each between 0 and its strictly nested value C2_ii/C1_ii; L-BFGS-B, 40 starts. The paper\'s per-bin version (one fraction per bin, same caps) gives 16.16.')

RES['meta'] = dict(value=dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), runtime_seconds=round(time.time() - t_start, 1),
                              grid='30x30 Default', n_matched=n, labels=labels, bins=bin_names),
                   definition='(meta)', script=THIS)

NUM = load_numbers()
NUM['unrestricted_class'] = py(RES)
save_numbers(NUM)
(OUT_DIR / 'unrestricted_class.log').write_text('\n'.join(LOG) + '\n')
say(f"\nwrote {OUT} [unrestricted_class]  ({time.time() - t_start:.0f}s)")
