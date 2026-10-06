#!/usr/bin/env python3
"""Shared set-up for the unrestricted class of joint laws (the Frechet class of the two published
covariances) and the exact combined DR1+DR2 statistic.

Imported by cross_release/unrestricted_class.py and cross_release/unrestricted_class_small_caps.py.
Everything here is deterministic (no random numbers).

Notation (see the docstring of unrestricted_class.py): C1 = L1^2, C2 = L2^2 on the 11 matched
quantities; an admissible cross-covariance is K = L1 R L2 with ||R||_op <= 1; whitened residuals
et1 = L1^{-1} e1_m, et2 = L2^{-1} e2_m; whitened mean shifts Dt1, Dt2 (G x 11); CONST holds the
DR1 factor on all 12 DR1 quantities and the unmatched DR2 QSO factor per grid point.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import linalg
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import ROOT, OUT_DIR, NUMBERS, add_code_path, load_numbers, save_numbers  # noqa: E402
add_code_path()
REPO = ROOT
OUT = NUMBERS

from data_loader import load_desi_data                                       # noqa: E402
from evalue_analysis import precompute_kernels                               # noqa: E402
from eprocess_hierarchical_mc import build_bin_matching, nearest_psd          # noqa: E402
import eprocess_joint as EJ                                                   # noqa: E402

THR = 20.0


def py(x):
    if isinstance(x, dict):
        return {str(k): py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [py(v) for v in x]
    if isinstance(x, np.ndarray):
        return py(x.tolist())
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if np.isfinite(x) else ('inf' if x > 0 else '-inf')
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ------------------------------------------------------------------ data (as in structured_families.py)
ds1 = load_desi_data(REPO / 'data' / 'dr1', 'DR1')
ds2 = load_desi_data(REPO / 'data' / 'dr2', 'DR2')
K1 = precompute_kernels(ds1.z_eff, ds1.quantities, ds1.cov, EJ.W0_GRID, EJ.WA_GRID)
K2 = precompute_kernels(ds2.z_eff, ds2.quantities, ds2.cov, EJ.W0_GRID, EJ.WA_GRID)
_, D1 = EJ.build_delta_matrix(ds1.z_eff, ds1.quantities, EJ.W0_GRID, EJ.WA_GRID)
_, D2 = EJ.build_delta_matrix(ds2.z_eff, ds2.quantities, EJ.W0_GRID, EJ.WA_GRID)
dr2_to_dr1, _ = build_bin_matching(ds1, ds2)
pairs = [(i, j) for j, i in enumerate(dr2_to_dr1) if i >= 0]
m1 = np.array([i for i, _ in pairs]); m2 = np.array([j for _, j in pairs])
u2 = np.array([j for j in range(len(ds2.data)) if dr2_to_dr1[j] < 0])
n = len(pairs)
C1 = ds1.cov
C1M = C1[np.ix_(m1, m1)]
C2M = ds2.cov[np.ix_(m2, m2)]
covu = nearest_psd(ds2.cov[np.ix_(u2, u2)])[0] + 1e-14 * np.eye(len(u2))
e1 = ds1.data - K1['mu_null']
e2 = ds2.data - K2['mu_null']
s1 = np.sqrt(np.diag(C1M)); s2 = np.sqrt(np.diag(C2M))
TRACER = {0.295: 'BGS', 0.51: 'LRG1', 0.706: 'LRG2', 0.934: 'LRG3+ELG1', 1.321: 'ELG2', 2.33: 'Lya'}
QN = {'DV_over_rs': 'DV', 'DM_over_rs': 'DM', 'DH_over_rs': 'DH'}
zz = [round(float(ds2.z_eff[j]), 3) for j in m2]
labels = [f"{TRACER[z]} {QN[ds2.quantities[j]]}" for z, j in zip(zz, m2)]
bins = sorted(set(zz))
idx = [np.array([k for k, z in enumerate(zz) if z == b]) for b in bins]
bin_names = [TRACER[b] for b in bins]
G = len(EJ.W0_GRID) * len(EJ.WA_GRID)

C1inv = np.linalg.inv(C1)
A1 = D1 @ C1inv
c1 = np.einsum('gi,ij,gj->g', D1, C1inv, D1)
Qu = np.linalg.inv(covu)
Du = D2[:, u2]
Au = Du @ Qu
cu = np.einsum('gi,ij,gj->g', Du, Qu, Du)
CONST = A1 @ e1 - 0.5 * c1 + Au @ e2[u2] - 0.5 * cu      # DR1 factor + unmatched DR2 term, per grid point


def sym_sqrt(C):
    w, V = np.linalg.eigh(C)
    return (V * np.sqrt(w)) @ V.T, (V / np.sqrt(w)) @ V.T


L1, L1i = sym_sqrt(C1M)
L2, L2i = sym_sqrt(C2M)
et1 = L1i @ e1[m1]
et2 = L2i @ e2[m2]
Dt1 = D1[:, m1] @ L1i          # (G, n): whitened DR1 mean shifts on the matched quantities
Dt2 = D2[:, m2] @ L2i          # (G, n): whitened DR2 mean shifts
I_n = np.eye(n)
gev = linalg.eigh(C2M, C1M, eigvals_only=True)
alpha_max = float(np.sqrt(gev.min()))

# correlations of DESI's DR1-DR2 consistency check (tracer counts from DR1 Table 1 and DR2 Results II Table I)
NTR = {0.295: (300017, 1188526), 0.51: (506905, 1052151), 0.706: (771875, 1613562),
       0.934: (1876164, 4540343), 1.321: (1415687, 3797271)}
rho_desi = np.array([np.sqrt(NTR[z][0] / NTR[z][1]) if z in NTR else 0.61 for z in zz])
aD = rho_desi * s2 / s1


def R_from_K(K):
    return L1i @ K @ L2i


def K_from_R(R):
    return L1 @ R @ L2


R13 = R_from_K(C1M / 3.0)                      # alpha = 1/3 member
Rws = R_from_K(C1M * 0.30965149996910124)      # scalar-family worst case
R_desi = R_from_K(C1M @ np.diag(aD))           # regression form Cov(X1, X2) = C1 A^T, A = diag(alpha_i)
R_nest = R_from_K(C2M)                         # matrix nested rule Cov(X1, X2) = Var(X2)


def terms(R):
    S = I_n - R.T @ R
    if np.linalg.eigvalsh(S).min() <= 1e-13:
        return None
    Q = np.linalg.inv(S)
    y = et2 - R.T @ et1
    delta = Dt2 - Dt1 @ R
    a = delta @ Q
    L = CONST + a @ y - 0.5 * np.einsum('gi,gi->g', a, delta)
    return dict(L=L, Q=Q, y=y, delta=delta, a=a)


def logM(R):
    t = terms(R)
    return np.inf if t is None else float(logsumexp(t['L']) - np.log(G))


def f_and_g(R):
    """log M(R) and its gradient with respect to R.

    With a_g = Q delta_g^T, w = Q y, Q = (I - R^T R)^{-1} and p_g the softmax weights of L_g:
    d log M / dR = sum_g p_g [ -Dt1_g w^T + Dt1_g a_g^T - et1 a_g^T + (R w) a_g^T + (R a_g) w^T - (R a_g) a_g^T ].
    """
    t = terms(R)
    if t is None:
        return np.inf, None
    L = t['L']
    lse = logsumexp(L)
    p = np.exp(L - lse)
    a = t['a']
    w = t['Q'] @ t['y']
    pD = p[:, None] * Dt1
    pa = p[:, None] * a
    sD = pD.sum(0)
    sa = pa.sum(0)
    grad = (-np.outer(sD, w) + pD.T @ a - np.outer(et1, sa) + np.outer(R @ w, sa)
            + np.outer(R @ sa, w) - R @ (pa.T @ a))
    return float(lse - np.log(G)), grad


def proj_full(R, rho):
    U, s, Vt = np.linalg.svd(R)
    return (U * np.minimum(s, rho)) @ Vt


def proj_block(R, rho):
    Rb = np.zeros_like(R)
    for ii in idx:
        Rb[np.ix_(ii, ii)] = proj_full(R[np.ix_(ii, ii)], rho)
    return Rb


def pgd(R0, rho, proj, sign=1.0, iters=4000, tol=1e-12):
    """Projected gradient on sign*log M over {||R|| <= rho}; BB step with Armijo backtracking."""
    R = proj(R0, rho)
    f, g = f_and_g(R)
    f, g = sign * f, sign * g
    eta = 0.1 / (np.linalg.norm(g) + 1e-12)
    for it in range(iters):
        while True:
            Rn = proj(R - eta * g, rho)
            fn, gn = f_and_g(Rn)
            fn, gn = sign * fn, (None if gn is None else sign * gn)
            dR = Rn - R
            if gn is not None and (fn <= f + 1e-4 * float(np.sum(g * dR)) or eta < 1e-15):
                break
            eta *= 0.5
        if float(np.linalg.norm(dR)) < tol * (1 + float(np.linalg.norm(R))) or abs(f - fn) < 1e-14:
            R, f, g = Rn, fn, gn
            break
        dg = gn - g
        den = float(np.sum(dR * dg))
        eta = min(max(float(np.sum(dR * dR)) / den if den > 1e-300 else 2 * eta, 1e-9), 1e4)
        R, f, g = Rn, fn, gn
    resid = float(np.linalg.norm(proj(R - 1e-3 * g, rho) - R)) / 1e-3
    return R, sign * f, it + 1, resid


def describe_R(R):
    U, s, Vt = np.linalg.svd(R)
    K = K_from_R(R)
    v1 = Vt[0]; u1 = U[:, 0]
    comp_v = {b: float(np.sum(v1[ii] ** 2)) for b, ii in zip(bin_names, idx)}
    comp_u = {b: float(np.sum(u1[ii] ** 2)) for b, ii in zip(bin_names, idx)}
    return dict(singular_values=s, top_right_singular_vector_bin_weights_DR2=comp_v,
                top_left_singular_vector_bin_weights_DR1=comp_u,
                implied_corr_X1i_X2i=dict(zip(labels, np.diag(K) / (s1 * s2))),
                implied_alpha_i_K_ii_over_C1_ii=dict(zip(labels, np.diag(K) / s1 ** 2)),
                frobenius_fraction_off_block=float(np.linalg.norm(R - proj_block(R, 10.0)) / max(np.linalg.norm(R), 1e-300)))


def make_starts(rng, rho, warm):
    """Start points for the capped-class optimisation (shared by both scripts)."""
    S = [np.zeros((n, n))]
    for Rb in (R13, R_nest, R_desi):
        nb = np.linalg.norm(Rb, 2)
        S.append(Rb * min(1.0, 0.999 * rho / nb))
    for sc in (0.1, 0.3):
        for _ in range(2):
            S.append(sc * rng.standard_normal((n, n)))
    for _ in range(3):
        u = rng.standard_normal(n); v = rng.standard_normal(n)
        S.append(rho * np.outer(u / np.linalg.norm(u), v / np.linalg.norm(v)))
    if warm is not None:
        S.append(warm)
    return S
