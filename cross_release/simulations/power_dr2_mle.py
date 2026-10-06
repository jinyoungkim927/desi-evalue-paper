#!/usr/bin/env python3
"""Power of the combination statistics when the DR2 MLE is true: the price of robustness.

Simulate universes where the ALTERNATIVE is true: data = w0waCDM prediction at
the DR2 BAO MLE (w0, wa) = (-0.856, -0.430) + noise, with the noise generated
under the one-fraction truth a_true = 1/3.  Residuals relative to LCDM are
    eps_release = (mu_alt - mu_LCDM)_release + noise.

Statistics (per draw), threshold 20 (alpha = 0.05):
  JOINT13 : anytime-valid sup_t M_t = max(M1, M_joint(alpha = 1/3))
  FAMINF  : max(M1, inf over the 7-alpha family of M_joint(alpha))
            family = {0.25, 0.275, 0.30, 1/3, 0.35, 0.375, 0.40}
  PROD    : M1 * M2  (product of the two snapshot mixture e-values)
  AVG     : (M1 + M2)/2  (dependence-robust average of snapshot e-values)
  M2      : DR2 snapshot mixture e-value alone

Also: medians of each statistic under the alternative, and under H0 (a = 1/3)
for context.  All arithmetic in log space; medians reported as exp(median log).

The output of record of this script is cross_release/inputs/power_dr2_mle_results.json, read by
cross_release/one_fraction_family.py. Running the script again writes a fresh copy to
results/simulations/ and prints the log to stdout.
Run from the repository root: python cross_release/simulations/power_dr2_mle.py
"""
import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, OUT_DIR, INPUTS, add_code_path  # noqa: E402
add_code_path()
REPO = ROOT
SIM_OUT = OUT_DIR / 'simulations'   # results/simulations/

from data_loader import load_desi_data
from cosmology import CosmologyParams, LCDM, compute_bao_predictions
from evalue_analysis import (precompute_kernels, mixture_log_e_from_residuals,
                             _build_theory_vector)
from eprocess_joint import (W0_GRID, WA_GRID, DR2_MLE, build_delta_matrix,
                            joint_kernel, joint_log_e, decomposition_for_alpha)
from eprocess_hierarchical_mc import build_bin_matching, nearest_psd

LOG20 = np.log(20.0)
A_TRUE = 1.0 / 3.0
ALPHA_FAMILY = [0.25, 0.275, 0.30, 1.0 / 3.0, 0.35, 0.375, 0.40]
N_ALT = 20000
N_H0 = 5000
SEED_ALT = 20260831
SEED_H0 = 20260832


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def main():
    t0 = time.time()
    ds1 = load_desi_data(REPO / 'data' / 'dr1', 'DR1')
    ds2 = load_desi_data(REPO / 'data' / 'dr2', 'DR2')
    n1, n2 = len(ds1.data), len(ds2.data)

    K1 = precompute_kernels(ds1.z_eff, ds1.quantities, ds1.cov, W0_GRID, WA_GRID)
    K2 = precompute_kernels(ds2.z_eff, ds2.quantities, ds2.cov, W0_GRID, WA_GRID)
    _, d1 = build_delta_matrix(ds1.z_eff, ds1.quantities, W0_GRID, WA_GRID)
    _, d2 = build_delta_matrix(ds2.z_eff, ds2.quantities, W0_GRID, WA_GRID)

    # --- bin matching and covariance blocks --------------------------------
    dr2_to_dr1, _ = build_bin_matching(ds1, ds2)
    pairs = [(i, j) for j, i in enumerate(dr2_to_dr1) if i >= 0]
    m1 = np.array([i for i, j in pairs], dtype=int)
    m2 = np.array([j for i, j in pairs], dtype=int)
    u2 = np.array([j for j in range(n2) if dr2_to_dr1[j] < 0], dtype=int)
    n_match, n_u = len(pairs), len(u2)
    C1M = ds1.cov[np.ix_(m1, m1)]
    C2M = ds2.cov[np.ix_(m2, m2)]
    cov_u, _ = nearest_psd(ds2.cov[np.ix_(u2, u2)])
    cov_u = cov_u + 1e-14 * np.eye(n_u)

    # --- kernels for the 7-alpha family (assumed alpha) --------------------
    kerns = {}
    for a in ALPHA_FAMILY:
        cy, min_eig, psd = decomposition_for_alpha(C1M, C2M, a)
        assert psd, f"alpha={a} inadmissible (min eig {min_eig})"
        cy = cy + 1e-14 * np.eye(n_match)
        kerns[a] = joint_kernel(d1, d2, m1, m2, u2, ds1.cov,
                                (1.0 - a) ** 2 * cy, cov_u, a)

    # --- sanity: observed values on the real data --------------------------
    e1o = (ds1.data - K1['mu_null'])[:, None]
    e2o = (ds2.data - K2['mu_null'])[:, None]
    M1_obs = float(np.exp(mixture_log_e_from_residuals(e1o, K1)[0]))
    M2_obs = float(np.exp(mixture_log_e_from_residuals(e2o, K2)[0]))
    Mj_obs = {}
    for a in ALPHA_FAMILY:
        y = e2o[m2] - a * e1o[m1]
        Mj_obs[a] = float(np.exp(joint_log_e(e1o, y, e2o[u2], kerns[a])[0]))
    faminf_obs = min(Mj_obs.values())
    print(f"[sanity] observed: M1 = {M1_obs:.2f}, M2 = {M2_obs:.2f}, "
          f"M_joint(1/3) = {Mj_obs[1.0/3.0]:.2f}, FAMINF = {faminf_obs:.2f}")
    for a in ALPHA_FAMILY:
        print(f"    alpha = {a:.4f}: M_joint = {Mj_obs[a]:.2f}")

    # --- alternative-mean offsets (residuals relative to LCDM) -------------
    alt = CosmologyParams(w0=DR2_MLE[0], wa=DR2_MLE[1])
    off1 = (_build_theory_vector(compute_bao_predictions(ds1.z_eff, alt),
                                 ds1.z_eff, ds1.quantities)
            - _build_theory_vector(compute_bao_predictions(ds1.z_eff, LCDM),
                                   ds1.z_eff, ds1.quantities))
    off2 = (_build_theory_vector(compute_bao_predictions(ds2.z_eff, alt),
                                 ds2.z_eff, ds2.quantities)
            - _build_theory_vector(compute_bao_predictions(ds2.z_eff, LCDM),
                                   ds2.z_eff, ds2.quantities))

    # --- noise machinery under one-fraction truth a_true = 1/3 ------------
    cy_t, _, psd_t = decomposition_for_alpha(C1M, C2M, A_TRUE)
    assert psd_t
    cy_t = cy_t + 1e-14 * np.eye(n_match)
    L1 = K1['L_chol']
    L23 = np.linalg.cholesky(cy_t)
    Lu = np.linalg.cholesky(cov_u)

    def simulate(n_draws, seed, with_signal, batch=1000):
        rng = np.random.default_rng(seed)
        logs = dict(M1=[], M2=[], **{f'J{a:.4f}': [] for a in ALPHA_FAMILY})
        done = 0
        while done < n_draws:
            b = min(batch, n_draws - done)
            z1 = L1 @ rng.standard_normal((n1, b))
            e23 = L23 @ rng.standard_normal((n_match, b))
            eu = Lu @ rng.standard_normal((n_u, b))
            eps2 = np.zeros((n2, b))
            eps2[m2] = A_TRUE * z1[m1] + (1.0 - A_TRUE) * e23
            eps2[u2] = eu
            eps1 = z1
            if with_signal:
                eps1 = eps1 + off1[:, None]
                eps2 = eps2 + off2[:, None]
            logs['M1'].append(mixture_log_e_from_residuals(eps1, K1))
            logs['M2'].append(mixture_log_e_from_residuals(eps2, K2))
            for a in ALPHA_FAMILY:
                y = eps2[m2] - a * eps1[m1]
                logs[f'J{a:.4f}'].append(joint_log_e(eps1, y, eps2[u2], kerns[a]))
            done += b
        return {k: np.concatenate(v) for k, v in logs.items()}

    def statistics(logs):
        lM1, lM2 = logs['M1'], logs['M2']
        lJ13 = logs[f'J{1.0/3.0:.4f}']
        lJmin = np.min(np.vstack([logs[f'J{a:.4f}'] for a in ALPHA_FAMILY]), axis=0)
        return dict(
            JOINT13=np.maximum(lM1, lJ13),
            FAMINF=np.maximum(lM1, lJmin),
            PROD=lM1 + lM2,
            AVG=np.logaddexp(lM1, lM2) - np.log(2.0),
            M2=lM2,
        )

    def summarize(stats, n):
        out = {}
        for name, ls in stats.items():
            k = int(np.sum(ls >= LOG20))
            lo, hi = wilson(k, n)
            med = float(np.exp(np.median(ls)))
            out[name] = dict(power=k / n, k=k, n=n, wilson=(float(lo), float(hi)),
                             median=med, median_log10=float(np.median(ls) / np.log(10)))
        return out

    print(f"[sim] alternative (N = {N_ALT}) ...")
    logs_alt = simulate(N_ALT, SEED_ALT, with_signal=True)
    stats_alt = statistics(logs_alt)
    sum_alt = summarize(stats_alt, N_ALT)

    print(f"[sim] H0 context (N = {N_H0}) ...")
    logs_h0 = simulate(N_H0, SEED_H0, with_signal=False)
    stats_h0 = statistics(logs_h0)
    sum_h0 = summarize(stats_h0, N_H0)

    print("\n=== POWER under the alternative (threshold 20) ===")
    for name in ('JOINT13', 'FAMINF', 'PROD', 'M2', 'AVG'):
        s = sum_alt[name]
        print(f"  {name:8s} power = {s['power']:.4f} ({s['k']}/{s['n']}) "
              f"Wilson95 [{s['wilson'][0]:.4f}, {s['wilson'][1]:.4f}]  "
              f"median = {s['median']:.4g} (log10 = {s['median_log10']:.2f})")
    print("\n=== H0 (a = 1/3) context medians / FPR at 20 ===")
    for name in ('JOINT13', 'FAMINF', 'PROD', 'M2', 'AVG'):
        s = sum_h0[name]
        print(f"  {name:8s} median = {s['median']:.4g}  "
              f"P(>=20) = {s['power']:.4f} ({s['k']}/{s['n']})")

    # Paired (same-draw) comparisons: McNemar discordant counts.
    def paired(a_name, b_name):
        ra = stats_alt[a_name] >= LOG20
        rb = stats_alt[b_name] >= LOG20
        n10 = int(np.sum(ra & ~rb))   # a rejects, b does not
        n01 = int(np.sum(~ra & rb))
        N = len(ra)
        gap = (n10 - n01) / N
        se = np.sqrt(max(n10 + n01 - (n10 - n01) ** 2 / N, 0)) / N
        z = (n10 - n01) / np.sqrt(n10 + n01) if (n10 + n01) else 0.0
        return dict(pair=f"{a_name} vs {b_name}", n10=n10, n01=n01,
                    gap=gap, gap_se=float(se), mcnemar_z=float(z))

    print("\n=== Paired comparisons (alternative draws) ===")
    paired_out = []
    for a, b in (('JOINT13', 'FAMINF'), ('JOINT13', 'M2'), ('FAMINF', 'M2'),
                 ('M2', 'AVG'), ('M2', 'PROD'), ('PROD', 'AVG')):
        r = paired(a, b)
        paired_out.append(r)
        print(f"  {r['pair']:18s} gap = {r['gap']:+.4f} +/- {r['gap_se']:.4f} "
              f"(n10 = {r['n10']}, n01 = {r['n01']}, z = {r['mcnemar_z']:+.1f})")

    gap = sum_alt['JOINT13']['power'] - sum_alt['FAMINF']['power']
    print(f"\nJOINT13 - FAMINF power gap = {gap:.4f}")

    out = dict(
        observed=dict(M1=M1_obs, M2=M2_obs, M_joint=Mj_obs,
                      FAMINF=faminf_obs),
        offsets=dict(off1=off1.tolist(), off2=off2.tolist()),
        alternative=sum_alt, h0=sum_h0, paired=paired_out,
        config=dict(alpha_family=ALPHA_FAMILY, a_true=A_TRUE, N_alt=N_ALT,
                    N_h0=N_H0, seeds=(SEED_ALT, SEED_H0), mle=DR2_MLE),
        seconds=time.time() - t0,
    )
    SIM_OUT.mkdir(parents=True, exist_ok=True)
    outp = SIM_OUT / 'power_dr2_mle_results.json'
    with open(outp, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[done] {time.time()-t0:.1f}s -> {outp}")


if __name__ == '__main__':
    main()
