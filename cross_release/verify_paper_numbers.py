#!/usr/bin/env python3
"""Recompute the numbers the paper quotes from the outputs of code/ and print each one against
its printed value. A check script: it reads results/*.json (or their copies of record) and the
data, recomputes, and records the values in the numbers file.

Sections
  1. Wilks / sigma conversions quoted in the paper (3.70, 3.69, 2.18, DESI 3.1 sigma, Bonferroni
     per-look thresholds 1.96 / 2.24 / 2.58)
  2. Table 1: the seven specifications against results/regrow_results.json, and whether every
     printed Markov p-value is rounded up; the Section 4.3 quintessence values
  3. Table 2: leave-one-out e-values and the fitted (w0, wa) per fold, recomputed with
     evalue_analysis.loocv_evalue and split_evalue_by_indices; shares 78.6% / 91%, six-bin
     average 2.54; per-bin REGROW and Default aggregates
  4. the other quoted values: E_UI 78 / 110 and the 0/200 synthetic-null bound; BAO+CMB 2.19
     and 7.07; no-LRG2 0.49; conditional mean 0.750 and 59%; background factor 163; Table 4
     columns; shell-prior numbers (delta*, M(1), M_max); cross-prediction table; split 1.43 and
     the alternating-split swing
  5. Table IV of DESI DR2 Results II versus the released CobayaSampler covariance: per-quantity
     error ratios, D_M-D_H correlations, M_DR2 under four hybrids (which part matters)
  6. the DR2-DR1 null test across cross-covariances a C1, with the check that a joint Gaussian
     law with the published marginals exists (C2 - a^2 C1 PSD)
  7. closed-form null means of the alpha=1/3 statistic under DR1-side coherent offsets
  8. weights and budget arithmetic (0.5755, 0.0435, 0.0389, 0.0857, 0.0589), the exact-tail
     Bonferroni with its Monte Carlo interval

Writes results/numbers.json["paper_numbers_check"] and results/verify_paper_numbers.log.
Run (from the repository root, about 10 s): python cross_release/verify_paper_numbers.py
Reads results/*.json written by code/*.py, or their copies of record in cross_release/inputs/results/.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import linalg, stats
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import ROOT, OUT_DIR, NUMBERS, add_code_path, load_numbers, save_numbers, result_json  # noqa: E402
add_code_path()
REPO = ROOT
OUT = NUMBERS
THIS = 'cross_release/verify_paper_numbers.py'

from data_loader import load_desi_data                                                     # noqa: E402
from evalue_analysis import (precompute_kernels, mixture_log_e_from_residuals,              # noqa: E402
                             loocv_evalue, split_evalue_by_indices)
from eprocess_hierarchical_mc import build_bin_matching, nearest_psd                        # noqa: E402
import eprocess_joint as EJ                                                                 # noqa: E402

t_start = time.time()
RES = {}
LOG = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    LOG.append(s)


def rec(key, value, definition):
    RES[key] = dict(value=value, definition=definition, script=THIS)


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


def J(name):
    return result_json(name)


NUM = load_numbers()
M1 = NUM['anchors']['M_DR1']['value']
M2 = NUM['anchors']['M_DR2']['value']

# ------------------------------------------------------------------ 1. sigma conversions
say("=== 1. sigma conversions ===")
ds = J('delta_star_results.json')
dchi2 = ds['DR2_MLE']['delta_chi2_exact']
p_wilks = float(stats.chi2.sf(dchi2, 2))
z_wilks = float(stats.norm.isf(p_wilks / 2))
sig = J('sigma_mc_results.json')
z_markov = float(stats.norm.isf(1 / (2 * M2)))
p_desi = float(stats.chi2.sf(12.5, 2))
z_desi = float(stats.norm.isf(p_desi / 2))
bonf = {n: float(stats.norm.isf(0.05 / n / 2)) for n in (1, 2, 4, 5)}
say(f"  Delta chi2 = {dchi2:.3f} -> Wilks p = {p_wilks:.3e}, two-sided sigma = {z_wilks:.3f} (paper 2.2e-4, 3.70)")
say(f"  MC tail {sig['mc']['p_emp']:.3e} (CI {sig['mc']['p_emp_ci']}), sigma_emp = {sig['mc']['sigma_emp_two_tail']:.3f} (paper 3.69); Markov sigma = {z_markov:.3f} (paper 2.18)")
say(f"  DESI BAO+CMB Delta chi2 = 12.5 -> p = {p_desi:.4f}, sigma = {z_desi:.2f} (paper 0.0019, 3.1); sqrt(12.5) = {math.sqrt(12.5):.2f}")
say(f"  Bonferroni per-look two-sided thresholds at overall 0.05: {bonf} (paper 1.96, 2.24, 2.58)")
rec('sigma_conversions', dict(delta_chi2_exact=dchi2, wilks_p=p_wilks, wilks_sigma_two_sided=z_wilks,
                              mc_tail=sig['mc']['p_emp'], mc_tail_ci95=sig['mc']['p_emp_ci'], sigma_emp=sig['mc']['sigma_emp_two_tail'],
                              markov_sigma=z_markov, desi_p=p_desi, desi_sigma=z_desi, bonferroni_thresholds=bonf),
    'Recomputed sigma conversions: Wilks for Delta chi2 = 16.86 on chi2(2) (two-sided z), the Monte Carlo tail of code/verify_sigma_mc.py, the Markov sigma Phi^-1(1 - 1/(2E)), DESI\'s 12.5, and the Bonferroni per-look thresholds.')

# ------------------------------------------------------------------ 2. Table 1
say("\n=== 2. Table 1 ===")
rg = J('regrow_results.json')
rows = {r['name']: (r['M_DR1'], r['M_DR2']) for r in rg['flat_priors']}
rows.update({f"REGROW_delta_{r['delta']:.0f}": (r['M_DR1'], r['M_DR2']) for r in rg['regrow_priors']})
printed = {'Narrow': ('2.60', '144', '0.007'), 'Default': ('1.05', '33.97', '0.0295'), 'Wide': ('0.320', '10.7', '0.094'),
           'Ong': ('0.160', '4.81', '0.208'), 'REGROW_delta_1': ('1.50', '7.58', '0.132'),
           'REGROW_delta_2': ('3.02', '72.5', '0.014'), 'REGROW_delta_3': ('5.25', '297', '0.0034')}
t1 = {}
for k, (a, b) in rows.items():
    p = 1 / b
    pr = float(printed[k][2])
    t1[k] = dict(M_DR1=a, M_DR2=b, p_exact=p, p_printed=pr, printed_not_below_bound=bool(pr >= p),
                 avg_with_DR1=0.5 * (a + b))
    say(f"  {k:15s} M1 {a:8.4f} ({printed[k][0]})  M2 {b:9.4f} ({printed[k][1]})  1/M2 = {p:.5f} printed {pr}  {'OK' if pr >= p else 'ROUNDED DOWN'}  avg {0.5 * (a + b):.2f}")
rec('table1', t1, 'Table 1 specifications from results/regrow_results.json; printed_not_below_bound checks that each printed Markov p-value is at or above 1/M_DR2 (never rounded down). avg_with_DR1 checks the caption claim that the equal-weight average reaches 20 only for Narrow and REGROW delta >= 2.')
tb = {r['name']: r for r in J('literature_priors.json')}
rec('thawing_freezing', dict(freezing_M_DR2=tb['Quintessence_freezing']['M_DR2'], thawing_M_DR2=tb['Quintessence_thawing_CL_band']['M_DR2'],
                             thawing_M_DR1=tb['Quintessence_thawing_CL_band']['M_DR1'], notes=tb['Quintessence_freezing']['notes']),
    'results/literature_priors.json: the freezing-prior value for the box centred at (-0.85, +0.15), evaluated by code/literature_priors.py on the 320 x 320 grid named in Section 4.3 (14.8; the grid-resolution study is resolution_and_covariance_checks.quintessence_resolution); thawing 1129 (paper 1.1e3) and DR1 thawing 14.7 (paper ~15).')
say(f"  freezing prior in results/literature_priors.json: {tb['Quintessence_freezing']['M_DR2']:.2f} (paper 14.8 on 320 x 320); thawing {tb['Quintessence_thawing_CL_band']['M_DR2']:.1f}, DR1 thawing {tb['Quintessence_thawing_CL_band']['M_DR1']:.1f}")

# ------------------------------------------------------------------ 3. Table 2 (LOO)
say("\n=== 3. Table 2: leave-one-out ===")
ds2 = load_desi_data(REPO / 'data' / 'dr2', 'DR2')
ds1 = load_desi_data(REPO / 'data' / 'dr1', 'DR1')
r = loocv_evalue(ds2.data, ds2.cov, ds2.z_eff, ds2.quantities, verbose=False)
fields = {k: v for k, v in vars(r).items() if not isinstance(v, np.ndarray)}
say("  loocv_evalue fields:", list(vars(r).keys()))
loo_saved = J('loo.json')
per_bin = {float(k): v for k, v in r.per_bin_e.items()}
zs = sorted(per_bin)
E = np.array([per_bin[z] for z in zs])
share = E / E.sum()
# per-fold fitted parameters via the same split machinery
fold = {}
for z in zs:
    test = np.where(np.isclose(ds2.z_eff, z))[0]
    train = np.array([i for i in range(len(ds2.data)) if i not in test])
    etr, ete = split_evalue_by_indices(ds2.data, ds2.cov, ds2.z_eff, ds2.quantities, train, test)
    fold[f'{z:.3f}'] = dict(E_loo_rerun=per_bin[z], E_loo_saved=loo_saved['per_bin_e'][f'{z:.3f}'],
                            E_split_by_indices=etr and ete.e_value, w0=ete.alt_params.w0, wa=ete.alt_params.wa)
    say(f"  z = {z:.3f}: E_k = {per_bin[z]:8.3f} (saved {loo_saved['per_bin_e'][f'{z:.3f}']:8.3f}; by indices {ete.e_value:8.3f})  fitted (w0, wa) = ({ete.alt_params.w0:.3f}, {ete.alt_params.wa:.3f})")
k2 = zs.index(0.706); k3 = zs.index(0.934)
six = (E.sum() - E[k2]) / 6
say(f"  average {E.mean():.3f} (saved {loo_saved['loo_average']:.3f}); LRG2 share {share[k2]:.4f}; LRG2+LRG3 share {share[k2] + share[k3]:.4f}; six-bin average without LRG2 {six:.3f}")
ta = J('per_bin_regrow.json'); bl = J('bin_level_results.json')
rec('table2_loo', dict(folds=fold, loo_average=float(E.mean()), lrg2_share=float(share[k2]), lrg2_plus_lrg3_share=float(share[k2] + share[k3]),
                       six_bin_average_without_lrg2=float(six),
                       regrow_delta2_per_bin=dict(zip(ta['per_bin_labels'], ta['per_bin_M_k_regrow_delta2'])),
                       regrow_product=ta['per_bin_product'], regrow_mean=ta['per_bin_arith_mean'],
                       default_per_bin=dict(zip(bl['bin_labels'], bl['M_k'])), default_product=bl['M_per_bin_product'],
                       default_mean=float(np.mean(bl['M_k'])), default_lrg2=bl['LRG2_only_M']),
    'Table 2 recomputed with code/evalue_analysis.py (loocv_evalue; fitted parameters from split_evalue_by_indices on the same folds), plus the per-bin REGROW delta=2 and Default-prior aggregates from results/per_bin_regrow.json and results/bin_level_results.json.')

# ------------------------------------------------------------------ 4. other quoted values
say("\n=== 4. other quoted values ===")
ui = J('universal_inference_results.json'); jc = J('joint_cmb.json'); ld = J('lrg2_drop.json'); ej = J('eprocess_joint.json')
bg = J('background_sensitivity.json'); cp = J('cross_prediction.json'); sv = J('split_variants.json'); pc = J('power_calibration.json')
cp_upper_0_of_200 = 1 - 0.05 ** (1 / 200)
mm = ej['main']['analytic']['marginal_conditional_mean']
mis = ej['main']['analytic']['E_M2joint_under_misspecified_truth']
dchi2_lg = ds['DR2_MLE']['delta_chi2_local_gaussian']
dmax = math.sqrt(dchi2_lg / 2 - 1)
M_shell = lambda d, dc: math.exp(0.5 * dc * d * d / (1 + d * d)) / (1 + d * d)   # noqa: E731
other = dict(
    E_UI_within=ui['within_dr2']['E_mean'], E_UI_cross=ui['cross_release']['E'], UI_h0_P_ge_20=ui['h0_calibration']['P_ge_20'],
    clopper_pearson_upper_0_of_200=cp_upper_0_of_200,
    BAO_CMB_mix_default=jc['E_mix']['default'], BAO_CMB_loo=jc['E_LOO_BAO_CMB'], BAO_CMB_mle=jc['mle'], BAO_CMB_delta_chi2=jc['delta_chi2'], BAO_CMB_sigma=jc['wilks_sigma_two_sided'],
    M_DR2_no_LRG2=ld['M_DR2_no_LRG2'], conditional_mean=mm['E_M2marg_given_obs_DR1'], frac_cond_exceeds=mm['frac_draws_cond_exceeds_M1'],
    E_joint_if_alpha_0p25=mis['alpha_true=0.250'], E_joint_if_independent=mis['DR1_DR2_independent'],
    background_factor=max(r_['M_DR2'] for r_ in bg) / min(r_['M_DR2'] for r_ in bg), background_rows={r_['label']: (r_['M_DR1'], r_['M_DR2'], r_['M_DR2_no_LRG2']) for r_ in bg},
    delta_star_lower=ds['delta_star_lower'], delta_star_upper=ds['delta_star_upper'], delta_max_formula=dmax, delta_max_saved=ds['delta_max'],
    M_max_formula=M_shell(dmax, dchi2_lg), M_max_saved=ds['M_max'], M_shell_at_1=M_shell(1.0, dchi2_lg), M_shell_at_1_with_16p5=M_shell(1.0, 16.5),
    fisher_distance_mle=rg['DR2_MLE']['fisher_distance_from_LCDM'], cross_prediction=cp,
    split_z1=sv['z1_LRG2_in_train'], split_alternating_ratio=sv['alternating_LRG2_in_test'] / sv['alternating_LRG2_in_train'],
    split_median_h1=pc['H1_w0waCDM_true']['median_e_split'])
for k, v in other.items():
    say(f"  {k}: {v}")
rec('other_quoted_values', other, 'Values quoted in the paper, read back from results/: E_UI (78.2 within DR2, 109.7 across), 0/200 synthetic-null bound (one-sided 95% Clopper-Pearson 0.0149), BAO+CMB (2.19, 7.07, MLE, 14.78, 3.42), no-LRG2 0.493, conditional mean 0.750 and 59.0%, misspecified means 161.9 and 2.44e13, background factor 162.9, shell-prior numbers, cross-prediction, split variants.')

# ------------------------------------------------------------------ 5. Table IV versus the released covariance
say("\n=== 5. Table IV (DR2 Results II) versus the released CobayaSampler covariance ===")
T4 = {0.295: ('DV', 0.075), 0.51: (0.167, 0.425, -0.459), 0.706: (0.177, 0.330, -0.404),
      0.934: (0.152, 0.193, -0.416), 1.321: (0.318, 0.221, -0.434), 1.484: (0.760, 0.516, -0.500),
      2.33: (0.531, 0.101, -0.431)}
QN = {'DV_over_rs': 'DV', 'DM_over_rs': 'DM', 'DH_over_rs': 'DH'}
TR = {0.295: 'BGS', 0.51: 'LRG1', 0.706: 'LRG2', 0.934: 'LRG3+ELG1', 1.321: 'ELG2', 1.484: 'QSO', 2.33: 'Lya'}
zq = [(round(float(z), 3), QN[q]) for z, q in zip(ds2.z_eff, ds2.quantities)]
sig_file = np.sqrt(np.diag(ds2.cov))
sig_t4 = np.array([T4[z][1] if q == 'DV' else (T4[z][0] if q == 'DM' else T4[z][1]) for z, q in zq])
corr_file = ds2.cov / np.outer(sig_file, sig_file)
corr_t4 = np.eye(len(zq))
for i, (z, q) in enumerate(zq):
    for j, (z2, q2) in enumerate(zq):
        if i != j and z == z2:
            corr_t4[i, j] = T4[z][2]


def M_from_cov(C):
    K = precompute_kernels(ds2.z_eff, ds2.quantities, C, EJ.W0_GRID, EJ.WA_GRID)
    return float(np.exp(mixture_log_e_from_residuals((ds2.data - K['mu_null'])[:, None], K)[0]))


C_t4 = corr_t4 * np.outer(sig_t4, sig_t4)
C_sig_t4_corr_file = corr_file * np.outer(sig_t4, sig_t4)
C_sig_file_corr_t4 = corr_t4 * np.outer(sig_file, sig_file)
C_l3 = ds2.cov.copy(); i3 = [i for i, (z, q) in enumerate(zq) if z == 0.934]; C_l3[np.ix_(i3, i3)] = C_t4[np.ix_(i3, i3)]
C_l3_sig_only = ds2.cov.copy()
for i in i3:
    for j in i3:
        C_l3_sig_only[i, j] = corr_file[i, j] * sig_t4[i] * sig_t4[j]
hyb = dict(released=M_from_cov(ds2.cov), tableIV_sigma_and_corr=M_from_cov(C_t4), tableIV_sigma_file_corr=M_from_cov(C_sig_t4_corr_file),
           file_sigma_tableIV_corr=M_from_cov(C_sig_file_corr_t4), only_LRG3ELG1_block_tableIV=M_from_cov(C_l3), only_LRG3ELG1_sigmas_tableIV=M_from_cov(C_l3_sig_only))
ratio = {f'{TR[z]} {q}': float(sig_file[i] / sig_t4[i] - 1) for i, (z, q) in enumerate(zq)}
rfile = {TR[z]: float(corr_file[i, i + 1]) for i, (z, q) in enumerate(zq) if i + 1 < len(zq) and zq[i + 1][0] == z}
rt4 = {TR[z]: T4[z][2] for z in T4 if z != 0.295}
for k, v in ratio.items():
    say(f"  sigma_file/sigma_TableIV - 1: {k:14s} {100 * v:6.2f}%")
say("  r(DM,DH) file:", {k: round(v, 3) for k, v in rfile.items()}, " Table IV:", rt4)
for k, v in hyb.items():
    say(f"  M_DR2 with {k:32s} = {v:8.3f}")
rec('tableIV_vs_released', dict(sigma_ratio_minus_1=ratio, r_file=rfile, r_tableIV=rt4, M_DR2_variants=hyb,
                                means_identical_to_tableIV_note='the released mean vector matches Table IV; only the covariance differs'),
    'DR2 Results II Table IV (errors and D_M-D_H correlations as printed; the mean vector is identical) against the released CobayaSampler desi_gaussian_bao_ALL_GCcomb covariance: per-quantity error ratios, correlations, and M_DR2 under the released covariance (33.97), the full Table IV covariance (68.4), Table IV errors with file correlations, file errors with Table IV correlations, and the LRG3+ELG1 block alone (52.9; errors only).')

# ------------------------------------------------------------------ 6. null test at Cov = 0.6 C1
say("\n=== 6. null test: admissibility of the quoted cross-covariances ===")
K1 = precompute_kernels(ds1.z_eff, ds1.quantities, ds1.cov, EJ.W0_GRID, EJ.WA_GRID)
dr2_to_dr1, _ = build_bin_matching(ds1, ds2)
pairs = [(i, j) for j, i in enumerate(dr2_to_dr1) if i >= 0]
m1 = np.array([i for i, _ in pairs]); m2 = np.array([j for _, j in pairs])
C1M = ds1.cov[np.ix_(m1, m1)]; C2M = ds2.cov[np.ix_(m2, m2)]
K2 = precompute_kernels(ds2.z_eff, ds2.quantities, ds2.cov, EJ.W0_GRID, EJ.WA_GRID)
e1 = ds1.data - K1['mu_null']; e2 = ds2.data - K2['mu_null']
d_c = e2[m2] - e1[m1]
n = len(pairs)
gev = linalg.eigh(C2M, C1M, eigvals_only=True)
amax = float(np.sqrt(gev.min()))


def chi2_at(a):
    V = (1 - 2 * a) * C1M + C2M
    c = float(d_c @ np.linalg.solve(V, d_c))
    return c, float(stats.chi2.sf(c, n)), float(np.linalg.eigvalsh(C2M - a * a * C1M).min()), float(np.linalg.eigvalsh(V).min())


nt = {}
for a in (0.0, 0.25, 1 / 3, 0.40, 0.45, 0.46, 0.47, 0.5, 0.6):
    c, p, mineig_joint, mineig_V = chi2_at(a)
    nt[f'{a:.4f}'] = dict(chi2=c, p=p, joint_law_exists=bool(mineig_joint >= 0), min_eig_C2_minus_a2_C1=mineig_joint, min_eig_Var_d=mineig_V)
    say(f"  alpha = {a:.4f}: chi2 = {c:7.2f} p = {p:.3g}  joint Gaussian law exists: {mineig_joint >= 0} (min eig C2 - a^2 C1 = {mineig_joint:.3e}; min eig Var d = {mineig_V:.3e})")
rec('null_test_admissibility', dict(alpha_max=amax, rows=nt),
    'z_eff-corrected DR2-DR1 null test chi2 = d^T [(1-2a) C1 + C2]^-1 d (11 dof) at several a, with the PSD check of the implied joint law [[C1, a C1],[a C1, C2]] (needs C2 - a^2 C1 PSD, i.e. a <= alpha_max = 0.4701). Above alpha_max (e.g. a = 0.6) no joint Gaussian law with the published marginals exists; the difference covariance (1-2a) C1 + C2 can still be positive definite, so the chi2 is computable but tests an empty hypothesis.')

# ------------------------------------------------------------------ 7. DR1-side offsets, closed form
say("\n=== 7. closed-form null mean of M_joint(1/3) under DR1-side coherent offsets ===")
_, D1 = EJ.build_delta_matrix(ds1.z_eff, ds1.quantities, EJ.W0_GRID, EJ.WA_GRID)
_, D2 = EJ.build_delta_matrix(ds2.z_eff, ds2.quantities, EJ.W0_GRID, EJ.WA_GRID)
a = 1 / 3
Sy = C2M - a * a * C1M
Qy = np.linalg.inv(Sy)
A1 = D1 @ np.linalg.inv(ds1.cov)
Dm = D2[:, m2] - a * D1[:, m1]
Ay = Dm @ Qy
s1 = np.sqrt(np.diag(C1M)); s2 = np.sqrt(np.diag(C2M))
G = len(EJ.W0_GRID) * len(EJ.WA_GRID)
proj_dr1 = A1[:, m1] @ s1 - a * (Ay @ s1)     # per-grid coefficient of a DR1-side offset s*sigma_DR1 on the matched quantities
proj_dr2 = Ay @ s2
means = {}
for s in (0.1, 0.5, 1.0, 2.0):
    means[f'{s}'] = dict(DR1_plus=float(np.exp(logsumexp(s * proj_dr1) - np.log(G))), DR1_minus=float(np.exp(logsumexp(-s * proj_dr1) - np.log(G))),
                         DR2_plus=float(np.exp(logsumexp(s * proj_dr2) - np.log(G))), DR2_minus=float(np.exp(logsumexp(-s * proj_dr2) - np.log(G))))
    say(f"  s = {s}: E[M] DR1-side +/-: {means[f'{s}']['DR1_plus']:.4g} / {means[f'{s}']['DR1_minus']:.4g};  DR2-side +/-: {means[f'{s}']['DR2_plus']:.4g} / {means[f'{s}']['DR2_minus']:.4g}")
rec('dr1_side_offset_means', dict(means=means, max_abs_coeff_dr1=float(np.abs(proj_dr1).max()), max_abs_coeff_dr2=float(np.abs(proj_dr2).max())),
    'Exact null mean of the alpha=1/3 combined statistic when every matched DR1 measurement is offset by s sigma_DR1 (same sign) versus the same offset pattern on DR2 (s sigma_DR2): E[M] = mean_g exp(c_g s) with c_g = A1_g.sigma_1 - alpha Ay_g.sigma_1 (DR1 side) or Ay_g.sigma_2 (DR2 side). The DR1-side coefficient nearly cancels because DR1 enters the combined statistic twice with opposite signs (Appendix B.2).')

# ------------------------------------------------------------------ 8. arithmetic
say("\n=== 8. weights and budget arithmetic ===")
w2_star = (20 - M1) / (M2 - M1)
wt = {k: 1 / (w[0] * M1 + w[1] * M2) for k, w in dict(equal=(0.5, 0.5), third_third=(1 / 3, 1 / 3), one_third_two_thirds=(1 / 3, 2 / 3), quarter_three_quarters=(0.25, 0.75), dr1_free_half=(0, 0.5)).items()}
cb = NUM['calibration_budget']; sf = NUM['simulated_fpr']; ntn = NUM['null_test']
arith = dict(weight_on_DR2_for_20=w2_star, p_bounds=wt,
             ratio_null_test_50pct_to_tolerance=ntn['offset_for_50pct_power']['value'] / cb['alpha_1_3_tolerance_b_star']['value'],
             ratio_coherent_crossing_to_tolerance=sf['unif_dr2_5pct_crossing']['value'] / cb['alpha_1_3_tolerance_b_star']['value'],
             ratio_adv_rms_to_tolerance=sf['adv_unit_5pct_crossing_per_bin_rms']['value'] / cb['alpha_1_3_tolerance_b_star']['value'],
             bonferroni_two_looks=2 * sig['mc']['p_emp'], bonferroni_ci95=[2 * x for x in sig['mc']['p_emp_ci']],
             exceedances=sig['mc']['exceed'], N=sig['mc']['N'],
             product_unrounded=M1 * M2, average=0.5 * (M1 + M2), p_average=2 / (M1 + M2))
for k, v in arith.items():
    say(f"  {k}: {v}")
rec('arithmetic', arith, 'Weights (DR2 weight needed for S >= 20; Markov bounds under the conventions of the Section 4.1 footnote), the ratios of the null-test 50% power offset and of the simulated 5% crossings to the alpha = 1/3 tolerance, and the exact-tail Bonferroni with its Monte Carlo 95% interval (45 exceedances in 200000 draws).')

RES['meta'] = dict(value=dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), runtime_seconds=round(time.time() - t_start, 1)),
                   definition='(meta)', script=THIS)
ALL = load_numbers()
ALL['paper_numbers_check'] = py(RES)
save_numbers(ALL)
(OUT_DIR / 'verify_paper_numbers.log').write_text('\n'.join(LOG) + '\n')
say(f"\nwrote {OUT} [paper_numbers_check]  ({time.time() - t_start:.0f}s)")
