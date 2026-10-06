# desi-evalue-paper

Code and data for *An E-value Reanalysis of DESI's Dynamical Dark Energy Signal*
(Jinyoung Kim, David F. Mota and Andrius Tamosiunas, 2026; arXiv:2607.28918, submitted to JCAP).

The code computes a mixture e-value for LCDM against w0waCDM from each DESI BAO data release
alone (DR1, DR2) at a fixed Planck background, and combines the releases by a weighted average
with weights fixed in advance, which keeps the false-detection probability at every release below
the stated level whatever the relation between the releases: DR2 alone gives M_DR2 = 33.97
(p <= 0.0295), the equal-weight average of DR1 and DR2 is 17.51 (p <= 0.0571), and the weights
fixed for DR3 in proportion to observing time give 11.44 (p <= 0.0874). It then computes the
model-based combinations that assume how the two releases are related, and their worst cases:
over all joint laws consistent with the two published covariances the combined statistic can be
0 or arbitrarily large; one noise fraction common to all bins gives 64.7 at worst, the
correlations DESI uses for its own DR1-DR2 consistency check give 28.9 to 33.7, and one fraction
per bin gives 16.2. Finally it localises the evidence: leave-one-out and per-bin e-values place it
in the LRG2 bin, without which M_DR2 falls to 0.49.

## Repository layout

- `code/`: the per-release analyses (Tables 1, 2, 4 and 5, Section 4.2 and 4.3, Appendices B.1
  and B.3 to B.8). Each script writes a JSON summary to `results/`.
- `cross_release/`: the cross-release analyses of Section 4.1 and Appendix B.2, the DR3 weight rule
  of Section 4.4, the resolution and covariance checks, the forecast checks and a script that
  recomputes the numbers the paper quotes. These scripts import from `code/` and write to
  `results/`.
  - `cross_release/simulations/`: the Monte Carlo experiments behind the simulated
    false-positive rates and power of Appendix B.2.
  - `cross_release/inputs/`: outputs of record that the cross-release scripts read: the JSON and
    logs of the simulations, and in `inputs/results/` the `results/*.json` written by `code/*.py`.
    A script reads `results/` when the corresponding `code/` script has been run and falls back
    to these copies otherwise, so every script runs from a fresh copy of the repository.
  - `cross_release/numbers/numbers.json`: the numbers file of record (see below).
- `figures/`: the scripts for the four figures of the paper; they write to `results/figures/`.
- `desi_evalue/`: a compact importable package with the main results as functions, pinned by the
  tests in `tests/`. It is independent of `code/`; both implement the same statistics.
- `data/`: the DESI DR1 and DR2 BAO mean vectors and covariances.
- `results/`: created on first run and not tracked.

## Requirements

Python 3 with numpy and scipy; matplotlib for the figure scripts; pytest for the tests. Tested
with Python 3.14.3, numpy 2.4.3, scipy 1.17.1 and matplotlib 3.10.8 on macOS.

```bash
pip install -r requirements.txt
```

No network access is needed. All scripts resolve paths relative to the repository root, so they
run from any location; the commands below are given from the repository root.

## Per-release analyses (`code/`)

| command | what it computes | paper |
|---|---|---|
| `python code/eprocess_demo.py` | Default-prior mixture e-values M_DR1 = 1.05, M_DR2 = 33.97; prior sensitivity | Sec. 4, Table 1 (Default row) |
| `python code/regrow_analysis.py` | Table 1: the four flat boxes and the three REGROW Fisher-ellipse priors; Fisher distance of the DR2 MLE from LCDM, 4.06 | Table 1; Sec. 4.3, App. B.5 |
| `python code/literature_priors.py` | Table 1 again, plus the Section 4.3 quintessence priors: freezing box M_DR2 = 14.8 on a 320 x 320 grid, thawing band 1.1e3 (1129) and its DR1 value 15; exits with an error if either moves by more than 1% | Sec. 4.3 |
| `python code/loo_per_bin.py` | Table 2: leave-one-out e-value of each bin (LRG2 55.98; average 10.17) and the per-bin REGROW delta = 2 column (product 4.46, mean 1.52) | Table 2, Sec. 4.2 |
| `python code/compute_lrg2_drop.py` | M_DR2 without LRG2: 0.49 | abstract, Sec. 4.2, Sec. 5, Table 4 |
| `python code/bin_level_eprocess.py` | per-bin Default-prior mixtures: product 1.8e-3, mean 3.5, LRG2 22.8 | App. B.4 |
| `python code/loo_benchmark.py` | parametric bootstrap of the leave-one-out concentration under a smooth w(z) and under LCDM (500 and 200 draws): 55%, 22%, 23%, 18%, 3.2%, 37%, 27% | Table 5, Sec. 4.2, App. B.4 |
| `python code/verify_sigma_mc.py` | Monte Carlo null tail of M_DR2 under the DR2 Gaussian likelihood, N = 2e5: sigma_emp = 3.69, tail 2.25e-4 | Sec. 4.1, App. B.1 |
| `python code/min_concentration_delta_star.py` | Delta chi^2 = 16.86 (3.70 sigma); Fisher-Gaussian shell family: delta* in [0.87, 13.5], delta_max 2.7, M_max 1.7e2 | Sec. 4.1, Sec. 4.3, App. B.5 |
| `python code/background_sensitivity.py` | Table 4: M_DR1, M_DR2 and M_DR2 without LRG2 under four fixed Planck backgrounds (M_DR2 from 8.70 to 1417, a factor 163) | Table 4, App. B.3 |
| `python code/universal_inference.py` | prior-free split-sample e-value within DR2, E_UI = 78 (400 splits) | Sec. 4.1 |
| `python code/joint_cmb_analysis.py` | BAO + compressed Planck: MLE (-0.818, -0.695), Delta chi^2 = 14.78 (3.42 sigma), leave-one-out 7.07, mixtures 7.2 / 2.19 / 0.91 / 0.091; stops on any gate failure | Sec. 4.1, App. B.7 |
| `python code/data_split_variants.py` | data-split e-values by bin partition: 1.43; the alternating split changes by a factor of about 66 | App. B.6 |
| `python code/power_calibration.py` | power of the data-split e-value by simulation (median 1.2 under w0wa) | App. B.6 |
| `python code/posterior_mean_cross_prediction.py` | cross-prediction e-values for Pantheon+, DES-Y5 and Union3: 9.6, 47 and 44 at the published MAPs, 342, 240 and 128 averaged over the posteriors | App. B.8 |
| `python code/dr3_power_analysis.py` | DR2 BAO-only MLE (-0.856, -0.430); Monte Carlo of the e-value added by two new DR3 bins at z = 1.7 and 2.5 under both truths (P(M_DR3 >= 20) = 0.18 under w0waCDM, 0.17 under LCDM) | Sec. 3, Fig. 1, Sec. 4.4 |
| `python code/eprocess_joint.py` | the model-based combination under one noise fraction alpha_1 = 1/3: 66.7, with the mean under other truths (162 at alpha_1 = 0.25, over 1e13 if the releases were independent) | App. B.2 |
| `python code/eprocess_hierarchical_mc.py` | Monte Carlo diagnostics of the one-fraction model; also provides the DR1-DR2 bin matching that `cross_release/` imports | App. B.2 |
| `python code/footnote_joint_fpr_check.py` | false-positive rate of per-look Wilks testing at DR1 and DR2 under the one-fraction model (0.085, below the independent-looks 0.0975 and Bonferroni's 0.10) | Sec. 1 (footnote) |
| `python code/lrg_residuals.py` | D_H/r_d residuals of the LRG2 and LRG3+ELG1 bins under LCDM (-2.1 and +0.3 sigma) and at the DR2 MLE | Fig. 1 (supporting) |

`code/data_loader.py`, `code/cosmology.py`, `code/evalue_analysis.py` and
`code/extended_analysis.py` are libraries imported by the scripts above. Run times on a laptop
(Apple silicon, one process): most scripts take seconds; `eprocess_demo.py` about 30 s,
`literature_priors.py` one to three minutes (the 320 x 320 freezing box),
`universal_inference.py` about 2 min, `posterior_mean_cross_prediction.py` about 6 min,
`loo_benchmark.py` about 10 min and `power_calibration.py` about 30 min. Their outputs of record
are in `cross_release/inputs/results/`; running the scripts reproduces them to better than one
part in a million (only timing fields differ).

## Cross-release analyses (`cross_release/`)

The scripts build on each other through the working copy `results/numbers.json`; run them in the
order listed, or run any one alone, in which case it starts from the copy of record
`cross_release/numbers/numbers.json`. The section of the numbers file each script writes is given
in brackets.

| command | what it computes | paper | time |
|---|---|---|---|
| `python cross_release/one_fraction_family.py` | [`anchors`, `null_test`, `alpha_max`, `family_scan`, `calibration_budget`, `simulated_fpr`, `simulations_context`, `series`, `figures`] M_DR1 = 1.0535, M_DR2 = 33.97, average 17.51, p <= 0.0295 and 0.0571; the DR1-DR2 null test (chi^2, pulls, variance ratios 0.22 to 0.49 with median 0.32); alpha_max = 0.4701; the one-fraction family with its worst case 64.7 at alpha_1 = 0.31 (p <= 0.0155) and M_joint(1/3) = 66.7; the calibration budget (alpha_1 = 1/3 and family rows of Table 3, tolerance 0.016 sigma); the simulated false-positive baseline 0.03%; runs `shift_bound.py` and reads `cross_release/inputs/` | abstract, Sec. 1, 4.1, 5, App. B.2, Table 3, Fig. 4 | 20 s |
| `python cross_release/structured_families.py` | [`structured_families`] the correlations of DESI's DR1-DR2 consistency check: implied fractions 0.25 to 0.49 (median 0.37) and the combined statistic 28.9 / 31.0 / 33.7 (p <= 0.035); the nested rule 26.0; one fraction per bin, worst case 16.2 (p <= 0.0619); the statistic without the DR2 quasar term, 90.2; the calibration tolerance of DR2 alone (0.011 sigma) and of the Narrow sub-box (0.062 sigma); the z_eff-corrected null test (chi^2 = 11.40, p = 0.41, largest pull 1.96 in LRG2 D_M, LRG2 bin chi^2 = 4.1); the weighting conventions of the Section 4.1 footnote (0.0435, 0.0389, 0.0857, 0.0589, 16.99, 0.576); the exact-tail Bonferroni 4.5e-4; the background worst case 8.70; M_DR2 with the Table IV errors and correlations (68.4; 52.9 for the LRG3+ELG1 block alone; 6.4% largest error excess) | Sec. 3, 4.1, 4.2, App. B.2, B.3, Table 3, Figs. 2 and 4 | 15 s |
| `python cross_release/weights_rule.py` | [`weights_rule`] the DR3 weight rule: weights in proportion to observing time, DR1 : DR2 : DR3 = 1 : 3 : 5 years, so 1/9, 3/9, 5/9; S_DR2 = 11.44, p <= 0.0874, and DR3 needs M_DR3 of about 15 for the average to reach 20; equal thirds over the three releases (11.68, p <= 0.0857); the information ratios implied by the published variances | abstract, Sec. 4.1, 4.4, 5, Fig. 2 | 1 s |
| `python cross_release/unrestricted_class.py` | [`unrestricted_class`] over all joint Gaussian laws consistent with the two published covariances: rank-one laws inside one bin that drive the model-based statistic to 0 and to arbitrarily large values; its worst and best case when the correlation between the releases is capped at rho in every direction (2.3, 0.09, 0.005, 0.003 at rho = 0.3, 0.5, 0.7, 0.75; block-diagonal by bin 6.2, 0.35, 0.016, 0.007); the alpha_1 = 1/3 member's largest such correlation, 0.709; one fraction per matched quantity, worst case 1.09 | abstract, Sec. 1, 4.1, 5, App. B.2, Fig. 2 | 5 to 12 min |
| `python cross_release/unrestricted_class_small_caps.py` | [`unrestricted_class_small_caps`] caps below 0.3: the worst case falls below 20 once the cap exceeds about 0.18 (0.22 block-diagonal) and below 1 at 0.35 (0.43); the collapse to 0 also holds when every same-quantity correlation is non-negative | App. B.2 | 40 s |
| `python cross_release/verify_paper_numbers.py` | [`paper_numbers_check`] recomputes the numbers the paper quotes from `results/` and prints each against its printed value: sigma conversions (3.70, 3.69, 2.18; Bonferroni thresholds 1.96, 2.24, 2.58), Table 1 with every Markov bound rounded up, Table 2 (LRG2 share 78.6%, LRG2 + LRG3+ELG1 91%, six-bin average 2.54), the Table IV decomposition, the null test at alpha_1 = 0.47 (chi^2 = 19.0, p = 0.061), the Bonferroni Monte Carlo interval 3.2 to 5.8e-4, and the values of Appendices B.3 to B.8 | Sec. 1, 4.1, 4.2, Tables 1, 2, 4, App. B.1 to B.8 | 10 s |
| `python cross_release/resolution_and_covariance_checks.py` | [`resolution_and_covariance_checks`] the quintessence priors at several grid resolutions; the released DR1 covariance against DR1 Table 1 (M_DR1 1.054 against 1.028); the DR1-side calibration tolerance 1.66 sigma_DR1, about a hundred times the DR2 side | Sec. 3, 4.3, App. B.2 | 30 s |
| `python cross_release/quintessence_resolution.py` | [`resolution_and_covariance_checks.quintessence_resolution`] the freezing box on 160 x 160 and 320 x 320 grids (14.8) and the thawing band with its upper w0 edge moved between -0.80 and -0.85 (0.98 to 1.13e3) | Sec. 4.3 | 3 to 4 min |
| `python cross_release/dr3_forecast.py` | [`dr3_forecast`] the two forecast statements of Section 4.4: the new high-redshift bins separate LCDM from the DR2 MLE by less than 0.5 sigma (0.45); halving the LRG2 errors gives a median e-value of 1.7e4 under the DR2 MLE and 0.011 under LCDM | Sec. 4.4 | 5 s |
| `python cross_release/verify_tableIV_vs_bao_data.py` | the Section 3 comparison of the released DR2 covariance with Table IV of DESI DR2 Results II (prints only) | Sec. 3 | 1 s |
| `python cross_release/shift_bound.py` | the data-allowed offset region against the breakdown radius of the alpha_1 = 1/3 statistic (also run by `one_fraction_family.py`) | App. B.2 (supporting) | 5 to 20 s |

`cross_release/unrestricted_class_core.py` is the set-up shared by the two `unrestricted_class`
scripts and `cross_release/_paths.py` resolves the paths; neither is run directly.

`cross_release/simulations/` holds the Monte Carlo experiments read by `one_fraction_family.py`:
`fpr_calibration_offsets.py` and `fpr_crossing.py` (false-positive rate of the alpha_1 = 1/3
statistic under coherent and adversarial calibration offsets, with the baseline rate of about
0.03% of Appendix B.2), `fpr_cross_release_worlds.py` (false-positive rates of six statistics in
four cross-release worlds), `power_dr2_mle.py` and `power_full_family.py` (power of the same
statistics under the DR2 MLE), and `mean_shift_robustness.py` (direction-free offset budgets).
Their outputs of record are in `cross_release/inputs/` and `cross_release/inputs/logs/`; running
them again (one to two minutes each, `power_dr2_mle.py` a few seconds) writes fresh copies to
`results/simulations/`.

## Figures (`figures/`)

| command | figure | paper |
|---|---|---|
| `python figures/figure1_bao_data.py` | the DR2 BAO residuals from LCDM with the w0waCDM prediction at the DR2 MLE, and the (w0, wa) plane with the Default box, the DR2 Fisher ellipses, the thawing band and the freezing box; reads `data/dr2/` | Fig. 1, Sec. 3 |
| `python figures/figure2_combination_bounds.py` | one row per statement about the combined DR1+DR2 evidence, in the order of Section 4.1, each at its e-value, with the model-based statements drawn hollow; reads the numbers file | Fig. 2, Sec. 4.1 |
| `python figures/figure3_loo_bins.py` | the leave-one-out e-value of each bin and the six aggregate e-values; reads `loo.json`, `lrg2_drop.json` and `per_bin_regrow.json` from `results/` or from the copies of record | Fig. 3, Sec. 4.2 |
| `python figures/figure4_cross_release_checks.py` | the model-based statistic across the one-fraction family with its minimum, alpha_1 = 1/3 and the per-bin worst case marked, and the DR2 minus DR1 pulls on the 11 matched quantities; reads the numbers file | Fig. 4, App. B.2 |

Each writes `results/figures/<script name>.pdf` and runs in a few seconds.

## The numbers file

`cross_release/numbers/numbers.json` records every number the paper prints, and the quantities
behind them, as `{"value": ..., "definition": ..., "script": ...}`, grouped in sections named
after the script that writes them (bracketed in the table above). The `figures` section lists
the four figures and the entries each one reads. Running the scripts writes a working copy to
`results/numbers.json`; the copy of record is never overwritten, so the two can be compared.

## The `desi_evalue` package

```python
from desi_evalue import analyses as A

dr1, dr2 = A.load_releases()
A.running_mixture(dr1, dr2)       # M_DR1 = 1.05, M_DR2 = 33.97
A.prior_sensitivity(dr1, dr2)     # Table 1
A.localisation(dr2)               # Table 2
A.joint_sequential(dr1, dr2)      # the one-fraction combination at alpha_1 = 1/3: 66.7
A.background_sensitivity(dr1, dr2)
```

Functions return their numbers rather than printing them.

Modules: `constants.py` (assumptions, fixed parameters, alternative choices), `cosmology.py`
(w0waCDM distances, Gaussian BAO likelihood), `data.py` (DR1/DR2 loading, bin structure,
cross-release matching), `evalues.py` (mixture, leave-one-out, split and universal-inference
e-values), `sequential.py` (the one-fraction joint construction), `analyses.py` (the paper's
results).

The prior grid is part of the test specification. A uniform mixture over a finite pre-specified
set of points is an e-value, so a different resolution is a different but equally valid test; the
Default box gives 33.97 at 30 x 30 and 35.75 at 120 x 120. The freezing box is small and its
integrand peaks sharply near one corner, so it is evaluated on a finer grid (`FREEZING_GRID_N` in
the package; 320 x 320 in `code/literature_priors.py`, as in Section 4.3).

The leave-one-out folds and the per-bin products are taken at bin level, not measurement level:
the two measurements within a redshift bin are anti-correlated, and the independence these
constructions rely on holds only across bins. The leave-one-out e-value is the average of the
per-bin E_k, which is an e-value by linearity of expectation although the training sets of the K
folds overlap.

## Tests

```bash
python -m pytest
```

The 21 tests pin the paper's numbers through the `desi_evalue` package and take about 40 s.
`pytest.ini` puts the repository root on the import path, so `pytest` also works from the root.

## Data

DESI DR1 and DR2 BAO measurements (mean and covariance) from
[CobayaSampler/bao_data](https://github.com/CobayaSampler/bao_data), the files DESI distributes
for likelihood analyses, unchanged. `data/dr1` and `data/dr2` hold the combined `ALL_GCcomb` files
the analysis uses and the per-tracer files from the same repository.

## Citation

```
Kim, J., Mota, D. F., Tamosiunas, A. (2026)
"An E-value Reanalysis of DESI's Dynamical Dark Energy Signal"
arXiv:2607.28918 (submitted to JCAP)
```

## License

MIT, see `LICENSE`.
