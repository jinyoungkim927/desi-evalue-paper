#!/usr/bin/env python3
"""Figure 3 of the paper (Section 4.2): figure3_loo_bins.pdf.

(a) the leave-one-out e-value of each DR2 redshift bin (Table 2), red where it reaches the
    threshold 20 = 1/0.05;
(b) the aggregate e-values: the full mixture over all seven bins (M_DR2 = 33.97), the
    leave-one-out average, the per-bin-independent product, the leave-one-out average and
    the full mixture without LRG2, and the look-elsewhere-corrected mean.

Values are read from the results JSON written by code/ (results/ if the code/ scripts have
been run, else the copies of record in cross_release/inputs/results/):
  loo.json             per-bin leave-one-out e-values and their average
  lrg2_drop.json       the full mixture with all bins and without LRG2
  per_bin_regrow.json  the per-bin-independent product and the look-elsewhere-corrected mean
The leave-one-out average without LRG2 is derived from loo.json (drop z = 0.706).

Run (from the repository root): python figures/figure3_loo_bins.py
Writes results/figures/figure3_loo_bins.pdf.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cross_release'))
from _paths import FIGURES, result_json  # noqa: E402

FIGURES.mkdir(parents=True, exist_ok=True)
OUT = FIGURES / "figure3_loo_bins.pdf"

plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 18,
                     'xtick.labelsize': 16, 'ytick.labelsize': 16})
THRESH = 20.0  # 1/alpha at alpha = 0.05

# --- per-bin LOO e-values (loo.json) ---
loo = result_json('loo.json')
per_bin = {float(z): e for z, e in loo['per_bin_e'].items()}
zeff = sorted(per_bin)                      # [0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330]
E_loo = [per_bin[z] for z in zeff]
bins = ['BGS', 'LRG1', 'LRG2', 'LRG3\n+ELG1', 'ELG2', 'QSO', r'Ly$\alpha$']
assert len(bins) == len(zeff) == 7
Z_LRG2 = 0.706

# --- aggregate e-values (lrg2_drop.json, loo.json, per_bin_regrow.json) ---
lrg2_drop = result_json('lrg2_drop.json')
regrow = result_json('per_bin_regrow.json')
loo_avg_no_lrg2 = np.mean([e for z, e in per_bin.items() if z != Z_LRG2])

agg_labels = ['Full mixture\n(all 7 bins)', 'LOO average\n(all 7 bins)',
              'Per-bin-independent\nproduct $\\prod_k M_k$', 'LOO average\n(no LRG2)',
              'Look-elsewhere-\ncorrected mean', 'Full mixture\n(no LRG2)']
agg_vals = [lrg2_drop['M_DR2_all_bins'], loo['loo_average'],
            regrow['per_bin_product'], loo_avg_no_lrg2,
            regrow['per_bin_arith_mean'], lrg2_drop['M_DR2_no_LRG2']]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6),
                               gridspec_kw={'width_ratios': [1.3, 1.0]})

# ---- panel (a) ----  red if the bin clears the rejection threshold, else blue
colors = ['crimson' if v >= THRESH else 'steelblue' for v in E_loo]
bars = ax1.bar(range(len(bins)), E_loo, color=colors, alpha=0.85,
               edgecolor='black', linewidth=0.5)
ax1.set_yscale('log')
ax1.axhline(1.0, color='gray', linewidth=1.0, alpha=0.6)
ax1.axhline(THRESH, color='gray', linewidth=1.3, linestyle='--',
            label=r'threshold $20 = 1/0.05$')
ax1.set_xticks(range(len(bins)))
# bin name over its effective redshift; the axis label names z_eff, so that the seven
# labels fit on one row at 14 pt without staggering
xtlabels = [f'{b}\n{z:.3f}' for b, z in zip(bins, zeff)]
ax1.set_xticklabels(xtlabels, fontsize=14)
ax1.set_xlabel(r'redshift bin and its $z_{\mathrm{eff}}$')
ax1.set_ylabel(r'Leave-one-out e-value $E_k^{\mathrm{LOO}}$')
ax1.set_title('(a) Per-bin evidence is localised to LRG2', fontsize=18)
ax1.set_ylim(0.4, 130)
for bar, v in zip(bars, E_loo):
    ax1.text(bar.get_x() + bar.get_width()/2, v*1.12, f'{v:.2f}',
             ha='center', va='bottom', fontsize=14)
# legend upper right, clear of the value label above the LRG2 bar
ax1.legend(fontsize=13, loc='upper right')

# ---- panel (b) ----  same rule: red clears the threshold (only the all-bins mixture), else blue
cols = ['crimson' if v >= THRESH else 'steelblue' for v in agg_vals]
y = np.arange(len(agg_labels))[::-1]
ax2.barh(y, agg_vals, color=cols, alpha=0.85, edgecolor='black', linewidth=0.5)
ax2.set_xscale('log')
ax2.axvline(THRESH, color='gray', linewidth=1.3, linestyle='--')
ax2.axvline(1.0, color='gray', linewidth=1.0, alpha=0.6)
ax2.set_yticks(y)
ax2.set_yticklabels(agg_labels, fontsize=14)
ax2.set_xlabel('e-value')
ax2.set_title(r'(b) Only the all-bins mixture rejects', fontsize=18)
ax2.set_xlim(0.3, 220)
for yi, v in zip(y, agg_vals):
    ax2.text(v*1.18, yi, f'{v:.2f}',
             va='center', fontsize=14)

plt.tight_layout()
plt.savefig(OUT, bbox_inches='tight', dpi=200)
plt.close(fig)
print(f'wrote {OUT}')
