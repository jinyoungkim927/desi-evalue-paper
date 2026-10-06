#!/usr/bin/env python3
"""Figure 4 of the paper (Appendix B.2): figure4_cross_release_checks.pdf.

(a) the model-based combined statistic M_joint(alpha_1) across the one-fraction family
    Cov(DR1, DR2) = alpha_1 C_DR1 on the matched quantities, over [0, alpha_1,max), with the
    family's minimum and alpha_1 = 1/3 marked, the region where C_DR2 - alpha_1^2 C_DR1 is
    not positive semi-definite hatched, the threshold 20 dashed and a horizontal line at the
    per-bin worst case (structured_families.per_bin_worst_case.nested_cap);
(b) DR2 - DR1 pulls on the 11 matched quantities, z_eff-corrected, at alpha_1 = 1/3
    (structured_families.null_test_zeff_corrected.pulls_1_3), Lya ordered D_M, D_H.

All numbers are read from the numbers file (the working copy results/numbers.json if present,
else the copy of record cross_release/numbers/numbers.json). The figure is 6.1 in wide, the
JCAP text width, and is included at width=\textwidth, so the fonts print at the sizes set
here.

Run (from the repository root): python figures/figure4_cross_release_checks.py
Writes results/figures/figure4_cross_release_checks.pdf.
"""
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cross_release'))
from _paths import FIGURES, load_numbers  # noqa: E402

NUM = load_numbers()
FIGURES.mkdir(parents=True, exist_ok=True)
OUT = FIGURES / 'figure4_cross_release_checks.pdf'

plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'font.family': 'sans-serif'})
BLUE, RED = 'steelblue', 'crimson'
THRESH = 20.0
SYM = r'\alpha_1'


def ceil_to(x, nd):
    f = 10 ** nd
    return math.ceil(x * f - 1e-9) / f


def val(d):
    return d['value'] if isinstance(d, dict) and 'value' in d else d


s = val(NUM['series'])
SF = NUM['structured_families']
a = np.array(s['alpha'])
M = np.array(s['M_joint'])
a_max = val(NUM['alpha_max']['alpha_max_exact'])
a_star = val(NUM['family_scan']['worst_case_alpha'])
M_star = val(NUM['family_scan']['worst_case_M'])
M13 = val(NUM['anchors']['M_joint_alpha_1_3'])
M_bin = val(SF['per_bin_worst_case'])['nested_cap'][0]
ntc = val(SF['null_test_zeff_corrected'])
chi2, pval = ntc['chi2_1_3']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.1, 2.6),
                               gridspec_kw={'width_ratios': [1.0, 1.1], 'wspace': 0.32})

# ---------------------------------------------------------------- panel (a)
YLO, YHI = 3, 3e4
XHI = 0.50
keep = M < YHI * 5
ax1.plot(a[keep], M[keep], '-', color=BLUE, lw=1.8, zorder=3)
ax1.set_yscale('log')
ax1.axvspan(a_max, XHI, facecolor='0.88', edgecolor='0.6', hatch='//', lw=0, zorder=1)
ax1.axvline(a_max, color='0.45', lw=1.0, zorder=2)
ax1.text((a_max + XHI) / 2, 250, 'no joint law', rotation=90, ha='center', va='center',
         fontsize=7.5, color='0.2',
         bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none'))
ax1.axhline(THRESH, color='gray', ls='--', lw=1.0, zorder=2)
ax1.text(0.012, THRESH * 1.12, 'threshold 20', color='0.3', fontsize=7.5, va='bottom', ha='left')
# per-bin worst case (one fraction per bin, nested caps)
ax1.axhline(M_bin, color='0.45', ls='-', lw=1.2, zorder=2)
ax1.text(0.012, M_bin * 0.88, f'one fraction per bin: {M_bin:.1f}', color='0.3', fontsize=7.5,
         va='top', ha='left')
# the family's minimum and the alpha_1 = 1/3 member
ax1.plot([a_star], [M_star], 'o', color=RED, ms=6, zorder=5,
         markeredgecolor='black', markeredgewidth=0.5)
ax1.plot([1 / 3], [M13], 'D', color='white', ms=5, zorder=6,
         markeredgecolor='black', markeredgewidth=1.0)
ax1.annotate(f'minimum {M_star:.1f} at ${SYM} = {a_star:.2f}$',
             xy=(a_star, M_star), xytext=(0.02, 160), fontsize=7.5, ha='left', va='bottom',
             arrowprops=dict(arrowstyle='-', color='0.3', lw=0.7, shrinkA=1, shrinkB=4))
ax1.annotate(f'${SYM} = 1/3$: {M13:.1f}',
             xy=(1 / 3, M13), xytext=(0.30, 1300), fontsize=7.5, ha='center', va='bottom',
             arrowprops=dict(arrowstyle='-', color='0.3', lw=0.7, shrinkA=1, shrinkB=4))
ax1.set_xlim(0, XHI)
ax1.set_ylim(YLO, YHI)
ax1.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
ax1.set_xlabel(f'noise fraction ${SYM}$')
ax1.set_ylabel(r'combined statistic $M^{\mathrm{joint}}_{\mathrm{DR2}}$')
ax1.set_title('(a) One noise fraction for all bins')
ax1.grid(True, which='major', alpha=0.2, lw=0.5)

# ---------------------------------------------------------------- panel (b)
labels = list(s['pull_labels'])
iL = [labels.index('Lya DM'), labels.index('Lya DH')]
order = [i for i in range(len(labels)) if i not in iL] + iL
labels = [labels[i] for i in order]
pulls = np.array([ntc['pulls_1_3'][l] for l in labels])
QTEX = {'DV': r'$D_V$', 'DM': r'$D_M$', 'DH': r'$D_H$'}
TR = {'Lya': r'Ly$\alpha$', 'LRG3+ELG1': 'LRG3\n+ELG1'}
x = np.arange(len(pulls))
ax2.axhspan(-1, 1, color='0.92', zorder=0)
for yv in (-2, 2):
    ax2.axhline(yv, color='gray', ls='--', lw=0.9, zorder=1)
ax2.axhline(0, color='0.3', lw=0.7, zorder=1)
ax2.bar(x, pulls, width=0.62, color=BLUE, alpha=0.85, edgecolor='black',
        linewidth=0.4, zorder=3)
i_max = int(np.argmax(np.abs(pulls)))
ax2.text(x[i_max], pulls[i_max] + 0.1, f'{pulls[i_max]:+.2f}', ha='center',
         va='bottom', fontsize=7.5, zorder=4,
         bbox=dict(boxstyle='square,pad=0.05', facecolor='white', edgecolor='none'))
ax2.set_xticks(x)
ax2.set_xticklabels([QTEX[l.split()[1]] for l in labels], fontsize=7.5)
groups = []
for i, l in enumerate(labels):
    tr = l.split()[0]
    if not groups or groups[-1][0] != tr:
        groups.append([tr, [i]])
    else:
        groups[-1][1].append(i)
for tr, idx in groups:
    xc = np.mean(idx)
    ax2.text(xc, -0.14, TR.get(tr, tr), transform=ax2.get_xaxis_transform(),
             ha='center', va='top', fontsize=7.5)
    if len(idx) > 1:
        ax2.plot([idx[0] - 0.3, idx[-1] + 0.3], [-0.125, -0.125], color='0.5', lw=0.6,
                 transform=ax2.get_xaxis_transform(), clip_on=False)
for tr, idx in groups[:-1]:
    ax2.axvline(idx[-1] + 0.5, color='0.8', lw=0.8, zorder=0)
ax2.text(0.98, 0.04, r'$\chi^2 = %.2f$ for 11 d.o.f. ($p = %.2f$)' % (chi2, pval),
         transform=ax2.transAxes, ha='right', va='bottom', fontsize=7.5, color='0.2',
         bbox=dict(boxstyle='square,pad=0.15', facecolor='white', edgecolor='none'))
ax2.set_xlim(-0.6, len(pulls) - 0.4)
ax2.set_ylim(-2.7, 2.7)
ax2.set_yticks([-2, -1, 0, 1, 2])
ax2.set_ylabel(r'pull of DR2 $-$ DR1 ($\sigma$) at $' + SYM + r' = 1/3$')
ax2.set_title('(b) Null test on the 11 matched quantities')

fig.tight_layout()
fig.savefig(OUT, bbox_inches='tight')
plt.close(fig)
print(f'wrote {OUT}')
