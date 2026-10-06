#!/usr/bin/env python3
"""Figure 2 of the paper (Section 4.1): figure2_combination_bounds.pdf.

One row per statement about the combined DR1+DR2 evidence, in the order of Section 4.1:
the three statements that need no model of the relation between the releases (DR2 as a
single look, the equal-weight average, the average with the weights fixed for DR3), then
the three model-based statements (one noise fraction common to all bins, the correlations
of DESI's DR1-DR2 consistency check, one noise fraction per bin). Each row is placed at its
e-value on a log axis; the threshold 20 (p = 0.05) is dashed. Model-based rows are drawn
hollow; red at or above 20, blue below.

Every plotted number is read from the numbers file (the working copy results/numbers.json if
present, else the copy of record cross_release/numbers/numbers.json):
  DR2 single look           anchors.M_DR2                                        33.97
  equal-weight average      anchors.average_M                                    17.51
  weights fixed for DR3     weights_rule.information_proportional.S_DR2          11.44
  one common fraction       family_scan.worst_case_M                             64.7
  consistency-check rule    structured_families.M_joint_desi_rule.min/max        28.9 to 33.7
  one fraction per bin      structured_families.per_bin_worst_case.nested_cap    16.2

The figure is 6.1 in wide, the JCAP text width, and is included at width=\textwidth, so
the fonts print at the sizes set here.

Run (from the repository root): python figures/figure2_combination_bounds.py
Writes results/figures/figure2_combination_bounds.pdf.
"""
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
OUT = FIGURES / 'figure2_combination_bounds.pdf'

plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8.5,
                     'ytick.labelsize': 9, 'font.family': 'sans-serif'})
BLUE, RED, GREY = 'steelblue', 'crimson', '0.45'
THRESH = 20.0


def val(d):
    return d['value'] if isinstance(d, dict) and 'value' in d else d


M2 = val(NUM['anchors']['M_DR2'])
S_avg = val(NUM['anchors']['average_M'])
SF = NUM['structured_families']
S_rule = val(NUM['weights_rule']['information_proportional'])['S_DR2']
M_bin = val(SF['per_bin_worst_case'])['nested_cap'][0]
desi = val(SF['M_joint_desi_rule'])
M_desi_min, M_desi_max = desi['min'], desi['max']
M_star = val(NUM['family_scan']['worst_case_M'])

# (label, value or (lo, hi), model-based?)
GROUPS = [
    ('No model of the relation between the releases', [
        ('DR2 alone, as a single look', M2, False),
        ('Equal-weight average of DR1 and DR2', S_avg, False),
        ('Average with the weights fixed for DR3', S_rule, False),
    ]),
    ('Under a model of the relation', [
        ('One noise fraction, all bins (worst case)', M_star, True),
        ("DESI's consistency-check correlations", (M_desi_min, M_desi_max), True),
        ('One noise fraction per bin (worst case)', M_bin, True),
    ]),
]

XL, XR = 4.0, 220.0
fig, ax = plt.subplots(figsize=(6.1, 2.7))
y = 0.0
ylabels, ypos = [], []
header_rows = []
for gi, (title, rows) in enumerate(GROUPS):
    if gi > 0:
        y -= 0.55                                  # gap between the groups
    header_rows.append((y, title))
    y -= 0.85
    for name, v, model in rows:
        ypos.append(y)
        ylabels.append(name)
        if isinstance(v, tuple):
            lo, hi = v
            col = RED if lo >= THRESH else BLUE
            ax.plot([lo, hi], [y, y], '-', color=col, lw=2.2, solid_capstyle='butt', zorder=3)
            for xv in (lo, hi):
                ax.plot([xv], [y], marker='|', color=col, ms=7, mew=1.6, zorder=4)
            ax.text(hi * 1.12, y, f'{lo:.1f} to {hi:.1f}', va='center', ha='left', fontsize=8.5)
        else:
            col = RED if v >= THRESH else BLUE
            face = 'white' if model else col
            ax.plot([v], [y], 'o', ms=7, markerfacecolor=face, markeredgecolor=col,
                    markeredgewidth=1.6, zorder=4)
            txt = f'{v:.2f}' if v in (M2, S_avg, S_rule) else f'{v:.1f}'   # as printed in the paper
            ax.text(v * 1.12, y, txt, va='center', ha='left', fontsize=8.5)
        y -= 1.0

ax.set_yticks(ypos)
ax.set_yticklabels(ylabels)
ax.tick_params(axis='y', length=0)
for yh, title in header_rows:
    ax.text(-0.02, yh, title, transform=ax.get_yaxis_transform(), ha='right', va='center',
            fontsize=8.5, style='italic', color='0.3')

ax.axvline(THRESH, color='0.4', ls='--', lw=1.0, zorder=1)
ax.text(THRESH, header_rows[0][0] + 0.55, '20 ($p = 0.05$)', ha='center', va='bottom',
        fontsize=8.5, color='0.3')
ax.set_xscale('log')
ax.set_xlim(XL, XR)
ax.set_ylim(y + 0.45, header_rows[0][0] + 0.55)
ax.set_xticks([5, 10, 20, 50, 100])
ax.set_xticklabels(['5', '10', '20', '50', '100'])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlabel(r'e-value $M$  (false-positive probability at most $1/M$)')
for side in ('top', 'right', 'left'):
    ax.spines[side].set_visible(False)
ax.grid(axis='x', color='0.9', lw=0.6, zorder=0)

fig.tight_layout()
fig.savefig(OUT, bbox_inches='tight')
plt.close(fig)
print(f'wrote {OUT}')
