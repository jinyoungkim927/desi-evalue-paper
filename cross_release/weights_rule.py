#!/usr/bin/env python3
"""The DR3 weight rule of Section 4.4: weights in proportion to observing time.

Section 4.4 fixes, before DR3 exists, the weights with which the per-release e-values of
DESI's five-year programme enter the weighted average of Proposition 5: each planned release
enters in proportion to the information it carries, taken as its observing time, so DR1 (1 yr)
: DR2 (3 yr) : DR3 (5 yr) = 1/9 : 3/9 : 5/9.  Equivalently the weights are the information
fractions 1/5, 3/5, 1 reached at the three planned looks, normalised to sum to one.

From the anchors M_DR1 and M_DR2 of the numbers file this script computes

  (a) the observing-time weights: S_DR2 = w1 M_DR1 + w2 M_DR2, the Markov bound 1/S_DR2
      (exact, and rounded UP as the paper prints it), and the M_DR3 at which the
      three-release average reaches 20;
  (b) the same for equal thirds over the three releases;
  (c) the information ratios implied by the published variances (diag C_DR2 over diag C_DR1
      on the 11 matched quantities, null_test section), the empirical support for taking
      observing time as the measure of information;
  (d) two variants of the rule (DR1 counted as its nominal 13 months; weights by information
      increment, 1:2:2, which is what a linear error-spending function allocates in a nested
      design).

Output: results/numbers.json["weights_rule"] (value, definition, script), other sections
untouched; a log beside it.

Run (from the repository root, under a second): python cross_release/weights_rule.py
Reads the numbers written by one_fraction_family.py and structured_families.py
(results/numbers.json, or the copy of record in cross_release/numbers/).
"""
import json
import sys
import time
from decimal import Decimal, ROUND_CEILING
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import OUT_DIR, NUMBERS, load_numbers, save_numbers  # noqa: E402
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = NUMBERS                             # the working copy written by save_numbers
LOGF = OUT_DIR / 'weights_rule.log'
THIS = 'cross_release/weights_rule.py'
THRESHOLD = 20.0

t0 = time.time()
RES, LOG = {}, []


def say(s=''):
    print(s)
    LOG.append(s)


def v(d):
    return d['value'] if isinstance(d, dict) and 'value' in d else d


def rec(key, value, definition):
    RES[key] = dict(value=value, definition=definition, script=THIS)


def ceil_at(x, nd):
    """x rounded UP at the nd-th decimal place (a p-bound is never rounded down)."""
    return float(Decimal(repr(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_CEILING))


def ceil_sig(x, n):
    """x rounded UP to n significant figures."""
    e = Decimal(repr(x)).adjusted()
    return float(Decimal(repr(x)).quantize(Decimal(1).scaleb(e - n + 1), rounding=ROUND_CEILING))


def fmt_p(p):
    """The paper prints bounds below 0.1 at four decimal places and bounds above at three."""
    nd = 4 if p < 0.1 else 3
    return f'{ceil_at(p, nd):.{nd}f}', nd


def combo(name, weights, M1, M2, years=None):
    """Weighted average at DR2 and what DR3 must deliver.  weights: Fractions (DR1, DR2, DR3)."""
    w1, w2, w3 = (float(w) for w in weights)
    assert abs(w1 + w2 + w3 - 1) < 1e-12, name
    S2 = w1 * M1 + w2 * M2
    p = 1 / S2
    p_str, nd = fmt_p(p)
    need = (THRESHOLD - S2) / w3 if w3 > 0 else float('inf')
    out = dict(weights=[w1, w2, w3], weights_exact=[str(w) for w in weights],
               S_DR2=S2, p_bound_exact=p, p_bound_printed=p_str,
               p_bound_rounded_up_at=f'{nd} decimal places',
               p_bound_ceil_4sf=ceil_sig(p, 4),
               M_DR3_needed_for_20=need,
               S_DR3_if_M_DR3_equals_20=S2 + w3 * THRESHOLD,
               S_DR3_if_M_DR3_equals_M_DR2=S2 + w3 * M2)
    if years is not None:
        out['observing_time_years'] = years
    say(f'{name}')
    say(f'  weights (DR1, DR2, DR3) = {[str(w) for w in weights]} = ({w1:.4f}, {w2:.4f}, {w3:.4f})')
    say(f'  S_DR2 = {S2:.6f}   1/S_DR2 = {p:.8f}  -> printed {p_str} (rounded up at {nd} dp; '
        f'rounded up at 4 s.f. {ceil_sig(p, 4):.5f})')
    say(f'  M_DR3 needed for the average to reach {THRESHOLD:g}: {need:.4f}')
    say(f'  S_DR3 if M_DR3 = 20: {S2 + w3 * THRESHOLD:.3f};  if M_DR3 = M_DR2: {S2 + w3 * M2:.3f}')
    return out


N = load_numbers()
M1 = v(N['anchors']['M_DR1'])
M2 = v(N['anchors']['M_DR2'])
S_eq = v(N['anchors']['average_M'])
VR = v(N['null_test']['variance_ratio_C2_over_C1'])
VR_LABELS = v(N['null_test']['labels'])
VR_MMM = v(N['null_test']['variance_ratio_min_median_max'])
THIRDS_CHECK = v(N['structured_families']['weights_at_DR2'])['third_each_three_releases']

say(f'anchors: M_DR1 = {M1:.10f}, M_DR2 = {M2:.10f}, equal-weight average {S_eq:.6f}')
rec('inputs', dict(M_DR1=M1, M_DR2=M2, equal_weight_average=S_eq,
                   variance_ratio_min_median_max=VR_MMM),
    'Anchors read from the numbers file: anchors.M_DR1, anchors.M_DR2, anchors.average_M; '
    'null_test.variance_ratio_min_median_max (diag C_DR2 / diag C_DR1 on the 11 matched quantities).')

# ------------------------------------------------------------------ the rule
YEARS = dict(DR1=1, DR2=3, DR3=5)
rec('observing_time_years', YEARS,
    'Observing time of each planned release of the five-year programme, in survey years as DESI labels '
    'them (Y1, Y3, Y5): DR1 covers the first year of main-survey observations (nominally 13 months, '
    '14 May 2021 to 14 June 2022), DR2 the first three years (to June 2024), DR3 the full five-year '
    'survey (completed 16 April 2026). Used as the proxy for the information each release carries.')
total_years = sum(YEARS.values())
frac_info = [Fraction(YEARS[k], YEARS['DR3']) for k in ('DR1', 'DR2', 'DR3')]      # 1/5, 3/5, 1
rec('information_fractions', [float(f) for f in frac_info],
    'Information fraction reached at each planned look, observing time over the five-year total: '
    '1/5, 3/5, 1. The information-proportional weights are these fractions normalised to sum to one.')

W_INFO = [Fraction(YEARS[k], total_years) for k in ('DR1', 'DR2', 'DR3')]            # 1/9, 3/9, 5/9
assert W_INFO == [Fraction(1, 9), Fraction(3, 9), Fraction(5, 9)]
assert [w / sum(frac_info) for w in frac_info] == W_INFO
say()
info = combo('(a) information-proportional weights 1/9, 3/9, 5/9 (observing time 1 : 3 : 5 yr)', W_INFO, M1, M2, YEARS)
info['S_DR2_over_equal_weight_average'] = info['S_DR2'] / S_eq
info['cost_factor_vs_equal_weight'] = S_eq / info['S_DR2']
info['M_DR3_needed_printed'] = 'about 15'
info['weights_sum'] = 1.0
say(f'  S_DR2 / equal-weight average = {info["S_DR2_over_equal_weight_average"]:.4f} '
    f'(cost factor {info["cost_factor_vs_equal_weight"]:.3f}); weights sum to 1, nothing is left for releases after DR3')
rec('information_proportional', info,
    'The rule of Section 4.4: weights in proportion to the observing time of each planned release '
    'of the five-year programme, DR1 : DR2 : DR3 = 1 : 3 : 5 yr, so w = (1/9, 3/9, 5/9) (sum 1; releases after '
    'the five-year programme lie outside the rule). S_DR2 = w1 M_DR1 + w2 M_DR2 from the unrounded anchors; '
    'p_bound_exact = 1/S_DR2; p_bound_printed is that bound rounded UP at the fourth decimal place, the '
    'paper\'s convention for bounds below 0.1 (the fourth significant figure is 0, so the 4 s.f. value is the '
    'same); M_DR3_needed_for_20 solves S_DR2 + w3 M_DR3 = 20 (the paper says "about 15"). Printed: 11.44, '
    '0.0874, about 15.')

say()
W_THIRDS = [Fraction(1, 3)] * 3
thirds = combo('(b) equal thirds 1/3, 1/3, 1/3', W_THIRDS, M1, M2)
thirds['cost_factor_vs_equal_weight'] = S_eq / thirds['S_DR2']
assert abs(thirds['S_DR2'] - THIRDS_CHECK['S']) < 1e-9 and abs(thirds['p_bound_exact'] - THIRDS_CHECK['p']) < 1e-12
say(f'  agrees with structured_families.weights_at_DR2.third_each_three_releases (S = {THIRDS_CHECK["S"]:.6f})')
rec('equal_thirds', thirds,
    'One third per release over DR1, DR2, DR3. Same quantities as information_proportional. '
    'Agrees with structured_families.weights_at_DR2.third_each_three_releases; its bound 0.0857 is the '
    'Section 4.1 footnote value for 1/3 per release over three releases.')

# ------------------------------------------------------------------ empirical support: variances
say()
info_ratio = [1 / r for r in VR]
mmm = dict(min=min(VR), median=sorted(VR)[len(VR) // 2], max=max(VR))
assert [mmm['min'], mmm['median'], mmm['max']] == VR_MMM
ir = dict(min=1 / mmm['max'], median=1 / mmm['median'], max=1 / mmm['min'])
time_ratio = YEARS['DR2'] / YEARS['DR1']
say(f'(c) published variances, DR2 over DR1, 11 matched quantities: min {mmm["min"]:.4f}, median {mmm["median"]:.4f}, '
    f'max {mmm["max"]:.4f}  (printed 0.22 to 0.49, median 0.32)')
say(f'    implied information ratio DR2 : DR1 = 1/ratio: {ir["min"]:.3f} to {ir["max"]:.3f}, median {ir["median"]:.3f}; '
    f'observing-time ratio {time_ratio:g}; median / time ratio = {ir["median"] / time_ratio:.3f}')
for lab, r in zip(VR_LABELS, VR):
    say(f'      {lab:14s} var ratio {r:.4f}  info ratio {1 / r:.3f}')
rec('information_ratio_from_published_variances',
    dict(variance_ratio_DR2_over_DR1=mmm, variance_ratio_printed='0.22 to 0.49, median 0.32',
         information_ratio_DR2_to_DR1=ir, observing_time_ratio_DR2_to_DR1=time_ratio,
         median_information_ratio_over_time_ratio=ir['median'] / time_ratio,
         per_quantity={lab: dict(variance_ratio=r, information_ratio=1 / r) for lab, r in zip(VR_LABELS, VR)}),
    'Empirical support for taking observing time as the measure of information: the published DR2 variances '
    'are about a third of DR1\'s on the 11 matched quantities (null_test.variance_ratio_C2_over_C1), '
    'so the information ratio DR2 : DR1 is 2.0 to 4.5 with median 3.15, within 5% of the observing-time ratio 3. '
    'Printed in Section 4.4 as "ratios 0.22 to 0.49, median 0.32".')

# ------------------------------------------------------------------ two variants of the rule
say()
Y13 = dict(DR1=Fraction(13, 12), DR2=Fraction(3), DR3=Fraction(5))
tot13 = sum(Y13.values())
W13 = [Y13[k] / tot13 for k in ('DR1', 'DR2', 'DR3')]
var13 = combo('variant (not used): DR1 counted as its nominal 13 months, weights 13/109, 36/109, 60/109', W13, M1, M2,
              {k: float(x) for k, x in Y13.items()})
say()
W_INC = [Fraction(1, 5), Fraction(2, 5), Fraction(2, 5)]
var_inc = combo('variant (not used): weights by information increment 1 : 2 : 2 yr (linear error spending in a nested design)',
                W_INC, M1, M2)
rec('variants_not_used',
    dict(dr1_as_13_months=var13, increment_proportional_1_2_2=var_inc),
    'Two variants of the rule, not used in the paper. (i) DR1 counted as its nominal 13 months '
    'instead of one survey year: the bound moves from 0.0874 to 0.0882, so the whole-year convention is '
    'immaterial. (ii) Weights in proportion to the information ADDED by each release (1, 2, 2 yr), which is what a '
    'linear error-spending function allocates to successive looks at nested data; the paper\'s rule weights the '
    'information each release CARRIES, because each release is scored whole by its own e-value rather than '
    'incrementally, and the analogy in Section 4.4 is to allocation by information fraction, not to the linear '
    'spending function itself.')

# ------------------------------------------------------------------ the strings the paper prints
rec('printed_strings',
    dict(weights='1/9, 3/9, 5/9', S_DR2='11.44', p_bound='0.0874', M_DR3_needed='about 15',
         variance_ratios='0.22 to 0.49, median 0.32', observing_time='one, three and five years',
         equal_thirds_p_bound='0.0857'),
    'The strings the paper prints for the rule (Section 4.4) and, for equal thirds, the bound in the Section 4.1 footnote.')

RES['meta'] = dict(value=dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), runtime_seconds=round(time.time() - t0, 2),
                              threshold=THRESHOLD),
                   definition='(meta)', script=THIS)

ALL = load_numbers()
ALL['weights_rule'] = RES
save_numbers(ALL)
LOGF.write_text('\n'.join(LOG) + '\n')
say(f'\nwrote {OUT} [weights_rule] and {LOGF}  ({time.time() - t0:.2f}s)')
