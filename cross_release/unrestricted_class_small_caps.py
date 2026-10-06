#!/usr/bin/env python3
"""Supplement to unrestricted_class.py: small canonical-correlation caps, and a positivity check.

  1. Worst case of the exact combined statistic over the capped classes ||R||_op <= rho for
     rho in {0.05, 0.10, 0.15, 0.20, 0.25} (full and block-diagonal-by-bin), to locate the cap at
     which the worst case falls below the threshold 20 (at rho = 0.3 it is already 2.3 / 6.2).
     At rho = 0 the class is the single law K = 0 and the statistic is 1025
     (anchors.M_joint_alpha_0).
  2. Does the collapse to 0 survive the restriction that each matched quantity's DR1 and DR2
     measurements are non-negatively correlated?  For each two-quantity bin, scan rank-one
     boundary laws R = u v^T (s -> 1) with y~.v = 0 and check whether some have both implied
     same-quantity correlations K_ii >= 0 and no grid point with d~_g.v = 0.

Writes results/numbers.json["unrestricted_class_small_caps"] and results/unrestricted_class_small_caps.log.
Run (from the repository root, about 40 s): python cross_release/unrestricted_class_small_caps.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unrestricted_class_core import *   # noqa: E402,F401,F403

THIS = 'cross_release/unrestricted_class_small_caps.py'
t_start = time.time()
rng = np.random.default_rng(20261006)
LOG, RES = [], {}


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    LOG.append(s)


def rec(key, value, definition):
    RES[key] = dict(value=value, definition=definition, script=THIS)


M0 = float(np.exp(logM(np.zeros((n, n)))))
say(f"K = 0 (independent releases, coherent mixture): M = {M0:.2f}")
caps = [0.05, 0.10, 0.15, 0.20, 0.25]
table = {}
for variant, proj in (('full', proj_full), ('block_diagonal', proj_block)):
    warm = None
    for rho in caps:
        best = None
        for R0 in make_starts(rng, rho, warm):
            R, f, it, resid = pgd(R0, rho, proj, sign=+1.0)
            if best is None or f < best[1]:
                best = (R, f, it, resid)
        warm = best[0]
        table.setdefault(variant, {})[f'{rho}'] = dict(inf_log_M=best[1], inf_M=float(np.exp(best[1])), inf_iters=best[2],
                                                        inf_pg_residual=best[3], inf_structure=describe_R(best[0]))
        say(f"  {variant:14s} rho = {rho:<5} inf M = {np.exp(best[1]):9.4g}  (resid {best[3]:.1e}, {best[2]} it)")
rec('small_caps_table', table, 'Worst case over the capped classes for caps below 0.3 (same optimiser and starts as unrestricted_class.py, 11 starts per cap, warm-started across caps). Upper bounds on the true infimum.')

# threshold crossing, combining with the main table if present
cross = {}
main = load_numbers().get('unrestricted_class', {}).get('capped_class_table', {}).get('value', {})
for variant in table:
    rows = {**{k: v['inf_log_M'] for k, v in table[variant].items()}, **{k: v['inf_log_M'] for k, v in main.get(variant, {}).items()}}
    rh = np.array([0.0] + [float(k) for k in rows]); lm = np.array([np.log(M0)] + list(rows.values()))
    o = np.argsort(rh); rh, lm = rh[o], lm[o]
    for name, lev in (('rho_where_inf_crosses_20', np.log(THR)), ('rho_where_inf_crosses_1', 0.0)):
        if lm.min() < lev < lm.max():
            k = int(np.argmax(lm < lev))
            cross[f'{variant}:{name}'] = float(rh[k - 1] + (lev - lm[k - 1]) * (rh[k] - rh[k - 1]) / (lm[k] - lm[k - 1]))
        else:
            cross[f'{variant}:{name}'] = None
    say(f"  {variant}: crossings {cross}")
rec('inf_threshold_crossings', cross, 'Cap rho at which the worst case falls below 20 and below 1, interpolating log M linearly between the tabulated caps (including rho = 0, M = 1025, and the caps of unrestricted_class.py).')

# positivity check of the collapse construction
say("\n=== collapse with non-negative same-quantity correlations ===")
pos = {}
for b, ii in zip(bin_names, idx):
    if len(ii) != 2:
        continue
    e1b, e2b = et1[ii], et2[ii]
    L1b, L2b = L1[np.ix_(ii, ii)], L2[np.ix_(ii, ii)]
    n1 = np.linalg.norm(e1b)
    perp1 = np.array([-e1b[1], e1b[0]]) / n1
    found = []
    for tv in np.linspace(0, 2 * np.pi, 721)[:-1]:
        v = np.array([np.cos(tv), np.sin(tv)])
        t = e2b @ v                                   # need u . e1b = t with |u| = 1 (s = 1)
        if abs(t) > n1:
            continue
        for sgn in (+1, -1):
            u = (t / n1 ** 2) * e1b + sgn * np.sqrt(max(0.0, 1 - (t / n1) ** 2)) * perp1
            Kb = L1b @ np.outer(u, v) @ L2b
            proj = Dt2[:, ii] @ v - Dt1[:, ii] @ u
            if np.all(np.diag(Kb) >= 0) and np.abs(proj).min() > 0:
                found.append((float(tv), int(sgn), float(np.abs(proj).min()), [float(x) for x in np.diag(Kb) / (s1[ii] * s2[ii])]))
    if found:
        tv, sgn, mp, corr = max(found, key=lambda z: z[2])
        v = np.array([np.cos(tv), np.sin(tv)]); t = e2b @ v
        u = (t / n1 ** 2) * e1b + sgn * np.sqrt(max(0.0, 1 - (t / n1) ** 2)) * perp1
        rows = {}
        for s in (0.9, 0.99, 0.999, 1 - 1e-5, 1 - 1e-6):
            R = np.zeros((n, n)); R[np.ix_(ii, ii)] = s * np.outer(u, v)
            rows[f'{s:.6f}'] = logM(R)
        pos[b] = dict(n_directions_found=len(found), example_same_quantity_correlations=corr, min_abs_projection=mp, log_M_vs_s=rows)
        say(f"  {b:10s}: {len(found)} boundary laws with both same-quantity correlations >= 0; example corr = {np.round(corr, 2)}, log M at s = 0.99, 0.999, 1-1e-6 -> {rows['0.990000']:.2f}, {rows['0.999000']:.2f}, {rows['0.999999']:.2f}")
    else:
        pos[b] = dict(n_directions_found=0)
        say(f"  {b:10s}: none found")
rec('collapse_with_nonnegative_same_quantity_correlation', pos,
    'Rank-one boundary laws (R = s u v^T inside one bin, y~.v = 0, s -> 1) scanned over 720 directions v (two u solutions each): the number of them for which both implied same-quantity correlations corr(X1_i, X2_i) are non-negative and no grid point has d~_g.v = 0, with an example and its log M as s -> 1. If any exist, the collapse to 0 survives the restriction to positively correlated releases.')

RES['meta'] = dict(value=dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), runtime_seconds=round(time.time() - t_start, 1)),
                   definition='(meta)', script=THIS)
NUM = load_numbers()
NUM['unrestricted_class_small_caps'] = py(RES)
save_numbers(NUM)
(OUT_DIR / 'unrestricted_class_small_caps.log').write_text('\n'.join(LOG) + '\n')
say(f"\nwrote {OUT} [unrestricted_class_small_caps]  ({time.time() - t_start:.0f}s)")
