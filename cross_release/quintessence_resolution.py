#!/usr/bin/env python3
"""Grid-resolution dependence of the Section 4.3 quintessence-prior e-values.

Section 4.3 quotes M_DR2^freezing = 14.8 ("does not reject") for the freezing box
w0 in [-0.95, -0.75], wa in [0, 0.3] on a 320 x 320 grid, and M_DR2^thawing = 1.1e3 for the
Caldwell-Linder band, with "moving the upper w0 edge between -0.80 and -0.85 changes it
between 0.98 and 1.13e3". The freezing box is small and its integrand peaks near one corner,
so its uniform mixture depends on the grid: 21.46 at 20 x 20 (resolution_and_covariance_checks.py
tabulates 20, 40 and 80 points per axis), converging to 14.8 at 320 x 320, the grid
code/literature_priors.py uses. This script evaluates the freezing box on 160 x 160 and
320 x 320 grids and varies the thawing band's upper w0 edge, with the functions of
code/literature_priors.py (flat_box_grid, mixture_log_e_grid).

Writes results/quintessence_resolution.json and
results/numbers.json["resolution_and_covariance_checks"]["quintessence_resolution"]
(value / definition / script).
Run (from the repository root, 3 to 4 minutes): python cross_release/quintessence_resolution.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import ROOT, OUT_DIR, add_code_path, load_numbers, save_numbers  # noqa: E402
add_code_path()
from data_loader import load_desi_data                                   # noqa: E402
from literature_priors import mixture_log_e_grid, flat_box_grid   # noqa: E402

THIS = 'cross_release/quintessence_resolution.py'
t0 = time.time()
ds1 = load_desi_data(ROOT / 'data' / 'dr1', 'DR1')
ds2 = load_desi_data(ROOT / 'data' / 'dr2', 'DR2')
out = {}
for nres in (160, 320):
    g = flat_box_grid(-0.95, -0.75, 0.0, 0.3, n_w0=nres, n_wa=nres)
    out[f'freezing_box_{nres}x{nres}'] = dict(M_DR2=float(np.exp(mixture_log_e_grid(g, ds2))),
                                              M_DR1=float(np.exp(mixture_log_e_grid(g, ds1))))
    print(nres, out[f'freezing_box_{nres}x{nres}'], f'[{time.time() - t0:.0f}s]', flush=True)


def band(w0lo, w0hi, nw=40, na=25):
    pts = []
    for w0 in np.linspace(w0lo, w0hi, nw):
        for wa in np.linspace(-3 * (1 + w0), -(1 + w0), na):
            pts.append((w0, wa))
    return np.array(pts)


for w0lo, w0hi in ((-1.0, -0.85), (-1.0, -0.80), (-1.0, -0.90), (-1.0, -0.875), (-1.0, -0.825)):
    out[f'thawing_band_w0_{w0lo}_{w0hi}'] = float(np.exp(mixture_log_e_grid(band(w0lo, w0hi), ds2)))
    print(w0lo, w0hi, out[f'thawing_band_w0_{w0lo}_{w0hi}'], flush=True)
out['meta'] = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), runtime_seconds=round(time.time() - t0, 1), script=THIS)

OUT_DIR.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUT_DIR / 'quintessence_resolution.json', 'w'), indent=1)
ALL = load_numbers()
ALL.setdefault('resolution_and_covariance_checks', {})['quintessence_resolution'] = dict(
    value=out,
    definition='Freezing box [-0.95,-0.75]x[0,0.3] on 160x160 and 320x320 grids (converges to 14.8, the value of '
               'Section 4.3, evaluated on 320x320 by code/literature_priors.py; 21.46 at 20x20), and '
               'the thawing Caldwell-Linder band with its upper w0 edge at -0.85 (paper), -0.80, -0.90, -0.875, -0.825 '
               '(40x25 grid).',
    script=THIS)
save_numbers(ALL)
print('wrote', OUT_DIR / 'quintessence_resolution.json', 'and', OUT_DIR / 'numbers.json', '[resolution_and_covariance_checks.quintessence_resolution]',
      f'({time.time() - t0:.0f}s)')
