#!/usr/bin/env python3
"""Sensitivity of the headline mixture e-values to the fixed Planck background.

Recomputes the Default-prior (30x30) running mixture on DR1, DR2, and DR2
without the LRG2 bin for four fixed backgrounds (h, Omega_m, r_d/Mpc): the
paper's baseline (a) and three self-consistent Planck 2018 columns (b)-(d).
Reproduces Table 4, the background-sensitivity table of Appendix B.3:
M_DR2 swings from 8.70 (+lensing+BAO) to 1417 (TT,TE,EE+lowE)
across columns -- two orders of magnitude -- while M_DR2 without LRG2 never
reaches the alpha = 0.05 threshold (max 5.55).

The background enters through the CosmologyParams field defaults, which every
grid alternative and the LCDM null inherit, so each row is computed in a fresh
subprocess with the BG_H / BG_OM / BG_RD environment overrides (cosmology.py)
set before interpreter start.

Writes results/background_sensitivity.json.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from data_loader import load_desi_data
from evalue_analysis import grow_evalue
from compute_lrg2_drop import drop_bin

ROOT = Path(__file__).resolve().parents[1]

# (row label, Planck 2018 column, h, Omega_m, r_d/Mpc). Row (a) mixes columns
# (h, Om from the +lensing+BAO chain; r_d from TT,TE,EE+lowE); (b)-(d) are
# self-consistent columns of Planck 2018 Table 2.
BACKGROUNDS = [
    ('a', 'this work',             0.6766, 0.3111, 147.05),
    ('b', 'TT,TE,EE+lowE+lensing', 0.6736, 0.3153, 147.09),
    ('c', '+lensing+BAO',          0.6766, 0.3111, 147.21),
    ('d', 'TT,TE,EE+lowE',         0.6727, 0.3166, 147.05),
]


def compute_one_background():
    """Default-prior mixture e-values under the background set by BG_* env vars
    (paper baseline when unset).

    Returns a dict {'M_DR1', 'M_DR2', 'M_DR2_no_LRG2'} of mixture e-values.
    Only meaningful for a changed background when run in a fresh process:
    the BG_* overrides are read when cosmology.py is imported.
    """
    dr1 = load_desi_data(ROOT / 'data' / 'dr1', 'DR1')
    dr2 = load_desi_data(ROOT / 'data' / 'dr2', 'DR2')
    dr2_no_lrg2 = drop_bin(dr2)

    def mixture(ds):
        return float(grow_evalue(ds.data, ds.cov, ds.z_eff, ds.quantities).e_value)

    return {
        'M_DR1': mixture(dr1),
        'M_DR2': mixture(dr2),
        'M_DR2_no_LRG2': mixture(dr2_no_lrg2),
    }


def main():
    rows = []
    for label, column, h, om, rd in BACKGROUNDS:
        env = dict(os.environ, BG_H=repr(h), BG_OM=repr(om), BG_RD=repr(rd))
        proc = subprocess.run([sys.executable, __file__, '--single'],
                              env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.exit(f"background ({label}) subprocess failed:\n{proc.stderr}")
        rows.append({'label': label, 'planck_column': column,
                     'h': h, 'omega_m': om, 'rd': rd,
                     **json.loads(proc.stdout)})

    header = (f"{'Background (h, Om, rd/Mpc)':<52s} {'M_DR1':>8s} "
              f"{'M_DR2':>9s} {'no LRG2':>8s}")
    print(header)
    print("-" * len(header))
    for r in rows:
        bg = f"({r['label']}) {r['planck_column']} ({r['h']}, {r['omega_m']}, {r['rd']})"
        print(f"{bg:<52s} {r['M_DR1']:>8.2f} {r['M_DR2']:>9.2f} "
              f"{r['M_DR2_no_LRG2']:>8.2f}")

    out_dir = ROOT / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'background_sensitivity.json', 'w') as f:
        json.dump(rows, f, indent=2)


if __name__ == '__main__':
    if '--single' in sys.argv[1:]:
        json.dump(compute_one_background(), sys.stdout)
    else:
        main()
