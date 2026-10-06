"""Check the Section 3 claims about the released DR2 covariance against Table IV.

Claims in Section 3 of the paper ("Data and likelihood construction"):
  - the CobayaSampler bao_data DR2 file has 13 measurements in 7 bins whose means
    match Table IV of DESI DR2 Results II (arXiv:2503.14738v3 / PRD 112, 083515);
  - the covariance is block-diagonal by bin, with within-bin D_M-D_H correlation
    between about -0.35 and -0.49;
  - the file errors exceed the Table IV posterior standard deviations by up to 6.4%
    (LRG3+ELG1 D_M), whose correlation is also weaker (-0.35 against -0.42), and by
    about 2% or less elsewhere.

Table IV values are transcribed from the LaTeX source of arXiv:2503.14738v3
(main.tex lines 575-581).

Run (from the repository root): python cross_release/verify_tableIV_vs_bao_data.py
Reads data/dr2/desi_gaussian_bao_ALL_GCcomb_{mean,cov}.txt, the files distributed in
https://github.com/CobayaSampler/bao_data (desi_bao_dr2); no network access is needed.
"""
from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DR2 = os.path.join(os.path.dirname(HERE), "data", "dr2")
# origin of the two files in data/dr2 (CobayaSampler/bao_data, desi_bao_dr2):
SOURCE = "https://raw.githubusercontent.com/CobayaSampler/bao_data/master/desi_bao_dr2/"
FILES = ["desi_gaussian_bao_ALL_GCcomb_mean.txt", "desi_gaussian_bao_ALL_GCcomb_cov.txt"]

# Table IV (arXiv:2503.14738v3): tracer -> (z_eff, D_M/r_d, sigma, D_H/r_d, sigma, r_MH)
# BGS reports only D_V/r_d = 7.942 +/- 0.075.
TABLE_IV = {
    "LRG1": (0.510, 13.588, 0.167, 21.863, 0.425, -0.459),
    "LRG2": (0.706, 17.351, 0.177, 19.455, 0.330, -0.404),
    "LRG3+ELG1": (0.934, 21.576, 0.152, 17.641, 0.193, -0.416),
    "ELG2": (1.321, 27.601, 0.318, 14.176, 0.221, -0.434),
    "QSO": (1.484, 30.512, 0.760, 12.817, 0.516, -0.500),
    "Lya": (2.330, 38.988, 0.531, 8.632, 0.101, -0.431),
}
BGS_DV = (0.295, 7.942, 0.075)


def fetch() -> tuple[list[tuple[float, float, str]], np.ndarray]:
    rows = []
    with open(os.path.join(DATA_DR2, FILES[0])) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            z, v, q = line.split()
            rows.append((float(z), float(v), q))
    cov = np.loadtxt(os.path.join(DATA_DR2, FILES[1]))
    return rows, cov


def main() -> None:
    rows, cov = fetch()
    assert cov.shape == (13, 13) and len(rows) == 13, "expected 13 measurements"
    sig = np.sqrt(np.diag(cov))
    zs = sorted({z for z, _, _ in rows})
    print(f"{len(rows)} measurements in {len(zs)} bins: z_eff = {zs}")

    # block-diagonal check
    off = [(i, j) for i in range(13) for j in range(13) if rows[i][0] != rows[j][0] and cov[i, j] != 0]
    print("non-zero off-block covariance entries:", len(off))

    print("\nmeans (file vs Table IV) and errors (file / Table IV - 1):")
    worst = (None, 0.0)
    for i, (z, v, q) in enumerate(rows):
        if q == "DV_over_rs":
            z4, v4, s4 = BGS_DV
            name = "BGS D_V"
        else:
            name = [k for k, t in TABLE_IV.items() if abs(t[0] - z) < 1e-6][0]
            t = TABLE_IV[name]
            v4, s4 = (t[1], t[2]) if q == "DM_over_rs" else (t[3], t[4])
            name += " D_M" if q == "DM_over_rs" else " D_H"
        excess = sig[i] / s4 - 1
        if excess > worst[1]:
            worst = (name, excess)
        print(f"  {name:16s} mean {v:9.4f} vs {v4:7.3f} (diff {v - v4:+.4f}); "
              f"sigma {sig[i]:.4f} vs {s4:.3f} -> {100 * excess:+.2f}%")
    print(f"largest excess: {worst[0]} {100 * worst[1]:.2f}%")

    print("\nwithin-bin D_M-D_H correlations (file vs Table IV r_MH):")
    for name, t in TABLE_IV.items():
        idx = [i for i, (z, _, q) in enumerate(rows) if abs(z - t[0]) < 1e-6]
        i, j = idx
        r = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])
        print(f"  {name:10s} {r:+.4f} vs {t[5]:+.3f}")


if __name__ == "__main__":
    main()
