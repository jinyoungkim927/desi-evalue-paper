"""DESI DR1/DR2 BAO measurements in the public CobayaSampler format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import BIN_EDGES_DR1, BIN_EDGES_DR2, BIN_MATCH_RTOL

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

_FILES = {
    "DR2": ("dr2", "desi_gaussian_bao_ALL_GCcomb_{}.txt"),
    "DR1": ("dr1", "desi_2024_gaussian_bao_ALL_GCcomb_{}.txt"),
}


def _bin_label(z, release):
    edges = BIN_EDGES_DR2 if release == "DR2" else BIN_EDGES_DR1
    return next(label for edge, label in edges if z < edge)


@dataclass(frozen=True)
class BAOData:
    """A BAO release: measurements, covariance, and their bin structure."""

    release: str
    z: np.ndarray
    values: np.ndarray
    cov: np.ndarray
    quantities: tuple[str, ...]
    bins: tuple[str, ...]

    def __len__(self):
        return len(self.values)

    @property
    def bin_names(self):
        """Distinct bin labels in redshift order."""
        return tuple(dict.fromkeys(self.bins))

    def indices_for(self, *bin_names):
        """Row indices belonging to the named bins."""
        wanted = set(bin_names)
        return np.array([i for i, b in enumerate(self.bins) if b in wanted], dtype=int)

    def subset(self, idx):
        """Restrict to the given rows, keeping the covariance block."""
        idx = np.asarray(idx, dtype=int)
        return BAOData(self.release, self.z[idx], self.values[idx],
                       self.cov[np.ix_(idx, idx)],
                       tuple(self.quantities[i] for i in idx),
                       tuple(self.bins[i] for i in idx))

    def drop_bins(self, *bin_names):
        """Restrict to everything outside the named bins."""
        drop = set(bin_names)
        return self.subset([i for i, b in enumerate(self.bins) if b not in drop])


def load(release="DR2", data_dir=DATA_DIR):
    """Load a DESI BAO release."""
    folder, pattern = _FILES[release]
    rows = np.genfromtxt(data_dir / folder / pattern.format("mean"), dtype=None,
                         encoding=None, names=("z", "value", "quantity"))
    rows = np.atleast_1d(rows)
    z = rows["z"].astype(float)
    cov = np.atleast_2d(np.loadtxt(data_dir / folder / pattern.format("cov")))
    if cov.shape[0] != len(z):
        raise ValueError(f"{release}: covariance is {cov.shape}, data has {len(z)} rows")
    return BAOData(release, z, rows["value"].astype(float), cov,
                   tuple(str(q) for q in rows["quantity"]),
                   tuple(_bin_label(zi, release) for zi in z))


def match_across_releases(dr1: BAOData, dr2: BAOData, rtol=BIN_MATCH_RTOL):
    """Pair up measurements present in both releases.

    A DR2 row matches a DR1 row when it is the same observable at the same
    effective redshift to within ``rtol``. Returns (dr1_idx, dr2_idx,
    dr2_unmatched); the unmatched DR2 rows are the genuinely new information.
    """
    pairs = [(i, j) for j, (zj, qj) in enumerate(zip(dr2.z, dr2.quantities))
             for i, (zi, qi) in enumerate(zip(dr1.z, dr1.quantities))
             if qi == qj and abs(zi - zj) <= rtol * zj]
    seen_dr1, seen_dr2 = set(), set()
    dr1_idx, dr2_idx = [], []
    for i, j in pairs:
        if i not in seen_dr1 and j not in seen_dr2:
            seen_dr1.add(i)
            seen_dr2.add(j)
            dr1_idx.append(i)
            dr2_idx.append(j)
    unmatched = [j for j in range(len(dr2)) if j not in seen_dr2]
    return np.array(dr1_idx), np.array(dr2_idx), np.array(unmatched, dtype=int)
