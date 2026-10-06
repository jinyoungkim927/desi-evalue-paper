"""
Shared data-loading library for the DESI e-value paper.

Loads DESI DR1/DR2 BAO mean vectors and covariances (CobayaSampler/bao_data
format) into a BAODataset: 13 DR2 measurements across 7 redshift bins, with
tracer labels (BGS, LRG1, LRG2 at z_eff=0.706, LRG3+ELG1, ELG2, QSO, Lya).

Imported by the analysis scripts that produce the paper's numbers/figures;
this module itself produces no paper output. Running it directly prints a
summary of each release for a quick sanity check.

References:
- DESI DR1: arXiv:2404.03002
- DESI DR2: arXiv:2503.14738
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class BAODataPoint:
    """Single BAO measurement."""
    z: float
    value: float
    quantity: str  # 'DM_over_rs', 'DH_over_rs', or 'DV_over_rs'
    tracer: Optional[str] = None  # BGS, LRG, ELG, QSO, Lya


@dataclass
class BAODataset:
    """Complete BAO dataset with measurements and covariance."""
    name: str
    z_eff: np.ndarray
    data: np.ndarray
    cov: np.ndarray
    quantities: List[str]  # quantity name per element ('DM_over_rs', ...)
    tracers: List[str]  # tracer label per element ('LRG2', ...)

    @property
    def errors(self) -> np.ndarray:
        """1-sigma errors from covariance diagonal."""
        return np.sqrt(np.diag(self.cov))

    @property
    def corr(self) -> np.ndarray:
        """Correlation matrix."""
        d = np.sqrt(np.diag(self.cov))
        return self.cov / np.outer(d, d)

    def summary(self) -> str:
        lines = [f"BAO Dataset: {self.name}", f"N measurements: {len(self.data)}", ""]
        lines.append(f"{'z':>6} {'Quantity':>12} {'Value':>12} {'Error':>10} {'Tracer':>8}")
        lines.append("-" * 52)
        for i, (z, q, v, e, t) in enumerate(zip(
            self.z_eff, self.quantities, self.data, self.errors, self.tracers
        )):
            lines.append(f"{z:6.3f} {q:>12} {v:12.4f} {e:10.4f} {t:>8}")
        return "\n".join(lines)


def load_mean_file(filepath: Path) -> List[BAODataPoint]:
    """
    Load BAO measurements from a mean file.

    Format: z value quantity (whitespace separated)
    Lines starting with # are comments.

    Returns: list of BAODataPoint.
    """
    data_points = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                z = float(parts[0])
                value = float(parts[1])
                quantity = parts[2]
                data_points.append(BAODataPoint(z=z, value=value, quantity=quantity))
    return data_points


def load_cov_file(filepath: Path) -> np.ndarray:
    """
    Load covariance matrix from a file.

    Format: whitespace-separated matrix values, one row per line.

    Returns: np.ndarray covariance matrix.
    """
    return np.loadtxt(filepath)


def infer_tracer(z: float, quantity: str, release: str = 'DR2') -> str:
    """Infer tracer type from redshift.

    Returns: str tracer name (e.g. 'BGS', 'LRG1', 'ELG2', 'QSO', 'Lya').
    """
    if release == 'DR2':
        if z < 0.35:
            return 'BGS'
        elif z < 0.55:
            return 'LRG1'
        elif z < 0.75:
            return 'LRG2'
        elif z < 1.0:
            return 'LRG3+ELG1'
        elif z < 1.4:
            return 'ELG2'
        elif z < 2.0:
            return 'QSO'
        else:
            return 'Lya'
    else:  # DR1
        if z < 0.35:
            return 'BGS'
        elif z < 0.55:
            return 'LRG1'
        elif z < 0.75:
            return 'LRG2'
        elif z < 1.0:
            return 'LRG+ELG'
        elif z < 1.35:
            return 'ELG'
        elif z < 2.0:
            return 'QSO'
        else:
            return 'Lya'


def load_desi_data(data_dir: Path, release: str = 'DR2') -> BAODataset:
    """
    Load complete DESI BAO dataset.

    Parameters:
    -----------
    data_dir : Path
        Directory containing the data files
    release : str
        'DR1' or 'DR2'

    Returns:
    --------
    BAODataset with all measurements and covariance
    """
    if release == 'DR2':
        mean_file = data_dir / 'desi_gaussian_bao_ALL_GCcomb_mean.txt'
        cov_file = data_dir / 'desi_gaussian_bao_ALL_GCcomb_cov.txt'
        name = 'DESI DR2'
    else:
        mean_file = data_dir / 'desi_2024_gaussian_bao_ALL_GCcomb_mean.txt'
        cov_file = data_dir / 'desi_2024_gaussian_bao_ALL_GCcomb_cov.txt'
        name = 'DESI DR1'

    data_points = load_mean_file(mean_file)

    z_eff = np.array([dp.z for dp in data_points])
    data = np.array([dp.value for dp in data_points])
    quantities = [dp.quantity for dp in data_points]
    tracers = [infer_tracer(dp.z, dp.quantity, release) for dp in data_points]

    cov = load_cov_file(cov_file)

    n = len(data)
    if cov.ndim == 1:
        cov = np.array([[cov[0]]])
    elif cov.shape[0] != n:
        raise ValueError(f"Covariance matrix shape {cov.shape} doesn't match data size {n}")

    return BAODataset(
        name=name,
        z_eff=z_eff,
        data=data,
        cov=cov,
        quantities=quantities,
        tracers=tracers
    )


def get_theory_vector(dataset: BAODataset, predictions: Dict) -> np.ndarray:
    """
    Extract theory predictions in the same order as the data vector.

    Parameters:
    -----------
    dataset : BAODataset
        The data to match
    predictions : dict
        Output from compute_bao_predictions() with arrays for each quantity

    Returns:
    --------
    np.ndarray matching the data vector structure
    """
    theory = np.zeros(len(dataset.data))

    for i, (z, q) in enumerate(zip(dataset.z_eff, dataset.quantities)):
        z_idx = np.argmin(np.abs(predictions['z'] - z))
        if 'DM' in q:
            theory[i] = predictions['DM_over_rd'][z_idx]
        elif 'DH' in q:
            theory[i] = predictions['DH_over_rd'][z_idx]
        elif 'DV' in q:
            theory[i] = predictions['DV_over_rd'][z_idx]
        else:
            raise ValueError(f"Unknown quantity: {q}")

    return theory


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1] / 'data'
    print(load_desi_data(base_dir / 'dr2', 'DR2').summary())
    print()
    print(load_desi_data(base_dir / 'dr1', 'DR1').summary())

