#!/usr/bin/env python3
"""
Support module for the CMB and supernova cross-prediction analyses.

Provides the compressed-CMB statistics and SN+CMB summary constraints used by
the paper's cross-prediction results:
- CMBCompressed / compute_cmb_predictions: the Chen-Huang-Wang (2019) Planck
  distance priors (r_s(z*) = 144.65 Mpc), imported by joint_cmb_analysis.py.
- PANTHEON_PLUS / DESY5 / UNION3: the per-compilation SN+CMB (w0, wa) MAP and
  posterior widths feeding the cross-prediction E values of Appendix B.8, imported by
  posterior_mean_cross_prediction.py.

This module computes no headline numbers itself; it supplies the inputs.

References:
- Chen, Huang & Wang (2019): arXiv:1808.05724 (Planck distance priors)
- Brout et al. (2022): Pantheon+
- DES Collaboration (2024): DES-SN5YR
- Rubin et al. (2023): Union3
"""

import numpy as np
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cosmology import CosmologyParams, DM, C_LIGHT_KM_S


@dataclass
class CMBCompressed:
    """
    Compressed CMB statistics from Planck 2018.

    These capture most of the constraining power of CMB for dark energy:
    - R: Shift parameter = sqrt(Omega_m) * H0 * D_M(z*) / c
    - la: Acoustic scale = pi * D_M(z*) / r_s(z*)
    - omega_b: Physical baryon density Omega_b h^2

    Values follow the compressed Planck 2018 TT,TE,EE+lowE likelihood of
    Chen, Huang & Wang (2019), arXiv:1808.05724 (distance priors from the
    Planck final release), with sound horizon at decoupling
    r_s(z*) = 144.65 Mpc -- distinct from the BAO drag-epoch r_d = 147.05 Mpc.
    """
    R: float = 1.7502
    la: float = 301.471
    omega_b: float = 0.02237

    sigma_R: float = 0.0046
    sigma_la: float = 0.090
    sigma_omega_b: float = 0.00015

    corr_R_la: float = 0.52
    corr_R_wb: float = -0.36
    corr_la_wb: float = -0.64

    z_star: float = 1089.92

    @property
    def data(self) -> np.ndarray:
        return np.array([self.R, self.la, self.omega_b])

    @property
    def cov(self) -> np.ndarray:
        """Covariance matrix from correlations and errors."""
        sigmas = np.array([self.sigma_R, self.sigma_la, self.sigma_omega_b])
        corr = np.array([
            [1.0, self.corr_R_la, self.corr_R_wb],
            [self.corr_R_la, 1.0, self.corr_la_wb],
            [self.corr_R_wb, self.corr_la_wb, 1.0],
        ])
        return np.outer(sigmas, sigmas) * corr


# Sound horizon at decoupling (z*), Chen et al. 2019 convention.
# NOT the BAO drag-epoch value (cosmo.rd = 147.05 Mpc).
RS_STAR = 144.65  # Mpc


def compute_cmb_predictions(cosmo: CosmologyParams, cmb: CMBCompressed) -> np.ndarray:
    """
    Compute CMB compressed statistics predictions.

    R = sqrt(Omega_m) * H0 * D_M(z*) / c   (comoving D_M, not D_A)
    la = pi * D_M(z*) / r_s(z*)            (r_s(z*) = 144.65 Mpc, not r_d)
    omega_b = Omega_b h^2                  (constant at fixed h)
    """
    D_M = DM(cmb.z_star, cosmo)  # comoving angular-diameter distance, Mpc

    # Shift parameter (uses comoving D_M; the historical D_A variant is
    # smaller by (1+z*) ~ 1090 and gives chi^2 ~ 1e4)
    H0 = 100 * cosmo.h  # km/s/Mpc
    R = np.sqrt(cosmo.omega_m) * H0 * D_M / C_LIGHT_KM_S

    la = np.pi * D_M / RS_STAR

    # Physical baryon density: constant in our fixed-background
    # parameterization (h is fixed; only w0, wa vary)
    omega_b = 0.02237 * (cosmo.h / 0.6766)**2

    return np.array([R, la, omega_b])


@dataclass
class SNConstraint:
    """
    Supernova constraints on (Omega_m, w0, wa) marginalized over other parameters.

    Gaussian approximation to the published SN+CMB joint posterior, built from
    the published MAP, marginal sigmas, and the w0-wa correlation.
    """
    name: str

    omega_m: float
    w0: float
    wa: float

    sigma_omega_m: float
    sigma_w0: float
    sigma_wa: float

    corr_w0_wa: float = -0.8


# SN + CMB (no DESI BAO) joint MAP values from each compilation's own paper.
# These are the methodologically correct values for cross-predicting DESI BAO,
# because they represent each SN sample's independent preference for (w0, wa)
# without any DESI input.
PANTHEON_PLUS = SNConstraint(
    name="Pantheon+",
    omega_m=0.336,           # approx Pantheon+ + Planck w0wa-CDM
    w0=-0.851,               # Brout et al. 2022, Pantheon+ + Planck CMB w0waCDM
    wa=-0.70,                # Brout et al. 2022
    sigma_omega_m=0.018,
    sigma_w0=0.095,
    sigma_wa=0.50,
    corr_w0_wa=-0.7
)

DESY5 = SNConstraint(
    name="DES-Y5",
    omega_m=0.325,           # DES Collab 2024 Table 2, DES-SN5YR + Planck 2020
    w0=-0.73,                # DES Collab 2024 Table 2, Flat-w0waCDM
    wa=-1.17,                # DES Collab 2024 Table 2
    sigma_omega_m=0.014,
    sigma_w0=0.11,
    sigma_wa=0.59,
    corr_w0_wa=-0.85
)

UNION3 = SNConstraint(
    name="Union3",
    omega_m=0.323,           # Rubin et al. 2023, SNe + CMB Flat w0-wa
    w0=-0.699,               # Rubin et al. 2023
    wa=-1.05,                # Rubin et al. 2023
    sigma_omega_m=0.014,
    sigma_w0=0.17,
    sigma_wa=0.78,
    corr_w0_wa=-0.75
)
