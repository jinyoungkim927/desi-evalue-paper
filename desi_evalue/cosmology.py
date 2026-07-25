"""Flat w0waCDM distances (Hogg 1999) with CPL w(a) = w0 + wa(1-a), and the
Gaussian BAO likelihood. The background is always an explicit argument."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.linalg import cho_factor, cho_solve

from .constants import BASELINE_BACKGROUND, C_LIGHT_KM_S, LCDM_THETA, OMEGA_R


@dataclass(frozen=True)
class Background:
    """Fixed background cosmology. Omega_de follows from flatness."""

    h: float = BASELINE_BACKGROUND[0]
    omega_m: float = BASELINE_BACKGROUND[1]
    rd: float = BASELINE_BACKGROUND[2]

    @property
    def omega_de(self) -> float:
        return 1.0 - self.omega_m - OMEGA_R

    @property
    def hubble_distance(self) -> float:
        return C_LIGHT_KM_S / (100.0 * self.h)


BASELINE = Background()


def inverse_E(z, theta, bg: Background):
    """1 / E(z) for the CPL dark-energy density. Vectorised over z."""
    w0, wa = theta
    a = 1.0 / (1.0 + z)
    de = bg.omega_de * a ** (-3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * (1.0 - a))
    return 1.0 / np.sqrt(bg.omega_m * (1.0 + z) ** 3 + OMEGA_R * (1.0 + z) ** 4 + de)


def _comoving_distance(z_unique, theta, bg: Background):
    """Comoving distance at each z. One quadrature per distinct redshift."""
    edges = np.concatenate([[0.0], z_unique])
    segments = [quad(inverse_E, lo, hi, args=(theta, bg))[0]
                for lo, hi in zip(edges[:-1], edges[1:])]
    return np.cumsum(segments) * bg.hubble_distance


def theory_vector(z, quantities, theta, bg: Background = BASELINE):
    """Predicted BAO observables in units of r_d, in the data's ordering."""
    z = np.asarray(z, dtype=float)
    z_unique, inverse = np.unique(z, return_inverse=True)
    dc = _comoving_distance(z_unique, theta, bg)[inverse]
    dh = bg.hubble_distance * inverse_E(z, theta, bg)

    out = np.empty(len(z))
    kind = np.array([q[:2] for q in quantities])
    out[kind == "DM"] = dc[kind == "DM"]
    out[kind == "DH"] = dh[kind == "DH"]
    dv = np.cbrt(z * dh * dc ** 2)
    out[kind == "DV"] = dv[kind == "DV"]
    return out / bg.rd


def null_vector(z, quantities, bg: Background = BASELINE):
    """Theory vector under LCDM."""
    return theory_vector(z, quantities, LCDM_THETA, bg)


def offsets(z, quantities, thetas, bg: Background = BASELINE):
    """Theory offsets from the null, delta[g] = mu(theta_g) - mu(H0)."""
    mu0 = null_vector(z, quantities, bg)
    return np.array([theory_vector(z, quantities, t, bg) - mu0 for t in thetas])


class GaussianLikelihood:
    """Gaussian likelihood with a fixed covariance, Cholesky-factorised once."""

    def __init__(self, cov):
        self.cov = np.atleast_2d(cov)
        self._chol = cho_factor(self.cov)

    def solve(self, x):
        """C^-1 x, for x of shape (n,) or (n, b)."""
        return cho_solve(self._chol, x)

    def chi2(self, residual):
        residual = np.asarray(residual)
        return float(residual @ self.solve(residual))

    def whiten_draws(self, n_draws, rng):
        """Zero-mean residual draws with this covariance, shape (n, n_draws)."""
        lower = np.linalg.cholesky(self.cov)
        return lower @ rng.standard_normal((self.cov.shape[0], n_draws))
