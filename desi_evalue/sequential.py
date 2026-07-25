"""Joint (DR1, DR2) martingale under the year-scaling model."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from . import constants as C
from .cosmology import BASELINE, Background, null_vector, offsets
from .data import match_across_releases


def years_23_covariance(cov_dr1, cov_dr2, alpha=C.ALPHA_Y1):
    """cov(eps_y23) implied by DR2 = mu + alpha*eps_y1 + (1-alpha)*eps_y23.

    Must be PSD for the model to be usable; the min eigenvalue is returned.
    """
    cov = (cov_dr2 - alpha**2 * cov_dr1) / (1.0 - alpha) ** 2
    return cov, float(np.linalg.eigvalsh(cov).min())


def largest_admissible_alpha(cov_dr1, cov_dr2, lo=C.ALPHA_Y1, hi=0.75, iters=60):
    """Largest year-1 share keeping cov(eps_y23) PSD."""
    if years_23_covariance(cov_dr1, cov_dr2, lo)[1] < 0:
        return float("nan")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if years_23_covariance(cov_dr1, cov_dr2, mid)[1] >= 0 else (lo, mid)
    return float(lo)


class JointEProcess:
    """Exact martingale on the DR1 -> DR2 filtration.

    Scores DR1 once, then only what DR2 adds: the years-2-3 innovation on the
    matched bins, plus any new bins. Ville's inequality applies to this, not to
    the snapshot sequence, which is not a martingale.
    """

    def __init__(self, dr1, dr2, thetas, alpha=C.ALPHA_Y1, bg: Background = BASELINE):
        self.alpha = alpha
        self.dr1_idx, self.dr2_idx, self.new_idx = match_across_releases(dr1, dr2)

        d1 = offsets(dr1.z, dr1.quantities, thetas, bg)
        d2 = offsets(dr2.z, dr2.quantities, thetas, bg)
        cov_y23, self.min_eigenvalue = years_23_covariance(
            dr1.cov[np.ix_(self.dr1_idx, self.dr1_idx)],
            dr2.cov[np.ix_(self.dr2_idx, self.dr2_idx)], alpha)
        self.cov_dr1 = dr1.cov
        # The innovation is tracked as the unscaled combination DR2_m - alpha*DR1_m,
        # which equals (1 - alpha) * eps_y23 and so carries this covariance.
        self.cov_innovation = (1.0 - alpha) ** 2 * cov_y23
        self.cov_new = dr2.cov[np.ix_(self.new_idx, self.new_idx)]

        # DR1 factor.
        self._a1, self._c1 = _affine(d1, dr1.cov)
        # Matched DR2 block, conditional on DR1: the innovation carries the
        # theory offset net of the alpha-weighted DR1 offset it already saw.
        d_new = d2[:, self.dr2_idx] - alpha * d1[:, self.dr1_idx]
        self._ay, self._cy = _affine(d_new, self.cov_innovation)
        # Bins present only in DR2.
        if len(self.new_idx):
            self._au, self._cu = _affine(d2[:, self.new_idx],
                                         dr2.cov[np.ix_(self.new_idx, self.new_idx)])
        else:
            self._au = np.zeros((len(thetas), 0))
            self._cu = np.zeros(len(thetas))

        mu1 = null_vector(dr1.z, dr1.quantities, bg)
        mu2 = null_vector(dr2.z, dr2.quantities, bg)
        self.eps1 = dr1.values - mu1
        eps2 = dr2.values - mu2
        # Years-2-3 content of DR2 on the matched bins.
        self.innovation = eps2[self.dr2_idx] - alpha * self.eps1[self.dr1_idx]
        self.new_residual = eps2[self.new_idx]
        self.n_thetas = len(thetas)

    def log_e(self, eps1=None, innovation=None, new_residual=None):
        """log of the joint mixture; defaults to the observed data."""
        eps1 = self.eps1 if eps1 is None else eps1
        innovation = self.innovation if innovation is None else innovation
        new_residual = self.new_residual if new_residual is None else new_residual
        flat = np.ndim(eps1) == 1
        e1, y = np.atleast_2d(eps1.T).T, np.atleast_2d(innovation.T).T
        log_lr = self._a1 @ e1 + self._ay @ y - 0.5 * (self._c1 + self._cy + self._cu)[:, None]
        if self._au.shape[1]:
            log_lr = log_lr + self._au @ np.atleast_2d(new_residual.T).T
        out = logsumexp(log_lr, axis=0) - np.log(self.n_thetas)
        return float(out[0]) if flat else out

    @property
    def e(self):
        return float(np.exp(self.log_e()))


def _affine(delta, cov):
    """Score matrix delta @ C^-1 and quadratic term diag(delta C^-1 delta')."""
    scores = np.linalg.solve(cov, delta.T).T
    return scores, np.einsum("gi,gi->g", delta, scores)


def ville_bound(alpha=C.ALPHA):
    """P(sup_t M_t >= 1/alpha | H0) <= alpha."""
    return alpha


def _draw(cov, n_draws, rng):
    if cov.shape[0] == 0:
        return np.zeros((0, n_draws))
    return np.linalg.cholesky(cov) @ rng.standard_normal((cov.shape[0], n_draws))


def running_supremum_tail(dr1_mixture, joint, n_draws=C.N_MC_MARTINGALE, seed=C.SEED):
    """Monte Carlo P(sup_t M_t >= 1/alpha | H0) for the joint process."""
    rng = np.random.default_rng(seed)
    eps1 = _draw(joint.cov_dr1, n_draws, rng)
    log_m1 = dr1_mixture.log_e_from_residuals(eps1)
    log_m2 = joint.log_e(eps1,
                         _draw(joint.cov_innovation, n_draws, rng),
                         _draw(joint.cov_new, n_draws, rng))
    return float((np.maximum(log_m1, log_m2) >= np.log(C.THRESHOLD)).mean())
