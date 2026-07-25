"""E-value constructions, all satisfying E[E | H0] <= 1.

The mixture averages the likelihood ratio over a pre-specified set of (w0, wa).
Averaging is what keeps it valid: the maximised ratio exp(dchi2/2) has infinite
null expectation for two extra parameters and is not an e-value. Sequential
statements need the martingale in sequential.py."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm

from . import constants as C
from .cosmology import BASELINE, Background, GaussianLikelihood, null_vector, offsets, theory_vector


class Mixture:
    """Mixture e-value over a pre-specified set of (w0, wa) points.

    log LR_g = delta_g' C^-1 eps - (1/2) delta_g' C^-1 delta_g is affine in the
    residual eps = data - mu(H0), so the score matrix and quadratic term are
    built once and each evaluation is a matrix product.
    """

    def __init__(self, dataset, thetas, bg: Background = BASELINE):
        self.dataset, self.thetas = dataset, np.asarray(thetas, dtype=float)
        self.likelihood = GaussianLikelihood(dataset.cov)
        self.mu_null = null_vector(dataset.z, dataset.quantities, bg)
        self.delta = offsets(dataset.z, dataset.quantities, self.thetas, bg)
        self.scores = self.likelihood.solve(self.delta.T).T
        self.penalty = np.einsum("gi,gi->g", self.delta, self.scores)
        self.residual = dataset.values - self.mu_null

    def log_e_from_residuals(self, eps):
        """log E for residuals of shape (n,) or (n, n_draws)."""
        eps = np.asarray(eps)
        flat = eps.ndim == 1
        log_lr = self.scores @ np.atleast_2d(eps.T).T - 0.5 * self.penalty[:, None]
        out = logsumexp(log_lr, axis=0) - np.log(len(self.thetas))
        return float(out[0]) if flat else out

    @property
    def log_e(self):
        return self.log_e_from_residuals(self.residual)

    @property
    def e(self):
        return float(np.exp(self.log_e))

    def null_draws(self, n_draws, rng):
        """log E on ``n_draws`` residual draws under H0."""
        return self.log_e_from_residuals(self.likelihood.whiten_draws(n_draws, rng))


def mixture_e(dataset, thetas, bg: Background = BASELINE):
    """Mixture e-value for a dataset and a pre-specified set of alternatives."""
    return Mixture(dataset, thetas, bg).e


def markov_p(e):
    """Markov p-value 1/E, valid at a fixed look under any dependence."""
    return float(min(1.0, 1.0 / e))


def rejects(e):
    return e >= C.THRESHOLD


# ---------------------------------------------------------------------------
# Point estimates and local geometry
# ---------------------------------------------------------------------------

def fit_mle(dataset, bg: Background = BASELINE):
    """Maximum-likelihood (w0, wa) at a fixed background."""
    likelihood = GaussianLikelihood(dataset.cov)

    def objective(theta):
        resid = dataset.values - theory_vector(dataset.z, dataset.quantities, theta, bg)
        return likelihood.chi2(resid)

    fit = minimize(objective, C.FIT_X0, bounds=C.FIT_BOUNDS, method="L-BFGS-B")
    return tuple(fit.x), float(fit.fun)


def delta_chi2(dataset, bg: Background = BASELINE):
    """chi2(LCDM) - chi2(MLE), the Wilks statistic on two extra parameters."""
    likelihood = GaussianLikelihood(dataset.cov)
    chi2_null = likelihood.chi2(dataset.values - null_vector(dataset.z, dataset.quantities, bg))
    _, chi2_alt = fit_mle(dataset, bg)
    return chi2_null - chi2_alt


def wilks_sigma(dchi2, k=C.N_EXTRA_PARAMS):
    """Two-sided sigma for a chi2(k) statistic. For the maximised ratio only:
    e-values are never converted to sigma (Appendix B.1)."""
    from scipy.stats import chi2 as chi2_dist
    return float(norm.isf(0.5 * chi2_dist.sf(dchi2, k)))


def fisher_at_null(dataset, bg: Background = BASELINE, step=C.FISHER_STEP):
    """Fisher information at LCDM, from the Hessian of the expected chi^2.

    That Hessian is 2F at the null, so F is half of it.
    """
    likelihood = GaussianLikelihood(dataset.cov)

    def expected_chi2(theta):
        d = theory_vector(dataset.z, dataset.quantities, theta, bg) - \
            null_vector(dataset.z, dataset.quantities, bg)
        return likelihood.chi2(d)

    w0, wa = C.LCDM_THETA
    h = step
    f00 = expected_chi2((w0 + h, wa)) - 2 * expected_chi2((w0, wa)) + expected_chi2((w0 - h, wa))
    f11 = expected_chi2((w0, wa + h)) - 2 * expected_chi2((w0, wa)) + expected_chi2((w0, wa - h))
    f01 = (expected_chi2((w0 + h, wa + h)) - expected_chi2((w0 + h, wa - h))
           - expected_chi2((w0 - h, wa + h)) + expected_chi2((w0 - h, wa - h))) / 4.0
    return 0.5 * np.array([[f00 / h**2, f01 / h**2], [f01 / h**2, f11 / h**2]])


def fisher_distance(theta, fisher, theta0=C.LCDM_THETA):
    """Distance from the null in Fisher-sigma units."""
    d = np.asarray(theta, dtype=float) - np.asarray(theta0, dtype=float)
    return float(np.sqrt(d @ fisher @ d))


# ---------------------------------------------------------------------------
# Test specifications: each returns a set of (w0, wa) points
# ---------------------------------------------------------------------------

def regrow_points(fisher, delta, n_points=C.REGROW_N_POINTS, theta0=C.LCDM_THETA):
    """Uniform sample on the Fisher-distance-delta ellipse around the null.

    All mass sits at distance delta: a pre-committed minimum effect size, paying
    the full Occam penalty for it.
    """
    eigvals, eigvecs = np.linalg.eigh(fisher)
    if eigvals.min() <= 0:
        raise ValueError(f"Fisher matrix is not positive definite: {eigvals}")
    phi = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    half_axes = delta / np.sqrt(eigvals)
    unit = np.stack([half_axes[0] * np.cos(phi), half_axes[1] * np.sin(phi)], axis=1)
    return unit @ eigvecs.T + np.asarray(theta0, dtype=float)


def shell_points(fisher, delta, n_grid=C.SHELL_N_GRID, n_sigma=C.SHELL_N_SIGMA,
                 theta0=C.LCDM_THETA):
    """Gaussian shell N(theta0, delta^2 F^-1) as a weighted grid."""
    cov = delta**2 * np.linalg.inv(fisher)
    scale = n_sigma * np.sqrt(np.diag(cov))
    axes = [np.linspace(c - s, c + s, n_grid) for c, s in zip(theta0, scale)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    d = pts - np.asarray(theta0, dtype=float)
    weights = np.exp(-0.5 * np.einsum("gi,ij,gj->g", d, np.linalg.inv(cov), d))
    keep = weights > weights.max() * 1e-6
    return pts[keep], weights[keep] / weights[keep].sum()


def thawing_points(n=C.GRID_N, w0_range=C.THAWING_W0_RANGE):
    """Caldwell-Linder thawing band, wa in [-3(1+w0), -(1+w0)]."""
    w0 = np.linspace(*w0_range, n)
    frac = np.linspace(0.0, 1.0, n)
    lo, hi = -3.0 * (1.0 + w0), -(1.0 + w0)
    grid_w0 = np.repeat(w0, n)
    grid_wa = (lo[:, None] + frac[None, :] * (hi - lo)[:, None]).ravel()
    return np.stack([grid_w0, grid_wa], axis=1)


def freezing_points(n=C.FREEZING_GRID_N):
    """Freezing (tracker, SUGRA) region."""
    return C.box_grid(C.FREEZING_BOX, n)


# ---------------------------------------------------------------------------
# Decompositions
# ---------------------------------------------------------------------------

def leave_one_bin_out(dataset, bg: Background = BASELINE):
    """Per-bin LOO e-values and their average.

    Each fold fits (w0, wa) on the other bins and scores the held-out one. The
    average is valid by linearity; the product is not, since folds share
    training data.
    """
    per_bin = {}
    for name in dataset.bin_names:
        held = dataset.subset(dataset.indices_for(name))
        rest = dataset.drop_bins(name)
        theta, _ = fit_mle(rest, bg)
        likelihood = GaussianLikelihood(held.cov)
        mu0 = null_vector(held.z, held.quantities, bg)
        mu1 = theory_vector(held.z, held.quantities, theta, bg)
        log_e = 0.5 * (likelihood.chi2(held.values - mu0) - likelihood.chi2(held.values - mu1))
        per_bin[name] = {"e": float(np.exp(log_e)), "theta": theta,
                         "z": float(np.median(held.z))}
    values = np.array([b["e"] for b in per_bin.values()])
    return {"per_bin": per_bin, "average": float(values.mean()),
            "shares": {k: v["e"] / values.sum() for k, v in per_bin.items()}}


def per_bin_mixture(dataset, thetas, bg: Background = BASELINE):
    """Per-bin mixture e-values, their product and their mean.

    The product is valid for the alternative where each bin takes its own
    (w0, wa), since the covariance is block-diagonal by bin and the prior is
    fixed in advance. The mean corrects for picking the most favourable bin.
    """
    per_bin = {name: mixture_e(dataset.subset(dataset.indices_for(name)), thetas, bg)
               for name in dataset.bin_names}
    values = np.array(list(per_bin.values()))
    return {"per_bin": per_bin, "product": float(values.prod()),
            "mean": float(values.mean())}


def split_evalue(dataset, train_idx, test_idx, bg: Background = BASELINE):
    """Fit the alternative on one part of the data, score it on the rest."""
    train, test = dataset.subset(train_idx), dataset.subset(test_idx)
    theta, _ = fit_mle(train, bg)
    likelihood = GaussianLikelihood(test.cov)
    mu0 = null_vector(test.z, test.quantities, bg)
    mu1 = theory_vector(test.z, test.quantities, theta, bg)
    log_e = 0.5 * (likelihood.chi2(test.values - mu0) - likelihood.chi2(test.values - mu1))
    return {"e": float(np.exp(log_e)), "theta": theta}


def universal_inference(dataset, n_splits=C.N_UI_SPLITS, seed=C.SEED,
                        bg: Background = BASELINE):
    """Prior-free split-sample e-value, averaged over random bin partitions.

    Splits at bin level, not measurement level: the two measurements inside a
    bin are anti-correlated, so a within-bin split would break the independence
    the construction needs. An average of e-values is an e-value.
    """
    rng = np.random.default_rng(seed)
    names = np.array(dataset.bin_names)
    half = len(names) // 2
    draws = []
    for _ in range(n_splits):
        shuffled = rng.permutation(names)
        train = dataset.indices_for(*shuffled[:half])
        test = dataset.indices_for(*shuffled[half:])
        draws.append(split_evalue(dataset, train, test, bg)["e"])
    return float(np.mean(draws))
