"""The paper's results, one function each. All return values; none print."""

from __future__ import annotations

import numpy as np

from . import constants as C
from . import data as _data
from .cosmology import BASELINE, Background
from .evalues import (Mixture, delta_chi2, fisher_at_null, fisher_distance, fit_mle,
                      freezing_points, leave_one_bin_out, markov_p, mixture_e,
                      per_bin_mixture, regrow_points, rejects, shell_points,
                      split_evalue, thawing_points, universal_inference, wilks_sigma)
from .sequential import JointEProcess, largest_admissible_alpha, years_23_covariance


def load_releases(data_dir=_data.DATA_DIR):
    return _data.load("DR1", data_dir), _data.load("DR2", data_dir)


def running_mixture(dr1, dr2, prior=C.DEFAULT_PRIOR, bg: Background = BASELINE):
    """M_DR1 and M_DR2 for one box prior (Table 1)."""
    thetas = C.box_grid(prior)
    m1, m2 = mixture_e(dr1, thetas, bg), mixture_e(dr2, thetas, bg)
    return {"prior": prior, "M_DR1": m1, "M_DR2": m2,
            "markov_p": markov_p(m2), "rejects": rejects(m2)}


def prior_sensitivity(dr1, dr2, bg: Background = BASELINE):
    """Table 1: all box priors and the REGROW ellipses."""
    rows = [running_mixture(dr1, dr2, name, bg) for name in C.BOX_PRIORS]
    fisher = fisher_at_null(dr2, bg)  # prior fixed once, from DR2; see constants
    for delta in C.REGROW_DELTAS:
        points = regrow_points(fisher, delta)
        m1 = mixture_e(dr1, points, bg)
        m2 = mixture_e(dr2, points, bg)
        rows.append({"prior": f"REGROW delta={delta:g}", "M_DR1": m1, "M_DR2": m2,
                     "markov_p": markov_p(m2), "rejects": rejects(m2)})
    return rows


def physical_priors(dr1, dr2, bg: Background = BASELINE):
    """Thawing and freezing quintessence specifications (Section 4.3)."""
    out = {}
    for name, thetas in (("thawing", thawing_points()), ("freezing", freezing_points())):
        out[name] = {"M_DR1": mixture_e(dr1, thetas, bg),
                     "M_DR2": mixture_e(dr2, thetas, bg)}
    return out


def likelihood_geometry(dr2, bg: Background = BASELINE):
    """MLE, Wilks statistic, and how far the MLE sits from the null."""
    theta, _ = fit_mle(dr2, bg)
    dchi2 = delta_chi2(dr2, bg)
    fisher = fisher_at_null(dr2, bg)
    return {"mle": theta, "delta_chi2": dchi2, "wilks_sigma": wilks_sigma(dchi2),
            "delta_mle": fisher_distance(theta, fisher)}


def localisation(dr2, bg: Background = BASELINE):
    """Table 2: LOO and per-bin-independent decompositions."""
    loo = leave_one_bin_out(dr2, bg)
    regrow = regrow_points(fisher_at_null(dr2, bg), 2.0)
    per_bin = per_bin_mixture(dr2, regrow, bg)
    without = mixture_e(dr2.drop_bins(C.DR3_DECISIVE_BIN), C.box_grid(C.DEFAULT_PRIOR), bg)
    six_bin = {k: v for k, v in loo["per_bin"].items() if k != C.DR3_DECISIVE_BIN}
    return {"loo": loo, "per_bin_regrow": per_bin,
            "M_DR2_without_LRG2": without,
            "loo_average_without_LRG2": float(np.mean([b["e"] for b in six_bin.values()])),
            "lrg2_share": loo["shares"][C.DR3_DECISIVE_BIN]}


def loo_concentration_bootstrap(dr2, bg: Background = BASELINE,
                                n_alt=C.N_BOOTSTRAP_ALT, n_null=C.N_BOOTSTRAP_NULL,
                                seed=C.SEED):
    """How often a smooth signal concentrates as sharply as the data (App B.5).

    What separates a localised feature from a smooth signal is which bin
    dominates, not how concentrated the evidence is.
    """
    from .cosmology import GaussianLikelihood, theory_vector
    rng = np.random.default_rng(seed)
    likelihood = GaussianLikelihood(dr2.cov)
    observed = leave_one_bin_out(dr2, bg)["shares"][C.DR3_DECISIVE_BIN]

    out = {}
    for label, theta, n in (("w0wa", C.DR2_BAO_MLE, n_alt), ("lcdm", C.LCDM_THETA, n_null)):
        truth = theory_vector(dr2.z, dr2.quantities, theta, bg)
        draws = likelihood.whiten_draws(n, rng)
        top_share, top_bin = [], []
        for k in range(n):
            synthetic = _data.BAOData(dr2.release, dr2.z, truth + draws[:, k], dr2.cov,
                                      dr2.quantities, dr2.bins)
            shares = leave_one_bin_out(synthetic, bg)["shares"]
            name = max(shares, key=shares.get)
            top_share.append(shares[name])
            top_bin.append(name)
        top_share = np.array(top_share)
        out[label] = {
            "median_max_share": float(np.median(top_share)),
            "frac_above_level": float((top_share >= C.LOO_CONCENTRATION_LEVEL).mean()),
            "frac_above_observed": float((top_share >= observed).mean()),
            "frac_lrg2_above_observed": float(np.mean(
                [(s >= observed) and (b == C.DR3_DECISIVE_BIN)
                 for s, b in zip(top_share, top_bin)])),
            "most_frequent_bin": max(set(top_bin), key=top_bin.count),
        }
    out["observed_lrg2_share"] = observed
    return out


def joint_sequential(dr1, dr2, prior=C.DEFAULT_PRIOR, alpha=C.ALPHA_Y1,
                     bg: Background = BASELINE):
    """Joint martingale across the two releases (Appendix B.2)."""
    thetas = C.box_grid(prior)
    joint = JointEProcess(dr1, dr2, thetas, alpha, bg)
    snapshot = mixture_e(dr2, thetas, bg)
    m1 = mixture_e(dr1, thetas, bg)
    return {"M_joint": joint.e, "M_snapshot": snapshot, "M_DR1": m1,
            "incremental": joint.e / m1,
            "anytime_valid_p": markov_p(joint.e),
            "cov_y23_min_eigenvalue": joint.min_eigenvalue,
            "n_matched": len(joint.dr1_idx), "n_new": len(joint.new_idx)}


def year_weight_robustness(dr1, dr2, prior=C.DEFAULT_PRIOR, bg: Background = BASELINE):
    """Joint value across assumed year-1 shares, and the admissible limit."""
    thetas = C.box_grid(prior)
    lo, hi = C.ALPHA_Y1_ROBUSTNESS
    grid = np.linspace(lo, hi, 7)
    values = {float(a): JointEProcess(dr1, dr2, thetas, a, bg).e for a in grid}
    matched1, matched2, _ = _data.match_across_releases(dr1, dr2)
    return {"by_alpha": values,
            "max_admissible_alpha": largest_admissible_alpha(
                dr1.cov[np.ix_(matched1, matched1)], dr2.cov[np.ix_(matched2, matched2)])}


def background_sensitivity(dr1, dr2, prior=C.DEFAULT_PRIOR):
    """Headline values across the Planck columns (Appendix B.3)."""
    rows = {}
    for name, (h, om, rd) in C.PLANCK_COLUMNS.items():
        bg = Background(h, om, rd)
        thetas = C.box_grid(prior)
        rows[name] = {
            "M_DR1": mixture_e(dr1, thetas, bg),
            "M_DR2": mixture_e(dr2, thetas, bg),
            "M_DR2_without_LRG2": mixture_e(dr2.drop_bins(C.DR3_DECISIVE_BIN), thetas, bg),
            "mle": fit_mle(dr2, bg)[0],
        }
    values = [r["M_DR2"] for r in rows.values()]
    rows["spread"] = max(values) / min(values)
    return rows


def shell_mixture(delta, dchi2, k=C.N_EXTRA_PARAMS):
    """Closed-form mixture e-value for the Gaussian shell N(theta0, delta^2 F^-1)."""
    return float(np.exp(0.5 * dchi2 * delta**2 / (1 + delta**2)) / (1 + delta**2) ** (k / 2))


def fisher_delta_chi2(dr2, bg: Background = BASELINE):
    """Delta chi^2 in the local Gaussian approximation, i.e. delta_MLE^2.

    Smaller than the exact-likelihood value; the shell formulas assume this one.
    """
    theta, _ = fit_mle(dr2, bg)
    return fisher_distance(theta, fisher_at_null(dr2, bg)) ** 2


def minimum_concentration(dr2, bg: Background = BASELINE):
    """Shell widths reaching the threshold: the narrowest aligned prior that
    gets there, and the width where the Occam factor pulls it back below."""
    dchi2 = fisher_delta_chi2(dr2, bg)
    grid = np.geomspace(0.2, 40.0, 20000)
    curve = np.array([shell_mixture(d, dchi2) for d in grid])
    roots = grid[np.flatnonzero(np.diff((curve >= C.THRESHOLD).astype(int)))]
    return {"delta_chi2": dchi2, "roots": [float(r) for r in roots],
            "delta_max": float(np.sqrt(dchi2 / C.N_EXTRA_PARAMS - 1)),
            "M_max": float(curve.max())}


def shell_vs_regrow(dr2, delta=1.0, bg: Background = BASELINE):
    """The two Fisher-delta families disagree at the boundary (Appendix B.6):
    REGROW pays the full Occam penalty, the shell spreads mass inward."""
    fisher = fisher_at_null(dr2, bg)
    return {"shell": shell_mixture(delta, fisher_delta_chi2(dr2, bg)),
            "regrow": mixture_e(dr2, regrow_points(fisher, delta), bg)}


def dr3_forecast(dr2, bg: Background = BASELINE, n_mc=C.N_MC_DR3, seed=C.SEED):
    """What DR3 would have to deliver (Section 4.4). New high-z bins barely
    discriminate; shrinking the error on the bin carrying the evidence does."""
    from .cosmology import GaussianLikelihood, theory_vector
    rng = np.random.default_rng(seed)
    thetas = C.box_grid(C.DEFAULT_PRIOR)
    out = {}

    z_new = np.array([b[0] for b in C.DR3_NEW_BINS for _ in range(2)])
    q_new = tuple(q for _ in C.DR3_NEW_BINS for q in ("DM_over_rs", "DH_over_rs"))
    cov_new = np.diag([s**2 for b in C.DR3_NEW_BINS for s in b[1:]])
    new_bins = _data.BAOData("DR3", z_new, np.zeros(len(z_new)), cov_new, q_new,
                             tuple(f"new{z:g}" for z in z_new))
    for label, theta in (("w0wa", C.DR2_BAO_MLE), ("lcdm", C.LCDM_THETA)):
        truth = theory_vector(z_new, q_new, theta, bg)
        draws = GaussianLikelihood(cov_new).whiten_draws(n_mc, rng)
        mix = Mixture(new_bins, thetas, bg)
        logs = mix.log_e_from_residuals(truth[:, None] + draws - mix.mu_null[:, None])
        out[f"new_bins_{label}"] = float(np.median(np.exp(logs)))

    shrunk = _data.BAOData(dr2.release, dr2.z, dr2.values, dr2.cov.copy(),
                           dr2.quantities, dr2.bins)
    rows = dr2.indices_for(C.DR3_DECISIVE_BIN)
    scaled = shrunk.cov.copy()
    scaled[np.ix_(rows, rows)] *= C.DR3_ERROR_SHRINK**2
    for label, theta in (("w0wa", C.DR2_BAO_MLE), ("lcdm", C.LCDM_THETA)):
        truth = theory_vector(dr2.z, dr2.quantities, theta, bg)
        tightened = _data.BAOData(dr2.release, dr2.z, truth, scaled, dr2.quantities, dr2.bins)
        out[f"tightened_{C.DR3_DECISIVE_BIN}_{label}"] = mixture_e(tightened, thetas, bg)
    return out


def data_split(dr2, split_z=C.SPLIT_Z, bg: Background = BASELINE):
    """Redshift-split e-value. Underpowered for wa; reported, not relied on."""
    low = np.flatnonzero(dr2.z < split_z)
    high = np.flatnonzero(dr2.z >= split_z)
    return split_evalue(dr2, low, high, bg)


def prior_free(dr2, bg: Background = BASELINE):
    """Universal-inference e-value, which uses no prior at all."""
    return {"E_UI": universal_inference(dr2, bg=bg)}
