"""Assumptions, fixed parameters, and alternative choices. Nothing data-derived."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Decision threshold
# ---------------------------------------------------------------------------
# Illustrative level used throughout the paper. An e-value of 1/ALPHA or more
# rejects; Ville's inequality bounds the false-positive rate by ALPHA at any
# stopping time and any number of looks.
ALPHA = 0.05
THRESHOLD = 1.0 / ALPHA  # 20

# ---------------------------------------------------------------------------
# Background cosmology
# ---------------------------------------------------------------------------
# (h, Omega_m, r_d/Mpc). BASELINE is the pre-specified choice quoted in the
# body; it mixes Planck 2018 columns, taking (h, Omega_m) from +lensing+BAO and
# r_d from the CMB-only chains. The alternatives are self-consistent single
# columns, and the headline e-value moves by two orders of magnitude across
# them -- this is the background-sensitivity result of Appendix B.3.
BASELINE_BACKGROUND = (0.6766, 0.3111, 147.05)

PLANCK_COLUMNS = {
    "baseline": (0.6766, 0.3111, 147.05),          # (a) as used in the body
    "TT,TE,EE+lowE+lensing": (0.6736, 0.3153, 147.09),  # (b)
    "+lensing+BAO": (0.6766, 0.3111, 147.21),      # (c) self-consistent w/ (h, Om)
    "TT,TE,EE+lowE": (0.6727, 0.3166, 147.05),     # (d)
}

OMEGA_R = 9.0e-5      # radiation density, held fixed
C_LIGHT_KM_S = 299792.458

# Null and the BAO-only MLE at the baseline background (paper Section 3).
LCDM_THETA = (-1.0, 0.0)
DR2_BAO_MLE = (-0.856, -0.430)

# Number of parameters the alternative adds over LCDM, for Occam/Wilks terms.
N_EXTRA_PARAMS = 2

# ---------------------------------------------------------------------------
# Test specifications on (w0, wa)
# ---------------------------------------------------------------------------
# Flat "box" priors, as (w0_min, w0_max, wa_min, wa_max). Default is the single
# pre-specified test; the others probe how much prior mass sits away from the
# data-preferred direction.
BOX_PRIORS = {
    "Narrow":  (-1.2, -0.8, -1.0, 0.5),
    "Default": (-1.5, -0.5, -2.0, 1.0),
    "Wide":    (-2.0,  0.0, -3.0, 2.0),
    "Ong":     (-3.0,  1.0, -3.0, 2.0),
}
DEFAULT_PRIOR = "Default"

# Grid resolution for box priors. The paper uses 30x30.
#
# The grid is part of the test specification, not a numerical detail: a uniform
# mixture over a finite, pre-specified set of points is an exact e-value, and a
# different resolution is a different (equally valid) test. Values do drift with
# n -- Default runs 33.97 at n=30 to 35.75 at n=120, and the much smaller
# FREEZING_BOX runs 21.5 at n=20 down to 15.5 at n=120, crossing the threshold.
# Change this only together with the numbers quoted in the paper.
GRID_N = 30
# The freezing box is small and its integrand peaks hard in the corner nearest
# the data, so a 30x30 grid is not converged: it reads 19.0 there against 14.8
# at 300x300. Integrate it finely enough that the quoted value is the integral
# rather than the discretisation.
FREEZING_GRID_N = 200

# REGROW: mass placed on the Fisher-distance-delta ellipse around LCDM.
REGROW_DELTAS = (1.0, 2.0, 3.0)
REGROW_N_POINTS = 120       # points sampled uniformly in eigenangle
FISHER_STEP = 5e-3          # finite-difference step for the Fisher matrix
# The ellipse is built from the DR2 Fisher matrix and that same set of points is
# used for both releases: a pre-specified prior is fixed once, not re-derived
# from whichever release is being scored.
REGROW_FISHER_FROM = "DR2"

# Gaussian-shell proxy pi_delta = N(theta_0, delta^2 F^-1) of Appendix B.6.
SHELL_N_GRID = 80
SHELL_N_SIGMA = 5

# Physically motivated priors (Section 4.3).
# Caldwell-Linder thawing band: wa in [-3(1+w0), -(1+w0)] for w0 in this range.
THAWING_W0_RANGE = (-1.0, -0.85)
# Freezing (tracker, SUGRA) region, as a box.
FREEZING_BOX = (-0.95, -0.75, 0.0, 0.3)

# Optimiser bounds and start for every (w0, wa) fit in the package.
FIT_BOUNDS = ((-2.0, 0.0), (-3.0, 2.0))
FIT_X0 = (-0.9, -0.5)

# ---------------------------------------------------------------------------
# Nested-release (year-scaling) model, Appendix B.2
# ---------------------------------------------------------------------------
# DR1 = mu + eps_y1;  DR2 = mu + ALPHA_Y1 * eps_y1 + (1 - ALPHA_Y1) * eps_y23.
# ALPHA_Y1 is fixed by survey design (DR2 is a three-year cumulative average,
# year 1 being one of the three), not fitted.
ALPHA_Y1 = 1.0 / 3.0
# Range re-run as a robustness check, and the largest share for which the
# decomposition stays positive semi-definite.
ALPHA_Y1_ROBUSTNESS = (0.25, 0.40)
ALPHA_Y1_MAX_ADMISSIBLE = 0.470
# Two bins count as the same physical measurement across releases when their
# effective redshifts agree to this fractional tolerance.
BIN_MATCH_RTOL = 0.01

# ---------------------------------------------------------------------------
# Monte Carlo settings
# ---------------------------------------------------------------------------
# Every stochastic result in the paper is seeded. Sizes are the ones quoted.
SEED = 20260710
N_MC_MARTINGALE = 100_000     # joint-null running-supremum tail (App B.2)
N_MC_SIGMA = 200_000          # sigma_emp calibration (App B.1)
N_MC_DR3 = 10_000             # DR3 forecast (Section 4.4)
N_BOOTSTRAP_ALT = 500         # LOO concentration under w0wa truth (App B.5)
N_BOOTSTRAP_NULL = 200        # LOO concentration under LCDM truth (App B.5)
N_UI_SPLITS = 400             # universal-inference cross-fits (App B.2)
N_UI_SYNTHETIC = 200          # synthetic-H0 trials for universal inference

# Concentration level the observed LOO decomposition is compared against.
LOO_CONCENTRATION_LEVEL = 0.75

# ---------------------------------------------------------------------------
# DR3 forecast (Section 4.4)
# ---------------------------------------------------------------------------
# Hypothetical new high-redshift bins, as (z_eff, sigma_DM/rd, sigma_DH/rd).
# These barely discriminate: LCDM and the DR2-preferred w0wa differ by ~0.5
# sigma there.
DR3_NEW_BINS = ((1.7, 0.7, 0.4), (2.5, 1.5, 0.4))
# Factor by which the existing-bin errors are shrunk in the decisive scenario.
DR3_ERROR_SHRINK = 0.5
DR3_DECISIVE_BIN = "LRG2"

# ---------------------------------------------------------------------------
# Compressed Planck CMB likelihood (Appendix B.8)
# ---------------------------------------------------------------------------
# Chen, Huang & Wang (2019) compressed statistic. Note r_s(z_*) differs from the
# BAO drag-epoch r_d above; both are needed and they are not interchangeable.
CMB_RS_STAR = 144.65   # Mpc
CMB_Z_STAR = 1089.92

# ---------------------------------------------------------------------------
# SN+CMB compilations used as pre-specified point alternatives (App B.9)
# ---------------------------------------------------------------------------
# (w0, wa) MAP and its covariance, from each published joint posterior.
SN_COMPILATIONS = {
    "Pantheon+": {
        "map": (-0.851, -0.70),
        "cov": ((0.058**2, -0.130 * 0.058 * 0.28), (-0.130 * 0.058 * 0.28, 0.28**2)),
    },
    "DES-Y5": {
        "map": (-0.730, -1.17),
        "cov": ((0.062**2, -0.150 * 0.062 * 0.32), (-0.150 * 0.062 * 0.32, 0.32**2)),
    },
    "Union3": {
        "map": (-0.699, -1.05),
        "cov": ((0.081**2, -0.160 * 0.081 * 0.37), (-0.160 * 0.081 * 0.37, 0.37**2)),
    },
}
SN_POSTERIOR_DRAWS = 500

# ---------------------------------------------------------------------------
# Redshift-bin structure
# ---------------------------------------------------------------------------
# Upper z edges defining the seven DESI bins, used to group the 13 DR2
# measurements. Bins are the unit of every leave-one-out and per-bin product,
# because the published covariance is block-diagonal by bin while the two
# measurements within a bin are anti-correlated.
BIN_EDGES_DR2 = ((0.35, "BGS"), (0.55, "LRG1"), (0.75, "LRG2"), (1.0, "LRG3+ELG1"),
                 (1.4, "ELG2"), (2.0, "QSO"), (np.inf, "Lya"))
BIN_EDGES_DR1 = ((0.35, "BGS"), (0.55, "LRG1"), (0.75, "LRG2"), (1.0, "LRG+ELG"),
                 (1.35, "ELG"), (2.0, "QSO"), (np.inf, "Lya"))

# Redshift at which the data-split test partitions the sample. Reported but not
# relied on: it is underpowered for wa (Appendix B.7).
SPLIT_Z = 1.0


def box_grid(name_or_box, n=GRID_N):
    """Flat prior grid as an (n*n, 2) array of (w0, wa) points."""
    box = BOX_PRIORS[name_or_box] if isinstance(name_or_box, str) else name_or_box
    w0_min, w0_max, wa_min, wa_max = box
    w0, wa = np.linspace(w0_min, w0_max, n), np.linspace(wa_min, wa_max, n)
    return np.stack(np.meshgrid(w0, wa, indexing="ij"), axis=-1).reshape(-1, 2)
