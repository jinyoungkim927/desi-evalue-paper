"""
Core e-value engine for DESI BAO model comparison (LCDM vs w0waCDM).

This module is imported by the analysis scripts in code/ and cross_release/,
which call these functions to produce the paper's headline numbers:
- uniform_mixture_evalue / grow_evalue on the 30x30 "Default" grid
  ([-1.5, -0.5] x [-2, 1]) -> M_DR2 = 33.97
- loocv_evalue -> LOO average e-value = 10.17

Key concepts:
- E-value: non-negative statistic with E[E] <= 1 under H0; combinable by
  multiplication (sequential) or averaging.
- Uniform mixture: Bayes factor with a discrete uniform prior over a grid of
  alternatives; valid e-value (E[E|H0] = 1 by construction).
- WARNING: the maximized likelihood ratio (plugging in the MLE) is NOT a valid
  e-value. For k>=2 extra parameters, E[exp(chi^2(k)/2)] = infinity under H0.
  Validity comes from data splitting (split_evalue) or mixtures (grow_evalue).

References:
- Shafer (2021), JRSS-A 184, 407-478
- Ramdas et al. (2023), Statistical Science 38(4)
- Vovk & Wang (2021), e-values: calibration, combination, applications
- Grünwald, de Heide & Koolen (2024), Safe Testing, JRSS-B
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Dict
from dataclasses import dataclass

from cosmology import (
    CosmologyParams, LCDM, compute_bao_predictions,
    chi_squared, log_likelihood
)


@dataclass
class EValueResult:
    """Results from e-value computation."""
    e_value: float
    log_e: float
    chi2_null: float
    chi2_alt: float
    delta_chi2: float
    null_params: CosmologyParams
    alt_params: CosmologyParams
    method: str

    @property
    def sigma_equivalent(self) -> float:
        """Convert to approximate sigma significance.

        Uses sigma = sqrt(2 * ln(E)), which equals sqrt(delta_chi2) when
        E = exp(delta_chi2 / 2) -- the standard cosmology convention for
        nested models. Approximate: e-values and p-values answer different
        questions.
        """
        if self.e_value <= 1:
            return 0.0
        return np.sqrt(2.0 * np.log(self.e_value))


def likelihood_ratio_evalue(
    data: np.ndarray,
    cov: np.ndarray,
    theory_null: np.ndarray,
    theory_alt: np.ndarray
) -> EValueResult:
    """
    Compute the likelihood-ratio statistic L(data|H1) / L(data|H0).

    This is a valid e-value ONLY when theory_alt is specified independently of
    the data (pre-registered alternative or a separate dataset). When theory_alt
    comes from MLE fitting on the same data it is a maximized likelihood ratio,
    NOT a valid e-value: for k>=2 extra parameters, E[exp(chi^2(k)/2) | H0] = inf.

    Used internally by split_evalue (valid: alternative fitted on disjoint
    training data) and for cross-dataset validation (also valid).
    """
    log_L_null = log_likelihood(data, theory_null, cov)
    log_L_alt = log_likelihood(data, theory_alt, cov)
    log_e = log_L_alt - log_L_null

    chi2_null = chi_squared(data, theory_null, cov)
    chi2_alt = chi_squared(data, theory_alt, cov)

    return EValueResult(
        e_value=np.exp(log_e),
        log_e=log_e,
        chi2_null=chi2_null,
        chi2_alt=chi2_alt,
        delta_chi2=chi2_null - chi2_alt,
        null_params=LCDM,
        alt_params=None,
        method='likelihood_ratio'
    )


def split_evalue(
    data: np.ndarray,
    cov: np.ndarray,
    z_values: np.ndarray,
    quantities: list,
    split_z: float = 1.0
) -> Tuple[EValueResult, EValueResult]:
    """
    Compute e-values using a redshift data split.

    Split into low-z (z < split_z) and high-z (z >= split_z); train the
    alternative on low-z and test on high-z. The test-set e-value is valid
    because the alternative is not fitted to the data used for testing.

    Returns (E_train, E_test).
    """
    low_z_mask = z_values < split_z
    high_z_mask = ~low_z_mask
    n_low = np.sum(low_z_mask)
    n_high = np.sum(high_z_mask)
    if n_low < 2 or n_high < 2:
        raise ValueError(f"Need at least 2 points in each split. Got {n_low} low-z, {n_high} high-z")

    low_z_idx = np.where(low_z_mask)[0]
    high_z_idx = np.where(high_z_mask)[0]
    data_low, data_high = data[low_z_idx], data[high_z_idx]
    cov_low = cov[np.ix_(low_z_idx, low_z_idx)]
    cov_high = cov[np.ix_(high_z_idx, high_z_idx)]
    z_low, z_high = z_values[low_z_idx], z_values[high_z_idx]
    q_low = [quantities[i] for i in low_z_idx]
    q_high = [quantities[i] for i in high_z_idx]

    def neg_log_likelihood(params):
        cosmo = CosmologyParams(w0=params[0], wa=params[1])
        theory = _build_theory_vector(compute_bao_predictions(z_low, cosmo), z_low, q_low)
        return -log_likelihood(data_low, theory, cov_low)

    result = minimize(
        neg_log_likelihood,
        x0=[-0.9, -0.5],
        bounds=[(-2.0, 0.0), (-3.0, 2.0)],
        method='L-BFGS-B'
    )
    alt_cosmo = CosmologyParams(w0=result.x[0], wa=result.x[1])

    theory_null_low = _build_theory_vector(compute_bao_predictions(z_low, LCDM), z_low, q_low)
    theory_null_high = _build_theory_vector(compute_bao_predictions(z_high, LCDM), z_high, q_high)
    theory_alt_low = _build_theory_vector(compute_bao_predictions(z_low, alt_cosmo), z_low, q_low)
    theory_alt_high = _build_theory_vector(compute_bao_predictions(z_high, alt_cosmo), z_high, q_high)

    # Train-set e-value is for reference only; it is NOT used for inference.
    e_train = likelihood_ratio_evalue(data_low, cov_low, theory_null_low, theory_alt_low)
    e_train.method = 'split_train'
    e_train.alt_params = alt_cosmo

    e_test = likelihood_ratio_evalue(data_high, cov_high, theory_null_high, theory_alt_high)
    e_test.method = 'split_test'
    e_test.alt_params = alt_cosmo

    return e_train, e_test


def split_evalue_by_indices(data, cov, z_values, quantities, train_idx, test_idx):
    """Data-split e-value for an arbitrary train/test partition of the measurements.

    Fits (w0, wa) on ``train_idx`` and evaluates the likelihood-ratio e-value on
    the held-out ``test_idx``. Generalises :func:`split_evalue` (a fixed-redshift
    split) to alternating-bin, random, or any index-based partition.

    Returns (e_train, e_test): EValueResult on the train (reference) and test
    (valid) sets.
    """
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)
    q_tr = [quantities[i] for i in train_idx]
    q_te = [quantities[i] for i in test_idx]
    d_tr, d_te = data[train_idx], data[test_idx]
    c_tr = cov[np.ix_(train_idx, train_idx)]
    c_te = cov[np.ix_(test_idx, test_idx)]
    z_tr, z_te = z_values[train_idx], z_values[test_idx]

    def neg_log_likelihood(params):
        cosmo = CosmologyParams(w0=params[0], wa=params[1])
        theory = _build_theory_vector(compute_bao_predictions(z_tr, cosmo), z_tr, q_tr)
        return -log_likelihood(d_tr, theory, c_tr)

    result = minimize(neg_log_likelihood, x0=[-0.9, -0.5],
                      bounds=[(-2.0, 0.0), (-3.0, 2.0)], method='L-BFGS-B')
    alt_cosmo = CosmologyParams(w0=result.x[0], wa=result.x[1])

    theory_null_te = _build_theory_vector(compute_bao_predictions(z_te, LCDM), z_te, q_te)
    theory_alt_te = _build_theory_vector(compute_bao_predictions(z_te, alt_cosmo), z_te, q_te)
    e_test = likelihood_ratio_evalue(d_te, c_te, theory_null_te, theory_alt_te)
    e_test.method = 'split_test'
    e_test.alt_params = alt_cosmo

    theory_null_tr = _build_theory_vector(compute_bao_predictions(z_tr, LCDM), z_tr, q_tr)
    theory_alt_tr = _build_theory_vector(compute_bao_predictions(z_tr, alt_cosmo), z_tr, q_tr)
    e_train = likelihood_ratio_evalue(d_tr, c_tr, theory_null_tr, theory_alt_tr)
    e_train.method = 'split_train'
    e_train.alt_params = alt_cosmo

    return e_train, e_test


def grow_evalue(
    data: np.ndarray,
    cov: np.ndarray,
    z_values: np.ndarray,
    quantities: list,
) -> EValueResult:
    """
    Uniform-mixture e-value: average L(data|theta)/L(data|H0) over a grid of (w0, wa).

    Equivalent to a Bayes factor with a discrete uniform prior; valid e-value
    (E[E|H0] = 1). This is NOT the GROW-optimal e-value of Grünwald et al.
    (2024) -- weights are uniform, not optimized -- so the result depends on the
    grid range. 'uniform_mixture_evalue' is an alias of this function.
    """
    # 30x30 "Default" prior grid used in the paper (M_DR2 = 33.97).
    # Run-scripts override the range via the sensitivity-analysis loop.
    w0_grid = np.linspace(-1.5, -0.5, 30)
    wa_grid = np.linspace(-2.0, 1.0, 30)

    theory_null = _build_theory_vector(
        compute_bao_predictions(z_values, LCDM), z_values, quantities)
    log_L_null = log_likelihood(data, theory_null, cov)

    log_ratios = []
    for w0 in w0_grid:
        for wa in wa_grid:
            cosmo = CosmologyParams(w0=w0, wa=wa)
            theory_alt = _build_theory_vector(
                compute_bao_predictions(z_values, cosmo), z_values, quantities)
            log_ratios.append(log_likelihood(data, theory_alt, cov) - log_L_null)
    log_ratios = np.array(log_ratios)

    # Log-sum-exp average over the uniform mixture (valid by Jensen's inequality).
    max_log_ratio = np.max(log_ratios)
    log_e = max_log_ratio + np.log(np.mean(np.exp(log_ratios - max_log_ratio)))

    best_idx = np.argmax(log_ratios)
    best_cosmo = CosmologyParams(
        w0=w0_grid[best_idx // len(wa_grid)],
        wa=wa_grid[best_idx % len(wa_grid)]
    )
    theory_best = _build_theory_vector(
        compute_bao_predictions(z_values, best_cosmo), z_values, quantities)
    chi2_null = chi_squared(data, theory_null, cov)
    chi2_alt = chi_squared(data, theory_best, cov)

    return EValueResult(
        e_value=np.exp(log_e),
        log_e=log_e,
        chi2_null=chi2_null,
        chi2_alt=chi2_alt,
        delta_chi2=chi2_null - chi2_alt,
        null_params=LCDM,
        alt_params=best_cosmo,
        method='uniform_mixture'
    )


uniform_mixture_evalue = grow_evalue


def sequential_evalue(
    datasets: list,
    covs: list,
    z_arrays: list,
    quantities_arrays: list
) -> EValueResult:
    """
    Sequential e-value: product of per-dataset e-values (E = E_1 * ... * E_n).

    Valid under optional stopping, so testing can stop once E exceeds a
    threshold while retaining type-I error control.
    """
    log_e_total = 0.0
    chi2_null_total = 0.0
    chi2_alt_total = 0.0

    for data, cov, z_values, quantities in zip(datasets, covs, z_arrays, quantities_arrays):
        e_result = grow_evalue(data, cov, z_values, quantities)
        log_e_total += e_result.log_e
        chi2_null_total += e_result.chi2_null
        chi2_alt_total += e_result.chi2_alt

    return EValueResult(
        e_value=np.exp(log_e_total),
        log_e=log_e_total,
        chi2_null=chi2_null_total,
        chi2_alt=chi2_alt_total,
        delta_chi2=chi2_null_total - chi2_alt_total,
        null_params=LCDM,
        alt_params=None,
        method='sequential_uniform_mixture'
    )


@dataclass
class LOOCVResult:
    """Results from LOOCV e-value computation.

    The valid LOO e-value is the AVERAGE of the per-bin E_k values (valid by
    linearity of expectation). The PRODUCT is NOT a valid e-value because the
    K leave-one-out training sets overlap heavily (any two folds share K-2 of
    K bins), so the per-fold alternative parameters and the per-fold E_k
    values are correlated across folds. We report both for transparency but
    only the average should be cited as the calibrated LOO evidence.
    """
    e_average: float  # AVERAGE of per-bin E_k (the valid LOO e-value)
    e_product_invalid: float  # PRODUCT of per-bin E_k (NOT valid; overlapping training sets)
    log_product: float  # Sum of log(E_k) = log(product), kept for numerical inspection
    per_bin_e: Dict[float, float]
    per_bin_log_e: Dict[float, float]
    per_bin_w0: Dict[float, float]
    per_bin_wa: Dict[float, float]
    chi2_null_total: float
    chi2_alt_total: float

    @property
    def e_value(self) -> float:
        """Alias for e_average (the valid LOO e-value)."""
        return self.e_average

    @property
    def log_e(self) -> float:
        """log of the AVERAGE e-value (for EValueResult-compatible interface)."""
        if self.e_average <= 0:
            return -np.inf
        return float(np.log(self.e_average))

    @property
    def sigma_equivalent(self) -> float:
        """sigma equivalent of the AVERAGE (the valid LOO e-value)."""
        if self.e_average <= 1:
            return 0.0
        return np.sqrt(2.0 * np.log(self.e_average))


def _identify_redshift_bins(z_values: np.ndarray) -> Dict[float, list]:
    """Group measurement indices by redshift bin.

    Returns dict mapping each unique redshift to a list of indices into the
    data/z_values arrays.
    """
    bins = {}
    for i, z in enumerate(z_values):
        # Round to avoid floating-point matching of nominally-equal redshifts.
        z_key = round(float(z), 4)
        bins.setdefault(z_key, []).append(i)
    return bins


def loocv_evalue(
    data: np.ndarray,
    cov: np.ndarray,
    z_values: np.ndarray,
    quantities: list,
    verbose: bool = True
) -> LOOCVResult:
    """
    Leave-One-Out Cross-Validation e-value over redshift bins.

    For each redshift bin: hold it out, fit (w0, wa) on the remaining
    measurements, and compute the held-out likelihood-ratio E_k. Each E_k uses
    training data disjoint from its test point, avoiding overfitting.

    The valid LOO e-value is the AVERAGE of the E_k (linearity of expectation);
    the product is reported but flagged invalid (overlapping training sets).

    Parameters
    ----------
    data : measurement vector (length 13 for DESI DR2)
    cov : full covariance matrix (block-diagonal by redshift bin)
    z_values : effective redshifts for each measurement
    quantities : quantity labels ('DM_over_rs', 'DH_over_rs', 'DV_over_rs')
    verbose : ignored (the function prints nothing)
    """
    bins = _identify_redshift_bins(z_values)

    per_bin_e = {}
    per_bin_log_e = {}
    per_bin_w0 = {}
    per_bin_wa = {}
    chi2_null_total = 0.0
    chi2_alt_total = 0.0

    for z_held_out, held_out_idx in sorted(bins.items()):
        held_out_idx = np.array(held_out_idx)
        train_idx = np.array([i for i in range(len(data)) if i not in held_out_idx])

        data_train = data[train_idx]
        cov_train = cov[np.ix_(train_idx, train_idx)]
        z_train = z_values[train_idx]
        q_train = [quantities[i] for i in train_idx]

        data_test = data[held_out_idx]
        cov_test = cov[np.ix_(held_out_idx, held_out_idx)]
        z_test = z_values[held_out_idx]
        q_test = [quantities[i] for i in held_out_idx]

        def neg_log_likelihood_train(params):
            cosmo = CosmologyParams(w0=params[0], wa=params[1])
            theory = _build_theory_vector(compute_bao_predictions(z_train, cosmo), z_train, q_train)
            return -log_likelihood(data_train, theory, cov_train)

        # Multi-start to avoid local minima in the (w0, wa) fit.
        best_result = None
        best_nll = np.inf
        starts = [
            [-0.9, -0.5],
            [-1.0, 0.0],
            [-0.75, -1.0],
            [-0.8, -0.8],
            [-1.1, 0.5],
        ]
        for x0 in starts:
            try:
                result = minimize(
                    neg_log_likelihood_train,
                    x0=x0,
                    bounds=[(-2.0, 0.0), (-4.0, 3.0)],
                    method='L-BFGS-B'
                )
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_result = result
            except Exception:
                continue

        w0_fit, wa_fit = best_result.x
        alt_cosmo = CosmologyParams(w0=w0_fit, wa=wa_fit)

        theory_null_test = _build_theory_vector(
            compute_bao_predictions(z_test, LCDM), z_test, q_test)
        theory_alt_test = _build_theory_vector(
            compute_bao_predictions(z_test, alt_cosmo), z_test, q_test)

        log_ek = (log_likelihood(data_test, theory_alt_test, cov_test)
                  - log_likelihood(data_test, theory_null_test, cov_test))

        per_bin_e[z_held_out] = np.exp(log_ek)
        per_bin_log_e[z_held_out] = log_ek
        per_bin_w0[z_held_out] = w0_fit
        per_bin_wa[z_held_out] = wa_fit
        chi2_null_total += chi_squared(data_test, theory_null_test, cov_test)
        chi2_alt_total += chi_squared(data_test, theory_alt_test, cov_test)

    # AVERAGE is the valid LOO e-value; product is flagged invalid (overlapping folds).
    e_average = float(np.mean(list(per_bin_e.values())))
    log_product = sum(per_bin_log_e.values())
    e_product_invalid = np.exp(log_product)

    return LOOCVResult(
        e_average=e_average,
        e_product_invalid=e_product_invalid,
        log_product=log_product,
        per_bin_e=per_bin_e,
        per_bin_log_e=per_bin_log_e,
        per_bin_w0=per_bin_w0,
        per_bin_wa=per_bin_wa,
        chi2_null_total=chi2_null_total,
        chi2_alt_total=chi2_alt_total,
    )


def _build_theory_vector(pred: dict, z_values: np.ndarray, quantities: list) -> np.ndarray:
    """Build a theory vector matching the data's (z, quantity) structure."""
    theory = np.zeros(len(quantities))
    for i, (z, q) in enumerate(zip(z_values, quantities)):
        z_idx = np.argmin(np.abs(pred['z'] - z))
        if 'DM' in q:
            theory[i] = pred['DM_over_rd'][z_idx]
        elif 'DH' in q:
            theory[i] = pred['DH_over_rd'][z_idx]
        elif 'DV' in q:
            theory[i] = pred['DV_over_rd'][z_idx]
        else:
            raise ValueError(f"Unknown BAO quantity: {q}")
    return theory


def mixture_log_e(data, cov, z_values, quantities, w0_grid, wa_grid):
    """Log mixture e-value over a (w0, wa) grid prior.

    Returns (log_e, per-grid log-likelihood-ratios); the ratios are returned so
    callers can reuse them for prior-sensitivity sweeps without recomputing the
    BAO predictions.
    """
    pred_null = compute_bao_predictions(z_values, LCDM)
    theory_null = _build_theory_vector(pred_null, z_values, quantities)
    log_L_null = log_likelihood(data, theory_null, cov)
    log_ratios = []
    for w0 in w0_grid:
        for wa in wa_grid:
            cosmo = CosmologyParams(w0=w0, wa=wa)
            pred = compute_bao_predictions(z_values, cosmo)
            theory_alt = _build_theory_vector(pred, z_values, quantities)
            log_L_alt = log_likelihood(data, theory_alt, cov)
            log_ratios.append(log_L_alt - log_L_null)
    log_ratios = np.array(log_ratios)
    m = np.max(log_ratios)
    log_e = m + np.log(np.mean(np.exp(log_ratios - m)))
    return log_e, log_ratios


def precompute_kernels(z_values, quantities, cov, w0_grid, wa_grid):
    """Precompute the affine mixture kernel for fast Monte-Carlo e-values under H0.

    Returns mu_null (null mean), L_chol (Cholesky of cov, to draw correlated
    residuals), and the per-grid score matrix A and quadratic term const, so that
    log-LR = A @ eps - const/2 for a residual eps = data - mu_null.
    """
    n = len(z_values)
    M = len(w0_grid) * len(wa_grid)
    Cinv = np.linalg.inv(cov)
    L_chol = np.linalg.cholesky(cov)
    pred_null = compute_bao_predictions(z_values, LCDM)
    mu_null = _build_theory_vector(pred_null, z_values, quantities)
    T_alt = np.zeros((M, n))
    g = 0
    for w0 in w0_grid:
        for wa in wa_grid:
            cosmo = CosmologyParams(w0=w0, wa=wa)
            pred = compute_bao_predictions(z_values, cosmo)
            T_alt[g] = _build_theory_vector(pred, z_values, quantities)
            g += 1
    delta = T_alt - mu_null[None, :]
    A = delta @ Cinv
    const = np.einsum('gi,ij,gj->g', delta, Cinv, delta)
    return dict(mu_null=mu_null, A=A, const=const, L_chol=L_chol, n=n, M=M)


def mixture_log_e_from_residuals(eps, kernels):
    """Vectorised mixture log e-value for residuals eps = data - mu_null, shape (n, B) -> (B,)."""
    log_LR = kernels['A'] @ eps - 0.5 * kernels['const'][:, None]
    m = log_LR.max(axis=0)
    return m + np.log(np.mean(np.exp(log_LR - m), axis=0))


if __name__ == "__main__":
    # Self-test on mock data drawn from a slightly non-LCDM model.
    np.random.seed(42)
    z_test = np.array([0.3, 0.5, 0.5, 0.7, 0.7, 1.0, 1.0, 1.3, 1.3])
    quantities = ['DV_over_rs', 'DM_over_rs', 'DH_over_rs',
                  'DM_over_rs', 'DH_over_rs', 'DM_over_rs', 'DH_over_rs',
                  'DM_over_rs', 'DH_over_rs']

    true_cosmo = CosmologyParams(w0=-0.9, wa=-0.3)
    theory_true = _build_theory_vector(
        compute_bao_predictions(z_test, true_cosmo), z_test, quantities)
    errors = theory_true * 0.02
    cov = np.diag(errors**2)
    data = theory_true + np.random.multivariate_normal(np.zeros(len(z_test)), cov)

    theory_lcdm = _build_theory_vector(
        compute_bao_predictions(z_test, LCDM), z_test, quantities)
    e_simple = likelihood_ratio_evalue(data, cov, theory_lcdm, theory_true)
    e_grow = grow_evalue(data, cov, z_test, quantities)

    print(f"likelihood_ratio_evalue: E = {e_simple.e_value:.3f}")
    print(f"grow_evalue (uniform mixture): E = {e_grow.e_value:.3f}")
