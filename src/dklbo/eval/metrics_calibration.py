"""Calibration / UQ metrics from [P2].

A well-calibrated GP is one where its uncertainty estimates (σ) are reliable:
  - 95% confidence intervals should contain ~95% of true values
  - Large σ predictions should correspond to large actual errors

Metrics implemented (all from [P2] Section 3):
  NLL          — negative log-likelihood (lower = better)
  ENCE         — expected normalised calibration error (target 0.06–0.10)
  Miscal. area — area between reliability curve and diagonal (target 0.04–0.07)
  Spearman     — correlation between |error| and σ (higher = model knows when wrong)
  Coverage@95  — fraction of true values inside the 95% interval (should be ~0.95)
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, spearmanr
from torch import Tensor


@dataclass
class CalibrationMetrics:
    nll: float             # negative log-likelihood
    ence: float            # expected normalised calibration error
    miscal_area: float     # miscalibration area (reliability curve vs diagonal)
    spearman_rho: float    # Spearman(|error|, σ)
    coverage_95: float     # fraction inside 95% CI


def compute_calibration_metrics(
    y_true: Tensor,
    y_pred_mean: Tensor,
    y_pred_std: Tensor,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Compute the full [P2] calibration suite.

    Parameters
    ----------
    y_true      : [N]  ground-truth values
    y_pred_mean : [N]  GP predicted means
    y_pred_std  : [N]  GP predicted standard deviations (NOT variance)
    n_bins      : number of bins for ENCE and reliability curve
    """
    y = y_true.numpy().astype(np.float64)
    mu = y_pred_mean.numpy().astype(np.float64)
    sigma = np.clip(y_pred_std.numpy().astype(np.float64), 1e-8, None)

    errors = np.abs(y - mu)

    # ---- NLL -------------------------------------------------------
    # NLL = -mean log N(y; mu, sigma^2)
    nll = float(np.mean(
        0.5 * np.log(2 * np.pi * sigma ** 2) + 0.5 * ((y - mu) / sigma) ** 2
    ))

    # ---- ENCE ------------------------------------------------------
    # Sort by sigma, split into n_bins bins.
    # For each bin: RMV = sqrt(mean(sigma^2)), RMSE = sqrt(mean(error^2))
    # ENCE = mean_over_bins( |RMV - RMSE| / RMV )
    sort_idx = np.argsort(sigma)
    sigma_s = sigma[sort_idx]
    errors_s = errors[sort_idx]
    bins = np.array_split(np.arange(len(sigma_s)), n_bins)
    ence_terms = []
    for b in bins:
        if len(b) == 0:
            continue
        rmv  = np.sqrt(np.mean(sigma_s[b] ** 2))
        rmse = np.sqrt(np.mean(errors_s[b] ** 2))
        if rmv > 1e-8:
            ence_terms.append(abs(rmv - rmse) / rmv)
    ence = float(np.mean(ence_terms)) if ence_terms else 0.0

    # ---- Reliability curve + miscalibration area -------------------
    # For confidence levels p in [0.05, 0.10, ..., 0.95]:
    #   expected fraction = p
    #   actual fraction   = mean( |error| ≤ z_{(1+p)/2} · sigma )
    # Miscal. area = mean( |actual - expected| ) over confidence levels
    conf_levels = np.linspace(0.05, 0.95, 19)
    actual_fracs = []
    for p in conf_levels:
        z = norm.ppf((1 + p) / 2)              # e.g. z=1.96 for p=0.95
        inside = (errors <= z * sigma).mean()
        actual_fracs.append(inside)
    actual_fracs = np.array(actual_fracs)
    miscal_area = float(np.mean(np.abs(actual_fracs - conf_levels)))

    # ---- Spearman(|error|, sigma) ----------------------------------
    # Does the model know when it's wrong?  Should be positive.
    rho, _ = spearmanr(errors, sigma)
    spearman_rho = float(rho)

    # ---- Coverage @ 95% -------------------------------------------
    z95 = norm.ppf(0.975)                      # = 1.96
    coverage_95 = float((errors <= z95 * sigma).mean())

    return CalibrationMetrics(
        nll=nll,
        ence=ence,
        miscal_area=miscal_area,
        spearman_rho=spearman_rho,
        coverage_95=coverage_95,
    )


def print_metrics_table(
    acc,
    cal,
    split: str = "test",
    surrogate: str = "ExactGP",
) -> None:
    """Print a formatted metrics table to stdout."""
    print(f"\n{'='*55}")
    print(f"  Surrogate: {surrogate}   Split: {split}")
    print(f"{'='*55}")
    print(f"  ACCURACY")
    print(f"    MAE          {acc.mae:.4f} eV")
    print(f"    RMSE         {acc.rmse:.4f} eV")
    print(f"    R²           {acc.r2:.4f}")
    print(f"  CALIBRATION  (targets from [P2])")
    print(f"    NLL          {cal.nll:.4f}        (lower = better)")
    print(f"    ENCE         {cal.ence:.4f}        (target 0.06–0.10)")
    print(f"    Miscal. area {cal.miscal_area:.4f}        (target 0.04–0.07)")
    print(f"    Spearman ρ   {cal.spearman_rho:.4f}        (higher = better)")
    print(f"    Coverage@95  {cal.coverage_95:.4f}        (target ≈0.95)")
    print(f"{'='*55}\n")
