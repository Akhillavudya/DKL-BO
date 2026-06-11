"""Post-hoc uncertainty recalibration (E2).

The GP may be over- or under-confident even after ARD (E1).
We fit a correction on the val set and apply it at inference time.

Two methods implemented
-----------------------
Temperature scaling (σ-scaling)
    Fit a single scalar τ that minimises val NLL.
    τ > 1  →  model was overconfident (σ too small); scale σ up
    τ < 1  →  model was underconfident; scale σ down
    Easy to apply in the BO loop: σ_cal = τ · σ_raw

Isotonic recalibration (Kuleshov 2018)
    Non-parametric map from predicted quantile to empirical quantile.
    Uses sklearn IsotonicRegression on the val probability-integral-transform.
    Used here as a diagnostic only (harder to invert to a single σ).

Usage
-----
    tau = fit_temperature(val_y, val_mean, val_std)
    sigma_cal = val_std * tau     # calibrated std
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from torch import Tensor


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def fit_temperature(
    y_true: Tensor,
    mu: Tensor,
    sigma: Tensor,
) -> float:
    """Fit temperature τ = sqrt(mean z²) on val set.

    The closed-form MLE solution that minimises val NLL over τ is:
        τ = sqrt( mean[ ((y - μ) / σ)² ] )  (RMS z-score)

    Returns
    -------
    tau : float
        Scale factor.  Apply as σ_cal = tau * σ_raw.
    """
    y = y_true.numpy().astype(np.float64)
    m = mu.numpy().astype(np.float64)
    s = np.clip(sigma.numpy().astype(np.float64), 1e-8, None)

    z2 = ((y - m) / s) ** 2
    tau = float(np.sqrt(np.mean(z2)))
    return tau


# ---------------------------------------------------------------------------
# Isotonic recalibration
# ---------------------------------------------------------------------------

def fit_isotonic(
    y_true: Tensor,
    mu: Tensor,
    sigma: Tensor,
) -> IsotonicRegression:
    """Fit an isotonic calibration map on the val PIT values (Kuleshov 2018).

    The probability-integral-transform (PIT) of a well-calibrated prediction
    is Uniform[0, 1].  We fit isotonic regression:
        predicted_quantile → empirical_quantile

    Returns
    -------
    iso : sklearn.isotonic.IsotonicRegression
        Fitted map.  Call iso.predict([p]) to get calibrated coverage at
        confidence level p.
    """
    y = y_true.numpy().astype(np.float64)
    m = mu.numpy().astype(np.float64)
    s = np.clip(sigma.numpy().astype(np.float64), 1e-8, None)

    pit = norm.cdf((y - m) / s)

    n = len(pit)
    sorted_pit = np.sort(pit)
    empirical   = np.arange(1, n + 1) / n

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(sorted_pit, empirical)
    return iso


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

@dataclass
class RecalibrationReport:
    tau: float
    nll_before: float
    nll_after: float
    coverage95_before: float
    coverage95_after: float
    miscal_area_before: float
    miscal_area_after: float


def recalibration_report(
    y_true: Tensor,
    mu: Tensor,
    sigma: Tensor,
    tau: float,
) -> RecalibrationReport:
    """Compute calibration metrics before and after temperature scaling."""
    y = y_true.numpy().astype(np.float64)
    m = mu.numpy().astype(np.float64)
    s_raw = np.clip(sigma.numpy().astype(np.float64), 1e-8, None)
    s_cal = s_raw * tau

    def _nll(s):
        return float(np.mean(
            0.5 * np.log(2 * np.pi * s ** 2) + 0.5 * ((y - m) / s) ** 2
        ))

    def _cov95(s):
        z = norm.ppf(0.975)
        return float((np.abs(y - m) <= z * s).mean())

    def _miscal(s):
        conf_levels = np.linspace(0.05, 0.95, 19)
        areas = []
        for p in conf_levels:
            z = norm.ppf((1 + p) / 2)
            actual = (np.abs(y - m) <= z * s).mean()
            areas.append(abs(actual - p))
        return float(np.mean(areas))

    return RecalibrationReport(
        tau=tau,
        nll_before=_nll(s_raw),
        nll_after=_nll(s_cal),
        coverage95_before=_cov95(s_raw),
        coverage95_after=_cov95(s_cal),
        miscal_area_before=_miscal(s_raw),
        miscal_area_after=_miscal(s_cal),
    )


def print_recalibration_report(report: RecalibrationReport) -> None:
    """Print a formatted before/after recalibration table."""
    print("\n" + "=" * 55)
    print("  Recalibration (temperature scaling)")
    print("=" * 55)
    print(f"  τ (temperature) = {report.tau:.4f}  "
          f"({'overconfident → scale σ up' if report.tau > 1 else 'underconfident → scale σ down'})")
    print(f"  {'Metric':<20} {'Before':>10}  {'After':>10}")
    print("  " + "-" * 43)
    print(f"  {'NLL':<20} {report.nll_before:>10.4f}  {report.nll_after:>10.4f}")
    print(f"  {'Coverage@95':<20} {report.coverage95_before:>10.4f}  {report.coverage95_after:>10.4f}")
    print(f"  {'Miscal. area':<20} {report.miscal_area_before:>10.4f}  {report.miscal_area_after:>10.4f}")
    print("=" * 55 + "\n")
