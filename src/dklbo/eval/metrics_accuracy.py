"""Predictive accuracy metrics: MAE, RMSE, R².

These measure how *correct* the mean predictions are, ignoring uncertainty.
From [P2]: accuracy and calibration are two separate axes — a model can be
accurate but badly calibrated (wrong error bars) or vice versa.
"""

from dataclasses import dataclass

import numpy as np
from torch import Tensor


@dataclass
class AccuracyMetrics:
    mae: float   # Mean Absolute Error (eV)
    rmse: float  # Root Mean Squared Error (eV)
    r2: float    # Coefficient of determination


def compute_accuracy_metrics(y_true: Tensor, y_pred: Tensor) -> AccuracyMetrics:
    """Compute MAE, RMSE, R² between true and predicted band gaps.

    Parameters
    ----------
    y_true : [N]  ground-truth HSE06 band gaps (eV)
    y_pred : [N]  model predicted means (eV)
    """
    y_true = y_true.numpy().astype(np.float64)
    y_pred = y_pred.numpy().astype(np.float64)

    residuals = y_pred - y_true
    mae  = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

    return AccuracyMetrics(mae=mae, rmse=rmse, r2=r2)
