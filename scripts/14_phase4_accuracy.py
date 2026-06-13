"""Phase 4 — prediction accuracy (MAE / RMSE / R²) on the held-out pool.

This measures how well each method's surrogate *predicts* the band gap — a different
question from "how good is the search" (which 12/13 cover). For each feature set we fit
the SAME GP on the phase-4 TRAIN split and predict the held-out POOL, so the comparison
is apples-to-apples:

  dkl_pretrained : 32-d embeddings from the encoder pre-trained on train
  dkl_cold       : 32-d embeddings from a random-init encoder
  std_gp         : 43 handcrafted descriptors

(Random has no model, so no accuracy metrics.)

Outputs
  results/phase4/accuracy.csv
  results/phase4/plots/accuracy_bars.png   (MAE, RMSE, R² — Phase-3 style)

Usage
-----
    python scripts/14_phase4_accuracy.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dklbo.eval.metrics_accuracy import compute_accuracy_metrics
from dklbo.eval.metrics_calibration import compute_calibration_metrics
from dklbo.models.surrogate import ExactGPSurrogate

PH4 = REPO / "results" / "phase4"
PLOTS = PH4 / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

FEATURES = {
    "dkl_pretrained": PH4 / "embeddings_pretrained.parquet",
    "std_gp":         REPO / "results" / "benchmark" / "descriptors_gap.parquet",
    "dkl_cold":       PH4 / "embeddings_cold.parquet",
}
LABELS = {"dkl_pretrained": "DKL (pre-trained)", "std_gp": "Standard GP",
          "dkl_cold": "DKL (cold)"}
COLORS = {"dkl_pretrained": "#2563EB", "std_gp": "#16A34A", "dkl_cold": "#F59E0B"}


def evaluate(path: Path, df: pd.DataFrame) -> dict:
    """Fit GP on train features, predict pool, return accuracy + calibration."""
    feats = pd.read_parquet(path).set_index("uid")
    tr = df[df.split4 == "train"]; po = df[df.split4 == "pool"]
    Xtr = feats.loc[tr.uid].to_numpy("float64")
    Xpo = feats.loc[po.uid].to_numpy("float64")

    # standardize features by train stats
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-9] = 1.0
    Xtr = torch.tensor((Xtr - mu) / sd); Xpo = torch.tensor((Xpo - mu) / sd)
    # standardize target by train stats (un-standardize before metrics)
    y = df.set_index("uid")["target"]
    ytr_raw = y.loc[tr.uid].to_numpy("float64")
    ypo_raw = y.loc[po.uid].to_numpy("float64")
    ym, ys = ytr_raw.mean(), ytr_raw.std() or 1.0
    ytr = torch.tensor((ytr_raw - ym) / ys)

    gp = ExactGPSurrogate(n_epochs=200, ard=True)
    gp.fit(Xtr, ytr); gp.eval_mode()
    mean_s, std_s = gp.predict(Xpo)
    mean = mean_s * ys + ym
    std = std_s * ys
    ypo = torch.tensor(ypo_raw, dtype=torch.float32)

    acc = compute_accuracy_metrics(ypo, mean.float())
    cal = compute_calibration_metrics(ypo, mean.float(), std.float())
    return {"mae": acc.mae, "rmse": acc.rmse, "r2": acc.r2,
            "coverage_95": cal.coverage_95}


def main():
    df = pd.read_parquet(REPO / "data" / "cache" / "metadata_phase4.parquet")
    n_pool = int((df.split4 == "pool").sum())

    rows = []
    for m, path in FEATURES.items():
        r = evaluate(path, df)
        r["method"] = m
        rows.append(r)
        print(f"  {LABELS[m]:18s}  MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}  "
              f"R²={r['r2']:.3f}  cov95={r['coverage_95']:.3f}")
    out = pd.DataFrame(rows)[["method", "mae", "rmse", "r2", "coverage_95"]]
    out.to_csv(PH4 / "accuracy.csv", index=False)

    # Bar plot: MAE, RMSE, R² (Phase-3 style)
    order = ["dkl_pretrained", "std_gp", "dkl_cold"]
    out = out.set_index("method").reindex(order)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, title, lower_better) in zip(axes, [
        ("mae", "MAE (eV) — lower is better", True),
        ("rmse", "RMSE (eV) — lower is better", True),
        ("r2", "R² — higher is better", False),
    ]):
        ax.bar([LABELS[m] for m in order], out[col],
               color=[COLORS[m] for m in order])
        ax.set_title(title, fontweight="bold")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(alpha=0.3, axis="y")
        for i, v in enumerate(out[col]):
            ax.text(i, v, f"{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=10)
    fig.suptitle(f"Phase 4 — band-gap prediction accuracy on held-out pool "
                 f"({n_pool} materials)", fontsize=15, fontweight="bold")
    fig.savefig(PLOTS / "accuracy_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {PH4/'accuracy.csv'}  and  {PLOTS/'accuracy_bars.png'}")


if __name__ == "__main__":
    main()
