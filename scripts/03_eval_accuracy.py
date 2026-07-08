"""Phase 2 (clean rebuild) — prediction accuracy: Std-GP vs DKL, on gap and emass.

The question
-----------
"How well does each method PREDICT the property?" — a different question from "how good
is the search" (that's Phase 3). For every (target, method) we fit the SAME ExactGP on the
`train` split and predict the held-out `pool`, so the only thing that differs is the
features:

  std_gp : 43 handcrafted descriptors      (data/cache/descriptors.parquet)
  dkl    : 32-d learned embeddings          (results/rebuild/embeddings_{target}.parquet)

Random has no predictive model, so it does not appear here.

Fairness: identical GP procedure (ARD Matérn-5/2, train-standardized features AND target)
on both feature sets — apples to apples. Metrics: MAE, RMSE, R² (accuracy) + coverage@95
(calibration). Run scripts/02_pretrain_encoder.py for BOTH targets first.

Outputs (results/rebuild/)
  accuracy.csv
  plots/accuracy.png

Usage
    python scripts/03_eval_accuracy.py
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

CACHE_DIR = REPO / "data" / "cache"
OUT_DIR = REPO / "results" / "rebuild"
PLOTS = OUT_DIR / "plots"

TARGETS = ["gap", "emass"]
LABELS = {"std_gp": "Standard GP", "dkl": "DKL"}
COLORS = {"std_gp": "#16A34A", "dkl": "#2563EB"}


def feature_paths(target: str) -> dict:
    return {
        "std_gp": CACHE_DIR / "descriptors.parquet",
        "dkl": OUT_DIR / f"embeddings_{target}.parquet",
    }


def evaluate(feat_path: Path, df: pd.DataFrame, target: str) -> dict:
    """Fit ExactGP on train features → predict pool. Return accuracy + calibration."""
    feats = pd.read_parquet(feat_path).set_index("uid")
    tr = df[df.split == "train"]
    po = df[df.split == "pool"]
    Xtr = feats.loc[tr.uid].to_numpy("float64")
    Xpo = feats.loc[po.uid].to_numpy("float64")

    # Standardize features by TRAIN statistics (no pool leakage).
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-9] = 1.0
    Xtr = torch.tensor((Xtr - mu) / sd)
    Xpo = torch.tensor((Xpo - mu) / sd)

    # Standardize the target by TRAIN statistics; un-standardize before metrics.
    y = df.set_index("uid")[target]
    ytr_raw = y.loc[tr.uid].to_numpy("float64")
    ypo_raw = y.loc[po.uid].to_numpy("float64")
    ym, ys = ytr_raw.mean(), ytr_raw.std() or 1.0
    ytr = torch.tensor((ytr_raw - ym) / ys)

    gp = ExactGPSurrogate(n_epochs=200, ard=True)
    gp.fit(Xtr, ytr)
    gp.eval_mode()
    mean_s, std_s = gp.predict(Xpo)
    mean = mean_s * ys + ym
    std = std_s * ys
    ypo = torch.tensor(ypo_raw, dtype=torch.float32)

    acc = compute_accuracy_metrics(ypo, mean.float())
    cal = compute_calibration_metrics(ypo, mean.float(), std.float())
    return {"mae": acc.mae, "rmse": acc.rmse, "r2": acc.r2,
            "coverage_95": cal.coverage_95}


def main() -> None:
    df = pd.read_parquet(CACHE_DIR / "master.parquet")
    n_pool = int((df.split == "pool").sum())
    PLOTS.mkdir(parents=True, exist_ok=True)

    rows = []
    for target in TARGETS:
        print(f"\n=== {target.upper()} prediction on held-out pool ({n_pool} materials) ===")
        for method, path in feature_paths(target).items():
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} missing. Run scripts/02_pretrain_encoder.py --target {target} first."
                )
            r = evaluate(path, df, target)
            r.update(target=target, method=method)
            rows.append(r)
            print(f"  {LABELS[method]:12s}  MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}  "
                  f"R²={r['r2']:.3f}  cov95={r['coverage_95']:.3f}")

    out = pd.DataFrame(rows)[["target", "method", "mae", "rmse", "r2", "coverage_95"]]
    out.to_csv(OUT_DIR / "accuracy.csv", index=False)

    # Plot: one row per target, three panels (MAE, RMSE, R²), Std-GP vs DKL bars.
    fig, axes = plt.subplots(len(TARGETS), 3, figsize=(14, 4 * len(TARGETS)))
    panels = [("mae", "MAE — lower better"), ("rmse", "RMSE — lower better"),
              ("r2", "R² — higher better")]
    for ri, target in enumerate(TARGETS):
        sub = out[out.target == target].set_index("method")
        for ax, (col, title) in zip(axes[ri], panels):
            methods = ["std_gp", "dkl"]
            ax.bar([LABELS[m] for m in methods], [sub.loc[m, col] for m in methods],
                   color=[COLORS[m] for m in methods])
            ax.set_title(f"{target}: {title}", fontweight="bold")
            ax.grid(alpha=0.3, axis="y")
            for i, m in enumerate(methods):
                v = sub.loc[m, col]
                ax.text(i, v, f"{v:.2f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=10)
    fig.suptitle(f"Phase 2 — prediction accuracy on held-out pool ({n_pool} materials)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(PLOTS / "accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {OUT_DIR/'accuracy.csv'}  and  {PLOTS/'accuracy.png'}")


if __name__ == "__main__":
    main()
