"""Phase 2 (SNUMAT side-test) — prediction accuracy: Std-GP vs DKL on the HSE gap.

"How well does each method PREDICT the band gap?" (a different question from "how good is the
search", which is Phase 3). For each method we fit the SAME ExactGP on the `train` split and
predict the held-out `pool`; only the features differ:

  std_gp : handcrafted descriptors   (data/cache/descriptors.parquet)
  dkl    : 32-d learned embeddings    (results/embeddings_gap.parquet)

Fairness: identical GP procedure (ARD Matérn-5/2, train-standardized features AND target) on
both feature sets. Metrics: MAE, RMSE, R² + coverage@95. Run script 02 first.

Outputs (results/): accuracy.csv, plots/accuracy.png
"""

import sys
from pathlib import Path

SNUMAT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SNUMAT_ROOT))
sys.path.insert(0, str(SNUMAT_ROOT.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from lib import config
from dklbo.eval.metrics_accuracy import compute_accuracy_metrics
from dklbo.eval.metrics_calibration import compute_calibration_metrics
from dklbo.models.surrogate import ExactGPSurrogate

PLOTS = config.RESULTS_DIR / "plots"
LABELS = {"std_gp": "Standard GP", "dkl": "DKL"}
COLORS = {"std_gp": "#16A34A", "dkl": "#2563EB"}


def feature_paths() -> dict:
    return {
        "std_gp": config.CACHE_DIR / "descriptors.parquet",
        "dkl": config.RESULTS_DIR / "embeddings_gap.parquet",
    }


def evaluate(feat_path: Path, df: pd.DataFrame) -> dict:
    """Fit ExactGP on train features → predict pool. Return accuracy + calibration."""
    feats = pd.read_parquet(feat_path).set_index("uid")
    tr = df[df.split == "train"]
    po = df[df.split == "pool"]
    Xtr = feats.loc[tr.uid].to_numpy("float64")
    Xpo = feats.loc[po.uid].to_numpy("float64")

    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-9] = 1.0
    Xtr = torch.tensor((Xtr - mu) / sd)
    Xpo = torch.tensor((Xpo - mu) / sd)

    y = df.set_index("uid")["gap"]
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
    df = pd.read_parquet(config.CACHE_DIR / "master.parquet")
    n_pool = int((df.split == "pool").sum())
    PLOTS.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"\n=== HSE GAP prediction on held-out pool ({n_pool} materials) ===")
    for method, path in feature_paths().items():
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Run script 02 first.")
        r = evaluate(path, df)
        r.update(method=method)
        rows.append(r)
        print(f"  {LABELS[method]:12s}  MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}  "
              f"R2={r['r2']:.3f}  cov95={r['coverage_95']:.3f}")

    out = pd.DataFrame(rows)[["method", "mae", "rmse", "r2", "coverage_95"]]
    out.to_csv(config.RESULTS_DIR / "accuracy.csv", index=False)

    sub = out.set_index("method")
    methods = ["std_gp", "dkl"]
    panels = [("mae", "MAE — lower better"), ("rmse", "RMSE — lower better"),
              ("r2", "R² — higher better")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (col, title) in zip(axes, panels):
        ax.bar([LABELS[m] for m in methods], [sub.loc[m, col] for m in methods],
               color=[COLORS[m] for m in methods])
        ax.set_title(f"gap: {title}", fontweight="bold")
        ax.grid(alpha=0.3, axis="y")
        for i, m in enumerate(methods):
            v = sub.loc[m, col]
            ax.text(i, v, f"{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=10)
    fig.suptitle(f"SNUMAT Phase 2 — HSE gap prediction on held-out pool ({n_pool} materials)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOTS / "accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {config.RESULTS_DIR/'accuracy.csv'}  and  {PLOTS/'accuracy.png'}")


if __name__ == "__main__":
    main()
