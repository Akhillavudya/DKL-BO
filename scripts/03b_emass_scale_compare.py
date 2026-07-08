"""Phase 2 (rebuild) — does modelling emass in LOG10 help? Compare raw vs log.

emass spans ~5 orders of magnitude (0.001–136), so a few huge values dominate raw-scale
errors. Paper 2 worked in log10(emass). This script measures emass prediction four ways:

    {Standard GP, DKL} x {raw emass, log10 emass}

For the LOG models we report BOTH:
  - log-space metrics (the natural scale the model was fit on), and
  - raw-scale metrics after back-transforming predictions (10**pred) — so raw and log
    models are compared on identical raw units (apples to apples).

Needs (run first):
  scripts/02_pretrain_encoder.py --target emass
  scripts/02_pretrain_encoder.py --target emass --log

Usage
    python scripts/03b_emass_scale_compare.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
import torch

from dklbo.eval.metrics_accuracy import compute_accuracy_metrics
from dklbo.models.surrogate import ExactGPSurrogate

CACHE_DIR = REPO / "data" / "cache"
OUT_DIR = REPO / "results" / "rebuild"


def fit_predict(feat_path: Path, df: pd.DataFrame, y_all: np.ndarray):
    """Fit ARD ExactGP on train features+target, return (pool_pred, pool_true) in y_all units."""
    feats = pd.read_parquet(feat_path).set_index("uid")
    tr = df[df.split == "train"]
    po = df[df.split == "pool"]
    Xtr = feats.loc[tr.uid].to_numpy("float64")
    Xpo = feats.loc[po.uid].to_numpy("float64")
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-9] = 1.0
    Xtr = torch.tensor((Xtr - mu) / sd)
    Xpo = torch.tensor((Xpo - mu) / sd)

    ys = pd.Series(y_all, index=df.uid)
    ytr_raw = ys.loc[tr.uid].to_numpy("float64")
    ypo_raw = ys.loc[po.uid].to_numpy("float64")
    ym, ystd = ytr_raw.mean(), ytr_raw.std() or 1.0
    ytr = torch.tensor((ytr_raw - ym) / ystd)

    gp = ExactGPSurrogate(n_epochs=200, ard=True)
    gp.fit(Xtr, ytr)
    gp.eval_mode()
    mean_s, _ = gp.predict(Xpo)
    pred = mean_s.numpy() * ystd + ym
    return pred, ypo_raw


def raw_metrics(pred_raw, true_raw):
    a = compute_accuracy_metrics(torch.tensor(true_raw, dtype=torch.float32),
                                 torch.tensor(pred_raw, dtype=torch.float32))
    return a.mae, a.rmse, a.r2


def main() -> None:
    df = pd.read_parquet(CACHE_DIR / "master.parquet")
    emass = df["emass"].to_numpy("float64")
    log_emass = np.log10(emass)

    descr = CACHE_DIR / "descriptors.parquet"
    feat = {
        ("std_gp", "raw"): descr,
        ("dkl", "raw"): OUT_DIR / "embeddings_emass.parquet",
        ("std_gp", "log"): descr,
        ("dkl", "log"): OUT_DIR / "embeddings_emass_log.parquet",
    }
    labels = {"std_gp": "Standard GP", "dkl": "DKL"}

    rows = []
    for (method, mode), path in feat.items():
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — train the encoder first.")
        if mode == "raw":
            pred, true = fit_predict(path, df, emass)
            mae, rmse, r2 = raw_metrics(pred, true)
            log_r2 = np.nan
        else:  # log: fit in log space, then back-transform to raw
            pred_log, true_log = fit_predict(path, df, log_emass)
            log_r2 = compute_accuracy_metrics(
                torch.tensor(true_log, dtype=torch.float32),
                torch.tensor(pred_log, dtype=torch.float32)).r2
            pred, true = 10.0 ** pred_log, 10.0 ** true_log
            mae, rmse, r2 = raw_metrics(pred, true)
        rows.append(dict(method=method, mode=mode, mae_raw=mae, rmse_raw=rmse,
                         r2_raw=r2, r2_log=log_r2))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "emass_scale_compare.csv", index=False)

    print(f"\nEmass prediction on held-out pool ({int((df.split=='pool').sum())} materials)")
    print("Raw-scale metrics (back-transformed for log models) + log-space R² for log models:\n")
    print(f"  {'method':12s} {'mode':4s} | {'MAE_raw':>9s} {'RMSE_raw':>9s} "
          f"{'R²_raw':>8s} {'R²_log':>8s}")
    print("  " + "-" * 58)
    for _, r in out.iterrows():
        rlog = f"{r.r2_log:8.3f}" if not np.isnan(r.r2_log) else f"{'—':>8s}"
        print(f"  {labels[r.method]:12s} {r['mode']:4s} | {r.mae_raw:9.3f} "
              f"{r.rmse_raw:9.3f} {r.r2_raw:8.3f} {rlog}")
    print(f"\nSaved → {OUT_DIR/'emass_scale_compare.csv'}")


if __name__ == "__main__":
    main()
