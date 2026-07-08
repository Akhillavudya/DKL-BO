"""Experiment — target-WINDOW band-gap search: Std-GP vs frozen DKL vs live-finetune DKL.

A different objective from max/min: we don't want the highest or lowest gap, we want gaps
that LAND INSIDE a band. Each cycle we pick the material the GP thinks is most likely to be
in-window, using the window-probability acquisition (see bo/acquisition.py:window):

    a(x) = P(lo <= gap <= hi) = Φ((hi-μ)/σ) - Φ((lo-μ)/σ)

Two windows, run separately and stored separately:
  visible_0p7_3p0 : 0.7-3.0 eV  — 65% of the pool qualifies (easy → expect a tie; sanity check)
  wide_3p0_4p5    : 3.0-4.5 eV  — only ~15% qualify, width 1.5 eV > model's ~0.73 eV MAE
                                  (rare BUT resolvable → the discriminating test)

Three methods (same per-seed init, same GP backend, same acquisition — only features/encoder
differ), plus Random as the floor:
  std_gp        43 handcrafted descriptors            (data/cache/descriptors.parquet)
  dkl_frozen    32-d pre-trained gap embeddings        (results/rebuild/embeddings_gap.parquet)
  dkl_finetune  encoder warm-started from encoder_gap.pt, fine-tuned live during search
  random        no model

Success metric: cumulative count of ACQUIRED materials whose true gap is in-window (post-init).
Init is identical across methods per seed, so it cancels out of the comparison.

Outputs (results/rebuild/)
  window_runs/{window}__{method}__seed{S}.csv   raw per-cycle records (idempotent/resumable)
  window_summary.csv                            per (window, method, seed) final in-window count
  window_stats.csv                              paired Wilcoxon vs Std-GP, per window
  plots/window_{window}.png                     cumulative in-window vs cycle + final-count bars

Usage
    python scripts/exp_window_bo.py                       # both windows, 4 methods, 30 seeds
    python scripts/exp_window_bo.py --seeds 2 --cycles 8  # smoke test
    python scripts/exp_window_bo.py --no_finetune         # skip the slow live-DKL method
"""

import argparse
import logging
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
from omegaconf import OmegaConf
from scipy.stats import wilcoxon

from dklbo.baselines.feature_bo_loop import FeatureBOLoop
from dklbo.bo.baselines import RandomBaseline
from dklbo.bo.loop import BOLoop
from dklbo.data.cache import GraphCache
from dklbo.models.cgcnn_encoder import CGCNNEncoder
from dklbo.models.dkl import DKLModel
from dklbo.models.surrogate import ExactGPSurrogate
from dklbo.utils.seed import seed_everything

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = REPO / "data" / "cache"
OUT = REPO / "results" / "rebuild"
RUNS = OUT / "window_runs"
PLOTS = OUT / "plots"

WINDOWS = {"visible_0p7_3p0": (0.7, 3.0), "wide_3p0_4p5": (3.0, 4.5)}
METHODS = ["std_gp", "dkl_frozen", "dkl_finetune", "random"]
LABELS = {"std_gp": "Standard GP", "dkl_frozen": "DKL (frozen)",
          "dkl_finetune": "DKL (fine-tuned, live)", "random": "Random"}
COLORS = {"std_gp": "#16A34A", "dkl_frozen": "#2563EB",
          "dkl_finetune": "#1E3A8A", "random": "#9CA3AF"}

ENCODER_KW = dict(atom_dim=90, bond_dim=10, hidden_dim=32, n_conv=3, n_fc=1,
                  pooling="attention")
GRAPH_PREPROC = {"radius": 8.0, "vacuum_cutoff": 4.0, "max_neighbors": 12,
                 "target": "gap_hse", "gap_min": 0.01}


def features(path: Path, pool: pd.DataFrame):
    d = pd.read_parquet(path).set_index("uid").loc[pool.uid]
    return d.index.tolist(), d.to_numpy("float64")


def build_dkl(device: str) -> DKLModel:
    enc = CGCNNEncoder(**ENCODER_KW)
    ckpt = OUT / "encoder_gap.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} missing — run scripts/02_pretrain_encoder.py --target gap")
    enc.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return DKLModel(encoder=enc, surrogate=ExactGPSurrogate(lr=0.01, n_epochs=100, ard=False),
                    encoder_lr=0.001, gp_lr=0.01, device=device, standardize=True)


def run_one(method, win_name, lo, hi, pool, loaded, cache, device, cfg) -> pd.DataFrame:
    seed = int(cfg.seed)
    if method == "random":
        return RandomBaseline(meta_df=pool, cfg=cfg, seed=seed, direction="max").run()
    if method == "dkl_finetune":
        dkl = build_dkl(device)
        return BOLoop(dkl=dkl, cache=cache, meta_df=pool, cfg=cfg,
                      seed=seed, direction="max").run()
    uids, X = loaded[method]
    return FeatureBOLoop(X=X, uids=uids, meta_df=pool, cfg=cfg, seed=seed,
                         direction="max", ard=True, gp_epochs=100).run()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--cycles", type=int, default=100)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--no_finetune", action="store_true",
                    help="skip the slow live-DKL method")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    methods = [m for m in METHODS if not (args.no_finetune and m == "dkl_finetune")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RUNS.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(CACHE_DIR / "master.parquet")
    pool = df[df.split == "pool"].reset_index(drop=True).copy()
    pool["target"] = pool["gap"].astype("float64").to_numpy()   # oracle = true gap (eV)
    loaded = {"std_gp": features(CACHE_DIR / "descriptors.parquet", pool),
              "dkl_frozen": features(OUT / "embeddings_gap.parquet", pool)}
    needs_cache = ("dkl_finetune" in methods)
    cache = GraphCache(str(CACHE_DIR), GRAPH_PREPROC, readonly=True) if needs_cache else None
    logger.info(f"pool={len(pool)} | windows={list(WINDOWS)} methods={methods} "
                f"seeds={args.seeds} cycles={args.cycles} device={device}")

    n_done = n_skip = 0
    for win_name, (lo, hi) in WINDOWS.items():
        base = {"acquisition": "window", "beta": 0.0, "xi": 0.01,
                "n_init": args.n_init, "n_cycles": args.cycles,
                "window_lo": lo, "window_hi": hi,
                "retrain_every_k": 5, "n_joint_epochs": 50,
                "n_pretrain_epochs": 20, "gp_refit_epochs": 50}
        for method in methods:
            for seed in seeds:
                out = RUNS / f"{win_name}__{method}__seed{seed}.csv"
                if out.exists():
                    n_skip += 1
                    continue
                seed_everything(seed)
                cfg = OmegaConf.create({**base, "seed": seed})
                res = run_one(method, win_name, lo, hi, pool, loaded, cache, device, cfg)
                res.insert(0, "window", win_name)
                res.insert(1, "method", method)
                res.insert(2, "seed", seed)
                res["in_window"] = res.gap_acquired.between(lo, hi)
                res["cumul_in_window"] = res.in_window.cumsum()
                res.to_csv(out, index=False)
                n_done += 1
                logger.info(f"  {win_name}/{method}/seed{seed} → "
                            f"in-window found={int(res.cumul_in_window.iloc[-1])}")
    if cache is not None:
        cache.close()
    logger.info(f"runs written={n_done} skipped={n_skip}. Summarizing…")
    summarize_and_plot()


def load_all() -> pd.DataFrame:
    frames = [pd.read_csv(c) for c in sorted(RUNS.glob("*.csv"))]
    if not frames:
        raise FileNotFoundError("No window runs found.")
    return pd.concat(frames, ignore_index=True)


def summarize_and_plot() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    df = load_all()

    # Per (window, method, seed): final cumulative in-window count.
    finals = (df.sort_values("cycle").groupby(["window", "method", "seed"])
              .tail(1)[["window", "method", "seed", "cumul_in_window"]]
              .rename(columns={"cumul_in_window": "final_in_window"}))
    finals.to_csv(OUT / "window_summary.csv", index=False)

    # Paired Wilcoxon vs Std-GP per window (matched by seed).
    rows = []
    for win in finals.window.unique():
        base = finals[(finals.window == win) & (finals.method == "std_gp")].set_index("seed")
        for m in [x for x in METHODS if x != "std_gp"]:
            cur = finals[(finals.window == win) & (finals.method == m)].set_index("seed")
            common = sorted(set(base.index) & set(cur.index))
            if len(common) < 2:
                continue
            x = cur.loc[common, "final_in_window"].to_numpy(float)
            y = base.loc[common, "final_in_window"].to_numpy(float)
            try:
                p = float(wilcoxon(x, y).pvalue) if not np.allclose(x, y) else float("nan")
            except ValueError:
                p = float("nan")
            rows.append({"window": win, "method": m, "n_seeds": len(common),
                         "method_mean": x.mean(), "std_gp_mean": y.mean(),
                         "mean_diff": (x - y).mean(), "wilcoxon_p": p,
                         "win_rate": float(np.mean(x > y))})
    stats = pd.DataFrame(rows)
    stats.to_csv(OUT / "window_stats.csv", index=False)

    # ── Plots: one figure per window (curve + final-count bars) ──
    for win, (lo, hi) in WINDOWS.items():
        sub = df[df.window == win]
        if sub.empty:
            continue
        present = [m for m in METHODS if m in sub.method.unique()]
        fig, (axc, axb) = plt.subplots(1, 2, figsize=(15, 5.5))
        for m in present:
            piv = sub[sub.method == m].pivot_table(index="seed", columns="cycle",
                                                    values="cumul_in_window")
            Y = piv.to_numpy(float)
            cyc = piv.columns.to_numpy()
            mean = Y.mean(0)
            n = max(Y.shape[0], 1)
            ci = 1.96 * Y.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
            axc.plot(cyc, mean, color=COLORS[m], lw=2.2, label=LABELS[m])
            axc.fill_between(cyc, mean - ci, mean + ci, color=COLORS[m], alpha=0.13)
        axc.set_xlabel("BO cycle")
        axc.set_ylabel("cumulative in-window materials found")
        axc.set_title(f"{win}  ({lo}–{hi} eV): in-window discoveries vs cycle",
                      fontweight="bold")
        axc.grid(alpha=0.3)
        axc.legend(fontsize=9)

        fwin = finals[finals.window == win]
        agg = fwin.groupby("method")["final_in_window"].agg(["mean", "std", "count"])
        agg = agg.reindex([m for m in present])
        ci = 1.96 * agg["std"] / np.sqrt(agg["count"].clip(lower=1))
        axb.bar([LABELS[m] for m in agg.index], agg["mean"],
                yerr=ci.to_numpy(), capsize=4,
                color=[COLORS[m] for m in agg.index])
        axb.set_title(f"{win}: final in-window found (mean ± 95% CI)", fontweight="bold")
        axb.set_ylabel("materials in-window after 100 cycles")
        axb.tick_params(axis="x", rotation=20)
        axb.grid(alpha=0.3, axis="y")
        for i, v in enumerate(agg["mean"]):
            axb.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(PLOTS / f"window_{win}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("\n" + "=" * 60)
    print("WINDOW SEARCH — final in-window count (mean over seeds), ★=p<0.05 vs Std-GP")
    print("=" * 60)
    for win in WINDOWS:
        fw = finals[finals.window == win]
        if fw.empty:
            continue
        print(f"\n  {win}")
        for m in [x for x in METHODS if x in fw.method.unique()]:
            mean = fw[fw.method == m].final_in_window.mean()
            tag = ""
            r = stats[(stats.window == win) & (stats.method == m)]
            if not r.empty:
                p = r.wilcoxon_p.iloc[0]
                tag = f"  {'★' if p < 0.05 else ' '} p={p:.3f} ({'+' if r.mean_diff.iloc[0]>0 else ''}{r.mean_diff.iloc[0]:.1f} vs Std-GP)"
            print(f"    {LABELS[m]:24s} {mean:6.1f}{tag}")
    print(f"\nSaved → window_summary.csv, window_stats.csv, plots/window_*.png")


if __name__ == "__main__":
    main()
