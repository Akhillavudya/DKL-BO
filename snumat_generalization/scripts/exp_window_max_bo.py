"""SNUMAT side-test — CONSTRAINED-MAX band-gap search in the visible window:
Std-GP vs frozen DKL vs live-finetune DKL vs Random.

Mirror of the C2DB experiment `scripts/exp_window_max_bo.py`, run on the 3D SNUMAT pool.
We want the HIGHEST gap that still fits under a hard ceiling. The objective is a
constrained maximisation:

    reward(x) = gap(x)      if  WIN_LO <= gap(x) <= WIN_HI
              = REJECT       otherwise

For the visible window 0.7-3.0 eV the single best material in the pool is the one whose
gap is closest to 3.0 from below. A plain max search would instead chase the materials
above 3.0 and never settle — so the search is driven by the window_max acquisition
(see dklbo/bo/acquisition.py:window_max), the posterior partial expectation

    a(x) = E[ gap * 1{WIN_LO <= gap <= WIN_HI} ]

which rewards a high predicted gap that stays inside the band and discounts the mass that
leaks past the ceiling. As sigma -> 0 it targets the true constrained maximum.

Four methods (same per-seed init, same GP backend, same acquisition — only the
features/encoder differ):
  std_gp        handcrafted descriptors                 (data/cache/descriptors.parquet)
  dkl_frozen    32-d pre-trained gap embeddings          (results/embeddings_gap.parquet)
  dkl_finetune  encoder warm-started from encoder_gap.pt, fine-tuned live during search
  random        no model

Metrics (all post-init; the init set is identical across methods per seed, so it cancels):
  best_inwin_gap   running max of acquired gaps that fall inside the window ("the maximum")
  cumul_top10      cumulative # acquired among the 10 highest in-window gaps
  cumul_top50      cumulative # acquired among the 50 highest in-window gaps
The top-10 / top-50 thresholds are derived from the in-window pool gaps (descending).

Outputs (results/)
  winmax_runs/{method}__seed{S}.csv     raw per-cycle records (idempotent/resumable)
  winmax_summary.csv                    per (method, seed) final best-gap / top10 / top50
  winmax_stats.csv                      paired Wilcoxon vs Std-GP, per metric
  winmax_crossover.csv                  cycle where each DKL method overtakes Std-GP, per metric
  plots/winmax_best_gap.png             best in-window gap vs cycle (the "maximum" curve)
  plots/winmax_top50.png                cumulative top-50 found vs cycle
  plots/winmax_crossover.png            2 panels: crossover cycle for best-gap / top50

Usage
    python snumat_generalization/scripts/exp_window_max_bo.py                       # 4 methods, 10 seeds
    python snumat_generalization/scripts/exp_window_max_bo.py --seeds 2 --cycles 8  # smoke test
    python snumat_generalization/scripts/exp_window_max_bo.py --no_finetune         # skip slow live-DKL
    python snumat_generalization/scripts/exp_window_max_bo.py --lo 0.7 --hi 3.0     # change the window
"""

import argparse
import logging
import sys
from pathlib import Path

SNUMAT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SNUMAT_ROOT))
sys.path.insert(0, str(SNUMAT_ROOT.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.stats import wilcoxon

from lib import config
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

CACHE_DIR = config.CACHE_DIR
OUT = config.RESULTS_DIR
RUNS = OUT / "winmax_runs"
PLOTS = OUT / "plots"

METHODS = ["std_gp", "dkl_frozen", "dkl_finetune", "random"]
LABELS = {"std_gp": "Standard GP", "dkl_frozen": "DKL (frozen)",
          "dkl_finetune": "DKL (fine-tuned, live)", "random": "Random"}
COLORS = {"std_gp": "#16A34A", "dkl_frozen": "#2563EB",
          "dkl_finetune": "#1E3A8A", "random": "#9CA3AF"}
# DKL methods compared against Std-GP for the crossover analysis.
DKL_METHODS = ["dkl_frozen", "dkl_finetune"]

# top-10 still computed in add_metrics / run CSVs; reporting focuses on best-gap + top50.
METRICS = {"best_inwin_gap": "best in-window gap (eV)",
           "cumul_top50": "cumulative top-50 in-window found"}

# 3D SNUMAT encoder: atom one-hot widened to 100 (no vacuum filter).
ENCODER_KW = dict(atom_dim=config.GRAPH_PREPROC["atom_feat_dim"], bond_dim=10,
                  hidden_dim=32, n_conv=3, n_fc=1, pooling="attention")


def features(path: Path, pool: pd.DataFrame):
    d = pd.read_parquet(path).set_index("uid").loc[pool.uid]
    return d.index.tolist(), d.to_numpy("float64")


def build_dkl(device: str) -> DKLModel:
    enc = CGCNNEncoder(**ENCODER_KW)
    ckpt = OUT / "encoder_gap.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} missing — run scripts/02_pretrain_encoder.py first")
    enc.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return DKLModel(encoder=enc, surrogate=ExactGPSurrogate(lr=0.01, n_epochs=100, ard=False),
                    encoder_lr=0.001, gp_lr=0.01, device=device, standardize=True)


def run_one(method, pool, loaded, cache, device, cfg) -> pd.DataFrame:
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


def inwin_thresholds(pool: pd.DataFrame, lo: float, hi: float):
    """Top-10 / top-50 gap thresholds among IN-WINDOW pool materials (descending)."""
    g = pool.loc[pool.target.between(lo, hi), "target"].sort_values(ascending=False)
    t10 = float(g.iloc[min(9, len(g) - 1)])
    t50 = float(g.iloc[min(49, len(g) - 1)])
    return t10, t50


def add_metrics(res: pd.DataFrame, lo: float, hi: float, t10: float, t50: float):
    """Annotate a per-cycle run with the constrained-max window metrics."""
    in_win = res.gap_acquired.between(lo, hi)
    inwin_gap = res.gap_acquired.where(in_win, other=np.nan)
    res["in_window"] = in_win
    res["best_inwin_gap"] = inwin_gap.cummax().ffill().fillna(lo)
    res["is_top10"] = in_win & (res.gap_acquired >= t10)
    res["is_top50"] = in_win & (res.gap_acquired >= t50)
    res["cumul_top10"] = res.is_top10.cumsum()
    res["cumul_top50"] = res.is_top50.cumsum()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=config.N_SEEDS)
    ap.add_argument("--cycles", type=int, default=config.N_CYCLES)
    ap.add_argument("--n_init", type=int, default=config.N_INIT)
    ap.add_argument("--lo", type=float, default=0.7)
    ap.add_argument("--hi", type=float, default=3.0)
    ap.add_argument("--no_finetune", action="store_true",
                    help="skip the slow live-DKL method")
    args = ap.parse_args()
    lo, hi = args.lo, args.hi
    seeds = list(range(args.seeds))
    methods = [m for m in METHODS if not (args.no_finetune and m == "dkl_finetune")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RUNS.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(CACHE_DIR / "master.parquet")
    pool = df[df.split == "pool"].reset_index(drop=True).copy()
    pool["target"] = pool["gap"].astype("float64").to_numpy()   # oracle = true gap (eV)
    t10, t50 = inwin_thresholds(pool, lo, hi)
    loaded = {"std_gp": features(CACHE_DIR / "descriptors.parquet", pool),
              "dkl_frozen": features(OUT / "embeddings_gap.parquet", pool)}
    needs_cache = ("dkl_finetune" in methods)
    cache = GraphCache(str(CACHE_DIR), config.GRAPH_PREPROC, readonly=True) if needs_cache else None
    n_inwin = int(pool.target.between(lo, hi).sum())
    logger.info(f"pool={len(pool)} in-window[{lo},{hi}]={n_inwin} "
                f"top10>={t10:.3f} top50>={t50:.3f} | methods={methods} "
                f"seeds={args.seeds} cycles={args.cycles} device={device}")

    base = {"acquisition": "window_max", "beta": 0.0, "xi": 0.01,
            "n_init": args.n_init, "n_cycles": args.cycles,
            "window_lo": lo, "window_hi": hi,
            "retrain_every_k": 5, "n_joint_epochs": 50,
            "n_pretrain_epochs": 20, "gp_refit_epochs": 50}

    n_done = n_skip = 0
    for method in methods:
        for seed in seeds:
            out = RUNS / f"{method}__seed{seed}.csv"
            if out.exists():
                n_skip += 1
                continue
            seed_everything(seed)
            cfg = OmegaConf.create({**base, "seed": seed})
            res = run_one(method, pool, loaded, cache, device, cfg)
            res.insert(0, "method", method)
            res.insert(1, "seed", seed)
            res = add_metrics(res, lo, hi, t10, t50)
            res.to_csv(out, index=False)
            n_done += 1
            logger.info(f"  {method}/seed{seed} → best_inwin={res.best_inwin_gap.iloc[-1]:.3f} "
                        f"top10={int(res.cumul_top10.iloc[-1])} "
                        f"top50={int(res.cumul_top50.iloc[-1])}")
    if cache is not None:
        cache.close()
    logger.info(f"runs written={n_done} skipped={n_skip}. Summarizing…")
    summarize_and_plot(lo, hi)


def load_all() -> pd.DataFrame:
    frames = [pd.read_csv(c) for c in sorted(RUNS.glob("*.csv"))]
    if not frames:
        raise FileNotFoundError("No winmax runs found.")
    return pd.concat(frames, ignore_index=True)


def mean_ci(piv: pd.DataFrame):
    """mean curve and 95% CI half-width across seeds for a [seed x cycle] pivot."""
    Y = piv.to_numpy(float)
    cyc = piv.columns.to_numpy()
    mean = Y.mean(0)
    n = max(Y.shape[0], 1)
    ci = 1.96 * Y.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return cyc, mean, ci


def crossover_cycle(cyc, dkl_mean, std_mean):
    """First cycle at which the DKL mean curve reaches/exceeds the Std-GP mean and
    stays >= it through the end. Returns the cycle (int) or None if it never does."""
    ge = dkl_mean >= std_mean
    for i in range(len(cyc)):
        if ge[i] and ge[i:].all():
            return int(cyc[i])
    return None


def summarize_and_plot(lo: float, hi: float) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    df = load_all()
    present = [m for m in METHODS if m in df.method.unique()]

    # ── Per (method, seed) finals for each metric ──
    finals = (df.sort_values("cycle").groupby(["method", "seed"]).tail(1)
              [["method", "seed", "best_inwin_gap", "cumul_top10", "cumul_top50"]])
    finals.to_csv(OUT / "winmax_summary.csv", index=False)

    # ── Paired Wilcoxon vs Std-GP, per metric ──
    rows = []
    for metric in METRICS:
        base = finals[finals.method == "std_gp"].set_index("seed")[metric]
        for m in [x for x in present if x != "std_gp"]:
            cur = finals[finals.method == m].set_index("seed")[metric]
            common = sorted(set(base.index) & set(cur.index))
            if len(common) < 2:
                continue
            x = cur.loc[common].to_numpy(float)
            y = base.loc[common].to_numpy(float)
            try:
                p = float(wilcoxon(x, y).pvalue) if not np.allclose(x, y) else float("nan")
            except ValueError:
                p = float("nan")
            rows.append({"metric": metric, "method": m, "n_seeds": len(common),
                         "method_mean": x.mean(), "std_gp_mean": y.mean(),
                         "mean_diff": (x - y).mean(), "wilcoxon_p": p,
                         "win_rate": float(np.mean(x > y))})
    pd.DataFrame(rows).to_csv(OUT / "winmax_stats.csv", index=False)

    # ── Mean curves per (method, metric) + crossover cycles ──
    curves = {}   # (method, metric) -> (cyc, mean, ci)
    for m in present:
        sub = df[df.method == m]
        for metric in METRICS:
            piv = sub.pivot_table(index="seed", columns="cycle", values=metric)
            curves[(m, metric)] = mean_ci(piv)

    cross_rows = []
    for metric in METRICS:
        if ("std_gp", metric) not in curves:
            continue
        cyc, std_mean, _ = curves[("std_gp", metric)]
        for m in DKL_METHODS:
            if (m, metric) not in curves:
                continue
            _, dkl_mean, _ = curves[(m, metric)]
            xc = crossover_cycle(cyc, dkl_mean, std_mean)
            cross_rows.append({"metric": metric, "dkl_method": m,
                               "crossover_cycle": xc,
                               "final_dkl": float(dkl_mean[-1]),
                               "final_std_gp": float(std_mean[-1])})
    cross = pd.DataFrame(cross_rows)
    cross.to_csv(OUT / "winmax_crossover.csv", index=False)

    # ── One curve figure per metric ──
    fignames = {"best_inwin_gap": "winmax_best_gap.png",
                "cumul_top50": "winmax_top50.png"}
    titles = {"best_inwin_gap": f"SNUMAT best in-window gap found ({lo}–{hi} eV) vs cycle",
              "cumul_top50": f"SNUMAT cumulative top-50 in-window found ({lo}–{hi} eV)"}
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for m in present:
            cyc, mean, ci = curves[(m, metric)]
            ax.plot(cyc, mean, color=COLORS[m], lw=2.2, label=LABELS[m])
            ax.fill_between(cyc, mean - ci, mean + ci, color=COLORS[m], alpha=0.13)
        if metric == "best_inwin_gap":
            ax.axhline(hi, color="#DC2626", ls="--", lw=1, alpha=0.7,
                       label=f"ceiling {hi} eV")
        ax.set_xlabel("BO cycle")
        ax.set_ylabel(METRICS[metric])
        ax.set_title(titles[metric], fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(PLOTS / fignames[metric], dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Crossover figure: one panel per metric, DKL vs Std-GP + crossover line ──
    fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 5.5))
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, METRICS):
        cyc, std_mean, std_ci = curves[("std_gp", metric)]
        ax.plot(cyc, std_mean, color=COLORS["std_gp"], lw=2.4, label=LABELS["std_gp"])
        ax.fill_between(cyc, std_mean - std_ci, std_mean + std_ci,
                        color=COLORS["std_gp"], alpha=0.12)
        for m in DKL_METHODS:
            if (m, metric) not in curves:
                continue
            cyc, dkl_mean, dkl_ci = curves[(m, metric)]
            ax.plot(cyc, dkl_mean, color=COLORS[m], lw=2.4, label=LABELS[m])
            ax.fill_between(cyc, dkl_mean - dkl_ci, dkl_mean + dkl_ci,
                            color=COLORS[m], alpha=0.12)
            xc = crossover_cycle(cyc, dkl_mean, std_mean)
            if xc is not None:
                ax.axvline(xc, color=COLORS[m], ls=":", lw=1.6, alpha=0.9)
                ax.annotate(f"{LABELS[m].split(' ')[0]} DKL\ncrosses @ {xc}",
                            xy=(xc, ax.get_ylim()[0]),
                            xytext=(xc + 2, std_mean.min()),
                            fontsize=8, color=COLORS[m])
        ax.set_xlabel("BO cycle")
        ax.set_ylabel(METRICS[metric])
        ax.set_title(f"Crossover — {metric}", fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"SNUMAT DKL vs Standard GP — crossover ({lo}–{hi} eV window-max search)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS / "winmax_crossover.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Console summary ──
    print("\n" + "=" * 64)
    print(f"SNUMAT WINDOW-MAX SEARCH ({lo}–{hi} eV) — finals (mean over seeds), "
          "★=p<0.05 vs Std-GP")
    print("=" * 64)
    stats = pd.read_csv(OUT / "winmax_stats.csv")
    for metric in METRICS:
        print(f"\n  {metric}")
        for m in present:
            mean = finals[finals.method == m][metric].mean()
            tag = ""
            r = stats[(stats.metric == metric) & (stats.method == m)]
            if not r.empty:
                p = r.wilcoxon_p.iloc[0]
                d = r.mean_diff.iloc[0]
                tag = f"  {'★' if p < 0.05 else ' '} p={p:.3f} ({'+' if d>0 else ''}{d:.2f} vs Std-GP)"
            print(f"    {LABELS[m]:24s} {mean:8.2f}{tag}")
    print("\n  Crossover cycle (DKL mean overtakes & stays above Std-GP):")
    for _, r in cross.iterrows():
        xc = "never" if pd.isna(r.crossover_cycle) else f"cycle {int(r.crossover_cycle)}"
        print(f"    {LABELS[r.dkl_method]:24s} on {r.metric:16s} → {xc}")
    print(f"\nSaved → winmax_summary.csv, winmax_stats.csv, winmax_crossover.csv, "
          f"plots/winmax_*.png")


if __name__ == "__main__":
    main()
