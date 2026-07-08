"""Acquisition functions for Bayesian Optimisation.

Each function takes (mean [N], std [N]) tensors and returns a score [N].
The material with the HIGHEST score is selected as the next candidate.

All acquisitions operate in a **maximize** convention: the BO loop always picks
argmax(score). Minimization tasks are handled upstream by the loop, which feeds
the GP a sign-flipped target (y_internal = -y_true); see `bo/loop.py`. So an
acquisition function never needs to know the optimisation direction.

To keep the loop's call site uniform, every function accepts the full keyword set
(`beta`, `best_f`, `xi`) and ignores the ones it does not use via `**_`.

UCB  = mean + β·std   (Upper Confidence Bound)
  β=0   → pure exploitation — always pick the highest predicted mean
  β=0.2 → our default from [P1] (professor's recommendation)
  β→∞  → pure exploration  — always pick the most uncertain material

EI   = Expected Improvement over the current best label (incumbent `best_f`).
  Standard analytic form. Used for the GP-BO vs DKL-BO benchmark so both methods
  share an identical, parameter-free acquisition.
"""

import torch
from torch import Tensor
from torch.distributions import Normal


def ucb(mean: Tensor, std: Tensor, beta: float = 0.2, **_) -> Tensor:
    """Upper Confidence Bound: mean + β·std."""
    return mean + beta * std


def ei(mean: Tensor, std: Tensor, best_f: float = 0.0, xi: float = 0.01, **_) -> Tensor:
    """Expected Improvement (maximize convention).

    EI(x) = (μ − f* − ξ)·Φ(z) + σ·φ(z),  with  z = (μ − f* − ξ) / σ
    where f* is the best labelled value so far (the incumbent), ξ a small jitter
    that discourages over-exploitation, Φ/φ the standard-normal CDF/PDF.
    EI is exactly 0 where σ = 0 (no information to gain).
    """
    std = std.clamp_min(0.0)
    safe_std = std.clamp_min(1e-9)
    improvement = mean - best_f - xi
    z = improvement / safe_std
    normal = Normal(torch.zeros_like(z), torch.ones_like(z))
    score = improvement * normal.cdf(z) + safe_std * torch.exp(normal.log_prob(z))
    # Where there is no uncertainty, EI collapses to 0 (cannot improve).
    return torch.where(std > 1e-9, score, torch.zeros_like(score))


def window(mean: Tensor, std: Tensor, window_lo: float = None,
           window_hi: float = None, **_) -> Tensor:
    """Target-window probability (maximize convention).

    Score(x) = P(window_lo ≤ y ≤ window_hi)  under the GP posterior N(μ, σ²)
             = Φ((window_hi − μ)/σ) − Φ((window_lo − μ)/σ)

    Picks the material most likely to land INSIDE the band. This is the standard
    acquisition for level-set / target-region search (vs EI's chase of extremes).
    Predictions must be in the SAME units as window_lo/hi — true eV here, which
    holds because direction='max' (sign=+1) feeds the GP the raw band gap.
    """
    if window_lo is None or window_hi is None:
        raise ValueError("window acquisition needs cfg.window_lo and cfg.window_hi")
    safe_std = std.clamp_min(1e-9)
    n = Normal(torch.zeros_like(mean), torch.ones_like(mean))
    return n.cdf((window_hi - mean) / safe_std) - n.cdf((window_lo - mean) / safe_std)


def window_max(mean: Tensor, std: Tensor, window_lo: float = None,
               window_hi: float = None, beta: float = 0.0, **_) -> Tensor:
    """Expected in-window gap — constrained max (maximize convention).

    Objective: maximise the band gap subject to a HARD ceiling, window_lo ≤ y ≤
    window_hi. A material above window_hi is rejected, so the best attainable
    target is the highest gap still inside the band (e.g. ≈2.997 eV for 0.7–3.0).

    Score(x) = E[ y · 1{window_lo ≤ y ≤ window_hi} ]   under the GP posterior N(μ, σ²)
             = μ·(Φ(βʰ) − Φ(αˡ)) − σ·(φ(βʰ) − φ(αˡ))
      with  αˡ = (window_lo − μ)/σ ,  βʰ = (window_hi − μ)/σ .

    This is the posterior *partial expectation* over the band: it rewards a high
    predicted gap that stays inside the window and discounts the probability mass
    that spills past the ceiling (so a confident 2.5 eV can beat an uncertain 2.99
    whose tail leaks above 3.0). As σ → 0 it collapses to argmax of the true
    in-window gap, i.e. it targets the genuine constrained maximum. An optional
    UCB-style bonus β·σ·P(in window) adds exploration; default β=0 (pure exploit).

    Predictions must be in the SAME units as window_lo/hi — true eV — which holds
    because direction='max' (sign=+1) feeds the GP the raw band gap.
    """
    if window_lo is None or window_hi is None:
        raise ValueError("window_max acquisition needs cfg.window_lo and cfg.window_hi")
    safe_std = std.clamp_min(1e-9)
    n = Normal(torch.zeros_like(mean), torch.ones_like(mean))
    al = (window_lo - mean) / safe_std
    bh = (window_hi - mean) / safe_std
    p_in = n.cdf(bh) - n.cdf(al)
    pdf_diff = torch.exp(n.log_prob(bh)) - torch.exp(n.log_prob(al))
    partial_exp = mean * p_in - safe_std * pdf_diff
    return partial_exp + beta * safe_std * p_in


def window_min(mean: Tensor, std: Tensor, window_lo: float = None,
               window_hi: float = None, beta: float = 0.0, **_) -> Tensor:
    """Expected in-window headroom — constrained min (maximize convention).

    Mirror of `window_max`. Objective: MINIMISE the band gap subject to a HARD
    floor, window_lo ≤ y ≤ window_hi. A material below window_lo is rejected, so
    the best attainable target is the LOWEST gap still inside the band (e.g.
    ≈0.7 eV for 0.7–3.0, i.e. closest to the floor from above).

    We reward the headroom below the ceiling, r(y) = window_hi − y, for in-window
    materials — maximal at the smallest in-window gap:

    Score(x) = E[ (window_hi − y) · 1{window_lo ≤ y ≤ window_hi} ]
             = (window_hi − μ)·(Φ(βʰ) − Φ(αˡ)) + σ·(φ(βʰ) − φ(αˡ))
      with  αˡ = (window_lo − μ)/σ ,  βʰ = (window_hi − μ)/σ .

    As σ → 0 it collapses to argmax of (window_hi − y) over in-window materials,
    i.e. the genuine constrained minimum. An optional UCB-style bonus
    β·σ·P(in window) adds exploration; default β=0 (pure exploit).

    Predictions must be in raw eV (same units as window_lo/hi) — which holds when
    the loop runs with direction='max' (sign=+1), so the GP sees the raw gap.
    """
    if window_lo is None or window_hi is None:
        raise ValueError("window_min acquisition needs cfg.window_lo and cfg.window_hi")
    safe_std = std.clamp_min(1e-9)
    n = Normal(torch.zeros_like(mean), torch.ones_like(mean))
    al = (window_lo - mean) / safe_std
    bh = (window_hi - mean) / safe_std
    p_in = n.cdf(bh) - n.cdf(al)
    pdf_diff = torch.exp(n.log_prob(bh)) - torch.exp(n.log_prob(al))
    partial_exp = (window_hi - mean) * p_in + safe_std * pdf_diff
    return partial_exp + beta * safe_std * p_in


def greedy(mean: Tensor, std: Tensor, **_) -> Tensor:
    """Pure exploitation: ignore uncertainty, always pick highest mean."""
    return mean.clone()


def random_acquisition(mean: Tensor, std: Tensor, **_) -> Tensor:
    """Pure random: ignore both mean and std (exploration baseline)."""
    return torch.rand_like(mean)


_REGISTRY = {
    "ucb":    ucb,
    "ei":     ei,
    "window": window,
    "window_max": window_max,
    "window_min": window_min,
    "greedy": greedy,
    "random": random_acquisition,
}


def get_acquisition(name: str):
    """Return acquisition function by name. Raises ValueError if unknown."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown acquisition '{name}'. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[name]
