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


def greedy(mean: Tensor, std: Tensor, **_) -> Tensor:
    """Pure exploitation: ignore uncertainty, always pick highest mean."""
    return mean.clone()


def random_acquisition(mean: Tensor, std: Tensor, **_) -> Tensor:
    """Pure random: ignore both mean and std (exploration baseline)."""
    return torch.rand_like(mean)


_REGISTRY = {
    "ucb":    ucb,
    "ei":     ei,
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
