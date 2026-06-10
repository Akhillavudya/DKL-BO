"""Acquisition functions for Bayesian Optimisation.

Each function takes (mean [N], std [N]) tensors and returns a score [N].
The material with the HIGHEST score is selected as the next candidate.

UCB  = mean + β·std   (Upper Confidence Bound)
  β=0   → pure exploitation — always pick the highest predicted mean
  β=0.2 → our default from [P1] (professor's recommendation)
  β→∞  → pure exploration  — always pick the most uncertain material

Why UCB works for material discovery:
  The model might be confidently wrong about some material.
  β>0 forces the loop to revisit uncertain regions so those errors get corrected.
  Without uncertainty, the loop gets stuck exploiting a local maximum.
"""

import torch
from torch import Tensor


def ucb(mean: Tensor, std: Tensor, beta: float = 0.2) -> Tensor:
    """Upper Confidence Bound: mean + β·std."""
    return mean + beta * std


def greedy(mean: Tensor, std: Tensor, **_) -> Tensor:
    """Pure exploitation: ignore uncertainty, always pick highest mean."""
    return mean.clone()


def random_acquisition(mean: Tensor, std: Tensor, **_) -> Tensor:
    """Pure random: ignore both mean and std (exploration baseline)."""
    return torch.rand_like(mean)


_REGISTRY = {
    "ucb":    ucb,
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
