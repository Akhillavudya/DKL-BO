"""GP surrogates behind a shared ABC interface.

Two concrete implementations:
  ExactGPSurrogate  — Matérn-5/2, exact inference.  O(N³) — use when N < ~1000.
  SVGPSurrogate     — Matérn-5/2, inducing points.   O(N·M²) — cap memory for large N.

Both implement BaseSurrogate so the rest of the code never needs to know which
one is active.  Swap via config: model=gp_svgp (one CLI flag, no code edit).

GP inputs are float64 for numerical stability during matrix inversions.
The encoder outputs float32; DKLModel handles the conversion.
"""

from abc import ABC, abstractmethod
from typing import List

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)
from torch import Tensor


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class BaseSurrogate(ABC):
    """Shared interface for ExactGP and SVGP.

    fit(X, y)       → trains the GP on embedding vectors X and targets y
    predict(X)      → returns (mean, std) predictions on new embeddings
    training_loss() → loss value for joint DKL backprop (MLL or ELBO)
    """

    @abstractmethod
    def fit(self, X: Tensor, y: Tensor, n_epochs: int = 100) -> List[float]:
        """Train on (X, y). Returns list of per-epoch loss values."""
        ...

    @abstractmethod
    def predict(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Return (mean [N], std [N]) on input embeddings X."""
        ...

    @abstractmethod
    def to(self, device: torch.device) -> "BaseSurrogate":
        ...

    @abstractmethod
    def train_mode(self) -> None:
        ...

    @abstractmethod
    def eval_mode(self) -> None:
        ...


# ---------------------------------------------------------------------------
# Exact GP implementation
# ---------------------------------------------------------------------------


class _ExactGPModel(ExactGP):
    """The inner GPyTorch model — Matérn-5/2 kernel as in [P1]."""

    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood: GaussianLikelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(MaternKernel(nu=2.5))

    def forward(self, x: Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class ExactGPSurrogate(BaseSurrogate):
    """Exact GP surrogate — gold standard for N < ~1000 labelled points.

    After joint DKL training, use .fit() to update the GP on the full
    train embeddings.  Between BO cycles, only .fit() needs to run (cheap)
    when K > 1 (retrain_every_k from configs/bo/ucb.yaml).
    """

    def __init__(self, lr: float = 0.01, n_epochs: int = 100, patience: int = 20):
        self.lr = lr
        self.n_epochs = n_epochs
        self.patience = patience
        self.model: _ExactGPModel | None = None
        self.likelihood: GaussianLikelihood | None = None
        self._device = torch.device("cpu")

    # ------------------------------------------------------------------
    # BaseSurrogate interface
    # ------------------------------------------------------------------

    def fit(self, X: Tensor, y: Tensor, n_epochs: int | None = None) -> List[float]:
        """Fit (or refit) the GP on embedding vectors X and band-gap targets y."""
        n_epochs = n_epochs or self.n_epochs
        X = X.double().to(self._device)
        y = y.double().to(self._device)

        self.likelihood = GaussianLikelihood().double().to(self._device)
        self.model = _ExactGPModel(X, y, self.likelihood).double().to(self._device)

        self.model.train()
        self.likelihood.train()

        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.likelihood.parameters()),
            lr=self.lr,
        )

        losses, best, no_improve = [], float("inf"), 0
        for _ in range(n_epochs):
            optimizer.zero_grad()
            loss = -mll(self.model(X), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if loss.item() < best - 1e-4:
                best, no_improve = loss.item(), 0
            else:
                no_improve += 1
            if no_improve >= self.patience:
                break

        return losses

    def predict(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (mean, std) on new embeddings X."""
        assert self.model is not None, "Call fit() before predict()"
        self.model.eval()
        self.likelihood.eval()
        X = X.double().to(self._device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self.model(X))
        return preds.mean.float().cpu(), preds.stddev.float().cpu()

    def joint_loss(self, X: Tensor, y: Tensor) -> Tensor:
        """MLL loss for joint DKL backprop (embeddings still have gradients)."""
        assert self.model is not None
        X = X.double()
        y = y.double()
        self.model.set_train_data(inputs=X, targets=y, strict=False)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        return -mll(self.model(X), y)

    def to(self, device) -> "ExactGPSurrogate":
        self._device = torch.device(device)
        if self.model is not None:
            self.model = self.model.to(self._device)
            self.likelihood = self.likelihood.to(self._device)
        return self

    def train_mode(self) -> None:
        if self.model is not None:
            self.model.train()
            self.likelihood.train()

    def eval_mode(self) -> None:
        if self.model is not None:
            self.model.eval()
            self.likelihood.eval()


# ---------------------------------------------------------------------------
# SVGP implementation (scaling fix from [P2])
# ---------------------------------------------------------------------------


class _SVGPModel(ApproximateGP):
    """Sparse Variational GP — O(N·M²) memory cap regardless of training set size."""

    def __init__(self, inducing_points: Tensor):
        vd = CholeskyVariationalDistribution(inducing_points.size(0))
        vs = VariationalStrategy(self, inducing_points, vd, learn_inducing_locations=True)
        super().__init__(vs)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(MaternKernel(nu=2.5))

    def forward(self, x: Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class SVGPSurrogate(BaseSurrogate):
    """SVGP surrogate — drop-in replacement for ExactGPSurrogate.

    Uses M inducing points (default 128) to approximate the full GP.
    Memory cost is O(N·M²) not O(N³), so it scales to any N.
    Trained with ELBO (variational lower bound) instead of exact MLL.

    From [P2]: this is the key architectural fix for the OOM cliff.
    """

    def __init__(self, n_inducing: int = 128, lr: float = 0.01, n_epochs: int = 200):
        self.n_inducing = n_inducing
        self.lr = lr
        self.n_epochs = n_epochs
        self.model: _SVGPModel | None = None
        self.likelihood: GaussianLikelihood | None = None
        self._device = torch.device("cpu")

    def fit(self, X: Tensor, y: Tensor, n_epochs: int | None = None) -> List[float]:
        n_epochs = n_epochs or self.n_epochs
        X = X.double().to(self._device)
        y = y.double().to(self._device)

        # Initialise inducing points as a random subset of training data
        idx = torch.randperm(X.size(0))[: min(self.n_inducing, X.size(0))]
        inducing_points = X[idx].clone().detach()

        self.likelihood = GaussianLikelihood().double().to(self._device)
        self.model = _SVGPModel(inducing_points).double().to(self._device)

        self.model.train()
        self.likelihood.train()

        elbo = VariationalELBO(self.likelihood, self.model, num_data=X.size(0))
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.likelihood.parameters()),
            lr=self.lr,
        )

        losses = []
        for _ in range(n_epochs):
            optimizer.zero_grad()
            loss = -elbo(self.model(X), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return losses

    def predict(self, X: Tensor) -> tuple[Tensor, Tensor]:
        assert self.model is not None, "Call fit() before predict()"
        self.model.eval()
        self.likelihood.eval()
        X = X.double().to(self._device)
        with torch.no_grad():
            preds = self.likelihood(self.model(X))
        return preds.mean.float().cpu(), preds.stddev.float().cpu()

    def joint_loss(self, X: Tensor, y: Tensor) -> Tensor:
        assert self.model is not None
        X_d, y_d = X.double(), y.double()
        elbo = VariationalELBO(self.likelihood, self.model, num_data=X_d.size(0))
        return -elbo(self.model(X_d), y_d)

    def to(self, device) -> "SVGPSurrogate":
        self._device = torch.device(device)
        if self.model is not None:
            self.model = self.model.to(self._device)
            self.likelihood = self.likelihood.to(self._device)
        return self

    def train_mode(self) -> None:
        if self.model is not None:
            self.model.train()
            self.likelihood.train()

    def eval_mode(self) -> None:
        if self.model is not None:
            self.model.eval()
            self.likelihood.eval()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_surrogate(cfg) -> BaseSurrogate:
    """Instantiate the correct surrogate from a Hydra config node."""
    kind = getattr(cfg, "surrogate_type", "exact")
    if kind == "svgp":
        return SVGPSurrogate(
            n_inducing=cfg.n_inducing,
            lr=cfg.lr,
            n_epochs=cfg.n_train_epochs,
        )
    return ExactGPSurrogate(
        lr=cfg.lr,
        n_epochs=cfg.n_train_epochs,
        patience=getattr(cfg, "patience", 20),
    )
