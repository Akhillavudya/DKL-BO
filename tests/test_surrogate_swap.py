"""Test that ExactGPSurrogate and SVGPSurrogate obey the same interface.

This is the 'surrogate swap' test from the plan (Section 4, tests/).
It verifies that switching between exact GP and SVGP is truly a one-line
config change — both produce (mean, std) of the correct shape and type.

If this test fails after any refactor, the DKL loop and BO loop will break
when you try to swap surrogates.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dklbo.models.surrogate import ExactGPSurrogate, SVGPSurrogate, BaseSurrogate

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

N_TRAIN = 50
N_TEST  = 20
D_EMB   = 32   # hidden_dim from cgcnn.yaml


@pytest.fixture
def synthetic_data():
    torch.manual_seed(0)
    X_train = torch.randn(N_TRAIN, D_EMB)
    y_train = torch.randn(N_TRAIN)
    X_test  = torch.randn(N_TEST, D_EMB)
    return X_train, y_train, X_test


@pytest.fixture
def exact_surrogate():
    return ExactGPSurrogate(lr=0.01, n_epochs=5, patience=10)


@pytest.fixture
def svgp_surrogate():
    return SVGPSurrogate(n_inducing=10, lr=0.01, n_epochs=5)


# ---------------------------------------------------------------------------
# Interface conformance tests (run for BOTH surrogates)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("surrogate_cls,kwargs", [
    (ExactGPSurrogate, {"lr": 0.01, "n_epochs": 5, "patience": 10}),
    (SVGPSurrogate,    {"n_inducing": 10, "lr": 0.01, "n_epochs": 5}),
])
def test_is_base_surrogate(surrogate_cls, kwargs):
    """Both surrogates must be instances of BaseSurrogate."""
    s = surrogate_cls(**kwargs)
    assert isinstance(s, BaseSurrogate), f"{surrogate_cls.__name__} is not a BaseSurrogate"


@pytest.mark.parametrize("surrogate_cls,kwargs", [
    (ExactGPSurrogate, {"lr": 0.01, "n_epochs": 5, "patience": 10}),
    (SVGPSurrogate,    {"n_inducing": 10, "lr": 0.01, "n_epochs": 5}),
])
def test_fit_returns_loss_list(surrogate_cls, kwargs, synthetic_data):
    """fit() must return a non-empty list of float losses."""
    X_train, y_train, _ = synthetic_data
    s = surrogate_cls(**kwargs)
    losses = s.fit(X_train, y_train)
    assert isinstance(losses, list), "fit() must return a list"
    assert len(losses) > 0, "fit() returned empty loss list"
    assert all(isinstance(v, float) for v in losses), "All losses must be float"


@pytest.mark.parametrize("surrogate_cls,kwargs", [
    (ExactGPSurrogate, {"lr": 0.01, "n_epochs": 5, "patience": 10}),
    (SVGPSurrogate,    {"n_inducing": 10, "lr": 0.01, "n_epochs": 5}),
])
def test_predict_output_shapes(surrogate_cls, kwargs, synthetic_data):
    """predict() must return (mean [N], std [N]) with correct shapes."""
    X_train, y_train, X_test = synthetic_data
    s = surrogate_cls(**kwargs)
    s.fit(X_train, y_train)
    mean, std = s.predict(X_test)

    assert mean.shape == (N_TEST,), f"mean shape {mean.shape} != ({N_TEST},)"
    assert std.shape  == (N_TEST,), f"std shape {std.shape} != ({N_TEST},)"


@pytest.mark.parametrize("surrogate_cls,kwargs", [
    (ExactGPSurrogate, {"lr": 0.01, "n_epochs": 5, "patience": 10}),
    (SVGPSurrogate,    {"n_inducing": 10, "lr": 0.01, "n_epochs": 5}),
])
def test_predict_std_is_positive(surrogate_cls, kwargs, synthetic_data):
    """GP uncertainty (std) must always be positive."""
    X_train, y_train, X_test = synthetic_data
    s = surrogate_cls(**kwargs)
    s.fit(X_train, y_train)
    _, std = s.predict(X_test)
    assert (std > 0).all(), "std contains non-positive values"


@pytest.mark.parametrize("surrogate_cls,kwargs", [
    (ExactGPSurrogate, {"lr": 0.01, "n_epochs": 5, "patience": 10}),
    (SVGPSurrogate,    {"n_inducing": 10, "lr": 0.01, "n_epochs": 5}),
])
def test_predict_output_is_cpu_float32(surrogate_cls, kwargs, synthetic_data):
    """predict() must return CPU float32 tensors regardless of training device."""
    X_train, y_train, X_test = synthetic_data
    s = surrogate_cls(**kwargs)
    s.fit(X_train, y_train)
    mean, std = s.predict(X_test)

    assert mean.device.type == "cpu", "mean must be on CPU"
    assert std.device.type  == "cpu", "std must be on CPU"
    assert mean.dtype == torch.float32, f"mean dtype {mean.dtype} != float32"
    assert std.dtype  == torch.float32, f"std dtype {std.dtype} != float32"


@pytest.mark.parametrize("surrogate_cls,kwargs", [
    (ExactGPSurrogate, {"lr": 0.01, "n_epochs": 5, "patience": 10}),
    (SVGPSurrogate,    {"n_inducing": 10, "lr": 0.01, "n_epochs": 5}),
])
def test_to_device_runs_without_error(surrogate_cls, kwargs, synthetic_data):
    """to(device) must not raise before or after fitting."""
    X_train, y_train, X_test = synthetic_data
    s = surrogate_cls(**kwargs)
    s.to("cpu")         # before fit — should not crash
    s.fit(X_train, y_train)
    s.to("cpu")         # after fit — should move model
    mean, std = s.predict(X_test)
    assert mean.shape == (N_TEST,)


@pytest.mark.parametrize("surrogate_cls,kwargs", [
    (ExactGPSurrogate, {"lr": 0.01, "n_epochs": 5, "patience": 10}),
    (SVGPSurrogate,    {"n_inducing": 10, "lr": 0.01, "n_epochs": 5}),
])
def test_joint_loss_is_scalar(surrogate_cls, kwargs, synthetic_data):
    """joint_loss() must return a scalar Tensor (for DKL backprop)."""
    X_train, y_train, _ = synthetic_data
    s = surrogate_cls(**kwargs)
    s.fit(X_train, y_train)
    X_with_grad = X_train.clone().requires_grad_(True)
    loss = s.joint_loss(X_with_grad, y_train)
    assert loss.shape == torch.Size([]), f"joint_loss shape {loss.shape} — must be scalar"
    assert loss.requires_grad, "joint_loss must have gradient"
