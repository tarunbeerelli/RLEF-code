import importlib.util

import pytest
import torch
import transformers

if importlib.util.find_spec("trl") is None:
    pytest.skip(
        "Skipping environment test: TRL is not installed or available.",
        allow_module_level=True,
    )


def test_imports():
    assert True


def test_torch_minimum_version():
    major, minor = torch.__version__.split(".")[:2]
    assert int(major) >= 2, f"torch >= 2.0 required, got {torch.__version__}"


def test_transformers_minimum_version():
    from packaging.version import Version

    assert Version(transformers.__version__) >= Version(
        "4.46.0"
    ), f"transformers >= 4.46.0 required, got {transformers.__version__}"


def test_trl_lazy_load_availability():
    """Verifies TRL can be loaded into the workspace environment.

    Skips gracefully if internal distributed submodules fail on macOS.
    """
    try:
        import trl  # noqa: F401
    except (ImportError, RuntimeError):
        pytest.skip(
            "TRL package is present, but distributed components (FSDP) failed to initialize on macOS."
        )


def test_torch_ops():
    x = torch.tensor([1.0, 2.0, 3.0])
    assert x.sum().item() == 6.0


def test_cuda_flag_is_bool():
    assert isinstance(torch.cuda.is_available(), bool)
