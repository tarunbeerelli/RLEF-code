import torch
import transformers


def test_imports():
    assert True


def test_torch_minimum_version():
    major, minor = torch.__version__.split(".")[:2]
    assert int(major) >= 2, f"torch >= 2.0 required, got {torch.__version__}"


def test_transformers_minimum_version():
    from packaging.version import Version

    assert Version(transformers.__version__) >= Version(
        "4.47.0"
    ), f"transformers >= 4.47.0 required, got {transformers.__version__}"


def test_trl_has_grpo():
    from trl import GRPOTrainer

    assert GRPOTrainer is not None


def test_torch_ops():
    x = torch.tensor([1.0, 2.0, 3.0])
    assert x.sum().item() == 6.0


def test_cuda_flag_is_bool():
    assert isinstance(torch.cuda.is_available(), bool)
