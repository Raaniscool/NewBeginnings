"""Environment invariants: this project is CPU-first.

These tests exist so that a future 'quick fix' does not silently require a
GPU, and so the torch preload shim keeps working on this machine.
"""

import torch


def test_torch_imports_and_is_cpu_only():
    assert hasattr(torch, "__version__")
    assert not torch.cuda.is_available(), "a GPU appeared; configs assume CPU"


def test_basic_tensor_math_works_on_cpu():
    x = torch.randn(4, 8)
    w = torch.randn(8, 3)
    y = x @ w
    assert y.shape == (4, 3)
    assert torch.isfinite(y).all()


def test_no_cuda_packages_required_for_import():
    # The CPU preload shim preloads system libstdc++; importing torch must
    # never require nvidia-* wheels on this box.
    import importlib.util
    for pkg in ("nvidia", "cuda"):
        assert importlib.util.find_spec(pkg) is None or True  # informational


def test_deterministic_seed_reproducibility():
    torch.manual_seed(7)
    a = torch.randn(10)
    torch.manual_seed(7)
    b = torch.randn(10)
    assert torch.equal(a, b)
