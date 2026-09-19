"""Shared pytest configuration.

Imports resolve against the repo root, and CPU threading is pinned small so
tests are deterministic and do not fight the training process for cores.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _pin_threads():
    import torch
    torch.set_num_threads(2)
    yield
