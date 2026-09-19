"""Configuration handling.

Everything in this project is configuration-driven. Configs are small JSON
files committed to git; behavior must be reproducible from config + code +
seeds alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict


# ---------------------------------------------------------------- config --

@dataclass
class ModelConfig:
    vocab_size: int = 8192            # overwritten with actual tokenizer size
    block_size: int = 192             # tokens of context
    n_layer: int = 4
    n_head: int = 8
    d_model: int = 256
    dropout: float = 0.1
    bias: bool = False                # biases in Linear layers

    def n_params_estimate(self) -> int:
        d, L, V, T = self.d_model, self.n_layer, self.vocab_size, self.block_size
        per_layer = 12 * d * d + (13 * d if self.bias else 0)
        layernorms = (2 * L + 1) * 2 * d
        embeddings = V * d + T * d     # lm_head is weight-tied to wte
        return L * per_layer + layernorms + embeddings


@dataclass
class TrainConfig:
    version: str = "v1"
    seed: int = 1234
    batch_size: int = 32
    block_size: int = 192             # must match model config
    epochs: int = 12
    max_steps: int = 0                # 0 = derive from epochs
    learning_rate: float = 3e-4
    min_lr_frac: float = 0.1
    warmup_steps: int = 80
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    dropout: float = 0.1              # informational; model holds the real one
    eval_interval: int = 200          # steps between validation evals
    eval_iters: int = 25              # batches averaged per eval
    log_interval: int = 25
    num_threads: int = 2
    ckpt_dir: str = "checkpoints/v1"
    data_dir: str = "data/tokenized"
    tokenizer_path: str = "artifacts/tokenizer/tokenizer.json"
    run_dir: str = "artifacts/runs/v1"
    model_config_path: str = "configs/v1_model.json"
    keep_best: bool = True


@dataclass
class TokenizerConfig:
    vocab_size: int = 8192
    min_frequency: int = 3
    special_tokens: tuple = ("<|bos|>", "<|eos|>", "<|pad|>")
    train_files: tuple = ()           # set by run.py: train split only


@dataclass
class DecodeConfig:
    """Pinned decoding for gate-relevant benchmark metrics."""
    name: str = "gate_pinned"
    temperature: float = 0.8
    top_k: int = 0                    # 0 = disabled
    top_p: float = 1.0                # 1.0 = disabled
    repetition_penalty: float = 1.0   # 1.0 = disabled (model knowledge only)
    max_new_tokens: int = 96
    seed: int = 99

    def is_raw_model(self) -> bool:
        """Gate rule: model-knowledge metrics may not use decoding crutches."""
        return self.repetition_penalty == 1.0 and self.top_p == 1.0 and self.top_k == 0


# ------------------------------------------------------------- io helpers --

def _coerce(cls, raw: Dict[str, Any]):
    names = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**raw)


def save_config(cfg, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)
        f.write("\n")


def load_config(cls, path: str | Path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return _coerce(cls, raw)
