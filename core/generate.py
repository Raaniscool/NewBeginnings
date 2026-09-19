"""Text generation: checkpoint loading + prompt-conditioned sampling."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import torch

from core.config import DecodeConfig, ModelConfig
from core.model import GPT
from core.tokenizer import BPETokenizer


def load_model(ckpt_path: str | Path, device: str = "cpu") -> GPT:
    payload = torch.load(Path(ckpt_path), map_location=device, weights_only=False)
    cfg = ModelConfig(**payload["model_config"])
    model = GPT(cfg)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


@torch.no_grad()
def generate_texts(
    model: GPT,
    tok: BPETokenizer,
    prompts: Sequence[str],
    decode: DecodeConfig,
    device: str = "cpu",
) -> List[str]:
    """Generate a continuation for each prompt.

    Decoding parameters come ONLY from the pinned DecodeConfig: gate metrics
    must reflect raw model knowledge, so repetition penalty / top-p stay at
    their neutral values for gate-relevant runs (is_raw_model() check).
    """
    torch.manual_seed(decode.seed)
    outs: List[str] = []
    for prompt in prompts:
        ids = tok.encode(prompt)
        if len(ids) > model.cfg.block_size - 1:
            ids = ids[-(model.cfg.block_size - 1):]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        y = model.generate(
            x,
            max_new_tokens=decode.max_new_tokens,
            temperature=decode.temperature,
            top_k=decode.top_k,
            top_p=decode.top_p,
            repetition_penalty=decode.repetition_penalty,
        )
        outs.append(tok.decode(y[0, len(ids):].tolist()))
    return outs
