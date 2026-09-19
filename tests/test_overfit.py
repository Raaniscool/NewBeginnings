"""THE OVERFIT TEST.

Project rule: before trusting any benchmark, the full pipeline
(tokenizer -> dataset -> model -> loss -> optimizer step) must prove it can
learn SOMETHING. The cheapest proof: a tiny model overfitted to a handful of
repeated sentences must (a) drive loss far below its starting point and
(b) reproduce a memorized continuation exactly under greedy decoding.

If this test ever fails, the pipeline is broken; no evaluation number from
it means anything.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from core.config import ModelConfig, TrainConfig
from core.dataset import Document, TokenDataset, tokenize_documents
from core.engine import Trainer, set_seed
from core.generate import generate_texts
from core.config import DecodeConfig
from core.model import GPT
from core.tokenizer import train_tokenizer

# A small, distinctive world: 8 sentences with surface variety.
WORLD = [
    "The brass key had been inside the green teapot the whole time.",
    "Salt on the doorstep keeps the snails out of the lettuce bed.",
    "Grandmother knitted the waves into every blue sweater she made.",
    "The night train whistled twice before it crossed the iron bridge.",
    "A fox will not bargain with a fox, muttered the old fur trader.",
    "Warm bread makes polite enemies of hungry neighbors.",
    "The lighthouse keeper counted ships the way children count stars.",
    "Under the third tile of the kitchen floor, the coins waited patiently.",
]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("overfit")
    corpus = tmp / "train.txt"
    corpus.write_text("\n".join(WORLD) * 500 + "\n")
    tok = train_tokenizer([corpus], vocab_size=260, min_frequency=2,
                          out_dir=tmp / "tk")
    flat = tokenize_documents(
        [Document("w", "stories", " ".join(WORLD) * 200, "g", "generated")], tok)
    for name in ("train", "val", "test"):
        np.save(tmp / f"{name}.npy", flat)
    mcfg = ModelConfig(vocab_size=tok.vocab_size, block_size=40, n_layer=2,
                       n_head=4, d_model=64, dropout=0.0, bias=False)
    tcfg = TrainConfig(
        version="overfit", seed=11, batch_size=16, block_size=40, epochs=1,
        max_steps=180, learning_rate=4e-3, min_lr_frac=0.1, warmup_steps=10,
        weight_decay=0.0, grad_clip=1.0, eval_interval=60, eval_iters=3,
        log_interval=60, num_threads=2, ckpt_dir=str(tmp / "ckpt"),
        data_dir=str(tmp), run_dir=str(tmp / "run"), model_config_path="")
    set_seed(11)
    model = GPT(mcfg)
    import math as _m
    n_params = model.num_params()
    from core.dataset import TokenDataset as _TD
    data = _TD(tmp)
    tr = Trainer(model, tcfg, data)
    # capture the initial loss deliberately (before any update)
    x, y = data.get_batch("train", 16, 40)
    model.eval()
    with torch.no_grad():
        _, init_loss = model(x, y)
    model.train()
    summary = tr.train()
    return {"tok": tok, "model": model, "init_loss": float(init_loss),
            "summary": summary, "params": n_params, "tmp": tmp, "mcfg": mcfg}


def test_tiny_setup_is_sane(world):
    assert world["params"] < 500_000           # genuinely tiny
    # random-init loss should be near ln(vocab)
    assert 0.8 * math.log(world["model"].cfg.vocab_size) < world["init_loss"] \
        < 1.35 * math.log(world["model"].cfg.vocab_size)


def test_loss_crashes_far_below_random(world):
    final_ppl = world["summary"]["final_val_ppl"]
    init_ppl = math.exp(world["init_loss"])
    # overfit proof: we must beat a 4x perplexity reduction on the tiny world
    assert final_ppl < init_ppl / 4.0, (
        f"pipeline failed to learn: ppl {init_ppl:.1f} -> {final_ppl:.1f}")
    assert final_ppl < math.e ** 2.5, f"still ~guessing: ppl {final_ppl:.1f}"


def test_model_memorized_a_continuation(world):
    tok, model = world["tok"], world["model"]
    model.eval()
    greedy = DecodeConfig(max_new_tokens=24, temperature=0.0, top_k=0,
                          top_p=1.0, repetition_penalty=1.0, seed=1,
                          name="memorize_check")
    prompt = "The brass key had been"
    out = generate_texts(model, tok, [prompt], greedy)[0]
    assert "inside the green teapot" in out, (
        f"model did not memorize its training world: {out!r}")


def test_training_produced_checkpoints_and_log(world):
    tmp = world["tmp"]
    assert (tmp / "ckpt" / "best.ckpt").exists()
    log = (tmp / "run" / "log.jsonl").read_text().splitlines()
    assert log, "training log is empty"
    assert world["summary"]["steps"] == 180
