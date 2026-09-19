"""Training-engine tests: checkpointing, resume, retention policy, LR
schedule, storage discipline. Heavy learning tests live in test_overfit.py.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from core.config import ModelConfig, TrainConfig
from core.dataset import Document, TokenDataset, tokenize_documents
from core.engine import Trainer, set_seed
from core.model import GPT
from core.tokenizer import train_tokenizer

SENTENCES = [
    "the cat sat on the warm mat by the fire and purred softly",
    "a young shepherd counted every sheep in the grey morning mist",
    "the baker opened his little shop before the village was awake",
    "cold rain drummed gently on the roof of the stone cottage",
    "the old lighthouse keeper climbed the stairs at dusk each day",
    "children laughed loudly in the garden behind the red school",
]


@pytest.fixture(scope="module")
def tiny_setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("engine")
    corpus = tmp / "train.txt"
    corpus.write_text(".\n".join(SENTENCES) * 600 + ".\n")
    tok = train_tokenizer([corpus], vocab_size=220, min_frequency=2,
                          out_dir=tmp / "tk")
    docs = [Document(f"d{i}", "stories", s + ".", "g", "generated")
            for i, s in enumerate(SENTENCES)]
    big = tokenize_documents(
        [Document("w", "stories", ". ".join(SENTENCES * 40) + ".", "g",
                  "generated")], tok)
    for name in ("train", "val", "test"):
        np.save(tmp / f"{name}.npy", big)
    tok.save(tmp / "tk")
    return tmp, tok


def make_trainer(tmp, max_steps, tok, seed=1234):
    tcfg = TrainConfig(
        version="test", seed=seed, batch_size=8, block_size=24, epochs=1,
        max_steps=max_steps, learning_rate=3e-3, min_lr_frac=0.1,
        warmup_steps=3, weight_decay=0.0, grad_clip=1.0, eval_interval=6,
        eval_iters=2, log_interval=5, num_threads=2,
        ckpt_dir=str(tmp / "ckpt"), data_dir=str(tmp),
        run_dir=str(tmp / "run"), model_config_path="",
    )
    mcfg = ModelConfig(vocab_size=tok.vocab_size, block_size=24, n_layer=2, n_head=4,
                       d_model=48, dropout=0.0, bias=False)
    set_seed(seed)
    model = GPT(mcfg)
    data = TokenDataset(tmp)
    return Trainer(model, tcfg, data), tcfg


def test_checkpoint_contains_everything_and_retention(tiny_setup, tmp_path):
    tmp, tok = tiny_setup
    work = tmp_path / "w1"
    work.mkdir()
    # reuse module-scope token files by pointing data_dir at tmp dir
    trainer, tcfg = make_trainer(tmp, max_steps=6, tok=tok)
    trainer.cfg.ckpt_dir = str(tmp_path / "ck")
    trainer.cfg.run_dir = str(tmp_path / "rn")
    Path(trainer.cfg.ckpt_dir).mkdir()
    Path(trainer.cfg.run_dir).mkdir()
    trainer.log_path = Path(trainer.cfg.run_dir) / "log.jsonl"
    summary = trainer.train()
    ck = Path(trainer.cfg.ckpt_dir)
    # retention: ONLY best + last (epoch may save both); no accumulating junk
    names = sorted(p.name for p in ck.glob("*.ckpt"))
    assert names == ["best.ckpt", "last.ckpt"], names
    payload = torch.load(ck / "last.ckpt", map_location="cpu", weights_only=False)
    for key in ("model_state", "opt_state", "step", "best_val", "train_config",
                "model_config", "rng"):
        assert key in payload, f"missing {key} in checkpoint"
    assert payload["step"] > 0
    assert summary["interrupted"] is False


def test_resume_continues_from_saved_step(tiny_setup, tmp_path):
    tmp, t = tiny_setup
    t1, _ = make_trainer(tmp, max_steps=8, tok=t)
    t1.cfg.ckpt_dir = str(tmp_path / "ckr"); Path(t1.cfg.ckpt_dir).mkdir()
    t1.cfg.run_dir = str(tmp_path / "rnr"); Path(t1.cfg.run_dir).mkdir()
    t1.log_path = Path(t1.cfg.run_dir) / "log.jsonl"
    t1.train()
    step_after_first = t1.step
    # new trainer continues the run
    t2, _ = make_trainer(tmp, max_steps=16, tok=t)
    t2.load(Path(t1.cfg.ckpt_dir) / "last.ckpt")
    assert t2.step == step_after_first
    assert t2.max_steps == 16  # a resumed run may have a new target
    assert t2.best_val == t1.best_val


def test_determinism_same_seed_same_loss_curve(tiny_setup, tmp_path):
    tmp, _ = tiny_setup

    def run_a_few(tag):
        tr, _ = make_trainer(tmp, max_steps=6, tok=tiny_setup[1], seed=777)
        tr.cfg.ckpt_dir = str(tmp_path / f"d{tag}c"); Path(tr.cfg.ckpt_dir).mkdir()
        tr.cfg.run_dir = str(tmp_path / f"d{tag}r"); Path(tr.cfg.run_dir).mkdir()
        tr.log_path = Path(tr.cfg.run_dir) / "log.jsonl"
        lost = []
        for s in range(6):
            x, y = tr.data.get_batch("train", 8, 24, generator=tr._rng)
            _, loss = tr.model(x, y)
            tr.opt.zero_grad(set_to_none=True)
            loss.backward()
            tr.opt.step()
            lost.append(round(float(loss), 6))
        return lost

    assert run_a_few("a") == run_a_few("b")


def test_lr_schedule_shape(tiny_setup):
    tmp, _ = tiny_setup
    tr, tcfg = make_trainer(tmp, max_steps=100, tok=tiny_setup[1])
    assert tr.max_steps == 100
    assert tr.lr_at(0) < tr.lr_at(tcfg.warmup_steps)
    peak = tr.lr_at(tcfg.warmup_steps)
    mid = tr.lr_at(50)
    end = tr.lr_at(99)
    assert peak > mid > end
    floor = tcfg.min_lr_frac * tcfg.learning_rate
    assert end <= floor + 0.05 * tcfg.learning_rate
    assert math.isclose(peak, tcfg.learning_rate, rel_tol=0.05)


def test_log_file_written_and_parseable(tiny_setup, tmp_path):
    tmp, _ = tiny_setup
    tr, _ = make_trainer(tmp, max_steps=10, tok=tiny_setup[1])
    tr.cfg.ckpt_dir = str(tmp_path / "lgc"); Path(tr.cfg.ckpt_dir).mkdir()
    tr.cfg.run_dir = str(tmp_path / "lgr"); Path(tr.cfg.run_dir).mkdir()
    tr.log_path = Path(tr.cfg.run_dir) / "log.jsonl"
    summary = tr.train()
    lines = [json.loads(x) for x in tr.log_path.read_text().splitlines() if x.strip()]
    assert any("train_loss" in r for r in lines)
    assert any("val_loss" in r for r in lines)
    assert summary["final_val_ppl"] > 1.0
    assert summary["avg_tokens_per_sec"] > 0
