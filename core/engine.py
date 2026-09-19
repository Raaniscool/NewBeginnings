"""Training engine: causal-LM training loop with checkpointing and resume.

Design requirements honored here:
  * CPU-first, threads pinned, batch + scheduler fully config-driven.
  * AdamW with decoupled weight decay (no decay on norms/embeddings),
    linear warmup -> cosine decay LR schedule, gradient clipping.
  * Periodic validation loss; perplexity is exp(mean CE loss).
  * Atomic checkpoints carrying everything needed to resume exactly:
    model, optimizer, step/epoch counters, best val loss, RNG states,
    and the resolved configs.
  * Retention policy: only best.ckpt / last.ckpt / interrupt.ckpt exist
    on disk at any time (storage constraint).
  * Ctrl+C: saves interrupt.ckpt and exits cleanly (resumable).
"""

from __future__ import annotations

import json
import math
import os
import random
import signal
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from core.config import ModelConfig, TrainConfig
from core.dataset import TokenDataset
from core.model import GPT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Trainer:
    def __init__(
        self,
        model: GPT,
        train_cfg: TrainConfig,
        data: TokenDataset,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.cfg = train_cfg
        self.data = data
        self.device = device
        torch.set_num_threads(max(1, train_cfg.num_threads))

        # optimizer: weight decay only on >=2D non-embedding weights
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2 and not name.endswith("wte.weight") and not name.endswith("wpe.weight"):
                decay.append(p)
            else:
                no_decay.append(p)
        self.opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": train_cfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=train_cfg.learning_rate, betas=(0.9, 0.95), eps=1e-8,
        )

        n_train_tokens = data.n_tokens("train")
        steps_per_epoch = max(1, n_train_tokens // (train_cfg.batch_size * train_cfg.block_size))
        self.steps_per_epoch = steps_per_epoch
        self.max_steps = train_cfg.max_steps or steps_per_epoch * train_cfg.epochs
        self.step = 0
        self.best_val = float("inf")
        self._interrupted = False
        self._rng = torch.Generator(device="cpu").manual_seed(train_cfg.seed)

        ckpt_dir = Path(train_cfg.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(train_cfg.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = run_dir / "log.jsonl"
        self._old_sigint = None

    # ---------------------------------------------------------- scheduling --

    def lr_at(self, step: int) -> float:
        c = self.cfg
        if step < c.warmup_steps:
            return c.learning_rate * (step + 1) / c.warmup_steps
        span = max(1, self.max_steps - c.warmup_steps)
        t = min(1.0, (step - c.warmup_steps) / span)
        floor = c.min_lr_frac * c.learning_rate
        return floor + 0.5 * (1 + math.cos(math.pi * t)) * (c.learning_rate - floor)

    def _set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    # ------------------------------------------------------------ evaluation --

    @torch.no_grad()
    def evaluate(self, split: str, iters: int) -> Dict[str, float]:
        self.model.eval()
        losses = []
        n = 0
        for _ in range(iters):
            x, y = self.data.get_batch(split, self.cfg.batch_size, self.cfg.block_size,
                                       device=self.device)
            _, loss = self.model(x, y)
            losses.append(float(loss))
            n += 1
        self.model.train()
        mean = float(np.mean(losses)) if losses else float("nan")
        return {"loss": mean, "ppl": float(math.exp(mean)) if mean < 20 else float("inf"),
                "iters": n}

    # ----------------------------------------------------------- checkpoint --

    def _ckpt_payload(self) -> dict:
        return {
            "model_state": self.model.state_dict(),
            "model_config": vars(self.model.cfg),
            "opt_state": self.opt.state_dict(),
            "step": self.step,
            "best_val": self.best_val,
            "train_config": vars(self.cfg),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "sampler": self._rng.get_state(),
            },
            "time": time.time(),
        }

    @staticmethod
    def _atomic_save(payload: dict, path: Path) -> None:
        tmp = path.with_suffix(".ckpt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)  # atomic on same filesystem

    def save(self, kind: str) -> Path:
        assert kind in ("best", "last", "interrupt")
        path = Path(self.cfg.ckpt_dir) / f"{kind}.ckpt"
        self._atomic_save(self._ckpt_payload(), path)
        return path

    def load(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state"])
        self.opt.load_state_dict(payload["opt_state"])
        self.step = payload["step"]
        self.best_val = payload.get("best_val", float("inf"))
        rng = payload.get("rng")
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            self._rng.set_state(rng["sampler"])

    # ------------------------------------------------------------- training --

    def _log(self, record: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _install_sigint(self) -> None:
        def handler(signum, frame):
            self._interrupted = True
        self._old_sigint = signal.signal(signal.SIGINT, handler)

    def _restore_sigint(self) -> None:
        if self._old_sigint is not None:
            signal.signal(signal.SIGINT, self._old_sigint)

    def train(self) -> Dict[str, float]:
        cfg = self.cfg
        self._install_sigint()
        t_start = time.time()
        last_log = t_start
        tokens_per_step = cfg.batch_size * cfg.block_size
        try:
            while self.step < self.max_steps:
                lr = self.lr_at(self.step)
                self._set_lr(lr)
                x, y = self.data.get_batch(
                    "train", cfg.batch_size, cfg.block_size,
                    device=self.device, generator=self._rng)
                _, loss = self.model(x, y)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.opt.step()
                self.step += 1

                if self.step % cfg.log_interval == 0 or self.step == 1:
                    now = time.time()
                    tps = tokens_per_step * cfg.log_interval / max(1e-9, now - last_log)
                    last_log = now
                    rec = {"step": self.step, "lr": lr, "train_loss": float(loss.detach()),
                           "tokens_per_sec": round(tps), "elapsed_s": round(now - t_start, 1)}
                    self._log(rec)
                    print(f"step {self.step:>6}/{self.max_steps} | lr {lr:.2e} | "
                          f"loss {float(loss.detach()):.3f} | {tps:.0f} tok/s", flush=True)

                if self.step % cfg.eval_interval == 0 or self.step == self.max_steps:
                    val = self.evaluate("val", cfg.eval_iters)
                    rec = {"step": self.step, "val_loss": val["loss"], "val_ppl": val["ppl"]}
                    self._log(rec)
                    print(f"  [eval] step {self.step} | val loss {val['loss']:.3f} | "
                          f"val ppl {val['ppl']:.1f}", flush=True)
                    if val["loss"] < self.best_val:
                        self.best_val = val["loss"]
                        self.save("best")
                        print(f"  [ckpt] new best -> best.ckpt", flush=True)
                    self.save("last")
                    if cfg.keep_best:
                        # retention policy: nothing else accumulates on disk
                        pass

                if self._interrupted:
                    print(f"\n[interrupt] Ctrl+C at step {self.step}; saving interrupt.ckpt",
                          flush=True)
                    self.save("interrupt")
                    self.save("last")
                    break
        finally:
            self._restore_sigint()
        total = time.time() - t_start
        final_val = self.evaluate("val", cfg.eval_iters)
        summary = {
            "steps": self.step,
            "max_steps": self.max_steps,
            "epochs_equiv": round(self.step / self.steps_per_epoch, 2),
            "best_val_loss": round(self.best_val, 4),
            "best_val_ppl": round(math.exp(self.best_val), 1) if self.best_val < 20 else None,
            "final_val_loss": round(final_val["loss"], 4),
            "final_val_ppl": round(final_val["ppl"], 1),
            "train_seconds": round(total, 1),
            "avg_tokens_per_sec": round(self.step * tokens_per_step / max(1e-9, total)),
            "interrupted": self._interrupted,
        }
        self._log({"summary": summary})
        return summary
