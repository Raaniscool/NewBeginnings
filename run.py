#!/usr/bin/env python3
"""NewBeginnings single entry point.

  python3 run.py build-corpus            assemble + dedupe + split the corpus
  python3 run.py train-tokenizer         BPE trained on the TRAIN split only
  python3 run.py build-dataset           tokenize splits -> data/tokenized/
  python3 run.py train [--resume PATH]   train (CPU), config-driven
  python3 run.py eval-test               final held-out test loss/perplexity
  python3 run.py generate --prompt ...   ad-hoc generation
  python3 run.py benchmark               fixed English benchmark suite
  python3 run.py cloze                   grammar + continuity cloze probes
  python3 run.py gate                    English Mastery Gate evaluation
  python3 run.py report                  milestone report
  python3 run.py storage                 disk usage breakdown

All behavior is config-driven; see configs/. Nothing pretrained anywhere.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def cmd_storage(_args) -> None:
    def size_of(p: Path) -> int:
        if not p.exists():
            return 0
        if p.is_file():
            return p.stat().st_size
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    print("== storage breakdown (workspace) ==")
    for name in ("data/seeds", "data/raw_downloads", "data/corpus", "data/tokenized",
                 "artifacts", "checkpoints", "benchmarks", "core", "tests", "configs"):
        n = size_of(ROOT / name)
        print(f"  {name:<22} {n/1e6:>9.2f} MB")
    tracked = size_of(ROOT) - size_of(ROOT / ".git")
    print(f"  {'TOTAL (excl .git)':<22} {tracked/1e6:>9.2f} MB")
    print(f"  {'.git':<22} {size_of(ROOT/'.git')/1e6:>9.2f} MB")
    total, used, free = shutil.disk_usage(ROOT)
    print(f"== filesystem: {used/1e9:.1f} GB used / {total/1e9:.0f} GB total "
          f"({free/1e9:.1f} GB free) ==")


def cmd_build_corpus(args) -> None:
    from core.corpus import build_corpus
    manifest = build_corpus(seed=args.seed, corpus_dir=args.out,
                            seeds_dir=args.seeds, downloads_dir=args.downloads,
                            verbose=True)
    print(json.dumps(manifest["categories"], indent=2))


def cmd_train_tokenizer(args) -> None:
    from core.tokenizer import BPETokenizer, train_tokenizer
    train_text = ROOT / "data/corpus/train.txt"
    if not train_text.exists():
        raise SystemExit("corpus not built; run `python3 run.py build-corpus` first")
    tok = train_tokenizer([train_text], vocab_size=args.vocab_size,
                          min_frequency=args.min_freq)
    tok.save()
    print(f"tokenizer trained on data/corpus/train.txt ONLY; "
          f"vocab_size={tok.vocab_size} bos={tok.bos_id} eos={tok.eos_id}")
    probe = "The quick brown fox jumps over the lazy dog. Isn't it lovely?"
    assert tok.decode(tok.encode(probe)) == probe, "tokenizer roundtrip failed"
    print("roundtrip probe: OK")


def cmd_build_dataset(_args) -> None:
    from core.dataset import TokenDataset, tokenize_documents, Document
    from core.tokenizer import BPETokenizer
    tok = BPETokenizer.load(ROOT / "artifacts/tokenizer/tokenizer.json")
    out = ROOT / "data/tokenized"
    out.mkdir(parents=True, exist_ok=True)
    import numpy as np
    meta = {}
    for split in ("train", "val", "test"):
        docs = []
        p = ROOT / f"data/corpus/{split}.jsonl"
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(Document.from_json(line))
        ids = tokenize_documents(docs, tok)
        np.save(out / f"{split}.npy", ids)
        meta[split] = {"n_docs": len(docs), "n_tokens": int(ids.size)}
        print(f"{split}: {len(docs)} docs, {ids.size} tokens")
    ds = TokenDataset(out)
    for split, mins in (("val", 5000), ("test", 5000)):
        if ds.n_tokens(split) < mins:
            raise SystemExit(f"FAIL: {split} split has {ds.n_tokens(split)} tokens (< {mins})"
                             " — refusing to proceed with an invalid split")
    meta["vocab_size"] = tok.vocab_size
    (out / "tokenized_meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def cmd_train(args) -> None:
    import torch
    from core.config import ModelConfig, TrainConfig, load_config
    from core.dataset import TokenDataset
    from core.engine import Trainer, set_seed
    from core.model import GPT

    tcfg = load_config(TrainConfig, args.config)
    mcfg = load_config(ModelConfig, tcfg.model_config_path)
    tok_path = Path(tcfg.tokenizer_path)
    meta = json.loads((tok_path.parent / "tokenizer_meta.json").read_text())
    mcfg.vocab_size = meta["vocab_size"]
    if mcfg.block_size != tcfg.block_size:
        raise SystemExit("model/train block_size mismatch")

    set_seed(tcfg.seed)
    model = GPT(mcfg)
    print(f"model params: {model.num_params():,} (estimate "
          f"{mcfg.n_params_estimate():,})")

    data = TokenDataset(tcfg.data_dir)
    trainer = Trainer(model, tcfg, data)
    if args.resume:
        trainer.load(args.resume)
        print(f"resumed from {args.resume} at step {trainer.step}")
    t0 = time.time()
    summary = trainer.train()
    summary["wall_clock_s"] = round(time.time() - t0, 1)
    summary["model_params"] = model.num_params()
    reports = ROOT / "artifacts/reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{tcfg.version}_train_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def _load_best(args):
    from core.generate import load_model
    from core.tokenizer import BPETokenizer
    ckpt = args.ckpt or f"checkpoints/{args.version}/best.ckpt"
    if not Path(ckpt).exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    model = load_model(ckpt)
    tok = BPETokenizer.load(ROOT / "artifacts/tokenizer/tokenizer.json")
    return model, tok, ckpt


def cmd_eval_test(args) -> None:
    from core.config import TrainConfig, load_config
    from core.dataset import TokenDataset
    from core.engine import Trainer
    model, tok, ckpt = _load_best(args)
    tcfg = load_config(TrainConfig, args.config or f"configs/{args.version}_train.json")
    data = TokenDataset(tcfg.data_dir)
    trainer = Trainer(model, tcfg, data)
    res = trainer.evaluate("test", iters=args.iters)
    out = {"version": args.version, "ckpt": str(ckpt),
           "test_loss": round(res["loss"], 4), "test_ppl": round(res["ppl"], 1),
           "iters": res["iters"]}
    reports = ROOT / "artifacts/reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{args.version}_test.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


def cmd_generate(args) -> None:
    from core.config import DecodeConfig
    from core.generate import generate_texts
    model, tok, ckpt = _load_best(args)
    decode = DecodeConfig(max_new_tokens=args.max_new, temperature=args.temperature,
                          top_p=args.top_p, top_k=args.top_k)
    for text in generate_texts(model, tok, [args.prompt], decode):
        print(f"PROMPT: {args.prompt}\n{text}\n")


def cmd_benchmark(args) -> None:
    from core.config import DecodeConfig, load_config
    model, tok, ckpt = _load_best(args)
    decode = load_config(DecodeConfig, args.decode or "configs/gate_decode.json")
    out_dir = ROOT / f"artifacts/benchmarks/{args.version}"
    from core.evaluate import run_benchmark
    metrics = run_benchmark(model, tok, args.bench or "benchmarks/english_v1.json",
                            decode, out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics["overall"], indent=2))
    print(f"outputs written to {out_dir}")


def cmd_cloze(args) -> None:
    model, tok, ckpt = _load_best(args)
    from core.evaluate import run_cloze
    out_dir = ROOT / f"artifacts/benchmarks/{args.version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, path in (("grammar", "benchmarks/cloze_grammar_v1.json"),
                       ("continuity", "benchmarks/cloze_continuity_v1.json")):
        result[name] = run_cloze(model, tok, ROOT / path)
        print(f"{name}: acc={result[name]['accuracy']} (n={result[name]['n']}, "
              f"chance={result[name]['chance']})")
    (out_dir / "cloze.json").write_text(json.dumps(result, indent=2) + "\n")


def cmd_gate(args) -> None:
    out_dir = ROOT / f"artifacts/benchmarks/{args.version}"
    bench_metrics = json.loads((out_dir / "metrics.json").read_text())
    cloze = json.loads((out_dir / "cloze.json").read_text())
    test = json.loads((Path(f"artifacts/reports/{args.version}_test.json")).read_text())
    prev = None
    if args.prev:
        p = ROOT / f"artifacts/benchmarks/{args.prev}/gate_result.json"
        if p.exists():
            prev = p
    from core.evaluate import evaluate_gate
    result = evaluate_gate(args.version, bench_metrics, cloze["grammar"],
                           cloze["continuity"], test["test_ppl"],
                           "configs/gate_v1.json", prev_metrics_path=prev)
    wrapped = {"version": args.version, "gate": result,
               "test_ppl": test["test_ppl"],
               "benchmark_metrics": bench_metrics, "cloze": cloze}
    (out_dir / "gate_result.json").write_text(json.dumps(wrapped, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))
    print("\nfloor checks:")
    for name, c in result["checks"].items():
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {name:<26} "
              f"value={c['value']:<10} floor={c['floor']}")
    print(f"\nVERDICT: {result['verdict']}")


def cmd_report(args) -> None:
    from core.report import build_report
    print(build_report(args.version))


def main() -> None:
    ap = argparse.ArgumentParser(description="NewBeginnings Phase 1 CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("storage")

    p = sub.add_parser("build-corpus")
    p.add_argument("--seed", type=int, default=20260919)
    p.add_argument("--out", default="data/corpus")
    p.add_argument("--seeds", default="data/seeds")
    p.add_argument("--downloads", default="data/raw_downloads")

    p = sub.add_parser("train-tokenizer")
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--min-freq", type=int, default=3)

    sub.add_parser("build-dataset")

    p = sub.add_parser("train")
    p.add_argument("--config", default="configs/v1_train.json")
    p.add_argument("--resume", default=None)

    p = sub.add_parser("eval-test")
    p.add_argument("--version", default="v1")
    p.add_argument("--config", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--iters", type=int, default=60)

    p = sub.add_parser("generate")
    p.add_argument("--version", default="v1")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)

    p = sub.add_parser("benchmark")
    p.add_argument("--version", default="v1")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--bench", default=None)
    p.add_argument("--decode", default=None)

    p = sub.add_parser("cloze")
    p.add_argument("--version", default="v1")
    p.add_argument("--ckpt", default=None)

    p = sub.add_parser("gate")
    p.add_argument("--version", default="v1")
    p.add_argument("--prev", default=None)

    p = sub.add_parser("report")
    p.add_argument("--version", default="v1")

    args = ap.parse_args()
    {
        "storage": cmd_storage,
        "build-corpus": cmd_build_corpus,
        "train-tokenizer": cmd_train_tokenizer,
        "build-dataset": cmd_build_dataset,
        "train": cmd_train,
        "eval-test": cmd_eval_test,
        "generate": cmd_generate,
        "benchmark": cmd_benchmark,
        "cloze": cmd_cloze,
        "gate": cmd_gate,
        "report": cmd_report,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
