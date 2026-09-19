# NewBeginnings

Building a language model **from scratch** (no pretrained weights, no pretrained
tokenizer, no external AI APIs) with one governing rule:

> ## HARD RULE: ENGLISH FIRST.
> The model must demonstrably master general English **before any programming
> content is introduced**. Lua/Luau/Roblox are locked behind the
> [English Mastery Gate](ENGLISH_GATE.md).

## Current status

**PHASE 1 — General English only.**

See [ROADMAP.md](ROADMAP.md) for the full phase plan and the lock on programming
content. See [ENGLISH_GATE.md](ENGLISH_GATE.md) for the exact criteria that must
be met before the project is allowed to move to programming concepts.

## What this project is (Phase 1)

- A small GPT-style causal Transformer written in PyTorch, trained from random
  initialization on a compact, diverse, hand-curated English corpus.
- A BPE tokenizer trained **from scratch, on the training split only**.
- A rigorous evaluation harness: held-out perplexity, generation benchmarks,
  repetition/diversity metrics, context-continuity probes, and an explicit
  English Mastery Gate.
- A test suite including an overfit test that proves the pipeline can learn.

## Repository layout

```
core/           library code (config, tokenizer, model, dataset, engine,
                generate, metrics, evaluate)
configs/        model/training/gate configurations (JSON)
scripts/        corpus building + maintenance scripts
tests/          pytest suite (including the overfit test)
benchmarks/     FIXED benchmark prompt suites
data/
  seeds/        hand-authored corpus source text (committed)
  raw_downloads/ public-domain downloads (gitignored; re-fetchable)
  corpus/       built corpus documents (gitignored; regenerable)
  tokenized/    tokenized datasets (gitignored; regenerable)
artifacts/
  tokenizer/    trained tokenizer JSON (committed: reproducibility)
  benchmarks/   benchmark outputs per model version (committed)
  reports/      milestone reports and metrics (committed)
checkpoints/    model checkpoints (gitignored: large)
run.py          single CLI entry point for everything
```

## Quick start

```bash
pip install -r requirements.txt
python3 run.py build-corpus            # assemble + dedupe the English corpus
python3 run.py train-tokenizer         # train BPE on the *train split only*
python3 run.py build-dataset           # tokenize into train/val/test arrays
pytest -q                              # run the full test suite
python3 run.py train --config configs/v1_train.json
python3 run.py benchmark --version v1  # run the fixed English benchmark
python3 run.py gate --version v1       # evaluate the English Mastery Gate
```

## Rules this repository enforces on itself

1. **No pretrained anything.** Model weights are random-init. The tokenizer is
   trained on our own training split. No external AI APIs.
2. **No code in the corpus.** The Phase 1 corpus is natural English only and is
   actively screened for code-like contamination (`{`, `}`, `function(`,
   `def `, `local `, `==`, `=>`, semicolon-heavy lines, etc.).
3. **Document-level splits.** Validation/test documents never share a source
   document with training.
4. **The tokenizer only ever sees the train split.** Prevents leakage through
   vocabulary statistics.
5. **Fixed benchmarks.** `benchmarks/english_v1.json` is the comparison point
   for every model version. Generation settings are pinned per version.
6. **Small checkpoints, small retention.** Only `best` + `last` + at most one
   interrupt checkpoint are kept on disk. Checkpoints never enter git.
7. **Honest reporting.** Every milestone report includes representative
   failures, not cherry-picked successes, and an explicit gate verdict.

## Storage policy

Arena workspace storage is finite. The policy is: commit small text (code,
seeds, configs, manifests, reports, benchmark outputs); regenerate everything
large or binary (corpus builds, tokenized arrays, checkpoints) from committed
sources. `python3 run.py storage` prints the current usage breakdown.
