# NewBeginnings Roadmap

## THE LOCK

```
┌──────────────────────────────────────────────────────────────────┐
│  Programming content of ANY kind is LOCKED until the English     │
│  Mastery Gate (ENGLISH_GATE.md) is PASSED by a released model    │
│  version. No Lua. No Luau. No Roblox. No Python. No pseudocode.  │
│  Not "a little bit mixed in". Not "just to help formatting".     │
│  The English foundation comes first, in full.                    │
└──────────────────────────────────────────────────────────────────┘
```

Enforcement mechanisms (not just intentions):

- `scripts/build_corpus.py` has a **code-contamination screen** that rejects
  any document that looks like source code. Phase 1 corpus = 0% code.
- The gate config (`configs/gate_v1.json`) is evaluated by
  `python3 run.py gate`; a model version only "unlocks" Phase 3 when
  `gate_result.json` says `passed: true`. That file is committed history.
- Milestone reports must include the explicit question: *"Is programming
  allowed yet?"* with the gate result as the answer.

## Phases

| # | Phase | Status |
|---|-------|--------|
| 1 | General English | **ACTIVE** |
| 2 | English mastery + strong evaluation | blocked (needs Phase 1) |
| 3 | General programming concepts | 🔒 locked by gate |
| 4 | Lua | 🔒 locked by gate |
| 5 | Luau | 🔒 locked by gate |
| 6 | Roblox development | 🔒 locked by gate |
| 7 | CodeBook (structured programming knowledge) | 🔒 locked by gate |
| 8 | Code execution + automated testing | 🔒 locked by gate |
| 9 | Debugging + regeneration | 🔒 locked by gate |
| 10 | Verification | 🔒 locked by gate |
| 11 | More advanced agentic behavior | 🔒 locked by gate |

## Phase 1 scope (current)

1. Compact, diverse, code-free English corpus (many registers, topics, lengths).
2. BPE tokenizer trained from scratch on the train split only.
3. Small GPT-style causal Transformer (CPU-trainable), trained from random init.
4. Proper document-level train/val/test splits; true held-out test set.
5. Fixed English benchmark suite + repetition/diversity/continuity metrics.
6. Versioned model releases (V1, V2, ...) with committed configs + results.
7. Honest gate evaluation. V1 is *expected* to fail the gate — that is a
   normal result, and it keeps the lock closed.

## Phase 2 scope (after Phase 1 is stable)

- Iterate on data quality/variety, tokenizer quality, training stability and
  duration — *not* raw model size — until the gate passes.
- Every V-next is compared against the SAME fixed benchmark.

## Future notes preserved for later phases

### The CodeBook idea (Phase 7; do NOT build now)

Each programming language gets a structured **CodeBook**: syntax, standard
APIs, common patterns, best practices, common errors, version-specific
behavior, security/performance considerations, examples, and relationships
between concepts. Generated code is checked against the relevant CodeBook
before execution. This is FUTURE work, recorded here so it is not forgotten.

### The eventual pipeline (Phases 8–11; do NOT build now)

```
user request → understand → plan → generate code → check vs CodeBook →
execute/test → detect errors → debug → regenerate → re-test → verify →
return → record verified lessons
```

## Model versioning

- Every major trained model is a version: `v1`, `v2`, ...
- Each version commits: model config, train config, tokenizer config + hash,
  corpus manifest hash, training log summary, benchmark outputs, gate result.
- Never overwrite a released version's artifacts.
- Do **not** increase model size without written evidence in the version's
  report that the bottleneck was capacity rather than data/training/eval.
