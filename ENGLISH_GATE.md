# The English Mastery Gate

This document is the **binding definition** of when this project may introduce
programming content. It was written **before the first model was trained**, on
purpose: the gate must not be invented after the fact to fit whatever the model
happens to achieve.

## Gate philosophy

1. **Loss going down is not mastery.** The gate uses behavioral evidence
   (held-out prediction, cloze probes, generation quality metrics across many
   distinct categories), not a single scalar.
2. **Mastery must be broad.** Passing requires minimum performance in *every*
   major category at once — a model that tells decent stories but degenerates
   on questions has not passed.
3. **Decoded text is not model knowledge.** Generation settings are pinned in
   `configs/gate_v1.json`. Metrics computed with fancier decoding
   (top-p, repetition penalty) are reported separately and **cannot** be used
   to pass the gate; gate metrics use the pinned decode configurations only.
4. **V1 is the baseline and cannot pass.** A version can only pass the gate by
   beating a previously released version AND meeting the absolute floors. The
   first possible passing version is therefore **v2 or later**. This prevents
   declaring victory over an arbitrary bar.
5. **No ratchet-down.** Gate floors may be *raised* freely; they may only be
   *lowered* with a written justification committed to this file's
   Calibration Log, and never merely to let a specific version pass.

## What is measured

Measured on the **fixed held-out test set** plus the **fixed benchmark suite**
(`benchmarks/english_v1.json`, decoding pinned in `configs/gate_v1.json`):

### A. Predictive quality (held-out)
- `test_ppl` — perplexity on the held-out test split.
- `grammar_cloze_acc` — accuracy on handcrafted multiple-choice grammar cloze
  probes: subject/verb agreement, tense consistency, pronoun case/gender/number,
  pluralization, articles, common function words. Chance = 0.50 (all probes are
  binary after grouping). Model picks by next-token log-probability; no sampling.
- `continuity_cloze_acc` — accuracy on entity-continuity cloze probes: after a
  sentence introduces "Sarah", does the model prefer "She" over "He/It/They"?
  After "the red book", does it prefer "the book/it" over a random entity?
  Chance = 0.50.

### B. Generation quality (benchmark suite, pinned decoding)
For each prompt the model produces a fixed-length continuation. Metrics are
computed per category and overall:

- `rep3_rate` — fraction of 3-grams that repeat earlier 3-grams within the same
  continuation (degeneration detector).
- `distinct_1`, `distinct_2` — ratio of unique unigrams / bigrams to total
  (lexical diversity).
- `tokens_to_first_rep3` — how many tokens before the first repeated 3-gram
  (early-collapse detector).
- `double_word_rate` — rate of immediate word repeats ("the the").
- `terminal_punct_rate` — fraction of continuations containing at least one
  sentence-ending punctuation mark (basic sentence-boundary awareness).
- `question_mark_rate` — on the `questions` category only: fraction of
  continuations containing a question mark where the prompt clearly asks a
  question in dialogue form (format-awareness probe).

### C. Category breadth
The benchmark has categories: `conversation`, `questions`, `explanations`,
`descriptions`, `instructions`, `stories`, `formal`, `grammar`, `continuity`.
Each category gets the same generation metrics computed independently.
"Category minimums" below must be met by the *worst* category, so one strong
category cannot hide a degenerate one.

## The floors (version 1.0 of this gate)

Justification follows each value.

| Metric | Floor | Why this number |
|---|---|---|
| `test_ppl` | ≤ 45.0 | On ~8k-vocab BPE English, a model that has genuinely internalized the local grammar/statistics of *our own compact corpus* (not the open internet) should reach the 30–50 range; this is a necessary-not-sufficient condition chosen to be achievable by a *well-trained* small model on this corpus, not by a distracted one. Uniform-random is ~9,000; an order-1 word model would sit in the low hundreds on this text. |
| `grammar_cloze_acc` | ≥ 0.80 | Chance is 0.50. These probes test the most frequent patterns of English (agreement, pronouns, tense). 0.80 = the model systematically encodes them with a modest error margin. |
| `continuity_cloze_acc` | ≥ 0.75 | Chance 0.50. Slightly lower than grammar: entity tracking across sentences is harder, but must be clearly above chance to proceed. |
| `rep3_rate` (overall) | ≤ 0.10 | Fewer than 1 in 10 trigrams repeating in a continuation. Unprompted trigram loops are the canonical small-model degeneration; healthy sampled English sits far below this. |
| `distinct_2` (overall) | ≥ 0.60 | Natural paragraph-scale English typically exceeds 0.7; 0.60 leaves room for short prompts while excluding echo-generators. |
| `tokens_to_first_rep3` (median) | ≥ 32 | The model must sustain roughly two dozen tokens before any trigram repeats — i.e., several clauses of non-looped text. |
| `double_word_rate` | ≤ 0.01 | "the the" style errors must be rare. |
| `terminal_punct_rate` | ≥ 0.95 | Nearly every continuation should show sentence-boundary awareness. |
| `question_mark_rate` (questions cat.) | ≥ 0.80 | Asking-format awareness in dialogue. |
| Worst-category `rep3_rate` | ≤ 0.15 | Breadth rule: no category may degenerate. |
| Worst-category `distinct_2` | ≥ 0.50 | Breadth rule: no echo-chamber category. |

### Improvement requirement (any candidate version vs the previous released version)

A candidate must also:
- improve (or hold within ±3% relative) on at least **8 of the 11** metrics above, and
- strictly improve `test_ppl` AND strictly improve (or hold within 1 point) both cloze accuracies.

This makes "better data, same architecture" and "longer training" the expected
ways forward, and blocks passing by luck of a single run.

## What passing does NOT mean

- It does not mean the model writes beautiful English.
- It does not mean the model is "done" learning English.
- It means the foundation is solid enough that adding a **new phase**
  (general programming concepts, later Lua/Luau/Roblox in their own phases)
  will not be built on mush. English training never fully stops either.

## Failure response

If a version fails the gate, the milestone report must say exactly which floors
failed, show representative failing generations, and the project stays in
Phase 1/2. The correct levers, in priority order: data quality and variety,
training stability/duration, tokenizer quality, evaluation depth — **and only
with written evidence, model capacity.**

## Calibration Log

- **2026-09-19 — Gate v1.0 created, before V1 training.** Floors above set
  pre-registration style, from known behavior of small Transformers on small
  corpora. V1 will be reported against these floors but, per §philosophy-4,
  cannot pass; V1's measurements will show how far the baseline is and give
  the improvement requirement its reference point.
