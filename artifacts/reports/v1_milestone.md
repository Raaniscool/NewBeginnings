# Milestone Report — v1

## Model
- architecture: GPT-style causal LM, 4 layers, 8 heads, d_model=256, block_size=192, dropout=0.1, bias=False, weight-tied embeddings
- parameters: 5,296,640 (actual; no pretrained weights anywhere; random init)
- tokenizer: byte-level BPE trained from scratch on TRAIN split only; vocab=8192; sha256=7948c252187e6f34…

## Data
- corpus: 1631 docs after dedupe, 1.39 Mchars, sha256=358a7eecefa7df37…
- origins: seed=247 docs, generated=1131 docs, downloaded=253 docs
- screened out (code-like): 0; exact dupes removed: 247; near dupes removed: 0
- category shares (by characters):
  - stories: 45.08% (384 docs)
  - knowledge: 13.13% (97 docs)
  - conversation: 10.36% (225 docs)
  - formal: 9.04% (102 docs)
  - descriptions: 6.28% (242 docs)
  - questions_and_answers: 5.97% (249 docs)
  - explanations: 5.12% (193 docs)
  - instructions: 5.01% (139 docs)
- split tokens: train=290,146 (1360 docs), val=37,407 (168 docs), test=41,471 (103 docs)
- splits are group-aware: no generator family or seed file spans two splits (enforced in tests)

## Training
- config: batch=32, block=192, AdamW(lr=0.0006, wd=0.1), warmup=100, cosine to 0.1×lr, grad clip=1.0, seed=1234
- steps: 1410 (≈30.0 epochs), 47.4 min wall clock, avg 3043 tok/s on 2 CPU threads
- best val loss: 5.565 (ppl 261.1); final val loss: 5.5992 (ppl 270.2)

## Evaluation
- held-out TEST: loss 5.3596, perplexity **212.6** (never used for any training decision)
- benchmark (fixed prompts, raw-model decoding, saved un-cherry-picked): rep3_rate=0.0256, distinct2=0.5421, double_word=0.00164, median tokens to first rep3=72.0, terminal punct=0.9722
  (raw-model decoding confirmed)
  worst repetition category: explanations (rep3=0.0748); worst diversity category: conversation (distinct2=0.7709)
- cloze probes: grammar acc=0.5116 (n=43, chance=0.5); continuity acc=0.4286 (n=42)
- sample generations (first 6 records, incl. any failures):

```
### [conversation] "Maya: Are you coming with us?\nJames: I don't know.\nMaya: Then we should"
 rested in patience.
```

```
### [conversation] 'Tom: Did you finish the report?\nAnna: Not yet.\nTom: When will'
 finish my tea with the top, could have to the left my tea.
```

```
### [conversation] 'Lena: I lost my keys again.\nMark: You should check'
 the same for me.
```

```
### [conversation] 'Sam: That movie was a waste of time.\nPriya: I actually liked it.\nSam: Really? What did you'
're going to come in a good?
```

```
### [conversation] "Mom: Dinner is ready.\nBen: I'll be there in a minute.\nMom: Don't take too long, or"
 deep.
```

```
### [conversation] 'Ava: Could you help me move these boxes?\nNoah: Of course. Where do you want'
 to spend?
```

## English Mastery Gate
- verdict: **FAIL — Phase 1/2 continues; programming remains locked**
- floors: 6/11 met
  - [FAIL] test_ppl: 212.6 (floor 45.0)
  - [FAIL] grammar_cloze_acc: 0.5116 (floor 0.8)
  - [FAIL] continuity_cloze_acc: 0.4286 (floor 0.75)
  - [PASS] rep3_rate: 0.0256 (floor 0.1)
  - [FAIL] distinct_2: 0.5421 (floor 0.6)
  - [PASS] tokens_to_first_rep3: 72.0 (floor 32)
  - [PASS] double_word_rate: 0.00164 (floor 0.01)
  - [PASS] terminal_punct_rate: 0.9722 (floor 0.95)
  - [FAIL] question_mark_rate: 0.7 (floor 0.8)
  - [PASS] worst_category_rep3: 0.0748 (floor 0.15)
  - [PASS] worst_category_distinct2: 0.7709 (floor 0.5)
  - improvement rule: not applicable (first released version defines the baseline)
  - note: v1 is the baseline version and cannot pass the gate by rule.

## Storage
- data/seeds: 0.26 MB
- data/corpus: 3.01 MB
- data/tokenized: 0.74 MB
- data/raw_downloads: 0.73 MB
- artifacts: 0.63 MB
- checkpoints: 127.24 MB
- torch install (external): 1236.96 MB
- filesystem: 2.5 GB used / 22 GB total; 19.0 GB free
  - checkpoint best.ckpt: 63.62 MB
  - checkpoint last.ckpt: 63.62 MB
  - retention policy: best + last only; nothing else accumulates

## Honest assessment
- v1 did NOT pass the gate. Programming stays locked.
- as the baseline version, v1 could not pass by rule; its role is to make v2's improvement measurable.

