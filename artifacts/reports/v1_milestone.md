# Milestone Report — v1

## Model
- architecture: GPT-style causal LM, 4 layers, 8 heads, d_model=256, block_size=192, dropout=0.1, bias=False, weight-tied embeddings
- parameters: 5,296,640 (no pretrained weights anywhere; random init)
- tokenizer: byte-level BPE trained from scratch on TRAIN split only; vocab=6502; sha256=effb3c27fc74a875…

## Data
- corpus: ? docs after dedupe, 1.83 Mchars, sha256=99cb48860ce693db…
- screened out (code-like): 0; exact dupes removed: 0; near dupes removed: 0
- category shares (by characters):
  - stories: 35.37% (1037 docs)
  - conversation: 18.19% (655 docs)
  - questions_and_answers: 13.54% (897 docs)
  - descriptions: 11.95% (689 docs)
  - instructions: 7.6% (308 docs)
  - explanations: 6.1% (332 docs)
  - formal: 4.2% (209 docs)
  - knowledge: 3.05% (91 docs)
- split tokens: train=358,507 (3588 docs), val=53,077 (340 docs), test=32,380 (290 docs)
- splits are group-aware: no generator family or seed file spans two splits (enforced in tests)

## Training
- config: batch=32, block=192, AdamW(lr=0.0006, wd=0.1), warmup=100, cosine to 0.1×lr, grad clip=1.0, seed=1234
- training summary: MISSING (run `python3 run.py train`)

## Evaluation
- held-out test: MISSING
- benchmark metrics: MISSING
- cloze probes: MISSING

## English Mastery Gate
- gate result: MISSING

## Storage
- data/seeds: 0.26 MB
- data/corpus: 4.23 MB
- data/tokenized: 0.89 MB
- data/raw_downloads: 0.00 MB
- artifacts: 0.43 MB
- checkpoints: 0.00 MB
- torch install (external): 1236.96 MB
- filesystem: 2.4 GB used / 22 GB total; 19.2 GB free
  - retention policy: best + last only; nothing else accumulates

## Honest assessment
- no evaluation artifacts yet; nothing can be concluded.

