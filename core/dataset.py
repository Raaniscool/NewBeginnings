"""Dataset construction: corpus loading, group-aware splits, tokenization.

Splitting rules (these are guarantees, not suggestions):
  * splits are made at the *document* level,
  * documents sharing a `group` key (same seed source file or same template
    family) always land in the same split, so near-duplicate patterns cannot
    leak across train/val/test,
  * if any split fails minimum-size requirements we raise immediately instead
    of silently training with a meaningless validation set.

Tokenized splits are stored as flat uint16 numpy arrays (ids with explicit
BOS/EOS per document). They are regenerable from data/corpus + tokenizer and
are therefore gitignored.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

from core.tokenizer import BPETokenizer

CORPUS_DIR = Path("data/corpus")
TOKENIZED_DIR = Path("data/tokenized")

# Minimum sizes below which we REFUSE to proceed (fail loudly).
MIN_VAL_DOCS = 25
MIN_TEST_DOCS = 25
MIN_VAL_TOKENS = 5_000
MIN_TEST_TOKENS = 5_000

DOC_RE = re.compile(r"\s+")


# --------------------------------------------------------------- documents --

@dataclass
class Document:
    doc_id: str
    category: str
    text: str
    group: str          # leakage-control group key (source file / template family)
    origin: str         # "seed" | "generated" | "publicdomain"

    def to_json(self) -> str:
        return json.dumps({"id": self.doc_id, "category": self.category,
                           "text": self.text, "group": self.group,
                           "origin": self.origin}, ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "Document":
        d = json.loads(line)
        return Document(d["id"], d["category"], d["text"], d["group"],
                        d.get("origin", "seed"))


def normalize_for_dedup(text: str) -> str:
    return DOC_RE.sub(" ", text.strip().lower())


def doc_key(text: str) -> str:
    return hashlib.sha256(normalize_for_dedup(text).encode("utf-8")).hexdigest()


def dedupe_documents(docs: Sequence[Document]) -> Tuple[List[Document], int]:
    """Remove exact (normalized) duplicates, keeping first occurrence."""
    seen, out = set(), []
    dropped = 0
    for d in docs:
        k = doc_key(d.text)
        if k in seen:
            dropped += 1
            continue
        seen.add(k)
        out.append(d)
    return out, dropped


# Contamination screen: Phase 1 corpus must NOT contain source code.
CODE_PATTERNS = [
    re.compile(p) for p in (
        r"\bfunction\s*\(", r"\bdef\s+\w+\s*\(", r"\blocal\s+\w+\s*=",
        r"\brequire\s*\(", r"\bprint\s*\(", r"\bimport\s+\w+", r"\bclass\s+\w+\s*[:(]",
        r"\bvar\s+\w+\s*=", r"\blet\s+\w+\s*=", r"\bconst\s+\w+\s*=",
        r"\bif\s*\(.+\)\s*\{", r"\bfor\s*\(.+\)\s*\{", r"\bwhile\s*\(.+\)\s*\{",
        r"\w+==\w+", r"=>", r"\{\s*\}", r";\s*$",
        r"\binstance\.new\b", r"\bgame\s*:\s*\w+\(", r"\bscript\.Parent\b",
        r"</\w+>", r"<\w+\s+\w+=", r"```",
    )
]


def looks_like_code(text: str) -> bool:
    """Heuristic contamination screen for the no-code rule."""
    hits = 0
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        for pat in CODE_PATTERNS:
            if pat.search(s):
                hits += 1
                break
        if hits >= 2:
            return True
    return False


# ----------------------------------------------------------------- splits --

def group_aware_split(
    docs: Sequence[Document],
    seed: int,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
) -> Dict[str, List[Document]]:
    """Split by GROUP so documents sharing a group stay in the same split.

    Groups are shuffled once with `seed`, then assigned greedily (largest
    groups first) to whichever split is furthest below its target mass, which
    keeps ratios close to val_frac/test_frac even with uneven group sizes.
    """
    rng = random.Random(seed)
    groups: Dict[str, List[Document]] = {}
    for d in docs:
        groups.setdefault(d.group, []).append(d)

    sizes = sorted(((len(v), g) for g, v in groups.items()), reverse=True)
    rng.shuffle(sizes)
    sizes.sort(key=lambda t: -t[0])

    total = len(docs)
    targets = {"val": val_frac * total, "test": test_frac * total}
    splits: Dict[str, List[Document]] = {"train": [], "val": [], "test": []}
    current = {"train": 0, "val": 0, "test": 0}

    for size, g in sizes:
        best = None
        best_deficit = float("inf")
        for name in ("val", "test"):
            deficit = targets[name] - current[name]
            if deficit > 0 and deficit - size < best_deficit:
                best, best_deficit = name, deficit - size
        if best is None:
            best = "train"
        # never let val/test overshoot massively: send leftover groups to train
        if best != "train" and current[best] >= targets[best]:
            best = "train"
        splits[best].extend(groups[g])
        current[best] += size

    for name in ("train", "val", "test"):
        rng.shuffle(splits[name])
    return splits


def assert_split_sane(splits: Dict[str, List[Document]]) -> None:
    problems = []
    if len(splits["val"]) < MIN_VAL_DOCS:
        problems.append(f"val split has {len(splits['val'])} docs (< {MIN_VAL_DOCS})")
    if len(splits["test"]) < MIN_TEST_DOCS:
        problems.append(f"test split has {len(splits['test'])} docs (< {MIN_TEST_DOCS})")
    # disjointness by document id AND by group
    ids = {n: {d.doc_id for d in splits[n]} for n in splits}
    groups = {n: {d.group for d in splits[n]} for n in splits}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if ids[a] & ids[b]:
            problems.append(f"doc-id leakage between {a} and {b}")
        if groups[a] & groups[b]:
            problems.append(f"group leakage between {a} and {b}")
    n_val_tokens = sum(len(normalize_for_dedup(d.text).split()) for d in splits["val"])
    n_test_tokens = sum(len(normalize_for_dedup(d.text).split()) for d in splits["test"])
    if n_val_tokens < MIN_VAL_TOKENS:
        problems.append(f"val has ~{n_val_tokens} whitespace tokens (< {MIN_VAL_TOKENS})")
    if n_test_tokens < MIN_TEST_TOKENS:
        problems.append(f"test has ~{n_test_tokens} whitespace tokens (< {MIN_TEST_TOKENS})")
    if problems:
        raise RuntimeError("Split sanity check FAILED: " + "; ".join(problems))


# ------------------------------------------------------------- tokenizing --

def tokenize_documents(
    docs: Sequence[Document],
    tok: BPETokenizer,
) -> np.ndarray:
    """Encode documents into one flat uint16 array: BOS ids EOS per document."""
    if tok.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError("vocab too large for uint16 storage")
    parts: List[np.ndarray] = []
    for d in docs:
        ids = tok.encode(d.text, add_bos=True, add_eos=True)
        parts.append(np.asarray(ids, dtype=np.uint16))
    if not parts:
        return np.zeros(0, dtype=np.uint16)
    return np.concatenate(parts)


class TokenDataset:
    """Random-window batch sampler over flat token arrays."""

    def __init__(self, data_dir: str | Path = TOKENIZED_DIR):
        data_dir = Path(data_dir)
        self.splits: Dict[str, np.ndarray] = {}
        for name in ("train", "val", "test"):
            p = data_dir / f"{name}.npy"
            if p.exists():
                self.splits[name] = np.load(p)
        if "train" not in self.splits:
            raise FileNotFoundError(
                f"no train tokens at {data_dir}/train.npy; run `python3 run.py build-dataset`")
        self.meta = {}
        mp = data_dir / "tokenized_meta.json"
        if mp.exists():
            self.meta = json.loads(mp.read_text())

    def n_tokens(self, split: str) -> int:
        return int(self.splits[split].size)

    def get_batch(self, split: str, batch_size: int, block_size: int,
                  device: str = "cpu", generator=None):
        import torch
        data = self.splits[split]
        hi = data.size - block_size - 1
        if hi <= 0:
            raise ValueError(f"split {split} too small ({data.size} tokens) for block_size {block_size}")
        if generator is not None:
            starts = torch.randint(hi, (batch_size,), generator=generator).tolist()
        else:
            starts = np.random.randint(0, hi, size=batch_size).tolist()
        x = np.stack([data[s:s + block_size] for s in starts]).astype(np.int64)
        y = np.stack([data[s + 1:s + block_size + 1] for s in starts]).astype(np.int64)
        return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))

    @staticmethod
    def iter_documents(corpus_dir: str | Path = CORPUS_DIR) -> Iterator[Document]:
        for name in ("train", "val", "test"):
            p = Path(corpus_dir) / f"{name}.jsonl"
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            yield Document.from_json(line)
