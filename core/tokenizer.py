"""Byte-level BPE tokenizer, trained FROM SCRATCH on our own English corpus.

We deliberately do NOT use any pretrained tokenizer. The `tokenizers` library
is used purely as a BPE *training* toolkit; the resulting vocabulary is
derived only from files passed to `train_tokenizer` -- which, by project rule,
is the *training split only*. The trained tokenizer JSON is committed to git
so tokenization is exactly reproducible.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Sequence

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer

BOS, EOS, PAD = "<|bos|>", "<|eos|>", "<|pad|>"
SPECIAL_TOKENS = [BOS, EOS, PAD]

TOKENIZER_DIR = Path("artifacts/tokenizer")
TOKENIZER_JSON = TOKENIZER_DIR / "tokenizer.json"
TOKENIZER_META = TOKENIZER_DIR / "tokenizer_meta.json"


def train_tokenizer(
    files: Sequence[str | Path],
    vocab_size: int = 8192,
    min_frequency: int = 3,
    out_dir: str | Path = TOKENIZER_DIR,
) -> "BPETokenizer":
    """Train a byte-level BPE tokenizer on the given files and save it.

    Files are streamed line-by-line so memory stays flat. The caller MUST
    pass only training-split files (enforced by run.py, verified in tests by
    construction of the caller).
    """
    files = [Path(f) for f in files]
    if not files:
        raise ValueError("train_tokenizer called with no files")
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f)

    tok = Tokenizer(BPE(unk_token=None))
    tok.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()

    def _iterator() -> Iterable[str]:
        for path in files:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if line:
                        yield line

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevelPreTokenizer.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(_iterator(), trainer=trainer)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_dir / "tokenizer.json"))
    return BPETokenizer(tok)


class BPETokenizer:
    """Thin wrapper around the trained tokenizer with project conventions."""

    def __init__(self, tok: Tokenizer):
        self._tok = tok

    # -- id access -----------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def bos_id(self) -> int:
        return self._tok.token_to_id(BOS)

    @property
    def eos_id(self) -> int:
        return self._tok.token_to_id(EOS)

    @property
    def pad_id(self) -> int:
        return self._tok.token_to_id(PAD)

    # -- encode/decode --------------------------------------------------
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self._tok.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=True)

    def token_to_id(self, token: str) -> int:
        tid = self._tok.token_to_id(token)
        if tid is None:
            raise KeyError(f"unknown special token: {token}")
        return tid

    # -- persistence ----------------------------------------------------
    def save(self, out_dir: str | Path = TOKENIZER_DIR) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(out_dir / "tokenizer.json"))
        digest = self.file_hash(out_dir / "tokenizer.json")
        meta = {"vocab_size": self.vocab_size, "sha256": digest,
                "special_tokens": SPECIAL_TOKENS,
                "note": "byte-level BPE trained from scratch on the TRAIN split only"}
        (out_dir / "tokenizer_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path = TOKENIZER_JSON) -> "BPETokenizer":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"tokenizer not found at {p}; run `python3 run.py train-tokenizer` first")
        return cls(Tokenizer.from_file(str(p)))

    @staticmethod
    def file_hash(path: str | Path) -> str:
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
