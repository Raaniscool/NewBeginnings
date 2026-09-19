"""Tokenizer tests: training-from-scratch, encode/decode, persistence,
determinism. A broken tokenizer silently poisons everything downstream, so
this file is deliberately strict.
"""

from __future__ import annotations

import json

import pytest

from core.tokenizer import BPETokenizer, train_tokenizer

TINY_CORPUS = (
    "the cat sat on the mat, and the cat was glad.\n"
    "how does a plane fly?\nit is a good question!\nthe river runs cold.\n"
    "Sarah picked up the book. She opened it, and she smiled.\n"
    "the lighthouse keeper climbed the long stairs every evening at dusk.\n"
    "what is the difference between a duck and a goose, my friend?\n"
)


@pytest.fixture()
def trained(tmp_path):
    corpus = tmp_path / "train_only.txt"
    corpus.write_text(TINY_CORPUS * 50, encoding="utf-8")
    tok = train_tokenizer([corpus], vocab_size=300, min_frequency=2,
                          out_dir=tmp_path / "tk")
    return tok, tmp_path


def test_special_ids_present_and_stable(trained):
    tok, _ = trained
    assert tok.bos_id is not None and tok.eos_id is not None and tok.pad_id is not None
    assert len({tok.bos_id, tok.eos_id, tok.pad_id}) == 3
    assert max(tok.bos_id, tok.eos_id, tok.pad_id) < tok.vocab_size


def test_vocab_size_bounded_by_target(trained):
    tok, _ = trained
    assert tok.vocab_size <= 300


def test_roundtrip_exact_including_newlines_and_punctuation(trained):
    tok, _ = trained
    s = ("Sarah picked up the book. She opened it, and she smiled.\n"
         "How does a plane fly?\nIt is a good question!")
    ids = tok.encode(s)
    assert ids, "encoding must not be empty"
    assert tok.decode(ids) == s


def test_bos_eos_appended_correctly(trained):
    tok, _ = trained
    ids = tok.encode("hello there", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id


def test_unicode_and_unseen_chars_do_not_crash(trained):
    tok, _ = trained
    ids = tok.encode("café — 日本語 😀 plain")
    assert isinstance(ids, list)
    # byte-level: anything is representable, decode must not crash either
    tok.decode(ids + [tok.eos_id])


def test_save_load_is_lossless(trained, tmp_path):
    tok, _ = trained
    tok.save(tmp_path / "saved")
    assert (tmp_path / "saved" / "tokenizer.json").exists()
    meta = json.loads((tmp_path / "saved" / "tokenizer_meta.json").read_text())
    assert meta["vocab_size"] == tok.vocab_size
    tok2 = BPETokenizer.load(tmp_path / "saved" / "tokenizer.json")
    s = "The lighthouse keeper climbed the stairs, and he was not afraid."
    assert tok2.encode(s, add_bos=True, add_eos=True) == tok.encode(s, add_bos=True, add_eos=True)


def test_training_deterministic_same_input_same_vocab(tmp_path):
    corpus = tmp_path / "c.txt"
    corpus.write_text(TINY_CORPUS * 50, encoding="utf-8")
    t1 = train_tokenizer([corpus], vocab_size=250, min_frequency=2,
                         out_dir=tmp_path / "a")
    t2 = train_tokenizer([corpus], vocab_size=250, min_frequency=2,
                         out_dir=tmp_path / "b")
    s = "the cat sat on the mat, and the river runs cold."
    assert t1.encode(s) == t2.encode(s)
    assert t1.vocab_size == t2.vocab_size


def test_missing_files_fail_loudly(tmp_path):
    with pytest.raises(ValueError):
        train_tokenizer([], vocab_size=100)
    with pytest.raises(FileNotFoundError):
        train_tokenizer([tmp_path / "nope.txt"], vocab_size=100)
    with pytest.raises(FileNotFoundError):
        BPETokenizer.load(tmp_path / "missing" / "tokenizer.json")
