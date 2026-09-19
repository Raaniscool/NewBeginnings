"""Dataset tests: dedupe, contamination screen, group-aware splits (no
leakage possible by construction), loud failures on invalid splits, tensor
batch shapes and the causal shift convention.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.dataset import (Document, TokenDataset, assert_split_sane,
                          dedupe_documents, group_aware_split, looks_like_code,
                          tokenize_documents)
from core.tokenizer import train_tokenizer


def docs_for_split_test(n_groups=30, per=3):
    rng_text = (
        "the quick brown fox jumps over the lazy dog near the river bank in the"
        " early morning light while birds sing in the trees above the meadow"
    ).split()
    out = []
    for g in range(n_groups):
        for i in range(per):
            body = " ".join(rng_text[(g + i + j) % len(rng_text)]
                            for j in range(60))
            out.append(Document(doc_id=f"d{g}_{i}", category="conversation",
                                text=body + f" token{g}{i}", group=f"g{g}",
                                origin="seed"))
    return out


def test_exact_dedupe_normalized():
    a = Document("a", "knowledge", "The Cat  Sat On\nthe MAT.", "g", "seed")
    b = Document("b", "knowledge", "the cat sat on the mat.", "g", "seed")
    c = Document("c", "knowledge", "A completely different sentence lives here.", "g", "seed")
    kept, dropped = dedupe_documents([a, b, c])
    assert dropped == 1 and len(kept) == 2
    assert [d.doc_id for d in kept] == ["a", "c"]


def test_code_contamination_screen():
    assert looks_like_code("def main():\n    print('hi')\n")
    assert looks_like_code("function go() {\n  let x = 1;\n  const y = 2;\n}")
    assert looks_like_code("local part = Instance.new('Part')\npart.Parent = script.Parent\n")
    assert not looks_like_code(
        "The inspector explained the procedure in plain language, and the "
        "whole office breathed a sigh of relief when the letter arrived.")
    assert not looks_like_code(
        "Mix the flour with the water, knead it gently, and then let the "
        "dough rest under a clean towel for an hour.")


def test_group_split_disjoint_by_group_and_deterministic():
    docs = docs_for_split_test()
    s1 = group_aware_split(docs, seed=42, val_frac=0.15, test_frac=0.15)
    s2 = group_aware_split(docs, seed=42, val_frac=0.15, test_frac=0.15)
    # deterministic given the seed
    for name in ("train", "val", "test"):
        assert sorted(d.doc_id for d in s1[name]) == sorted(d.doc_id for d in s2[name])
    groups = {n: {d.group for d in s1[n]} for n in s1}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (groups[a] & groups[b]), f"group leakage: {a} vs {b}"
    total = sum(len(s1[n]) for n in ("train", "val", "test"))
    assert total == len(docs)


def test_split_sanity_fails_loudly_on_tiny_splits():
    docs = docs_for_split_test(n_groups=4, per=2)
    splits = group_aware_split(docs, seed=1, val_frac=0.3, test_frac=0.3)
    with pytest.raises(RuntimeError) as exc:
        assert_split_sane(splits)
    msg = str(exc.value)
    assert "Split sanity check FAILED" in msg


def test_split_sanity_passes_on_sized_corpus():
    docs = docs_for_split_test(n_groups=80, per=3)
    splits = group_aware_split(docs, seed=7, val_frac=0.07, test_frac=0.07)
    assert_split_sane(splits)  # should not raise


@pytest.fixture(scope="module")
def tok_and_data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tdata")
    corpus = tmp / "train.txt"
    sentences = [
        "the cat sat on the mat and purred softly in the warm sun",
        "a young shepherd counted his sheep every morning at dawn",
        "the baker opened his shop before the village was awake",
        "cold rain drummed on the roof of the little stone cottage",
    ]
    corpus.write_text(".\n".join(sentences) * 400 + ".\n")
    tok = train_tokenizer([corpus], vocab_size=200, min_frequency=2,
                          out_dir=tmp / "tk")
    return tok, tmp, sentences


def test_tokenize_documents_layout(tok_and_data):
    tok, _, sentences = tok_and_data
    docs = [Document(f"d{i}", "stories", s + ".", "g", "generated")
            for i, s in enumerate(sentences)]
    ids = tokenize_documents(docs, tok)
    assert ids.dtype == np.uint16
    # each doc = bos ... eos
    assert (ids == tok.bos_id).sum() >= len(docs)
    assert (ids == tok.eos_id).sum() >= len(docs)


def test_batch_shapes_and_causal_shift(tok_and_data):
    tok, tmp, sentences = tok_and_data
    docs = [Document("x", "stories", (" ".join(sentences) + ".") * 30, "g",
                     "generated")]
    ids = tokenize_documents(docs, tok)
    np.save(tmp / "train.npy", ids)
    np.save(tmp / "val.npy", ids)
    np.save(tmp / "test.npy", ids)
    ds = TokenDataset(tmp)
    x, y = ds.get_batch("train", batch_size=8, block_size=24, device="cpu")
    assert x.shape == (8, 24) and y.shape == (8, 24)
    # y must be exactly x shifted by one against the TRUE token stream
    data = ds.splits["train"]
    arrx, arry = x.numpy(), y.numpy()
    for b in range(8):
        # find the window in the raw array by matching the first 8 tokens
        starts = [s for s in range(len(data) - 9)
                  if (data[s:s + 8] == arrx[b, :8]).all()]
        assert starts, "sampled window not found in source array"
        s = starts[0]
        assert (data[s:s + 24] == arrx[b]).all()
        assert (data[s + 1:s + 25] == arry[b]).all()


def test_get_batch_respects_generator_seeding(tok_and_data):
    import torch
    tok, tmp, sentences = tok_and_data
    docs = [Document("x", "stories", (" ".join(sentences) + ".") * 30, "g",
                     "generated")]
    ids = tokenize_documents(docs, tok)
    for name in ("train", "val", "test"):
        np.save(tmp / f"{name}.npy", ids)
    ds = TokenDataset(tmp)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    x1, _ = ds.get_batch("train", 4, 16, generator=g1)
    x2, _ = ds.get_batch("train", 4, 16, generator=g2)
    assert torch.equal(x1, x2)


def test_missing_tokenized_dir_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        TokenDataset(tmp_path)
    assert "build-dataset" in str(exc.value)
