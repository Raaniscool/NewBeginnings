"""Generation metrics, cloze probes, and the English Mastery Gate logic.

The gate is the project contract: an ABSOLUTE floor version (V1) must never
pass by rule, and later versions must beat floors AND the improvement rule.
These tests pin that behavior so nobody can accidentally widen the door.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from core.config import DecodeConfig, ModelConfig
from core.generate import generate_texts, load_model
from core.evaluate import evaluate_gate, run_cloze
from core.metrics import (aggregate_generation_metrics, double_word_rate,
                          rep_n_rate, tokens_to_first_rep_n)
from core.model import GPT
from core.tokenizer import train_tokenizer
from core.dataset import Document, tokenize_documents
import numpy as np


# ------------------------------------------------------------- metrics -----

def test_rep_metrics_detect_degeneration():
    good = ("The morning was cold and clear. Marta opened the shop early, "
            "swept the floor, and set out the bread while the village slept.")
    bad = "the river and the river and the river and the river and the river and the river and the river"
    assert rep_n_rate(bad, 3) > 0.8
    assert rep_n_rate(good, 3) < 0.05
    assert double_word_rate(bad) > 0.5
    assert double_word_rate(good) == 0.0
    assert tokens_to_first_rep_n(good, 3) > 20
    assert tokens_to_first_rep_n(bad, 3) <= 3


def test_aggregate_metrics_keys_and_ranges():
    texts = ["One two three, four five six.", "A different sentence follows here!",
             "He asked a plain question? She answered it, plainly."]
    m = aggregate_generation_metrics(texts)
    for key in ("rep3_rate", "distinct_1", "distinct_2", "tokens_to_first_rep3_median",
                "double_word_rate", "terminal_punct_rate", "mean_words", "n_texts"):
        assert key in m
    assert 0.0 <= m["rep3_rate"] <= 1.0
    assert 0.0 <= m["distinct_2"] <= 1.0
    assert m["terminal_punct_rate"] == 1.0
    assert m["n_texts"] == 3


# ------------------------------------------------------------ generate -----

@pytest.fixture(scope="module")
def tok_model(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("genfix")
    corpus = tmp / "c.txt"
    corpus.write_text(("the cat sat on the mat. the dog ran in the yard. "
                       "she smiled and waved. he asked a quiet question?\n") * 300)
    tok = train_tokenizer([corpus], vocab_size=200, min_frequency=2,
                          out_dir=tmp / "tk")
    cfg = ModelConfig(vocab_size=tok.vocab_size, block_size=24, n_layer=2,
                      n_head=4, d_model=48, dropout=0.0, bias=False)
    torch.manual_seed(0)
    model = GPT(cfg)
    model.eval()
    return tok, model, tmp


def test_generate_greedy_is_deterministic(tok_model):
    tok, model, _ = tok_model
    d = DecodeConfig(max_new_tokens=12, temperature=0.0, top_k=0, top_p=1.0,
                     repetition_penalty=1.0, seed=1, name="greedy_test")
    a = generate_texts(model, tok, ["the cat"], d)[0]
    b = generate_texts(model, tok, ["the cat"], d)[0]
    assert a == b
    assert len(a) > 0


def test_generate_seed_controlled_sampling(tok_model):
    tok, model, _ = tok_model
    d = DecodeConfig(max_new_tokens=12, temperature=0.9, seed=42)
    a = generate_texts(model, tok, ["the cat"], d)
    b = generate_texts(model, tok, ["the cat"], d)
    assert a == b, "same seed must reproduce sampling (gate comparability)"


def test_checkpoint_roundtrip_via_load_model(tok_model, tmp_path):
    tok, model, _ = tok_model
    ckpt = {
        "model_state": model.state_dict(),
        "model_config": vars(model.cfg),
        "opt_state": {}, "step": 5, "best_val": 1.0,
    }
    p = tmp_path / "m.ckpt"
    torch.save(ckpt, p)
    m2 = load_model(p)
    assert m2.num_params() == model.num_params()
    d = DecodeConfig(max_new_tokens=8, temperature=0.0)
    assert generate_texts(m2, tok, ["the dog"], d) == generate_texts(model, tok, ["the dog"], d)


# -------------------------------------------------------- cloze probes -----

def test_cloze_scoring_runs_and_reports_chance(tok_model, tmp_path):
    tok, model, _ = tok_model
    probes = {
        "probes": [
            {"context": "The cat ",
             "choices": ["sat on the mat.", "quantum entangled mat."],
             "answer_index": 0, "tags": ["agreement"]},
            {"context": "She asked ",
             "choices": ["a question.", "the saw mill."],
             "answer_index": 0, "tags": ["continuity"]},
        ]
    }
    p = tmp_path / "cloze.json"
    p.write_text(json.dumps(probes))
    res = run_cloze(model, tok, p)
    assert res["n"] == 2
    assert 0.0 <= res["accuracy"] <= 1.0
    assert res["chance"] == 0.5
    assert "agreement" in res["per_tag"]


# ---------------------------------------------------------------- gate -----

def _bench_metrics(rep3=0.05, dist2=0.7, worst_rep3=0.08, worst_dist2=0.6):
    cat = {"rep3_rate": rep3, "distinct_2": dist2}
    return {
        "overall": {"rep3_rate": rep3, "distinct_2": dist2,
                    "tokens_to_first_rep3_median": 64,
                    "double_word_rate": 0.002,
                    "terminal_punct_rate": 0.99,
                    "question_mark_rate": 0.9},
        "per_category": {"qa": cat, "stories": {"rep3_rate": worst_rep3,
                                                "distinct_2": worst_dist2}},
    }


GOOD_CLOZE = {"accuracy": 0.9, "n": 40, "chance": 0.5, "per_tag": {}}
GATE_CFG = Path("configs/gate_v1.json")


def test_gate_v1_baseline_can_never_pass(tmp_path):
    res = evaluate_gate("v1", _bench_metrics(), GOOD_CLOZE, GOOD_CLOZE,
                        test_ppl=20.0, gate_cfg_path=GATE_CFG)
    assert res["all_floors_met"] is True     # floors CAN be met...
    assert res["passed"] is False            # ...but V1 must never PASS
    assert "baseline" in res["note"]
    assert "FAIL" in res["verdict"]


def test_gate_floors_fail_loud(tmp_path):
    bad = _bench_metrics(rep3=0.5, worst_rep3=0.6)
    res = evaluate_gate("v2", bad, {"accuracy": 0.4, "n": 10, "chance": 0.5,
                                    "per_tag": {}}, GOOD_CLOZE, test_ppl=90.0,
                        gate_cfg_path=GATE_CFG)
    assert res["passed"] is False
    failed = [n for n, c in res["checks"].items() if not c["pass"]]
    assert {"test_ppl", "grammar_cloze_acc", "rep3_rate",
            "worst_category_rep3"} <= set(failed)


def test_gate_v2_needs_improvement_vs_v1(tmp_path):
    prev = {"version": "v1", "gate": {"checks": {
        n: {"value": v, "floor": f, "pass": True}
        for n, v, f in [
            ("test_ppl", 22.0, 45.0), ("grammar_cloze_acc", 0.82, 0.8),
            ("continuity_cloze_acc", 0.78, 0.75), ("rep3_rate", 0.07, 0.1),
            ("distinct_2", 0.65, 0.6), ("tokens_to_first_rep3_median", 50, 32),
            ("double_word_rate", 0.004, 0.01), ("terminal_punct_rate", 0.96, 0.95),
            ("question_mark_rate", 0.85, 0.8), ("worst_category_rep3", 0.1, 0.15),
            ("worst_category_distinct2", 0.55, 0.5),
        ]}}}
    p = tmp_path / "prev_gate.json"
    p.write_text(json.dumps(prev))
    # improvement on ppl (20 < 22), cloze held (0.9 >= 0.82)
    res = evaluate_gate("v2", _bench_metrics(), GOOD_CLOZE, GOOD_CLOZE,
                        test_ppl=20.0, gate_cfg_path=GATE_CFG,
                        prev_metrics_path=p)
    assert res["improvement"]["applicable"] is True
    assert res["improvement"]["strict_test_ppl_improved"] is True
    # identical ppl must NOT pass the strict rule
    res2 = evaluate_gate("v2", _bench_metrics(), GOOD_CLOZE, GOOD_CLOZE,
                         test_ppl=22.0, gate_cfg_path=GATE_CFG,
                         prev_metrics_path=p)
    assert res2["improvement"]["strict_test_ppl_improved"] is False
    assert res2["passed"] is False


def test_gate_v2_without_prev_cannot_pass(tmp_path):
    res = evaluate_gate("v2", _bench_metrics(), GOOD_CLOZE, GOOD_CLOZE,
                        test_ppl=20.0, gate_cfg_path=GATE_CFG,
                        prev_metrics_path=tmp_path / "nonexistent.json")
    assert res["passed"] is False
    assert res["all_floors_met"] is True
