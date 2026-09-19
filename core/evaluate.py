"""Evaluation: fixed benchmark suite, cloze probes, and the Mastery Gate.

The gate evaluation never *declares* mastery; it measures against
pre-registered floors (configs/gate_v1.json) plus the improvement rule
versus the previous released version, and records the result.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch

from core.config import DecodeConfig
from core.generate import generate_texts
from core.metrics import aggregate_generation_metrics
from core.model import GPT
from core.tokenizer import BPETokenizer

BENCH_DIR = Path("benchmarks")
ARTIFACT_BENCH = Path("artifacts/benchmarks")


def load_benchmark(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not data.get("prompts"):
        raise ValueError(f"benchmark file {path} has no prompts")
    return data


def run_benchmark(
    model: GPT,
    tok: BPETokenizer,
    bench_path: str | Path,
    decode: DecodeConfig,
    out_dir: str | Path,
    device: str = "cpu",
) -> dict:
    """Run the FIXED generation benchmark; save outputs (never cherry-picked)
    and per-category + overall metrics."""
    bench = load_benchmark(bench_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    categories: Dict[str, List[str]] = {}
    lines: List[str] = []
    gen_records = []
    prompts_by_cat: Dict[str, List[str]] = {}
    for item in bench["prompts"]:
        prompts_by_cat.setdefault(item["category"], []).append(item["prompt"])

    for category, prompts in prompts_by_cat.items():
        texts = generate_texts(model, tok, prompts, decode, device=device)
        categories[category] = texts
        for prompt, text in zip(prompts, texts):
            gen_records.append({"category": category, "prompt": prompt,
                                "continuation": text})
            lines.append(f"### [{category}] {prompt!r}\n{text}\n")

    per_cat = {}
    for category, texts in categories.items():
        m = aggregate_generation_metrics(texts)
        if category == "questions":
            q = [1 for t in texts if "?" in t]
            m["question_mark_rate"] = round(len(q) / len(texts), 4)
        per_cat[category] = m

    all_texts = [t for texts in categories.values() for t in texts]
    overall = aggregate_generation_metrics(all_texts)
    overall["question_mark_rate"] = per_cat.get("questions", {}).get("question_mark_rate", 0.0)

    (out / "generations.txt").write_text("\n".join(lines), encoding="utf-8")
    (out / "generations.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in gen_records) + "\n",
        encoding="utf-8")
    metrics = {"overall": overall, "per_category": per_cat,
               "decode": vars(decode), "benchmark": str(bench_path),
               "raw_model_decoding": decode.is_raw_model()}
    return metrics


@torch.no_grad()
def run_cloze(model: GPT, tok: BPETokenizer, cloze_path: str | Path,
              device: str = "cpu", limit: int = 0) -> dict:
    """Multiple-choice next-text probes. Score = sum of token log-probs of the
    continuation given the context. Chance = 1/n_choices."""
    with Path(cloze_path).open("r", encoding="utf-8") as f:
        items = json.load(f)["probes"]
    if limit:
        items = items[:limit]
    correct = 0
    per_tag: Dict[str, List[bool]] = {}
    for item in items:
        ctx = item["context"]
        choice_scores = []
        ctx_ids = tok.encode(ctx)
        for choice in item["choices"]:
            choice_ids = tok.encode(choice)
            full_ids = ctx_ids + choice_ids
            boundary = len(ctx_ids)
            # if too long, drop context from the left (keep the choice intact)
            overflow = len(full_ids) - model.cfg.block_size
            if overflow > 0:
                full_ids = full_ids[overflow:]
                boundary = max(0, boundary - overflow)
            x = torch.tensor([full_ids], dtype=torch.long, device=device)
            logits, _ = model(x)
            logp = torch.log_softmax(logits.float(), dim=-1)
            score = 0.0
            for pos in range(boundary, len(full_ids)):
                score += float(logp[0, pos - 1, full_ids[pos]])
            choice_scores.append(score / max(1, (len(full_ids) - boundary)))
        pred = max(range(len(choice_scores)), key=lambda i: choice_scores[i])
        ok = pred == item["answer_index"]
        correct += ok
        for tag in item.get("tags", ["all"]):
            per_tag.setdefault(tag, []).append(ok)
    n = len(items)
    return {
        "accuracy": round(correct / n, 4),
        "n": n,
        "per_tag": {t: round(sum(v) / len(v), 4) for t, v in per_tag.items()},
        "chance": round(1.0 / max(2, len(items[0]["choices"])), 4),
    }


# ------------------------------------------------------------------ gate --

def evaluate_gate(
    version: str,
    bench_metrics: dict,
    cloze_grammar: dict,
    cloze_continuity: dict,
    test_ppl: float,
    gate_cfg_path: str | Path,
    prev_metrics_path: Optional[str | Path] = None,
) -> dict:
    with Path(gate_cfg_path).open("r", encoding="utf-8") as f:
        gate = json.load(f)
    floors = gate["floors"]

    overall = bench_metrics["overall"]
    per_cat = bench_metrics["per_category"]
    worst_rep3 = max(m["rep3_rate"] for m in per_cat.values())
    worst_dist2 = min(m["distinct_2"] for m in per_cat.values())

    checks = {
        "test_ppl": {"value": round(test_ppl, 2), "floor": floors["test_ppl_max"],
                     "pass": test_ppl <= floors["test_ppl_max"]},
        "grammar_cloze_acc": {"value": cloze_grammar["accuracy"], "floor": floors["grammar_cloze_min"],
                              "pass": cloze_grammar["accuracy"] >= floors["grammar_cloze_min"]},
        "continuity_cloze_acc": {"value": cloze_continuity["accuracy"], "floor": floors["continuity_cloze_min"],
                                 "pass": cloze_continuity["accuracy"] >= floors["continuity_cloze_min"]},
        "rep3_rate": {"value": overall["rep3_rate"], "floor": floors["rep3_max"],
                      "pass": overall["rep3_rate"] <= floors["rep3_max"]},
        "distinct_2": {"value": overall["distinct_2"], "floor": floors["distinct2_min"],
                       "pass": overall["distinct_2"] >= floors["distinct2_min"]},
        "tokens_to_first_rep3": {"value": overall["tokens_to_first_rep3_median"],
                                 "floor": floors["first_rep3_min"],
                                 "pass": overall["tokens_to_first_rep3_median"] >= floors["first_rep3_min"]},
        "double_word_rate": {"value": overall["double_word_rate"], "floor": floors["double_word_max"],
                             "pass": overall["double_word_rate"] <= floors["double_word_max"]},
        "terminal_punct_rate": {"value": overall["terminal_punct_rate"], "floor": floors["terminal_punct_min"],
                                "pass": overall["terminal_punct_rate"] >= floors["terminal_punct_min"]},
        "question_mark_rate": {"value": overall.get("question_mark_rate", 0.0),
                               "floor": floors["question_mark_min"],
                               "pass": overall.get("question_mark_rate", 0.0) >= floors["question_mark_min"]},
        "worst_category_rep3": {"value": worst_rep3, "floor": floors["worst_cat_rep3_max"],
                                "pass": worst_rep3 <= floors["worst_cat_rep3_max"]},
        "worst_category_distinct2": {"value": worst_dist2, "floor": floors["worst_cat_distinct2_min"],
                                     "pass": worst_dist2 >= floors["worst_cat_distinct2_min"]},
    }

    floors_passed = sum(1 for c in checks.values() if c["pass"])
    all_floors = floors_passed == len(checks)

    improvement = {"applicable": False, "reason": "first released version defines the baseline"}
    if prev_metrics_path and Path(prev_metrics_path).exists():
        prev = json.loads(Path(prev_metrics_path).read_text())
        prev_checks_version = prev.get("gate", {}).get("checks", {})
        better_or_held = 0
        compared = 0
        strict_ppl = strict_cloze = True
        for name, cur in checks.items():
            old = prev_checks_version.get(name, {}).get("value")
            if old is None:
                continue
            compared += 1
            higher_better = name not in ("test_ppl", "rep3_rate", "double_word_rate",
                                         "worst_category_rep3")
            if higher_better:
                ok = cur["value"] >= old * 0.97
            else:
                ok = cur["value"] <= old * 1.03 if old else True
            better_or_held += ok
            if name == "test_ppl" and not (cur["value"] < old):
                strict_ppl = False
            if name in ("grammar_cloze_acc", "continuity_cloze_acc") and (cur["value"] < old - 0.01):
                strict_cloze = False
        improvement = {
            "applicable": True,
            "metrics_held_or_improved": better_or_held,
            "metrics_compared": compared,
            "required": gate["improvement_rule"]["min_held_or_improved"],
            "strict_test_ppl_improved": strict_ppl,
            "cloze_not_regressed": strict_cloze,
            "pass": (better_or_held >= gate["improvement_rule"]["min_held_or_improved"]
                     and strict_ppl and strict_cloze),
        }

    is_v1 = version.lower().endswith("v1")
    passed = all_floors and (improvement["applicable"] and improvement["pass"]) and not is_v1
    result = {
        "version": version,
        "passed": passed,
        "floors_passed": floors_passed,
        "floors_total": len(checks),
        "all_floors_met": all_floors,
        "checks": checks,
        "improvement": improvement,
        "note": ("v1 is the baseline version and cannot pass the gate by rule."
                 if is_v1 else ""),
        "verdict": "PASS — programming phase may be unlocked" if passed
                   else "FAIL — Phase 1/2 continues; programming remains locked",
    }
    return result
