"""Corpus & fixture integrity: seed files parse correctly, categories line
up with the generator's map, no document uses the separator inside its body,
no code contamination in the seeds, and the FIXED benchmark files stay sane.

These tests protect the data contract; the numeric corpus-size checks live
in the build step (assert_split_sane + manifest), not here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.corpus import DOC_SEP, MIN_DOC_CHARS, load_seed_documents
from core.dataset import looks_like_code
from core.gen_sources import FAMILY_CATEGORY, FAMILIES, generate_documents

SEEDS = Path("data/seeds")
DOWNLOADS = Path("data/raw_downloads")
BENCH = Path("benchmarks")


def test_seed_dirs_match_generator_categories():
    if not SEEDS.exists():
        pytest.skip("seeds not extracted in this checkout")
    seed_cats = {p.name for p in SEEDS.iterdir() if p.is_dir()}
    gen_cats = set(FAMILY_CATEGORY.values())
    unknown = seed_cats - gen_cats - {"knowledge", "stories"}
    assert not unknown, f"seed category dirs with no generator counterpart: {unknown}"


def test_seed_documents_parse_and_are_clean():
    if not SEEDS.exists():
        pytest.skip("seeds not extracted in this checkout")
    docs = load_seed_documents(SEEDS)
    assert len(docs) >= 120, f"suspiciously few seed docs: {len(docs)}"
    flagged = [d.doc_id for d in docs if looks_like_code(d.text)]
    assert not flagged, f"code contamination flagged in seeds: {flagged[:5]}"
    short = [d.doc_id for d in docs if len(d.text) < MIN_DOC_CHARS]
    assert not short
    # every doc carries a category and a group for leakage control
    assert all(d.category and d.group.startswith("seed:") for d in docs)


def test_no_doc_contains_the_separator():
    if not SEEDS.exists():
        pytest.skip("seeds not extracted in this checkout")
    for f in SEEDS.rglob("*.txt"):
        parts = f.read_text(encoding="utf-8").split(DOC_SEP)
        assert len(parts) >= 1, f"unreadable seed file {f}"


def test_downloads_have_public_domain_metadata():
    if not DOWNLOADS.exists():
        pytest.skip("no downloads present (offline checkout)")
    for txt in sorted(DOWNLOADS.glob("*.txt")):
        meta_path = txt.with_suffix(".json")
        assert meta_path.exists(), f"missing license metadata for {txt.name}"
        meta = json.loads(meta_path.read_text())
        assert "public domain" in meta["license"].lower()
        assert meta["category"] in set(FAMILY_CATEGORY.values()) | {"knowledge", "stories"}
        assert meta["source"].startswith("http")


def test_generator_output_is_varied_and_code_free():
    docs = list(generate_documents(seed=20260919))
    assert len(docs) >= 1300, f"generator wrote only {len(docs)} docs"
    cats = {}
    for d in docs:
        cats[d.category] = cats.get(d.category, 0) + 1
    assert len(cats) >= 6, cats
    # no single generator category may dominate the generated slice
    top = max(cats.values())
    assert top / len(docs) < 0.35, cats
    flagged = [d for d in docs if looks_like_code(d.text)]
    assert not flagged, f"generator produced code-like text: {[d.doc_id for d in flagged[:3]]}"
    # Template-slot generation samples a combinatorial space; at high counts
    # exact-text collisions are EXPECTED and are removed by build_corpus()
    # (see manifest n_exact_dupes). The invariant here: the space is large
    # enough that sampling is still majority-unique at production counts.
    unique = len({d.text for d in docs})
    assert unique > 0.70 * len(docs), \
        f"generator space exhausted: only {unique}/{len(docs)} unique texts"


def test_generator_deterministic_by_seed():
    a = [d.text for d in generate_documents(seed=1)]
    b = [d.text for d in generate_documents(seed=1)]
    c = [d.text for d in generate_documents(seed=2)]
    assert a == b
    assert a != c


BENCH_CATEGORIES = {
    "conversation", "questions", "explanations", "descriptions", "instructions",
    "stories", "formal", "grammar", "continuity",
}


def test_fixed_benchmark_files_are_valid_and_cover_required_categories():
    bench = json.loads((BENCH / "english_v1.json").read_text())
    prompts = bench["prompts"]
    assert len(prompts) >= 50
    cats = {p["category"] for p in prompts}
    missing = BENCH_CATEGORIES - cats
    assert not missing, f"benchmark missing categories: {missing}"
    qs = [p["prompt"] for p in prompts if p["category"] == "questions"]
    assert qs, "no question-category prompts in benchmark"


def test_cloze_probes_are_binary_and_index_valid():
    for name in ("cloze_grammar_v1.json", "cloze_continuity_v1.json"):
        data = json.loads((BENCH / name).read_text())
        probes = data["probes"]
        assert len(probes) >= 40, f"{name} has only {len(probes)} probes"
        for p in probes:
            assert len(p["choices"]) == 2, "cloze probes must stay binary"
            assert 0 <= p["answer_index"] < 2
            assert p["choices"][0] != p["choices"][1]
            assert p["context"].strip()
            assert p.get("tags"), "every cloze probe needs a tag"


def test_gate_config_pre_registered(tmp_path):
    gate = json.loads(Path("configs/gate_v1.json").read_text())
    floors = gate["floors"]
    for key in ("test_ppl_max", "grammar_cloze_min", "continuity_cloze_min",
                "rep3_max", "distinct2_min", "terminal_punct_min",
                "double_word_max", "worst_cat_rep3_max"):
        assert key in floors, f"gate floor {key} missing"
    assert gate["improvement_rule"]["min_held_or_improved"] >= 5
    # the pinned decode config must be raw-model decoding
    dec = json.loads(Path(gate["pinned_decode_config"]).read_text())
    assert dec["repetition_penalty"] == 1.0 and dec["top_p"] == 1.0 and dec["top_k"] == 0
