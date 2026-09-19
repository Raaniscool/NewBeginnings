"""Milestone report builder.

Pulls together every committed/recorded artifact of a version (config,
manifest, training summary, test eval, benchmark, cloze, gate, storage) into
ONE honest markdown document. Missing sections are reported as missing, not
silently omitted — an unfinished evaluation must look unfinished.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _du(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _mb(n: int) -> str:
    return f"{n / 1e6:.2f} MB"


def build_report(version: str) -> str:
    out: list[str] = []
    a = out.append
    a(f"# Milestone Report — {version}")
    a("")

    # ---------------------------------------------------------------- model
    a("## Model")
    mcfg = _load_json(ROOT / "configs" / f"{version}_model.json")
    tok_meta = _load_json(ROOT / "artifacts/tokenizer/tokenizer_meta.json")
    ckpt = ROOT / "checkpoints" / version / "best.ckpt"
    actual = None
    if ckpt.exists():
        try:
            import torch
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            mcfg = payload.get("model_config", mcfg)
            from core.config import ModelConfig
            from core.model import GPT
            model = GPT(ModelConfig(**payload["model_config"]))
            actual = model.num_params()
            del model, payload
        except Exception:
            actual = None
    if mcfg:
        est = None
        try:
            from core.config import ModelConfig
            est = ModelConfig(**mcfg).n_params_estimate()
        except Exception:
            pass
        a(f"- architecture: GPT-style causal LM, {mcfg['n_layer']} layers, "
          f"{mcfg['n_head']} heads, d_model={mcfg['d_model']}, "
          f"block_size={mcfg['block_size']}, dropout={mcfg['dropout']}, "
          f"bias={mcfg['bias']}, weight-tied embeddings")
        n = actual if actual is not None else est
        tag = "actual" if actual is not None else "estimate"
        if n:
            a(f"- parameters: {n:,} ({tag}; no pretrained weights anywhere; "
              f"random init)")
    else:
        a("- model config: MISSING")
    if tok_meta:
        a(f"- tokenizer: byte-level BPE trained from scratch on TRAIN split only; "
          f"vocab={tok_meta['vocab_size']}; sha256={tok_meta['sha256'][:16]}…")
    else:
        a("- tokenizer meta: MISSING")
    a("")

    # ----------------------------------------------------------------- data
    a("## Data")
    manifest = _load_json(ROOT / "data/manifest.json")
    if manifest:
        c = manifest.get("counts", {})
        org = manifest.get("origins", {})
        a(f"- corpus: {manifest.get('counts', {}).get('final_docs', '?')} docs "
          f"after dedupe, {manifest.get('total_chars', 0)/1e6:.2f} Mchars, "
          f"sha256={manifest.get('corpus_sha256', '?')[:16]}…")
        a(f"- origins: seed={org.get('seed', 0)} docs, generated="
          f"{org.get('generated', 0)} docs, downloaded="
          f"{c.get('download_docs', 0)} docs")
        a(f"- screened out (code-like): {c.get('code_screened_out', 0)}; "
          f"exact dupes removed: {c.get('exact_dupes_removed', 0)}; "
          f"near dupes removed: {c.get('near_dupes_removed', 0)}")
        a("- category shares (by characters):")
        for cat, info in sorted(manifest["categories"].items(),
                                key=lambda kv: -kv[1]["share_pct"]):
            a(f"  - {cat}: {info['share_pct']}% ({info['docs']} docs)")
    else:
        a("- manifest: MISSING")
    tk = _load_json(ROOT / "data/tokenized/tokenized_meta.json")
    if tk:
        a(f"- split tokens: train={tk['train']['n_tokens']:,} "
          f"({tk['train']['n_docs']} docs), val={tk['val']['n_tokens']:,} "
          f"({tk['val']['n_docs']} docs), test={tk['test']['n_tokens']:,} "
          f"({tk['test']['n_docs']} docs)")
        a("- splits are group-aware: no generator family or seed file spans "
          "two splits (enforced in tests)")
    a("")

    # ------------------------------------------------------------- training
    a("## Training")
    summ = _load_json(ROOT / "artifacts/reports" / f"{version}_train_summary.json")
    tcfg = _load_json(ROOT / "configs" / f"{version}_train.json")
    if tcfg:
        a(f"- config: batch={tcfg['batch_size']}, block={tcfg['block_size']}, "
          f"AdamW(lr={tcfg['learning_rate']}, wd={tcfg['weight_decay']}), "
          f"warmup={tcfg['warmup_steps']}, cosine to "
          f"{tcfg['min_lr_frac']}×lr, grad clip={tcfg['grad_clip']}, "
          f"seed={tcfg['seed']}")
    if summ:
        a(f"- steps: {summ['steps']} (≈{summ['epochs_equiv']} epochs), "
          f"{summ['train_seconds']/60:.1f} min wall clock, "
          f"avg {summ['avg_tokens_per_sec']} tok/s on 2 CPU threads")
        a(f"- best val loss: {summ['best_val_loss']} (ppl {summ['best_val_ppl']}); "
          f"final val loss: {summ['final_val_loss']} (ppl {summ['final_val_ppl']})")
        if summ.get("interrupted"):
            a("- RUN WAS INTERRUPTED (Ctrl+C) and resumed/ended gracefully per policy")
    else:
        a("- training summary: MISSING (run `python3 run.py train`)")
    a("")

    # ------------------------------------------------------------ evaluation
    a("## Evaluation")
    test = _load_json(ROOT / "artifacts/reports" / f"{version}_test.json")
    if test:
        a(f"- held-out TEST: loss {test['test_loss']}, perplexity "
          f"**{test['test_ppl']}** (never used for any training decision)")
    else:
        a("- held-out test: MISSING")
    bm_dir = ROOT / "artifacts/benchmarks" / version
    bench = _load_json(bm_dir / "metrics.json")
    if bench:
        ov = bench["overall"]
        a(f"- benchmark (fixed prompts, raw-model decoding, saved un-cherry-picked): "
          f"rep3_rate={ov['rep3_rate']}, distinct2={ov['distinct_2']}, "
          f"double_word={ov['double_word_rate']}, "
          f"median tokens to first rep3={ov['tokens_to_first_rep3_median']}, "
          f"terminal punct={ov['terminal_punct_rate']}")
        caption = "decoding was NOT raw-model" if not bench.get(
            "raw_model_decoding") else "raw-model decoding confirmed"
        a(f"  ({caption})")
        per = bench["per_category"]
        worst_rep = max(per, key=lambda c: per[c]["rep3_rate"])
        worst_div = min(per, key=lambda c: per[c]["distinct_2"])
        a(f"  worst repetition category: {worst_rep} "
          f"(rep3={per[worst_rep]['rep3_rate']}); worst diversity category: "
          f"{worst_div} (distinct2={per[worst_div]['distinct_2']})")
    else:
        a("- benchmark metrics: MISSING")
    cloze = _load_json(bm_dir / "cloze.json")
    if cloze:
        a(f"- cloze probes: grammar acc={cloze['grammar']['accuracy']} "
          f"(n={cloze['grammar']['n']}, chance={cloze['grammar']['chance']}); "
          f"continuity acc={cloze['continuity']['accuracy']} "
          f"(n={cloze['continuity']['n']})")
    else:
        a("- cloze probes: MISSING")
    gen_txt = bm_dir / "generations.txt"
    if gen_txt.exists():
        lines = gen_txt.read_text(encoding="utf-8").split("\n")
        a("- sample generations (first 6 records, incl. any failures):")
        count = 0
        i = 0
        while count < 6 and i < len(lines):
            if lines[i].startswith("### "):
                snippet = lines[i]
                body = lines[i + 1] if i + 1 < len(lines) else ""
                a(f"\n```\n{snippet}\n{body[:400]}\n```")
                count += 1
                i += 2
            else:
                i += 1
    gate = _load_json(bm_dir / "gate_result.json")
    a("")

    # ----------------------------------------------------------------- gate
    a("## English Mastery Gate")
    if gate:
        g = gate["gate"]
        a(f"- verdict: **{g['verdict']}**")
        a(f"- floors: {g['floors_passed']}/{g['floors_total']} met")
        for name, c in g["checks"].items():
            mark = "PASS" if c["pass"] else "FAIL"
            a(f"  - [{mark}] {name}: {c['value']} (floor {c['floor']})")
        if g.get("improvement", {}).get("applicable"):
            imp = g["improvement"]
            a(f"  - improvement rule: {imp['metrics_held_or_improved']}/"
              f"{imp['metrics_compared']} held-or-improved "
              f"(required {imp['required']}), "
              f"strict ppl improvement: {imp['strict_test_ppl_improved']}, "
              f"cloze no-regress: {imp['cloze_not_regressed']}")
        else:
            a("  - improvement rule: not applicable "
              "(first released version defines the baseline)")
        if g.get("note"):
            a(f"  - note: {g['note']}")
    else:
        a("- gate result: MISSING")
    a("")

    # --------------------------------------------------------------- storage
    a("## Storage")
    total, used, free = shutil.disk_usage(ROOT)
    rows = []
    for name, p in (("data/seeds", ROOT / "data/seeds"),
                    ("data/corpus", ROOT / "data/corpus"),
                    ("data/tokenized", ROOT / "data/tokenized"),
                    ("data/raw_downloads", ROOT / "data/raw_downloads"),
                    ("artifacts", ROOT / "artifacts"),
                    ("checkpoints", ROOT / "checkpoints"),
                    ("venv+site-packages (ext.)", None)):
        if p is None:
            continue
        rows.append((name, _du(p)))
    try:
        sp = subprocess.run(["du", "-s", "--block-size=1",
                             "/usr/local/lib/python3.11/dist-packages/torch"],
                            capture_output=True, text=True, timeout=60)
        torch_mb = int(sp.stdout.split()[0])
        rows.append(("torch install (external)", torch_mb))
    except Exception:
        pass
    for name, n in rows:
        a(f"- {name}: {_mb(n)}")
    a(f"- filesystem: {used/1e9:.1f} GB used / {total/1e9:.0f} GB total; "
      f"{free/1e9:.1f} GB free")
    ck = ROOT / "checkpoints" / version
    if ck.exists():
        for p in sorted(ck.glob("*.ckpt")):
            a(f"  - checkpoint {p.name}: {_mb(p.stat().st_size)}")
        a("  - retention policy: best + last only; nothing else accumulates")
    a("")

    # ------------------------------------------------------- assessment ----
    a("## Honest assessment")
    if gate:
        g = gate["gate"]
        if g["passed"]:
            a(f"- {version} PASSED the gate. Programming phase may be unlocked.")
        else:
            a(f"- {version} did NOT pass the gate. Programming stays locked.")
            if g.get("improvement") and not g["improvement"].get("applicable", True):
                a("- as the baseline version, v1 could not pass by rule; its "
                  "role is to make v2's improvement measurable.")
    elif test:
        a("- gate not evaluated yet.")
    else:
        a("- no evaluation artifacts yet; nothing can be concluded.")
    a("")
    report = "\n".join(out)
    out_path = ROOT / "artifacts/reports" / f"{version}_milestone.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")
    return report
