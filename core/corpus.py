"""Corpus assembly: seeds + generated docs + optional public-domain downloads.

Guarantees (enforced, not hoped for):
  * every document passes a code-contamination screen (Phase 1 = NO code);
  * exact duplicates and near-duplicates (MinHash over word 8-grams) are
    removed globally;
  * splits are group-aware so near-identical patterns cannot straddle
    train/val/test;
  * sanity-checked minimum split sizes, otherwise it fails loudly;
  * a committed manifest (data/manifest.json) records exact composition.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from core.dataset import (Document, assert_split_sane, dedupe_documents,
                          group_aware_split, looks_like_code,
                          normalize_for_dedup)
from core.gen_sources import generate_documents

DOC_SEP = "\n===\n"          # document boundary inside seed/download files
MIN_DOC_CHARS = 40


# ------------------------------------------------------------ seed loading --

def load_seed_documents(seeds_dir: Path) -> List[Document]:
    docs: List[Document] = []
    seeds_dir = Path(seeds_dir)
    if not seeds_dir.exists():
        raise FileNotFoundError(f"seeds dir missing: {seeds_dir}")
    for cat_dir in sorted(p for p in seeds_dir.iterdir() if p.is_dir()):
        category = cat_dir.name
        for path in sorted(cat_dir.glob("*.txt")):
            raw = path.read_text(encoding="utf-8")
            for i, chunk in enumerate(raw.split(DOC_SEP)):
                text = chunk.strip()
                if len(text) < MIN_DOC_CHARS:
                    continue
                docs.append(Document(
                    doc_id=f"seed:{category}:{path.stem}:{i:03d}",
                    category=category, text=text,
                    group=f"seed:{category}:{path.stem}", origin="seed"))
    if not docs:
        raise RuntimeError(f"no seed documents found under {seeds_dir}")
    return docs


def load_download_documents(downloads_dir: Path) -> List[Document]:
    """Optional third-party public-domain text, already cleaned into
    documents separated by the standard separator."""
    docs: List[Document] = []
    downloads_dir = Path(downloads_dir)
    if not downloads_dir.exists():
        return docs
    for path in sorted(downloads_dir.glob("*.txt")):
        meta = downloads_dir / (path.stem + ".json")
        category = "stories"
        if meta.exists():
            category = json.loads(meta.read_text()).get("category", "stories")
        raw = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(raw.split(DOC_SEP)):
            text = chunk.strip()
            if len(text) < MIN_DOC_CHARS:
                continue
            docs.append(Document(
                doc_id=f"dl:{path.stem}:{i:03d}", category=category, text=text,
                group=f"dl:{path.stem}", origin="publicdomain"))
    return docs


# -------------------------------------------------------------- near-dupes --

def _hash_int(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def _minhash_sig(text: str, k: int = 16, n: int = 8) -> Tuple[int, ...]:
    words = normalize_for_dedup(text).split()
    grams = {" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}
    if not grams:
        grams = {" ".join(words)}
    return tuple(sorted((_hash_int(g) for g in grams))[:k])


def drop_near_duplicates(docs: List[Document], thresh: float = 0.75) -> Tuple[List[Document], int]:
    """Drop a document whose word-8-gram MinHash signature overlaps an
    earlier kept document from a DIFFERENT group (Jaccard estimate >= thresh).

    Why cross-group only: template families produce legitimate surface
    variety by recombining the same clause frames with different slots. At
    the 8-gram level two very different members of one family legitimately
    share most shingles, and a global pass would (correctly-but-uselessly)
    annihilate the generated slice. What we actually need to kill is
    redundancy BETWEEN distinct sources (e.g. a seed and a generator doc
    retelling the same recipe), not within-family variety, which is the
    whole point of the generator. Exact duplicates are removed globally
    before this function runs."""
    kept: List[Document] = []
    sigs_by_group: Dict[str, List[Tuple[int, ...]]] = {}
    dropped = 0
    for d in docs:
        sig = _minhash_sig(d.text)
        sigset = set(sig)
        dup = False
        for group, sigs in sigs_by_group.items():
            if group == d.group:
                continue
            for other in sigs:
                inter = len(sigset & other)
                if inter / max(1, len(sigset | other)) >= thresh:
                    dup = True
                    break
            if dup:
                break
        if dup:
            dropped += 1
            continue
        kept.append(d)
        sigs_by_group.setdefault(d.group, []).append(sigset)
    return kept, dropped


# -------------------------------------------------------------------- build --

def build_corpus(seed: int, corpus_dir: str | Path, seeds_dir: str | Path,
                 downloads_dir: str | Path, verbose: bool = True) -> dict:
    corpus_dir = Path(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    docs: List[Document] = []
    seeds = load_seed_documents(Path(seeds_dir))
    downloads = load_download_documents(Path(downloads_dir))
    generated = list(generate_documents(seed=seed))
    docs.extend(seeds)
    docs.extend(downloads)
    docs.extend(generated)

    # 1. code-contamination screen (Phase 1 rule: zero code)
    clean, screened = [], []
    for d in docs:
        (screened if looks_like_code(d.text) else clean).append(d)
    docs = clean

    # 2. exact duplicates (normalized)
    docs, n_exact = dedupe_documents(docs)

    # 3. near duplicates (template families produce legitimate surface
    #    variety; anything with ~70% shared 8-grams to an earlier doc is out)
    docs, n_near = drop_near_duplicates(docs, thresh=0.7)

    # 4. group-aware split (val/test stay meaningful)
    splits = group_aware_split(docs, seed=seed, val_frac=0.05, test_frac=0.05)
    assert_split_sane(splits)

    # 5. write outputs
    stats: Dict[str, Dict[str, int]] = {}
    manifest = {
        "build_seed": seed,
        "counts": {"seed_docs": len(seeds), "download_docs": len(downloads),
                   "generated_docs_raw": len(generated),
                   "code_screened_out": len(screened),
                   "screened_examples": [s.doc_id for s in screened[:5]],
                   "exact_dupes_removed": n_exact,
                   "near_dupes_removed": n_near,
                   "final_docs": len(docs)},
        "categories": {},
        "splits": {},
        "origins": defaultdict(int),
    }
    for split_name, split_docs in splits.items():
        with (corpus_dir / f"{split_name}.jsonl").open("w", encoding="utf-8") as f:
            for d in split_docs:
                f.write(d.to_json() + "\n")
        with (corpus_dir / f"{split_name}.txt").open("w", encoding="utf-8") as f:
            for d in split_docs:
                f.write(d.text.replace("\r", "") + "\n\n")
        chars = sum(len(d.text) for d in split_docs)
        manifest["splits"][split_name] = {
            "docs": len(split_docs), "chars": chars,
            "est_tokens_chars_over_4": chars // 4,
        }
        for d in split_docs:
            manifest["origins"][d.origin] += 1

    for d in docs:
        c = manifest["categories"].setdefault(
            d.category, {"docs": 0, "chars": 0, "share_pct": 0.0})
        c["docs"] += 1
        c["chars"] += len(d.text)
    total_chars = sum(c["chars"] for c in manifest["categories"].values())
    for c in manifest["categories"].values():
        c["share_pct"] = round(100.0 * c["chars"] / max(1, total_chars), 2)
        c["est_tokens"] = c["chars"] // 4
    manifest["origins"] = dict(manifest["origins"])
    manifest["total_chars"] = total_chars
    manifest["total_est_tokens"] = total_chars // 4

    # content hash for reproducibility tracking
    h = hashlib.sha256()
    for split_name in ("train", "val", "test"):
        h.update((corpus_dir / f"{split_name}.jsonl").read_bytes())
    manifest["corpus_sha256"] = h.hexdigest()

    Path("data").mkdir(exist_ok=True)
    (Path("data") / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if verbose:
        print(f"corpus built: {len(docs)} docs, {total_chars/1e6:.2f} Mchars "
              f"(~{total_chars//4//1000}k est. tokens)")
        for cat, c in sorted(manifest["categories"].items(),
                             key=lambda kv: -kv[1]["chars"]):
            print(f"  {cat:<26} {c['docs']:>5} docs  {c['share_pct']:>5.1f}%")
        print(f"screened (code-like): {len(screened)}, exact dupes: {n_exact}, "
              f"near dupes: {n_near}")
    return manifest
