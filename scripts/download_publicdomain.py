#!/usr/bin/env python3
"""Download and clean a small, FIXED set of public-domain texts.

Why this exists:
  * storage is a first-class constraint -> we fetch tiny tarballs from the
    GITenberg GitHub mirrors of Project Gutenberg, keep ONLY the plain-text
    file, strip the PG header/license boilerplate, and cap each work;
  * legality -> every work here is a Project Gutenberg eBook, i.e. public
    domain in the US; the eBook number is recorded in the sidecar metadata;
  * reproducibility -> the same script with the same WORKS list produces
    identical files. Output lands in data/raw_downloads/ (gitignored; this
    script is the committed record of how to regenerate it).

Output format per work: data/raw_downloads/<slug>.txt where documents are
separated by the standard DOC_SEP, plus <slug>.json metadata.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
import urllib.request
from pathlib import Path

DOC_SEP = "\n===\n"
OUT_DIR = Path("data/raw_downloads")
MAX_CHARS_PER_WORK = 130_000        # hard cap per work (storage budget)

WORKS = [
    # (owner/repo, eBook no., slug, category, title)
    ("GITenberg/Just-So-Stories_32488", 32488, "justso", "stories",
     "Just So Stories (Rudyard Kipling, 1902)"),
    ("GITenberg/Aesop-s-Fables--a-new-translation_11339", 11339, "aesop",
     "stories", "Aesop's Fables (a new translation)"),
    ("GITenberg/The-Wind-in-the-Willows_27805", 27805, "willows", "stories",
     "The Wind in the Willows (Kenneth Grahame, 1908)"),
    ("GITenberg/The-Wonderful-Wizard-of-Oz_55", 55, "oz", "stories",
     "The Wonderful Wizard of Oz (L. Frank Baum, 1900)"),
    ("GITenberg/The-Art-of-War_132", 132, "artofwar", "knowledge",
     "The Art of War (Sun Tzu, tr. Lionel Giles, 1910)"),
    ("GITenberg/The-Elements-of-Style_37134", 37134, "strunk", "formal",
     "The Elements of Style (William Strunk Jr., 1918)"),
]

START_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*\n", re.I)
END_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*", re.I)
ILLUS_RE = re.compile(r"\[(Illustration|Illustration:|Image)[^\]]*\]", re.I)
TRANSCRIBER_RE = re.compile(r"(Transcriber'?s? Notes?:|Etext transcriber)", re.I)
HEADING_RE = re.compile(r"^\s*(CHAPTER\s+[IVXLC0-9]+|[IVXLC]{2,}\.?|Chapter\s+\d+)\s*$")
ALLCAPS_HEADING_RE = re.compile(r"^[A-Z][A-Z ,;:'\-–—.?!ÆSŒ]{3,70}$")

BOILER_DENSE_RE = re.compile(
    r"project gutenberg|gutenberg\.net|pgdp|ebook|produced by|proofreading team"
    r"|copyright|transcriber",
    re.I)

TRANSCRIBER_BLOCK_RE = re.compile(r"Transcriber'?s? Note:?.*?(?=\n\s*\n)", re.I | re.S)


def fetch_text(repo: str) -> str:
    url = f"https://codeload.github.com/{repo}/tar.gz/refs/heads/master"
    req = urllib.request.Request(url, headers={"User-Agent": "NewBeginnings/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        payload = r.read()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tf:
        members = [m for m in tf.getmembers()
                   if m.isfile() and (m.name.endswith("-8.txt") or
                                      (m.name.lower().endswith(".txt")
                                       and "/" not in m.name.split("/", 1)[1]))]
        if not members:
            # fallback: largest .txt anywhere
            members = [m for m in tf.getmembers()
                       if m.isfile() and m.name.lower().endswith(".txt")]
        if not members:
            raise RuntimeError(f"no text file found in {repo}")
        member = max(members, key=lambda m: m.size)
        data = tf.extractfile(member).read()
    return data.decode("utf-8", errors="replace")


def strip_gutenberg_boilerplate(text: str) -> str:
    m = START_RE.search(text)
    if m:
        text = text[m.end():]
    m = END_RE.search(text)
    if m:
        text = text[:m.start()]
    # drop trailing transcriber notes sections
    m = TRANSCRIBER_RE.search(text)
    if m and m.start() > len(text) * 0.5:
        text = text[:m.start()]
    return text


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = ILLUS_RE.sub("", text)
    text = TRANSCRIBER_BLOCK_RE.sub("", text)
    # many editions place "Produced by ..." right after the START marker
    text = re.sub(r"^\s*Produced by[^\n]*(\n[^\n]*){0,3}", "", text, count=1)
    # drop HTML-ish tags that occasionally survive
    text = re.sub(r"<[^>]+>", " ", text)
    # straighten whitespace runs
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_documents(text: str) -> list[str]:
    """Split on chapter/story headings; fall back to paragraph windows."""
    lines = text.split("\n")
    docs, current = [], []

    def flush():
        chunk = "\n".join(current).strip()
        if len(chunk) > 400:
            docs.append(chunk)

    for line in lines:
        stripped = line.strip()
        is_heading = bool(HEADING_RE.match(stripped)) or (
            bool(ALLCAPS_HEADING_RE.match(stripped)) and len(stripped) > 8
            and stripped[0].isupper())
        if is_heading and current and sum(len(x) for x in current) > 400:
            flush()
            current = []
        current.append(line)
    flush()

    if len(docs) < 3:
        # fallback: window by ~450 words at paragraph boundaries
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        docs, buf, n = [], [], 0
        for p in paras:
            buf.append(p)
            n += len(p.split())
            if n >= 450:
                docs.append("\n\n".join(buf))
                buf, n = [], 0
        if buf and sum(len(x) for x in buf) > 400:
            docs.append("\n\n".join(buf))
    return docs


def _looks_like_prose(doc: str) -> bool:
    ws = [w for w in re.findall(r"[A-Za-z']+", doc) if len(w) > 1]
    if not ws:
        return False
    lower = sum(1 for w in ws if any(c.islower() for c in w))
    funcs = sum(1 for w in ws if w.lower() in
                ("the", "of", "and", "a", "to", "in", "is", "it"))
    return (lower / len(ws) > 0.6) and (funcs / len(ws) > 0.04)


def _strip_leading_junk(doc: str) -> str:
    """Remove title-page/contents junk glued to a doc's head: drop every
    leading line until the first prose-looking line."""
    lines = doc.split("\n")
    i = 0
    while i < len(lines):
        ws = re.findall(r"[A-Za-z']+", lines[i])
        if (len(ws) >= 3 and
                sum(1 for w in ws if any(c.islower() for c in w)) / len(ws) > 0.6):
            break  # prose begins here
        i += 1
    return "\n".join(lines[i:]).strip()


def process(raw: str) -> list[str]:
    body = strip_gutenberg_boilerplate(raw)
    body = clean_text(body)
    docs = split_into_documents(body)
    # front matter (title pages, contents, illustration lists) can be glued
    # onto the first real chunk; strip per-document head junk.
    docs = [_strip_leading_junk(d) for d in docs]
    docs = [d for d in docs if len(d) > 300]
    docs = [d for d in docs if len(BOILER_DENSE_RE.findall(d)) < 3
            and _looks_like_prose(d)]
    return docs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for repo, ebook_no, slug, category, title in WORKS:
        print(f"downloading {title} ...")
        try:
            raw = fetch_text(repo)
        except Exception as exc:  # keep going: the slice is deliberately small
            print(f"  SKIPPED {slug}: {exc}")
            continue
        docs = process(raw)
        # cap politely at a document boundary
        total, kept = 0, []
        for d in docs:
            if total + len(d) > MAX_CHARS_PER_WORK:
                break
            kept.append(d)
            total += len(d)
        out = OUT_DIR / f"{slug}.txt"
        out.write_text(DOC_SEP.join(d.strip() for d in kept) + "\n", encoding="utf-8")
        meta = {
            "title": title,
            "ebook_no": ebook_no,
            "source": f"https://github.com/{repo} (Project Gutenberg eBook #{ebook_no})",
            "license": "public domain in the US (Project Gutenberg terms)",
            "category": category,
            "documents": len(kept),
            "chars": total,
        }
        (OUT_DIR / f"{slug}.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"  {slug}: {len(kept)} documents, {total/1000:.0f} kchars")
    print("done. raw_downloads is gitignored; this script is the record.")


if __name__ == "__main__":
    main()
