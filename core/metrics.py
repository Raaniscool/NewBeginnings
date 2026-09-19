"""Repetition / diversity / coherence-proxy metrics for generated text.

These are heuristic proxies, and that is their role: they catch the classic
small-model failure modes (degenerate loops, echoing, sentence-boundary
blindness) quantitatively and comparably across model versions. They are
reported SEPARATELY from training metrics, and gate-relevant values are
computed only with the pinned raw-model decoding configuration.
"""

from __future__ import annotations

import re
import statistics
from typing import Dict, Iterable, List, Sequence

WORD_RE = re.compile(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]")
SENT_END = frozenset(".!?")

BANAL_FILLER = {"", " "}


def words(text: str) -> List[str]:
    return WORD_RE.findall(text)


def ngrams(items: Sequence[str], n: int) -> List[tuple]:
    return [tuple(items[i:i + n]) for i in range(len(items) - n + 1)]


def distinct_n(texts: Iterable[str], n: int) -> float:
    total, unique = 0, set()
    for t in texts:
        grams = ngrams([w.lower() for w in words(t)], n)
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0


def rep_n_rate(text: str, n: int = 3) -> float:
    """Fraction of n-grams in `text` that already occurred earlier in it."""
    grams = ngrams([w.lower() for w in words(text)], n)
    seen = set()
    repeats = 0
    for g in grams:
        if g in seen:
            repeats += 1
        else:
            seen.add(g)
    return repeats / len(grams) if grams else 0.0


def tokens_to_first_rep_n(text: str, n: int = 3) -> int:
    """Words produced before the first n-gram repeats (len if never)."""
    toks = [w.lower() for w in words(text)]
    seen = set()
    for i in range(n, len(toks) + 1):
        g = tuple(toks[i - n:i])
        if g in seen:
            return i - 1
        seen.add(g)
    return len(toks)


def double_word_rate(text: str) -> float:
    toks = [w.lower() for w in words(text) if w.strip()]
    if len(toks) < 2:
        return 0.0
    doubles = sum(1 for a, b in zip(toks, toks[1:]) if a == b and a.isalpha())
    return doubles / (len(toks) - 1)


def has_terminal_punct(text: str) -> bool:
    return any(c in SENT_END for c in text.strip())


def aggregate_generation_metrics(texts: Sequence[str]) -> Dict[str, float]:
    if not texts:
        raise ValueError("aggregate_generation_metrics: no texts")
    rep3 = [rep_n_rate(t, 3) for t in texts]
    t_first = [tokens_to_first_rep_n(t, 3) for t in texts]
    dbl = [double_word_rate(t) for t in texts]
    term = [has_terminal_punct(t) for t in texts]
    return {
        "rep3_rate": round(statistics.mean(rep3), 4),
        "distinct_1": round(distinct_n(texts, 1), 4),
        "distinct_2": round(distinct_n(texts, 2), 4),
        "tokens_to_first_rep3_median": statistics.median(t_first),
        "double_word_rate": round(statistics.mean(dbl), 5),
        "terminal_punct_rate": round(sum(term) / len(term), 4),
        "mean_words": round(statistics.mean(len(words(t)) for t in texts), 1),
        "n_texts": len(texts),
    }
