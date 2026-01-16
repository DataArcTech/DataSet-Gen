from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


_EN_STOP = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "which",
    "who",
    "why",
    "with",
}

_TOKEN_RE = re.compile(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{2,}|\d+(?:\.\d+)?")


def _tokens(text: str, *, max_terms: int = 48) -> List[str]:
    out: List[str] = []
    for m in _TOKEN_RE.finditer(str(text or "").lower()):
        t = m.group(0).strip()
        if not t:
            continue
        if t in _EN_STOP:
            continue
        out.append(t)
        if len(out) >= max_terms:
            break
    # add a small number of bigrams to improve near-duplicate sensitivity
    bigrams: List[str] = []
    for i in range(len(out) - 1):
        bg = out[i] + "_" + out[i + 1]
        bigrams.append(bg)
        if len(bigrams) >= 16:
            break
    return out + bigrams


def _hash64(token: str) -> int:
    # stable 64-bit hash
    digest = hashlib.md5(token.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def simhash64(text: str) -> int:
    """
    Compute a 64-bit SimHash over word tokens (and a few bigrams).
    """
    vec = [0] * 64
    for t in _tokens(text):
        h = _hash64(t)
        w = 2 if "_" in t else 1
        for i in range(64):
            bit = (h >> i) & 1
            vec[i] += w if bit else -w
    out = 0
    for i, v in enumerate(vec):
        if v >= 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


@dataclass(frozen=True)
class SemanticDedupReport:
    total: int
    clusters: int
    duplicate_items: int
    largest_cluster_size: int
    top_clusters: List[List[int]]
    sample_pairs: List[Tuple[int, int, int]]


def _bucket_keys(sig: int) -> Tuple[int, int, int, int]:
    # 4 bands x 16 bits
    return (
        sig & 0xFFFF,
        (sig >> 16) & 0xFFFF,
        (sig >> 32) & 0xFFFF,
        (sig >> 48) & 0xFFFF,
    )


def semantic_dedup_clusters(
    texts: List[str],
    *,
    max_hamming: int = 3,
    max_pairs: int = 2000,
    max_top_clusters: int = 10,
) -> Tuple[Dict[int, List[int]], SemanticDedupReport]:
    """
    Find near-duplicate clusters via SimHash bucketing.

    Returns:
    - clusters: cluster_id -> list[item_index]
    - report: summary + some sample pairs (i,j,dist)
    """
    n = len(texts)
    sigs = [simhash64(t) for t in texts]

    # buckets for candidate generation
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for i, s in enumerate(sigs):
        for band, key in enumerate(_bucket_keys(s)):
            buckets.setdefault((band, key), []).append(i)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    sample_pairs: List[Tuple[int, int, int]] = []
    compared = set()
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        # compare within bucket
        for i in range(len(idxs)):
            a = idxs[i]
            for j in range(i + 1, len(idxs)):
                b = idxs[j]
                key = (a, b) if a < b else (b, a)
                if key in compared:
                    continue
                compared.add(key)
                d = hamming(sigs[a], sigs[b])
                if d <= max_hamming:
                    union(a, b)
                    if len(sample_pairs) < max_pairs:
                        sample_pairs.append((key[0], key[1], d))

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(i)

    # normalize cluster ids
    norm_clusters: Dict[int, List[int]] = {}
    for k, v in clusters.items():
        norm_clusters[len(norm_clusters)] = sorted(v)

    multi = [v for v in norm_clusters.values() if len(v) >= 2]
    duplicate_items = sum(len(v) for v in multi) - len(multi)
    largest = max((len(v) for v in norm_clusters.values()), default=0)
    top_clusters = sorted(multi, key=len, reverse=True)[:max_top_clusters]

    report = SemanticDedupReport(
        total=n,
        clusters=len(norm_clusters),
        duplicate_items=int(duplicate_items),
        largest_cluster_size=int(largest),
        top_clusters=top_clusters,
        sample_pairs=sample_pairs[:200],
    )
    return norm_clusters, report

