from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dataset_gen.config import AppConfig
from dataset_gen.toolkit.doc_toolkit import DocToolkit


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
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
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


def extract_keywords(question: str, *, max_terms: int = 8) -> List[str]:
    toks: List[str] = []
    for m in _TOKEN_RE.finditer(str(question or "").lower()):
        t = m.group(0).strip()
        if not t:
            continue
        if t in _EN_STOP:
            continue
        if t not in toks:
            toks.append(t)
        if len(toks) >= max_terms:
            break
    return toks


def _alt_keywords(question: str, base: List[str], *, max_terms: int = 10) -> List[str]:
    # A lightweight second query: include some bigrams and keep numbers.
    toks = extract_keywords(question, max_terms=max_terms)
    out = list(base)
    for i in range(len(toks) - 1):
        bg = toks[i] + " " + toks[i + 1]
        if bg not in out:
            out.append(bg)
        if len(out) >= max_terms:
            break
    # ensure at least something
    return out[:max_terms] if out else toks[:max_terms]


@dataclass(frozen=True)
class ReachabilityItemResult:
    ok: bool
    searches: int
    reads: int
    hit_sections: List[str]
    found_evidence_section_ids: List[str]


@dataclass(frozen=True)
class ReachabilityReport:
    total: int
    ok: int
    ok_rate: float
    avg_searches: float
    avg_reads: float


def reachability_stress_test(
    cfg: AppConfig,
    *,
    items: Sequence[Dict[str, Any]],
    max_searches: int = 2,
    search_limit: int = 10,
    max_reads: int = 2,
) -> Tuple[ReachabilityReport, List[Dict[str, Any]]]:
    """
    Fixed-budget reachability: from question alone, can we retrieve at least one
    gold evidence section_id via search/read within the budget?
    """
    toolkits: Dict[str, DocToolkit] = {}

    def get_tk(doc_id: str) -> Optional[DocToolkit]:
        if doc_id in toolkits:
            return toolkits[doc_id]
        try:
            tk = DocToolkit(cfg, doc_id=doc_id)
        except Exception:
            return None
        toolkits[doc_id] = tk
        return tk

    results: List[Dict[str, Any]] = []
    ok = 0
    total_searches = 0
    total_reads = 0

    try:
        for it in items:
            q = str(it.get("question") or "")
            used_docs = it.get("used_doc_ids") or []
            if not isinstance(used_docs, list) or not used_docs:
                used_docs = []
            evidence = it.get("evidence_section_ids") or []
            if not isinstance(evidence, list):
                evidence = []
            evidence_set = {str(x) for x in evidence if str(x)}

            searches = 0
            reads = 0
            found: set[str] = set()
            hit_sections: List[str] = []

            base_kw = extract_keywords(q, max_terms=8)
            kw_sets: List[List[str]] = [base_kw]
            if max_searches >= 2:
                kw_sets.append(_alt_keywords(q, base_kw, max_terms=10))

            for kws in kw_sets[:max_searches]:
                if not kws:
                    continue
                for doc_id in used_docs:
                    tk = get_tk(str(doc_id))
                    if tk is None:
                        continue
                    searches += 1
                    hits = tk.search(keywords=kws, limit=search_limit)
                    for h in hits:
                        if h.section_id:
                            hit_sections.append(str(h.section_id))
                            if str(h.section_id) in evidence_set:
                                found.add(str(h.section_id))
                    # read top hits to simulate normal usage (optional for reachability)
                    if hits and max_reads > 0:
                        to_read = []
                        for h in hits:
                            if h.section_id and h.section_id not in to_read:
                                to_read.append(h.section_id)
                            if len(to_read) >= max_reads:
                                break
                        if to_read:
                            reads += 1
                            try:
                                tk.read(section_ids=to_read, goal="collect evidence", max_chars=1200)
                            except Exception:
                                pass
                if found:
                    break
            ok_item = bool(found)
            if ok_item:
                ok += 1
            total_searches += searches
            total_reads += reads
            results.append(
                {
                    "question": q,
                    "used_doc_ids": used_docs,
                    "evidence_section_ids": list(evidence_set),
                    "ok": ok_item,
                    "searches": searches,
                    "reads": reads,
                    "found_evidence_section_ids": sorted(found),
                    "hit_section_ids": hit_sections[:40],
                }
            )
    finally:
        for tk in toolkits.values():
            try:
                tk.close()
            except Exception:
                pass

    total = len(items)
    rep = ReachabilityReport(
        total=int(total),
        ok=int(ok),
        ok_rate=(ok / total) if total else 0.0,
        avg_searches=(total_searches / total) if total else 0.0,
        avg_reads=(total_reads / total) if total else 0.0,
    )
    return rep, results

