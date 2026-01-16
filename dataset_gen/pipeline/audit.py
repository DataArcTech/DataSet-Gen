import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dataset_gen.config import AppConfig
from dataset_gen.storage.doc_store import DocStore
from dataset_gen.evaluation.semantic_dedup import semantic_dedup_clusters
from dataset_gen.evaluation.section_coverage import compute_section_coverage
from dataset_gen.evaluation.reachability import reachability_stress_test


_PUNCT_RE = re.compile(r"[，。！？,.!?;；:：()（）\\[\\]{}\"'“”‘’]+")


def _norm_question(q: str) -> str:
    s = str(q or "").strip().lower()
    s = re.sub(r"\\s+", " ", s)
    s = _PUNCT_RE.sub("", s)
    return s[:500]


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def audit_dataset(
    *,
    cfg: Optional[AppConfig] = None,
    qa_jsonl: Path,
    debug_jsonl: Optional[Path] = None,
    semantic_dedup: bool = True,
    semantic_max_hamming: int = 3,
    coverage_quota_max_share: float = 0.35,
    reachability: bool = False,
    reachability_max_searches: int = 2,
    reachability_search_limit: int = 10,
    reachability_max_reads: int = 2,
) -> Dict[str, Any]:
    qa_items = list(_iter_jsonl(qa_jsonl))
    total = len(qa_items)

    seen: set[str] = set()
    dup = 0
    for it in qa_items:
        key = _norm_question(str(it.get("question") or ""))
        if not key:
            continue
        if key in seen:
            dup += 1
        else:
            seen.add(key)

    out: Dict[str, Any] = {
        "qa_jsonl": str(qa_jsonl),
        "total": total,
        "duplicate_questions": dup,
        "unique_questions": len(seen),
    }

    if semantic_dedup and total > 0:
        texts = [str(it.get("question") or "") for it in qa_items]
        _, rep = semantic_dedup_clusters(texts, max_hamming=int(semantic_max_hamming))
        out["semantic_dedup"] = {
            "method": "simhash64",
            "max_hamming": int(semantic_max_hamming),
            "clusters": rep.clusters,
            "duplicate_items": rep.duplicate_items,
            "largest_cluster_size": rep.largest_cluster_size,
            "top_clusters": rep.top_clusters,
            "sample_pairs": rep.sample_pairs,
        }

    if debug_jsonl and debug_jsonl.exists():
        dbg_items = list(_iter_jsonl(debug_jsonl))
        by_diff: Dict[str, int] = {}
        by_kind: Dict[str, int] = {}
        evidence_sections: List[int] = []
        evidence_docs: List[int] = []
        multimodal = 0

        for it in dbg_items:
            diff = str(it.get("difficulty") or "")
            kind = str(it.get("kind") or "")
            if diff:
                by_diff[diff] = by_diff.get(diff, 0) + 1
            if kind:
                by_kind[kind] = by_kind.get(kind, 0) + 1
            sec_ids = it.get("evidence_section_ids") or []
            if isinstance(sec_ids, list):
                evidence_sections.append(len({str(x) for x in sec_ids if str(x)}))
            doc_ids = it.get("used_doc_ids") or []
            if isinstance(doc_ids, list):
                evidence_docs.append(len({str(x) for x in doc_ids if str(x)}))

            # Cheap multimodal proxy: any read in trajectory has has_table true or non-empty image_urls.
            traj = it.get("trajectory") or []
            has_mm = False
            if isinstance(traj, list):
                for step in traj:
                    if not isinstance(step, dict):
                        continue
                    obs = step.get("observation") if isinstance(step.get("observation"), dict) else {}
                    reads = obs.get("reads") or []
                    if not isinstance(reads, list):
                        continue
                    for r in reads:
                        if not isinstance(r, dict):
                            continue
                        if r.get("has_table") is True:
                            has_mm = True
                            break
                        imgs = r.get("image_urls") or []
                        if isinstance(imgs, list) and len(imgs) > 0:
                            has_mm = True
                            break
                    if has_mm:
                        break
            if has_mm:
                multimodal += 1

        out["debug_jsonl"] = str(debug_jsonl)
        out["debug_total"] = len(dbg_items)
        out["by_difficulty"] = by_diff
        out["by_kind"] = by_kind
        out["multimodal_items_estimate"] = multimodal
        out["avg_evidence_sections"] = (sum(evidence_sections) / len(evidence_sections)) if evidence_sections else 0.0
        out["avg_used_docs"] = (sum(evidence_docs) / len(evidence_docs)) if evidence_docs else 0.0

        # Coverage / section tree quota checks (requires canonical sections; best-effort via DocStore).
        if cfg is not None:
            try:
                store = DocStore(cfg)
                # Aggregate evidence across all items; multi-doc friendly.
                used_section_ids: List[str] = []
                used_doc_ids: set[str] = set()
                for it in dbg_items:
                    for sid in (it.get("evidence_section_ids") or []):
                        if isinstance(sid, str) and sid:
                            used_section_ids.append(sid)
                    for did in (it.get("used_doc_ids") or []):
                        if isinstance(did, str) and did:
                            used_doc_ids.add(did)

                coverage_by_doc: Dict[str, Any] = {}
                for did in sorted(used_doc_ids):
                    try:
                        doc = store.get_doc(did)
                    except Exception:
                        continue
                    canon_path = doc.get("canonical_path")
                    if not isinstance(canon_path, str) or not canon_path:
                        continue
                    try:
                        canon = json.loads(Path(canon_path).read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    sections = canon.get("sections") if isinstance(canon, dict) else None
                    if not isinstance(sections, list):
                        continue
                    # filter evidence for this doc_id based on section_id prefix convention
                    did_section_ids = [sid for sid in used_section_ids if isinstance(sid, str) and sid.startswith(did + "_")]
                    rep = compute_section_coverage(
                        sections=sections,
                        used_section_ids=did_section_ids,
                        quota_max_share=float(coverage_quota_max_share),
                    )
                    coverage_by_doc[did] = {
                        "top_level_total": rep.top_level_total,
                        "top_level_covered": rep.top_level_covered,
                        "top_level_coverage_ratio": rep.top_level_coverage_ratio,
                        "distinct_sections_used": rep.distinct_sections_used,
                        "section_total": rep.section_total,
                        "max_top_level_share": rep.max_top_level_share,
                        "top_level_counts": rep.top_level_counts,
                        "quota_violations": rep.quota_violations,
                    }
                if coverage_by_doc:
                    out["section_coverage"] = {
                        "quota_max_share": float(coverage_quota_max_share),
                        "by_doc_id": coverage_by_doc,
                    }
            except Exception:
                pass

        if reachability and cfg is not None and dbg_items:
            # Fixed-budget search/read reachability to any gold evidence section_id.
            rep, details = reachability_stress_test(
                cfg,
                items=dbg_items,
                max_searches=int(reachability_max_searches),
                search_limit=int(reachability_search_limit),
                max_reads=int(reachability_max_reads),
            )
            out["reachability"] = {
                "max_searches": int(reachability_max_searches),
                "search_limit": int(reachability_search_limit),
                "max_reads": int(reachability_max_reads),
                "total": rep.total,
                "ok": rep.ok,
                "ok_rate": rep.ok_rate,
                "avg_searches": rep.avg_searches,
                "avg_reads": rep.avg_reads,
                "samples": details[:50],
            }

    return out
