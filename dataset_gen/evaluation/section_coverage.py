from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class SectionNode:
    section_id: str
    title: str
    level: int
    parent_id: Optional[str]


@dataclass(frozen=True)
class CoverageReport:
    top_level_total: int
    top_level_covered: int
    top_level_coverage_ratio: float
    distinct_sections_used: int
    section_total: int
    max_top_level_share: float
    top_level_counts: List[Tuple[str, int]]
    quota_violations: List[Dict[str, Any]]


def _to_nodes(sections: List[Dict[str, Any]]) -> Dict[str, SectionNode]:
    out: Dict[str, SectionNode] = {}
    for s in sections or []:
        if not isinstance(s, dict):
            continue
        sid = str(s.get("section_id") or "")
        if not sid:
            continue
        pid = s.get("parent_id")
        out[sid] = SectionNode(
            section_id=sid,
            title=str(s.get("title") or ""),
            level=int(s.get("level") or 0),
            parent_id=(str(pid) if pid is not None else None),
        )
    return out


def _top_level_id(nodes: Dict[str, SectionNode], sid: str) -> Optional[str]:
    cur = nodes.get(sid)
    if cur is None:
        return None
    # Define top-level as the first ancestor whose parent_id is None.
    seen = set()
    while cur.parent_id:
        if cur.section_id in seen:
            break
        seen.add(cur.section_id)
        nxt = nodes.get(cur.parent_id)
        if nxt is None:
            break
        cur = nxt
    return cur.section_id


def compute_section_coverage(
    *,
    sections: List[Dict[str, Any]],
    used_section_ids: Iterable[str],
    quota_max_share: float = 0.35,
) -> CoverageReport:
    nodes = _to_nodes(sections)
    top_levels = [n.section_id for n in nodes.values() if n.parent_id is None and n.level > 0]
    top_set = set(top_levels)

    used = {str(x) for x in used_section_ids if str(x)}
    used = {x for x in used if x in nodes}

    used_top: List[str] = []
    for sid in used:
        tid = _top_level_id(nodes, sid)
        if tid:
            used_top.append(tid)

    counts: Dict[str, int] = {}
    for tid in used_top:
        if tid in top_set:
            counts[tid] = counts.get(tid, 0) + 1

    total_used = len(used_top) if used_top else 0
    max_share = 0.0
    if total_used > 0 and counts:
        max_share = max(v / total_used for v in counts.values())

    top_level_counts = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    top_covered = len({t for t in used_top if t in top_set})
    top_total = len(top_levels)
    ratio = (top_covered / top_total) if top_total > 0 else 0.0

    violations: List[Dict[str, Any]] = []
    if total_used > 0:
        for tid, c in top_level_counts:
            share = c / total_used
            if share > float(quota_max_share):
                node = nodes.get(tid)
                violations.append(
                    {
                        "top_level_section_id": tid,
                        "title": (node.title if node else ""),
                        "count": c,
                        "share": share,
                        "quota_max_share": quota_max_share,
                    }
                )

    return CoverageReport(
        top_level_total=int(top_total),
        top_level_covered=int(top_covered),
        top_level_coverage_ratio=float(ratio),
        distinct_sections_used=int(len(used)),
        section_total=int(len(nodes)),
        max_top_level_share=float(max_share),
        top_level_counts=top_level_counts[:30],
        quota_violations=violations,
    )

