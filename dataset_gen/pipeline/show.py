import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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


def _one_line(text: str, *, max_chars: int) -> str:
    s = " ".join(str(text or "").split())
    if max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 1)] + "…"


def show_dataset_items(
    *,
    debug_jsonl: Path,
    n: int = 3,
    kind: Optional[str] = None,
    difficulty: Optional[str] = None,
    evidence_chars: int = 260,
) -> List[Dict[str, Any]]:
    """
    Return a compact list of items suitable for printing: question/answer + compact evidence snippets.
    """
    out: List[Dict[str, Any]] = []
    for it in _iter_jsonl(debug_jsonl):
        if kind and str(it.get("kind") or "") != kind:
            continue
        if difficulty and str(it.get("difficulty") or "") != difficulty:
            continue

        evidence: List[Dict[str, Any]] = []
        traj = it.get("trajectory") or []
        if isinstance(traj, list):
            seen_sections: set[str] = set()
            for step in traj:
                if not isinstance(step, dict) or step.get("tool") != "read":
                    continue
                obs = step.get("observation") if isinstance(step.get("observation"), dict) else {}
                reads = obs.get("reads") or []
                if not isinstance(reads, list):
                    continue
                for r in reads:
                    if not isinstance(r, dict):
                        continue
                    sid = str(r.get("section_id") or "")
                    if not sid or sid in seen_sections:
                        continue
                    seen_sections.add(sid)
                    evidence.append(
                        {
                            "section_id": sid,
                            "section_title": r.get("section_title"),
                            "page_idxs": r.get("page_idxs"),
                            "has_table": r.get("has_table"),
                            "image_urls": r.get("image_urls"),
                            "snippet": _one_line(r.get("text") or "", max_chars=evidence_chars),
                        }
                    )
                    if len(evidence) >= 5:
                        break
                if len(evidence) >= 5:
                    break

        out.append(
            {
                "question": it.get("question"),
                "answer": it.get("answer"),
                "difficulty": it.get("difficulty"),
                "kind": it.get("kind"),
                "evidence_section_ids": it.get("evidence_section_ids"),
                "derived": it.get("derived"),
                "evidence": evidence,
            }
        )
        if len(out) >= int(n):
            break
    return out

