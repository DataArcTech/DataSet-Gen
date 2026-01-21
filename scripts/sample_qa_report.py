#!/usr/bin/env python3
"""
Sample N QA items from a generated dataset and write a Markdown report with debug/evidence.

Typical usage:
  python3 DataSet-Gen/scripts/sample_qa_report.py \
    --run-dir outputs/dataset_gen_20260119_203200 \
    --n 20 --seed 42
"""
import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
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


def _norm_q(s: str) -> str:
    # Keep consistent with docdancer_io.write_items_jsonl() / merge logic.
    s = str(s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\u3000\t\r\n]+", " ", s)
    s = re.sub(r"[，。！？,.!?;；:：()（）\[\]{}\"'“”‘’]+", "", s)
    return s[:500]


def _load_doc_store(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    p = run_dir / "doc_store.json"
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}
    docs = obj.get("docs") if isinstance(obj, dict) else None
    return docs if isinstance(docs, dict) else {}


def _md_escape(s: str) -> str:
    # Minimal escaping for fenced Markdown context; keep readable.
    return str(s or "").replace("\r\n", "\n").replace("\r", "\n")


def _pick_reads_from_trajectory(traj: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    reads: List[Dict[str, Any]] = []
    for step in traj or []:
        if not isinstance(step, dict):
            continue
        if str(step.get("tool") or "") != "read":
            continue
        obs = step.get("observation")
        if not isinstance(obs, dict):
            continue
        rr = obs.get("reads") or []
        if isinstance(rr, list):
            for r in rr:
                if isinstance(r, dict) and r.get("text"):
                    reads.append(r)
    return reads


def _best_doc_labels(
    *,
    used_doc_ids: List[str],
    reads: List[Dict[str, Any]],
    doc_store: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, str]]:
    """
    Return list[(doc_id, label)] in order, where label is best-effort PDF filename/stem/title.
    """
    out: List[Tuple[str, str]] = []
    seen: set[str] = set()

    def add(did: str, label: str) -> None:
        did = str(did or "").strip()
        if not did or did in seen:
            return
        seen.add(did)
        out.append((did, str(label or "").strip()))

    # Prefer read metadata (most accurate for this item).
    for r in reads:
        did = str(r.get("doc_id") or "").strip()
        if not did:
            continue
        label = r.get("doc_filename") or r.get("doc_title") or ""
        if not label:
            rec = doc_store.get(did) or {}
            label = rec.get("filename") or Path(str(rec.get("source_path") or "")).name
        add(did, label or did)

    # Backfill used_doc_ids.
    for did in used_doc_ids or []:
        rec = doc_store.get(did) or {}
        label = rec.get("filename") or Path(str(rec.get("source_path") or "")).name or did
        add(did, label)

    return out


def build_report_item_md(
    *,
    idx: int,
    qa: Dict[str, Any],
    dbg: Optional[Dict[str, Any]],
    doc_store: Dict[str, Dict[str, Any]],
) -> str:
    q = str(qa.get("question") or "").strip()
    a = str(qa.get("answer") or "").strip()

    difficulty = str((dbg or {}).get("difficulty") or "")
    kind = str((dbg or {}).get("kind") or "")
    used_doc_ids = (dbg or {}).get("used_doc_ids") if isinstance((dbg or {}).get("used_doc_ids"), list) else []
    evidence_chunk_ids = (dbg or {}).get("evidence_chunk_ids") if isinstance((dbg or {}).get("evidence_chunk_ids"), list) else []
    evidence_section_ids = (dbg or {}).get("evidence_section_ids") if isinstance((dbg or {}).get("evidence_section_ids"), list) else []
    traj = (dbg or {}).get("trajectory") if isinstance((dbg or {}).get("trajectory"), list) else []
    reads = _pick_reads_from_trajectory(traj)
    doc_labels = _best_doc_labels(used_doc_ids=used_doc_ids, reads=reads, doc_store=doc_store)

    parts: List[str] = []
    parts.append(f"## Sample {idx}\n")
    parts.append(f"- Difficulty: `{difficulty or 'unknown'}`  Kind: `{kind or 'unknown'}`\n")
    if doc_labels:
        parts.append("- Source PDF(s):\n")
        for did, label in doc_labels:
            parts.append(f"  - `{did}`: `{label}`\n")
    else:
        parts.append("- Source PDF(s): (unknown)\n")
    parts.append("\n")
    parts.append("### Question\n\n")
    parts.append(_md_escape(q) + "\n\n")
    parts.append("### Answer\n\n")
    parts.append(_md_escape(a) + "\n\n")

    parts.append("### Evidence IDs\n\n")
    parts.append(f"- evidence_chunk_ids ({len(evidence_chunk_ids)}): `{', '.join([str(x) for x in evidence_chunk_ids])}`\n")
    parts.append(f"- evidence_section_ids ({len(evidence_section_ids)}): `{', '.join([str(x) for x in evidence_section_ids])}`\n\n")

    parts.append("### Evidence Snippets (from debug trajectory)\n\n")
    if not reads:
        parts.append("_No read evidence found in debug trajectory._\n\n")
    else:
        # Keep report readable: show up to 6 read snippets.
        for j, r in enumerate(reads[:6], start=1):
            did = str(r.get("doc_id") or "").strip()
            doc_name = str(r.get("doc_filename") or r.get("doc_title") or "").strip()
            if not doc_name and did:
                rec = doc_store.get(did) or {}
                doc_name = str(rec.get("filename") or Path(str(rec.get("source_path") or "")).name or "").strip()
            sid = str(r.get("section_id") or "").strip()
            st = str(r.get("section_title") or "").strip()
            cids = r.get("chunk_ids") if isinstance(r.get("chunk_ids"), list) else []
            page_idxs = r.get("page_idxs") if isinstance(r.get("page_idxs"), list) else []
            has_table = r.get("has_table")
            image_urls = r.get("image_urls") if isinstance(r.get("image_urls"), list) else []
            txt = str(r.get("text") or "").strip()
            if len(txt) > 1800:
                txt = txt[:1800] + "\n…(truncated)…"

            parts.append(f"#### Read {j}\n\n")
            parts.append(f"- doc_id: `{did}`\n")
            parts.append(f"- doc_filename/title: `{doc_name or ''}`\n")
            parts.append(f"- section_id: `{sid}`\n")
            parts.append(f"- section_title: `{st}`\n")
            parts.append(f"- chunk_ids: `{', '.join([str(x) for x in cids])}`\n")
            parts.append(f"- page_idxs: `{', '.join([str(x) for x in page_idxs])}`\n")
            parts.append(f"- has_table: `{has_table}`\n")
            parts.append(f"- image_urls_count: `{len(image_urls)}`\n\n")
            parts.append("```text\n")
            parts.append(_md_escape(txt) + "\n")
            parts.append("```\n\n")

    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None, help="Run dir containing qa.mix.jsonl and doc_store.json.")
    ap.add_argument("--qa-jsonl", default=None, help="QA jsonl path (overrides --run-dir default).")
    ap.add_argument("--debug-jsonl", default=None, help="Debug jsonl path (overrides --run-dir default).")
    ap.add_argument("--n", type=int, default=20, help="Number of items to sample.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--out", default=None, help="Output Markdown path.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    qa_path = Path(args.qa_jsonl).resolve() if args.qa_jsonl else (run_dir / "qa.mix.jsonl" if run_dir else None)
    dbg_path = (
        Path(args.debug_jsonl).resolve()
        if args.debug_jsonl
        else ((run_dir / "qa.mix.jsonl.debug.jsonl") if run_dir else None)
    )

    if qa_path is None or not qa_path.exists():
        raise SystemExit(f"QA jsonl not found: {qa_path}")
    if dbg_path is None or not dbg_path.exists():
        raise SystemExit(f"Debug jsonl not found: {dbg_path}")

    if args.out:
        out_path = Path(args.out).resolve()
    else:
        out_path = (run_dir / f"qa.sample{int(args.n)}.md") if run_dir else qa_path.with_suffix(".sample.md")

    doc_store = _load_doc_store(run_dir) if run_dir else {}

    qa_items = list(_iter_jsonl(qa_path))
    if not qa_items:
        raise SystemExit(f"No QA items found in: {qa_path}")

    dbg_by_q: Dict[str, Dict[str, Any]] = {}
    for d in _iter_jsonl(dbg_path):
        q = d.get("question")
        if not isinstance(q, str) or not q.strip():
            continue
        k = _norm_q(q)
        if k and k not in dbg_by_q:
            dbg_by_q[k] = d

    rnd = random.Random(int(args.seed))
    n = max(1, int(args.n))
    picked = rnd.sample(qa_items, k=min(n, len(qa_items)))

    header = []
    header.append("# QA Sample Report\n\n")
    header.append(f"- qa_jsonl: `{qa_path}`\n")
    header.append(f"- debug_jsonl: `{dbg_path}`\n")
    if run_dir:
        header.append(f"- run_dir: `{run_dir}`\n")
    header.append(f"- sampled_n: `{len(picked)}` (requested `{n}`)\n")
    header.append(f"- seed: `{args.seed}`\n\n")
    header.append("Notes:\n")
    header.append("- Evidence snippets come from the debug trajectory (read tool outputs).\n")
    header.append("- `Source PDF(s)` are derived from evidence `doc_filename/doc_title` and fall back to doc_store.json.\n\n")

    body: List[str] = []
    for i, qa in enumerate(picked, start=1):
        q = str(qa.get("question") or "")
        dbg = dbg_by_q.get(_norm_q(q))
        body.append(build_report_item_md(idx=i, qa=qa, dbg=dbg, doc_store=doc_store))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(header) + "".join(body), encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

