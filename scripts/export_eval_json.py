#!/usr/bin/env python3
"""
Export DataSet-Gen outputs to an evaluation-friendly JSON format.

Inputs (from a run_dir):
  - qa.mix.jsonl: clean QA pairs (no citations)
  - qa.mix.jsonl.debug.jsonl: debug-only fields (includes citations + citation_map)
  - doc_store.json + canonical/<doc_id>/canonical.json: chunk corpus

Outputs (to project outputs/):
  - <name>-qa.json: list[{query, answer, evidence_list, ...}]
  - <name>-doc.json: list[{title, body, ...}]  (chunk-level corpus)
  - <name>-assets/: copied multimodal assets (images/pages/crops) for all docs

Only `query`, `answer`, `evidence_list` are guaranteed; other fields are best-effort and optional.
"""
import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


_CIT_RE = re.compile(r"\[(\d+)\]")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
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


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_name(s: str) -> str:
    s = str(s or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "dataset"


def _excerpt(text: str, *, max_chars: int = 420) -> str:
    t = str(text or "").strip()
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t)
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    filename: str
    title: str
    canonical_path: Path
    mineru_dir: Optional[Path]


def _load_doc_store(run_dir: Path) -> Dict[str, DocMeta]:
    store_path = run_dir / "doc_store.json"
    obj = json.loads(store_path.read_text(encoding="utf-8", errors="ignore"))
    docs = obj.get("docs") if isinstance(obj, dict) else None
    out: Dict[str, DocMeta] = {}
    if not isinstance(docs, dict):
        return out
    for doc_id, rec in docs.items():
        if not isinstance(rec, dict):
            continue
        did = str(rec.get("doc_id") or doc_id).strip()
        if not did:
            continue
        canon = rec.get("canonical_path")
        if not isinstance(canon, str) or not canon:
            continue
        canonical_path = Path(canon)
        if not canonical_path.exists():
            continue
        filename = str(rec.get("filename") or "").strip()
        title = filename.rsplit(".", 1)[0] if filename else did
        md_path = rec.get("mineru_markdown_path")
        mineru_dir = None
        if isinstance(md_path, str) and md_path:
            mineru_dir = Path(md_path).parent
        out[did] = DocMeta(
            doc_id=did,
            filename=filename,
            title=title,
            canonical_path=canonical_path,
            mineru_dir=mineru_dir,
        )
    return out


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _emit_json_array_stream(path: Path, items: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        for it in items:
            if not first:
                f.write(",\n")
            first = False
            f.write(json.dumps(it, ensure_ascii=False, indent=2))
            n += 1
        f.write("\n]\n")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Run dir, e.g. outputs/dataset_gen_20260119_203200")
    ap.add_argument("--out-dir", default="outputs", help="Output dir for exported JSON files (default: outputs/)")
    ap.add_argument("--name", default="", help="Dataset name prefix (default: derived from run-dir basename)")
    ap.add_argument(
        "--qa-jsonl",
        default="qa.mix.jsonl",
        help="QA jsonl filename under run-dir (default: qa.mix.jsonl)",
    )
    ap.add_argument(
        "--debug-jsonl",
        default="qa.mix.jsonl.debug.jsonl",
        help="Debug jsonl filename under run-dir (default: qa.mix.jsonl.debug.jsonl)",
    )
    ap.add_argument("--copy-assets", default="1", choices=["0", "1"], help="Copy multimodal assets (default: 1)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    name = _safe_name(args.name or run_dir.name)

    qa_jsonl_arg = Path(str(args.qa_jsonl))
    dbg_jsonl_arg = Path(str(args.debug_jsonl))
    qa_jsonl = qa_jsonl_arg if qa_jsonl_arg.is_absolute() else (run_dir / qa_jsonl_arg)
    dbg_jsonl = dbg_jsonl_arg if dbg_jsonl_arg.is_absolute() else (run_dir / dbg_jsonl_arg)

    if not qa_jsonl.exists():
        # Best-effort fallback: look for a qa.mix.jsonl under the run_dir.
        found = list(run_dir.glob("**/qa.mix.jsonl"))
        if found:
            qa_jsonl = found[0]
    if not dbg_jsonl.exists():
        found = list(run_dir.glob("**/qa.mix.jsonl.debug.jsonl"))
        if found:
            dbg_jsonl = found[0]

    # Load debug (jsonl); map by question_plain for stable joining.
    dbg_by_q: Dict[str, Dict[str, Any]] = {}
    for obj in _iter_jsonl(dbg_jsonl):
        q_plain = str(obj.get("question_plain") or obj.get("question") or "").strip()
        if not q_plain:
            continue
        dbg_by_q[q_plain] = obj

    # Load doc store, copy assets (all docs) and write doc corpus; also build lookup for referenced chunks.
    doc_meta = _load_doc_store(run_dir)
    assets_root = out_dir / f"{name}-assets"
    do_copy_assets = str(args.copy_assets) == "1"

    chunk_lookup: Dict[str, Dict[str, Any]] = {}

    def _doc_items() -> Iterable[Dict[str, Any]]:
        for did, meta in sorted(doc_meta.items(), key=lambda x: x[0]):
            # Copy multimodal assets once per doc.
            if do_copy_assets and meta.mineru_dir and meta.mineru_dir.exists():
                for sub in ("images", "pages", "crops"):
                    src = meta.mineru_dir / sub
                    dst = assets_root / did / sub
                    _copy_tree(src, dst)

            canon = json.loads(meta.canonical_path.read_text(encoding="utf-8", errors="ignore"))
            chunks = canon.get("chunks") if isinstance(canon, dict) else None
            if not isinstance(chunks, list):
                continue

            # Precompute page/crop file maps (best-effort) for this doc.
            pages_dir = (assets_root / did / "pages") if (assets_root / did / "pages").exists() else None
            crops_dir = (assets_root / did / "crops") if (assets_root / did / "crops").exists() else None

            def page_path(page_idx: Optional[int]) -> Optional[str]:
                if pages_dir is None or page_idx is None:
                    return None
                cand = pages_dir / f"page_{page_idx + 1:04d}.png"
                if cand.exists():
                    return str(cand.relative_to(out_dir))
                cand2 = pages_dir / f"page_{page_idx + 1}.png"
                if cand2.exists():
                    return str(cand2.relative_to(out_dir))
                return None

            def crop_paths(page_idx: Optional[int], *, limit: int = 4) -> List[str]:
                if crops_dir is None or page_idx is None:
                    return []
                pat = f"crop_*_p{page_idx + 1:04d}_*.png"
                out: List[str] = []
                for p in sorted(crops_dir.glob(pat))[:limit]:
                    if p.exists():
                        out.append(str(p.relative_to(out_dir)))
                return out

            for ch in chunks:
                if not isinstance(ch, dict):
                    continue
                chunk_id = str(ch.get("chunk_id") or "").strip()
                if not chunk_id:
                    continue
                body = str(ch.get("text") or "").strip()
                section_title = str(ch.get("section_title") or "").strip()
                page_idx = ch.get("page_idx") if isinstance(ch.get("page_idx"), int) else None
                image_urls = ch.get("image_urls") if isinstance(ch.get("image_urls"), list) else []

                # Map image_urls -> copied assets relative paths (best-effort).
                rel_images: List[str] = []
                for u in image_urls[:8]:
                    if not isinstance(u, str) or not u.strip():
                        continue
                    fn = Path(u).name
                    p = assets_root / did / "images" / fn
                    if p.exists():
                        rel_images.append(str(p.relative_to(out_dir)))

                item: Dict[str, Any] = {
                    "id": chunk_id,
                    "title": meta.title if not section_title else f"{meta.title} - {section_title}",
                    "source": meta.filename or meta.title,
                    "url": f"doc:{did}#{chunk_id}",
                    "body": body,
                    # Extra optional fields for traceability / multimodal retrieval.
                    "doc_id": did,
                    "chunk_id": chunk_id,
                    "section_id": ch.get("section_id"),
                    "section_title": section_title or None,
                    "page_idx": page_idx,
                }
                et = ch.get("element_types")
                if isinstance(et, list) and et:
                    item["element_types"] = et
                if image_urls:
                    item["image_urls"] = image_urls[:8]
                if rel_images:
                    item["image_paths"] = rel_images
                pp = page_path(page_idx)
                if pp:
                    item["page_image_path"] = pp
                cps = crop_paths(page_idx)
                if cps:
                    item["crop_paths"] = cps

                # Build a compact chunk lookup for QA evidence_list. This is the retrieval-corpus primary key.
                chunk_lookup[chunk_id] = {
                    "id": chunk_id,
                    "doc_id": did,
                    "chunk_id": chunk_id,
                    "title": item.get("title"),
                    "source": item.get("source"),
                    "url": item.get("url"),
                    "page_idx": page_idx,
                    "fact": _excerpt(body, max_chars=520),
                }
                if rel_images:
                    chunk_lookup[chunk_id]["image_paths"] = rel_images
                if pp:
                    chunk_lookup[chunk_id]["page_image_path"] = pp
                if cps:
                    chunk_lookup[chunk_id]["crop_paths"] = cps

                yield item

    doc_out = out_dir / f"{name}-doc.json"
    doc_n = _emit_json_array_stream(doc_out, _doc_items())

    # Build QA export
    qa_out = out_dir / f"{name}-qa.json"

    def _qa_items() -> Iterable[Dict[str, Any]]:
        for qa in _iter_jsonl(qa_jsonl):
            q = str(qa.get("question") or "").strip()
            a = str(qa.get("answer") or "").strip()
            if not q or not a:
                continue
            if a in {"UNANSWERABLE", "无法回答", "無法回答"}:
                yield {"query": q, "answer": a, "evidence_list": []}
                continue
            dbg = dbg_by_q.get(q) or {}
            dbg_obj = dbg.get("debug") if isinstance(dbg.get("debug"), dict) else {}
            citation_map = dbg_obj.get("citation_map") if isinstance(dbg_obj.get("citation_map"), list) else []
            awc = dbg_obj.get("answer_with_citations") if isinstance(dbg_obj.get("answer_with_citations"), str) else ""

            evidence_list: List[Dict[str, Any]] = []
            if citation_map and awc:
                by_eid = {m.get("eid"): m for m in citation_map if isinstance(m, dict) and isinstance(m.get("eid"), int)}
                seen_chunks: Set[str] = set()
                for eid_s in _CIT_RE.findall(awc):
                    try:
                        eid = int(eid_s)
                    except Exception:
                        continue
                    m = by_eid.get(eid) or {}
                    cids = m.get("chunk_ids") if isinstance(m.get("chunk_ids"), list) else []
                    cid = str(cids[0]).strip() if cids else ""
                    if not cid or cid in seen_chunks:
                        continue
                    seen_chunks.add(cid)
                    base = chunk_lookup.get(cid) or {}
                    ev: Dict[str, Any] = {
                        "id": cid,
                        "chunk_id": cid,
                        "doc_id": str(m.get("doc_id") or base.get("doc_id") or "").strip() or None,
                        "fact": base.get("fact") or "",
                    }
                    # Optional helper fields if available.
                    for k in ("title", "source", "url", "page_idx", "image_paths", "page_image_path", "crop_paths"):
                        if k in base and base.get(k) is not None:
                            ev[k] = base.get(k)
                    evidence_list.append(ev)
            else:
                # Fallback: use evidence_chunk_ids if present in debug.
                for cid in (dbg.get("evidence_chunk_ids") or []):
                    if not isinstance(cid, str) or not cid.strip():
                        continue
                    base = chunk_lookup.get(cid) or {}
                    evidence_list.append(
                        {
                            "id": cid,
                            "chunk_id": cid,
                            "doc_id": base.get("doc_id"),
                            "fact": base.get("fact") or "",
                        }
                    )

            qa_item: Dict[str, Any] = {
                "query": q,
                "answer": a,
                "evidence_list": evidence_list,
            }
            # Optional fields if present in debug.
            if dbg.get("difficulty"):
                qa_item["difficulty"] = dbg.get("difficulty")
            if dbg.get("kind"):
                qa_item["kind"] = dbg.get("kind")
            yield qa_item

    qa_n = _emit_json_array_stream(qa_out, _qa_items())

    print(
        json.dumps(
            {
                "ok": True,
                "run_dir": str(run_dir),
                "name": name,
                "qa_json": str(qa_out),
                "doc_json": str(doc_out),
                "assets_dir": str(assets_root) if do_copy_assets else None,
                "doc_items": doc_n,
                "qa_items": qa_n,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
