import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from dataset_gen.config import AppConfig
from dataset_gen.lang import detect_prompt_lang
from dataset_gen.mineru_client import MinerUHttpClient, MinerUParseRequest
from dataset_gen.processing.canonicalize import canonicalize_markdown, write_canonical
from dataset_gen.storage.doc_store import DocStore
from dataset_gen.indexing.sqlite_fts import SqliteFtsIndex


def _list_pdfs(input_path: Path, pattern: str) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted([p for p in input_path.rglob(pattern) if p.is_file()])


def ingest_one_or_many(
    cfg: AppConfig,
    *,
    input_path: Path,
    pattern: str,
    mineru_url: str,
    timeout_s: int,
    parse_format: str,
    lang: str,
    formula_enable: bool,
    table_enable: bool,
    start_page: int,
    end_page: Optional[int],
    caption_mode: Optional[str] = None,
    caption_max_images: Optional[int] = None,
    keep_source: bool,
) -> List[Dict[str, Any]]:
    store = DocStore(cfg)
    client = MinerUHttpClient(base_url=mineru_url, timeout_s=timeout_s)
    pdfs = _list_pdfs(input_path, pattern)
    results: List[Dict[str, Any]] = []

    for pdf in pdfs:
        rec = store.upsert_source(pdf)
        doc_root = cfg.output_dir
        if keep_source:
            cfg.docs_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pdf, cfg.docs_dir / f"{rec.doc_id}_{pdf.name}")

        parsed_root = cfg.parsed_dir / "mineru" / rec.doc_id
        parsed_root.mkdir(parents=True, exist_ok=True)

        parse_req = MinerUParseRequest(
            lang=lang,
            formula_enable=formula_enable,
            table_enable=table_enable,
            start_page=start_page,
            end_page=end_page,
            output_format=parse_format,
            caption_mode=caption_mode,
            caption_max_images=caption_max_images,
        )
        parse_res = client.parse(file_path=pdf, req=parse_req)
        task_id = str(parse_res.get("task_id") or "")
        if not task_id:
            raise RuntimeError(f"MinerU parse did not return task_id: {parse_res}")

        local_task_dir = client.sync_task(task_id, parsed_root)

        md_rel = parse_res.get("markdown_rel_path") or parse_res.get("markdown_path")
        cl_rel = parse_res.get("content_list_rel_path") or parse_res.get("content_list_path")
        am_rel = parse_res.get("asset_manifest_rel_path") or parse_res.get("asset_manifest_path")

        markdown_path = (local_task_dir / md_rel) if md_rel else None
        content_list_path = (local_task_dir / cl_rel) if cl_rel else None
        asset_manifest_path = (local_task_dir / am_rel) if am_rel else None

        if markdown_path is None or not markdown_path.exists():
            # For content_list only, there might be no markdown.
            markdown_path = None

        doc_lang = None
        if markdown_path is not None and markdown_path.exists():
            try:
                md_text = markdown_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                md_text = ""
            fb_text = ""
            if content_list_path is not None and content_list_path.exists():
                try:
                    import json

                    obj = json.loads(content_list_path.read_text(encoding="utf-8", errors="ignore"))
                    items = []
                    if isinstance(obj, list) and obj and isinstance(obj[0], list):
                        for sub in obj:
                            if isinstance(sub, list):
                                items.extend([x for x in sub if isinstance(x, dict)])
                    elif isinstance(obj, list):
                        items = [x for x in obj if isinstance(x, dict)]
                    parts = []
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        if str(it.get("type") or "") != "text":
                            continue
                        t = it.get("text")
                        if isinstance(t, str) and t.strip():
                            parts.append(t.strip())
                            if sum(len(p) for p in parts) >= 2000:
                                break
                    fb_text = "\n".join(parts)
                except Exception:
                    fb_text = ""
            doc_lang = detect_prompt_lang(md_text, fb_text)

        canonical_path = cfg.canonical_dir / rec.doc_id / "canonical.json"
        index_path = cfg.indexes_dir / rec.doc_id / "chunks.sqlite3"

        if markdown_path is not None:
            canonical = canonicalize_markdown(
                doc_id=rec.doc_id,
                markdown_path=markdown_path,
                content_list_path=content_list_path if content_list_path and content_list_path.exists() else None,
            )
            write_canonical(canonical, canonical_path)

            idx = SqliteFtsIndex(index_path)
            idx.rebuild(
                [
                    {
                        "chunk_id": c.chunk_id,
                        "section_id": getattr(c, "section_id", "") or "",
                        "section_title": c.section_title,
                        "text": c.text,
                    }
                    for c in canonical.chunks
                ]
            )
            idx.close()

        store.update_doc(
            rec.doc_id,
            mineru_task_id=task_id,
            mineru_task_local_dir=str(local_task_dir),
            mineru_markdown_path=str(markdown_path) if markdown_path else None,
            mineru_content_list_path=str(content_list_path) if content_list_path else None,
            mineru_asset_manifest_path=str(asset_manifest_path) if asset_manifest_path and asset_manifest_path.exists() else None,
            canonical_path=str(canonical_path) if canonical_path.exists() else None,
            index_path=str(index_path) if index_path.exists() else None,
            extra={
                "mineru_parse": parse_res,
                "mineru_url": mineru_url,
                "output_dir": str(doc_root),
                "doc_language": doc_lang,
            },
        )

        results.append(
            {
                "doc_id": rec.doc_id,
                "file": str(pdf),
                "mineru_task_id": task_id,
                "mineru_task_local_dir": str(local_task_dir),
                "canonical_path": str(canonical_path) if canonical_path.exists() else None,
                "index_path": str(index_path) if index_path.exists() else None,
            }
        )

    return results
