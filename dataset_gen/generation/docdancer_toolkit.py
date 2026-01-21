import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dataset_gen.config import AppConfig
from dataset_gen.toolkit.doc_toolkit import DocToolkit, SearchResult

from .docdancer_utils import looks_like_table


class MultiDocToolkit:
    def __init__(
        self,
        cfg: AppConfig,
        doc_ids: List[str],
        *,
        assets: Dict[str, Dict[str, Any]],
        doc_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.cfg = cfg
        self.doc_ids = doc_ids
        self._tks: Dict[str, DocToolkit] = {doc_id: DocToolkit(cfg, doc_id=doc_id) for doc_id in doc_ids}
        self._assets = assets
        # Optional doc identity hints (e.g. filename/title) for self-contained questions.
        self._doc_meta = doc_meta or {}

    def close(self) -> None:
        for tk in self._tks.values():
            tk.close()

    def search(self, *, keywords: List[str], limit: int = 20) -> List[Dict[str, Any]]:
        all_hits: List[Tuple[str, SearchResult]] = []
        per_doc = max(3, limit)
        for doc_id, tk in self._tks.items():
            hits = tk.search(keywords=keywords, limit=per_doc)
            for h in hits:
                all_hits.append((doc_id, h))
        # FTS bm25 lower is better. Missing scores are 0.0 (fallback) -> treat as middle.
        all_hits.sort(key=lambda x: float(x[1].score))
        out: List[Dict[str, Any]] = []
        for doc_id, h in all_hits[:limit]:
            meta = self._tks[doc_id].get_chunk_info(h.chunk_id) or {}
            dm = self._doc_meta.get(doc_id) or {}
            out.append(
                {
                    "doc_id": doc_id,
                    "doc_filename": dm.get("filename"),
                    "doc_title": dm.get("title"),
                    "section_id": h.section_id or meta.get("section_id"),
                    "chunk_id": h.chunk_id,
                    "score": h.score,
                    "page_idx": meta.get("page_idx"),
                    "section_title": meta.get("section_title"),
                    "snippet": h.snippet,
                }
            )
        return out

    def read(
        self,
        *,
        section_ids: Optional[List[str]] = None,
        chunk_ids: Optional[List[str]] = None,
        goal: str,
        max_chars: int = 5000,
        max_chunks_per_section: int = 8,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        sec_ids = [str(s).strip() for s in (section_ids or []) if str(s).strip()]
        cids = [str(c).strip() for c in (chunk_ids or []) if str(c).strip()]

        # If only chunk_ids are provided, infer section_ids from them.
        if not sec_ids and cids:
            for cid in cids:
                doc_id = cid.split("_", 1)[0]
                tk = self._tks.get(doc_id)
                if not tk:
                    continue
                meta = tk.get_chunk_info(cid) or {}
                sid = str(meta.get("section_id") or "").strip()
                if sid and sid not in sec_ids:
                    sec_ids.append(sid)

        for sid in sec_ids:
            doc_id = sid.split("_", 1)[0]
            tk = self._tks.get(doc_id)
            if not tk:
                continue
            rr = tk.read(
                section_ids=[sid],
                chunk_ids=cids,
                goal=goal,
                max_chars=max_chars,
                max_chunks_per_section=int(max_chunks_per_section),
            )
            for r in rr:
                images_dir = (self._assets.get(doc_id) or {}).get("images_dir")
                pages_dir = (self._assets.get(doc_id) or {}).get("pages_dir")
                crops_dir = (self._assets.get(doc_id) or {}).get("crops_dir")
                dm = self._doc_meta.get(doc_id) or {}
                abs_images: List[str] = []
                if isinstance(images_dir, str) and images_dir and r.image_urls:
                    for u in r.image_urls[:3]:
                        name = Path(str(u)).name
                        abs_path = Path(images_dir) / name
                        if abs_path.exists():
                            abs_images.append(str(abs_path))
                abs_pages: List[str] = []
                if isinstance(pages_dir, str) and pages_dir and r.page_idxs:
                    for pidx in r.page_idxs[:2]:
                        if not isinstance(pidx, int):
                            continue
                        # Prefer new padded naming, fall back to simple patterns.
                        candidates = [
                            Path(pages_dir) / f"page_{pidx + 1:04d}.png",
                            Path(pages_dir) / f"page_{pidx + 1}.png",
                            Path(pages_dir) / f"p{pidx}.png",
                        ]
                        for cand in candidates:
                            if cand.exists():
                                abs_pages.append(str(cand))
                                break
                abs_crops: List[str] = []
                if isinstance(crops_dir, str) and crops_dir and r.page_idxs:
                    for pidx in r.page_idxs[:2]:
                        if not isinstance(pidx, int):
                            continue
                        # Crops use padded page index. We include a small number to limit cost.
                        pat = f"crop_*_p{pidx + 1:04d}_*.png"
                        for cand in sorted(Path(crops_dir).glob(pat))[:2]:
                            if cand.exists():
                                abs_crops.append(str(cand))
                out.append(
                    {
                        "doc_id": doc_id,
                        "doc_filename": dm.get("filename"),
                        "doc_title": dm.get("title"),
                        "section_id": r.section_id,
                        "chunk_ids": list(r.chunk_ids),
                        "section_title": r.section_title,
                        "page_idxs": list(r.page_idxs),
                        "start_char": r.start_char,
                        "end_char": r.end_char,
                        "image_urls": r.image_urls,
                        "image_paths": abs_images,
                        "page_image_paths": abs_pages,
                        "crop_paths": abs_crops,
                        "has_table": looks_like_table(r.text),
                        "text": r.text,
                    }
                )
        return out

    def outline(self, *, max_sections_per_doc: int = 40) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for doc_id, tk in self._tks.items():
            secs = getattr(tk._canonical, "sections", []) or []
            packed: List[Dict[str, Any]] = []
            for s in secs[:max_sections_per_doc]:
                packed.append(
                    {
                        "section_id": s.section_id,
                        "title": s.title,
                        "level": s.level,
                        "parent_id": s.parent_id,
                    }
                )
            dm = self._doc_meta.get(doc_id) or {}
            out.append({"doc_id": doc_id, "doc_filename": dm.get("filename"), "doc_title": dm.get("title"), "sections": packed})
        return out

