import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dataset_gen.config import AppConfig
from dataset_gen.indexing.sqlite_fts import SqliteFtsIndex
from dataset_gen.processing.canonicalize import load_canonical


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    section_id: str
    score: float
    snippet: str


@dataclass(frozen=True)
class ReadResult:
    section_id: str
    chunk_ids: List[str]
    section_title: str
    text: str
    image_urls: List[str]
    goal: str
    page_idxs: List[int]
    start_char: int
    end_char: int


class DocToolkit:
    def __init__(self, cfg: AppConfig, *, doc_id: str):
        self.cfg = cfg
        self.doc_id = doc_id
        self.canonical_path = cfg.canonical_dir / doc_id / "canonical.json"
        self.index_path = cfg.indexes_dir / doc_id / "chunks.sqlite3"

        if not self.canonical_path.exists():
            raise FileNotFoundError(f"Canonical doc not found: {self.canonical_path}")

        self._canonical = load_canonical(self.canonical_path)
        self._chunks_by_id = {c.chunk_id: c for c in self._canonical.chunks}
        self._sections_by_id = {s.section_id: s for s in getattr(self._canonical, "sections", []) or []}
        self._chunk_ids_by_section: Dict[str, List[str]] = {}
        for c in self._canonical.chunks:
            sid = getattr(c, "section_id", "") or ""
            if not sid:
                continue
            self._chunk_ids_by_section.setdefault(sid, []).append(c.chunk_id)
        self._index = SqliteFtsIndex(self.index_path)

    def close(self) -> None:
        self._index.close()

    def get_chunk_info(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        c = self._chunks_by_id.get(chunk_id)
        if not c:
            return None
        return {
            "chunk_id": c.chunk_id,
            "section_id": getattr(c, "section_id", "") or "",
            "section_title": c.section_title,
            "page_idx": getattr(c, "page_idx", None),
            "start_char": int(getattr(c, "start_char", 0)),
            "end_char": int(getattr(c, "end_char", 0)),
            "image_urls": list(getattr(c, "image_urls", [])),
        }

    def get_section_info(self, section_id: str) -> Optional[Dict[str, Any]]:
        s = self._sections_by_id.get(section_id)
        if not s:
            return None
        chunk_ids = list(self._chunk_ids_by_section.get(section_id) or [])
        page_idxs: List[int] = []
        for cid in chunk_ids:
            c = self._chunks_by_id.get(cid)
            if c is None:
                continue
            p = getattr(c, "page_idx", None)
            if isinstance(p, int):
                page_idxs.append(p)
        return {
            "section_id": s.section_id,
            "title": s.title,
            "level": s.level,
            "parent_id": s.parent_id,
            "start_char": s.start_char,
            "end_char": s.end_char,
            "chunk_ids": chunk_ids,
            "page_idxs": sorted(set(page_idxs)),
        }

    def search(self, *, keywords: List[str], limit: int = 20) -> List[SearchResult]:
        hits = self._index.search(keywords, limit=limit)
        out: List[SearchResult] = []
        for h in hits:
            meta = self.get_chunk_info(h.chunk_id) or {}
            out.append(
                SearchResult(
                    chunk_id=h.chunk_id,
                    section_id=str(meta.get("section_id") or ""),
                    score=h.score,
                    snippet=h.snippet,
                )
            )
        return out

    def read(
        self,
        *,
        section_ids: Optional[List[str]] = None,
        chunk_ids: Optional[List[str]] = None,
        goal: str,
        max_chars: int = 8000,
        max_chunks_per_section: int = 8,
    ) -> List[ReadResult]:
        """
        Read by section_id (preferred), falling back to chunk_ids.
        Returns one aggregated ReadResult per section.
        """
        wanted_sections = [str(s).strip() for s in (section_ids or []) if str(s).strip()]
        wanted_chunks = [str(c).strip() for c in (chunk_ids or []) if str(c).strip()]

        # If only chunk_ids were provided, map them to sections.
        if not wanted_sections and wanted_chunks:
            for cid in wanted_chunks:
                meta = self.get_chunk_info(cid) or {}
                sid = str(meta.get("section_id") or "").strip()
                if sid and sid not in wanted_sections:
                    wanted_sections.append(sid)

        out: List[ReadResult] = []
        for sid in wanted_sections:
            sec = self._sections_by_id.get(sid)
            chunk_ids_for_section = list(self._chunk_ids_by_section.get(sid) or [])
            # If the caller provided explicit chunk_ids, treat them as a hard filter so we can
            # read a single chunk (useful for "easy = single-chunk" constraints).
            if wanted_chunks:
                filtered = [c for c in chunk_ids_for_section if c in wanted_chunks]
                if filtered:
                    chunk_ids_for_section = filtered
            if not chunk_ids_for_section:
                # Fallback: if we don't know this section, still try direct chunk reads for that doc.
                if wanted_chunks:
                    chunk_ids_for_section = [c for c in wanted_chunks if c in self._chunks_by_id][:max_chunks_per_section]
                if not chunk_ids_for_section:
                    continue

            parts: List[str] = []
            image_urls: List[str] = []
            page_idxs: List[int] = []
            start_char = None
            end_char = None
            used_chunk_ids: List[str] = []
            for cid in chunk_ids_for_section[:max_chunks_per_section]:
                c = self._chunks_by_id.get(cid)
                if not c:
                    continue
                used_chunk_ids.append(cid)
                parts.append(str(c.text or "").strip())
                for u in getattr(c, "image_urls", []) or []:
                    if u not in image_urls:
                        image_urls.append(u)
                p = getattr(c, "page_idx", None)
                if isinstance(p, int):
                    page_idxs.append(p)
                sc = int(getattr(c, "start_char", 0))
                ec = int(getattr(c, "end_char", 0))
                start_char = sc if start_char is None else min(start_char, sc)
                end_char = ec if end_char is None else max(end_char, ec)

            text = "\n\n".join([p for p in parts if p]).strip()
            if not text:
                continue
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n…(truncated)…"

            title = (sec.title if sec else "") or (self.get_chunk_info(used_chunk_ids[0]) or {}).get("section_title") or ""
            out.append(
                ReadResult(
                    section_id=sid,
                    chunk_ids=used_chunk_ids,
                    section_title=str(title),
                    text=text,
                    image_urls=image_urls,
                    goal=goal,
                    page_idxs=sorted(set(page_idxs)),
                    start_char=int(start_char or 0),
                    end_char=int(end_char or 0),
                )
            )

        return out

    def dump_debug(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "canonical_path": str(self.canonical_path),
            "index_path": str(self.index_path),
            "chunks": len(self._canonical.chunks),
            "sections": len(getattr(self._canonical, "sections", []) or []),
        }

    @staticmethod
    def load_canonical_json(path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))
