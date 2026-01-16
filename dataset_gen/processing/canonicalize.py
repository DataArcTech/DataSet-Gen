import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_HEADING_RE = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$", re.MULTILINE)
_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<url>[^)]+)\)")
_WS_RE = re.compile(r"\s+")


@dataclass
class CanonicalSection:
    section_id: str
    title: str
    level: int
    parent_id: Optional[str]
    start_char: int
    end_char: int

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CanonicalSection":
        pid = d.get("parent_id")
        return CanonicalSection(
            section_id=str(d.get("section_id") or ""),
            title=str(d.get("title") or ""),
            level=int(d.get("level") or 0),
            parent_id=(str(pid) if pid is not None else None),
            start_char=int(d.get("start_char") or 0),
            end_char=int(d.get("end_char") or 0),
        )


@dataclass
class CanonicalChunk:
    chunk_id: str
    section_id: str
    section_title: str
    heading_level: int
    text: str
    start_char: int
    end_char: int
    image_urls: List[str]
    page_idx: Optional[int] = None
    element_types: Optional[List[str]] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CanonicalChunk":
        return CanonicalChunk(
            chunk_id=str(d.get("chunk_id") or ""),
            section_id=str(d.get("section_id") or ""),
            section_title=str(d.get("section_title") or ""),
            heading_level=int(d.get("heading_level") or 0),
            text=str(d.get("text") or ""),
            start_char=int(d.get("start_char") or 0),
            end_char=int(d.get("end_char") or 0),
            image_urls=list(d.get("image_urls") or []),
            page_idx=(int(d["page_idx"]) if d.get("page_idx") is not None else None),
            element_types=(list(d.get("element_types") or []) or None),
        )


@dataclass
class CanonicalDoc:
    doc_id: str
    source_markdown: str
    sections: List[CanonicalSection]
    chunks: List[CanonicalChunk]
    metadata: Dict[str, Any]


def _iter_sections(md: str) -> List[Dict[str, Any]]:
    matches = list(_HEADING_RE.finditer(md))
    if not matches:
        return [{"title": "", "level": 0, "start": 0, "end": len(md)}]
    sections: List[Dict[str, Any]] = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(md)
        sections.append({"title": m.group("title").strip(), "level": len(m.group("level")), "start": start, "end": end})
    if sections and sections[0]["start"] > 0:
        sections.insert(0, {"title": "", "level": 0, "start": 0, "end": sections[0]["start"]})
    return sections


def _build_section_tree(doc_id: str, md: str) -> List[CanonicalSection]:
    """
    Build a hierarchical outline (section tree) from Markdown headings.

    Notes:
    - This does not try to "understand" MinerU layout JSON.
    - It creates stable section_id values based on heading order.
    """
    entries = _iter_sections(md)
    sections: List[CanonicalSection] = []
    stack: List[CanonicalSection] = []
    for idx, ent in enumerate(entries):
        level = int(ent.get("level") or 0)
        title = str(ent.get("title") or "").strip()
        sid = f"{doc_id}_sec{idx}"

        while stack and stack[-1].level >= level and level > 0:
            stack.pop()
        parent_id = stack[-1].section_id if (stack and level > 0) else None

        sec = CanonicalSection(
            section_id=sid,
            title=title,
            level=level,
            parent_id=parent_id,
            start_char=int(ent.get("start") or 0),
            end_char=int(ent.get("end") or 0),
        )
        sections.append(sec)
        if level > 0:
            stack.append(sec)
    return sections


def _load_content_list_items(content_list_path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(content_list_path.read_text(encoding="utf-8", errors="ignore"))
    if not isinstance(obj, list):
        return []
    # MinerU may output:
    # - v1: list[dict]
    # - v2: list[list[dict]]
    if obj and isinstance(obj[0], list):
        flat: List[Dict[str, Any]] = []
        for sub in obj:
            if isinstance(sub, list):
                flat.extend([x for x in sub if isinstance(x, dict)])
        return flat
    return [x for x in obj if isinstance(x, dict)]


def _strip_noise(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").replace("\\", "").strip())


def _find_anchor_pos(md: str, anchor: str) -> Optional[int]:
    if not md or not anchor:
        return None
    hay = md.lower()
    needle = _strip_noise(anchor).lower()
    if not needle:
        return None
    idx = hay.find(needle)
    if idx >= 0:
        return idx
    prefix = needle[:80].strip()
    if len(prefix) >= 12:
        idx = hay.find(prefix)
        if idx >= 0:
            return idx
    return None


def _page_anchor_positions(md: str, items: List[Dict[str, Any]]) -> Dict[int, int]:
    """
    Try to map content_list page_idx -> markdown char positions, using early text anchors per page.
    """
    by_page: Dict[int, List[str]] = {}
    for it in items:
        try:
            page_idx = int(it.get("page_idx"))
        except Exception:
            continue
        t = str(it.get("type") or "")
        if t != "text":
            continue
        txt = it.get("text")
        if not isinstance(txt, str) or len(txt.strip()) < 18:
            continue
        by_page.setdefault(page_idx, []).append(_strip_noise(txt))

    out: Dict[int, int] = {}
    for page_idx, texts in by_page.items():
        anchor = ""
        for t in texts[:3]:
            if len(t) >= 18:
                anchor = t
                break
        if not anchor:
            continue
        pos = _find_anchor_pos(md, anchor)
        if pos is not None:
            out[page_idx] = pos
    return out


def _section_for_pos(sections: List[CanonicalSection], pos: int) -> Optional[CanonicalSection]:
    if not sections:
        return None
    best: Optional[CanonicalSection] = None
    for s in sections:
        if s.start_char <= pos < s.end_char:
            if best is None or s.level >= best.level:
                best = s
    return best or sections[0]


def _canonicalize_from_content_list(
    *,
    doc_id: str,
    markdown_path: Optional[Path],
    content_list_path: Path,
    max_chars_per_chunk: int,
) -> CanonicalDoc:
    items = _load_content_list_items(content_list_path)
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for it in items:
        page_idx = it.get("page_idx")
        if page_idx is None:
            continue
        try:
            page = int(page_idx)
        except Exception:
            continue
        by_page.setdefault(page, []).append(it)

    md = markdown_path.read_text(encoding="utf-8", errors="ignore") if markdown_path and markdown_path.exists() else ""
    metadata: Dict[str, Any] = {
        "markdown_path": str(markdown_path) if markdown_path else None,
        "content_list_path": str(content_list_path),
    }

    sections = _build_section_tree(doc_id, md) if md else []
    page_pos = _page_anchor_positions(md, items) if md and items else {}

    chunks: List[CanonicalChunk] = []
    start_char = 0
    for page_idx in sorted(by_page.keys()):
        page_items = by_page[page_idx]
        texts: List[str] = []
        element_types: List[str] = []
        image_urls: List[str] = []

        for it in page_items:
            t = str(it.get("type") or "")
            if t:
                element_types.append(t)
            if t == "text":
                txt = it.get("text")
                if isinstance(txt, str) and txt.strip():
                    texts.append(txt.strip())
            elif t == "image":
                # Prefer MinerU captions if available.
                cap = it.get("image_caption") or it.get("caption") or it.get("text")
                if isinstance(cap, str) and cap.strip():
                    texts.append(f"[图像说明] {cap.strip()}")
                img_path = it.get("img_path") or it.get("image_path") or ""
                if isinstance(img_path, str) and img_path.strip():
                    image_urls.append(f"images/{Path(img_path).name}")
            elif t == "table":
                # Some MinerU outputs include table text; if present, include as text.
                txt = it.get("text")
                if isinstance(txt, str) and txt.strip():
                    texts.append(f"[表格] {txt.strip()}")

        page_text = "\n".join(texts).strip()
        if not page_text:
            continue
        if len(page_text) > max_chars_per_chunk:
            page_text = page_text[:max_chars_per_chunk] + "\n…(truncated)…"

        chunk_id = f"{doc_id}_p{page_idx}"
        section_id = ""
        section_title = f"Page {page_idx + 1}"
        heading_level = 0
        if sections and md:
            pos = page_pos.get(page_idx)
            if pos is None:
                # Fallback: try to locate a short snippet from this page text in markdown.
                snippet = _strip_noise(page_text)[:120]
                pos = _find_anchor_pos(md, snippet) if snippet else None
            if pos is not None:
                sec = _section_for_pos(sections, pos)
                if sec:
                    section_id = sec.section_id
                    section_title = sec.title
                    heading_level = int(sec.level)
        if not section_id:
            # Ensure non-empty section_id for downstream tools.
            section_id = f"{doc_id}_sec0"
            if not sections:
                sections = [
                    CanonicalSection(
                        section_id=section_id,
                        title="",
                        level=0,
                        parent_id=None,
                        start_char=0,
                        end_char=len(md),
                    )
                ]

        chunks.append(
            CanonicalChunk(
                chunk_id=chunk_id,
                section_id=section_id,
                section_title=section_title,
                heading_level=heading_level,
                text=page_text,
                start_char=start_char,
                end_char=start_char + len(page_text),
                image_urls=sorted(set(image_urls)),
                page_idx=page_idx,
                element_types=sorted(set(element_types)) if element_types else None,
            )
        )
        start_char += len(page_text) + 1

    return CanonicalDoc(doc_id=doc_id, source_markdown=md, sections=sections, chunks=chunks, metadata=metadata)


def canonicalize_markdown(
    *,
    doc_id: str,
    markdown_path: Path,
    content_list_path: Optional[Path],
    max_chars_per_chunk: int = 4000,
) -> CanonicalDoc:
    # Prefer MinerU content_list for stable page-aware chunking (closer to DocDancer's section/page abstraction).
    if content_list_path and content_list_path.exists():
        return _canonicalize_from_content_list(
            doc_id=doc_id,
            markdown_path=markdown_path,
            content_list_path=content_list_path,
            max_chars_per_chunk=max_chars_per_chunk,
        )

    md = markdown_path.read_text(encoding="utf-8", errors="ignore")
    metadata: Dict[str, Any] = {"markdown_path": str(markdown_path)}

    sections = _build_section_tree(doc_id, md) if md else []
    if not sections:
        sections = [
            CanonicalSection(
                section_id=f"{doc_id}_sec0",
                title="",
                level=0,
                parent_id=None,
                start_char=0,
                end_char=len(md),
            )
        ]

    chunks: List[CanonicalChunk] = []
    section_entries = _iter_sections(md)
    for s_idx, s in enumerate(section_entries):
        s_text = md[s["start"] : s["end"]].strip()
        if not s_text:
            continue
        images = [m.group("url").strip() for m in _IMAGE_RE.finditer(s_text)]
        if len(s_text) <= max_chars_per_chunk:
            chunk_id = f"{doc_id}_s{s_idx}_c0"
            chunks.append(
                CanonicalChunk(
                    chunk_id=chunk_id,
                    section_id=f"{doc_id}_sec{s_idx}",
                    section_title=s["title"],
                    heading_level=int(s["level"]),
                    text=s_text,
                    start_char=int(s["start"]),
                    end_char=int(s["end"]),
                    image_urls=images,
                )
            )
            continue

        # Fallback: split by paragraphs.
        paras = [p.strip() for p in re.split(r"\n\s*\n+", s_text) if p.strip()]
        buf: List[str] = []
        buf_len = 0
        c_idx = 0
        start_char = s["start"]
        for p in paras:
            if buf_len + len(p) + 2 > max_chars_per_chunk and buf:
                text = "\n\n".join(buf)
                chunk_id = f"{doc_id}_s{s_idx}_c{c_idx}"
                images = [m.group("url").strip() for m in _IMAGE_RE.finditer(text)]
                chunks.append(
                    CanonicalChunk(
                        chunk_id=chunk_id,
                        section_id=f"{doc_id}_sec{s_idx}",
                        section_title=s["title"],
                        heading_level=int(s["level"]),
                        text=text,
                        start_char=start_char,
                        end_char=start_char + len(text),
                        image_urls=images,
                    )
                )
                c_idx += 1
                buf = []
                buf_len = 0
                start_char = start_char + len(text)

            buf.append(p)
            buf_len += len(p) + 2

        if buf:
            text = "\n\n".join(buf)
            chunk_id = f"{doc_id}_s{s_idx}_c{c_idx}"
            images = [m.group("url").strip() for m in _IMAGE_RE.finditer(text)]
            chunks.append(
                CanonicalChunk(
                    chunk_id=chunk_id,
                    section_id=f"{doc_id}_sec{s_idx}",
                    section_title=s["title"],
                    heading_level=int(s["level"]),
                    text=text,
                    start_char=start_char,
                    end_char=start_char + len(text),
                    image_urls=images,
                )
            )

    return CanonicalDoc(doc_id=doc_id, source_markdown=md, sections=sections, chunks=chunks, metadata=metadata)


def write_canonical(doc: CanonicalDoc, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "doc_id": doc.doc_id,
        "metadata": doc.metadata,
        "sections": [asdict(s) for s in (doc.sections or [])],
        "chunks": [asdict(c) for c in doc.chunks],
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def load_canonical(path: Path) -> CanonicalDoc:
    raw = json.loads(path.read_text(encoding="utf-8"))
    sections = [
        CanonicalSection.from_dict(s) for s in (raw.get("sections") or []) if isinstance(s, dict) and s.get("section_id")
    ]
    chunks = [CanonicalChunk.from_dict(c) for c in (raw.get("chunks") or []) if isinstance(c, dict)]
    if not sections:
        # Back-compat: canonical.json generated before section tree existed.
        fallback_id = f"{raw['doc_id']}_sec0"
        sections = [
            CanonicalSection(
                section_id=fallback_id,
                title="",
                level=0,
                parent_id=None,
                start_char=0,
                end_char=0,
            )
        ]
        for c in chunks:
            if not getattr(c, "section_id", ""):
                c.section_id = fallback_id  # type: ignore[misc]
    return CanonicalDoc(
        doc_id=raw["doc_id"],
        source_markdown="",
        sections=sections,
        chunks=chunks,
        metadata=raw.get("metadata") or {},
    )
