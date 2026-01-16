import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PageRenderConfig:
    dpi: int = 110
    image_format: str = "png"


@dataclass(frozen=True)
class CropConfig:
    enabled: bool = False
    table: bool = True
    figure: bool = True
    image: bool = False
    padding_px: int = 6


def _iter_blocks_v2(content_list_v2: Any) -> List[List[Dict[str, Any]]]:
    """
    MinerU content_list_v2 is typically: list[page] where page is list[block].
    """
    if isinstance(content_list_v2, list) and content_list_v2 and isinstance(content_list_v2[0], list):
        pages = []
        for page in content_list_v2:
            if isinstance(page, list):
                pages.append([b for b in page if isinstance(b, dict)])
        return pages
    return []


def _page_bbox_extent(blocks: Sequence[Dict[str, Any]]) -> Tuple[float, float]:
    mx = 0.0
    my = 0.0
    for b in blocks:
        bb = b.get("bbox")
        if isinstance(bb, list) and len(bb) == 4:
            try:
                mx = max(mx, float(bb[2]))
                my = max(my, float(bb[3]))
            except Exception:
                continue
    return mx, my


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def generate_page_screenshots(
    *,
    source_pdf_path: Path,
    out_dir: Path,
    start_page: int = 0,
    end_page: Optional[int] = None,
    render: PageRenderConfig = PageRenderConfig(),
) -> List[Dict[str, Any]]:
    """
    Render PDF pages to images using PyMuPDF (fitz).
    Output filenames: page_<1-based, zero-padded 4>.png
    """
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyMuPDF (fitz) is required for page screenshot generation") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(source_pdf_path))
    try:
        total = int(doc.page_count)
        s = max(0, int(start_page))
        e = int(end_page) if end_page is not None else (total - 1)
        e = min(total - 1, max(s, e))

        zoom = float(render.dpi) / 72.0
        mat = fitz.Matrix(zoom, zoom)
        results: List[Dict[str, Any]] = []
        for page_idx in range(s, e + 1):
            page = doc.load_page(page_idx)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            filename = f"page_{page_idx + 1:04d}.{render.image_format}"
            dst = out_dir / filename
            pix.save(str(dst))
            results.append(
                {
                    "page_idx": page_idx,
                    "filename": filename,
                    "relative_path": f"{out_dir.name}/{filename}",
                    "absolute_path": str(dst),
                    "width": int(getattr(pix, "width", 0) or 0),
                    "height": int(getattr(pix, "height", 0) or 0),
                    "dpi": int(render.dpi),
                }
            )
        return results
    finally:
        doc.close()


def generate_block_crops_from_page_images(
    *,
    page_images: List[Dict[str, Any]],
    content_list_v2: Any,
    out_dir: Path,
    crop: CropConfig,
) -> List[Dict[str, Any]]:
    """
    Crop table/figure/image blocks using content_list_v2 bbox coordinates.

    Notes:
    - Requires Pillow for decoding and saving crops.
    - Uses a heuristic scaling: bbox coords are assumed proportional to the rendered page image,
      using max bbox extent per page as the reference coordinate system.
    """
    if not crop.enabled:
        return []

    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError("Pillow (PIL) is required for crop generation") from exc

    pages = _iter_blocks_v2(content_list_v2)
    if not pages:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    images_by_page: Dict[int, Path] = {}
    for entry in page_images:
        page_idx = _safe_int(entry.get("page_idx"), -1)
        abs_path = entry.get("absolute_path")
        if page_idx >= 0 and isinstance(abs_path, str) and abs_path:
            images_by_page[page_idx] = Path(abs_path)

    results: List[Dict[str, Any]] = []
    counters: Dict[Tuple[int, str], int] = {}

    for page_idx, blocks in enumerate(pages):
        page_img_path = images_by_page.get(page_idx)
        if not page_img_path or not page_img_path.exists():
            continue
        extent_x, extent_y = _page_bbox_extent(blocks)
        if extent_x <= 0 or extent_y <= 0:
            continue

        with Image.open(page_img_path) as im:
            w, h = im.size
            sx = float(w) / float(extent_x)
            sy = float(h) / float(extent_y)

            for b in blocks:
                btype = str(b.get("type") or "")
                if btype == "table" and not crop.table:
                    continue
                if btype in ("figure",) and not crop.figure:
                    continue
                if btype in ("image",) and not crop.image:
                    continue
                if btype not in ("table", "figure", "image"):
                    continue

                bb = b.get("bbox")
                if not (isinstance(bb, list) and len(bb) == 4):
                    continue
                try:
                    x0, y0, x1, y1 = [float(x) for x in bb]
                except Exception:
                    continue

                pad = int(max(0, crop.padding_px))
                left = max(0, int(x0 * sx) - pad)
                top = max(0, int(y0 * sy) - pad)
                right = min(w, int(x1 * sx) + pad)
                bottom = min(h, int(y1 * sy) + pad)
                if right - left < 10 or bottom - top < 10:
                    continue

                key = (page_idx, btype)
                counters[key] = counters.get(key, 0) + 1
                idx = counters[key]
                filename = f"crop_{btype}_p{page_idx + 1:04d}_{idx:03d}.png"
                dst = out_dir / filename
                im.crop((left, top, right, bottom)).save(dst, format="PNG")

                results.append(
                    {
                        "page_idx": page_idx,
                        "type": btype,
                        "bbox": bb,
                        "filename": filename,
                        "relative_path": f"{out_dir.name}/{filename}",
                        "absolute_path": str(dst),
                    }
                )

    return results


def load_json(path: Optional[Path]) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

