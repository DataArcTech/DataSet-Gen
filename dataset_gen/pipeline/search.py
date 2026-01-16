from typing import Any, Dict, List

from dataset_gen.config import AppConfig
from dataset_gen.toolkit.doc_toolkit import DocToolkit


def search_doc(cfg: AppConfig, *, doc_id: str, keywords: List[str], limit: int = 20) -> Dict[str, Any]:
    tk = DocToolkit(cfg, doc_id=doc_id)
    try:
        hits = tk.search(keywords=keywords, limit=limit)
        return {
            "doc": tk.dump_debug(),
            "keywords": keywords,
            "hits": [
                {"chunk_id": h.chunk_id, "section_id": h.section_id, "score": h.score, "snippet": h.snippet} for h in hits
            ],
        }
    finally:
        tk.close()
