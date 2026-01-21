import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from dataset_gen.config import AppConfig

try:  # Unix-only
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class DocRecord:
    doc_id: str
    filename: str
    source_path: str
    sha256: str
    created_at: float
    updated_at: float
    mineru_task_id: Optional[str] = None
    mineru_task_local_dir: Optional[str] = None
    mineru_markdown_path: Optional[str] = None
    mineru_content_list_path: Optional[str] = None
    mineru_asset_manifest_path: Optional[str] = None
    canonical_path: Optional[str] = None
    index_path: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class DocStore:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

    def _lock_path(self) -> Path:
        return self.cfg.metadata_path.with_suffix(self.cfg.metadata_path.suffix + ".lock")

    def _with_lock(self, *, shared: bool = False):
        """
        Cross-process lock for doc_store.json updates.
        This allows running multiple `dataset_gen ingest` processes in parallel safely.
        """
        if fcntl is None:
            # Best-effort fallback (no locking on non-Unix platforms).
            from contextlib import nullcontext

            return nullcontext()
        lock_path = self._lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        class _LockCtx:
            def __enter__(self_nonlocal):
                self_nonlocal.f = lock_path.open("w")
                lock_kind = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
                fcntl.flock(self_nonlocal.f.fileno(), lock_kind)
                return self_nonlocal

            def __exit__(self_nonlocal, exc_type, exc, tb):
                try:
                    fcntl.flock(self_nonlocal.f.fileno(), fcntl.LOCK_UN)
                finally:
                    self_nonlocal.f.close()
                return False

        return _LockCtx()

    def _load(self) -> Dict[str, Any]:
        if not self.cfg.metadata_path.exists():
            return {"version": 1, "docs": {}}
        return json.loads(self.cfg.metadata_path.read_text(encoding="utf-8"))

    def _save(self, data: Dict[str, Any]) -> None:
        # Atomic write to avoid partial files if the process is interrupted.
        tmp = self.cfg.metadata_path.with_suffix(self.cfg.metadata_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.cfg.metadata_path)

    def upsert_source(self, source_pdf: Path) -> DocRecord:
        source_pdf = source_pdf.resolve()
        digest = sha256_file(source_pdf)
        doc_id = digest[:16]
        now = time.time()
        with self._with_lock(shared=False):
            data = self._load()
            docs = data.setdefault("docs", {})
            existing = docs.get(doc_id)
            if existing:
                existing["updated_at"] = now
                self._save(data)
                return DocRecord(**existing)

            rec = DocRecord(
                doc_id=doc_id,
                filename=source_pdf.name,
                source_path=str(source_pdf),
                sha256=digest,
                created_at=now,
                updated_at=now,
                extra={},
            )
            docs[doc_id] = asdict(rec)
            self._save(data)
            return rec

    def update_doc(self, doc_id: str, **fields: Any) -> DocRecord:
        with self._with_lock(shared=False):
            data = self._load()
            docs = data.setdefault("docs", {})
            if doc_id not in docs:
                raise KeyError(f"Unknown doc_id: {doc_id}")
            docs[doc_id].update(fields)
            docs[doc_id]["updated_at"] = time.time()
            self._save(data)
            return DocRecord(**docs[doc_id])

    def get_doc(self, doc_id: str) -> Dict[str, Any]:
        # Best-effort lock: allows consistent reads while parallel ingests are running.
        with self._with_lock(shared=True):
            data = self._load()
            doc = (data.get("docs") or {}).get(doc_id)
            if not doc:
                raise KeyError(f"Unknown doc_id: {doc_id}")
            return doc

    def list_docs(self) -> Dict[str, Dict[str, Any]]:
        with self._with_lock(shared=True):
            data = self._load()
            docs = data.get("docs") or {}
            if not isinstance(docs, dict):
                return {}
            return docs
