import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from requests import Response
import os
import random


@dataclass(frozen=True)
class MinerUParseRequest:
    backend: str = "vlm-transformers"
    parse_method: str = "auto"
    lang: str = "ch"
    formula_enable: bool = True
    table_enable: bool = True
    start_page: int = 0
    end_page: Optional[int] = None
    output_format: str = "mm_md"  # mm_md | md_only | content_list
    caption_mode: Optional[str] = None
    caption_max_images: Optional[int] = None


class MinerUHttpClient:
    def __init__(self, base_url: str, timeout_s: int = 900):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = int(timeout_s)
        self.session = requests.Session()

    def health(self) -> Dict[str, Any]:
        resp = self.session.get(f"{self.base_url}/health", timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def _post_multipart(self, url: str, *, file_path: Path, data: Dict[str, str]) -> Response:
        with file_path.open("rb") as f:
            resp = self.session.post(
                url,
                data=data,
                files={"file": (file_path.name, f, "application/octet-stream")},
                timeout=self.timeout_s,
            )
        return resp

    def _progress(self, msg: str) -> None:
        if str(os.environ.get("MINERU_CLIENT_PROGRESS", "0")).lower() in ("1", "true", "yes", "y"):
            print(msg, file=os.sys.stderr, flush=True)

    def _request_with_retries(self, fn, *, what: str, max_tries: int = 6) -> Response:
        base_sleep = float(os.environ.get("MINERU_CLIENT_RETRY_BASE_S", "1.0"))
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_tries + 1):
            try:
                return fn()
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= max_tries:
                    raise
                sleep_s = min(20.0, base_sleep * (2 ** (attempt - 1))) + random.random() * 0.25
                self._progress(f"[mineru_client] {what} failed (attempt {attempt}/{max_tries}), retrying in {sleep_s:.1f}s: {exc}")
                time.sleep(sleep_s)
        raise RuntimeError(f"{what} failed after retries: {last_exc}")

    def get_result(self, task_id: str) -> Dict[str, Any]:
        """
        Preferred for long parses: returns either final ParseResult or a status dict.
        """
        resp = self._request_with_retries(
            lambda: self.session.get(f"{self.base_url}/task/{task_id}/result", timeout=self.timeout_s),
            what=f"GET /task/{task_id}/result",
        )
        resp.raise_for_status()
        return resp.json()

    def parse(self, *, file_path: Path, req: MinerUParseRequest) -> Dict[str, Any]:
        data = {
            "backend": req.backend,
            "parse_method": req.parse_method,
            "lang": req.lang,
            "formula_enable": "true" if req.formula_enable else "false",
            "table_enable": "true" if req.table_enable else "false",
            "start_page": str(req.start_page),
            "output_format": req.output_format,
        }
        if req.end_page is not None:
            data["end_page"] = str(req.end_page)
        if req.caption_mode:
            data["caption_mode"] = str(req.caption_mode)
        if req.caption_max_images is not None:
            data["caption_max_images"] = str(int(req.caption_max_images))
        started = time.perf_counter()

        # Prefer async parse (avoid long-lived HTTP connections that can be dropped by proxies).
        # Fallback to synchronous /parse if the server doesn't support /parse_async.
        async_url = f"{self.base_url}/parse_async"
        sync_url = f"{self.base_url}/parse"

        resp: Optional[Response] = None
        try:
            # Submit async job (quick request). Retry transient failures a few times.
            resp = self._request_with_retries(
                lambda: self._post_multipart(async_url, file_path=file_path, data=data),
                what="POST /parse_async",
                max_tries=4,
            )
            if resp.status_code == 404:
                raise requests.HTTPError("parse_async_not_supported", response=resp)
            resp.raise_for_status()

            queued = resp.json()
            task_id = str(queued.get("task_id") or "")
            if not task_id:
                raise RuntimeError(f"Missing task_id from /parse_async: {queued}")
            self._progress(f"[mineru_client] submitted parse_async task_id={task_id} file={file_path.name}")

            # Poll until we get a terminal ParseResult.
            # Use timeout_s as the overall wall-clock budget.
            deadline = started + float(self.timeout_s)
            backoff = 1.0
            last_print = 0.0
            while True:
                now = time.perf_counter()
                if now > deadline:
                    raise TimeoutError(f"MinerU async parse timed out after {self.timeout_s}s (task_id={task_id})")
                try:
                    result = self.get_result(task_id)
                except Exception:
                    # Transient network errors: backoff and retry.
                    time.sleep(min(10.0, backoff))
                    backoff = min(10.0, backoff * 1.5)
                    continue

                status = str(result.get("status") or "")
                if now - last_print >= 30.0:
                    elapsed = now - started
                    self._progress(f"[mineru_client] task_id={task_id} status={status} elapsed={elapsed:.0f}s")
                    last_print = now
                if status in {"success", "failed"}:
                    result["_client_wall_time_s"] = time.perf_counter() - started
                    result["_client_parse_mode"] = "async"
                    return result

                time.sleep(min(10.0, backoff))
                backoff = min(10.0, backoff * 1.2)

        except requests.HTTPError as exc:
            # Only fall back when server doesn't implement /parse_async.
            if getattr(exc, "response", None) is not None and exc.response is not None and exc.response.status_code == 404:
                self._progress("[mineru_client] /parse_async not supported; falling back to /parse (long-lived request)")
                resp2 = self._request_with_retries(
                    lambda: self._post_multipart(sync_url, file_path=file_path, data=data),
                    what="POST /parse",
                    max_tries=2,
                )
                resp2.raise_for_status()
                out = resp2.json()
                out["_client_wall_time_s"] = time.perf_counter() - started
                out["_client_parse_mode"] = "sync"
                return out
            raise

    def get_manifest(self, task_id: str) -> Dict[str, Any]:
        resp = self._request_with_retries(
            lambda: self.session.get(f"{self.base_url}/task/{task_id}/manifest", timeout=self.timeout_s),
            what=f"GET /task/{task_id}/manifest",
        )
        resp.raise_for_status()
        return resp.json()

    def download_task_file(self, task_id: str, rel_path: str, dst: Path) -> None:
        rel_path = str(rel_path).lstrip("/")
        resp = self._request_with_retries(
            lambda: self.session.get(
                f"{self.base_url}/task/{task_id}/file/{rel_path}",
                timeout=self.timeout_s,
                stream=True,
            ),
            what=f"GET /task/{task_id}/file/{rel_path}",
        )
        resp.raise_for_status()
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("wb") as out:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out.write(chunk)

    def sync_task(self, task_id: str, dst_root: Path) -> Path:
        manifest = self.get_manifest(task_id)
        files = manifest.get("files", [])
        local_task_root = dst_root / task_id
        total = len(files) if isinstance(files, list) else 0
        for i, entry in enumerate(files if isinstance(files, list) else []):
            rel_path = entry.get("path")
            if not rel_path:
                continue
            if i % 25 == 0:
                self._progress(f"[mineru_client] syncing task_id={task_id}: {i}/{total} files")
            self.download_task_file(task_id, rel_path, local_task_root / rel_path)
        self._progress(f"[mineru_client] sync done task_id={task_id}: {total}/{total} files")
        return local_task_root
