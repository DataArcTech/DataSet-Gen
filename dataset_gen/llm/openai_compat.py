import json
import os
import time
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import requests


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: Union[str, List[Dict[str, Any]]]


def _join_url(base: str, suffix: str) -> str:
    return base.rstrip("/") + "/" + suffix.lstrip("/")


def build_chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return _join_url(base, "chat/completions")
    return _join_url(base, "v1/chat/completions")


class OpenAICompatChatClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_s: int = 120,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = int(timeout_s)
        self.session = requests.Session()

    @staticmethod
    def from_env(timeout_s: int = 120, *, model: Optional[str] = None) -> "OpenAICompatChatClient":
        api_key = os.environ.get("OPENAI_API_KEY") or ""
        base_url = os.environ.get("OPENAI_BASE_URL") or ""
        env_model = os.environ.get("OPENAI_CHAT_MODEL") or ""
        final_model = model or env_model
        if not api_key or not base_url or not final_model:
            missing = [
                k
                for k, v in [
                    ("OPENAI_API_KEY", api_key),
                    ("OPENAI_BASE_URL", base_url),
                    ("OPENAI_CHAT_MODEL", env_model),
                ]
                if not v
            ]
            raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")
        return OpenAICompatChatClient(api_key=api_key, base_url=base_url, model=final_model, timeout_s=timeout_s)

    def chat(
        self,
        *,
        messages: List[ChatMessage],
        temperature: float = 0.6,
        max_tokens: int = 1024,
        response_format_json: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        url = build_chat_completions_url(self.base_url)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}
        if extra:
            payload.update(extra)

        # Basic retry for transient HTTP/network errors (long runs can hit occasional disconnects).
        max_tries = int(os.environ.get("OPENAI_HTTP_MAX_RETRIES", "6"))
        base_sleep = float(os.environ.get("OPENAI_HTTP_RETRY_BASE_S", "1.0"))
        last_exc: Optional[Exception] = None

        for attempt in range(1, max_tries + 1):
            try:
                resp = self.session.post(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    timeout=self.timeout_s,
                )
                # Retry common transient statuses.
                if resp.status_code in (408, 409, 429, 500, 502, 503, 504):
                    if attempt >= max_tries:
                        resp.raise_for_status()
                    sleep_s = min(20.0, base_sleep * (2 ** (attempt - 1))) + random.random() * 0.25
                    time.sleep(sleep_s)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= max_tries:
                    raise
                sleep_s = min(20.0, base_sleep * (2 ** (attempt - 1))) + random.random() * 0.25
                time.sleep(sleep_s)
                continue
        else:
            raise RuntimeError(f"OpenAI-compatible request failed after retries: {last_exc}")

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"No choices returned: {data}")
        msg = (choices[0].get("message") or {}).get("content")
        if not isinstance(msg, str):
            raise RuntimeError(f"Invalid response: {data}")
        return msg
