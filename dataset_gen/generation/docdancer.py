import json
import random
import re
import time
import os
import base64
import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from dataset_gen.config import AppConfig
from dataset_gen.llm.openai_compat import ChatMessage, OpenAICompatChatClient
from dataset_gen.storage.doc_store import DocStore
from dataset_gen.toolkit.doc_toolkit import DocToolkit, ReadResult, SearchResult
from dataset_gen.tools.code_python import PythonSandboxLimits, run_code_python
from dataset_gen.prompts.docdancer import DocDancerPrompts, PromptLang


Difficulty = Literal["easy", "hard", "unanswerable"]
ItemKind = Literal["qa", "calc"]


@dataclass
class ToolCall:
    intent: str
    tool: Literal["search", "read", "finish"]
    args: Dict[str, Any]


@dataclass
class TrajectoryStep:
    step: int
    intent: str
    tool: str
    args: Dict[str, Any]
    observation: Dict[str, Any]


@dataclass
class GeneratedItem:
    question: str
    answer: str
    difficulty: Difficulty
    kind: ItemKind
    used_doc_ids: List[str]
    evidence_section_ids: List[str]
    evidence_chunk_ids: List[str]
    trajectory: List[TrajectoryStep]
    derived: Optional[Dict[str, Any]] = None


def _extract_first_json(text: str) -> Dict[str, Any]:
    """
    Robust-ish: find first {...} JSON object inside text.
    """
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found")
    # naive brace matching
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start : i + 1]
                return json.loads(snippet)
    raise ValueError("Unclosed JSON object")


def _looks_like_table(text: str) -> bool:
    # cheap heuristic: markdown table rows
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    has_pipe = any("|" in ln for ln in lines[:8])
    has_sep = any(re.match(r"^[|]?[ \t]*-{3,}", ln) for ln in lines[:8])
    return has_pipe and has_sep


PROHIBITED_LOC_RE = re.compile(
    r"(第\s*\d+\s*页|page\s*\d+|section_id|chunk_id|Figure\s*\d+|Table\s*\d+|图\s*\d+|表\s*\d+|第\s*\d+\s*章|chapter\s*\d+)",
    re.IGNORECASE,
)

_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)(%|‰)?")
_TITLE_TERM_RE = re.compile(r"[\u4e00-\u9fffA-Za-z]{2,10}")
_TITLE_STOP = {
    "目录",
    "图像",
    "Images",
    "Table",
    "Figure",
    "附录",
    "参考",
    "References",
    "Introduction",
    "Abstract",
    "Conclusion",
    "Related",
    "Work",
}

_KW_TOKEN_RE = re.compile(r"[0-9A-Za-z\u4e00-\u9fff]{2,14}")


def _keywords_from_text(text: str, *, max_terms: int = 10) -> List[str]:
    tokens: List[str] = []
    for m in _KW_TOKEN_RE.finditer(str(text or "")):
        t = m.group(0).strip()
        if not t:
            continue
        # avoid boilerplate
        if t in _TITLE_STOP:
            continue
        if t not in tokens:
            tokens.append(t)
        if len(tokens) >= max_terms:
            break
    return tokens


def _extract_number_tokens(text: str, *, max_items: int = 50) -> List[str]:
    out: List[str] = []
    for m in _NUMBER_RE.finditer(str(text or "")):
        val = (m.group(1) or "").strip()
        suf = (m.group(2) or "").strip()
        if not val:
            continue
        token = val + suf
        if token not in out:
            out.append(token)
        if len(out) >= max_items:
            break
    return out


def _normalize_number_haystack(text: str) -> str:
    # Make it easier to match numbers regardless of thousands separators.
    return str(text or "").replace(",", "")


def _extract_inputs_keys_used(code: str) -> set[str]:
    """
    Parse code and return keys accessed as INPUTS['key'] or INPUTS[\"key\"].
    """
    keys: set[str] = set()
    try:
        tree = ast.parse(str(code or ""))
    except Exception:
        return keys

    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        v = node.value
        if not (isinstance(v, ast.Name) and v.id == "INPUTS"):
            continue

        slc = node.slice
        # Python 3.10: slice is an expr node.
        if isinstance(slc, ast.Constant) and isinstance(slc.value, str):
            keys.add(slc.value)
            continue
        # Back-compat: ast.Index(Constant)
        if isinstance(slc, ast.Index) and isinstance(slc.value, ast.Constant) and isinstance(slc.value.value, str):
            keys.add(slc.value.value)
            continue
    return keys


def _suggest_keywords_from_outline(outline: List[Dict[str, Any]], *, max_terms: int = 12) -> List[List[str]]:
    titles: List[str] = []
    for d in outline:
        for s in (d.get("sections") or []):
            if isinstance(s, dict):
                t = s.get("title")
                if isinstance(t, str) and t.strip():
                    titles.append(t.strip())

    terms: List[str] = []
    for t in titles:
        for m in _TITLE_TERM_RE.finditer(t):
            token = m.group(0).strip()
            if not token or token in _TITLE_STOP:
                continue
            if token not in terms:
                terms.append(token)
            if len(terms) >= max_terms:
                break
        if len(terms) >= max_terms:
            break

    suggestions: List[List[str]] = []
    for term in terms[:max_terms]:
        suggestions.append([term])
        if len(suggestions) >= 8:
            break
    return suggestions


class MultiDocToolkit:
    def __init__(self, cfg: AppConfig, doc_ids: List[str], *, assets: Dict[str, Dict[str, Any]]):
        self.cfg = cfg
        self.doc_ids = doc_ids
        self._tks: Dict[str, DocToolkit] = {doc_id: DocToolkit(cfg, doc_id=doc_id) for doc_id in doc_ids}
        self._assets = assets

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
            out.append(
                {
                    "doc_id": doc_id,
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
            rr = tk.read(section_ids=[sid], chunk_ids=cids, goal=goal, max_chars=max_chars)
            for r in rr:
                images_dir = (self._assets.get(doc_id) or {}).get("images_dir")
                pages_dir = (self._assets.get(doc_id) or {}).get("pages_dir")
                crops_dir = (self._assets.get(doc_id) or {}).get("crops_dir")
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
                        "has_table": _looks_like_table(r.text),
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
            out.append({"doc_id": doc_id, "sections": packed})
        return out

def _validate_calc_spec(*, difficulty: Difficulty, spec: Dict[str, Any]) -> Tuple[bool, str]:
    if set(spec.keys()) != {"question", "inputs", "code"}:
        return False, "Output must contain only question+inputs+code"
    q = spec.get("question")
    code = spec.get("code")
    inputs = spec.get("inputs")
    if not isinstance(q, str) or not q.strip():
        return False, "Empty question"
    if PROHIBITED_LOC_RE.search(q):
        return False, "Contains prohibited location references"
    if "?" not in q and "？" not in q:
        return False, "Question should contain a question mark"
    if not isinstance(code, str) or not code.strip():
        return False, "Empty code"
    if "INPUTS" not in code:
        return False, "Code must reference INPUTS"
    if not any(op in code for op in ["+", "-", "*", "/", "**"]):
        return False, "Code must include arithmetic"
    if not isinstance(inputs, dict):
        return False, "inputs must be an object"
    if difficulty == "easy" and len(inputs) < 1:
        return False, "easy calc must include at least 1 input"
    if difficulty == "hard" and len(inputs) < 2:
        return False, "hard calc must include at least 2 inputs"
    # Keep input payload small and JSON-friendly.
    if len(inputs) > 20:
        return False, "Too many inputs"
    for k, v in inputs.items():
        if not isinstance(k, str) or not k.strip():
            return False, "Invalid inputs key"
        if not isinstance(v, (int, float, str)):
            return False, "inputs values must be number or string"
    if difficulty == "unanswerable":
        return False, "Unanswerable cannot be a calc item"
    if difficulty == "hard":
        # Encourage multi-hop: require referencing at least 2 distinct INPUTS keys.
        keys_used = _extract_inputs_keys_used(code)
        if len(keys_used) < 2:
            return False, "hard calc code must use at least 2 distinct INPUTS keys"
    return True, ""


def _validate_qa(*, difficulty: Difficulty, qa: Dict[str, Any], prompt_lang: PromptLang = "en") -> Tuple[bool, str]:
    unanswerable = DocDancerPrompts(lang=prompt_lang).unanswerable_answer()
    if set(qa.keys()) != {"question", "answer"}:
        return False, "Output must contain only question+answer"
    q = qa.get("question")
    a = qa.get("answer")
    if not isinstance(q, str) or not q.strip():
        return False, "Empty question"
    if not isinstance(a, str) or not a.strip():
        return False, "Empty answer"
    if PROHIBITED_LOC_RE.search(q) or PROHIBITED_LOC_RE.search(a):
        return False, "Contains prohibited location references"
    if "?" not in q and "？" not in q:
        return False, "Question should contain a question mark"
    # Keep answers short (evaluation-friendly).
    if difficulty != "unanswerable" and len(a.strip()) > 220:
        return False, "Answer too long"
    if difficulty == "unanswerable" and a.strip() != unanswerable:
        return False, f"Unanswerable answer must be exactly '{unanswerable}'"
    return True, ""


def _evidence_satisfies(
    *,
    difficulty: Difficulty,
    evidence: List[Dict[str, Any]],
    require_multi_doc: bool,
    min_page_gap: int,
    hard_min_evidence_sections: int = 2,
) -> bool:
    section_evidence = [e for e in evidence if isinstance(e, dict) and e.get("section_id")]
    if difficulty == "easy":
        return len(section_evidence) >= 1
    if difficulty == "unanswerable":
        # still require at least some attempts / reads.
        return len(section_evidence) >= 1

    hard_min = max(2, int(hard_min_evidence_sections))
    if len(section_evidence) < hard_min:
        return False

    docs = {e.get("doc_id") for e in section_evidence if e.get("doc_id")}
    if require_multi_doc:
        return len(docs) >= 2

    pages: List[int] = []
    for e in section_evidence:
        for p in (e.get("page_idxs") or []):
            if isinstance(p, int):
                pages.append(p)
    if len(pages) >= 2 and (max(pages) - min(pages) >= min_page_gap):
        return True

    # Fallback: if page_idx missing, use char distance.
    starts = [int(e.get("start_char") or 0) for e in section_evidence]
    return (max(starts) - min(starts)) >= 20000


def _local_verify_constraints(
    *,
    kind: ItemKind,
    difficulty: Difficulty,
    require_multi_doc: bool,
    min_page_gap: int,
    hard_min_evidence_sections: int = 2,
    evidence: List[Dict[str, Any]],
    question: str,
    answer: str,
    prompt_lang: PromptLang = "en",
) -> Tuple[bool, str]:
    unanswerable = DocDancerPrompts(lang=prompt_lang).unanswerable_answer()
    if PROHIBITED_LOC_RE.search(question) or PROHIBITED_LOC_RE.search(answer):
        return False, "Contains prohibited location references"
    if difficulty == "unanswerable" and answer.strip() != unanswerable:
        return False, f"Unanswerable answer must be exactly '{unanswerable}'"

    # For non-calculation QA, enforce that numeric tokens in the answer appear in evidence
    # (reduces hallucinated numbers/units).
    if kind == "qa" and difficulty != "unanswerable":
        nums = _extract_number_tokens(answer, max_items=6)
        if nums:
            hay = ""
            for e in evidence:
                if isinstance(e, dict):
                    if isinstance(e.get("text"), str):
                        hay += "\n" + e["text"]
            hay_norm = _normalize_number_haystack(hay)
            for token in nums:
                token_norm = _normalize_number_haystack(token)
                if token_norm and token_norm not in hay_norm:
                    return False, f"Answer numeric token not found in evidence: {token}"

    section_evidence = [e for e in evidence if isinstance(e, dict) and e.get("section_id")]
    section_ids = [str(e.get("section_id")) for e in section_evidence if e.get("section_id")]
    if difficulty == "easy":
        if len(set(section_ids)) != 1:
            return False, "Easy must use exactly 1 evidence section"
        return True, ""

    if difficulty == "unanswerable":
        if len(set(section_ids)) < 1:
            return False, "Unanswerable must include at least 1 evidence section (attempted reads)"
        return True, ""

    # hard
    hard_min = max(2, int(hard_min_evidence_sections))
    if len(set(section_ids)) < hard_min:
        return False, f"Hard must use >={hard_min} evidence sections"
    doc_ids = {e.get("doc_id") for e in section_evidence if e.get("doc_id")}
    if require_multi_doc:
        if len(doc_ids) < 2:
            return False, "Hard requires cross-doc evidence"
        return True, ""

    pages: List[int] = []
    for e in section_evidence:
        for p in (e.get("page_idxs") or []):
            if isinstance(p, int):
                pages.append(p)
    if len(pages) >= 2 and (max(pages) - min(pages) >= min_page_gap):
        return True, ""
    return False, "Hard requires sufficient page gap within a single doc"


def _has_multimodal_evidence(section_evidence: List[Dict[str, Any]]) -> bool:
    for e in section_evidence:
        if e.get("has_table") is True:
            return True
        imgs = e.get("image_urls") or []
        if isinstance(imgs, list) and len(imgs) > 0:
            return True
    return False


def _call_llm_json(
    llm: OpenAICompatChatClient,
    *,
    messages: List[ChatMessage],
    temperature: float,
    max_tokens: int,
    prompt_lang: PromptLang = "en",
    tries: int = 3,
) -> Dict[str, Any]:
    prompts = DocDancerPrompts(lang=prompt_lang)
    last_err: Optional[Exception] = None
    for attempt in range(tries):
        content = llm.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format_json=True,
        )
        try:
            return _extract_first_json(content)
        except Exception as exc:
            last_err = exc
            messages = [
                *messages,
                ChatMessage(
                    role="user",
                    content=prompts.json_retry(),
                ),
            ]
            continue
    raise RuntimeError(f"LLM did not return valid JSON: {last_err}")


def _guess_data_url_mime(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suf == ".webp":
        return "image/webp"
    return "image/png"


def _summarize_reads_for_goal(
    llm: OpenAICompatChatClient,
    *,
    goal: str,
    reads: List[Dict[str, Any]],
    read_with_images: bool = False,
    prompt_lang: PromptLang = "en",
) -> str:
    prompts = DocDancerPrompts(lang=prompt_lang)
    compact_reads = []
    image_paths: List[str] = []
    for r in reads[:4]:
        compact_reads.append(
            {
                "doc_id": r.get("doc_id"),
                "section_id": r.get("section_id"),
                "section_title": r.get("section_title"),
                "page_idxs": r.get("page_idxs"),
                "has_table": r.get("has_table"),
                "image_urls": r.get("image_urls"),
                "text": (str(r.get("text") or "")[:1200] + "…") if r.get("text") else "",
            }
        )
        if read_with_images:
            # Prefer crops / page screenshots for multimodal grounding when available.
            preferred = []
            preferred.extend(list(r.get("crop_paths") or [])[:2])
            preferred.extend(list(r.get("page_image_paths") or [])[:2])
            preferred.extend(list(r.get("image_paths") or [])[:2])
            for p in preferred[:2]:
                if isinstance(p, str) and p and p not in image_paths:
                    image_paths.append(p)
            if len(image_paths) >= 2:
                break

    user_content: Any
    if read_with_images and image_paths:
        parts: List[Dict[str, Any]] = [
            {"type": "text", "text": json.dumps({"goal": goal, "reads": compact_reads}, ensure_ascii=False, indent=2)}
        ]
        for p in image_paths[:2]:
            try:
                img_path = Path(p)
                mime = _guess_data_url_mime(img_path)
                b = img_path.read_bytes()
                data_url = f"data:{mime};base64," + base64.b64encode(b).decode("ascii")
                parts.append({"type": "image_url", "image_url": {"url": data_url}})
            except Exception:
                continue
        user_content = parts
    else:
        user_content = json.dumps({"goal": goal, "reads": compact_reads}, ensure_ascii=False, indent=2)
    messages = [
        ChatMessage(role="system", content=prompts.reader_system()),
        ChatMessage(
            role="user",
            content=user_content,
        ),
    ]
    try:
        out = _call_llm_json(llm, messages=messages, temperature=0.2, max_tokens=260, prompt_lang=prompt_lang, tries=2)
        s = out.get("summary")
        if isinstance(s, str) and s.strip():
            return s.strip()
    except Exception:
        pass
    return ""


def explore(
    *,
    llm: OpenAICompatChatClient,
    toolkit: MultiDocToolkit,
    difficulty: Difficulty,
    require_multi_doc: bool,
    min_page_gap: int,
    max_steps: int = 12,
    search_limit: int = 8,
    seed: Optional[int] = None,
    read_with_images: bool = False,
    guided_keywords: Optional[List[str]] = None,
    prompt_lang: PromptLang = "en",
) -> List[TrajectoryStep]:
    rnd = random.Random(seed)
    prompts = DocDancerPrompts(lang=prompt_lang)
    sys = prompts.tool_schema()
    intro = prompts.exploration_intro(difficulty=difficulty, require_multi_doc=require_multi_doc, min_page_gap=min_page_gap)
    outline = toolkit.outline()
    history: List[TrajectoryStep] = []
    used_signatures: set[str] = set()

    for step in range(1, max_steps + 1):
        outline_suggestions = _suggest_keywords_from_outline(outline) if step == 1 else []
        user_state = {
            "difficulty": difficulty,
            "require_multi_doc": require_multi_doc,
            "min_page_gap": min_page_gap,
            "step": step,
            "outline": outline if step == 1 else None,
            "guided_keywords": guided_keywords if (step == 1 and guided_keywords) else None,
            "suggested_search_keywords": (
                (prompts.suggested_search_keyword_groups() + outline_suggestions) if step == 1 else None
            ),
            "history": [
                {
                    "intent": h.intent,
                    "tool": h.tool,
                    "args": h.args,
                    "observation": h.observation,
                }
                for h in history[-6:]
            ],
        }
        messages = [
            ChatMessage(role="system", content=sys),
            ChatMessage(role="system", content=intro),
            ChatMessage(role="user", content=json.dumps(user_state, ensure_ascii=False, indent=2)),
        ]
        action = _call_llm_json(llm, messages=messages, temperature=0.6, max_tokens=400, prompt_lang=prompt_lang)
        tool = action.get("tool")
        intent = str(action.get("intent") or "").strip()
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        if tool not in ("search", "read", "finish"):
            tool = "search"

        sig = json.dumps({"tool": tool, "args": args}, ensure_ascii=False, sort_keys=True)
        if sig in used_signatures:
            # Nudge: avoid repetition by forcing a small variation.
            if tool == "search":
                kws = list(args.get("keywords") or [])
                if kws:
                    pool = [t for grp in prompts.suggested_search_keyword_groups() for t in grp if isinstance(t, str) and t.strip()]
                    kws = kws + [rnd.choice(pool or ["definition"])]
                args["keywords"] = kws
            sig = json.dumps({"tool": tool, "args": args}, ensure_ascii=False, sort_keys=True)
        used_signatures.add(sig)

        if tool == "finish":
            history.append(
                TrajectoryStep(step=step, intent=intent or "finish", tool="finish", args={}, observation={"status": "finished"})
            )
            break

        if tool == "search":
            keywords = args.get("keywords")
            if not isinstance(keywords, list) or not keywords:
                base = prompts.suggested_search_keyword_groups()
                fallback = (base[0][0] if base and base[0] else "definition")
                keywords = [args.get("keyword") or fallback]
            keywords = [str(k).strip() for k in keywords if str(k).strip()]
            hits = toolkit.search(keywords=keywords, limit=search_limit)
            obs = {"keywords": keywords, "hits": hits}
            history.append(TrajectoryStep(step=step, intent=intent, tool="search", args={"keywords": keywords}, observation=obs))
            continue

        if tool == "read":
            section_ids = args.get("section_ids")
            chunk_ids = args.get("chunk_ids")  # back-compat if model still outputs chunk_ids
            if not isinstance(section_ids, list):
                section_ids = []
            if not isinstance(chunk_ids, list):
                chunk_ids = []

            if not section_ids:
                # if no section_ids, try to reuse last search results
                last_hits: List[Dict[str, Any]] = []
                for h in reversed(history):
                    if h.tool == "search":
                        last_hits = list((h.observation or {}).get("hits") or [])
                        break
                section_ids = [
                    x.get("section_id")
                    for x in last_hits[:3]
                    if isinstance(x, dict) and x.get("section_id")
                ]
                if not section_ids:
                    chunk_ids = [
                        x.get("chunk_id")
                        for x in last_hits[:3]
                        if isinstance(x, dict) and x.get("chunk_id")
                    ]

            section_ids = [str(s).strip() for s in (section_ids or []) if str(s).strip()]
            chunk_ids = [str(c).strip() for c in (chunk_ids or []) if str(c).strip()]
            goal = str(args.get("goal") or "").strip() or prompts.default_read_goal()
            reads = toolkit.read(section_ids=section_ids[:3], chunk_ids=chunk_ids[:4], goal=goal, max_chars=4500)
            summary = (
                _summarize_reads_for_goal(
                    llm, goal=goal, reads=reads, read_with_images=read_with_images, prompt_lang=prompt_lang
                )
                if reads
                else ""
            )
            obs = {"goal": goal, "reads": reads, "summary_for_goal": summary}
            history.append(
                TrajectoryStep(
                    step=step,
                    intent=intent,
                    tool="read",
                    args={"section_ids": section_ids[:3], "chunk_ids": chunk_ids[:4], "goal": goal},
                    observation=obs,
                )
            )
            continue

    return history


def synthesize(
    *,
    llm: OpenAICompatChatClient,
    difficulty: Difficulty,
    require_multi_doc: bool,
    min_page_gap: int,
    evidence: List[Dict[str, Any]],
    tries: int = 3,
    prompt_lang: PromptLang = "en",
) -> Dict[str, Any]:
    prompts = DocDancerPrompts(lang=prompt_lang)
    prompt = prompts.synthesis_qa_prompt(
        difficulty=difficulty,
        require_multi_doc=require_multi_doc,
        min_page_gap=min_page_gap,
        evidence_json=json.dumps(evidence, ensure_ascii=False, indent=2),
    )
    messages = [
        ChatMessage(role="system", content=prompts.synthesis_system()),
        ChatMessage(role="user", content=prompt),
    ]
    last_err = ""
    for _ in range(tries):
        qa = _call_llm_json(llm, messages=messages, temperature=0.6, max_tokens=600, prompt_lang=prompt_lang)
        ok, reason = _validate_qa(difficulty=difficulty, qa=qa, prompt_lang=prompt_lang)
        if ok:
            return qa
        last_err = reason
        retry = prompts.synthesis_retry_invalid(kind="qa", reason=reason)
        messages.append(ChatMessage(role="user", content=retry))
    raise RuntimeError(f"Synthesis failed: {last_err}")


def synthesize_calc_spec(
    *,
    llm: OpenAICompatChatClient,
    difficulty: Difficulty,
    require_multi_doc: bool,
    min_page_gap: int,
    evidence: List[Dict[str, Any]],
    tries: int = 3,
    prompt_lang: PromptLang = "en",
) -> Dict[str, Any]:
    prompts = DocDancerPrompts(lang=prompt_lang)
    prompt = prompts.synthesis_calc_prompt(
        difficulty=difficulty,
        require_multi_doc=require_multi_doc,
        min_page_gap=min_page_gap,
        evidence_json=json.dumps(evidence, ensure_ascii=False, indent=2),
    )
    messages = [
        ChatMessage(role="system", content=prompts.synthesis_system()),
        ChatMessage(role="user", content=prompt),
    ]
    last_err = ""
    for _ in range(tries):
        spec = _call_llm_json(llm, messages=messages, temperature=0.6, max_tokens=900, prompt_lang=prompt_lang)
        ok, reason = _validate_calc_spec(difficulty=difficulty, spec=spec)
        if ok:
            return spec
        last_err = reason
        retry = prompts.synthesis_retry_invalid(kind="calc", reason=reason)
        messages.append(ChatMessage(role="user", content=retry))
    raise RuntimeError(f"Calc synthesis failed: {last_err}")


def _select_evidence_from_trajectory(
    trajectory: List[TrajectoryStep], *, max_reads: int = 4, prefer_multimodal: bool = False
) -> Tuple[List[Dict[str, Any]], List[str]]:
    reads: List[Dict[str, Any]] = []
    summaries: List[str] = []
    for step in trajectory:
        if step.tool != "read":
            continue
        for r in (step.observation or {}).get("reads") or []:
            if not isinstance(r, dict):
                continue
            # Keep evidence compact to reduce token cost.
            reads.append(
                {
                    "doc_id": r.get("doc_id"),
                    "section_id": r.get("section_id"),
                    "chunk_ids": r.get("chunk_ids"),
                    "section_title": r.get("section_title"),
                    "page_idxs": r.get("page_idxs"),
                    "start_char": r.get("start_char"),
                    "end_char": r.get("end_char"),
                    "image_urls": r.get("image_urls"),
                    "has_table": r.get("has_table"),
                    "text": (str(r.get("text") or "")[:900] + "…") if r.get("text") else "",
                }
            )
        s = (step.observation or {}).get("summary_for_goal")
        if isinstance(s, str) and s.strip():
            summaries.append(s.strip())
    # Keep last N reads; they tend to be most relevant.
    if not prefer_multimodal:
        return reads[-max_reads:], summaries[-max_reads:]

    # Prefer a diverse set (multimodal/table/image) while keeping recency.
    picked: List[Dict[str, Any]] = []
    seen_sections: set[str] = set()

    def is_multimodal(e: Dict[str, Any]) -> bool:
        if e.get("has_table") is True:
            return True
        imgs = e.get("image_urls") or []
        return isinstance(imgs, list) and len(imgs) > 0

    for e in reversed(reads):
        sid = str(e.get("section_id") or "")
        if not sid or sid in seen_sections:
            continue
        if is_multimodal(e):
            picked.append(e)
            seen_sections.add(sid)
        if len(picked) >= max_reads:
            break
    if len(picked) < max_reads:
        for e in reversed(reads):
            sid = str(e.get("section_id") or "")
            if not sid or sid in seen_sections:
                continue
            picked.append(e)
            seen_sections.add(sid)
            if len(picked) >= max_reads:
                break

    picked.reverse()
    return picked, summaries[-max_reads:]


def generate_docdancer_items(
    cfg: AppConfig,
    *,
    doc_ids: List[str],
    total: int,
    easy_max_ratio: float = 0.10,
    unanswerable_ratio: float = 0.15,
    hard_multi_doc_ratio: float = 0.5,
    calc_ratio: float = 0.0,
    hard_min_evidence_sections: int = 2,
    min_page_gap: int = 3,
    max_steps: int = 12,
    search_limit: int = 10,
    seed: Optional[int] = None,
    verify_with_llm: bool = False,
    llm_timeout_s: int = 180,
    explore_model: Optional[str] = None,
    synth_model: Optional[str] = None,
    judge_model: Optional[str] = None,
    hard_require_multimodal: bool = False,
    read_with_images: bool = False,
    prompt_lang: PromptLang = "en",
) -> Iterable[GeneratedItem]:
    rnd = random.Random(seed)

    easy_target = int(total * easy_max_ratio)
    unans_target = int(total * unanswerable_ratio + 0.5)
    easy_target = min(easy_target, total)
    unans_target = min(unans_target, total - easy_target)
    hard_target = total - easy_target - unans_target

    schedule: List[Difficulty] = (
        (["easy"] * easy_target) + (["unanswerable"] * unans_target) + (["hard"] * hard_target)
    )
    rnd.shuffle(schedule)

    store = DocStore(cfg)
    docs = store.list_docs()

    # Only keep docs that have the canonical + index artifacts we need for DocToolkit.
    available: List[str] = []
    for did in doc_ids:
        rec = docs.get(did) if isinstance(docs, dict) else None
        if not isinstance(rec, dict):
            continue
        canon = rec.get("canonical_path")
        idx = rec.get("index_path")
        if not (isinstance(canon, str) and canon and Path(canon).exists()):
            continue
        if not (isinstance(idx, str) and idx and Path(idx).exists()):
            continue
        available.append(did)

    if not available:
        raise RuntimeError(
            "No ingested docs available for generation (canonical/index missing). "
            "Run ingest first and ensure canonical.json + chunks.sqlite3 exist."
        )

    explore_model_final = explore_model or os.environ.get("OPENAI_EXPLORE_MODEL") or "gpt-4o-mini"
    synth_model_final = synth_model or os.environ.get("OPENAI_SYNTH_MODEL") or None
    llm_explore = OpenAICompatChatClient.from_env(timeout_s=llm_timeout_s, model=explore_model_final)
    llm_synth = OpenAICompatChatClient.from_env(timeout_s=llm_timeout_s, model=synth_model_final)
    judge_model_final = judge_model or os.environ.get("OPENAI_JUDGE_MODEL") or llm_synth.model
    llm_judge = OpenAICompatChatClient.from_env(timeout_s=llm_timeout_s, model=judge_model_final)

    for idx, difficulty in enumerate(schedule, start=1):
        # Select documents for this item.
        require_multi_doc = False
        selected_docs: List[str]
        if difficulty == "hard" and len(available) >= 2 and rnd.random() < hard_multi_doc_ratio:
            require_multi_doc = True
            selected_docs = rnd.sample(available, k=2)
        else:
            selected_docs = [rnd.choice(available)]

        assets: Dict[str, Dict[str, Any]] = {}
        for did in selected_docs:
            try:
                d = store.get_doc(did)
            except Exception:
                continue
            md_path = d.get("mineru_markdown_path")
            images_dir = None
            pages_dir = None
            crops_dir = None
            asset_manifest_path = d.get("mineru_asset_manifest_path")
            if isinstance(md_path, str) and md_path:
                cand = Path(md_path).parent / "images"
                if cand.exists():
                    images_dir = str(cand)
                pages_cand = Path(md_path).parent / "pages"
                if pages_cand.exists():
                    pages_dir = str(pages_cand)
                crops_cand = Path(md_path).parent / "crops"
                if crops_cand.exists():
                    crops_dir = str(crops_cand)

            # If an asset manifest exists, prefer its declared directories (back-compat safe).
            if isinstance(asset_manifest_path, str) and asset_manifest_path:
                try:
                    manifest = json.loads(Path(asset_manifest_path).read_text(encoding="utf-8"))
                    # These are relative paths under method_dir; we resolve via manifest file parent.
                    method_dir = Path(asset_manifest_path).parent
                    if (method_dir / "pages").exists():
                        pages_dir = str((method_dir / "pages").resolve())
                    if (method_dir / "crops").exists():
                        crops_dir = str((method_dir / "crops").resolve())
                except Exception:
                    pass
            assets[did] = {
                "images_dir": images_dir,
                "pages_dir": pages_dir,
                "crops_dir": crops_dir,
                "asset_manifest_path": asset_manifest_path,
            }

        toolkit = MultiDocToolkit(cfg, selected_docs, assets=assets)
        try:
            guided_keywords: Optional[List[str]] = None
            for attempt in range(1, 6):
                traj = explore(
                    llm=llm_explore,
                    toolkit=toolkit,
                    difficulty=difficulty,
                    require_multi_doc=require_multi_doc,
                    min_page_gap=min_page_gap,
                    max_steps=min(max_steps, 8) if difficulty == "easy" else max_steps,
                    search_limit=search_limit,
                    seed=(None if seed is None else (seed + idx * 100 + attempt)),
                    read_with_images=read_with_images,
                    guided_keywords=guided_keywords,
                    prompt_lang=prompt_lang,
                )
                evidence, summaries = _select_evidence_from_trajectory(
                    traj,
                    max_reads=5 if difficulty == "hard" else 3,
                    prefer_multimodal=(difficulty == "hard"),
                )
                if summaries:
                    # Provide higher-level hints without replacing raw evidence.
                    evidence.append({"notes": "\n".join(summaries[:3])})
                if not _evidence_satisfies(
                    difficulty=difficulty,
                    evidence=evidence,
                    require_multi_doc=require_multi_doc,
                    min_page_gap=min_page_gap,
                    hard_min_evidence_sections=hard_min_evidence_sections,
                ):
                    continue

                section_evidence = [e for e in evidence if isinstance(e, dict) and e.get("section_id")]
                kind: ItemKind = "qa"
                spec: Dict[str, Any] = {}
                outcome: Dict[str, Any] = {}
                if difficulty != "unanswerable" and float(calc_ratio) > 0.0 and rnd.random() < float(calc_ratio):
                    kind = "calc"

                qa: Dict[str, Any]
                if kind == "calc":
                    try:
                        spec = synthesize_calc_spec(
                            llm=llm_synth,
                            difficulty=difficulty,
                            require_multi_doc=require_multi_doc,
                            min_page_gap=min_page_gap,
                            evidence=evidence,
                            tries=3,
                            prompt_lang=prompt_lang,
                        )

                        # Ground calc inputs: numeric inputs should come from the raw evidence text (not reader summaries).
                        hay = "\n".join(
                            str(e.get("text") or "")
                            for e in section_evidence
                            if isinstance(e, dict) and isinstance(e.get("text"), str) and e.get("text")
                        )
                        hay_norm = _normalize_number_haystack(hay)
                        inputs_obj = spec.get("inputs") if isinstance(spec.get("inputs"), dict) else {}
                        grounded = True
                        for _, v in inputs_obj.items():
                            tokens = _extract_number_tokens(str(v), max_items=4)
                            for tok in tokens:
                                tok_norm = _normalize_number_haystack(tok)
                                if tok_norm and tok_norm not in hay_norm:
                                    grounded = False
                                    break
                            if not grounded:
                                break
                        if not grounded:
                            continue

                        outcome = run_code_python(
                            code=str(spec["code"]),
                            inputs=inputs_obj,
                            limits=PythonSandboxLimits(timeout_s=6.0),
                        )
                        if outcome.get("exec_status") != "ok":
                            continue
                        answer = str(outcome.get("result_text") or "").strip()
                        if not answer:
                            continue
                        qa = {"question": str(spec["question"]), "answer": answer}

                        ok, reason = _validate_qa(difficulty=difficulty, qa=qa, prompt_lang=prompt_lang)
                        if not ok:
                            continue
                        ok, reason = _local_verify_constraints(
                            kind="calc",
                            difficulty=difficulty,
                            require_multi_doc=require_multi_doc,
                            min_page_gap=min_page_gap,
                            hard_min_evidence_sections=hard_min_evidence_sections,
                            evidence=evidence,
                            question=str(qa["question"]),
                            answer=str(qa["answer"]),
                            prompt_lang=prompt_lang,
                        )
                        if not ok:
                            continue
                    except Exception:
                        continue
                else:
                    try:
                        qa = synthesize(
                            llm=llm_synth,
                            difficulty=difficulty,
                            require_multi_doc=require_multi_doc,
                            min_page_gap=min_page_gap,
                            evidence=evidence,
                            tries=3,
                            prompt_lang=prompt_lang,
                        )
                    except Exception:
                        continue
                    ok, reason = _local_verify_constraints(
                        kind="qa",
                        difficulty=difficulty,
                        require_multi_doc=require_multi_doc,
                        min_page_gap=min_page_gap,
                        hard_min_evidence_sections=hard_min_evidence_sections,
                        evidence=evidence,
                        question=str(qa["question"]),
                        answer=str(qa["answer"]),
                        prompt_lang=prompt_lang,
                    )
                    if not ok:
                        continue
                if difficulty == "hard" and hard_require_multimodal:
                    if not _has_multimodal_evidence(section_evidence):
                        continue
                if verify_with_llm:
                    # For calc items, attach derived computation so the judge can validate determinism/support.
                    judge_evidence = [e for e in evidence if not (isinstance(e, dict) and set(e.keys()) == {"notes"})]
                    if kind == "calc":
                        judge_evidence.append(
                            {
                                "derived": {
                                    "tool": "python_sandbox",
                                    "code": str(spec.get("code") or "")[:1200],
                                    "inputs": spec.get("inputs") if isinstance(spec.get("inputs"), dict) else {},
                                    "result_text": str(outcome.get("result_text") or "")[:500],
                                    "exec_status": outcome.get("exec_status"),
                                }
                            }
                        )
                    judge_out = judge_item_with_llm(
                        llm_judge,
                        kind=kind,
                        difficulty=difficulty,
                        require_multi_doc=require_multi_doc,
                        min_page_gap=min_page_gap,
                        hard_min_evidence_sections=hard_min_evidence_sections,
                        evidence=judge_evidence,
                        question=str(qa["question"]),
                        answer=str(qa["answer"]),
                        prompt_lang=prompt_lang,
                    )
                    if not (
                        judge_out.get("supported") is True
                        and judge_out.get("unique") is True
                        and judge_out.get("difficulty_ok") is True
                    ):
                        issues_text = " ".join([str(x) for x in (judge_out.get("issues") or [])])
                        guided_keywords = _keywords_from_text(issues_text + " " + str(qa.get("question") or ""), max_terms=10)
                        continue
                    guided_keywords = None

                evidence_section_ids = [str(e.get("section_id")) for e in section_evidence if e.get("section_id")]
                evidence_chunk_ids: List[str] = []
                for e in section_evidence:
                    for cid in (e.get("chunk_ids") or []):
                        if isinstance(cid, str) and cid:
                            evidence_chunk_ids.append(cid)

                derived: Optional[Dict[str, Any]] = None
                if kind == "calc":
                    derived = {
                        "tool": "python_sandbox",
                        "inputs": spec.get("inputs") if isinstance(spec.get("inputs"), dict) else {},
                        "code": str(spec.get("code") or ""),
                        "result_text": str(outcome.get("result_text") or ""),
                        "exec_status": outcome.get("exec_status"),
                        "elapsed_ms": outcome.get("elapsed_ms"),
                    }
                yield GeneratedItem(
                    question=str(qa["question"]).strip(),
                    answer=str(qa["answer"]).strip(),
                    difficulty=difficulty,
                    kind=kind,
                    used_doc_ids=selected_docs,
                    evidence_section_ids=evidence_section_ids,
                    evidence_chunk_ids=evidence_chunk_ids,
                    trajectory=traj,
                    derived=derived,
                )
                break
            else:
                # give up this slot
                yield GeneratedItem(
                    question="",
                    answer="",
                    difficulty=difficulty,
                    kind="qa",
                    used_doc_ids=selected_docs,
                    evidence_section_ids=[],
                    evidence_chunk_ids=[],
                    trajectory=[],
                    derived=None,
                )
        finally:
            toolkit.close()


def judge_item_with_llm(
    llm: OpenAICompatChatClient,
    *,
    kind: ItemKind,
    difficulty: Difficulty,
    require_multi_doc: bool,
    min_page_gap: int,
    hard_min_evidence_sections: int = 2,
    evidence: List[Dict[str, Any]],
    question: str,
    answer: str,
    prompt_lang: PromptLang = "en",
) -> Dict[str, Any]:
    """
    Return judge output dict: supported/unique/difficulty_ok/issues.
    """
    prompts = DocDancerPrompts(lang=prompt_lang)
    messages = [
        ChatMessage(role="system", content=prompts.judge_system()),
        ChatMessage(
            role="user",
            content=prompts.judge_prompt(
                kind=kind,
                difficulty=difficulty,
                require_multi_doc=require_multi_doc,
                min_page_gap=min_page_gap,
                hard_min_evidence_sections=int(hard_min_evidence_sections),
                evidence_json=json.dumps(evidence, ensure_ascii=False, indent=2),
                question=question,
                answer=answer,
            ),
        ),
    ]
    out = _call_llm_json(llm, messages=messages, temperature=0.0, max_tokens=400, prompt_lang=prompt_lang, tries=2)
    if not isinstance(out, dict):
        return {"supported": False, "unique": False, "difficulty_ok": False, "issues": ["bad_judge_output"]}
    out.setdefault("issues", [])
    if not isinstance(out.get("issues"), list):
        out["issues"] = [str(out.get("issues"))]
    return out


def write_items_jsonl(
    *,
    items: Iterable[GeneratedItem],
    out_jsonl_path: Path,
    write_debug: bool = True,
    resume: bool = False,
) -> int:
    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path = out_jsonl_path.with_suffix(out_jsonl_path.suffix + ".debug.jsonl")
    count = 0
    seen_questions: set[str] = set()

    def norm_q(s: str) -> str:
        s = str(s or "").strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[\u3000\t\r\n]+", " ", s)
        s = re.sub(r"[，。！？,.!?;；:：()（）\\[\\]{}\"'“”‘’]+", "", s)
        return s[:500]

    mode = "a" if (resume and out_jsonl_path.exists()) else "w"
    dbg_mode = "a" if (resume and debug_path.exists()) else "w"
    if resume and out_jsonl_path.exists():
        try:
            for ln in out_jsonl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                if isinstance(obj, dict) and obj.get("question"):
                    key = norm_q(str(obj.get("question") or ""))
                    if key:
                        seen_questions.add(key)
        except Exception:
            pass

    with out_jsonl_path.open(mode, encoding="utf-8") as f_out, debug_path.open(dbg_mode, encoding="utf-8") as f_dbg:
        for it in items:
            if not it.question or not it.answer:
                continue
            key = norm_q(it.question)
            if key and key in seen_questions:
                continue
            if key:
                seen_questions.add(key)
            f_out.write(json.dumps({"question": it.question, "answer": it.answer}, ensure_ascii=False) + "\n")
            if write_debug:
                f_dbg.write(
                    json.dumps(
                        {
                            "question": it.question,
                            "answer": it.answer,
                            "difficulty": it.difficulty,
                            "kind": it.kind,
                            "used_doc_ids": it.used_doc_ids,
                            "evidence_section_ids": it.evidence_section_ids,
                            "evidence_chunk_ids": it.evidence_chunk_ids,
                            "trajectory": [asdict(s) for s in it.trajectory],
                            "derived": it.derived,
                            "created_at": time.time(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            count += 1
    return count
