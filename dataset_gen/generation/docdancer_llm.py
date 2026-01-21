import base64
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dataset_gen.llm.openai_compat import ChatMessage, OpenAICompatChatClient
from dataset_gen.prompts.docdancer import DocDancerPrompts, PromptLang

from .docdancer_toolkit import MultiDocToolkit
from .docdancer_types import Difficulty, ItemKind, TrajectoryStep
from .docdancer_utils import extract_first_json, suggest_keywords_from_outline
from .docdancer_validation import validate_calc_spec, validate_qa


def call_llm_json(
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
    for _ in range(tries):
        content = llm.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format_json=True,
        )
        try:
            return extract_first_json(content)
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


def guess_data_url_mime(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suf == ".webp":
        return "image/webp"
    return "image/png"


def summarize_reads_for_goal(
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
                mime = guess_data_url_mime(img_path)
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
        ChatMessage(role="user", content=user_content),
    ]
    try:
        out = call_llm_json(llm, messages=messages, temperature=0.2, max_tokens=260, prompt_lang=prompt_lang, tries=2)
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
        outline_suggestions = suggest_keywords_from_outline(outline) if step == 1 else []
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
        action = call_llm_json(llm, messages=messages, temperature=0.6, max_tokens=400, prompt_lang=prompt_lang)
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
                    pool = [
                        t
                        for grp in prompts.suggested_search_keyword_groups()
                        for t in grp
                        if isinstance(t, str) and t.strip()
                    ]
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
                if difficulty == "easy":
                    # Easy = single-chunk extraction: prefer reading one specific chunk.
                    chunk_ids = [x.get("chunk_id") for x in last_hits[:1] if isinstance(x, dict) and x.get("chunk_id")]
                    section_ids = []
                if not chunk_ids and not section_ids:
                    section_ids = [x.get("section_id") for x in last_hits[:3] if isinstance(x, dict) and x.get("section_id")]
                if not chunk_ids and not section_ids:
                    chunk_ids = [x.get("chunk_id") for x in last_hits[:3] if isinstance(x, dict) and x.get("chunk_id")]

            section_ids = [str(s).strip() for s in (section_ids or []) if str(s).strip()]
            chunk_ids = [str(c).strip() for c in (chunk_ids or []) if str(c).strip()]
            goal = str(args.get("goal") or "").strip() or prompts.default_read_goal()
            reads = toolkit.read(
                section_ids=section_ids[:3],
                chunk_ids=chunk_ids[:4],
                goal=goal,
                max_chars=4500,
                max_chunks_per_section=(1 if difficulty == "easy" else 8),
            )
            do_summary = os.environ.get("DOCDANCER_SUMMARIZE_READS", "1").strip() in {"1", "true", "yes", "on"}
            summary = (
                summarize_reads_for_goal(
                    llm, goal=goal, reads=reads, read_with_images=read_with_images, prompt_lang=prompt_lang
                )
                if (do_summary and reads)
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
        qa = call_llm_json(llm, messages=messages, temperature=0.6, max_tokens=600, prompt_lang=prompt_lang)
        ok, reason = validate_qa(difficulty=difficulty, qa=qa, prompt_lang=prompt_lang)
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
        spec = call_llm_json(llm, messages=messages, temperature=0.6, max_tokens=900, prompt_lang=prompt_lang)
        ok, reason = validate_calc_spec(difficulty=difficulty, spec=spec)
        if ok:
            return spec
        last_err = reason
        retry = prompts.synthesis_retry_invalid(kind="calc", reason=reason)
        messages.append(ChatMessage(role="user", content=retry))
    raise RuntimeError(f"Calc synthesis failed: {last_err}")


def select_evidence_from_trajectory(
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
                    "doc_filename": r.get("doc_filename"),
                    "doc_title": r.get("doc_title"),
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
    out = call_llm_json(llm, messages=messages, temperature=0.0, max_tokens=400, prompt_lang=prompt_lang, tries=2)
    if not isinstance(out, dict):
        return {"supported": False, "unique": False, "difficulty_ok": False, "issues": ["bad_judge_output"]}
    out.setdefault("issues", [])
    if not isinstance(out.get("issues"), list):
        out["issues"] = [str(out.get("issues"))]
    return out
import os
