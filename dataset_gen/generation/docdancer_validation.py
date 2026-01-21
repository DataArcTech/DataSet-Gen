from typing import Any, Dict, List, Tuple

from dataset_gen.prompts.docdancer import DocDancerPrompts, PromptLang

from .docdancer_types import Difficulty, ItemKind
from .docdancer_utils import (
    PROHIBITED_LOC_RE,
    doc_identity_ok,
    extract_inputs_keys_used,
    extract_number_tokens,
    normalize_number_haystack,
)


def validate_calc_spec(*, difficulty: Difficulty, spec: Dict[str, Any]) -> Tuple[bool, str]:
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
        keys_used = extract_inputs_keys_used(code)
        if len(keys_used) < 2:
            return False, "hard calc code must use at least 2 distinct INPUTS keys"
    return True, ""


def validate_qa(*, difficulty: Difficulty, qa: Dict[str, Any], prompt_lang: PromptLang = "en") -> Tuple[bool, str]:
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


def evidence_satisfies(
    *,
    difficulty: Difficulty,
    evidence: List[Dict[str, Any]],
    require_multi_doc: bool,
    min_page_gap: int,
    hard_min_evidence_sections: int = 2,
) -> bool:
    section_evidence = [e for e in evidence if isinstance(e, dict) and e.get("section_id")]
    chunk_ids: List[str] = []
    for e in section_evidence:
        for cid in (e.get("chunk_ids") or []):
            if isinstance(cid, str) and cid:
                chunk_ids.append(cid)
    uniq_chunks = sorted(set(chunk_ids))

    if difficulty == "easy":
        # Easy should be answerable from a single chunk.
        if uniq_chunks:
            return len(uniq_chunks) == 1
        return len(section_evidence) >= 1
    if difficulty == "unanswerable":
        # still require at least some attempts / reads.
        return bool(uniq_chunks) or len(section_evidence) >= 1

    hard_min = max(2, int(hard_min_evidence_sections))
    # Hard must be multi-hop: require >=N distinct chunks (more robust than "sections" for page-chunked canonicals).
    if uniq_chunks and len(uniq_chunks) < hard_min:
        return False
    if not uniq_chunks and len(section_evidence) < hard_min:
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


def local_verify_constraints(
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
    if not doc_identity_ok(question=question, evidence=evidence, require_multi_doc=require_multi_doc):
        return False, "Question missing explicit document identity (must mention doc_title/doc_filename from evidence)"

    # For non-calculation QA, enforce that numeric tokens in the answer appear in evidence
    # (reduces hallucinated numbers/units).
    if kind == "qa" and difficulty != "unanswerable":
        nums = extract_number_tokens(answer, max_items=6)
        if nums:
            hay = ""
            for e in evidence:
                if isinstance(e, dict):
                    if isinstance(e.get("text"), str):
                        hay += "\n" + e["text"]
            hay_norm = normalize_number_haystack(hay)
            for token in nums:
                token_norm = normalize_number_haystack(token)
                if token_norm and token_norm not in hay_norm:
                    return False, f"Answer numeric token not found in evidence: {token}"

    section_evidence = [e for e in evidence if isinstance(e, dict) and e.get("section_id")]
    section_ids = [str(e.get("section_id")) for e in section_evidence if e.get("section_id")]
    chunk_ids: List[str] = []
    for e in section_evidence:
        for cid in (e.get("chunk_ids") or []):
            if isinstance(cid, str) and cid:
                chunk_ids.append(cid)
    uniq_chunks = sorted(set(chunk_ids))
    if difficulty == "easy":
        # "Easy" is single-chunk direct extraction.
        if uniq_chunks and len(uniq_chunks) != 1:
            return False, "Easy must use exactly 1 evidence chunk"
        if not uniq_chunks and len(set(section_ids)) != 1:
            return False, "Easy must use exactly 1 evidence section"
        return True, ""

    if difficulty == "unanswerable":
        if uniq_chunks:
            return True, ""
        if len(set(section_ids)) < 1:
            return False, "Unanswerable must include at least 1 evidence section (attempted reads)"
        return True, ""

    # hard
    hard_min = max(2, int(hard_min_evidence_sections))
    if uniq_chunks and len(uniq_chunks) < hard_min:
        return False, f"Hard must use >={hard_min} evidence chunks"
    if not uniq_chunks and len(set(section_ids)) < hard_min:
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


def has_multimodal_evidence(section_evidence: List[Dict[str, Any]]) -> bool:
    for e in section_evidence:
        if e.get("has_table") is True:
            return True
        imgs = e.get("image_urls") or []
        if isinstance(imgs, list) and len(imgs) > 0:
            return True
    return False

