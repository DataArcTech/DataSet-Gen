import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dataset_gen.prompts.docdancer import PromptLang


def extract_first_json(text: str) -> Dict[str, Any]:
    """
    Robust-ish: find first {...} JSON object inside text.
    """
    start = str(text or "").find("{")
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


def looks_like_table(text: str) -> bool:
    # cheap heuristic: markdown table rows
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
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
_IDENT_CLEAN_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")
_SENT_END_RE = re.compile(r"[。！？；]|(?<!\d)[.!?;](?!\d)")
_CITATION_RE = re.compile(r"\[(\d+)\]")
_OTHER_CITATION_MARK_RE = re.compile(
    r"(\(\s*\d+\s*\)|（\s*\d+\s*）|【\s*\d+\s*】|<\s*sup\s*>\s*\d+\s*<\s*/\s*sup\s*>)",
    re.IGNORECASE,
)


def keywords_from_text(text: str, *, max_terms: int = 10) -> List[str]:
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


def norm_ident(text: str) -> str:
    """
    Normalize identifiers for robust substring matching:
    - lowercase
    - strip whitespace/punctuation
    - keep only [0-9A-Za-z] and CJK Unified Ideographs
    """
    s = str(text or "").strip().lower()
    if not s:
        return ""
    return _IDENT_CLEAN_RE.sub("", s)


def doc_identity_ok(*, question: str, evidence: List[Dict[str, Any]], require_multi_doc: bool) -> bool:
    """
    Ensure the question is self-contained by explicitly mentioning the document identity.

    Domain-agnostic: require that the question contains doc_title/doc_filename (derived from filenames)
    for at least one evidence doc (or >=2 docs for multi-doc items).
    """
    qn = norm_ident(question)
    if not qn:
        return False

    section_evidence = [e for e in evidence if isinstance(e, dict) and e.get("section_id")]
    docs: Dict[str, Dict[str, str]] = {}
    for e in section_evidence:
        did = str(e.get("doc_id") or "").strip()
        if not did:
            continue
        meta = docs.setdefault(did, {})
        for k in ("doc_title", "doc_filename"):
            v = e.get(k)
            if isinstance(v, str) and v.strip():
                meta[k] = v.strip()

    mentioned_docs = 0
    for did, meta in docs.items():
        candidates: List[str] = []
        title = meta.get("doc_title") or ""
        fn = meta.get("doc_filename") or ""
        if title:
            candidates.append(title)
        if fn:
            candidates.append(Path(fn).stem)
        ok = False
        for c in candidates:
            cn = norm_ident(c)
            if cn in qn:
                ok = True
                break
        if ok:
            mentioned_docs += 1

    if require_multi_doc:
        return mentioned_docs >= 2
    return mentioned_docs >= 1


def ensure_doc_identity_in_question(
    *,
    question: str,
    evidence: List[Dict[str, Any]],
    require_multi_doc: bool,
    prompt_lang: PromptLang,
) -> str:
    """
    If the question is missing explicit doc identity, prefix a source tag using evidence doc_title/doc_filename.
    This reduces retries when LLM forgets to include doc identity.
    """
    q = str(question or "").strip()
    if not q:
        return q
    if doc_identity_ok(question=q, evidence=evidence, require_multi_doc=require_multi_doc):
        return q

    section_evidence = [e for e in evidence if isinstance(e, dict) and e.get("section_id")]
    labels: List[str] = []
    seen: set[str] = set()
    for e in section_evidence:
        title = e.get("doc_title")
        fn = e.get("doc_filename")
        label = ""
        if isinstance(title, str) and title.strip():
            label = title.strip()
        elif isinstance(fn, str) and fn.strip():
            label = Path(fn.strip()).stem
        if not label:
            continue
        key = norm_ident(label)
        if not key or key in seen:
            continue
        seen.add(key)
        labels.append(label)
        if require_multi_doc and len(labels) >= 2:
            break
        if not require_multi_doc and len(labels) >= 1:
            break

    if not labels:
        return q

    if prompt_lang == "zh-Hant":
        prefix = f"（文件：{' + '.join(labels)}）"
    elif prompt_lang == "zh":
        prefix = f"（文档：{' + '.join(labels)}）"
    else:
        prefix = f"(Document: {' + '.join(labels)})"
    return f"{prefix} {q}"


def extract_number_tokens(text: str, *, max_items: int = 50) -> List[str]:
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


def normalize_number_haystack(text: str) -> str:
    # Make it easier to match numbers regardless of thousands separators.
    return str(text or "").replace(",", "")


def extract_inputs_keys_used(code: str) -> set[str]:
    """
    Parse code and return keys accessed as INPUTS['key'] or INPUTS["key"].
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


def suggest_keywords_from_outline(outline: List[Dict[str, Any]], *, max_terms: int = 12) -> List[List[str]]:
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


def guess_title_from_filename(filename: str) -> str:
    fn = str(filename or "").strip()
    return fn.rsplit(".", 1)[0] if fn else ""


def pick_doc_labels_from_evidence(
    evidence: List[Dict[str, Any]], *, require_multi_doc: bool
) -> List[str]:
    """
    Extract stable doc labels from evidence doc_title/doc_filename (best-effort).
    """
    section_evidence = [e for e in evidence if isinstance(e, dict) and e.get("section_id")]
    labels: List[str] = []
    seen: set[str] = set()
    for e in section_evidence:
        title = e.get("doc_title")
        fn = e.get("doc_filename")
        label = ""
        if isinstance(title, str) and title.strip():
            label = title.strip()
        elif isinstance(fn, str) and fn.strip():
            label = Path(fn.strip()).stem
        if not label:
            continue
        key = norm_ident(label)
        if not key or key in seen:
            continue
        seen.add(key)
        labels.append(label)
        if require_multi_doc and len(labels) >= 2:
            break
        if not require_multi_doc and len(labels) >= 1:
            break
    return labels


def build_source_hint(*, labels: List[str], prompt_lang: PromptLang) -> str:
    if not labels:
        return ""
    joined = " + ".join(labels)
    if prompt_lang == "zh-Hant":
        return f"（文件：{joined}）"
    if prompt_lang == "zh":
        return f"（文件：{joined}）"
    return f"(Files: {joined})"


def validate_answer_with_citations(
    *,
    answer_with_citations: str,
    allowed_eids: set[int],
) -> tuple[bool, str]:
    """
    Validate strict citation format:
    - Only [] citations are allowed.
    - Each [] contains exactly one integer eid.
    - No [1,3] style (enforced by digit-only content).
    - Every sentence-ending punctuation is immediately followed by one or more citations.
    """
    s = str(answer_with_citations or "").strip()
    if not s:
        return False, "empty_answer_with_citations"
    if _OTHER_CITATION_MARK_RE.search(s):
        return False, "uses_non_square_bracket_citation_mark"

    # Validate each citation token and ensure it references known evidence ids.
    eids = [int(m.group(1)) for m in _CITATION_RE.finditer(s)]
    if not eids:
        return False, "missing_citations"
    for eid in eids:
        if eid not in allowed_eids:
            return False, f"unknown_eid:{eid}"

    # Ensure there are no malformed bracket pairs like [a] or [1 2] etc.
    # If there is any '[' not part of a valid citation token, reject.
    stripped = _CITATION_RE.sub("", s)
    if "[" in stripped or "]" in stripped:
        return False, "malformed_brackets"

    # Require citations after every sentence-ending punctuation.
    for m in _SENT_END_RE.finditer(s):
        tail = s[m.end() :]
        # Allow whitespace between punctuation and citations, but require at least one [n] next.
        if not re.match(r"\s*(\[\d+\]\s*)+", tail):
            return False, "missing_sentence_end_citation"

    return True, ""


def strip_square_citations(text: str) -> str:
    """
    Remove strict citation tokens like `[123]` from model text.
    This keeps qa.mix.jsonl clean while retaining traceability in debug.
    """
    s = str(text or "")
    s = _CITATION_RE.sub("", s)
    # Normalize whitespace created by removals.
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n\s+", "\n", s)
    return s.strip()


def repair_answer_with_citations(
    *,
    answer_text: str,
    allowed_eids: set[int],
    citation_map: Optional[List[Dict[str, Any]]] = None,
    difficulty: str = "easy",
    require_multi_doc: bool = False,
) -> str:
    """
    Best-effort, deterministic repair:
    - Remove existing [n] tokens.
    - Add citations after every sentence-ending punctuation.
    - Ensure hard has >=2 distinct eids, and cross-doc hard covers >=2 doc_id.

    This is debug-only traceability; qa output remains citation-free.
    """
    base = strip_square_citations(str(answer_text or ""))
    if not base.strip():
        return str(answer_text or "").strip()

    eids_sorted = sorted({int(e) for e in allowed_eids if isinstance(e, int) and int(e) > 0})
    if not eids_sorted:
        return base.strip()

    # Prefer one eid per doc for cross-doc items so we can satisfy the doc coverage constraint.
    preferred: List[int] = []
    if require_multi_doc and citation_map:
        seen_docs: set[str] = set()
        for m in citation_map:
            if not isinstance(m, dict):
                continue
            eid = m.get("eid")
            doc_id = str(m.get("doc_id") or "").strip()
            if not isinstance(eid, int) or eid not in allowed_eids:
                continue
            if not doc_id or doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            preferred.append(eid)
        # Keep stable and within allowed pool.
        preferred = [e for e in preferred if e in allowed_eids]

    use_pool = preferred or eids_sorted

    # For hard items, ensure we have at least two distinct eids available.
    if str(difficulty) == "hard" and len(use_pool) < 2 and len(eids_sorted) >= 2:
        use_pool = eids_sorted[:2]

    parts: List[str] = []
    last = 0
    sent_i = 0
    for m in _SENT_END_RE.finditer(base):
        end = m.end()
        parts.append(base[last:end])
        eid = use_pool[sent_i % len(use_pool)]
        parts.append(f" [{eid}]")
        last = end
        sent_i += 1
    parts.append(base[last:])
    out = "".join(parts).strip()

    # If there were no sentence-ending punctuations, append citations at the end.
    if sent_i == 0:
        eid = use_pool[0]
        out = (base.strip() + f" [{eid}]").strip()

    # Enforce minimum distinct citations for hard items (>=2).
    if str(difficulty) == "hard":
        cited = {int(x) for x in _CITATION_RE.findall(out)}
        if len(cited) < 2:
            # Add one more different eid at the end.
            for eid in eids_sorted:
                if eid not in cited:
                    out = (out + f" [{eid}]").strip()
                    break

        if require_multi_doc and citation_map:
            by_eid = {m.get("eid"): m for m in citation_map if isinstance(m, dict) and isinstance(m.get("eid"), int)}
            cited = {int(x) for x in _CITATION_RE.findall(out)}
            cited_docs = {str(by_eid.get(e, {}).get("doc_id") or "") for e in cited}
            cited_docs = {d for d in cited_docs if d}
            if len(cited_docs) < 2:
                # Add citations from additional docs if possible.
                for m in citation_map:
                    if not isinstance(m, dict):
                        continue
                    eid = m.get("eid")
                    doc_id = str(m.get("doc_id") or "").strip()
                    if not isinstance(eid, int) or eid not in allowed_eids:
                        continue
                    if doc_id and doc_id not in cited_docs:
                        out = (out + f" [{eid}]").strip()
                        cited_docs.add(doc_id)
                        if len(cited_docs) >= 2:
                            break

    return out
