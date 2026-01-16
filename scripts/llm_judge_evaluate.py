#!/usr/bin/env python3
"""
LLM-judge phase (evaluation): local Gate checks + LLM-as-a-judge scoring.

Input: a run directory created by scripts/llm_judge_generate.py
Output: quality_report.json + failed_items.jsonl + judged_items.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bootstrap_imports() -> None:
    sys.path.insert(0, str(_repo_root()))


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


PROHIBITED_LOC_RE = re.compile(
    r"(第\s*\d+\s*页|page\s*\d+|section_id|chunk_id|Figure\s*\d+|Table\s*\d+|图\s*\d+|表\s*\d+|第\s*\d+\s*章|chapter\s*\d+)",
    re.IGNORECASE,
)

PUNCT_RE = re.compile(r"[，。！？,.!?;；:：()（）\[\]{}\"'“”‘’]+")
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)(%|‰)?")


def _norm_question(q: str) -> str:
    s = str(q or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = PUNCT_RE.sub("", s)
    return s[:500]


def _normalize_number_haystack(text: str) -> str:
    return str(text or "").replace(",", "")


def _extract_number_tokens(text: str, *, max_items: int = 8) -> List[str]:
    out: List[str] = []
    for m in NUMBER_RE.finditer(str(text or "")):
        val = (m.group(1) or "").strip()
        suf = (m.group(2) or "").strip()
        tok = (val + suf).strip()
        if tok and tok not in out:
            out.append(tok)
        if len(out) >= max_items:
            break
    return out


def _extract_inputs_keys_used(code: str) -> set[str]:
    out: set[str] = set()
    for m in re.finditer(r"INPUTS\s*\[\s*(['\"])(?P<k>[^'\"]+)\1\s*\]", str(code or "")):
        k = (m.group("k") or "").strip()
        if k:
            out.add(k)
    return out


def _section_doc_id(section_id: str) -> Optional[str]:
    sid = str(section_id or "")
    if "_" not in sid:
        return None
    return sid.split("_", 1)[0] or None


@dataclass(frozen=True)
class GateResult:
    ok: bool
    reasons: List[str]


def gate_check_item(
    *,
    item: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    evidence_text: str,
    prompt_lang: str,
    hard_min_evidence_sections: int,
    min_page_gap: int,
    require_multi_doc: bool,
) -> GateResult:
    reasons: List[str] = []

    q = item.get("question")
    a = item.get("answer")
    difficulty = str(item.get("difficulty") or "")
    kind = str(item.get("kind") or "qa")
    evidence_section_ids = item.get("evidence_section_ids") or []
    derived = item.get("derived")

    if not isinstance(q, str) or not q.strip():
        reasons.append("empty_question")
    if not isinstance(a, str) or not a.strip():
        reasons.append("empty_answer")

    if isinstance(q, str) and ("?" not in q and "？" not in q):
        reasons.append("missing_question_mark")

    if isinstance(q, str) and PROHIBITED_LOC_RE.search(q):
        reasons.append("prohibited_location_in_question")
    if isinstance(a, str) and PROHIBITED_LOC_RE.search(a):
        reasons.append("prohibited_location_in_answer")

    if difficulty not in {"easy", "hard", "unanswerable"}:
        reasons.append("bad_difficulty")

    if difficulty != "unanswerable" and isinstance(a, str) and len(a.strip()) > 220:
        reasons.append("answer_too_long")

    sec_ids: List[str] = []
    if isinstance(evidence_section_ids, list):
        for sid in evidence_section_ids:
            if isinstance(sid, str) and sid.strip():
                sec_ids.append(sid.strip())
    distinct_secs = len(set(sec_ids))

    if difficulty == "easy":
        if distinct_secs != 1:
            reasons.append("easy_requires_exactly_1_evidence_section")
    elif difficulty == "unanswerable":
        if distinct_secs < 1:
            reasons.append("unanswerable_requires_at_least_1_evidence_section")
    elif difficulty == "hard":
        if distinct_secs < max(2, int(hard_min_evidence_sections)):
            reasons.append(f"hard_requires_at_least_{max(2, int(hard_min_evidence_sections))}_evidence_sections")

    # Difficulty constraints aligned with generator:
    if difficulty == "hard":
        section_evidence = [e for e in (evidence or []) if isinstance(e, dict) and e.get("section_id")]
        doc_ids = {str(e.get("doc_id")) for e in section_evidence if isinstance(e.get("doc_id"), str) and e.get("doc_id")}
        if require_multi_doc:
            if len(doc_ids) < 2:
                reasons.append("hard_requires_multi_doc")
        else:
            pages: List[int] = []
            starts: List[int] = []
            for e in section_evidence:
                for p in (e.get("page_idxs") or []):
                    if isinstance(p, int):
                        pages.append(p)
                sc = e.get("start_char")
                if isinstance(sc, int):
                    starts.append(sc)
            ok_gap = False
            if len(pages) >= 2 and (max(pages) - min(pages) >= int(min_page_gap)):
                ok_gap = True
            if (not ok_gap) and starts and (max(starts) - min(starts) >= 20000):
                ok_gap = True
            if not ok_gap:
                reasons.append("hard_page_span_too_small")

    if kind == "qa" and difficulty != "unanswerable" and isinstance(a, str):
        nums = _extract_number_tokens(a, max_items=6)
        if nums:
            hay_norm = _normalize_number_haystack(evidence_text)
            for tok in nums:
                tok_norm = _normalize_number_haystack(tok)
                if tok_norm and tok_norm not in hay_norm:
                    reasons.append(f"answer_number_not_in_evidence:{tok}")
                    break

    if kind == "calc":
        if difficulty == "unanswerable":
            reasons.append("unanswerable_cannot_be_calc")
        if not isinstance(derived, dict):
            reasons.append("calc_missing_derived")
        else:
            for k in ["tool", "inputs", "code", "result_text", "exec_status"]:
                if k not in derived:
                    reasons.append(f"calc_derived_missing:{k}")
            code = derived.get("code")
            inputs = derived.get("inputs")
            if not isinstance(code, str) or not code.strip():
                reasons.append("calc_empty_code")
            else:
                if "INPUTS" not in code:
                    reasons.append("calc_code_must_reference_INPUTS")
                if not any(op in code for op in ["+", "-", "*", "/", "**"]):
                    reasons.append("calc_code_must_include_arithmetic")
                if difficulty == "hard" and len(_extract_inputs_keys_used(code)) < 2:
                    reasons.append("hard_calc_must_use_at_least_2_INPUTS_keys")
                if PROHIBITED_LOC_RE.search(code):
                    reasons.append("calc_code_contains_prohibited_location")
            if not isinstance(inputs, dict):
                reasons.append("calc_inputs_not_object")
            else:
                hay_norm = _normalize_number_haystack(evidence_text)
                grounded = True
                for _, v in inputs.items():
                    for tok in _extract_number_tokens(str(v), max_items=4):
                        if _normalize_number_haystack(tok) not in hay_norm:
                            grounded = False
                            break
                    if not grounded:
                        break
                if not grounded:
                    reasons.append("calc_inputs_not_grounded_in_evidence")

    if difficulty == "unanswerable" and isinstance(a, str):
        from dataset_gen.prompts.docdancer import DocDancerPrompts

        unans = DocDancerPrompts(lang=prompt_lang).unanswerable_answer()
        if a.strip() != unans:
            reasons.append("unanswerable_answer_token_mismatch")

    return GateResult(ok=(len(reasons) == 0), reasons=reasons)


def _build_evidence_from_section_ids(
    *,
    cfg: Any,
    section_ids: List[str],
) -> Tuple[List[Dict[str, Any]], str]:
    from dataset_gen.toolkit.doc_toolkit import DocToolkit

    by_doc: Dict[str, List[str]] = defaultdict(list)
    for sid in section_ids:
        did = _section_doc_id(sid)
        if not did:
            continue
        by_doc[did].append(sid)

    evidence: List[Dict[str, Any]] = []
    text_parts: List[str] = []
    for doc_id, sids in sorted(by_doc.items()):
        tk = DocToolkit(cfg, doc_id=doc_id)
        try:
            reads = tk.read(section_ids=sids, goal="evidence_for_evaluation", max_chars=8000)
            for r in reads:
                evidence.append(
                    {
                        "doc_id": doc_id,
                        "section_id": r.section_id,
                        "chunk_ids": list(r.chunk_ids),
                        "title": r.section_title,
                        "page_idxs": list(r.page_idxs),
                        "start_char": int(r.start_char),
                        "end_char": int(r.end_char),
                        "image_urls": list(r.image_urls),
                        "has_table": ("|" in r.text and "\n" in r.text),
                        "text": r.text,
                    }
                )
                text_parts.append(r.text)
        finally:
            tk.close()

    return evidence, "\n".join(text_parts)


def _reservoir_sample_debug_items(
    *,
    debug_files: List[Path],
    k: int,
    seed: int,
) -> List[Tuple[Path, int, Dict[str, Any]]]:
    rnd = random.Random(int(seed))
    reservoir: List[Tuple[Path, int, Dict[str, Any]]] = []
    seen = 0
    for dbg_path in debug_files:
        for idx, it in enumerate(_iter_jsonl(dbg_path)):
            seen += 1
            if len(reservoir) < k:
                reservoir.append((dbg_path, idx, it))
                continue
            j = rnd.randint(0, seen - 1)
            if j < k:
                reservoir[j] = (dbg_path, idx, it)
    reservoir.sort(key=lambda x: (str(x[0]), int(x[1])))
    return reservoir


def main() -> int:
    _bootstrap_imports()

    from dataset_gen.config import AppConfig
    from dataset_gen.env import find_env_file, load_dotenv
    from dataset_gen.llm.openai_compat import ChatMessage, OpenAICompatChatClient
    from dataset_gen.prompts.docdancer import DocDancerPrompts
    from dataset_gen.tools.code_python import PythonSandboxLimits, run_code_python

    env_path = find_env_file(_repo_root())
    if env_path:
        load_dotenv(env_path, override=False)

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="./outputs/llm_judge_run", help="Run dir produced by llm_judge_generate.py.")
    ap.add_argument("--qa-dir", default=None, help="Override QA dir (default: <run-dir>/qa).")
    ap.add_argument("--out-dir", default=None, help="Override output dir (default: <run-dir>/eval_llm_judge).")
    ap.add_argument("--output-dir", default=None, help="Override dataset_gen output dir (default: <run-dir>/dataset_gen_output).")

    ap.add_argument("--prompt-lang", default="en", choices=["en", "zh", "zh-Hant"])
    ap.add_argument("--hard-min-evidence-sections", type=int, default=2)
    ap.add_argument("--min-page-gap", type=int, default=3)
    ap.add_argument("--require-multi-doc", action="store_true")

    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument("--no-llm", action="store_true", help="Only run local Gate checks (skip LLM judge).")
    ap.add_argument(
        "--max-items",
        type=int,
        default=200,
        help="Evaluate at most N items (default: 200; <=0 means all). If the run has more items, a seeded sample is used.",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed used when sampling --max-items.")

    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    qa_dir = Path(args.qa_dir) if args.qa_dir else (run_dir / "qa")
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "eval_llm_judge")
    out_dir.mkdir(parents=True, exist_ok=True)

    debug_files = sorted(qa_dir.glob("*.jsonl.debug.jsonl"))
    if not debug_files:
        raise SystemExit(f"No debug JSONL files found under: {qa_dir}")

    ds_output_dir = Path(args.output_dir) if args.output_dir else (run_dir / "dataset_gen_output")
    cfg = AppConfig(output_dir=ds_output_dir)

    llm: Optional[OpenAICompatChatClient] = None
    prompts = DocDancerPrompts(lang=args.prompt_lang)
    if not args.no_llm:
        llm = OpenAICompatChatClient.from_env(timeout_s=180, model=str(args.judge_model))

    total = 0
    gate_ok = 0
    judge_ok = 0
    aligned_ok = 0
    by_reason = Counter()
    by_difficulty = Counter()
    by_kind = Counter()
    judge_counts = Counter()
    dup_norm_questions = 0
    seen_norm_q: set[str] = set()

    judged_out = (out_dir / "judged_items.jsonl").open("w", encoding="utf-8")
    failed_out = (out_dir / "failed_items.jsonl").open("w", encoding="utf-8")

    max_items = None if int(args.max_items) <= 0 else int(args.max_items)

    def eval_one(*, dbg_path: Path, idx: int, it: Dict[str, Any]) -> None:
        nonlocal total, gate_ok, judge_ok, aligned_ok, dup_norm_questions

        total += 1
        doc_hint = it.get("used_doc_ids")
        doc_id = None
        if isinstance(doc_hint, list) and doc_hint and isinstance(doc_hint[0], str):
            doc_id = doc_hint[0]
        doc_id = doc_id or _section_doc_id((it.get("evidence_section_ids") or [""])[0]) or "unknown"

        difficulty = str(it.get("difficulty") or "")
        kind = str(it.get("kind") or "qa")
        by_difficulty[difficulty] += 1
        by_kind[kind] += 1

        qn = _norm_question(str(it.get("question") or ""))
        if qn and qn in seen_norm_q:
            dup_norm_questions += 1
        elif qn:
            seen_norm_q.add(qn)

        sec_ids = [s for s in (it.get("evidence_section_ids") or []) if isinstance(s, str) and s.strip()]
        evidence, evidence_text = _build_evidence_from_section_ids(cfg=cfg, section_ids=sec_ids)

        gate = gate_check_item(
            item=it,
            evidence=evidence,
            evidence_text=evidence_text,
            prompt_lang=str(args.prompt_lang),
            hard_min_evidence_sections=int(args.hard_min_evidence_sections),
            min_page_gap=int(args.min_page_gap),
            require_multi_doc=bool(args.require_multi_doc),
        )
        if not gate.ok:
            for r in gate.reasons:
                by_reason[r] += 1
            failed_out.write(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "source_debug_jsonl": str(dbg_path),
                        "idx": idx,
                        "stage": "gate",
                        "reasons": gate.reasons,
                        "question": it.get("question"),
                        "answer": it.get("answer"),
                        "difficulty": difficulty,
                        "kind": kind,
                        "evidence_section_ids": sec_ids,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            judged_out.write(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "source_debug_jsonl": str(dbg_path),
                        "idx": idx,
                        "gate_ok": False,
                        "gate_reasons": gate.reasons,
                        "judge": None,
                        "question": it.get("question"),
                        "answer": it.get("answer"),
                        "difficulty": difficulty,
                        "kind": kind,
                        "evidence_section_ids": sec_ids,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            return

        gate_ok += 1

        if kind == "calc":
            derived = it.get("derived") if isinstance(it.get("derived"), dict) else {}
            code = str(derived.get("code") or "")
            inputs = derived.get("inputs") if isinstance(derived.get("inputs"), dict) else {}
            out = run_code_python(code=code, inputs=inputs, limits=PythonSandboxLimits(timeout_s=6.0))
            if out.get("exec_status") != "ok":
                reason = "calc_replay_exec_failed"
                by_reason[reason] += 1
                failed_out.write(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "source_debug_jsonl": str(dbg_path),
                            "idx": idx,
                            "stage": "gate",
                            "reasons": [reason],
                            "question": it.get("question"),
                            "answer": it.get("answer"),
                            "difficulty": difficulty,
                            "kind": kind,
                            "evidence_section_ids": sec_ids,
                            "replay": out,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                return
            answer = str(it.get("answer") or "").strip()
            result_text = str(out.get("result_text") or "").strip()
            if answer != result_text:
                reason = "calc_replay_result_mismatch"
                by_reason[reason] += 1
                failed_out.write(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "source_debug_jsonl": str(dbg_path),
                            "idx": idx,
                            "stage": "gate",
                            "reasons": [reason],
                            "question": it.get("question"),
                            "answer": answer,
                            "expected_result_text": result_text,
                            "difficulty": difficulty,
                            "kind": kind,
                            "evidence_section_ids": sec_ids,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                return

        judge_out: Optional[Dict[str, Any]] = None
        supported_ok: Optional[bool] = None
        unique_ok: Optional[bool] = None
        judge_difficulty_ok: Optional[bool] = None
        per_item_aligned: Optional[bool] = None

        if llm is not None:
            judge_evidence = list(evidence)
            if kind == "calc":
                d = it.get("derived") if isinstance(it.get("derived"), dict) else {}
                judge_evidence.append(
                    {
                        "derived": {
                            "tool": "python_sandbox",
                            "code": str(d.get("code") or "")[:1200],
                            "inputs": d.get("inputs") if isinstance(d.get("inputs"), dict) else {},
                            "result_text": str(d.get("result_text") or "")[:500],
                            "exec_status": d.get("exec_status"),
                        }
                    }
                )
            messages = [
                ChatMessage(role="system", content=prompts.judge_system()),
                ChatMessage(
                    role="user",
                    content=prompts.judge_prompt(
                        kind=str(kind),
                        difficulty=str(difficulty),
                        require_multi_doc=bool(args.require_multi_doc),
                        min_page_gap=int(args.min_page_gap),
                        hard_min_evidence_sections=int(args.hard_min_evidence_sections),
                        evidence_json=json.dumps(judge_evidence, ensure_ascii=False, indent=2),
                        question=str(it.get("question") or ""),
                        answer=str(it.get("answer") or ""),
                    ),
                ),
            ]
            raw = llm.chat(messages=messages, temperature=0.0, max_tokens=400, response_format_json=True)
            try:
                judge_out = json.loads(raw)
            except Exception:
                judge_out = {"supported": False, "unique": False, "difficulty_ok": False, "issues": ["bad_judge_output"]}

            supported_ok = judge_out.get("supported") is True
            unique_ok = judge_out.get("unique") is True
            judge_difficulty_ok = judge_out.get("difficulty_ok") is True

            for k in ["supported", "unique", "difficulty_ok"]:
                if judge_out.get(k) is True:
                    judge_counts[k] += 1

            if supported_ok and unique_ok and judge_difficulty_ok:
                judge_ok += 1
            else:
                by_reason["judge_failed"] += 1
                failed_out.write(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "source_debug_jsonl": str(dbg_path),
                            "idx": idx,
                            "stage": "judge",
                            "reasons": ["judge_failed"],
                            "judge": judge_out,
                            "question": it.get("question"),
                            "answer": it.get("answer"),
                            "difficulty": difficulty,
                            "kind": kind,
                            "evidence_section_ids": sec_ids,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            per_item_aligned = supported_ok and unique_ok
            if per_item_aligned:
                aligned_ok += 1
            else:
                if not supported_ok:
                    by_reason["judge_supported_false"] += 1
                if not unique_ok:
                    by_reason["judge_unique_false"] += 1

        judged_out.write(
            json.dumps(
                {
                    "doc_id": doc_id,
                    "source_debug_jsonl": str(dbg_path),
                    "idx": idx,
                    "gate_ok": True,
                    "gate_reasons": [],
                    "judge": judge_out,
                    "judge_supported_ok": supported_ok,
                    "judge_unique_ok": unique_ok,
                    "judge_difficulty_ok": judge_difficulty_ok,
                    "aligned_pass": per_item_aligned,
                    "question": it.get("question"),
                    "answer": it.get("answer"),
                    "difficulty": difficulty,
                    "kind": kind,
                    "evidence_section_ids": sec_ids,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    try:
        if max_items is None:
            for dbg_path in debug_files:
                for idx, it in enumerate(_iter_jsonl(dbg_path)):
                    eval_one(dbg_path=dbg_path, idx=idx, it=it)
        else:
            sampled = _reservoir_sample_debug_items(debug_files=debug_files, k=max_items, seed=int(args.seed))
            for dbg_path, idx, it in sampled:
                eval_one(dbg_path=dbg_path, idx=int(idx), it=it)
    finally:
        judged_out.close()
        failed_out.close()

    def rate(n: int, d: int) -> float:
        return 0.0 if d <= 0 else float(n) / float(d)

    report: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "qa_dir": str(qa_dir),
        "dataset_gen_output_dir": str(ds_output_dir),
        "total_items": total,
        "gate_pass": gate_ok,
        "gate_pass_rate": rate(gate_ok, total),
        "judge_pass": (judge_ok if not args.no_llm else None),
        "judge_pass_rate": (rate(judge_ok, gate_ok) if (not args.no_llm) else None),
        "aligned_pass": (aligned_ok if not args.no_llm else None),
        "aligned_pass_rate": (rate(aligned_ok, gate_ok) if (not args.no_llm) else None),
        "judge_true_rates_over_gate": (
            {k: rate(int(judge_counts[k]), gate_ok) for k in ["supported", "unique", "difficulty_ok"]} if not args.no_llm else None
        ),
        "duplicate_questions_norm": dup_norm_questions,
        "unique_questions_norm": len(seen_norm_q),
        "by_difficulty": dict(by_difficulty),
        "by_kind": dict(by_kind),
        "top_failure_reasons": by_reason.most_common(30),
        "settings": {
            "prompt_lang": args.prompt_lang,
            "hard_min_evidence_sections": int(args.hard_min_evidence_sections),
            "min_page_gap": int(args.min_page_gap),
            "require_multi_doc": bool(args.require_multi_doc),
            "judge_model": (None if args.no_llm else str(args.judge_model)),
        },
        "artifacts": {
            "quality_report_json": str(out_dir / "quality_report.json"),
            "failed_items_jsonl": str(out_dir / "failed_items.jsonl"),
            "judged_items_jsonl": str(out_dir / "judged_items.jsonl"),
        },
    }

    (out_dir / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
