import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .docdancer_types import GeneratedItem


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
                            # Keep QA clean in qa.mix.jsonl; debug can have augmented question/answer (sources/citations).
                            "question_plain": it.question,
                            "answer_plain": it.answer,
                            "question": (it.debug or {}).get("question") if isinstance(it.debug, dict) and (it.debug or {}).get("question") else it.question,
                            "answer": (it.debug or {}).get("answer") if isinstance(it.debug, dict) and (it.debug or {}).get("answer") else it.answer,
                            "difficulty": it.difficulty,
                            "kind": it.kind,
                            "used_doc_ids": it.used_doc_ids,
                            "evidence_section_ids": it.evidence_section_ids,
                            "evidence_chunk_ids": it.evidence_chunk_ids,
                            "trajectory": [asdict(s) for s in it.trajectory],
                            "derived": it.derived,
                            "debug": it.debug,
                            "created_at": time.time(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            count += 1
    return count
