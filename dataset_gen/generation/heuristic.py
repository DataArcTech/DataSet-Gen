import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class QAPair:
    question: str
    answer: str


def _extract_headings(text: str, max_items: int = 30) -> List[str]:
    headings: List[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            if title and title not in headings:
                headings.append(title)
        if len(headings) >= max_items:
            break
    return headings


def _extract_candidate_terms(text: str, max_items: int = 50) -> List[str]:
    # Extremely lightweight: find common insurance-ish keywords.
    seeds = [
        "责任",
        "保险责任",
        "责任免除",
        "除外",
        "定义",
        "释义",
        "等待期",
        "免赔额",
        "赔付比例",
        "保额",
        "保险金额",
        "续保",
        "犹豫期",
        "理赔",
        "投保",
        "生效",
        "终止",
        "费率",
        "保费",
    ]
    out: List[str] = []
    for s in seeds:
        if s in text:
            out.append(s)
    # Also capture some “定义：xxx” patterns.
    for m in re.finditer(r"(?P<term>[^\\s:：]{2,15})\\s*[：:]\\s*(?P<def>[^\\n]{5,80})", text):
        term = m.group("term").strip()
        if term and term not in out:
            out.append(term)
        if len(out) >= max_items:
            break
    return out[:max_items]


def generate_heuristic_qa(markdown: str, limit: int = 50) -> Iterable[QAPair]:
    headings = _extract_headings(markdown)
    terms = _extract_candidate_terms(markdown)
    produced: List[QAPair] = []

    for h in headings:
        produced.append(QAPair(question=f"文档中是否包含“{h}”相关章节？请回答：包含/不包含。", answer="包含"))
        if len(produced) >= limit:
            return produced

    for t in terms:
        produced.append(QAPair(question=f"请在文档中定位“{t}”并用一句话概括其含义或相关要求。", answer="见文档相关条款。"))
        if len(produced) >= limit:
            return produced

    # Fallback
    produced.append(QAPair(question="该文档主要讲的是什么？", answer="见文档内容。"))
    return produced[:limit]

