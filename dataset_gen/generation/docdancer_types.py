from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


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
    # Extra fields only written to .debug.jsonl (qa jsonl stays clean).
    debug: Optional[Dict[str, Any]] = None
