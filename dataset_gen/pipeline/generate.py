import json
from pathlib import Path
from typing import Literal, Optional

from dataset_gen.config import AppConfig
from dataset_gen.env import find_env_file, load_dotenv
from dataset_gen.generation.heuristic import generate_heuristic_qa
from dataset_gen.generation.docdancer import generate_docdancer_items, write_items_jsonl
from dataset_gen.prompts.docdancer import PromptLang
from dataset_gen.storage.doc_store import DocStore


def generate_qa(
    cfg: AppConfig,
    *,
    doc_id: str,
    doc_ids: Optional[list[str]] = None,
    out_jsonl_path: Path,
    mode: Literal["heuristic", "docdancer"] = "heuristic",
    limit: int = 50,
    easy_max_ratio: float = 0.10,
    unanswerable_ratio: float = 0.15,
    hard_multi_doc_ratio: float = 0.50,
    calc_ratio: float = 0.0,
    hard_min_evidence_sections: int = 2,
    min_page_gap: int = 3,
    max_steps: int = 12,
    search_limit: int = 10,
    seed: Optional[int] = None,
    write_debug: bool = True,
    resume: bool = False,
    verify_with_llm: bool = False,
    llm_timeout_s: int = 180,
    explore_model: Optional[str] = None,
    synth_model: Optional[str] = None,
    judge_model: Optional[str] = None,
    hard_require_multimodal: bool = False,
    read_with_images: bool = False,
    prompt_lang: PromptLang = "en",
) -> int:
    store = DocStore(cfg)
    if mode == "heuristic":
        doc = store.get_doc(doc_id)
        md_path = doc.get("mineru_markdown_path")
        if not md_path:
            raise FileNotFoundError("mineru_markdown_path missing; ingest with output_format=mm_md first.")
        markdown = Path(md_path).read_text(encoding="utf-8", errors="ignore")
        out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with out_jsonl_path.open("w", encoding="utf-8") as f:
            for qa in generate_heuristic_qa(markdown, limit=limit):
                f.write(json.dumps({"question": qa.question, "answer": qa.answer}, ensure_ascii=False) + "\n")
                count += 1
        return count

    if mode == "docdancer":
        env_path = find_env_file(Path.cwd())
        if env_path:
            load_dotenv(env_path, override=False)
        selected = doc_ids or [doc_id]
        already = 0
        if resume and out_jsonl_path.exists():
            try:
                already = sum(1 for _ in out_jsonl_path.open("r", encoding="utf-8") if _.strip())
            except Exception:
                already = 0
        remaining = max(0, int(limit) - int(already))
        if remaining <= 0:
            return int(already)
        items = generate_docdancer_items(
            cfg,
            doc_ids=selected,
            total=remaining,
            easy_max_ratio=easy_max_ratio,
            unanswerable_ratio=unanswerable_ratio,
            hard_multi_doc_ratio=hard_multi_doc_ratio,
            calc_ratio=calc_ratio,
            hard_min_evidence_sections=hard_min_evidence_sections,
            min_page_gap=min_page_gap,
            max_steps=max_steps,
            search_limit=search_limit,
            seed=seed,
            verify_with_llm=verify_with_llm,
            llm_timeout_s=llm_timeout_s,
            explore_model=explore_model,
            synth_model=synth_model,
            judge_model=judge_model,
            hard_require_multimodal=hard_require_multimodal,
            read_with_images=read_with_images,
            prompt_lang=prompt_lang,
        )
        written = write_items_jsonl(items=items, out_jsonl_path=out_jsonl_path, write_debug=write_debug, resume=resume)
        return int(already) + int(written)

    raise ValueError(f"Unknown mode: {mode}")
