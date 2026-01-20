import argparse
import json
import os
from pathlib import Path

from dataset_gen.config import AppConfig
from dataset_gen.pipeline.ingest import ingest_one_or_many
from dataset_gen.pipeline.search import search_doc
from dataset_gen.pipeline.generate import generate_qa
from dataset_gen.pipeline.audit import audit_dataset
from dataset_gen.pipeline.show import show_dataset_items


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("DATASET_GEN_OUTPUT_DIR", "./data"),
        help="Output root directory (default: ./data or env DATASET_GEN_OUTPUT_DIR).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dataset_gen", description="Insurance Doc QA dataset generator (MVP).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Parse PDF(s) via MinerU and build canonical + index.")
    _add_common_args(p_ingest)
    p_ingest.add_argument("--input", required=True, help="PDF file or folder.")
    p_ingest.add_argument("--pattern", default="*.pdf", help="Glob pattern when --input is a folder.")
    p_ingest.add_argument("--mineru-url", default=os.environ.get("MINERU_SERVER_URL", "http://localhost:8899"))
    p_ingest.add_argument("--timeout-s", type=int, default=int(os.environ.get("MINERU_TIMEOUT_S", "900")))
    p_ingest.add_argument("--parse-format", default="mm_md", choices=["mm_md", "md_only", "content_list"])
    p_ingest.add_argument("--lang", default="ch")
    p_ingest.add_argument("--no-table", action="store_true")
    p_ingest.add_argument("--no-formula", action="store_true")
    p_ingest.add_argument("--start-page", type=int, default=0)
    p_ingest.add_argument("--end-page", type=int, default=None)
    p_ingest.add_argument(
        "--caption-mode",
        default=None,
        choices=["off", "content_list", "llm", "content_list_then_llm"],
        help="Optional override for server caption mode (MinerU postprocess).",
    )
    p_ingest.add_argument(
        "--caption-max-images",
        type=int,
        default=None,
        help="Optional override for max images to caption (MinerU postprocess).",
    )
    p_ingest.add_argument("--keep-source", action="store_true", help="Copy original PDFs into docs/ mirror.")

    p_search = sub.add_parser("search", help="Keyword search within one ingested doc.")
    _add_common_args(p_search)
    p_search.add_argument("--doc-id", required=True)
    p_search.add_argument("--keywords", nargs="+", required=True)
    p_search.add_argument("--limit", type=int, default=20)

    p_gen = sub.add_parser("generate", help="Generate QA pairs for one ingested doc.")
    _add_common_args(p_gen)
    p_gen.add_argument("--doc-id", required=True)
    p_gen.add_argument("--doc-ids", nargs="+", default=None, help="Optional doc_id list for cross-doc generation.")
    p_gen.add_argument("--out", required=True, help="Output JSONL path.")
    p_gen.add_argument("--mode", default="heuristic", choices=["heuristic", "docdancer"])
    p_gen.add_argument("--limit", type=int, default=50)
    p_gen.add_argument(
        "--append-n",
        type=int,
        default=None,
        help="Append exactly N new items (incremental mode). When set, --limit becomes ignored (docdancer).",
    )
    p_gen.add_argument("--easy-max-ratio", type=float, default=0.10, help="Ratio of easy questions (docdancer).")
    p_gen.add_argument("--unanswerable-ratio", type=float, default=0.15, help="Ratio of unanswerable questions (docdancer).")
    p_gen.add_argument(
        "--hard-multi-doc-ratio",
        type=float,
        default=0.50,
        help="For hard items, probability to enforce cross-doc evidence when >=2 docs are available (docdancer).",
    )
    p_gen.add_argument(
        "--hard-min-evidence-sections",
        type=int,
        default=2,
        help="For hard items, require at least N distinct evidence chunks (docdancer).",
    )
    p_gen.add_argument(
        "--calc-ratio",
        type=float,
        default=0.0,
        help="Ratio of calc questions (computed via python sandbox). Calc items are generated as hard (docdancer).",
    )
    p_gen.add_argument(
        "--min-page-gap",
        type=int,
        default=3,
        help="For hard single-doc items, require evidence page gap >= N (docdancer).",
    )
    p_gen.add_argument("--max-steps", type=int, default=12, help="Max exploration steps per item (docdancer).")
    p_gen.add_argument("--search-limit", type=int, default=10, help="Max hits returned per search step (docdancer).")
    p_gen.add_argument("--seed", type=int, default=None, help="Random seed (docdancer).")
    p_gen.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable writing .debug.jsonl with trajectories and evidence (docdancer).",
    )
    p_gen.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing output files and continue generation (docdancer).",
    )
    p_gen.add_argument(
        "--verify-with-llm",
        action="store_true",
        help="Use an LLM judge to verify support + difficulty constraints; enables rejection sampling (docdancer).",
    )
    p_gen.add_argument(
        "--llm-timeout-s",
        type=int,
        default=180,
        help="Timeout seconds for OpenAI-compatible LLM calls (docdancer).",
    )
    p_gen.add_argument(
        "--explore-model",
        default=None,
        help="Override model for exploration/reading (docdancer). Default: env OPENAI_EXPLORE_MODEL or gpt-4o-mini.",
    )
    p_gen.add_argument(
        "--synth-model",
        default=None,
        help="Override model for synthesis (docdancer). Default: env OPENAI_SYNTH_MODEL or OPENAI_CHAT_MODEL.",
    )
    p_gen.add_argument(
        "--judge-model",
        default=None,
        help="Override model for LLM judge (docdancer). Default: env OPENAI_JUDGE_MODEL or synth model.",
    )
    p_gen.add_argument(
        "--hard-require-multimodal",
        action="store_true",
        help="Require hard items to include at least one table/image evidence section (DocDancer-style multi-element).",
    )
    p_gen.add_argument(
        "--read-with-images",
        action="store_true",
        help="Attach extracted images to the auxiliary reader summaries when available.",
    )
    p_gen.add_argument(
        "--prompt-lang",
        default="auto",
        choices=["auto", "en", "zh", "zh-Hant"],
        help="Prompt language for LLM calls (default: auto; per-doc detection when available).",
    )
    p_gen.add_argument(
        "--anchor-doc",
        action="store_true",
        help="Anchor generation to --doc-id (easy/unanswerable use that doc; hard may add one extra doc for cross-doc).",
    )

    p_info = sub.add_parser("info", help="Show ingested document metadata.")
    _add_common_args(p_info)
    p_info.add_argument("--doc-id", required=True)

    p_audit = sub.add_parser("audit", help="Audit a generated QA dataset JSONL (and optional debug JSONL).")
    _add_common_args(p_audit)
    p_audit.add_argument("--qa-jsonl", required=True, help="Path to generated qa JSONL (question/answer).")
    p_audit.add_argument("--debug-jsonl", default=None, help="Optional path to .debug.jsonl for richer stats.")
    p_audit.add_argument(
        "--no-semantic-dedup",
        action="store_true",
        help="Disable semantic near-duplicate analysis (SimHash).",
    )
    p_audit.add_argument(
        "--semantic-max-hamming",
        type=int,
        default=3,
        help="Max Hamming distance for SimHash near-duplicate clustering (default: 3).",
    )
    p_audit.add_argument(
        "--coverage-quota-max-share",
        type=float,
        default=0.35,
        help="Quota check: flag if any top-level section exceeds this share of evidence usage (default: 0.35).",
    )
    p_audit.add_argument(
        "--reachability",
        action="store_true",
        help="Run fixed-budget reachability stress test (search/read) to hit gold evidence sections.",
    )
    p_audit.add_argument("--reachability-max-searches", type=int, default=2)
    p_audit.add_argument("--reachability-search-limit", type=int, default=10)
    p_audit.add_argument("--reachability-max-reads", type=int, default=2)

    p_show = sub.add_parser("show", help="Show generated items with evidence snippets (from .debug.jsonl).")
    p_show.add_argument("--debug-jsonl", required=True, help="Path to .debug.jsonl produced by docdancer mode.")
    p_show.add_argument("--n", type=int, default=3, help="Number of items to show.")
    p_show.add_argument("--kind", default=None, choices=["qa", "calc"], help="Optional filter by kind.")
    p_show.add_argument(
        "--difficulty",
        default=None,
        choices=["easy", "hard", "unanswerable"],
        help="Optional filter by difficulty.",
    )
    p_show.add_argument("--evidence-chars", type=int, default=260, help="Max characters per evidence snippet.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "ingest":
        cfg = AppConfig(output_dir=Path(args.output_dir))
        results = ingest_one_or_many(
            cfg,
            input_path=Path(args.input),
            pattern=args.pattern,
            mineru_url=args.mineru_url,
            timeout_s=args.timeout_s,
            parse_format=args.parse_format,
            lang=args.lang,
            formula_enable=not args.no_formula,
            table_enable=not args.no_table,
            start_page=args.start_page,
            end_page=args.end_page,
            caption_mode=args.caption_mode,
            caption_max_images=args.caption_max_images,
            keep_source=args.keep_source,
        )
        print(json.dumps({"ok": True, "results": results}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "search":
        cfg = AppConfig(output_dir=Path(args.output_dir))
        out = search_doc(cfg, doc_id=args.doc_id, keywords=args.keywords, limit=args.limit)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    if args.command == "generate":
        cfg = AppConfig(output_dir=Path(args.output_dir))
        out_path = Path(args.out)
        generated = generate_qa(
            cfg,
            doc_id=args.doc_id,
            doc_ids=args.doc_ids,
            out_jsonl_path=out_path,
            mode=args.mode,
            limit=args.limit,
            append_n=args.append_n,
            easy_max_ratio=args.easy_max_ratio,
            unanswerable_ratio=args.unanswerable_ratio,
            hard_multi_doc_ratio=args.hard_multi_doc_ratio,
            calc_ratio=args.calc_ratio,
            hard_min_evidence_sections=args.hard_min_evidence_sections,
            min_page_gap=args.min_page_gap,
            max_steps=args.max_steps,
            search_limit=args.search_limit,
            seed=args.seed,
            write_debug=not args.no_debug,
            resume=args.resume,
            verify_with_llm=args.verify_with_llm,
            llm_timeout_s=args.llm_timeout_s,
            explore_model=args.explore_model,
            synth_model=args.synth_model,
            judge_model=args.judge_model,
            hard_require_multimodal=args.hard_require_multimodal,
            read_with_images=args.read_with_images,
            prompt_lang=args.prompt_lang,
            anchor_doc=args.anchor_doc,
        )
        print(json.dumps({"ok": True, "written": str(out_path), "count": generated}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "info":
        from dataset_gen.storage.doc_store import DocStore

        cfg = AppConfig(output_dir=Path(args.output_dir))
        store = DocStore(cfg)
        doc = store.get_doc(args.doc_id)
        print(json.dumps(doc, ensure_ascii=False, indent=2))
        return 0

    if args.command == "audit":
        cfg = AppConfig(output_dir=Path(args.output_dir))
        out = audit_dataset(
            cfg=cfg,
            qa_jsonl=Path(args.qa_jsonl),
            debug_jsonl=(Path(args.debug_jsonl) if args.debug_jsonl else None),
            semantic_dedup=(not args.no_semantic_dedup),
            semantic_max_hamming=args.semantic_max_hamming,
            coverage_quota_max_share=args.coverage_quota_max_share,
            reachability=args.reachability,
            reachability_max_searches=args.reachability_max_searches,
            reachability_search_limit=args.reachability_search_limit,
            reachability_max_reads=args.reachability_max_reads,
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    if args.command == "show":
        items = show_dataset_items(
            debug_jsonl=Path(args.debug_jsonl),
            n=args.n,
            kind=args.kind,
            difficulty=args.difficulty,
            evidence_chars=args.evidence_chars,
        )
        print(json.dumps({"ok": True, "items": items}, ensure_ascii=False, indent=2))
        return 0

    raise SystemExit(f"Unknown command: {args.command}")
