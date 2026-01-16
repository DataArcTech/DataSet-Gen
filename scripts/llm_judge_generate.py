#!/usr/bin/env python3
"""
LLM-judge phase (generation): ingest PDFs and synthesize QA pairs.

This script is intentionally dataset-agnostic: you provide an input PDF or a folder of PDFs.
It will:
1) ingest PDFs via MinerU server (build canonical + index)
2) generate QA pairs (DocDancer-style) with an OpenAI-compatible LLM

It writes a run directory with:
- dataset_gen_output/ (DocStore, canonical, indexes, parsed mirror)
- qa/ (per-doc QA JSONL + debug JSONL)
- ingest_manifest.json / generate_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bootstrap_imports() -> None:
    # Allow running without installing as a package.
    root = _repo_root()
    sys.path.insert(0, str(root))


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _mineru_healthcheck(mineru_url: str, timeout_s: float = 3.0) -> bool:
    try:
        import requests

        r = requests.get(mineru_url.rstrip("/") + "/health", timeout=timeout_s)
        return r.status_code == 200
    except Exception:
        return False


def _list_pdfs(input_path: Path, pattern: str) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted([p for p in input_path.rglob(pattern) if p.is_file()])


def _sample(pdfs: List[Path], *, sample_n: Optional[int], seed: int) -> List[Path]:
    if sample_n is None:
        return pdfs
    n = int(sample_n)
    if n <= 0 or n >= len(pdfs):
        return pdfs
    rnd = random.Random(int(seed))
    tmp = list(pdfs)
    rnd.shuffle(tmp)
    # stable order for nicer diffs/manifests
    return sorted(tmp[:n])


def main() -> int:
    _bootstrap_imports()

    from dataset_gen.config import AppConfig
    from dataset_gen.env import find_env_file, load_dotenv
    from dataset_gen.pipeline.generate import generate_qa
    from dataset_gen.pipeline.ingest import ingest_one_or_many
    from dataset_gen.storage.doc_store import DocStore

    # Load repo root .env (OpenAI-compatible) if present.
    env_path = find_env_file(_repo_root())
    if env_path:
        load_dotenv(env_path, override=False)

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="A PDF file or a folder containing PDFs.")
    ap.add_argument("--pattern", default="*.pdf", help="Glob pattern when --input is a folder (default: *.pdf).")

    ap.add_argument("--run-dir", default="./outputs/llm_judge_run", help="Run output directory.")
    ap.add_argument("--output-dir", default=None, help="Override dataset_gen output dir (default: <run-dir>/dataset_gen_output).")
    ap.add_argument("--qa-dir", default=None, help="Override QA dir (default: <run-dir>/qa).")

    ap.add_argument("--mineru-url", default=os.environ.get("MINERU_SERVER_URL", "http://127.0.0.1:18899"))
    ap.add_argument("--mineru-timeout-s", type=int, default=int(os.environ.get("MINERU_TIMEOUT_S", "7200")))
    ap.add_argument("--skip-mineru-healthcheck", action="store_true")
    ap.add_argument("--parse-format", default="mm_md", choices=["mm_md", "md_only", "content_list"])
    ap.add_argument("--lang", default="en", help="MinerU parse lang (default: en).")
    ap.add_argument("--start-page", type=int, default=0)
    ap.add_argument("--end-page", type=int, default=None)
    ap.add_argument("--keep-source", action="store_true", help="Copy original PDFs into docs/ mirror under output dir.")
    ap.add_argument("--force-ingest", action="store_true", help="Re-run MinerU ingest even if canonical/index already exists.")

    ap.add_argument("--sample-docs", type=int, default=5, help="Randomly sample N PDFs (default: 5; <=0 means all).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (used for sampling + generation).")
    ap.add_argument("--questions-per-doc", type=int, default=10)
    ap.add_argument("--gen-batch-size", type=int, default=2, help="Generate in small resume-able batches (default: 2).")
    ap.add_argument("--no-resume", action="store_true", help="Do not resume; overwrite QA outputs for each doc.")

    ap.add_argument("--prompt-lang", default="en", choices=["en", "zh", "zh-Hant"])
    ap.add_argument("--min-page-gap", type=int, default=3)
    ap.add_argument("--hard-min-evidence-sections", type=int, default=2)
    ap.add_argument("--easy-max-ratio", type=float, default=0.10)
    ap.add_argument("--unanswerable-ratio", type=float, default=0.15)
    ap.add_argument("--calc-ratio", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--search-limit", type=int, default=10)

    ap.add_argument("--gen-model", default=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    ap.add_argument("--verify-with-llm", action="store_true", help="Enable rejection sampling with an LLM judge.")
    ap.add_argument("--judge-model", default="gpt-4o", help="Judge model used when --verify-with-llm is set.")

    ap.add_argument("--fail-fast", action="store_true", help="Stop immediately on first error.")

    args = ap.parse_args()

    input_path = Path(args.input)
    pdfs = _list_pdfs(input_path, str(args.pattern))
    if not pdfs:
        raise SystemExit(f"No PDFs found under: {input_path}")

    sample_n = int(args.sample_docs)
    sample_n = None if sample_n <= 0 else sample_n
    pdfs = _sample(pdfs, sample_n=sample_n, seed=int(args.seed))

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ds_output_dir = Path(args.output_dir) if args.output_dir else (run_dir / "dataset_gen_output")
    qa_dir = Path(args.qa_dir) if args.qa_dir else (run_dir / "qa")
    qa_dir.mkdir(parents=True, exist_ok=True)

    cfg = AppConfig(output_dir=ds_output_dir)
    store = DocStore(cfg)

    _log(f"Selected {len(pdfs)} PDFs. run_dir={run_dir}")

    if not args.skip_mineru_healthcheck:
        if not _mineru_healthcheck(str(args.mineru_url)):
            raise SystemExit(f"MinerU healthcheck failed: {args.mineru_url}")

    # Enable MinerU client keep-alive logs by default (useful over SSH/proxies).
    os.environ.setdefault("MINERU_CLIENT_PROGRESS", "1")

    ingest_manifest: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "dataset_gen_output_dir": str(ds_output_dir),
        "mineru_url": str(args.mineru_url),
        "parse_format": str(args.parse_format),
        "docs": [],
    }
    ingest_manifest_path = run_dir / "ingest_manifest.json"

    _log(f"Ingest: {len(pdfs)} docs (timeout_s={args.mineru_timeout_s})")
    for i, pdf in enumerate(pdfs, start=1):
        rec = store.upsert_source(pdf)
        doc = store.list_docs().get(rec.doc_id, {})
        canon_ok = bool(doc.get("canonical_path")) and Path(str(doc.get("canonical_path"))).exists()
        index_ok = bool(doc.get("index_path")) and Path(str(doc.get("index_path"))).exists()
        if (canon_ok and index_ok) and (not args.force_ingest):
            _log(f"[ingest {i}/{len(pdfs)}] skip doc_id={rec.doc_id} file={pdf.name} (already ingested)")
            ingest_manifest["docs"].append({"pdf": str(pdf), "doc_id": rec.doc_id, "ingested": False, "reason": "already_ingested"})
            ingest_manifest_path.write_text(json.dumps(ingest_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            continue

        _log(f"[ingest {i}/{len(pdfs)}] start doc_id={rec.doc_id} file={pdf.name}")
        t0 = time.perf_counter()
        try:
            res = ingest_one_or_many(
                cfg,
                input_path=pdf,
                pattern="*.pdf",
                mineru_url=str(args.mineru_url),
                timeout_s=int(args.mineru_timeout_s),
                parse_format=str(args.parse_format),
                lang=str(args.lang),
                formula_enable=True,
                table_enable=True,
                start_page=int(args.start_page),
                end_page=args.end_page,
                caption_mode=None,
                caption_max_images=None,
                keep_source=bool(args.keep_source),
            )
            ingest_manifest["docs"].append({"pdf": str(pdf), "doc_id": rec.doc_id, "ingested": True, "ingest_result": res[0] if res else None})
            _log(f"[ingest {i}/{len(pdfs)}] done doc_id={rec.doc_id} elapsed_s={time.perf_counter()-t0:.1f}")
        except Exception as exc:
            _log(f"[ingest {i}/{len(pdfs)}] ERROR doc_id={rec.doc_id}: {exc}")
            ingest_manifest["docs"].append({"pdf": str(pdf), "doc_id": rec.doc_id, "ingested": False, "error": str(exc)})
            if args.fail_fast:
                raise
        finally:
            ingest_manifest_path.write_text(json.dumps(ingest_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Generate
    gen_manifest: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "qa_dir": str(qa_dir),
        "dataset_gen_output_dir": str(ds_output_dir),
        "questions_per_doc": int(args.questions_per_doc),
        "prompt_lang": str(args.prompt_lang),
        "gen_model": str(args.gen_model),
        "judge_model": str(args.judge_model),
        "verify_with_llm": bool(args.verify_with_llm),
        "docs": [],
    }
    gen_manifest_path = run_dir / "generate_manifest.json"

    _log(
        f"Generate: {len(pdfs)} docs x {args.questions_per_doc} questions/doc "
        f"(resume={not args.no_resume}, verify_with_llm={args.verify_with_llm})"
    )

    for i, pdf in enumerate(pdfs, start=1):
        rec = store.upsert_source(pdf)
        out_path = qa_dir / f"{rec.doc_id}.jsonl"
        _log(f"[gen {i}/{len(pdfs)}] start doc_id={rec.doc_id} out={out_path.name}")
        t0 = time.perf_counter()
        try:
            target = int(args.questions_per_doc)
            resume = (not bool(args.no_resume))
            batch = int(args.gen_batch_size)

            def _run_one(limit_now: int) -> int:
                return generate_qa(
                    cfg,
                    doc_id=rec.doc_id,
                    doc_ids=None,
                    out_jsonl_path=out_path,
                    mode="docdancer",
                    limit=int(limit_now),
                    easy_max_ratio=float(args.easy_max_ratio),
                    unanswerable_ratio=float(args.unanswerable_ratio),
                    hard_multi_doc_ratio=0.0,  # single-doc by default in this helper
                    calc_ratio=float(args.calc_ratio),
                    hard_min_evidence_sections=int(args.hard_min_evidence_sections),
                    min_page_gap=int(args.min_page_gap),
                    max_steps=int(args.max_steps),
                    search_limit=int(args.search_limit),
                    seed=int(args.seed),
                    write_debug=True,
                    resume=bool(resume),
                    verify_with_llm=bool(args.verify_with_llm),
                    llm_timeout_s=180,
                    explore_model=str(args.gen_model),
                    synth_model=str(args.gen_model),
                    judge_model=str(args.judge_model),
                    hard_require_multimodal=False,
                    read_with_images=False,
                    prompt_lang=args.prompt_lang,
                )

            if (not resume) or batch <= 0:
                count = _run_one(target)
            else:
                count = 0
                max_loops = max(3, (target // max(1, batch)) + 10)
                for _ in range(max_loops):
                    if count >= target:
                        break
                    limit_now = min(target, count + batch)
                    new_count = _run_one(limit_now)
                    if new_count <= count:
                        _log(f"[gen {i}/{len(pdfs)}] no progress (count={count}/{target}); stopping this doc")
                        break
                    count = new_count
                    _log(f"[gen {i}/{len(pdfs)}] progress doc_id={rec.doc_id} {count}/{target}")

            gen_manifest["docs"].append({"pdf": str(pdf), "doc_id": rec.doc_id, "qa_jsonl": str(out_path), "count": int(count)})
            _log(f"[gen {i}/{len(pdfs)}] done doc_id={rec.doc_id} count={count} elapsed_s={time.perf_counter()-t0:.1f}")
        except Exception as exc:
            _log(f"[gen {i}/{len(pdfs)}] ERROR doc_id={rec.doc_id}: {exc}")
            gen_manifest["docs"].append({"pdf": str(pdf), "doc_id": rec.doc_id, "qa_jsonl": str(out_path), "error": str(exc)})
            if args.fail_fast:
                raise
        finally:
            gen_manifest_path.write_text(json.dumps(gen_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    _log(f"Done. manifests: {ingest_manifest_path}, {gen_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

