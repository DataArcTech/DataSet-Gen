import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dataset_gen.config import AppConfig
from dataset_gen.llm.openai_compat import OpenAICompatChatClient
from dataset_gen.prompts.docdancer import PromptLang
from dataset_gen.storage.doc_store import DocStore
from dataset_gen.tools.code_python import PythonSandboxLimits, run_code_python

from .docdancer_io import write_items_jsonl
from .docdancer_llm import (
    explore,
    judge_item_with_llm,
    select_evidence_from_trajectory as _select_evidence_from_trajectory,
    synthesize,
    synthesize_calc_spec,
)
from .docdancer_toolkit import MultiDocToolkit
from .docdancer_types import Difficulty, GeneratedItem, ItemKind
from .docdancer_utils import (
    extract_number_tokens as _extract_number_tokens,
    keywords_from_text as _keywords_from_text,
    guess_title_from_filename as _guess_title_from_filename,
    normalize_number_haystack as _normalize_number_haystack,
    pick_doc_labels_from_evidence as _pick_doc_labels_from_evidence,
    build_source_hint as _build_source_hint,
    validate_answer_with_citations as _validate_answer_with_citations,
    strip_square_citations as _strip_square_citations,
    repair_answer_with_citations as _repair_answer_with_citations,
)
from .docdancer_validation import (
    evidence_satisfies as _evidence_satisfies,
    has_multimodal_evidence as _has_multimodal_evidence,
    local_verify_constraints as _local_verify_constraints,
    validate_qa as _validate_qa,
)

__all__ = [
    "Difficulty",
    "ItemKind",
    "GeneratedItem",
    "generate_docdancer_items",
    "judge_item_with_llm",
    "write_items_jsonl",
]

def generate_docdancer_items(
    cfg: AppConfig,
    *,
    doc_ids: List[str],
    total: int,
    easy_max_ratio: float = 0.10,
    unanswerable_ratio: float = 0.15,
    hard_multi_doc_ratio: float = 0.5,
    calc_ratio: float = 0.0,
    hard_min_evidence_sections: int = 2,
    min_page_gap: int = 3,
    max_steps: int = 12,
    search_limit: int = 10,
    seed: Optional[int] = None,
    verify_with_llm: bool = False,
    llm_timeout_s: int = 180,
    explore_model: Optional[str] = None,
    synth_model: Optional[str] = None,
    judge_model: Optional[str] = None,
    hard_require_multimodal: bool = False,
    read_with_images: bool = False,
    prompt_lang: PromptLang = "en",
    anchor_doc: bool = False,
    anchor_doc_id: Optional[str] = None,
) -> Iterable[GeneratedItem]:
    rnd = random.Random(seed)
    trace = os.environ.get("DOCDANCER_TRACE", "0").strip().lower() in {"1", "true", "yes", "on"}

    def _tr(msg: str) -> None:
        if trace:
            print(f"[docdancer] {msg}", file=sys.stderr)

    # We model four "types" in output:
    # - easy (qa): single-chunk, direct extraction
    # - unanswerable (qa)
    # - calc (hard): python-sandbox grounded calculation
    # - hard (qa): multi-hop; can be single-doc (page gap) or cross-doc
    #
    # Ratios are applied to total and converted to integer targets.
    easy_target = int(total * float(easy_max_ratio))
    unans_target = int(total * float(unanswerable_ratio) + 0.5)
    calc_target = int(total * float(calc_ratio) + 0.5)
    easy_target = max(0, min(easy_target, total))
    unans_target = max(0, min(unans_target, total - easy_target))
    calc_target = max(0, min(calc_target, total - easy_target - unans_target))
    hard_target = total - easy_target - unans_target - calc_target

    store = DocStore(cfg)
    docs = store.list_docs()

    # Only keep docs that have the canonical + index artifacts we need for DocToolkit.
    available: List[str] = []
    doc_lang: Dict[str, str] = {}
    doc_assets: Dict[str, Dict[str, Any]] = {}
    doc_meta_by_id: Dict[str, Dict[str, Any]] = {}
    for did in doc_ids:
        rec = docs.get(did) if isinstance(docs, dict) else None
        if not isinstance(rec, dict):
            continue
        canon = rec.get("canonical_path")
        idx = rec.get("index_path")
        if not (isinstance(canon, str) and canon and Path(canon).exists()):
            continue
        if not (isinstance(idx, str) and idx and Path(idx).exists()):
            continue
        available.append(did)
        extra = rec.get("extra") if isinstance(rec.get("extra"), dict) else {}
        lang = str((extra or {}).get("doc_language") or "").strip()
        if lang:
            doc_lang[did] = lang

        filename = str(rec.get("filename") or "").strip()
        title = _guess_title_from_filename(filename)
        doc_meta_by_id[did] = {
            "filename": filename or None,
            "title": title or None,
            "source_path": str(rec.get("source_path") or "").strip() or None,
        }

        # Precompute asset directories once per doc to avoid repeated DocStore I/O and filesystem checks.
        md_path = rec.get("mineru_markdown_path")
        images_dir = None
        pages_dir = None
        crops_dir = None
        asset_manifest_path = rec.get("mineru_asset_manifest_path")
        if isinstance(md_path, str) and md_path:
            cand = Path(md_path).parent / "images"
            if cand.exists():
                images_dir = str(cand)
            pages_cand = Path(md_path).parent / "pages"
            if pages_cand.exists():
                pages_dir = str(pages_cand)
            crops_cand = Path(md_path).parent / "crops"
            if crops_cand.exists():
                crops_dir = str(crops_cand)

        if isinstance(asset_manifest_path, str) and asset_manifest_path:
            try:
                _ = json.loads(Path(asset_manifest_path).read_text(encoding="utf-8"))
                method_dir = Path(asset_manifest_path).parent
                if (method_dir / "pages").exists():
                    pages_dir = str((method_dir / "pages").resolve())
                if (method_dir / "crops").exists():
                    crops_dir = str((method_dir / "crops").resolve())
            except Exception:
                pass

        doc_assets[did] = {
            "images_dir": images_dir,
            "pages_dir": pages_dir,
            "crops_dir": crops_dir,
            "asset_manifest_path": asset_manifest_path,
        }

    if not available:
        raise RuntimeError(
            "No ingested docs available for generation (canonical/index missing). "
            "Run ingest first and ensure canonical.json + chunks.sqlite3 exist."
        )

    anchor_doc_id_final = (str(anchor_doc_id).strip() if anchor_doc_id else "")
    anchor_lang = doc_lang.get(anchor_doc_id_final) if anchor_doc_id_final else None

    # For "hard qa" we want a controlled split:
    # - (1 - hard_multi_doc_ratio): single-doc with page gap constraint
    # - hard_multi_doc_ratio: cross-doc evidence
    hard_qa_total = int(hard_target)
    hard_qa_multi = int(hard_qa_total * float(hard_multi_doc_ratio) + 0.5)
    hard_qa_multi = max(0, min(hard_qa_multi, hard_qa_total))
    if len(available) < 2:
        hard_qa_multi = 0
    if anchor_doc and anchor_doc_id_final in available and anchor_lang:
        same_lang = [d for d in available if doc_lang.get(d) == anchor_lang]
        if len(same_lang) < 2:
            hard_qa_multi = 0
    hard_qa_single = hard_qa_total - hard_qa_multi

    schedule: List[Tuple[Difficulty, ItemKind, bool]] = (
        ([("easy", "qa", False)] * easy_target)
        + ([("unanswerable", "qa", False)] * unans_target)
        + ([("hard", "calc", False)] * calc_target)
        + ([("hard", "qa", False)] * hard_qa_single)
        + ([("hard", "qa", True)] * hard_qa_multi)
    )
    rnd.shuffle(schedule)

    explore_model_final = explore_model or os.environ.get("OPENAI_EXPLORE_MODEL") or "gpt-4o-mini"
    synth_model_final = synth_model or os.environ.get("OPENAI_SYNTH_MODEL") or None
    llm_explore = OpenAICompatChatClient.from_env(timeout_s=llm_timeout_s, model=explore_model_final)
    llm_synth = OpenAICompatChatClient.from_env(timeout_s=llm_timeout_s, model=synth_model_final)
    judge_model_final = judge_model or os.environ.get("OPENAI_JUDGE_MODEL") or llm_synth.model
    llm_judge = OpenAICompatChatClient.from_env(timeout_s=llm_timeout_s, model=judge_model_final)

    for idx, (difficulty, kind, require_multi_doc) in enumerate(schedule, start=1):
        # Select documents for this item.
        selected_docs: List[str]
        if anchor_doc and anchor_doc_id_final and anchor_doc_id_final in available:
            primary = anchor_doc_id_final
        else:
            primary = rnd.choice(available)

        if require_multi_doc and len(available) >= 2:
            if anchor_doc and primary in available:
                other_pool = [d for d in available if d != primary]
                if anchor_lang:
                    other_pool = [d for d in other_pool if doc_lang.get(d) == anchor_lang] or other_pool
                other = rnd.choice(other_pool) if other_pool else primary
                selected_docs = [primary, other] if other != primary else [primary]
            else:
                # Best-effort keep same language when doc languages are known.
                if doc_lang.get(primary):
                    same = [d for d in available if d != primary and doc_lang.get(d) == doc_lang.get(primary)]
                    if same:
                        selected_docs = [primary, rnd.choice(same)]
                    else:
                        selected_docs = rnd.sample(available, k=2)
                else:
                    selected_docs = rnd.sample(available, k=2)
        else:
            selected_docs = [primary] if (anchor_doc and primary in available) else [rnd.choice(available)]

        assets = {did: (doc_assets.get(did) or {}) for did in selected_docs}
        doc_meta = {did: (doc_meta_by_id.get(did) or {}) for did in selected_docs}
        toolkit = MultiDocToolkit(cfg, selected_docs, assets=assets, doc_meta=doc_meta)
        try:
            guided_keywords: Optional[List[str]] = None
            for attempt in range(1, 6):
                _tr(f"slot={idx}/{len(schedule)} attempt={attempt} difficulty={difficulty} kind={kind} require_multi_doc={require_multi_doc} docs={selected_docs}")
                traj = explore(
                    llm=llm_explore,
                    toolkit=toolkit,
                    difficulty=difficulty,
                    require_multi_doc=require_multi_doc,
                    min_page_gap=min_page_gap,
                    max_steps=min(max_steps, 8) if difficulty == "easy" else max_steps,
                    search_limit=search_limit,
                    seed=(None if seed is None else (seed + idx * 100 + attempt)),
                    read_with_images=read_with_images,
                    guided_keywords=guided_keywords,
                    prompt_lang=prompt_lang,
                )
                if trace:
                    n_search = sum(1 for s in traj if getattr(s, "tool", "") == "search")
                    n_read = sum(1 for s in traj if getattr(s, "tool", "") == "read")
                    _tr(f"  traj: steps={len(traj)} search={n_search} read={n_read}")
                max_reads = 1 if difficulty == "easy" else (5 if difficulty == "hard" else 3)
                evidence, summaries = _select_evidence_from_trajectory(
                    traj,
                    max_reads=max_reads,
                    prefer_multimodal=(difficulty == "hard"),
                    difficulty=difficulty,
                    require_multi_doc=require_multi_doc,
                    min_page_gap=min_page_gap,
                )
                if summaries:
                    # Provide higher-level hints without replacing raw evidence.
                    evidence.append({"notes": "\n".join(summaries[:3])})
                if not _evidence_satisfies(
                    difficulty=difficulty,
                    evidence=evidence,
                    require_multi_doc=require_multi_doc,
                    min_page_gap=min_page_gap,
                    hard_min_evidence_sections=hard_min_evidence_sections,
                ):
                    _tr("  reject: evidence_satisfies=false")
                    continue

                section_evidence = [e for e in evidence if isinstance(e, dict) and e.get("section_id")]
                spec: Dict[str, Any] = {}
                outcome: Dict[str, Any] = {}
                if difficulty == "easy" and kind != "qa":
                    # Safety: easy must be a direct QA item.
                    continue
                if difficulty == "unanswerable" and kind != "qa":
                    continue

                qa: Dict[str, Any]
                if kind == "calc":
                    try:
                        spec = synthesize_calc_spec(
                            llm=llm_synth,
                            difficulty=difficulty,
                            require_multi_doc=require_multi_doc,
                            min_page_gap=min_page_gap,
                            evidence=evidence,
                            tries=3,
                            prompt_lang=prompt_lang,
                        )

                        # Ground calc inputs: numeric inputs should come from the raw evidence text (not reader summaries).
                        hay = "\n".join(
                            str(e.get("text") or "")
                            for e in section_evidence
                            if isinstance(e, dict) and isinstance(e.get("text"), str) and e.get("text")
                        )
                        hay_norm = _normalize_number_haystack(hay)
                        inputs_obj = spec.get("inputs") if isinstance(spec.get("inputs"), dict) else {}
                        grounded = True
                        for _, v in inputs_obj.items():
                            tokens = _extract_number_tokens(str(v), max_items=4)
                            for tok in tokens:
                                tok_norm = _normalize_number_haystack(tok)
                                if tok_norm and tok_norm not in hay_norm:
                                    grounded = False
                                    break
                            if not grounded:
                                break
                        if not grounded:
                            continue

                        outcome = run_code_python(
                            code=str(spec["code"]),
                            inputs=inputs_obj,
                            limits=PythonSandboxLimits(timeout_s=6.0),
                        )
                        if outcome.get("exec_status") != "ok":
                            continue
                        answer = str(outcome.get("result_text") or "").strip()
                        if not answer:
                            continue
                        qa = {"question": str(spec.get("question") or ""), "answer": answer}

                        ok, reason = _validate_qa(difficulty=difficulty, qa=qa, prompt_lang=prompt_lang)
                        if not ok:
                            continue
                        ok, reason = _local_verify_constraints(
                            kind="calc",
                            difficulty=difficulty,
                            require_multi_doc=require_multi_doc,
                            min_page_gap=min_page_gap,
                            hard_min_evidence_sections=hard_min_evidence_sections,
                            evidence=evidence,
                            question=str(qa["question"]),
                            answer=str(qa["answer"]),
                            prompt_lang=prompt_lang,
                        )
                        if not ok:
                            continue
                    except Exception:
                        continue
                else:
                    try:
                        qa = synthesize(
                            llm=llm_synth,
                            difficulty=difficulty,
                            require_multi_doc=require_multi_doc,
                            min_page_gap=min_page_gap,
                            evidence=evidence,
                            tries=3,
                            prompt_lang=prompt_lang,
                        )
                    except Exception:
                        _tr("  reject: synthesize_exception")
                        continue
                    ok, reason = _local_verify_constraints(
                        kind="qa",
                        difficulty=difficulty,
                        require_multi_doc=require_multi_doc,
                        min_page_gap=min_page_gap,
                        hard_min_evidence_sections=hard_min_evidence_sections,
                        evidence=evidence,
                        question=str(qa["question"]),
                        answer=str(qa["answer"]),
                        prompt_lang=prompt_lang,
                    )
                    if not ok:
                        _tr(f"  reject: local_verify_constraints(before_judge) {reason}")
                        continue
                if difficulty == "hard" and hard_require_multimodal:
                    if not _has_multimodal_evidence(section_evidence):
                        _tr("  reject: hard_require_multimodal_no_evidence")
                        continue
                if verify_with_llm:
                    # For calc items, attach derived computation so the judge can validate determinism/support.
                    judge_evidence = [e for e in evidence if not (isinstance(e, dict) and set(e.keys()) == {"notes"})]
                    if kind == "calc":
                        judge_evidence.append(
                            {
                                "derived": {
                                    "tool": "python_sandbox",
                                    "code": str(spec.get("code") or "")[:1200],
                                    "inputs": spec.get("inputs") if isinstance(spec.get("inputs"), dict) else {},
                                    "result_text": str(outcome.get("result_text") or "")[:500],
                                    "exec_status": outcome.get("exec_status"),
                                }
                            }
                        )
                    # Number evidence entries so the judge can cite as [eid].
                    numbered: List[Dict[str, Any]] = []
                    citation_map: List[Dict[str, Any]] = []
                    eid = 1
                    for evi in judge_evidence:
                        if not isinstance(evi, dict):
                            continue
                        # Only number readable evidence chunks/sections; derived notes are not citable.
                        if evi.get("section_id"):
                            # Prefer chunk-level citations: split one read that spans multiple chunk_ids
                            # into multiple citable entries (bounded) so the judge can cite [1][2][3].
                            cids = evi.get("chunk_ids") if isinstance(evi.get("chunk_ids"), list) else []
                            cids = [str(x) for x in cids if isinstance(x, str) and x.strip()]
                            if cids:
                                for cid in cids[: max(1, int(hard_min_evidence_sections))]:
                                    ee = dict(evi)
                                    ee["chunk_ids"] = [cid]
                                    ee["chunk_id"] = cid
                                    ee["eid"] = eid
                                    numbered.append(ee)
                                    citation_map.append(
                                        {
                                            "eid": eid,
                                            "doc_id": ee.get("doc_id"),
                                            "doc_filename": ee.get("doc_filename"),
                                            "doc_title": ee.get("doc_title"),
                                            "section_id": ee.get("section_id"),
                                            "chunk_ids": [cid],
                                            "page_idxs": ee.get("page_idxs"),
                                        }
                                    )
                                    eid += 1
                            else:
                                ee = dict(evi)
                                ee["eid"] = eid
                                numbered.append(ee)
                                citation_map.append(
                                    {
                                        "eid": eid,
                                        "doc_id": ee.get("doc_id"),
                                        "doc_filename": ee.get("doc_filename"),
                                        "doc_title": ee.get("doc_title"),
                                        "section_id": ee.get("section_id"),
                                        "chunk_ids": ee.get("chunk_ids"),
                                        "page_idxs": ee.get("page_idxs"),
                                    }
                                )
                                eid += 1
                        else:
                            numbered.append(evi)

                    judge_out = judge_item_with_llm(
                        llm_judge,
                        kind=kind,
                        difficulty=difficulty,
                        require_multi_doc=require_multi_doc,
                        min_page_gap=min_page_gap,
                        hard_min_evidence_sections=hard_min_evidence_sections,
                        evidence=numbered,
                        question=str(qa["question"]),
                        answer=str(qa["answer"]),
                        prompt_lang=prompt_lang,
                    )
                    if not (
                        judge_out.get("supported") is True
                        and judge_out.get("unique") is True
                        and judge_out.get("difficulty_ok") is True
                    ):
                        issues_text = " ".join([str(x) for x in (judge_out.get("issues") or [])])
                        guided_keywords = _keywords_from_text(issues_text + " " + str(qa.get("question") or ""), max_terms=10)
                        _tr(
                            f"  reject: judge_flags supported={judge_out.get('supported')} unique={judge_out.get('unique')} "
                            f"difficulty_ok={judge_out.get('difficulty_ok')} issues={judge_out.get('issues')}"
                        )
                        continue
                    guided_keywords = None

                    # Extract answer_with_citations from judge, retry once if missing/invalid.
                    allowed = {m.get("eid") for m in citation_map if isinstance(m, dict) and isinstance(m.get("eid"), int)}
                    allowed_eids = {int(x) for x in allowed if isinstance(x, int)}
                    if trace and difficulty == "hard":
                        _tr(f"  cite_pool: allowed_eids={sorted(allowed_eids)} citation_map_n={len(citation_map)}")
                    awc = judge_out.get("answer_with_citations")
                    if not isinstance(awc, str):
                        awc = ""
                    # Unanswerable requires the exact token in plain QA output; do not force citations there.
                    if difficulty == "unanswerable":
                        ok_cit, _ = (True, "")
                        awc = str(qa.get("answer") or "").strip()
                    else:
                        ok_cit, _ = (
                            _validate_answer_with_citations(answer_with_citations=awc, allowed_eids=allowed_eids)
                            if allowed_eids
                            else (False, "no_eids")
                        )
                    if (not ok_cit) and allowed_eids:
                        # Retry with a stronger instruction (judge_prompt already includes rules; this is a backstop).
                        judge_out2 = judge_item_with_llm(
                            llm_judge,
                            kind=kind,
                            difficulty=difficulty,
                            require_multi_doc=require_multi_doc,
                            min_page_gap=min_page_gap,
                            hard_min_evidence_sections=hard_min_evidence_sections,
                            evidence=numbered,
                            question=str(qa["question"]),
                            answer=str(qa["answer"]),
                            prompt_lang=prompt_lang,
                        )
                        awc2 = judge_out2.get("answer_with_citations") if isinstance(judge_out2, dict) else ""
                        if isinstance(awc2, str) and awc2.strip():
                            if difficulty == "unanswerable":
                                awc = str(qa.get("answer") or "").strip()
                                ok_cit = True
                            else:
                                ok_cit2, _ = _validate_answer_with_citations(answer_with_citations=awc2, allowed_eids=allowed_eids)
                                if ok_cit2:
                                    awc = awc2
                                    ok_cit = True
                                else:
                                    # Deterministic formatting repair (debug-only): enforce per-sentence citations.
                                    awc_fixed = _repair_answer_with_citations(
                                        answer_text=awc2,
                                        allowed_eids=allowed_eids,
                                        citation_map=citation_map,
                                        difficulty=difficulty,
                                        require_multi_doc=require_multi_doc,
                                    )
                                    ok_fix, _ = _validate_answer_with_citations(
                                        answer_with_citations=awc_fixed, allowed_eids=allowed_eids
                                    )
                                    if ok_fix:
                                        awc = awc_fixed
                                        ok_cit = True

                    # If we cannot obtain a valid cited answer, discard and retry; debug citations are required.
                    if allowed_eids and difficulty != "unanswerable":
                        ok_final, _ = _validate_answer_with_citations(answer_with_citations=awc, allowed_eids=allowed_eids)
                        if not ok_final:
                            awc_fixed = _repair_answer_with_citations(
                                answer_text=awc,
                                allowed_eids=allowed_eids,
                                citation_map=citation_map,
                                difficulty=difficulty,
                                require_multi_doc=require_multi_doc,
                            )
                            ok_final2, _ = _validate_answer_with_citations(
                                answer_with_citations=awc_fixed, allowed_eids=allowed_eids
                            )
                            if not ok_final2:
                                _tr("  reject: citations_invalid")
                                continue
                            awc = awc_fixed
                        # Multi-hop: for hard items require multiple distinct citations (minimum 2).
                        import re as _re

                        cited = {int(x) for x in _re.findall(r"\[(\d+)\]", str(awc))}
                        if difficulty == "hard":
                            if len(cited) < 2:
                                # If the judge keeps reusing the same eid everywhere (still format-valid),
                                # deterministically spread citations across available eids.
                                awc_fixed = _repair_answer_with_citations(
                                    answer_text=awc,
                                    allowed_eids=allowed_eids,
                                    citation_map=citation_map,
                                    difficulty=difficulty,
                                    require_multi_doc=require_multi_doc,
                                )
                                ok_fix, _ = _validate_answer_with_citations(
                                    answer_with_citations=awc_fixed, allowed_eids=allowed_eids
                                )
                                if ok_fix:
                                    awc = awc_fixed
                                    cited = {int(x) for x in _re.findall(r"\[(\d+)\]", str(awc))}
                            if len(cited) < 2:
                                _tr(f"  reject: hard_requires_2_distinct_citations cited={sorted(cited)}")
                                continue
                            if require_multi_doc:
                                by_eid = {m.get("eid"): m for m in citation_map if isinstance(m, dict) and isinstance(m.get("eid"), int)}
                                cited_docs = {str(by_eid.get(e, {}).get("doc_id") or "") for e in cited}
                                cited_docs = {d for d in cited_docs if d}
                                if len(cited_docs) < 2:
                                    # Same idea: add missing doc coverage via repair.
                                    awc_fixed = _repair_answer_with_citations(
                                        answer_text=awc,
                                        allowed_eids=allowed_eids,
                                        citation_map=citation_map,
                                        difficulty=difficulty,
                                        require_multi_doc=require_multi_doc,
                                    )
                                    ok_fix, _ = _validate_answer_with_citations(
                                        answer_with_citations=awc_fixed, allowed_eids=allowed_eids
                                    )
                                    if ok_fix:
                                        awc = awc_fixed
                                        cited = {int(x) for x in _re.findall(r"\[(\d+)\]", str(awc))}
                                        cited_docs = {str(by_eid.get(e, {}).get("doc_id") or "") for e in cited}
                                        cited_docs = {d for d in cited_docs if d}
                                if len(cited_docs) < 2:
                                    _tr("  reject: hard_crossdoc_requires_2_docs_in_citations")
                                    continue

                    # Prefer the (possibly rewritten) judged answer as the final clean answer for QA items.
                    # Keep citations only in debug.
                    final_answer_plain = str(qa.get("answer") or "").strip()
                    if kind == "qa" and difficulty != "unanswerable":
                        final_answer_plain = _strip_square_citations(str(awc))
                        qa["answer"] = final_answer_plain
                        ok2, reason2 = _local_verify_constraints(
                            kind="qa",
                            difficulty=difficulty,
                            require_multi_doc=require_multi_doc,
                            min_page_gap=min_page_gap,
                            hard_min_evidence_sections=hard_min_evidence_sections,
                            evidence=evidence,
                            question=str(qa["question"]),
                            answer=str(qa["answer"]),
                            prompt_lang=prompt_lang,
                        )
                        if not ok2:
                            _tr(f"  reject: local_verify_constraints(after_judge) {reason2}")
                            continue

                    # Attach debug-only info (sources + cited answer + citation map).
                    labels = _pick_doc_labels_from_evidence(evidence, require_multi_doc=require_multi_doc)
                    source_hint = _build_source_hint(labels=labels, prompt_lang=prompt_lang)
                    debug: Dict[str, Any] = {
                        "source_hint": source_hint,
                        "citation_map": citation_map,
                        "answer_with_citations": awc,
                        "question_len_chars": len(str(qa.get("question") or "").strip()),
                        "answer_len_chars_plain": len(str(qa.get("answer") or "").strip()),
                        "answer_len_chars_cited": len(str(awc or "").strip()),
                    }
                else:
                    # Still keep a lightweight source hint in debug when not verifying with LLM.
                    labels = _pick_doc_labels_from_evidence(evidence, require_multi_doc=require_multi_doc)
                    source_hint = _build_source_hint(labels=labels, prompt_lang=prompt_lang)
                    debug = {
                        "source_hint": source_hint,
                        "question_len_chars": len(str(qa.get("question") or "").strip()),
                        "answer_len_chars_plain": len(str(qa.get("answer") or "").strip()),
                    }

                evidence_section_ids = [str(e.get("section_id")) for e in section_evidence if e.get("section_id")]
                evidence_chunk_ids: List[str] = []
                for e in section_evidence:
                    for cid in (e.get("chunk_ids") or []):
                        if isinstance(cid, str) and cid:
                            evidence_chunk_ids.append(cid)

                derived: Optional[Dict[str, Any]] = None
                if kind == "calc":
                    derived = {
                        "tool": "python_sandbox",
                        "inputs": spec.get("inputs") if isinstance(spec.get("inputs"), dict) else {},
                        "code": str(spec.get("code") or ""),
                        "result_text": str(outcome.get("result_text") or ""),
                        "exec_status": outcome.get("exec_status"),
                        "elapsed_ms": outcome.get("elapsed_ms"),
                    }
                yield GeneratedItem(
                    question=str(qa["question"]).strip(),
                    answer=str(qa["answer"]).strip(),
                    difficulty=difficulty,
                    kind=kind,
                    used_doc_ids=selected_docs,
                    evidence_section_ids=evidence_section_ids,
                    evidence_chunk_ids=evidence_chunk_ids,
                    trajectory=traj,
                    derived=derived,
                    debug={
                        **(debug or {}),
                        # Preferred debug presentation fields (writer uses these for debug question/answer).
                        "question": str((debug or {}).get("source_hint") or "").strip() + (" " if (debug or {}).get("source_hint") else "") + str(qa["question"]).strip(),
                        "answer": (str((debug or {}).get("answer_with_citations") or "").strip() or str(qa["answer"]).strip()),
                    },
                )
                _tr("  accept")
                break
            else:
                # give up this slot
                yield GeneratedItem(
                    question="",
                    answer="",
                    difficulty=difficulty,
                    kind="qa",
                    used_doc_ids=selected_docs,
                    evidence_section_ids=[],
                    evidence_chunk_ids=[],
                    trajectory=[],
                    derived=None,
                )
        finally:
            toolkit.close()
