# DataSet-Gen: Evidence-Grounded Document QA Dataset Synthesis

Ref: https://arxiv.org/abs/2601.05163

DataSet-Gen is a document QA dataset generator for **general-domain PDFs**. It builds a minimal document toolkit (`search`/`read`) on top of parsed PDFs, then synthesizes evaluation-oriented QA pairs with strict **evidence grounding**, **uniqueness**, and **difficulty(Multi-Hop)** constraints.

This repo also includes a standalone MinerU parsing service (`mineru/`) and reference scripts for the **LLM-as-a-judge** evaluation phase.

Chinese README: `README.zh.md`

## 1) Project Functionality

- **Ingest PDFs** via a parsing backend (MinerU) and build:
  - a canonical document representation (section tree + chunks)
  - a lightweight search index (SQLite FTS)
- **Generate QA pairs** (JSONL) with:
  - evidence-grounded answers (numeric token checks, calc replay)
  - hard/easy/unanswerable difficulty scheduling
  - optional rejection sampling using an LLM judge (`--verify-with-llm`)
- **Audit & inspect** generated datasets:
  - duplicates, semantic near-duplicates, section coverage, reachability stress test
  - show evidence snippets from `.debug.jsonl`

Outputs are evaluation-friendly JSONL files:
- `*.jsonl`: `{question, answer}`
- `*.jsonl.debug.jsonl`: per-item difficulty/kind/evidence ids/trajectories/derived calc replay info (may include extracted text; do not redistribute if you cannot share the source documents)

## 2) Usage (Core + MinerU)

### 2.1 Install

Python: `>= 3.10` (`python3`)

Install (editable install recommended):

```bash
python3 -m pip install -r requirements.txt
# optional (installs `dataset-gen` entrypoint)
python3 -m pip install -e .
```

Create a local `.env` (OpenAI-compatible). You can start from `.env.example`:

```bash
cp .env.example .env
# then edit .env
```

### 2.2 Start MinerU parsing service (included)

This repo includes a standalone FastAPI MinerU server under `mineru/`.

Start a local server on port `18899`:

```bash
./scripts/start_mineru_server_18899.sh
curl http://127.0.0.1:18899/health
```

Default multimodal settings (overridable via environment variables):
- Captioning: `content_list_then_llm` (requires an OpenAI-compatible multimodal chat endpoint)
- Caption limit: no limit (`MINERU_CAPTION_MAX_IMAGES=0`)
- Page screenshots and block crops: enabled (used by downstream `--read-with-images`)

Config knobs (set before starting the server):
- `MINERU_CAPTION_MODE`: `off|content_list|llm|content_list_then_llm`
- `MINERU_CAPTION_MAX_IMAGES`: `0` means no limit
- `MINERU_CHAT_API_BASE_URL` / `MINERU_CHAT_API_KEY` / `MINERU_CHAT_MODEL`: captioning chat endpoint
- `MINERU_DUMP_PAGE_SCREENSHOTS=1` / `MINERU_DUMP_BLOCK_CROPS=1` / `MINERU_CROP_IMAGES=1`

Notes:
- The server requires FastAPI deps (see `mineru/requirements.txt`).
- The server runtime must have upstream MinerU installed and working (CUDA/models/backends). See `mineru/README.md`.
- If you already run MinerU elsewhere, set `MINERU_SERVER_URL=http://<host>:<port>`.

### 2.3 Ingest a PDF (build canonical + index)

```bash
python3 -m dataset_gen ingest \
  --input /path/to/your.pdf \
  --output-dir ./outputs/dataset_gen_run \
  --mineru-url http://127.0.0.1:18899 \
  --parse-format mm_md
```

### 2.4 Generate QA (DocDancer-style)

DocDancer mode uses only two tools (`search`/`read`) to explore evidence, then synthesizes QA.

```bash
python3 -m dataset_gen generate \
  --output-dir ./outputs/dataset_gen_run \
  --doc-id <DOC_ID> \
  --out ./outputs/dataset_gen_run/qa.docdancer.jsonl \
  --mode docdancer \
  --prompt-lang en \
  --limit 100 \
  --min-page-gap 3 \
  --hard-min-evidence-sections 2 \
  --unanswerable-ratio 0.15 \
  --easy-max-ratio 0.10 \
  --verify-with-llm
```

Resume generation:

```bash
python3 -m dataset_gen generate --resume \
  --output-dir ./outputs/dataset_gen_run \
  --doc-id <DOC_ID> \
  --out ./outputs/dataset_gen_run/qa.docdancer.jsonl \
  --mode docdancer \
  --limit 100
```

### 2.5 Audit and inspect

```bash
python3 -m dataset_gen audit \
  --output-dir ./outputs/dataset_gen_run \
  --qa-jsonl ./outputs/dataset_gen_run/qa.docdancer.jsonl \
  --debug-jsonl ./outputs/dataset_gen_run/qa.docdancer.jsonl.debug.jsonl
```

```bash
python3 -m dataset_gen show \
  --debug-jsonl ./outputs/dataset_gen_run/qa.docdancer.jsonl.debug.jsonl \
  --n 3
```

## 3) Evaluation

Current evaluation uses **LLM-as-a-judge** plus local Gate checks (format/leakage, evidence constraints, numeric grounding, calc replay).

In the future, we will add training-based evaluation (fine-tune on synthesized data and compare on held-out benchmarks with strict leakage controls).

### 3.1 Reference scripts (included)

This repo includes two helper scripts under `scripts/`:
- `scripts/llm_judge_generate.py`: sample PDFs, ingest via MinerU, generate QA with an OpenAI-compatible LLM
- `scripts/llm_judge_evaluate.py`: evaluate generated QA using an LLM judge + local Gate

Example:

```bash
python3 scripts/llm_judge_generate.py \
  --input /path/to/pdfs \
  --mineru-url http://127.0.0.1:18899 \
  --sample-docs 5 --questions-per-doc 10 \
  --gen-model gpt-4o-mini --verify-with-llm --judge-model gpt-4o
```

```bash
python3 scripts/llm_judge_evaluate.py \
  --run-dir ./outputs/llm_judge_run \
  --judge-model gpt-4o --max-items 200
```

### 3.2 What we test (without shipping datasets)

We do not include third-party datasets in this repo. We validate the pipeline on public document QA datasets (e.g. MMLongBench-Doc) by:
- running end-to-end ingestion + generation on a sampled subset of documents
- auditing with Gate checks
- evaluating with an independent LLM judge rubric (fixed model/prompt)

### 3.3 Example results (MMLongBench-Doc sample run)

We tested on a sampled subset of **MMLongBench-Doc** (dataset PDFs not included). Environment and settings:

- MinerU parsing service running locally at `http://127.0.0.1:18899`
- Generation model: `gpt-4o-mini`
- Judge model: `gpt-4o`
- Sampling: `--sample-docs 5`, `--questions-per-doc 10`, `--verify-with-llm`

Run commands (paths are examples):

```bash
./scripts/start_mineru_server_18899.sh

python3 scripts/llm_judge_generate.py \
  --input datasets/MMLongBench-Doc/data/documents \
  --run-dir outputs/stage1_mmlongbench_doc \
  --mineru-url http://127.0.0.1:18899 \
  --sample-docs 5 --questions-per-doc 10 \
  --gen-model gpt-4o-mini --verify-with-llm --judge-model gpt-4o

python3 scripts/llm_judge_evaluate.py \
  --run-dir outputs/stage1_mmlongbench_doc \
  --out-dir outputs/stage1_mmlongbench_doc/eval_after_verify \
  --judge-model gpt-4o --max-items 200

python3 scripts/llm_judge_evaluate.py \
  --run-dir outputs/stage1_mmlongbench_doc \
  --out-dir outputs/stage1_mmlongbench_doc/eval_aligned \
  --judge-model gpt-4o --max-items 200
```

Reported metrics are read from these two reports:
- `outputs/stage1_mmlongbench_doc/eval_after_verify/quality_report.json`
- `outputs/stage1_mmlongbench_doc/eval_aligned/quality_report.json`

Comparison:

| Report | Total | Gate pass | Judge pass (supported+unique+difficulty_ok) | Aligned pass (supported+unique) | supported | unique | difficulty_ok | Difficulty mix |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `eval_after_verify` | 42 | 100% | 85.7% | 97.6% | 97.6% | 100% | 88.1% | hard=30, unanswerable=8, easy=4 |
| `eval_aligned` | 46 | 100% | 78.3% | 95.7% | 95.7% | 100% | 78.3% | hard=46 |

Notes:
- “Gate pass” is local rules (format/leakage/evidence constraints).
- “Judge pass” is LLM judge strict pass (supported+unique+difficulty_ok).
- “Aligned pass” ignores judge difficulty and focuses on supported+unique (useful when iterating difficulty rubrics).

Your exact numbers may vary with document samples, MinerU backend settings, and model versions/prompts.
