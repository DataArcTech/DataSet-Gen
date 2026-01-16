# DataSet-Gen：基于证据的文档 QA 数据合成

DataSet-Gen 是一个面向**通用领域 PDF 文档**的 QA 数据集生成工具。它在解析后的文档上构建最小工具集（`search`/`read`），再合成用于评测的问答对，并强调：
- **证据支持（evidence-grounded）**
- **唯一性（unique / unambiguous）**
- **难度约束（easy / hard / unanswerable）（多跳推理问答问题合成）**

本仓库同时包含一个可独立部署的 MinerU 解析服务（`mineru/`），以及用于 **LLM-as-a-judge** 评测阶段的脚本（`scripts/`）。

English README: `README.md`

## 1. 项目功能

- **PDF 入库（ingest）**：调用 MinerU 服务解析 PDF，构建
  - 统一的 canonical 表示（section tree + chunks）
  - 轻量搜索索引（SQLite FTS）
- **QA 合成（generate）**：
  - 通过 `search/read` 探索证据，再生成 `question/answer`
  - 支持 hard/easy/unanswerable 的比例与约束
  - 可选开启 `--verify-with-llm` 做拒绝采样（LLM judge 不通过则重采样）
- **质量审计与抽查（audit/show）**：
  - 格式/去重/覆盖统计，可选语义去重与可达性压力测试
  - 从 `.debug.jsonl` 直接查看证据片段与生成轨迹

输出为评测友好的 JSONL：
- `*.jsonl`：`{question, answer}`
- `*.jsonl.debug.jsonl`：包含 difficulty/kind、evidence ids、trajectory、calc derived 等追溯信息

## 2. 使用方法（项目本体 + MinerU）

### 2.1 安装

Python：`>= 3.10`（`python3`）

建议以 editable 模式安装：

```bash
python3 -m pip install -e .
```

创建本地 `.env`（OpenAI 兼容）。可从 `.env.example` 复制一份开始：

```bash
cp .env.example .env
# 然后编辑 .env
```

### 2.2 启动 MinerU 解析服务（仓库内置）

本仓库提供一个基于 FastAPI 的 MinerU 服务封装（`mineru/`）。

本地启动（18899 端口）：

```bash
./scripts/start_mineru_server_18899.sh
curl http://127.0.0.1:18899/health
```

说明：
- 服务依赖 FastAPI/uvicorn 等（脚本会提示缺失依赖如何安装）。
- 运行环境需要已正确安装并可用的上游 MinerU（CUDA/模型/后端）。详见 `mineru/README.md`。
- 如果你有独立部署的 MinerU 服务，可设置 `MINERU_SERVER_URL=http://<host>:<port>`。

### 2.3 入库一个 PDF（构建 canonical + index）

```bash
python3 -m dataset_gen ingest \
  --input /path/to/your.pdf \
  --output-dir ./outputs/dataset_gen_run \
  --mineru-url http://127.0.0.1:18899 \
  --parse-format mm_md
```

### 2.4 合成 QA（DocDancer 风格：仅 search/read）

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

断点续跑：

```bash
python3 -m dataset_gen generate --resume \
  --output-dir ./outputs/dataset_gen_run \
  --doc-id <DOC_ID> \
  --out ./outputs/dataset_gen_run/qa.docdancer.jsonl \
  --mode docdancer \
  --limit 100
```

### 2.5 审计与抽查

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

## 3. 评测

当前评测阶段采用 **LLM-as-a-judge** + 本地 Gate：
- Gate：格式/泄露检测、证据与难度约束、数值一致性、calc 可复算等
- Judge：独立 LLM 对 supported/unique/difficulty_ok 打分并给出 issues

后续将加入训练型评测：用合成数据微调模型，并在固定 benchmark 上做对比评测（含严格防泄漏约束）。

### 3.1 LLM judge 阶段脚本（仓库内置）

仓库 `scripts/` 目录提供两段式脚本：
- `scripts/llm_judge_generate.py`：抽样 PDF → 入库 → 生成 QA
- `scripts/llm_judge_evaluate.py`：对生成结果进行 Gate + LLM judge 评测，输出报告与失败样本

示例：

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

### 3.2 不包含第三方数据集

本仓库不分发第三方数据集文件。我们会在公开 benchmark 的**抽样子集**上做端到端测试（解析→生成→评测），以验证流程稳定性并迭代失败模式。

### 3.3 示例结果（MMLongBench-Doc 抽样运行）

我们在 **MMLongBench-Doc** 的抽样子集上做过一次端到端测试（数据集 PDF 不包含在仓库中）。运行环境与设置：

- MinerU 解析服务：本地 `http://127.0.0.1:18899`
- 生成模型：`gpt-4o-mini`
- 评测模型：`gpt-4o`
- 抽样：`--sample-docs 5`，每文档最多 `--questions-per-doc 10`，开启 `--verify-with-llm`

复现实验（路径仅示例）：

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

对比数据来自两份报告：
- `outputs/stage1_mmlongbench_doc/eval_after_verify/quality_report.json`
- `outputs/stage1_mmlongbench_doc/eval_aligned/quality_report.json`

对比表：

| 报告 | 总数 | Gate 通过率 | Judge 严格通过率（supported+unique+difficulty_ok） | Aligned 通过率（supported+unique） | supported | unique | difficulty_ok | 难度分布 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `eval_after_verify` | 42 | 100% | 85.7% | 97.6% | 97.6% | 100% | 88.1% | hard=30, unanswerable=8, easy=4 |
| `eval_aligned` | 46 | 100% | 78.3% | 95.7% | 95.7% | 100% | 78.3% | hard=46 |

说明：
- Gate：本地规则（格式/泄露/证据约束等）。
- Judge 严格通过率：LLM judge 同时满足 supported+unique+difficulty_ok。
- Aligned 通过率：忽略 judge 的 difficulty，只看 supported+unique（便于迭代难度口径时做稳定对比）。

注：具体数值会随抽样文档、MinerU 后端配置、模型版本/提示词变化而波动。
