from dataclasses import dataclass
from typing import Literal, TypeVar


PromptLang = Literal["en", "zh", "zh-Hant"]
RetryKind = Literal["qa", "calc"]

T = TypeVar("T")


@dataclass(frozen=True)
class DocDancerPrompts:
    lang: PromptLang = "en"

    def _t(self, en: T, zh: T, zht: T) -> T:
        if self.lang == "zh":
            return zh
        if self.lang == "zh-Hant":
            return zht
        return en

    def tool_schema(self) -> str:
        return self._t(
            en=(
                "You may only use two tools: search and read.\n\n"
                "search: keyword full-text search.\n"
                "Input JSON: {\"intent\": \"...\", \"tool\": \"search\", \"args\": {\"keywords\": [\"...\", \"...\"]}}\n\n"
                "read: read text evidence by section_ids or chunk_ids, and extract evidence relevant to a goal.\n"
                "Input JSON: {\"intent\": \"...\", \"tool\": \"read\", \"args\": {\"section_ids\": [\"...\"], \"chunk_ids\": [\"...\"], \"goal\": \"...\"}}\n"
                "Notes:\n"
                "- Provide either section_ids or chunk_ids (or both).\n"
                "- For easy items, prefer reading exactly ONE chunk_id.\n\n"
                "When you believe evidence is sufficient, finish:\n"
                "Input JSON: {\"intent\": \"...\", \"tool\": \"finish\", \"args\": {}}\n\n"
                "Strict requirement: every response must be exactly one JSON object and nothing else."
            ),
            zh=(
                "你只能使用两个工具：search 和 read。\n\n"
                "search：关键词全文检索。\n"
                "输入 JSON：{\"intent\": \"...\", \"tool\": \"search\", \"args\": {\"keywords\": [\"...\", \"...\"]}}\n\n"
                "read：按 section_ids 或 chunk_ids 读取文本，并按 goal 抽取证据。\n"
                "输入 JSON：{\"intent\": \"...\", \"tool\": \"read\", \"args\": {\"section_ids\": [\"...\"], \"chunk_ids\": [\"...\"], \"goal\": \"...\"}}\n"
                "说明：\n"
                "- 你可以只提供 section_ids 或只提供 chunk_ids（也可以都提供）。\n"
                "- 对于 easy 题，优先只读取 1 个 chunk_id。\n\n"
                "当你认为证据足够时结束：\n"
                "输入 JSON：{\"intent\": \"...\", \"tool\": \"finish\", \"args\": {}}\n\n"
                "严格要求：每次回复必须只包含一个 JSON 对象，不能输出任何额外文本。"
            ),
            zht=(
                "你只能使用兩個工具：search 和 read。\n\n"
                "search：關鍵詞全文檢索。\n"
                "輸入 JSON：{\"intent\": \"...\", \"tool\": \"search\", \"args\": {\"keywords\": [\"...\", \"...\"]}}\n\n"
                "read：按 section_ids 或 chunk_ids 讀取文字，並按 goal 抽取證據。\n"
                "輸入 JSON：{\"intent\": \"...\", \"tool\": \"read\", \"args\": {\"section_ids\": [\"...\"], \"chunk_ids\": [\"...\"], \"goal\": \"...\"}}\n"
                "說明：\n"
                "- 你可以只提供 section_ids 或只提供 chunk_ids（也可以都提供）。\n"
                "- 對於 easy 題，優先只讀取 1 個 chunk_id。\n\n"
                "當你認為證據足夠時結束：\n"
                "輸入 JSON：{\"intent\": \"...\", \"tool\": \"finish\", \"args\": {}}\n\n"
                "嚴格要求：每次回覆必須只包含一個 JSON 物件，不能輸出任何額外文字。"
            ),
        )

    def unanswerable_answer(self) -> str:
        return self._t(
            en="UNANSWERABLE",
            zh="无法回答",
            zht="無法回答",
        )

    def json_retry(self) -> str:
        return self._t(
            en="Output exactly one JSON object (no Markdown, no explanation) that can be parsed by json.loads.",
            zh="请严格只输出一个 JSON 对象（不要 Markdown，不要解释），并确保可被 json.loads 解析。",
            zht="請嚴格只輸出一個 JSON 物件（不要 Markdown，不要解釋），並確保可被 json.loads 解析。",
        )

    def suggested_search_keyword_groups(self) -> list[list[str]]:
        return self._t(
            en=[
                ["definition", "term", "glossary"],
                ["table", "amount", "ratio"],
                ["workflow", "process", "steps"],
                ["example", "case", "illustration"],
                ["calculation", "formula", "rate"],
            ],
            zh=[
                ["定义", "术语", "解释"],
                ["表", "金额", "比例"],
                ["流程", "步骤", "条件"],
                ["示例", "举例", "说明"],
                ["计算", "公式", "费率"],
            ],
            zht=[
                ["定義", "術語", "解釋"],
                ["表", "金額", "比例"],
                ["流程", "步驟", "條件"],
                ["示例", "舉例", "說明"],
                ["計算", "公式", "費率"],
            ],
        )

    def exploration_intro(self, *, difficulty: str, require_multi_doc: bool, min_page_gap: int) -> str:
        if self.lang == "en":
            if difficulty == "easy":
                constraints = (
                    "- Single-hop allowed: evidence from 1 chunk is sufficient.\n"
                    "- Prefer direct extraction questions.\n"
                    "- Prefer reading exactly ONE chunk_id.\n"
                )
            elif difficulty == "unanswerable":
                constraints = "- The goal is an unanswerable question: the documents do not provide a unique answer.\n- Use search/read to attempt answering and confirm the gap/ambiguity.\n"
            else:
                constraints = (
                    "- Must be multi-hop: require evidence from at least 2 different chunks.\n"
                    + ("- Must be cross-document: evidence from at least 2 documents.\n" if require_multi_doc else f"- Must have a clear span: page gap >= {min_page_gap} within a single document (or obvious section separation).\n")
                    + "- Avoid simple paraphrase or single-location lookup.\n"
                )

            return (
                "You are exploring processed long documents to discover meaningful, non-random question opportunities and collect evidence.\n"
                "You must use tool calls to obtain information; do not invent content.\n\n"
                "Exploration guidance:\n"
                "- Questions do NOT have to be business-specific, but should be concrete, information-rich, and uniquely answerable from evidence.\n"
                "- Prefer titles/definitions/tables/numbers/workflows/examples as \"askable\" anchors.\n"
                "- After search hits, quickly read relevant sections to capture citeable evidence.\n"
                "- Avoid repeating the same search/read.\n\n"
                f"Target difficulty: {difficulty}\n"
                f"{constraints}\n"
                "Output requirement: return only one JSON (intent+tool+args)."
            )

        # zh / zh-Hant
        if difficulty == "easy":
            constraints = "- 允许单跳：证据来自 1 个 chunk 即可。\n- 问题应为直接抽取型。\n- 优先只读取 1 个 chunk_id。\n"
        elif difficulty == "unanswerable":
            constraints = "- 目标是生成“无法回答”的问题：文档中没有给出唯一答案。\n- 你需要用 search/read 尝试寻找答案，并确认缺失/歧义。\n"
        else:
            constraints = (
                "- 必须多跳：至少需要 2 个不同 chunk 的证据才能回答。\n"
                + ("- 必须跨文档：证据来自至少 2 份不同文档。\n" if require_multi_doc else f"- 必须跨跨度：同一文档页跨度至少 {min_page_gap}（或明显章节跨度）。\n")
                + "- 不能是简单同义复述或单段落定位。\n"
            )
        body_zh = (
            "你正在探索一批已解析的长文档内容，目标是发现“有实际意义”的可提问点并收集证据。\n"
            "你必须通过工具调用来获取信息，不要凭空编造。\n\n"
            "探索策略：\n"
            "- 问题可以与业务不相关，但应当具体、有信息量，且可由证据唯一确定。\n"
            "- 优先从章节标题/定义/表格/数值/流程/示例中寻找“可问点”，避免纯随机。\n"
            "- 每次 search 命中后，应尽快 read 相关 section 以获得可引用证据。\n"
            "- 避免重复同样的 search/read。\n\n"
            f"本次目标难度：{difficulty}\n"
            f"{constraints}\n"
            "输出要求：只输出一个 JSON（intent+tool+args），不输出其他文字。"
        )
        body_zht = body_zh.replace("证据", "證據").replace("输出", "輸出").replace("难度", "難度").replace("相关", "相關")
        return body_zh if self.lang == "zh" else body_zht

    def default_read_goal(self) -> str:
        return self._t(
            en="Extract key information relevant to potential questions (definitions/numbers/tables/workflows/conditions/examples).",
            zh="提取与潜在可提问点相关的关键信息（定义/数值/表格/流程/条件/例子）。",
            zht="提取與潛在可提問點相關的關鍵資訊（定義/數值/表格/流程/條件/例子）。",
        )

    def reader_system(self) -> str:
        return self._t(
            en="You are a reader. Extract only information relevant to the goal from the provided content. Output JSON: {\"summary\": string}.",
            zh="你是一个阅读器。只基于给定内容提取与目标相关的关键信息，避免猜测。输出 JSON：{summary: string}",
            zht="你是一個閱讀器。只基於給定內容提取與目標相關的關鍵資訊，避免猜測。輸出 JSON：{summary: string}",
        )

    def synthesis_system(self) -> str:
        return self._t(
            en="You are a rigorous dataset generation assistant.",
            zh="你是一个严谨的数据集生成助手。",
            zht="你是一個嚴謹的資料集生成助手。",
        )

    def synthesis_retry_invalid(self, *, kind: RetryKind, reason: str) -> str:
        if kind == "calc":
            return self._t(
                en=f"Your last output was invalid: {reason}. Regenerate; output JSON(question,inputs,code) only.",
                zh=f"上一次输出不合格：{reason}。请重新生成，仍然只输出 JSON(question,inputs,code)。",
                zht=f"上一次輸出不合格：{reason}。請重新生成，仍然只輸出 JSON(question,inputs,code)。",
            )
        return self._t(
            en=f"Your last output was invalid: {reason}. Regenerate; output JSON(question,answer) only.",
            zh=f"上一次输出不合格：{reason}。请重新生成，仍然只输出 JSON(question,answer)。",
            zht=f"上一次輸出不合格：{reason}。請重新生成，仍然只輸出 JSON(question,answer)。",
        )

    def synthesis_qa_prompt(self, *, difficulty: str, require_multi_doc: bool, min_page_gap: int, evidence_json: str) -> str:
        if self.lang == "en":
            base = (
                "Generate one QA pair based only on the provided evidence snippets.\n"
                "Hard requirements:\n"
                "- Output exactly one JSON object with only two fields: question and answer.\n"
                "- The question must be natural, unambiguous, and ask exactly one thing with a unique answer.\n"
                "- The question must be document-grounded (not answerable by common sense).\n"
                "- Do NOT mention page/chunk/section ids, tools, search, read, or trajectories.\n"
                "- Do NOT use explicit location markers such as \"page X\", \"Figure X\", \"Table X\"; instead use 1-3 content hints (term, clause name, table header, key phrase).\n"
            )
            if difficulty == "unanswerable":
                diff = f"- This must be unanswerable. The answer must be exactly: {self.unanswerable_answer()}.\n"
            elif difficulty == "easy":
                diff = "- Easy: direct extraction; one evidence chunk is sufficient.\n"
            else:
                diff = (
                    "- Hard: must be multi-hop; require synthesizing at least two evidence chunks.\n"
                    + ("- Evidence must come from at least two different documents.\n" if require_multi_doc else f"- Evidence must have a clear span: page gap >= {min_page_gap} within one document.\n")
                    + "- Keep the answer short (entity/number/phrase/list).\n"
                )
            return base + diff + "\nEvidence (JSON list):\n" + evidence_json

        # zh / zh-Hant
        base = (
            "你将基于给定的证据（来自文档内容的片段）生成一个问答对，用于评估。\n"
            "硬性要求：\n"
            "- 只能输出一个 JSON 对象，且只能包含两个字段：question 和 answer。\n"
            "- question 必须自然、明确、只问一个问题，对应唯一答案。\n"
            "- question 必须高度依赖文档，不能凭常识回答。\n"
            "- question/answer 中不得提及 page/chunk/section id、工具、搜索、阅读、轨迹。\n"
            "- 不要出现“第X页/图X/表X/章节编号”等显式定位信息；请用 1-3 个内容线索引导定位（例如：术语、条款名、表头字段、关键短语）。\n"
        )
        if difficulty == "unanswerable":
            diff = f"- 该题必须不可回答，answer 必须严格为：{self.unanswerable_answer()}。\n"
        elif difficulty == "easy":
            diff = "- 该题为简单抽取型，可由单一 chunk 证据直接得到答案。\n"
        else:
            diff = (
                "- 该题必须多跳：需要综合至少两条 chunk 证据才能得到答案。\n"
                + ("- 证据必须来自至少两份不同文档。\n" if require_multi_doc else f"- 证据必须来自同一文档中跨度明显的两处（页码差至少 {min_page_gap}）。\n")
                + "- 答案尽量短：实体/数值/短语/列表，避免长段解释。\n"
            )
        body_zh = base + diff + "\n证据如下（JSON 列表）：\n" + evidence_json
        body_zht = body_zh.replace("证据", "證據").replace("输出", "輸出").replace("必须", "必須").replace("页码", "頁碼")
        return body_zh if self.lang == "zh" else body_zht

    def synthesis_calc_prompt(self, *, difficulty: str, require_multi_doc: bool, min_page_gap: int, evidence_json: str) -> str:
        if self.lang == "en":
            base = (
                "Generate one CALCULATION item based on the evidence.\n"
                "The final answer must come from executing Python code in a sandbox (NOT from mental math).\n\n"
                "Output must be exactly one JSON object with only three fields: question, inputs, code.\n"
                "- question: one clear question that depends on evidence.\n"
                "- inputs: JSON object of extracted parameters (numbers or strings only).\n"
                "- code: Python code string that uses INPUTS and sets variable result as the final answer (prefer a string with unit if needed).\n"
                "Hard constraints:\n"
                "- Do NOT mention page/chunk/section ids, tools, trajectories, or explicit location markers.\n"
                "- code must reference INPUTS and perform arithmetic (+ - * / **).\n"
                "- For hard items, code must reference at least TWO distinct INPUTS keys (e.g., INPUTS['a'] and INPUTS['b']).\n"
                "- No file/network I/O.\n"
            )
            if difficulty == "easy":
                diff = "- Easy: allow computing from one evidence chunk.\n"
            else:
                diff = (
                    "- Hard: must combine information from at least two different evidence chunks.\n"
                    + ("- Must be cross-document (>=2 doc_id).\n" if require_multi_doc else f"- Must have a clear span: page gap >= {min_page_gap} within one document.\n")
                )
            return base + diff + "\nEvidence (JSON list):\n" + evidence_json

        # zh / zh-Hant
        base = (
            "你将基于给定证据生成一个【计算题】问答，用于评估。\n"
            "必须让答案来自 Python 代码沙箱执行结果（而不是你自己心算/估算）。\n\n"
            "输出必须是一个 JSON 对象，且只能包含三个字段：question, inputs, code。\n"
            "- question: 自然语言问题，只问一个问题，且必须依赖证据。\n"
            "- inputs: 你从 evidence 中抽取的输入参数（JSON object），值必须是数字或字符串。\n"
            "- code: Python 代码字符串；只能使用 INPUTS 里的参数；必须设置变量 result 作为最终答案（建议为字符串，包含单位/百分号等）。\n"
            "硬性约束：\n"
            "- question 中不得出现 page/chunk/section id、工具、轨迹，以及“第X页/图X/表X”等显式定位。\n"
            "- code 必须引用 INPUTS，并且包含至少一次算术运算（+ - * / **）。\n"
            "- 不要在 code 里进行文件/网络 I/O。\n"
        )
        if difficulty == "easy":
            diff = "- 难度 easy：允许只用 1 个 chunk 证据中的数字完成计算。\n"
        else:
            diff = (
                "- 难度 hard：计算必须综合至少两个不同 chunk 证据的信息。\n"
                + ("- 且证据必须跨文档（至少 2 个 doc_id）。\n" if require_multi_doc else f"- 且证据必须来自同一文档中跨度明显的两处（页码差至少 {min_page_gap}）。\n")
            )
        body_zh = base + diff + "\n证据如下（JSON 列表）：\n" + evidence_json
        body_zht = body_zh.replace("证据", "證據").replace("输出", "輸出").replace("必须", "必須").replace("页码", "頁碼")
        return body_zh if self.lang == "zh" else body_zht

    def judge_system(self) -> str:
        return self._t(
            en="You are a strict JSON reviewer.",
            zh="你是一个严格的 JSON 审核器。",
            zht="你是一個嚴格的 JSON 審核器。",
        )

    def judge_prompt(
        self,
        *,
        kind: str,
        difficulty: str,
        require_multi_doc: bool,
        min_page_gap: int,
        hard_min_evidence_sections: int = 2,
        evidence_json: str,
        question: str,
        answer: str,
    ) -> str:
        if self.lang == "en":
            extra = ""
            if kind == "calc":
                extra = "If evidence contains derived.python_sandbox, treat result_text as deterministic computation output for calc items.\n"
            rules = [
                f"- difficulty={difficulty}",
            ]
            if difficulty == "hard" and require_multi_doc:
                rules.append(
                    f"- hard must cite >= {int(hard_min_evidence_sections)} distinct evidence chunk_id "
                    "(count unique chunk_ids across evidence)."
                )
                rules.append("- hard must be cross-document (>=2 distinct doc_id among evidence).")
            if difficulty == "hard" and not require_multi_doc:
                rules.append(
                    f"- hard must cite >= {int(hard_min_evidence_sections)} distinct evidence chunk_id "
                    "(count unique chunk_ids across evidence)."
                )
                rules.append(
                    f"- hard single-doc page_gap is defined as page_span = max(page_idxs) - min(page_idxs) across ALL evidence pages; require page_span >= {min_page_gap}.\n"
                    "  Do NOT require every adjacent page gap >= threshold."
                )
            if difficulty == "unanswerable":
                rules.append(f"- unanswerable answer must be exactly '{self.unanswerable_answer()}'.")
            rules.append(
                "- No explicit location markers or ids in QUESTION/ANSWER (page/chunk/section ids, Figure/Table numbers, etc.).\n"
                "  Evidence may contain such markers; do not penalize the item for markers appearing inside evidence."
            )
            return (
                "You are a strict dataset reviewer. Judge the QA only using the evidence.\n"
                + extra
                + "Output must be JSON:\n"
                "{\n"
                '  \"supported\": boolean,\n'
                '  \"unique\": boolean,\n'
                '  \"difficulty_ok\": boolean,\n'
                '  \"issues\": [string]\n'
                "}\n\n"
                "Difficulty constraints:\n"
                + "\n".join(rules)
                + "\n\nEvidence (JSON):\n"
                + evidence_json
                + "\n\nQuestion:\n"
                + question
                + "\n\nAnswer:\n"
                + answer
            )

        # zh / zh-Hant
        extra = ""
        if kind == "calc":
            extra = "如果 evidence 中包含 derived.python_sandbox，则其 result_text 视为确定性的计算结果（用于校验计算题）。\n"
        rules = [f"- difficulty={difficulty}"]
        if difficulty == "hard" and require_multi_doc:
            rules.append(f"- hard 必须引用至少 {int(hard_min_evidence_sections)} 个不同 evidence chunk_id（按 chunk_ids 去重统计）。")
            rules.append("- hard 必须跨文档（证据中至少 2 个不同 doc_id）。")
        if difficulty == "hard" and not require_multi_doc:
            rules.append(f"- hard 必须引用至少 {int(hard_min_evidence_sections)} 个不同 evidence chunk_id（按 chunk_ids 去重统计）。")
            rules.append(
                f"- hard 单文档页跨度定义为：page_span = max(page_idxs) - min(page_idxs)（跨所有证据页）；要求 page_span >= {min_page_gap}。\n"
                "  不要求相邻页差都满足阈值。"
            )
        if difficulty == "unanswerable":
            rules.append(f"- unanswerable 的答案必须是“{self.unanswerable_answer()}”。")
        rules.append(
            "- 禁止在 question/answer 中出现 page/chunk/section id、工具、图/表编号等显式定位；证据中出现这些字样不算违规。"
        )
        body_zh = (
            "你是一个严格的评估数据集审稿人。请仅基于证据判断 question/answer 是否合格。\n"
            + extra
            + "输出必须是 JSON：\n"
            "{\n"
            '  "supported": boolean,\n'
            '  "unique": boolean,\n'
            '  "difficulty_ok": boolean,\n'
            '  "issues": [string]\n'
            "}\n\n"
            "难度约束：\n"
            + "\n".join(rules)
            + "\n\n证据（JSON）：\n"
            + evidence_json
            + "\n\n待审问题：\n"
            + question
            + "\n\n待审答案：\n"
            + answer
        )
        body_zht = body_zh.replace("证据", "證據").replace("输出", "輸出").replace("难度", "難度").replace("必须", "必須")
        return body_zh if self.lang == "zh" else body_zht
