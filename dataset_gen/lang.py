from __future__ import annotations

from typing import Optional

from dataset_gen.prompts.docdancer import PromptLang


# Very small heuristic language detector (no external deps).
# We follow MinerU's idea (CJK ratio -> zh vs en) and extend it to zh-Hant vs zh.
_CJK_RANGE = ("\u4e00", "\u9fff")

# Common simplified/traditional variant pairs. We count occurrences to guess zh vs zh-Hant.
# This is heuristic (not perfect), but works well for typical PDFs.
_VARIANT_PAIRS: list[tuple[str, str]] = [
    ("这", "這"),
    ("里", "裡"),
    ("后", "後"),
    ("为", "為"),
    ("国", "國"),
    ("东", "東"),
    ("发", "發"),
    ("应", "應"),
    ("与", "與"),  # note: can appear in both, still useful in aggregate
    ("条", "條"),
    ("责", "責"),
    ("权", "權"),
    ("险", "險"),
    ("综", "綜"),
    ("简", "簡"),
    ("证", "證"),
    ("标", "標"),
    ("额", "額"),
    ("计", "計"),
    ("须", "須"),
    ("际", "際"),
    ("续", "續"),
    ("赔", "賠"),
    ("审", "審"),
    ("终", "終"),
    ("并", "並"),
    ("买", "買"),
    ("卖", "賣"),
]


def _sample_text(markdown_text: str, fallback_text: str = "") -> str:
    sample = (markdown_text or "").strip()
    if not sample:
        sample = (fallback_text or "").strip()
    return sample[:4000]


def detect_prompt_lang(markdown_text: str, fallback_text: str = "") -> PromptLang:
    """
    Return prompt language for the LLM:
      - "en" for English-like docs
      - "zh" for Simplified Chinese
      - "zh-Hant" for Traditional Chinese (heuristic)
    """
    sample = _sample_text(markdown_text, fallback_text)
    if not sample:
        return "en"

    cjk = sum(1 for ch in sample if _CJK_RANGE[0] <= ch <= _CJK_RANGE[1])
    # Keep this threshold fairly low so short Chinese PDFs still get zh prompts.
    if cjk < 12 or (cjk / max(1, len(sample))) < 0.01:
        return "en"

    simp = 0
    trad = 0
    for s, t in _VARIANT_PAIRS:
        simp += sample.count(s)
        trad += sample.count(t)

    # If we observe clear Traditional signals, return zh-Hant.
    # Otherwise default to zh.
    if trad >= 3 and trad >= (simp + 2):
        return "zh-Hant"
    return "zh"


def coerce_prompt_lang(value: Optional[str]) -> PromptLang:
    v = str(value or "").strip()
    if v in {"en", "zh", "zh-Hant"}:
        return v  # type: ignore[return-value]
    return "en"
