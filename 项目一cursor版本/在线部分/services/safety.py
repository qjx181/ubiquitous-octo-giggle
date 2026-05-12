import logging
import re

logger = logging.getLogger(__name__)


def extract_citation_indices(answer: str) -> list[int]:
    ids = []
    try:
        for m in re.finditer(r"\[(?:联网资料|来源)(\d+)\]", answer):
            ids.append(int(m.group(1)))
    except Exception:
        logger.exception("extract citation indices failed")
    return ids


def validate_citations(answer: str, refs: list) -> list[int]:
    bad = []
    try:
        cited = extract_citation_indices(answer)
        if not cited:
            return []
        for idx in cited:
            if idx <= 0 or idx > len(refs):
                bad.append(idx)
    except Exception:
        logger.exception("validate citations failed")
        return []
    return bad


def verify_and_enforce_sources(answer: str, rag_context: str, web_context: str, refs: list | None = None) -> str:
    try:
        has_context = bool(rag_context or web_context or (refs and len(refs) > 0))
        if not has_context:
            return answer
        cleaned = (answer or "").strip()
        if not cleaned:
            return (
                "无法从当前检索到的资料中直接给出确定结论。\n"
                "请根据检索到的资料综合判断。"
            )
        # 只做最小化校验，不向前端追加任何检索摘要或来源详情
        if re.search(r"\[(?:联网资料|来源)\d+\]", cleaned):
            return cleaned
        return cleaned
    except Exception:
        logger.exception("verify and enforce sources failed")
        return answer
