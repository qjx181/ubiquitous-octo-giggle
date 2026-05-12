"""测试 safety.py：引用校验和来源强制"""
import pytest

from services.safety import (
    extract_citation_indices,
    validate_citations,
    verify_and_enforce_sources,
)


# ============ extract_citation_indices ============

class TestExtractCitationIndices:
    def test_normal_citation(self):
        """[联网资料1] 格式应提取出 1"""
        assert extract_citation_indices("根据[联网资料1]的记载") == [1]

    def test_multiple_citations(self):
        """多个引用编号应全部提取"""
        ans = "根据[联网资料1]和[来源2]以及[联网资料3]"
        assert extract_citation_indices(ans) == [1, 2, 3]

    def test_no_citations(self):
        """没有引用返回空列表"""
        assert extract_citation_indices("没有引用的回答") == []

    def test_empty_string(self):
        assert extract_citation_indices("") == []

    def test_none_input(self):
        assert extract_citation_indices(None) == []

    def test_garbled_citation(self):
        """缺少闭合括号 ] 的引用格式应不匹配"""
        ans = "根据[联网资料和[来源2"
        assert extract_citation_indices(ans) == []
        # 正则要求 \] 闭合，[来源2 没有 ] 所以不匹配，这是正确的行为


# ============ validate_citations ============

class TestValidateCitations:
    def test_valid_citations(self):
        """所有引用编号都在 refs 范围内"""
        refs = [{"rank": 1}, {"rank": 2}, {"rank": 3}]
        answer = "根据[联网资料1]和[来源3]"
        assert validate_citations(answer, refs) == []

    def test_invalid_citation(self):
        """引用了不存在的编号"""
        refs = [{"rank": 1}]
        answer = "根据[来源5]"
        bad = validate_citations(answer, refs)
        assert bad == [5]

    def test_zero_index(self):
        """引用编号为 0 应视为无效"""
        refs = [{"rank": 1}]
        answer = "根据[联网资料0]"
        bad = validate_citations(answer, refs)
        assert bad == [0]

    def test_empty_refs(self):
        """没有 refs 时任何引用都是无效的"""
        answer = "根据[联网资料1]"
        assert validate_citations(answer, []) == [1]


# ============ verify_and_enforce_sources ============

class TestVerifyAndEnforceSources:
    def test_no_context_returns_answer(self):
        """没有上下文时直接返回原答案"""
        result = verify_and_enforce_sources("你好", "", "", [])
        assert result == "你好"

    def test_answer_with_citation(self):
        """有引用标记的回答直接返回"""
        result = verify_and_enforce_sources(
            "根据[联网资料1]的规定",
            "民法典相关法条",
            "",
            [{"rank": 1}],
        )
        assert "根据[联网资料1]" in result

    def test_answer_without_citation_returns_cleaned(self):
        """有上下文但没有引用的回答，直接返回清洗后的内容"""
        result = verify_and_enforce_sources(
            "  回答内容  ",
            "民法典知识库内容",
            "",
            [{"rank": 1}],
        )
        assert result == "回答内容"

    def test_empty_answer_with_context(self):
        """空回答但有上下文时返回默认提示"""
        result = verify_and_enforce_sources(
            "",
            "民法典知识库",
            "",
            [{"rank": 1}],
        )
        assert "无法" in result

    def test_whitespace_answer_with_context(self):
        """仅有空格的回答"""
        result = verify_and_enforce_sources(
            "   ",
            "民法典知识库",
            "",
            [{"rank": 1}],
        )
        assert "无法" in result

    def test_exception_returns_original(self):
        """异常发生时返回原始答案"""
        # None 作为 refs 可能导致异常
        result = verify_and_enforce_sources("测试回答", "上下文", "", None)
        assert result == "测试回答"
