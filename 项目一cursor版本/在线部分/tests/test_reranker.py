"""测试 reranker.py：重排序逻辑（mock sentence_transformers）"""
import sys
from unittest.mock import MagicMock, patch

# 在 import reranker 之前 mock sentence_transformers
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["sentence_transformers"].CrossEncoder = MagicMock()

import pytest

from reranker import build_candidate_text


class TestBuildCandidateText:
    def test_with_title_and_source(self):
        """正常情况拼接 source/title/text"""
        item = {
            "title": "民法典合同编",
            "source_file": "law.pdf",
            "content": "合同相关法律条文",
        }
        result = build_candidate_text(item)
        assert "law.pdf" in result
        assert "民法典合同编" in result
        assert "合同相关法律条文" in result

    def test_without_title(self):
        """没有 title 时有 article_no"""
        item = {
            "article_no": "第120条",
            "path": "/data/law.txt",
            "text": "内容",
        }
        result = build_candidate_text(item)
        assert "第120条" in result
        assert "/data/law.txt" in result
        assert "内容" in result

    def test_minimal_item(self):
        """只有 content 的情况"""
        item = {"content": "仅有的内容"}
        result = build_candidate_text(item)
        assert result == "仅有的内容"

    def test_empty_item(self):
        """空字典返回空字符串"""
        assert build_candidate_text({}) == ""


class TestRerankCandidates:
    """rerank_candidates 需要加载实际模型才能完整测试"""

    def test_input_format(self):
        """验证 rerank_candidates 期望的输入格式"""
        sample = (
            {"id": "1", "title": "测试", "content": "内容", "page": "1", "source_file": "test.txt"},
            0.95,
            "semantic",
        )
        # build_candidate_text 能处理这个格式的 item
        text = build_candidate_text(sample[0])
        assert "内容" in text
