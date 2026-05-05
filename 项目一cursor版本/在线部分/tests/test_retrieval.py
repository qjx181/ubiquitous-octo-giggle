"""测试 retrieval.py：核心检索逻辑（mock 所有外部依赖在 import 之前）

注意：这个文件在 sys.modules 中 mock pymilvus 和 sentence_transformers 等
第三方包，但不 mock services 包本身（避免干扰其他测试文件）。
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ===== 在 import 任何项目模块前，先 mock 外部第三方包 =====
# 不 mock services 包（保持真实），只 mock 第三方包
sys.modules["pymilvus"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["rank_bm25"] = MagicMock()
sys.modules["rank_bm25"].BM25Okapi = MagicMock()

# reranker 需要在 import 前 mock，因为 retrieval 会 from reranker import ...
reranker_module = MagicMock()
reranker_module.load_reranker = MagicMock(return_value=MagicMock())
reranker_module.rerank_candidates = MagicMock(return_value=[])
reranker_module.build_candidate_text = MagicMock(return_value="")
sys.modules["reranker"] = reranker_module

# web_fallback 也需要 mock
wf_mock = MagicMock()
wf_mock.is_professional_query = MagicMock(return_value=False)
wf_mock.crawl_web_knowledge = MagicMock(return_value=[])
wf_mock.web_collection_name = MagicMock(return_value="web_memory_general")
sys.modules["web_fallback"] = wf_mock

# data_loader mock
dl_mock = MagicMock()
dl_mock.load_jsonl = MagicMock(return_value=[])
dl_mock.build_bm25_index = MagicMock(return_value=None)
dl_mock.tokenize = MagicMock(return_value=["test"])
dl_mock.filter_bm25_items = MagicMock(return_value=[])
sys.modules["data_loader"] = dl_mock

# 注意：不要 mock 'services' 或 'services.knowledge_store' 在 sys.modules 级别！
# 让 Python 正常解析 services 包路径。

# ===== 现在导入待测模块 =====
# pytest 的 conftest.py 已经将 在线部分/ 添加到 sys.path
from services.retrieval import (
    get_embedding_model,
    get_reranker_model,
    get_collection,
    cached_encode,
    cached_semantic_search,
    cached_vector_search,
    cached_web_search,
    cached_web_milvus_search,
    get_jsonl_items,
    get_bm25_index,
    get_bm25_items,
    get_query_jsonl_path,
    get_query_collection_name,
    get_query_items,
    is_greeting_or_smalltalk,
    is_knowledge_query,
    should_use_local_knowledge,
    is_web_query,
    build_context,
    _normalize_text,
    _item_dedupe_key,
)


class TestGreetingDetection:
    def test_typical_greetings(self):
        for greeting in ["你好", "您好", "hello", "hi", "嗨", "早上好", "谢谢"]:
            assert is_greeting_or_smalltalk(greeting), f"{greeting} 应被识别为问候"

    def test_long_query_not_greeting(self):
        """超过 12 字符的不是问候"""
        assert not is_greeting_or_smalltalk("民法典关于合同的规定有哪些")
        assert not is_greeting_or_smalltalk("高血压患者饮食注意事项")

    def test_empty_query(self):
        assert is_greeting_or_smalltalk("")
        assert is_greeting_or_smalltalk(None)

    def test_greeting_longer_than_12(self):
        """超过12个字符的短句才不算问候"""
        # "你好" + 标点 + 更多文字超过12字符就不算问候了
        assert not is_greeting_or_smalltalk("你好，民法典合同的规定有哪些")
        assert not is_greeting_or_smalltalk("你好医生我血压有点高怎么办")


class TestKnowledgeQuery:
    def test_medical_keywords_for_doctor(self):
        """
        BUG 检测: is_knowledge_query 在检查医学关键词之前，
        先检查 JSONL 文件是否存在。如果文件不存在，直接返回 False。
        这意味着即使查询明显是医学问题，也会返回 False。
        """
        with patch("pathlib.Path.exists", return_value=True):
            assert is_knowledge_query("高血压的症状", "医生")
            assert is_knowledge_query("收缩压正常范围", "医生")
            assert not is_knowledge_query("今天天气怎么样", "医生")

    def test_lawyer_delegates_to_web_fallback(self):
        # web_fallback 被 mock 为返回 False
        result = is_knowledge_query("合同纠纷", "律师")
        assert result is False

    def test_greeting_is_not_knowledge(self):
        assert not is_knowledge_query("你好", "医生")


class TestNormalizeAndDedupe:
    def test_normalize_text(self):
        assert _normalize_text("  hello   world  ") == "hello world"
        assert _normalize_text(None) == ""
        assert _normalize_text("") == ""

    def test_dedupe_key_consistency(self):
        item1 = {"content": "测试内容", "title": "标题", "source_file": "文件"}
        item2 = {"content": "测试内容", "title": "标题", "source_file": "文件"}
        assert _item_dedupe_key(item1) == _item_dedupe_key(item2)

    def test_dedupe_key_different(self):
        item1 = {"content": "内容A", "title": "标题", "source_file": "文件"}
        item2 = {"content": "内容B", "title": "标题", "source_file": "文件"}
        assert _item_dedupe_key(item1) != _item_dedupe_key(item2)

    def test_dedupe_key_fallback_fields(self):
        """没有 content 时用 text 字段"""
        item = {"text": "正文", "title": "标题", "path": "path/to/file"}
        key = _item_dedupe_key(item)
        assert "正文" in key


class TestQueryLookups:
    def test_get_query_jsonl_path_lawyer(self):
        path = get_query_jsonl_path("律师")
        assert "民法典" in path

    def test_get_query_jsonl_path_doctor(self):
        path = get_query_jsonl_path("医生")
        assert "高血压" in path

    def test_get_query_jsonl_path_unknown(self):
        assert get_query_jsonl_path("未知角色") == ""

    def test_get_query_collection_name_lawyer(self):
        assert get_query_collection_name("律师") == "civil_code_rag"

    def test_get_query_collection_name_doctor(self):
        assert get_query_collection_name("医生") == "hypertension_rag_chunks"

    def test_get_query_collection_name_unknown(self):
        assert get_query_collection_name("未知") == ""


class TestBuildContext:
    """build_context 是检索核心入口"""

    def test_greeting_returns_empty(self):
        context, refs, seconds, total, bm25_hits, vec_hits, score = build_context("你好", "律师")
        assert context == ""
        assert refs == []

    def test_unknown_role_returns_empty(self):
        context, refs, seconds, total, bm25_hits, vec_hits, score = build_context("民法典合同", "未知角色")
        assert context == ""
        assert refs == []

    def test_empty_query(self):
        context, refs, seconds, total, bm25_hits, vec_hits, score = build_context("", "律师")
        assert context == ""

    def test_none_query(self):
        context, refs, seconds, total, bm25_hits, vec_hits, score = build_context(None, "律师")
        assert context == ""


class TestIsWebQuery:
    def test_greeting_not_web_query(self):
        assert not is_web_query("你好", "律师")

    def test_delegates_to_professional_check(self):
        # web_fallback mock 返回 False
        assert not is_web_query("高血压", "医生")


class TestCacheBehavior:
    def test_get_embedding_model_cached(self):
        m1 = get_embedding_model()
        m2 = get_embedding_model()
        assert m1 is m2

    def test_get_query_jsonl_path_cached(self):
        p1 = get_query_jsonl_path("律师")
        p2 = get_query_jsonl_path("律师")
        assert p1 == p2
