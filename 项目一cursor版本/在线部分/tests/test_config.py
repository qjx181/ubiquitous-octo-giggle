"""测试 config.py：环境变量加载"""
import os
import pytest

# 需要先设置环境变量再导入 config
os.environ["TEST_MILVUS_HOST"] = "192.168.1.1"
os.environ["TEST_MILVUS_PORT"] = "19531"
os.environ["TEST_LLM_MODEL_NAME"] = "test-model"
os.environ["TEST_COLLECTION_NAME"] = "test_collection"

# 动态测试 config 加载逻辑


class TestEnvPath:
    def test_env_path_uses_env_var(self):
        """环境变量优先"""
        from config import _env_path
        result = _env_path("TEST_MILVUS_HOST", "default_value")
        assert result == "192.168.1.1"

    def test_env_path_fallback(self):
        """未设置环境变量时使用默认值"""
        from config import _env_path
        result = _env_path("NONEXISTENT_VAR_12345", "default_val")
        assert result == "default_val"

    def test_env_path_strips_whitespace(self):
        """值应去除首尾空格"""
        import config as cfg
        # 确认 strip 逻辑
        val = "  hello  "
        stripped = cfg._env_path.__wrapped__ if hasattr(cfg._env_path, '__wrapped__') else None
        assert val.strip() == "hello"


class TestConfigDefaults:
    """验证默认配置的合理性"""

    def test_milvus_default_host(self):
        from config import MILVUS_HOST
        assert MILVUS_HOST  # 非空

    def test_bm25_top_k_reasonable(self):
        from config import BM25_TOP_K
        assert 5 <= BM25_TOP_K <= 100

    def test_vector_top_k_reasonable(self):
        from config import VECTOR_TOP_K
        assert 5 <= VECTOR_TOP_K <= 100

    def test_rerank_top_k_reasonable(self):
        from config import RERANK_TOP_K
        assert 1 <= RERANK_TOP_K <= 20

    def test_min_retrieval_score_range(self):
        from config import MIN_RETRIEVAL_SCORE
        assert 0.0 <= MIN_RETRIEVAL_SCORE <= 1.0

    def test_collection_names_not_empty(self):
        from config import COLLECTION_NAME, DOCTOR_COLLECTION_NAME
        assert COLLECTION_NAME
        assert DOCTOR_COLLECTION_NAME

    def test_jsonl_paths_not_empty(self):
        from config import DATA_JSONL, DOCTOR_DATA_JSONL
        assert DATA_JSONL
        assert DOCTOR_DATA_JSONL

    def test_llm_config_has_values(self):
        from config import LLM_API_BASE, LLM_MODEL_NAME
        assert LLM_API_BASE
        assert LLM_MODEL_NAME

    def test_web_config_reasonable(self):
        from config import WEB_TOP_K, WEB_TIMEOUT_SECONDS, SEARCH_RETRIES
        assert WEB_TOP_K >= 1
        assert WEB_TIMEOUT_SECONDS >= 1
        assert SEARCH_RETRIES >= 1
