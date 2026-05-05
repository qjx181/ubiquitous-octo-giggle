"""测试 test_llm_client.py - 修复中文字符bytes语法问题"""
import pytest
from unittest.mock import patch, MagicMock

import requests
from llm_client import call_llm, stream_llm


class TestCallLLM:
    def test_call_llm_success(self):
        """正常 Ollama 响应"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "这是回答"}}
        mock_resp.status_code = 200

        with patch("llm_client.requests.post", return_value=mock_resp):
            result = call_llm([{"role": "user", "content": "你好"}])
            assert result == "这是回答"

    def test_call_llm_empty_content_raises(self):
        """空内容应抛出异常"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": ""}}
        mock_resp.status_code = 200

        with patch("llm_client.requests.post", return_value=mock_resp):
            with pytest.raises(RuntimeError) as exc:
                call_llm([{"role": "user", "content": "你好"}])
            assert "Ollama" in str(exc.value)

    def test_call_llm_missing_message_key(self):
        """缺少 message 键应抛出异常"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "something wrong"}
        mock_resp.status_code = 200

        with patch("llm_client.requests.post", return_value=mock_resp):
            with pytest.raises(RuntimeError):
                call_llm([{"role": "user", "content": "你好"}])

    def test_call_llm_network_error(self):
        """网络错误应抛出异常"""
        with patch("llm_client.requests.post", side_effect=requests.exceptions.ConnectionError("连接超时")):
            with pytest.raises(RuntimeError) as exc:
                call_llm([{"role": "user", "content": "你好"}])
            assert "Ollama" in str(exc.value)


class TestStreamLLM:
    def test_stream_llm_success(self):
        """流式响应解析 - 使用纯ascii bytes避免中文字符的问题"""
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.iter_lines.return_value = [
            b'{"message":{"content":"chunk1"}}',
            b'{"message":{"content":"chunk2"}}',
            b'{"message":{"content":"chunk3"},"done":true}',
        ]

        with patch("llm_client.requests.post", return_value=mock_resp):
            chunks = list(stream_llm([{"role": "user", "content": "hello"}]))
            assert chunks == ["chunk1", "chunk2", "chunk3"]

    def test_stream_llm_handles_data_prefix(self):
        """处理 SSE data: 前缀"""
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.iter_lines.return_value = [
            b"data: {\"message\":{\"content\":\"answer text\"}}",
        ]

        with patch("llm_client.requests.post", return_value=mock_resp):
            chunks = list(stream_llm([{"role": "user", "content": "你好"}]))
            assert chunks == ["answer text"]

    def test_stream_llm_handles_string_input(self):
        """处理 str 类型行（非 bytes）"""
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.iter_lines.return_value = [
            'data: {"message":{"content":"answer1"}}',
            '{"message":{"content":"done"},"done":true}',
        ]

        with patch("llm_client.requests.post", return_value=mock_resp):
            chunks = list(stream_llm([{"role": "user", "content": "你好"}]))
            assert chunks == ["answer1", "done"]

    def test_stream_llm_errors(self):
        """网络错误时抛出 RuntimeError"""
        mock_resp = MagicMock()
        mock_resp.__enter__.side_effect = requests.exceptions.ConnectionError("连接失败")

        with patch("llm_client.requests.post", return_value=mock_resp):
            with pytest.raises(RuntimeError) as exc:
                list(stream_llm([{"role": "user", "content": "你好"}]))
            assert "Ollama" in str(exc.value)

    def test_stream_llm_skip_invalid_json(self):
        """非 JSON 行应跳过"""
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.iter_lines.return_value = [
            b"not json at all",
            b'{"message":{"content":"valid response"},"done":true}',
        ]

        with patch("llm_client.requests.post", return_value=mock_resp):
            chunks = list(stream_llm([{"role": "user", "content": "你好"}]))
            assert chunks == ["valid response"]
