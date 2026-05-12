"""
backend/app/core/llm_client.py — LLM客户端（单一职责）
=======================================================
<<<<<<< HEAD
职责：调用LLM服务生成答案

支持的LLM提供商：
  1. SGLang    — 本地部署，高性能推理引擎
  2. Ollama    — 本地部署，简单易用
  3. 第三方API — 硅基流动、DeepSeek、OpenAI等

功能：
  1. 统一接口适配多种LLM提供商
  2. 支持多轮对话上下文
  3. 失败自动重试（指数退避）
  4. 流式输出支持

使用方式：
  通过环境变量 LLM_PROVIDER 切换提供商：
    set LLM_PROVIDER=ollama     # 使用Ollama
    set LLM_PROVIDER=sglang     # 使用SGLang
    set LLM_PROVIDER=siliconflow # 使用硅基流动
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional, Generator

import requests

from ..config import settings
=======
作用：封装调用各种LLM服务的通用接口
原理：
  1. 统一接口适配多种LLM提供商（SGLang/Ollama/第三方API）
  2. 支持非流式和流式两种调用方式
  3. 失败自动重试（指数退避，逐步增加等待时间）
  4. 根据provider不同自动选择API格式和响应解析方式
"""

import json     # 作用：解析JSON格式的API响应
import logging  # 作用：记录日志
import time     # 作用：重试时等待一段时间再试
from typing import List, Dict, Any, Optional, Generator  # 作用：类型注解

import requests  # 作用：发送HTTP请求调用LLM

from ..config import settings  # 作用：读取配置
>>>>>>> 6d37b33 (提交信息)

logger = logging.getLogger(__name__)


class LLMClient:
    """
<<<<<<< HEAD
    LLM客户端

    统一封装多种LLM提供商的API调用

=======
    作用：LLM客户端，统一封装多种LLM提供商的API调用
    原理：根据self.provider的值，自动选择对应的API地址、请求格式、响应解析方式
>>>>>>> 6d37b33 (提交信息)
    使用示例：
        client = LLMClient()
        answer = client.generate("公司的营业收入是多少？", context=["...", "..."])
    """

    def __init__(self):
        """
<<<<<<< HEAD
        初始化LLM客户端

        根据配置自动选择LLM提供商
=======
        作用：初始化LLM客户端
        原理：从config读取LLM_PROVIDER配置（如"ollama"或"sglang"），
              根据不同的provider设置不同的host、model、API路径
        逻辑：
          1. 读取LLM_PROVIDER，决定用哪个引擎
          2. 不同的provider有不同的地址和模型名
          3. 设置对应的API URL（/v1/chat/completions 或 /api/chat）
>>>>>>> 6d37b33 (提交信息)
        """
        self.provider = settings.LLM_PROVIDER.lower()
        self.timeout = settings.LLM_TIMEOUT
        self.max_retries = settings.LLM_MAX_RETRIES

<<<<<<< HEAD
        # 根据提供商初始化配置
        if self.provider == "sglang":
            self.host = settings.SGLANG_HOST
            # 作用：使用实际模型名而非 "default"，避免 400 Bad Request
            # 原理：SGLang OpenAPI 兼容接口要求 model 参数与服务端实际加载的模型一致
            # 否则部分 SGLang 版本会返回 400
=======
        if self.provider == "sglang":
            self.host = settings.SGLANG_HOST
>>>>>>> 6d37b33 (提交信息)
            self.model = settings.SGLANG_MODEL
            self.api_key = ""
            self.chat_url = f"{self.host}/v1/chat/completions"

        elif self.provider == "ollama":
            self.host = settings.OLLAMA_HOST
            self.model = settings.OLLAMA_MODEL
            self.api_key = ""
<<<<<<< HEAD
            # Ollama 的 chat API 路径
=======
>>>>>>> 6d37b33 (提交信息)
            self.chat_url = f"{self.host}/api/chat"

        elif self.provider in ("siliconflow", "deepseek", "openai"):
            self.host = settings.API_HOST
            self.model = settings.API_MODEL
            self.api_key = settings.API_KEY
            self.chat_url = f"{self.host}/v1/chat/completions"

        else:
            raise ValueError(f"不支持的LLM提供商: {self.provider}")

        logger.info(f"LLM客户端初始化完成 | 提供商: {self.provider} | 模型: {self.model} | 地址: {self.host}")

    def generate(
        self,
        query: str,
        context: List[str],
        history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False
    ) -> str:
        """
<<<<<<< HEAD
        生成答案

        参数：
            query: 用户问题
            context: 检索到的相关文本（作为RAG上下文）
            history: 对话历史（多轮对话）
            stream: 是否流式输出

        返回：
            生成的答案文本

        异常：
            RuntimeError: 生成失败或超时
        """
        # 构建消息列表
        messages = self._build_messages(query, context, history)

        # 根据提供商调用对应的生成方法
=======
        作用：调用LLM生成答案（非流式，一次性返回完整答案）
        参数：
            query: 用户问题
            context: 检索到的相关文本列表
            history: 对话历史
            stream: 是否流式（默认False，一次性返回）
        返回：
            生成的答案文本
        """
        messages = self._build_messages(query, context, history)

>>>>>>> 6d37b33 (提交信息)
        if self.provider == "ollama":
            return self._generate_ollama(messages, stream)
        else:
            return self._generate_openai_compatible(messages, stream)

    def generate_stream(
        self,
        query: str,
        context: List[str],
        history: Optional[List[Dict[str, str]]] = None
    ) -> Generator[str, None, None]:
        """
<<<<<<< HEAD
        流式生成答案

        参数：
            query: 用户问题
            context: 检索到的相关文本
            history: 对话历史

        返回：
            生成器，每次yield一个token
=======
        作用：流式生成答案，每次yield一个token
        参数：
            query: 用户问题
            context: 检索到的文本
            history: 对话历史
        返回：
            生成器，每次yield一个token字符串
>>>>>>> 6d37b33 (提交信息)
        """
        messages = self._build_messages(query, context, history)

        if self.provider == "ollama":
            yield from self._generate_stream_ollama(messages)
        else:
            yield from self._generate_stream_openai(messages)

    def _generate_openai_compatible(self, messages: List[Dict[str, str]], stream: bool = False) -> str:
        """
<<<<<<< HEAD
        调用OpenAI兼容API生成答案

        适用：SGLang、硅基流动、DeepSeek、OpenAI
=======
        作用：调用OpenAI兼容API（SGLang、硅基流动、DeepSeek、OpenAI等），非流式
        原理：构建符合OpenAI格式的请求体 → POST请求 → 解析响应
        重试：指数退避 1s→2s→4s
>>>>>>> 6d37b33 (提交信息)
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "stream": stream,
        }

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(self.max_retries):
            try:
                logger.info(f"调用LLM [{self.provider}] (尝试 {attempt + 1}/{self.max_retries})")

                response = requests.post(
                    self.chat_url,
                    json=payload,
                    headers=headers if headers else None,
                    timeout=self.timeout
                )
                response.raise_for_status()

                result = response.json()
                answer = result["choices"][0]["message"]["content"]

                logger.info("LLM调用成功")
                return answer

            except requests.exceptions.Timeout:
                logger.warning(f"LLM调用超时 (尝试 {attempt + 1})")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError("LLM调用超时")

            except requests.exceptions.ConnectionError:
                logger.warning(f"LLM连接失败 (尝试 {attempt + 1})")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError("LLM连接失败，请检查服务是否启动")

            except Exception as e:
                logger.error(f"LLM调用失败: {e}")
<<<<<<< HEAD
=======
                logger.error(f"请求URL: {self.chat_url}")
                logger.error(f"请求体: {json.dumps(payload, ensure_ascii=False, indent=2)[:500]}")
                if 'response' in locals():
                    logger.error(f"响应状态: {response.status_code}")
                    logger.error(f"响应内容: {response.text[:500]}")
>>>>>>> 6d37b33 (提交信息)
                raise RuntimeError(f"LLM调用失败: {e}")

        raise RuntimeError("LLM调用失败，已达到最大重试次数")

    def _generate_ollama(self, messages: List[Dict[str, str]], stream: bool = False) -> str:
        """
<<<<<<< HEAD
        调用Ollama API生成答案

        Ollama的API格式与OpenAI略有不同
        """
        # Ollama 使用 "content" 而不是 "messages" 中的标准格式
        # 但新版 Ollama 也支持 OpenAI 兼容格式
=======
        作用：调用Ollama API生成答案（非流式）
        原理：Ollama的API格式和OpenAI略有不同
        重试：指数退避 1s→2s→4s
        """
>>>>>>> 6d37b33 (提交信息)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": settings.LLM_TEMPERATURE,
                "num_predict": settings.LLM_MAX_TOKENS,
            }
        }

        headers = {"Content-Type": "application/json"}

        for attempt in range(self.max_retries):
            try:
                logger.info(f"调用LLM [ollama] (尝试 {attempt + 1}/{self.max_retries})")

                response = requests.post(
                    self.chat_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                response.raise_for_status()

                result = response.json()
<<<<<<< HEAD
                # Ollama 返回格式: {"message": {"role": "assistant", "content": "..."}}
=======
>>>>>>> 6d37b33 (提交信息)
                answer = result["message"]["content"]

                logger.info("LLM调用成功")
                return answer

            except requests.exceptions.Timeout:
                logger.warning(f"LLM调用超时 (尝试 {attempt + 1})")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError("LLM调用超时")

            except requests.exceptions.ConnectionError:
                logger.warning(f"LLM连接失败 (尝试 {attempt + 1})")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(
                    "Ollama连接失败，请检查:\n"
                    "1. Ollama是否已安装: https://ollama.com\n"
                    "2. Ollama服务是否已启动: ollama serve\n"
                    "3. 模型是否已下载: ollama pull qwen2.5:7b"
                )

            except Exception as e:
                logger.error(f"LLM调用失败: {e}")
                raise RuntimeError(f"LLM调用失败: {e}")

        raise RuntimeError("LLM调用失败，已达到最大重试次数")

    def _generate_stream_openai(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """
<<<<<<< HEAD
        OpenAI兼容流式生成
=======
        作用：OpenAI兼容API的流式生成
        原理：stream=True时，服务端一行一行返回SSE格式数据
>>>>>>> 6d37b33 (提交信息)
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "stream": True,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                self.chat_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"]
                            if "content" in delta:
                                yield delta["content"]
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            raise RuntimeError(f"流式生成失败: {e}")

    def _generate_stream_ollama(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """
<<<<<<< HEAD
        Ollama流式生成
=======
        作用：Ollama流式生成
        原理：Ollama流式每行一个完整JSON，包含message.content和done字段
>>>>>>> 6d37b33 (提交信息)
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": settings.LLM_TEMPERATURE,
                "num_predict": settings.LLM_MAX_TOKENS,
            }
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(
                self.chat_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        if "message" in chunk and "content" in chunk["message"]:
                            yield chunk["message"]["content"]
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            raise RuntimeError(f"流式生成失败: {e}")

    def _build_messages(
        self,
        query: str,
        context: List[str],
        history: Optional[List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """
<<<<<<< HEAD
        构建消息列表

        结构：
          1. 系统提示（角色定义 + 回答规范）
          2. 上下文（检索到的相关文本）
          3. 对话历史（多轮对话）
          4. 用户问题
        """
        messages = []

        # 系统提示
        system_prompt = self._build_system_prompt()
        messages.append({"role": "system", "content": system_prompt})

        # 上下文（RAG）
        if context:
            context_text = "\n\n".join(context)
            messages.append({
                "role": "system",
                "content": f"以下是与问题相关的参考信息：\n\n{context_text}"
            })

        # 对话历史
        if history:
            messages.extend(history)

        # 用户问题
=======
        作用：构建LLM调用的消息列表
        结构：系统提示 + 检索上下文 + 对话历史 + 用户问题
        """
        messages = []
        system_prompt = self._build_system_prompt()

        if context:
            context_text = "\n\n".join(context)
            system_prompt += f"\n\n以下是与问题相关的参考信息：\n\n{context_text}"

        messages.append({"role": "system", "content": system_prompt})

        if history:
            messages.extend(history)

>>>>>>> 6d37b33 (提交信息)
        messages.append({"role": "user", "content": query})

        return messages

    def _build_system_prompt(self) -> str:
        """
<<<<<<< HEAD
        构建系统提示

        定义AI助手的角色和回答规范
=======
        作用：构建系统提示词，定义AI助手的行为规范
>>>>>>> 6d37b33 (提交信息)
        """
        return """你是一个专业的招股说明书分析助手。你的任务是：

1. 基于提供的参考信息回答问题
2. 如果参考信息不足，明确告知用户
3. 回答要简洁、准确、专业
4. 涉及财务数据时，注意单位（万元、亿元等）
5. 表格数据要准确引用
6. 不确定的内容不要编造

回答格式：
- 直接给出答案
- 必要时引用来源（页码）
- 复杂问题分点说明"""


<<<<<<< HEAD
# 全局客户端实例（单例模式）
=======
>>>>>>> 6d37b33 (提交信息)
_llm_client = None


def get_llm_client() -> LLMClient:
<<<<<<< HEAD
    """
    获取LLM客户端单例

    返回：
        LLMClient实例
    """
=======
>>>>>>> 6d37b33 (提交信息)
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
