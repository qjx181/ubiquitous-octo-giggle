"""
backend/app/core/llm_client.py — LLM客户端（异步版）
======================================================
【作用】封装调用各种LLM服务的统一接口（async/await）
【原理】
  1. 使用 httpx.AsyncClient 实现真正的异步I/O，不阻塞事件循环
  2. 非流式和流式都支持 async
  3. 失败自动重试（指数退避），超时/连接错误才重试，其他错误直接抛
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator

import httpx

from ..config import settings

# 【作用】创建一个 logger 对象，专门记录当前模块（llm_client.py）的日志
logger = logging.getLogger(__name__)


class LLMClient:
    """
    【作用】LLM客户端，统一封装多种LLM提供商的异步API调用

    【API返回类型和返回值结构】
      generate():          返回 str（LLM生成的完整答案文本）
      generate_stream():   返回 AsyncGenerator[str, None]（逐token异步生成器）
      _build_messages():   返回 List[Dict]（[{role, content}, ...] 格式的对话消息列表）

    【设计原理】
      1. 支持 4 种 provider：sglang、ollama、siliconflow/deepseek/openai
      2. 两个 API 协议分支：
         - OpenAI 兼容协议（sglang/siliconflow/deepseek/openai）：{host}/v1/chat/completions
         - Ollama 原生协议：{host}/api/chat
      3. provider 在 __init__ 中通过 settings.LLM_PROVIDER 决定
      4. 注意：当被 LLMRouter 通过 __new__ 创建时，不走 __init__
         所以这里的 init 逻辑只在"直接使用 LLMClient()"时生效

    【边界情况处理】
      - TimeoutException：指数退避重试，最多 max_retries 次
      - ConnectError：同样指数退避重试
      - 其他异常：直接抛 RuntimeError，不重试（因为重试也没用）
      - 流式中断：async for 读取异常时抛 RuntimeError

    【和类似方案对比】
      - 不直接用 requests：httpx.AsyncClient 支持真正的异步 I/O，不阻塞事件循环
      - 不需要 requests 的线程池方案：async/await 更简洁，资源开销更小
      - 不暴露底层 httpx 给调用方：统一用 generate/generate_stream 接口

    【面试官可能问】
      Q: 为什么用 httpx 而不是 aiohttp？
      A: httpx 的 AsyncClient 接口和 requests 高度一致（post、json、raise_for_status），
         迁移成本低。aiohttp 的 API 风格差异较大。

      Q: 4xx 错误为什么不重试？
      A: 4xx 是请求本身的问题（如 API Key 过期、模型名不存在），重试一万次一样报错。
         重试只应对瞬态故障（网络超时、连接重置），不应对永久故障。

      Q: 为什么流式接口没有重试？
      A: 流式如果中途断了，前面已经 yield 的 token 已经发出去了，重试会重复。
         由上层 LLMRouter 做引擎级故障转移（切换到备用引擎），比重试更合理。
    """

    def __init__(self):
        # 【作用】从全局配置中读取 LLM 提供商的名字，转小写便于后续判断
        self.provider = settings.LLM_PROVIDER.lower()
        # 【作用】读取 LLM 调用超时时间（秒），用于 httpx.AsyncClient 的 timeout 参数
        self.timeout = settings.LLM_TIMEOUT
        # 【作用】读取最大重试次数，默认为 3
        self.max_retries = settings.LLM_MAX_RETRIES

        # 【作用】根据不同的 provider 构建对应的 API 地址和认证信息
        # 【逻辑】sglang 和 ollama 是本地/内网部署，不需要 API Key
        if self.provider == "sglang":
            self.host = settings.SGLANG_HOST
            self.model = settings.SGLANG_MODEL
            self.api_key = ""
            # 【原理】SGLang 兼容 OpenAI 的 chat/completions 接口格式
            self.chat_url = f"{self.host}/v1/chat/completions"

        elif self.provider == "ollama":
            self.host = settings.OLLAMA_HOST
            self.model = settings.OLLAMA_MODEL
            self.api_key = ""
            # 【原理】Ollama 使用自己的 /api/chat 接口，不是 OpenAI 兼容格式
            self.chat_url = f"{self.host}/api/chat"

        elif self.provider in ("siliconflow", "deepseek", "openai"):
            self.host = settings.API_HOST
            self.model = settings.API_MODEL
            self.api_key = settings.API_KEY
            # 【原理】这些云服务都兼容 OpenAI 的 API 格式，所以用同样的 URL 路径
            self.chat_url = f"{self.host}/v1/chat/completions"

        else:
            # 【作用】如果配置文件里写了不支持的提供商名称，直接报错阻止启动
            # 【原理】尽早失败（fail-fast）比运行时才发现问题好
            raise ValueError(f"不支持的LLM提供商: {self.provider}")

        logger.info(f"LLM客户端初始化完成 | 提供商: {self.provider} | 模型: {self.model} | 地址: {self.host}")

    async def generate(
        self,
        query: str,
        context: List[str],
        history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False
    ) -> str:
        """async 非流式生成，内部 await HTTP 调用，不阻塞事件循环

        【函数做什么】
          接收用户问题 + 检索到的上下文 + 对话历史，调用 LLM 生成答案

        【参数】
          query:   用户的问题文本
          context: 检索到的参考文档片段列表（每个元素是一段文本）
          history: 多轮对话历史，格式 [{"role": "user"/"assistant", "content": "..."}]
          stream:  是否流式输出（非流式模式下始终为 False）

        【返回值】
          str: LLM 生成的答案文本

        【设计原理】
          1. 无论 provider 是哪个，都先调用 _build_messages 拼出统一格式的 messages
          2. 然后根据 provider 选择对应的生成方法
          3. 生成方法内部处理了重试逻辑

        【面试官可能问】
          Q: 为什么不把 stream 参数去掉，让调用方决定用哪个方法？
          A: generate() 和 generate_stream() 分开更清晰。
             如果合到一个方法，调用方需要处理 str 和 AsyncGenerator 两种返回类型，
             代码复杂度会增加。
        """
        messages = self._build_messages(query, context, history)

        if self.provider == "ollama":
            return await self._generate_ollama(messages, stream)
        else:
            return await self._generate_openai_compatible(messages, stream)

    async def generate_stream(
        self,
        query: str,
        context: List[str],
        history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        """async 流式生成，yield 每个 token

        【函数做什么】
          和 generate() 一样接收问题+上下文+历史，但逐 token 返回（打字机效果）

        【返回值】
          AsyncGenerator[str, None]: 异步生成器，每次 yield 一个 token 的文本

        【设计原理】
          1. 同样先拼 messages
          2. 然后根据 provider 选择对应的流式生成方法
          3. 流式方法使用 httpx 的 client.stream 保持长连接
          4. 逐行读取 SSE 事件流，每次 yield 一个 token

        【边界情况处理】
          - 流式中途断连：抛 RuntimeError，由上层 LLMRouter 做故障转移
          - 没有重试：流式场景重试会导致 token 重复

        【面试官可能问】
          Q: 为什么 generate_stream 不是 generate(stream=True) 的一个分支？
          A: Python 类型系统不支持"返回值可能是 str 也可能是 AsyncGenerator"的优雅写法。
             分开两个方法，调用方只看方法名就知道返回什么类型。
        """
        messages = self._build_messages(query, context, history)

        if self.provider == "ollama":
            async for token in self._generate_stream_ollama(messages):
                yield token
        else:
            async for token in self._generate_stream_openai(messages):
                yield token

    def _build_payload(self, stream: bool = False) -> dict:
        """构建API请求体（公共部分）

        【函数做什么】
          生成发给 LLM API 的请求体字典，包含模型名、参数等

        【参数】
          stream: 是否启用流式输出

        【返回值】
          dict: 请求体字典，包含 model、temperature、max_tokens 等字段

        【设计原理】
          1. Ollama 的请求体格式和 OpenAI 兼容格式不同
          2. 温度参数 temperature 控制输出的随机性（0=确定，1=随机）
          3. max_tokens 控制最大生成长度，防止无限生成

        【边界情况】
          - Ollama 用 options 嵌套结构传参
          - OpenAI 兼容协议直接在顶层传 temperature 和 max_tokens
        """
        if self.provider == "ollama":
            return {
                "model": self.model,
                "messages": [],
                "stream": stream,
                "options": {
                    "temperature": settings.LLM_TEMPERATURE,
                    "num_predict": settings.LLM_MAX_TOKENS,
                }
            }
        return {
            "model": self.model,
            "messages": [],
            "temperature": settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "stream": stream,
        }

    def _build_headers(self) -> dict:
        """构建HTTP请求头

        【函数做什么】
          生成 HTTP 请求头，如果需要鉴权则添加 Authorization 字段

        【返回值】
          dict: 请求头字典，可能包含 Authorization: Bearer xxx
        """
        headers = {}
        if self.api_key:
            # 【原理】OpenAI 兼容协议的鉴权格式是 Bearer token
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _generate_openai_compatible(self, messages: List[Dict], stream: bool = False) -> str:
        """async 调用 OpenAI 兼容 API

        【函数做什么】
          向 OpenAI 兼容的 API 发送 POST 请求，获取 LLM 生成的答案

        【参数】
          messages: 对话消息列表 [{"role": "system"/"user"/"assistant", "content": "..."}]
          stream:   是否流式（非流式模式下为 False）

        【返回值】
          str: LLM 生成的答案文本

        【设计原理 — 重试机制】
          1. 最大重试 3 次（由 settings.LLM_MAX_RETRIES 控制）
          2. 指数退避: 第 1 次重试等 1 秒，第 2 次等 2 秒，第 3 次等 4 秒
          3. 只对 TimeoutException 和 ConnectError 重试——它们是瞬态故障
          4. 其他异常（如 4xx、解析错误）直接抛——重试也没用

        【面试官可能问】
          Q: 指数退避为什么是 2 ** attempt 而不是 attempt ** 2？
          A: 指数退避（2^0=1s, 2^1=2s, 2^2=4s）是被广泛采用的标准策略。
             线性退避增长太慢（1,2,3），指数退避增长更快，能更快地拉开间隔，
             避免在服务恢复期间持续发送请求造成"惊群效应"。

          Q: raise_for_status() 把 4xx 和 5xx 都转成了异常，为什么 4xx 不单独处理？
          A: raise_for_status() 统一处理了所有 HTTP 错误码。4xx（如 401、404）会被
             最外层的 except Exception 捕获，直接抛给上层。5xx 也一样——服务端错误
             重试大概率是浪费时间。
        """
        payload = self._build_payload(stream)
        payload["messages"] = messages
        headers = self._build_headers()

        for attempt in range(self.max_retries):
            try:
                logger.info(f"调用LLM [{self.provider}] (尝试 {attempt + 1}/{self.max_retries})")
                # 【作用】创建异步 HTTP 客户端，设置超时时间
                # 【原理】async with 确保请求完成后自动释放连接资源
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    # 【作用】发送 POST 请求到 LLM API
                    response = await client.post(self.chat_url, json=payload, headers=headers)
                    # 【作用】如果状态码不是 2xx，会抛 httpx.HTTPStatusError
                    response.raise_for_status()
                    # 【作用】解析返回的 JSON 响应体
                    result = response.json()
                    # 【作用】从 OpenAI 兼容格式中提取答案文本
                    # 【格式】{"choices": [{"message": {"content": "答案文本"}}]}
                    answer = result["choices"][0]["message"]["content"]
                    logger.info("LLM调用成功")
                    return answer
            except httpx.TimeoutException:
                # 【作用】请求超时，可能是网络波动或 LLM 负载过高
                logger.warning(f"LLM调用超时 (尝试 {attempt + 1})")
                if attempt < self.max_retries - 1:
                    # 【原理】指数退避：第 1 次等 1s，第 2 次等 2s
                    await asyncio.sleep(2 ** attempt)
                    continue
                # 【作用】最后一次尝试也失败了，不再重试
                raise RuntimeError("LLM调用超时")
            except httpx.ConnectError:
                # 【作用】连接失败，可能是 LLM 服务未启动或网络不通
                logger.warning(f"LLM连接失败 (尝试 {attempt + 1})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise RuntimeError("LLM连接失败，请检查服务是否启动")
            except Exception as e:
                # 【作用】其他异常（4xx、解析错误等）直接抛，不重试
                # 【原理】4xx 是请求错误（如 API Key 失效），重试不会解决问题
                logger.error(f"LLM调用失败: {e}")
                raise RuntimeError(f"LLM调用失败: {e}")
        # 【作用】兜底：如果 for 循环因为某种原因没有 return 也没 raise，走这里
        raise RuntimeError("LLM调用失败，已达到最大重试次数")

    async def _generate_ollama(self, messages: List[Dict], stream: bool = False) -> str:
        """async 调用 Ollama API

        【函数做什么】
          和 _generate_openai_compatible 功能相同，但适配 Ollama 的 API 格式

        【返回值】
          str: LLM 生成的答案文本

        【设计原理】
          Ollama 的 API 和 OpenAI 兼容 API 有两个主要区别：
          1. URL 路径不同：/api/chat vs /v1/chat/completions
          2. 返回格式不同：result["message"]["content"] vs result["choices"][0]["message"]["content"]
          3. 参数结构不同：options.temperature vs 顶层 temperature
        """
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
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(self.chat_url, json=payload, headers=headers)
                    response.raise_for_status()
                    result = response.json()
                    # 【作用】Ollama 返回格式：{"message": {"content": "答案"}, ...}
                    answer = result["message"]["content"]
                    logger.info("LLM调用成功")
                    return answer
            except httpx.TimeoutException:
                logger.warning(f"LLM调用超时 (尝试 {attempt + 1})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise RuntimeError("LLM调用超时")
            except httpx.ConnectError:
                logger.warning(f"LLM连接失败 (尝试 {attempt + 1})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise RuntimeError("Ollama连接失败，请检查Ollama是否启动")
            except Exception as e:
                logger.error(f"LLM调用失败: {e}")
                raise RuntimeError(f"LLM调用失败: {e}")
        raise RuntimeError("LLM调用失败，已达到最大重试次数")

    async def _generate_stream_openai(self, messages: List[Dict]) -> AsyncGenerator[str, None]:
        """async 流式调用 OpenAI 兼容 API

        【函数做什么】
          保持 HTTP 长连接，逐行读取 SSE 事件流，每收到一个 token 就 yield

        【返回值】
          AsyncGenerator[str, None]: 异步生成器，每次 yield 一个 token

        【设计原理 — SSE 协议】
          1. 使用 httpx 的 client.stream 方法建立长连接（不是一次性请求）
          2. OpenAI SSE 格式的每行数据以 "data: " 开头
          3. 收到 "data: [DONE]" 表示流结束
          4. 每行数据是 JSON 格式：{"choices": [{"delta": {"content": "token"}}]}
          5. 非内容事件（如 role 事件）的 delta 中没有 content 字段

        【边界情况】
          - 空行：跳过
          - JSON 解析失败：跳过该行继续读下一行
          - 长连接中断：抛 RuntimeError
          - 流式场景没有重试（已在 generate_stream 的 docstring 中解释原因）
        """
        payload = self._build_payload(stream=True)
        payload["messages"] = messages
        headers = self._build_headers()

        try:
            # 【作用】建立异步 HTTP 长连接
            # 【原理】client.stream 不会等整个响应完成，而是逐块接收
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", self.chat_url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    # 【作用】逐行读取 SSE 事件流
                    # 【原理】aiter_lines() 是 httpx 的异步逐行迭代器
                    #        每次收到一个换行符就 yield 一行数据
                    async for line in response.aiter_lines():
                        # 【作用】跳过空行（SSE 协议中用空行分隔事件）
                        if not line:
                            continue
                        # 【作用】SSE 协议的每一行都以 "data: " 开头
                        # 【原理】line[6:] 去掉 "data: " 前缀（6 个字符），提取真正的数据
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                # 【作用】SSE 结束标记，收到后停止读取
                                break
                            try:
                                # 【作用】解析 JSON 格式的数据行
                                chunk = json.loads(data)
                                # 【作用】从 OpenAI 流式格式中提取增量内容
                                # 【格式】{"choices": [{"delta": {"content": "token"}}]}
                                # 【注意】第一个事件可能是 role 事件（delta 中没有 content）
                                delta = chunk["choices"][0]["delta"]
                                if "content" in delta:
                                    yield delta["content"]
                            except json.JSONDecodeError:
                                # 【作用】如果某行 JSON 格式不对，跳过不影响后续
                                continue
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            raise RuntimeError(f"流式生成失败: {e}")

    async def _generate_stream_ollama(self, messages: List[Dict]) -> AsyncGenerator[str, None]:
        """async 流式调用 Ollama API

        【函数做什么】
          和 _generate_stream_openai 功能相同，但适配 Ollama 的流式格式

        【设计原理 — Ollama 流式格式】
          Ollama 不使用 SSE 协议的 "data: " 前缀格式。
          每行直接返回一个完整的 JSON 对象：
          {"message": {"content": "token"}, "done": false}
          最后一行：{"message": {"content": ""}, "done": true}

        【和 OpenAI 流式的区别】
          - OpenAI: 行以 "data: " 开头，[DONE] 结束
          - Ollama: 行就是 JSON，done=true 结束
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
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", self.chat_url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            # 【作用】Ollama 返回格式：{"message": {"content": "token"}, "done": false/true}
                            # 【逻辑】先检查 message 和 content 字段是否存在
                            if "message" in chunk and "content" in chunk["message"]:
                                yield chunk["message"]["content"]
                            if chunk.get("done", False):
                                # 【作用】done=true 表示流结束
                                break
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            raise RuntimeError(f"流式生成失败: {e}")

    def _build_messages(self, query: str, context: List[str], history: Optional[List[Dict]]) -> List[Dict]:
        """构建LLM对话消息列表

        【函数做什么】
          将用户问题、检索到的上下文、对话历史拼接成 LLM API 要求的 messages 格式

        【参数】
          query:   用户问题文本
          context: 检索到的参考文档片段列表，每个元素是一段文本
          history: 多轮对话历史，格式 [{"role": "user"/"assistant", "content": "..."}]

        【返回值】
          List[Dict]: 对话消息列表
            格式：[{"role": "system", "content": "系统提示语"},
                   {"role": "user", "content": "历史对话1"},
                   {"role": "assistant", "content": "历史回答1"},
                   {"role": "user", "content": "当前问题"}]

        【设计原理 — 为什么 context 拼进 system 而不是 user】
          把检索到的参考信息放在 system prompt 尾部，而不是作为 user message 发送。
          原因：system prompt 的权重通常高于 user message，模型会更重视这些参考信息。
          如果放在 user message 里，模型可能把它们和用户的问题混在一起，导致"参考信息
          和用户问题之间的边界模糊"，模型可能误把参考信息当成对话的一部分去"回答"。

          但这不是绝对正确的：
          - 某些微调模型对 system/user 权重没区别，放在哪都一样
          - 某些模型（如 Llama 3）的 system prompt 支持度有限
          当前选择是一种"经验设计"，在招股说明书分析场景下经验证效果良好。

        【边界情况】
          - context 为空列表：system prompt 不加参考信息，只包含指令
          - history 为 None：不追加历史对话，只有 system + user
          - history 为空列表：和 None 效果相同

        【面试官可能问】
          Q: 为什么不用 template-based 的方案（如 Jinja2）？
          A: 当前对话结构固定，用简单的字符串拼接就够了。
             如果未来需要支持不同的 prompt 模板（多语言、不同场景），再用 Jinja2 不迟。

          Q: context 过多导致超出 max_tokens 怎么办？
          A: 当前设计不截断。由调用方（retrieval.py）通过 top_k 控制检索数量，
             一般 3-5 段文档不太可能超限。如果确实需要截断，应该在 _build_messages
             之前做，而不是在这里。
        """
        messages = []
        system_prompt = self._build_system_prompt()
        if context:
            # 【作用】把多段参考文档用两个换行拼成一段文本
            # 【原理】join 是高效的字符串拼接方式，避免用 + 号在循环中拼接
            context_text = "\n\n".join(context)
            # 【作用】把参考信息追加到 system prompt 末尾
            # 【设计意图】让模型知道"这些是参考信息，基于它们回答"
            system_prompt += f"\n\n以下是与问题相关的参考信息：\n\n{context_text}"
        messages.append({"role": "system", "content": system_prompt})
        if history:
            # 【作用】追加对话历史（多轮对话场景）
            # 【原理】extend 比多次 append 更高效，一次把整个列表加进来
            messages.extend(history)
        messages.append({"role": "user", "content": query})
        return messages

    def _build_system_prompt(self) -> str:
        """生成系统提示语

        【函数做什么】
          返回招股说明书分析助手的 system prompt，包含回答规则

        【返回值】
          str: 系统提示语文本

        【设计原理】
          6 条规则覆盖了 RAG 助手的核心行为约束：
          1. 基于参考信息回答——约束模型不要超出检索到的范围
          2. 信息不足告知用户——避免幻觉
          3. 简洁准确专业——回答风格约束
          4. 注意财务单位——领域特化（本项目面向招股说明书）
          5. 准确引用表格——领域特化
          6. 不确定不编造——再次强调避免幻觉

          回答格式部分还要求了引用页码、复杂问题分点——这是为了让回答
          结构清晰、可溯源。

        【面试官可能问】
          Q: 为什么 system prompt 不放在配置文件中？
          A: 目前只有这一个场景（招股说明书分析），写死在代码里最直接。
             如果未来需要支持不同的应用场景（如法律文档分析、医疗问答），
             应该把 system prompt 提取成配置项或数据库记录。
        """
        return """你是一个专业的招股说明书分析助手。你的任务是：

1. 基于提供的参考信息回答问题
2. 如果参考信息不足，明确告知用户
3. 回答要简洁、准确、专业
4. 涉及财务数据时，注意单位（万元、亿元等）
5. 表格数据要准确引用
6. 不确定的内容不要编造

重要规则：
- 当参考信息中包含多个公司的数据时，优先回答母公司（发行人）的信息
- 子公司信息通常在表格中标注为"子公司"或注册资本较小
- 母公司的注册资本通常在5,000万元以上，成立时间较早
- 如果问题问的是"公司"，通常指的是母公司而非子公司

faithfulness铁律（违反即为错误回答）：
- 每一条事实性声明必须能在参考信息中找到原文对应，不得补充参考信息中没有的内容
- 不要自行归纳分类（如把零散风险点归纳为"技术风险/经营风险"），除非参考信息本身就有这个分类
- 不要使用你自己的知识来补充或扩展参考信息，即使你知道答案
- 如果参考信息只提到了部分内容，只回答那部分，不要补充"还包括..."之类的扩展

回答格式：
- 直接给出答案
- 必要时引用来源（页码）
- 复杂问题分点说明"""


# 【作用】模块级全局变量，用作 LLMClient 的单例缓存
_llm_client = None


def get_llm_client() -> LLMClient:
    """获取 LLMClient 单例

    【函数做什么】
      单例工厂：全局只初始化一次 LLMClient，后续都返回同一个实例

    【返回值】
      LLMClient 实例

    【设计原理】
      和 retrieval.py 中的 get_retrieval_service() 同样的单例模式。
      延迟初始化 + 全局复用。
      注意：当被 LLMRouter 使用时，不走这里的单例。
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
