"""backend/app/core/llm_router.py — LLM智能路由

优先级：问候语 → Ollama | 复杂问题 → SGLang > DeepSeek > Ollama
故障转移：SGLang失败 → DeepSeek → Ollama
"""

import logging
import re
from typing import List, Dict, Optional

from ..config import settings
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

# 问候语模式（匹配这些直接走Ollama，不需要大模型）
GREETING_PATTERNS = [
    r"^(你好|您好|嗨|哈喽|hi|hello|hey|早上好|下午好|晚上好|早安|晚安)[\s!！。.~～]*$",
]


class LLMRouter:
    """LLM路由器：统一管理多个LLM引擎，智能选择和故障转移"""

    def __init__(self):
        # 检测SGLang服务是否在运行
        self.sglang_available = self._check_service(settings.SGLANG_HOST, "SGLang")
        # 检测DeepSeek服务是否在运行（需要API Key）
        self.deepseek_available = self._check_service(settings.DEEPSEEK_HOST, "DeepSeek") if settings.DEEPSEEK_KEY else False
        # 检测Ollama服务是否在运行
        self.ollama_available = self._check_service(settings.OLLAMA_HOST, "Ollama")

        self.sglang_client = None
        self.deepseek_client = None
        self.ollama_client = None

        if self.sglang_available:
            self.sglang_client = self._create_client(
                provider="sglang",
                host=settings.SGLANG_HOST,
                model=settings.SGLANG_MODEL
            )
            logger.info(f"✅ SGLang 已连接: {settings.SGLANG_MODEL}")

        if self.deepseek_available:
            self.deepseek_client = self._create_client(
                provider="deepseek",
                host=settings.DEEPSEEK_HOST,
                model=settings.DEEPSEEK_MODEL,
                api_key=settings.DEEPSEEK_KEY
            )
            logger.info(f"✅ DeepSeek 已连接: {settings.DEEPSEEK_MODEL}")

        if self.ollama_available:
            self.ollama_client = self._create_client(
                provider="ollama",
                host=settings.OLLAMA_HOST,
                model=settings.OLLAMA_MODEL
            )
            logger.info(f"✅ Ollama 已连接: {settings.OLLAMA_MODEL}")

        if not self.sglang_available and not self.deepseek_available and not self.ollama_available:
            logger.error("❌ 没有可用的LLM服务！请启动 SGLang (端口30000) 或 Ollama (端口11434) 或配置 DeepSeek API")

    def _check_service(self, host: str, name: str) -> bool:
        """检测LLM服务是否已经启动"""
        try:
            import httpx
            # DeepSeek是云端API，直接返回True（有Key就认为可用）
            if name == "DeepSeek":
                return True
            root_url = host.replace("/v1/chat/completions", "").replace("/api/chat", "")
            resp = httpx.get(f"{root_url}/health", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def _create_client(self, provider: str, host: str, model: str, api_key: str = "") -> LLMClient:
        """创建LLM客户端（绕过__init__的单一provider限制）

        原理：用__new__创建空对象，手动设置属性
        """
        client = LLMClient.__new__(LLMClient)
        client.provider = provider
        client.host = host
        client.model = model
        client.api_key = api_key
        client.max_retries = 3
        client.timeout = 60

        # 根据provider设置API地址
        if provider == "sglang":
            client.chat_url = f"{host}/v1/chat/completions"
        elif provider == "ollama":
            client.chat_url = f"{host}/api/chat"
        elif provider == "deepseek":
            # DeepSeek host 已含 /v1，直接拼 /chat/completions
            client.chat_url = f"{host}/chat/completions"

        return client

    def _is_greeting(self, query: str) -> bool:
        """判断是否是问候语"""
        return any(re.match(p, query.strip(), re.IGNORECASE) for p in GREETING_PATTERNS)

    def _select_client(self, query: str, history: Optional[List[Dict]] = None) -> LLMClient:
        """智能选择 LLM 客户端

        规则：
          1. 问候语 → Ollama（简单任务，不需要大模型）
          2. 其他 → SGLang > DeepSeek > Ollama
        """
        # 问候语优先走Ollama
        if self._is_greeting(query):
            if self.ollama_available:
                logger.info(f"[智能调度] 问候语 → Ollama | {query[:50]}")
                return self.ollama_client  # type: ignore  # ollama_available=True 时一定不为 None
            # Ollama不可用则走正常优先级
            logger.info("[智能调度] 问候语但Ollama不可用，走正常优先级")

        available_clients = []
        if self.sglang_available:
            available_clients.append(("sglang", self.sglang_client))
        if self.deepseek_available:
            available_clients.append(("deepseek", self.deepseek_client))
        if self.ollama_available:
            available_clients.append(("ollama", self.ollama_client))

        if not available_clients:
            raise RuntimeError("没有可用的LLM服务")

        # 统一优先级：SGLang > DeepSeek > Ollama
        for provider, client in available_clients:
            logger.info(f"[智能调度] {provider} | {query[:50]}...")
            return client  # type: ignore  # available_clients 不为空，client 一定有值

        raise RuntimeError("没有可用的LLM服务")  # 理论上不会走到这里

    def _get_fallback_client(self, current_client: LLMClient) -> Optional[LLMClient]:
        """获取备用客户端（故障转移）

        优先级：SGLang > DeepSeek > Ollama
        """
        fallback_clients = []
        if self.sglang_available and current_client != self.sglang_client:
            fallback_clients.append(self.sglang_client)
        if self.deepseek_available and current_client != self.deepseek_client:
            fallback_clients.append(self.deepseek_client)
        if self.ollama_available and current_client != self.ollama_client:
            fallback_clients.append(self.ollama_client)

        return fallback_clients[0] if fallback_clients else None

    async def generate(self, query: str, context: List[str], history: Optional[List[Dict]] = None) -> str:
        """生成答案（非流式）"""
        client = self._select_client(query, history)
        try:
            return await client.generate(query=query, context=context, history=history)
        except Exception as e:
            logger.warning(f"LLM调用失败: {e}")
            fallback = self._get_fallback_client(client)
            if fallback:
                logger.info(f"故障转移到备用引擎")
                return await fallback.generate(query=query, context=context, history=history)
            raise

    async def generate_stream(self, query: str, context: List[str], history: Optional[List[Dict]] = None):
        """生成答案（流式）"""
        client = self._select_client(query, history)
        try:
            async for chunk in client.generate_stream(query=query, context=context, history=history):
                yield chunk
        except Exception as e:
            logger.warning(f"LLM流式调用失败: {e}")
            fallback = self._get_fallback_client(client)
            if fallback:
                logger.info(f"故障转移到备用引擎")
                async for chunk in fallback.generate_stream(query=query, context=context, history=history):
                    yield chunk
            else:
                raise


_llm_router = None


def get_llm_router() -> LLMRouter:
    """单例工厂"""
    global _llm_router
    if _llm_router is None:
        _llm_router = LLMRouter()
    return _llm_router
