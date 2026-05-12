"""
backend/app/core/llm_router.py — LLM智能调度器（单一职责）
================================================================
作用：根据任务复杂度智能选择LLM推理引擎，实现SGLang和Ollama之间的自动分流
原理：
  1. 自动检测可用服务，无需手动配置
  2. 简单任务（事实性问答）→ Ollama轻量模型（省显存、响应快）
  3. 复杂任务（分析/推理）→ SGLang大模型（高质量、深度分析）
  4. 主引擎故障时自动切换到备用引擎（故障降级）
  5. 只有一个服务时，所有任务走该服务（单服务兼容）
"""

import logging  # 作用：记录日志，方便排查问题
import re       # 作用：正则表达式，用于匹配简单任务模式
import requests # 作用：发送HTTP请求，检测服务是否可用
from typing import List, Dict, Any, Optional  # 作用：类型注解，让代码更清晰

from ..config import settings    # 作用：读取配置文件中的参数
from .llm_client import LLMClient  # 作用：LLM调用客户端，负责实际发请求

logger = logging.getLogger(__name__)  # 作用：创建当前模块的日志记录器


class LLMRouter:
    """
    作用：LLM智能调度器，自动选择最优引擎处理用户问题
    原理：
      1. 自动发现：启动时检测SGLang和Ollama是否运行
      2. 智能分流：根据问题复杂度选择轻量或大模型
      3. 故障降级：主引擎报错自动切换到备用引擎
      4. 单服务兼容：只有一个服务时，所有任务走该服务
    """

    # 复杂任务关键词（触发使用大模型SGLang）
    # 作用：如果用户问题中包含这些词，说明需要深度分析，交给大模型
    COMPLEX_KEYWORDS = [
        # 分析类——需要综合理解多个信息点
        "分析", "比较", "对比", "评估", "评价",
        # 趋势类——需要看时间序列变化
        "趋势", "变化", "发展", "预测", "展望",
        # 总结类——需要从多个段落中提炼核心
        "总结", "归纳", "概述", "概括", "提炼",
        # 推理类——需要多步逻辑推导
        "为什么", "原因", "影响", "意义", "作用",
        # 代码类——需要生成或理解代码
        "代码", "程序", "脚本", "函数", "算法",
        # 深度类——需要全面深入的回答
        "详细", "深入", "全面", "系统", "综合",
    ]

    # 简单任务特征（使用轻量模型Ollama）
    # 作用：通过正则匹配"一问一答"型的问题，这些直接查文档就能回答，不需要深度分析
    SIMPLE_PATTERNS = [
        r"^\d+年.*是多少",           # 如 "2023年营收是多少"
        r"^什么是",                  # 如 "什么是营业收入"
        r"^.*的?定义",               # 如 "营业收入的定义"
        r"^第?\d+页",                # 如 "第15页说了什么"
        r"^.*是\d+",                 # 如 "营收是1000万吗"
    ]

    def __init__(self):
        """
        作用：初始化智能调度器
        原理：自动检测可用的LLM服务，检测到了就创建对应的客户端
        逻辑：
          1. 从config读取复杂度阈值
          2. 检测SGLang是否可用（localhost:30000）
          3. 检测Ollama是否可用（localhost:11434）
          4. 分别创建可用的客户端
          5. 如果两个都不可用，抛异常退出
        """
        # 读取复杂度阈值（config.py中LLM_COMPLEX_THRESHOLD，默认100字符）
        # 作用：问题超过这个长度就算复杂任务
        self.complex_threshold = settings.LLM_COMPLEX_THRESHOLD

        # 检测SGLang服务是否在运行
        # 原理：发一个GET请求到SGLang地址，看能不能连上
        # 注意：Windows无法通过localhost访问WSL2服务，必须使用WSL2 IP地址
        sglang_check_host = "http://172.23.190.86:30000"
        self.sglang_available = self._check_service(
            sglang_check_host,  # 使用WSL2 IP地址检测
            "SGLang"
        )
        # 检测Ollama服务是否在运行
        self.ollama_available = self._check_service(
            settings.OLLAMA_HOST,  # 如 http://localhost:11434
            "Ollama"
        )

        # 先占位，后面根据检测结果来初始化
        self.sglang_client = None
        self.ollama_client = None

        # 如果SGLang可用，创建SGLang客户端
        # 作用：把SGLang的地址、模型名等信息封装成一个LLMClient对象
        if self.sglang_available:
            self.sglang_client = self._create_client(
                provider="sglang",          # 标记这个客户端是SGLang
                host=settings.SGLANG_HOST,   # SGLang服务地址
                model=settings.SGLANG_MODEL  # 模型名，如 Qwen2.5-1.5B-Instruct
            )
            logger.info(f"✅ SGLang 已连接: {settings.SGLANG_MODEL}")

        # 如果Ollama可用，创建Ollama客户端
        if self.ollama_available:
            self.ollama_client = self._create_client(
                provider="ollama",          # 标记这个客户端是Ollama
                host=settings.OLLAMA_HOST,   # Ollama服务地址
                model=settings.OLLAMA_MODEL  # 模型名，如 qwen2.5:0.5b
            )
            logger.info(f"✅ Ollama 已连接: {settings.OLLAMA_MODEL}")

        # 安全校验：两个服务至少有一个可用才能继续
        # 原理：没有LLM服务，RAG系统的最后一步"生成"就无法完成
        if not self.sglang_available and not self.ollama_available:
            logger.error("❌ 没有可用的LLM服务！")
            logger.error("请启动 SGLang (端口30000) 或 Ollama (端口11434)")
            raise RuntimeError("没有可用的LLM服务")

        logger.info("LLM智能调度器初始化完成")

    def _check_service(self, host: str, name: str) -> bool:
        """
        作用：检测LLM服务是否已经启动
        原理：发一个简单的HTTP GET请求，能连上说明服务在运行
        逻辑：
          1. 从host地址中提取根路径（去掉/v1/chat/completions或/api/chat后缀）
          2. 用requests.get()发请求，超时2秒
          3. 能收到响应（不管状态码）→ 服务可用
          4. 连接被拒绝 → 服务没启动
          5. 其他异常 → 也算不可用
        参数：
            host: 服务地址，如 http://localhost:11434
            name: 服务名，如 "Ollama"（仅用于日志）
        返回：
            True=可用，False=不可用
        """
        try:
            # 去掉API路径后缀，拿到服务根路径
            # 原理：SGLang的聊天API是 /v1/chat/completions，Ollama是 /api/chat
            # 但检测服务时只需要访问根路径就够了，不需要完整API路径
            root_url = host.replace("/v1/chat/completions", "").replace("/api/chat", "")
            # 发GET请求，2秒没响应就算超时
            response = requests.get(root_url, timeout=2)
            # 只要请求没抛异常，不管返回什么状态码，都认为服务在运行
            logger.debug(f"{name} 状态码: {response.status_code}")
            return True
        except requests.exceptions.ConnectionError:
            # 连接被拒绝 → 服务没启动
            logger.debug(f"{name} 未启动 ({host})")
            return False
        except Exception as e:
            # 其他异常（如DNS解析失败）→ 也算不可用
            logger.debug(f"{name} 检测失败: {e}")
            return False

    def _create_client(self, provider: str, host: str, model: str) -> LLMClient:
        """
        作用：创建一个LLM客户端对象，封装了服务地址和模型信息
        原理：不调用__init__，直接用__new__创建空对象，然后手动赋值属性
              因为LLMClient的__init__会从config读provider，
              但我们这里要手动指定provider（绕过config的单一provider限制）
        逻辑：
          1. 用__new__创建LLMClient的空实例
          2. 手动设置provider、host、model、api_key等属性
          3. 根据provider类型设置不同的API地址格式
            - SGLang: /v1/chat/completions（OpenAI兼容格式）
            - Ollama: /api/chat（Ollama原生格式）
        参数：
            provider: "sglang" 或 "ollama"
            host: 服务地址
            model: 模型名称
        返回：
            LLMClient实例
        """
        # 用__new__创建空对象，不走__init__
        # 作用：我们自己在外面设置属性，比走__init__更灵活
        client = LLMClient.__new__(LLMClient)
        client.provider = provider      # 标记是sglang还是ollama
        # 强制使用WSL2 IP地址，因为Windows无法通过localhost访问WSL2服务
        # 注意：WSL2 IP可能会变化，如果重启后连接失败，需要重新获取IP
        if provider == "sglang":
            client.host = "http://172.23.190.86:30000"
            client.model = "default"
        else:
            client.host = host              # 服务地址
            client.model = model            # 模型名称
        client.api_key = ""             # 本地服务不需要API Key
        client.timeout = settings.LLM_TIMEOUT         # 超时时间（秒）
        client.max_retries = settings.LLM_MAX_RETRIES  # 最大重试次数

        # 设置API地址
        # 原理：SGLang和Ollama的API接口路径不同
        # SGLang兼容OpenAI格式，Ollama有自己的格式
        if provider == "sglang":
            client.chat_url = f"{client.host}/v1/chat/completions"
        else:
            client.chat_url = f"{client.host}/api/chat"

        return client

    def generate(
        self,
        query: str,
        context: List[str],
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        作用：智能生成答案——根据问题复杂度选择引擎，主引擎失败自动切备用
        原理：
          1. 先选引擎（_select_client）
          2. 调用选中引擎的generate方法（非流式，一次性返回）
          3. 如果失败，尝试换成另一个引擎
          4. 两个都失败，抛异常
        参数：
            query: 用户问题
            context: 检索到的相关文本列表（作为RAG上下文）
            history: 对话历史（多轮对话用）
        返回：
            生成的答案文本字符串
        """
        # 第一步：选引擎
        client = self._select_client(query, history)

        # 第二步：调用选中的引擎（非流式）
        try:
            return client.generate(query, context, history)
        except Exception as e:
            logger.warning(f"主引擎失败: {e}")
            fallback = self._get_fallback_client(client)
            if fallback:
                logger.info(f"切换到备用引擎: {fallback.provider}")
                return fallback.generate(query, context, history)
            raise

    def generate_stream(
        self,
        query: str,
        context: List[str],
        history: Optional[List[Dict[str, str]]] = None
    ):
        """
        作用：流式智能生成——逐步返回答案token，前端可以实现打字机效果
        原理：和generate()一样，但用yield逐步返回
        参数：
            query: 用户问题
            context: 检索到的文本
            history: 对话历史
        返回：
            生成器，每次yield一个token字符串
        """
        client = self._select_client(query, history)

        try:
            # yield from = 把内部生成器的每个值逐个传出来
            yield from client.generate_stream(query, context, history)
        except Exception as e:
            logger.warning(f"主引擎失败: {e}")
            fallback = self._get_fallback_client(client)
            if fallback:
                logger.info(f"切换到备用引擎: {fallback.provider}")
                yield from fallback.generate_stream(query, context, history)
            raise

    def _select_client(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> LLMClient:
        """
        作用：根据可用服务和任务复杂度，选择最优的LLM客户端
        原理：
          1. 只有一个服务 → 不用判断，直接用那个
          2. 两个服务都有 → 调用_analyze_complexity判断复杂度
             - 复杂 → SGLang（大模型，深度分析）
             - 简单 → Ollama（轻量模型，快速响应）
        参数：
            query: 用户问题
            history: 对话历史
        返回：
            选中的LLMClient实例
        """
        # 情况1：只有SGLang，没有Ollama
        if self.sglang_available and not self.ollama_available:
            logger.debug("只有 SGLang 可用")
            return self.sglang_client

        # 情况2：只有Ollama，没有SGLang
        if self.ollama_available and not self.sglang_available:
            logger.debug("只有 Ollama 可用")
            return self.ollama_client

        # 情况3：两个都有，按复杂度分流
        # 这是智能分流最核心的一行
        complexity = self._analyze_complexity(query, history)

        if complexity == "complex":
            # 复杂任务 → 大模型SGLang
            logger.info(f"[智能调度] 复杂任务 → SGLang | 问题: {query[:50]}...")
            return self.sglang_client
        else:
            # 简单任务 → 轻量模型Ollama
            logger.info(f"[智能调度] 简单任务 → Ollama | 问题: {query[:50]}...")
            return self.ollama_client

    def _get_fallback_client(self, current_client: LLMClient) -> Optional[LLMClient]:
        """
        作用：获取当前引擎的备用引擎
        原理：如果当前是SGLang，备用就是Ollama，反之亦然
        参数：
            current_client: 当前失败的客户端
        返回：
            备用客户端，如果没有备用则返回None
        """
        # 当前是SGLang，且Ollama可用 → 切到Ollama
        if current_client == self.sglang_client and self.ollama_available:
            return self.ollama_client
        # 当前是Ollama，且SGLang可用 → 切到SGLang
        if current_client == self.ollama_client and self.sglang_available:
            return self.sglang_client
        # 没有可用的备用引擎
        return None

    def _analyze_complexity(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        作用：分析一个问题属于"简单任务"还是"复杂任务"
        原理：按优先级依次判断，一旦命中某个条件就返回结果：
          1. 有对话历史 → 复杂（多轮对话需要上下文理解）
          2. 包含复杂关键词 → 复杂（分析/对比/总结等需要深度推理）
          3. 问题超过长度阈值 → 复杂（长问题通常需要综合分析）
          4. 匹配简单模式 → 简单（一问一答式的问题）
          5. 都不匹配 → 默认简单（宁可用轻量模型跑，省资源）
        参数：
            query: 用户问题
            history: 对话历史
        返回：
            "simple" 或 "complex"
        """
        # 规则1：有对话历史 → 复杂任务
        # 原理：多轮对话需要理解上下文，轻量模型容易"忘记"前面说过什么
        if history and len(history) > 0:
            logger.debug("复杂度判断: 有对话历史 → complex")
            return "complex"

        # 规则2：关键词匹配
        # 原理：包含"分析""为什么""总结"等词，说明用户想要深度回答
        for keyword in self.COMPLEX_KEYWORDS:
            if keyword in query:
                logger.debug(f"复杂度判断: 关键词'{keyword}' → complex")
                return "complex"

        # 规则3：问题长度超过阈值
        # 原理：长问题往往包含多个子问题或详细背景，需要综合回答
        if len(query) > self.complex_threshold:
            logger.debug(f"复杂度判断: 长度{len(query)}>{self.complex_threshold} → complex")
            return "complex"

        # 规则4：简单模式匹配
        # 原理：如果匹配了"SIMPLE_PATTERNS"中的正则，说明是一问一答式
        for pattern in self.SIMPLE_PATTERNS:
            if re.match(pattern, query):
                logger.debug(f"复杂度判断: 匹配简单模式 → simple")
                return "simple"

        # 规则5：默认返回简单
        # 原理：没有明确的复杂信号，就按简单处理，省资源
        logger.debug("复杂度判断: 默认 → simple")
        return "simple"

    def get_stats(self) -> Dict[str, Any]:
        """
        作用：获取调度器的当前状态信息，方便调试和监控
        返回：包含各引擎可用状态、地址、模型名的字典
        """
        return {
            "sglang": {
                "available": self.sglang_available,
                "host": settings.SGLANG_HOST if self.sglang_available else None,
                "model": settings.SGLANG_MODEL if self.sglang_available else None,
            },
            "ollama": {
                "available": self.ollama_available,
                "host": settings.OLLAMA_HOST if self.ollama_available else None,
                "model": settings.OLLAMA_MODEL if self.ollama_available else None,
            },
            "threshold": self.complex_threshold,
        }


# 全局调度器实例（单例模式）
# 作用：整个程序只创建一个LLMRouter实例
# 原理：_llm_router是模块级变量，第一次调用get_llm_router()时创建，
#       后续调用直接返回已有的实例，避免重复检测服务和创建客户端
_llm_router = None


def get_llm_router() -> LLMRouter:
    """
    作用：获取LLM智能调度器单例
    原理：全局变量_llm_router，第一次调用时创建，后续直接返回
    返回：
        LLMRouter实例
    """
    global _llm_router  # 声明要使用模块级的全局变量
    if _llm_router is None:  # 第一次调用，创建实例
        _llm_router = LLMRouter()
    return _llm_router  # 返回已有的实例
