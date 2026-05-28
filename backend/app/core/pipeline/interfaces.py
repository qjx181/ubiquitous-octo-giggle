"""pipeline/interfaces.py — RAG管道阶段标准接口定义

受 HiveWard GatewayOpenClawAdapter 启发，为 RAG 管道的每个阶段定义 Protocol
接口。任何实现了对应 Protocol 的类都可以作为该阶段的 Adapter 被 Pipeline 调用。

设计原则（借鉴 HiveWard）：
  - 每个阶段有明确输入/输出类型，互不依赖
  - Adapter 模式：Pipeline 不关心背后是哪个实现，只认接口
  - 替代/新增阶段不需要改 Pipeline 代码

注意：使用 Protocol（鸭子类型）而非 ABC，好处是现有类不需要显式继承，
只要方法签名匹配就能作为 Adapter 传入 Pipeline。这保留了现有代码的独立性。

面试官可能问：
  Q: 为什么用 Protocol 而不是 ABC？
  A: Protocol 是结构子类型（structural subtyping），ABC 是名义子类型
     （nominal subtyping）。用 Protocol 的好处：
     1. 不需要现有类显式继承接口基类
     2. 第三方库的类只要方法签名匹配就能直接用
     3. 测试时可以用 mock 对象而不需要 mock 框架

  Q: 你的 Pipeline 和 LangChain LCEL 的 Runnable 有什么区别？
  A: LangChain LCEL 是通用的链式调用框架，支持复杂的 DAG、并行、条件分支。
     我的 Pipeline 更薄——只做线性编排+统一的 Context 数据流传递。
     优点是简单、可读性强、调试方便；缺点是缺少 DAG 能力。
     如果需要 DAG，可以借鉴 HiveWard 的 Blueprint 设计（React Flow 画布）。
"""

from typing import Protocol, List, Dict, Optional, AsyncIterator, runtime_checkable


# ========================================================
# 数据类：管道上下文，在各阶段之间传递
# ========================================================

class PipelineContext:
    """管道上下文：在阶段之间传递的不可变数据包

    设计思路（借鉴 HiveWard 蓝图节点间的 data flow）：
      - 每个阶段读取自己需要的字段，写入自己负责的字段
      - 不修改其他阶段的数据（PipelineContext 记录每个阶段输出但不覆盖）
      - 方便调试：任意时刻 dump context 就能看到完整管道状态

    边界情况：
      - 各阶段可能产生相同的字段名？→ 通过 stage_name 前缀区分
      - 某个阶段跳过？→ 该阶段对应的字段保持初始值 None
    """
    # 输入（管道入口设置）
    question: str = ""
    top_k: int = 3
    content_type: Optional[str] = None
    history: Optional[List[Dict]] = None
    kb_name: Optional[str] = None  # 知识库名称过滤（可选，用于多公司场景）

    # Query理解阶段输出
    normalized_query: Optional[str] = None
    clarification_prompt: Optional[str] = None
    sub_queries: Optional[List[str]] = None

    # Embedding阶段输出
    query_vector: Optional[List[float]] = None

    # 向量检索阶段输出
    vector_results: Optional[List[Dict]] = None

    # 混合检索阶段输出
    hybrid_results: Optional[List[Dict]] = None

    # 重排序阶段输出
    reranked_results: Optional[List[Dict]] = None

    # LLM生成阶段输出
    answer: Optional[str] = None

    # 引用来源（SourcesBuilder 阶段写入）
    sources: Optional[List[Dict]] = None

    # 管道总耗时
    total_time: Optional[float] = None

    # 元数据（调试/监控用）
    stage_times: Dict[str, float] = None  # type: ignore[assignment]
    stage_errors: Dict[str, str] = None  # type: ignore[assignment]

    def __init__(self, **kwargs):
        self.stage_times = {}
        self.stage_errors = {}
        for k, v in kwargs.items():
            setattr(self, k, v)


# ========================================================
# 接口定义：每个管道阶段一个 Protocol
# ========================================================

@runtime_checkable
class QueryAdapter(Protocol):
    """Query理解阶段接口

    输入：原始用户问题 + 对话历史
    输出：规范化查询、子查询列表、是否需要澄清
    """
    async def process_query(self, question: str,
                            history: Optional[List[Dict]] = None) -> Dict:
        """处理用户问题

        Returns:
            {"normalized_query": str,
             "sub_queries": list[str] | None,
             "needs_clarification": bool,
             "clarification_prompt": str | None}
        """
        ...


@runtime_checkable
class EmbeddingAdapter(Protocol):
    """Embedding编码阶段接口

    输入：文本
    输出：1024维向量（BGE-M3归一化向量，内积=余弦相似度）
    """
    async def encode(self, text: str) -> List[float]:
        """单条文本编码"""
        ...

    async def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """批量文本编码"""
        ...


@runtime_checkable
class VectorSearchAdapter(Protocol):
    """向量检索阶段接口

    输入：向量 + 过滤条件
    输出：检索结果列表（包含 text, score, page, section 等元数据）
    """
    async def search(self, vector: List[float], top_k: int,
                     content_type: Optional[str] = None,
                     kb_name: Optional[str] = None) -> List[Dict]:
        """向量相似度搜索"""
        ...


@runtime_checkable
class HybridSearchAdapter(Protocol):
    """混合检索阶段接口（向量+BM25融合）

    输入：文本查询 + 向量检索结果
    输出：融合后的检索结果列表
    """
    async def search(self, query: str, vector_results: List[Dict],
                     top_k: int) -> List[Dict]:
        """混合检索（BM25 + 向量）"""
        ...


@runtime_checkable
class RerankAdapter(Protocol):
    """重排序阶段接口

    输入：文本查询 + 候选文档列表
    输出：重排序后的文档列表（带 rerank_score 字段）
    """
    async def rerank(self, query: str, candidates: List[Dict],
                     top_k: int) -> List[Dict]:
        """Cross-Encoder 精排"""
        ...


@runtime_checkable
class LLMGenerationAdapter(Protocol):
    """LLM生成阶段接口

    输入：问题 + 检索上下文
    输出：生成的答案文本
    """
    async def generate(self, query: str, context: List[str],
                       history: Optional[List[Dict]] = None) -> str:
        """非流式生成"""
        ...

    async def generate_stream(self, query: str, context: List[str],
                              history: Optional[List[Dict]] = None) -> AsyncIterator[str]:
        """流式生成"""
        ...


@runtime_checkable
class PipelineStage(Protocol):
    """通用管道阶段接口

    每个阶段都是一个 callable，接收 PipelineContext，返回 PipelineContext。
    这样不管什么阶段的处理逻辑，对 Pipeline 编排器而言都是统一的。
    """
    name: str  # 阶段名称，用于日志和调试

    async def __call__(self, ctx: PipelineContext) -> PipelineContext:
        ...
