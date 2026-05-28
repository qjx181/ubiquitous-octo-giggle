"""pipeline/adapters.py — 现有服务的 Adapter 实现

每个 Adapter 封装一个现有的服务组件，将其适配为 PipelineStage 接口。
这样 Pipeline 编排器不需要知道服务内部的实现细节。

设计思路（借鉴 HiveWard GatewayOpenClawAdapter 的 RuntimeAdapter 模式）：
  HiveWard 的 RuntimeAdapter 为 OpenClaw/Codex/Claude Code 提供统一接口：
    listModels() / listAgents() / startTask() / waitForTask()
  我们的 Adapter 做同样的事——为 RAG 各阶段提供统一接口：
    process_query() / encode() / search() / rerank() / generate()

区别：
  HiveWard 的 Adapter 跨 Agent 类型（OpenClaw ↔ Codex ↔ Claude Code）
  我们的 Adapter 跨实现（BGE-M3 ↔ 其他embedding / MinerU ↔ PyMuPDF / ...）

实现细节：
  - 每个 Adapter 接受现有服务的实例，不走全局单例（方便测试和替换）
  - Adapter 不会破坏现有代码：现有模块可以继续直接使用
  - 所有 Adapter 都实现 __call__(ctx) → ctx 接口，可被 Pipeline 编排

面试官可能问：
  Q: Adapter 层是不是多余的？直接调现有服务不行吗？
  A: 对于当前只有一个实现的情况，确实可以说"extra layer of indirection"。
     但 Adapter 的价值在于：
     1. 切换实现时不需要改 Pipeline 代码（比如 BGE-M3 → 其他embedding）
     2. 测试时可以注入 mock Adapter，不需要加载真实模型
     3. 每个阶段职责明确，新成员看 Adapter 就知道每个阶段做什么
     这在 HiveWard 的 RuntimeAdapter 上已经验证过了——它能在不修改蓝图
     逻辑的前提下，把执行引擎从 OpenClaw 换成 Codex。

  Q: 为什么 Adapter 不直接继承 PipelineStage Protocol？
  A: Protocol 是结构子类型，不需要显式继承。
     只要类有 name 属性和 __call__ 方法签名匹配，Pipeline 就能用。
     继承反而带来耦合——Adapter 必须从某个基类派生。
"""

import logging
import time
from typing import List, Dict, Optional

from .interfaces import PipelineContext, PipelineStage

logger = logging.getLogger(__name__)


# ========================================================
# Query理解阶段 Adapter
# ========================================================

class QueryUnderstandingAdapter:
    """Query理解阶段 Adapter

    包装 query_understanding 模块，适配 PipelineStage 接口。
    如果 query_understanding 不可用，透传原始问题。
    """
    name = "query_understanding"

    def __init__(self, service=None):
        """初始化 Adapter

        Args:
            service: query_understanding 服务实例（或 None）
        """
        self._service = service

    async def __call__(self, ctx: PipelineContext) -> PipelineContext:
        start = time.perf_counter()
        if self._service is None:
            ctx.normalized_query = ctx.question
            ctx.clarification_prompt = None
            ctx.stage_times[self.name] = time.perf_counter() - start
            return ctx

        try:
            u = self._service.understand(ctx.question,
                                         conversation_history=ctx.history)
            if u.needs_clarification:
                ctx.clarification_prompt = u.clarification_prompt
                ctx.normalized_query = ctx.question  # 保持原样
            else:
                ctx.normalized_query = u.normalized_query
                if u.sub_queries and len(u.sub_queries) > 1:
                    ctx.sub_queries = u.sub_queries
        except Exception as e:
            logger.warning(f"[{self.name}] 处理失败，使用原始问题: {e}")
            ctx.normalized_query = ctx.question
            ctx.stage_errors[self.name] = str(e)

        ctx.stage_times[self.name] = time.perf_counter() - start
        return ctx


# ========================================================
# Embedding编码阶段 Adapter
# ========================================================

class EmbeddingAdapter:
    """Embedding编码阶段 Adapter

    包装 embedding_service，将文本转为向量。
    """
    name = "embedding"

    def __init__(self, service):
        self._service = service

    async def __call__(self, ctx: PipelineContext) -> PipelineContext:
        start = time.perf_counter()
        try:
            q = ctx.normalized_query or ctx.question
            ctx.query_vector = self._service.encode(q)
        except Exception as e:
            logger.error(f"[{self.name}] 编码失败: {e}")
            ctx.stage_errors[self.name] = str(e)
            raise  # Embedding 是基础依赖，失败必须抛
        ctx.stage_times[self.name] = time.perf_counter() - start
        return ctx


# ========================================================
# 向量检索阶段 Adapter
# ========================================================

class VectorSearchAdapter:
    """向量检索阶段 Adapter

    包装 Milvus 客户端，执行向量相似度搜索。

    对比 HiveWard 的设计：HiveWard 的 listChannels() / listSessions()
    是获取 Agent 会话列表，我们的 search() 是获取语义相似文档。
    共同点是都做"从外部存储中检索结果"。
    """
    name = "vector_search"

    def __init__(self, milvus_client):
        self._client = milvus_client

    async def __call__(self, ctx: PipelineContext) -> PipelineContext:
        start = time.perf_counter()
        try:
            if ctx.query_vector is None:
                raise RuntimeError("向量检索缺少 query_vector，Embedding 阶段可能未执行")

            # 多取候选（top_k*6），让 reranker 有更多候选可排序
            candidate_k = ctx.top_k * 6
            ctx.vector_results = self._client.search(
                embedding=ctx.query_vector,
                top_k=candidate_k,
                content_type=ctx.content_type,
                kb_name=ctx.kb_name
            )
        except Exception as e:
            logger.error(f"[{self.name}] 检索失败: {e}")
            ctx.stage_errors[self.name] = str(e)
            ctx.vector_results = []
        ctx.stage_times[self.name] = time.perf_counter() - start
        return ctx


# ========================================================
# 混合检索阶段 Adapter
# ========================================================

class HybridSearchAdapter:
    """混合检索阶段 Adapter（向量 + BM25）

    包装 bm25_retriever，将向量检索结果和 BM25 结果融合。
    如果混合检索不可用，直接透传向量检索结果。
    """
    name = "hybrid_search"

    def __init__(self, hybrid_retriever=None):
        self._retriever = hybrid_retriever

    async def __call__(self, ctx: PipelineContext) -> PipelineContext:
        start = time.perf_counter()
        if self._retriever is None:
            ctx.hybrid_results = ctx.vector_results
            ctx.stage_times[self.name] = time.perf_counter() - start
            return ctx

        q = ctx.normalized_query or ctx.question
        try:
            rr = self._retriever.search(
                query=q,
                vector_results=[(r, r.get("score", 0.0)) for r in (ctx.vector_results or [])],
                top_k=ctx.top_k
            )
            ctx.hybrid_results = [{**d, "score": s} for d, s in rr]
        except Exception as e:
            logger.warning(f"[{self.name}] 混合检索失败，降级为纯向量检索: {e}")
            ctx.hybrid_results = ctx.vector_results
            ctx.stage_errors[self.name] = str(e)

        ctx.stage_times[self.name] = time.perf_counter() - start
        return ctx


# ========================================================
# 重排序阶段 Adapter
# ========================================================

class RerankAdapter:
    """重排序阶段 Adapter

    包装 RerankerService，用 Cross-Encoder 精排检索结果。
    如果 Reranker 不可用，透传上一阶段的结果。
    """
    name = "reranker"

    def __init__(self, reranker_service=None):
        self._service = reranker_service

    async def __call__(self, ctx: PipelineContext) -> PipelineContext:
        start = time.perf_counter()
        if self._service is None:
            ctx.reranked_results = ctx.hybrid_results
            ctx.stage_times[self.name] = time.perf_counter() - start
            return ctx

        candidates = ctx.hybrid_results or ctx.vector_results or []
        if not candidates:
            ctx.reranked_results = []
            ctx.stage_times[self.name] = time.perf_counter() - start
            return ctx

        q = ctx.normalized_query or ctx.question
        try:
            docs = [{**r, "content": r.get("text", "")} for r in candidates]
            rd = self._service.rerank_documents(query=q, documents=docs,
                                                top_k=ctx.top_k)
            # 移除临时字段，保留 rerank_score
            ctx.reranked_results = []
            for d in rd:
                item = {k: v for k, v in d.items()
                        if k not in ("content", "rerank_score")}
                item["rerank_score"] = d.get("rerank_score", 0.0)
                ctx.reranked_results.append(item)
        except Exception as e:
            logger.warning(f"[{self.name}] 重排序失败，使用重排前结果: {e}")
            ctx.reranked_results = candidates
            ctx.stage_errors[self.name] = str(e)

        ctx.stage_times[self.name] = time.perf_counter() - start
        return ctx


# ========================================================
# 分数阈值过滤阶段 Adapter
# ========================================================

class ScoreThresholdAdapter:
    """分数阈值过滤阶段 Adapter

    对最终检索结果做分数阈值下限过滤，低于阈值的弃掉。
    保持至少1个结果，避免 LLM 无上下文可用。
    """
    name = "score_filter"

    def __init__(self, min_score: float = 0.3):
        self._min_score = min_score

    async def __call__(self, ctx: PipelineContext) -> PipelineContext:
        start = time.perf_counter()
        results = ctx.reranked_results or []
        if not results or self._min_score <= 0:
            ctx.stage_times[self.name] = time.perf_counter() - start
            return ctx

        filtered = [r for r in results if r.get("score", 0) >= self._min_score]
        if not filtered and results:
            ctx.reranked_results = [results[0]]  # 至少保留1个
        else:
            ctx.reranked_results = filtered[:ctx.top_k]

        ctx.stage_times[self.name] = time.perf_counter() - start
        return ctx


# ========================================================
# LLM生成阶段 Adapter
# ========================================================

class LLMGenerationAdapter:
    """LLM生成阶段 Adapter

    包装 LLMRouter，根据检索结果生成答案。
    """
    name = "llm_generation"

    def __init__(self, llm_router):
        self._router = llm_router

    async def __call__(self, ctx: PipelineContext) -> PipelineContext:
        start = time.perf_counter()
        results = ctx.reranked_results or []
        context = [r["text"] for r in results] if results else []
        q = ctx.normalized_query or ctx.question

        try:
            ctx.answer = await self._router.generate(
                query=q, context=context, history=ctx.history
            )
        except Exception as e:
            logger.error(f"[{self.name}] 生成失败: {e}")
            ctx.stage_errors[self.name] = str(e)
            raise

        ctx.stage_times[self.name] = time.perf_counter() - start
        return ctx


# ========================================================
# 来源构建阶段 Adapter（输出整理）
# ========================================================

class SourcesBuilder:
    """引用来源整理 Adapter

    从检索结果中提取简洁的引用来源列表，用于前端展示。
    这是管道的最后一步，不改变业务数据，只做格式整理。

    对比 HiveWard 设计：
      HiveWard 的 History 页面也做类似的事——把运行轨迹整理为可审计的记录。
      我们的 SourcesBuilder 是把检索结果整理为前端可展示的引用卡片。
    """
    name = "sources_builder"

    async def __call__(self, ctx: PipelineContext) -> PipelineContext:
        start = time.perf_counter()
        results = ctx.reranked_results or []
        ctx.sources = [
            {
                "text": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                "page": r["page_start"],
                "section": r["section"],
                "content_type": r["content_type"],
                "score": round(r["score"], 4),
            }
            for r in results
        ] if results else []
        ctx.stage_times[self.name] = time.perf_counter() - start
        return ctx
