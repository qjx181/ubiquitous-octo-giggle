"""pipeline/pipeline.py — Pipeline 编排器

作用：串联多个管道阶段，按顺序执行，提供统一的日志和监控接口。

设计思路（借鉴 HiveWard Blueprint 的节点编排思想）：
  HiveWard 的 Blueprint 画布上，节点按连线顺序执行：
    CEO → Leader → Blueprint → Harness Agent → Run → Approval → History
  我们的 Pipeline 做类似的事——阶段按顺序张成一条线：
    QueryUnderstanding → Embedding → VectorSearch → HybridSearch → Reranker → ScoreFilter → LLMGeneration → SourcesBuilder

  区别：
    HiveWard 的 Blueprint 支持 DAG（并行、条件分支、审批暂停）
    我们的 Pipeline 目前只支持线性顺序（串行执行）

  未来可以借鉴 HiveWard 的 DAG 设计：
    - Manager 节点 → 条件路由到不同 Slot
    - Slot 节点 → 并行执行
    - 审批节点 → 人工确认后继续
    但目前 RAG 管道的线性特征已经足够。

核心能力：
  1. 阶段注册（按顺序添加）
  2. 执行编排（遍历阶段，前一个的输出是后一个的输入）
  3. 错误隔离（某阶段失败可以跳过或降级）
  4. 耗时统计（每个阶段独立计时，方便定位瓶颈）

面试官可能问：
  Q: Pipeline 和直接顺序调用的区别是什么？
  A: 直接调用：retrieval.py 的 _run_retrieval_pipeline 里写了完整的调用链，
     各阶段耦合在一起。Pipeline 模式：
     1. 每个阶段独立，可以单独测试
     2. 添加/移除/替换阶段不需要改 Pipeline 代码
     3. 可以插桩监控（每个阶段的耗时、错误数自动记录）
     4. 未来可以支持条件跳过（比如澄清后不继续检索）

  Q: 为什么不直接用 LangChain 的 LCEL？
  A: LCEL 功能更强大，但代价是依赖 LangChain 生态。
     项目当前依赖自定义实现（没有用 LangChain），为了一件事引入整个框架
     得不偿失。轻量 Pipeline 实现成本低、可控性强。
"""

import logging
import time
from typing import List, Optional

from .interfaces import PipelineContext, PipelineStage

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """RAG 检索管道编排器

    使用方式：
        pipeline = RetrievalPipeline()
        pipeline.add_stage(QueryUnderstandingAdapter(qu))
        pipeline.add_stage(EmbeddingAdapter(emb))
        pipeline.add_stage(VectorSearchAdapter(mv))
        pipeline.add_stage(HybridSearchAdapter(hybrid))
        pipeline.add_stage(RerankAdapter(reranker))
        pipeline.add_stage(ScoreThresholdAdapter(threshold))
        pipeline.add_stage(LLMGenerationAdapter(router))
        pipeline.add_stage(SourcesBuilder())

        ctx = PipelineContext(question="公司营收是多少？", top_k=3)
        ctx = await pipeline.run(ctx)

        # 结果在 ctx.answer, ctx.sources, ctx.stage_times 里

    设计对比 HiveWard：
      HiveWard 的 Blueprint 运行时做了类似的事——按连线顺序遍历节点，
      每个节点执行自己的逻辑，把结果传给下游。但 Blueprint 的节点更复杂：
      - 节点可以暂停（等待审批）
      - 可以执行并行分支（Manager 分发到多个 Slot）
      - 可以回滚（失败时回溯到上一个安全节点）
      我们的 Pipeline 目前只做"串行+降级"，是 Blueprint 的一个简化子集。
    """

    def __init__(self):
        self._stages: List[PipelineStage] = []
        self._stage_map: dict = {}  # name → stage 快速查找

    def add_stage(self, stage: PipelineStage) -> "RetrievalPipeline":
        """添加一个管道阶段

        Args:
            stage: 实现 PipelineStage 接口的对象（必须有 name 和 __call__）
        """
        self._stages.append(stage)
        self._stage_map[stage.name] = stage
        return self  # 链式调用

    def remove_stage(self, name: str) -> "RetrievalPipeline":
        """按名称移除一个阶段"""
        self._stages = [s for s in self._stages if s.name != name]
        self._stage_map.pop(name, None)
        return self

    def insert_after(self, after_name: str, stage: PipelineStage) -> "RetrievalPipeline":
        """在指定阶段后插入新阶段"""
        for i, s in enumerate(self._stages):
            if s.name == after_name:
                self._stages.insert(i + 1, stage)
                self._stage_map[stage.name] = stage
                return self
        raise ValueError(f"阶段 '{after_name}' 不存在")

    def insert_before(self, before_name: str, stage: PipelineStage) -> "RetrievalPipeline":
        """在指定阶段前插入新阶段"""
        for i, s in enumerate(self._stages):
            if s.name == before_name:
                self._stages.insert(i, stage)
                self._stage_map[stage.name] = stage
                return self
        raise ValueError(f"阶段 '{before_name}' 不存在")

    def get_stage_names(self) -> List[str]:
        """获取当前所有阶段的名称列表"""
        return [s.name for s in self._stages]

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """执行完整的管道

        Args:
            ctx: 管道上下文（包含输入问题等）

        Returns:
            执行完毕的 PipelineContext，包含所有阶段的输出
        """
        total_start = time.perf_counter()

        for stage in self._stages:
            # 如果 Clarification 已触发，跳过后续检索和生成阶段
            if ctx.clarification_prompt and stage.name not in (
                    "sources_builder", "query_understanding"):
                logger.info(f"[Pipeline] 需澄清，跳过阶段: {stage.name}")
                continue

            # 如果检索结果为空，跳过后续检索阶段但保留 LLM 生成（让 LLM 处理空上下文）
            if stage.name == "llm_generation" and not ctx.vector_results:
                logger.info("[Pipeline] 检索结果为空，仍执行 LLM 生成（空上下文）")
            elif stage.name in ("hybrid_search", "reranker", "score_filter") and not (ctx.vector_results or ctx.hybrid_results):
                logger.info(f"[Pipeline] 无检索结果，跳过阶段: {stage.name}")
                continue

            try:
                ctx = await stage(ctx)
                logger.debug(f"[Pipeline] {stage.name} 完成, "
                             f"耗时 {ctx.stage_times.get(stage.name, 0):.3f}s")
            except Exception as e:
                logger.error(f"[Pipeline] {stage.name} 阶段崩溃: {e}")
                ctx.stage_errors[stage.name] = str(e)
                # 致命阶段（Embedding, LLM生成）失败，终止管道
                if stage.name in ("embedding", "llm_generation"):
                    raise
                # 非致命阶段失败，继续下一个阶段
                continue

        ctx.total_time = time.perf_counter() - total_start

        # 记录管道摘要
        stage_summary = ", ".join(
            f"{s.name}={ctx.stage_times.get(s.name, 0):.3f}s"
            for s in self._stages
        )
        logger.info(f"[Pipeline] 完成 (总耗时 {ctx.total_time:.3f}s): {stage_summary}")

        return ctx


def build_default_pipeline(
    query_understanding=None,
    embedding_service=None,
    milvus_client=None,
    hybrid_retriever=None,
    reranker_service=None,
    min_score: float = 0.3,
    llm_router=None,
) -> RetrievalPipeline:
    """构建默认的 RAG 管道

    这是工厂函数，根据当前可用的服务组装 Pipeline。
    某个服务为 None 时，对应的 Adapter 会执行降级逻辑。

    对比 HiveWard 的 setup 流程：
      HiveWard 的 hiveward setup 做环境检测并生成配置。
      我们的 build_default_pipeline 做类似的事——根据探测结果组装适配的管道。
      不同之处：
      - HiveWard 检测 Agent 是否可用（OpenClaw/Codex/Claude Code）
      - 我们检测 RAG 组件是否可用（Reranker/混合检索/Redis）

    各阶段的依赖关系（有依赖的阶段放在后面）：
      1. query_understanding    — 无依赖
      2. embedding              — 依赖 normalized_query
      3. vector_search          — 依赖 query_vector
      4. hybrid_search          — 依赖 vector_results
      5. reranker               — 依赖 hybrid_results
      6. score_filter           — 依赖 reranked_results
      7. llm_generation         — 依赖 reranked_results
      8. sources_builder        — 依赖 reranked_results

    Args:
        query_understanding: Query 理解服务实例（None 则跳过）
        embedding_service: Embedding 服务实例（必须，否则管道无法运行）
        milvus_client: Milvus 客户端实例（必须）
        hybrid_retriever: 混合检索器实例（None 则跳过）
        reranker_service: Reranker 服务实例（None 则跳过）
        min_score: 最低分数阈值
        llm_router: LLM 路由器实例（必须）

    Returns:
        配置好的 RetrievalPipeline 实例
    """
    from .adapters import (
        QueryUnderstandingAdapter,
        EmbeddingAdapter,
        VectorSearchAdapter,
        HybridSearchAdapter,
        RerankAdapter,
        ScoreThresholdAdapter,
        LLMGenerationAdapter,
        SourcesBuilder,
    )

    pipeline = RetrievalPipeline()

    pipeline.add_stage(QueryUnderstandingAdapter(query_understanding))
    pipeline.add_stage(EmbeddingAdapter(embedding_service))
    pipeline.add_stage(VectorSearchAdapter(milvus_client))
    pipeline.add_stage(HybridSearchAdapter(hybrid_retriever))
    pipeline.add_stage(RerankAdapter(reranker_service))
    pipeline.add_stage(ScoreThresholdAdapter(min_score))
    pipeline.add_stage(LLMGenerationAdapter(llm_router))
    pipeline.add_stage(SourcesBuilder())

    return pipeline
