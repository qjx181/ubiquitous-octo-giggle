# -*- coding: utf-8 -*-
"""示例：如何使用 DAG 执行引擎处理对比分析类查询

场景：用户问"对比腾讯和阿里的营收"
  → Intent: COMPARISON
  → DAG 路由：
    ├─ 分支1：检索腾讯营收数据（并行）
    └─ 分支2：检索阿里营收数据（并行）
  → Join：汇聚两个分支结果
  → LLMGeneration：生成对比分析
"""

import asyncio
from backend.app.core.pipeline import (
    DAGPipeline, DAGNode, NodeType, DAGBuilder,
    PipelineContext, check_comparison_intent
)


# 假设我们有以下 Adapter（实际实现需要根据项目代码）
class QueryUnderstandingAdapter:
    """Query 理解 Adapter"""
    def __init__(self, qu_service):
        self.qu_service = qu_service
    
    async def __call__(self, ctx: PipelineContext) -> PipelineContext:
        # 调用 Query 理解服务
        result = self.qu_service.understand(ctx.question)
        ctx.normalized_query = result.normalized_query
        ctx.sub_queries = result.sub_queries
        return ctx


class RetrievalAdapter:
    """检索 Adapter"""
    def __init__(self, retrieval_service, entity_filter=None):
        self.retrieval_service = retrieval_service
        self.entity_filter = entity_filter
    
    async def __call__(self, ctx: PipelineContext) -> PipelineContext:
        # 执行检索
        query = ctx.normalized_query or ctx.question
        if self.entity_filter:
            query = f"{self.entity_filter} {query}"
        results = self.retrieval_service.search(query, top_k=ctx.top_k)
        ctx.vector_results = results
        return ctx


class LLMGenerationAdapter:
    """LLM 生成 Adapter"""
    def __init__(self, llm_router):
        self.llm_router = llm_router
    
    async def __call__(self, ctx: PipelineContext) -> PipelineContext:
        # 生成答案
        context = "\n".join([r.get("content", "") for r in (ctx.vector_results or [])])
        prompt = f"基于以下信息回答问题：\n{context}\n\n问题：{ctx.question}"
        ctx.answer = self.llm_router.generate(prompt)
        return ctx


async def example_comparison_query():
    """示例：对比分析类查询"""
    
    # 1. 创建服务实例（假设已经初始化）
    # qu_service = QueryUnderstandingService()
    # retrieval_service = RetrievalService()
    # llm_router = LLMRouter()
    
    # 2. 创建 Adapter
    # qu_adapter = QueryUnderstandingAdapter(qu_service)
    # retrieval_tencent = RetrievalAdapter(retrieval_service, entity_filter="腾讯")
    # retrieval_alibaba = RetrievalAdapter(retrieval_service, entity_filter="阿里")
    # llm_adapter = LLMGenerationAdapter(llm_router)
    
    # 3. 构建 DAG
    # 方式一：手动构建
    # root = DAGNode("query_understanding", NodeType.STAGE, stage=qu_adapter)
    # router = DAGNode("comparison_router", NodeType.ROUTER, condition=check_comparison_intent)
    # branch1 = DAGNode("retrieval_tencent", NodeType.STAGE, stage=retrieval_tencent)
    # branch2 = DAGNode("retrieval_alibaba", NodeType.STAGE, stage=retrieval_alibaba)
    # join = DAGNode("join", NodeType.JOIN)
    # llm = DAGNode("llm_generation", NodeType.STAGE, stage=llm_adapter)
    #
    # root.add_child(router)
    # router.add_children([branch1, branch2])
    # branch1.add_child(join)
    # branch2.add_child(join)
    # join.add_child(llm)
    #
    # pipeline = DAGPipeline(root)
    
    # 方式二：使用 DAGBuilder（更简洁）
    # builder = DAGBuilder()
    # builder.add_stage("query_understanding", qu_adapter)
    #        .add_router("comparison_router", check_comparison_intent)
    #        .add_fork("parallel_retrieval")
    #        .add_stage("retrieval_tencent", retrieval_tencent)
    #        .add_stage("retrieval_alibaba", retrieval_alibaba)
    #        .end_fork()
    #        .end_router()
    #        .add_stage("llm_generation", llm_adapter)
    #
    # pipeline = builder.build()
    
    # 4. 执行
    # ctx = PipelineContext(question="对比腾讯和阿里的营收", top_k=3)
    # ctx = await pipeline.run(ctx)
    #
    # print(f"答案: {ctx.answer}")
    # print(f"阶段耗时: {ctx.stage_times}")
    
    print("示例代码（需要实际服务才能运行）")


async def example_with_state_machine():
    """示例：结合状态机使用"""
    
    # from backend.app.core.pipeline import create_state_machine, PipelineState
    #
    # sm = create_state_machine()
    #
    # # 设置回调
    # sm.on_state_change = lambda old, new: print(f"状态变化: {old} → {new}")
    # sm.on_rollback = lambda old, target: print(f"回滚: {old} → {target}")
    #
    # # 执行 Pipeline
    # ctx = PipelineContext(question="对比腾讯和阿里的营收")
    #
    # try:
    #     # Query 理解
    #     sm.transition(PipelineState.QUERY_UNDERSTANDING, ctx)
    #     ctx = await qu_adapter(ctx)
    #
    #     # Embedding
    #     sm.transition(PipelineState.EMBEDDING, ctx)
    #     ctx = await embedding_adapter(ctx)
    #
    #     # 检索
    #     sm.transition(PipelineState.RETRIEVAL, ctx)
    #     ctx = await retrieval_adapter(ctx)
    #
    #     # 重排序
    #     sm.transition(PipelineState.RERANKING, ctx)
    #     ctx = await rerank_adapter(ctx)
    #
    #     # LLM 生成
    #     sm.transition(PipelineState.LLM_GENERATION, ctx)
    #     ctx = await llm_adapter(ctx)
    #
    #     # 完成
    #     sm.transition(PipelineState.COMPLETED, ctx)
    #
    # except Exception as e:
    #     # 失败时回滚
    #     sm.transition(PipelineState.FAILED, ctx)
    #     ctx = sm.rollback(PipelineState.RETRIEVAL)
    #     # 重试检索
    #     ctx = await retrieval_adapter(ctx)
    #
    # # 获取进度
    # progress = sm.get_progress()
    # print(f"进度: {progress}")
    
    print("示例代码（需要实际服务才能运行）")


async def example_with_query_learner():
    """示例：结合查询学习器使用"""
    
    # from backend.app.core.evolution import get_query_learner
    #
    # learner = get_query_learner("path/to/history.json")
    #
    # # 记录查询
    # learner.record_query(
    #     query="腾讯营收是多少",
    #     normalized_query="腾讯营收",
    #     intent="financial",
    #     intent_confidence=0.85,
    #     entities=[{"type": "ORG", "text": "腾讯"}],
    #     retrieval_results=[...],
    #     answer="腾讯2023年营收...",
    #     user_feedback=4.5
    # )
    #
    # # 获取意图提示
    # hints = learner.get_intent_hints("阿里营收")
    # print(f"意图提示: {hints}")
    # # 输出: {"financial": 0.15}
    #
    # # 获取检索参数
    # params = learner.get_retrieval_params("financial")
    # print(f"检索参数: {params}")
    # # 输出: {"top_k": 5, "threshold": 0.6}
    #
    # # 获取实体建议
    # suggestions = learner.get_entity_suggestions("ORG", "腾")
    # print(f"实体建议: {suggestions}")
    # # 输出: ["腾讯", "腾讯音乐", "腾讯云"]
    #
    # # 获取统计信息
    # stats = learner.get_stats()
    # print(f"统计: {stats}")
    
    print("示例代码（需要实际服务才能运行）")


if __name__ == "__main__":
    asyncio.run(example_comparison_query())
    asyncio.run(example_with_state_machine())
    asyncio.run(example_with_query_learner())
