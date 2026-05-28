"""pipeline — RAG管道阶段的Adapter模式实现

借鉴 HiveWard GatewayOpenClawAdapter 的设计模式：
  将 RAG 管道的每个阶段（Query理解、Embedding、检索、重排、生成）
  抽象为 Plugin 接口，可独立替换、测试、插桩。

使用方式：
  pipeline = build_default_pipeline(
      query_understanding=qu_service,
      embedding_service=emb_service,
      milvus_client=milvus_client,
      ...
  )
  ctx = PipelineContext(question="公司营收多少？")
  ctx = await pipeline.run(ctx)
  print(ctx.answer)
"""

from .interfaces import PipelineContext
from .pipeline import RetrievalPipeline, build_default_pipeline
from .dag_engine import DAGPipeline, DAGNode, NodeType, DAGBuilder, check_comparison_intent, check_multi_entity, check_high_confidence
from .state_machine import PipelineStateMachine, PipelineState, create_state_machine, with_state_machine

__all__ = [
    "PipelineContext", 
    "RetrievalPipeline", 
    "build_default_pipeline",
    "DAGPipeline",
    "DAGNode",
    "NodeType",
    "DAGBuilder",
    "check_comparison_intent",
    "check_multi_entity",
    "check_high_confidence",
    "PipelineStateMachine",
    "PipelineState",
    "create_state_machine",
    "with_state_machine"
]
