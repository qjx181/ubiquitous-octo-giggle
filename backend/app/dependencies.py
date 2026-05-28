"""
backend/app/dependencies.py — FastAPI依赖注入
===============================================
作用：集中导出所有的单例获取函数，方便API层使用
原理：FastAPI的依赖注入系统可以自动管理依赖的生命周期
      使用Depends()装饰器，FastAPI会自动调用这些函数并注入结果

好处：
  - API层不需要关心服务怎么创建，只需要调用
  - 统一管理所有单例的获取方式
  - 减少重复导入

使用示例：
  @router.post("/chat")
  async def chat(request: ChatRequest, service: RetrievalService = Depends(get_retrieval_service)):
      result = service.query(request.question)
      return result

面试官可能问：
  Q: FastAPI的Depends依赖注入有什么好处？
  A: (1) 自动管理依赖的生命周期，不需要手动new
     (2) 每个请求独立获取依赖，线程安全
     (3) 替换实现简单（如测试时Mock），改Depends参数即可

  Q: 为什么用__all__控制导出？
  A: __all__明确声明哪些是公共接口，IDE和静态检查工具可以识别。
     避免from .dependencies import *时意外导入内部模块。
     这是Python模块设计的良好实践。
"""

from .core.retrieval import RetrievalService, get_retrieval_service
from .core.embedding_service import EmbeddingService, get_embedding_service
from .db.milvus_client import MilvusClientWrapper, get_milvus_client
from .core.llm_router import LLMRouter, get_llm_router
from .core.query_understanding import QueryUnderstandingService, get_query_understanding_service
from .core.reranker import RerankerService, get_reranker_service
from .core.bm25_retriever import HybridRetriever, get_hybrid_retriever
from .db.redis_client import RedisCache, SessionManager, QueryCache, get_redis_cache, get_session_manager, get_query_cache


# __all__的作用：from .dependencies import * 时只导入这些
# 明确告诉其他模块"我们提供了哪些服务"
__all__ = [
    "get_retrieval_service",
    "get_embedding_service",
    "get_milvus_client",
    "get_llm_router",
    "get_query_understanding_service",
    "get_reranker_service",
    "get_hybrid_retriever",
    "get_redis_cache",
    "get_session_manager",
    "get_query_cache",
    "RetrievalService",
    "EmbeddingService",
    "MilvusClientWrapper",
    "LLMRouter",
    "QueryUnderstandingService",
    "RerankerService",
    "HybridRetriever",
    "RedisCache",
    "SessionManager",
    "QueryCache",
]
