"""
backend/app/dependencies.py — FastAPI依赖注入
<<<<<<< HEAD
==============================================
职责：提供FastAPI的依赖注入函数

功能：
  1. 获取检索服务实例
  2. 获取Embedding服务实例
  3. 获取Milvus客户端实例
  4. 获取LLM智能调度器实例

原理：
  FastAPI的依赖注入系统可以自动管理依赖的生命周期
  使用Depends()装饰器，FastAPI会自动调用这些函数并注入结果
=======
===============================================
作用：集中导出所有的单例获取函数，方便API层使用
原理：FastAPI的依赖注入系统可以自动管理依赖的生命周期
      使用Depends()装饰器，FastAPI会自动调用这些函数并注入结果

好处：
  - API层不需要关心服务怎么创建，只需要调用
  - 统一管理所有单例的获取方式
  - 减少重复导入
>>>>>>> 6d37b33 (提交信息)

使用示例：
  @router.post("/chat")
  async def chat(request: ChatRequest, service: RetrievalService = Depends(get_retrieval_service)):
      result = service.query(request.question)
      return result
"""

from .core.retrieval import RetrievalService, get_retrieval_service
from .core.embedding_service import EmbeddingService, get_embedding_service
<<<<<<< HEAD
from .db.milvus_client import MilvusClient, get_milvus_client
from .core.llm_router import LLMRouter, get_llm_router


=======
from .db.milvus_client import MilvusClientWrapper, get_milvus_client
from .core.llm_router import LLMRouter, get_llm_router
from .core.query_understanding import QueryUnderstandingService, get_query_understanding_service
from .core.reranker import RerankerService, get_reranker_service
from .core.bm25_retriever import BM25Retriever, get_bm25_retriever, HybridRetriever, get_hybrid_retriever
from .db.redis_client import RedisCache, SessionManager, QueryCache, get_redis_cache, get_session_manager, get_query_cache


# __all__的作用：from .dependencies import * 时只导入这些
# 明确告诉其他模块"我们提供了哪些服务"
>>>>>>> 6d37b33 (提交信息)
__all__ = [
    "get_retrieval_service",
    "get_embedding_service",
    "get_milvus_client",
    "get_llm_router",
<<<<<<< HEAD
    "RetrievalService",
    "EmbeddingService",
    "MilvusClient",
    "LLMRouter",
=======
    "get_query_understanding_service",
    "get_reranker_service",
    "get_bm25_retriever",
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
    "BM25Retriever",
    "HybridRetriever",
    "RedisCache",
    "SessionManager",
    "QueryCache",
>>>>>>> 6d37b33 (提交信息)
]
