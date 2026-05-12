# -*- coding: utf-8 -*-
"""检查所有模块能否正常导入"""
import sys
import os

# 添加backend目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

errors = []

# 检查Query理解模块
try:
    from app.core.query_understanding import QueryUnderstandingService, IntentType, Entity, QueryUnderstandingResult
    print("✅ query_understanding.py - OK")
except Exception as e:
    errors.append(f"❌ query_understanding.py - {e}")

# 检查BM25模块
try:
    from app.core.bm25_retriever import BM25Retriever, HybridRetriever
    print("✅ bm25_retriever.py - OK")
except Exception as e:
    errors.append(f"❌ bm25_retriever.py - {e}")

# 检查重排序模块
try:
    from app.core.reranker import CrossEncoderReranker, RerankerService, RerankResult
    print("✅ reranker.py - OK")
except Exception as e:
    errors.append(f"❌ reranker.py - {e}")

# 检查认证模块
try:
    from app.core.auth import JWTAuth, UserRole, User, PermissionChecker, UserManager
    print("✅ auth.py - OK")
except Exception as e:
    errors.append(f"❌ auth.py - {e}")

# 检查Redis模块
try:
    from app.db.redis_client import RedisCache, SessionManager, QueryCache, CacheManager
    print("✅ redis_client.py - OK")
except Exception as e:
    errors.append(f"❌ redis_client.py - {e}")

# 检查其他核心模块
try:
    from app.core.embedding_service import EmbeddingService
    print("✅ embedding_service.py - OK")
except Exception as e:
    errors.append(f"❌ embedding_service.py - {e}")

try:
    from app.core.llm_client import LLMClient
    print("✅ llm_client.py - OK")
except Exception as e:
    errors.append(f"❌ llm_client.py - {e}")

try:
    from app.core.llm_router import LLMRouter
    print("✅ llm_router.py - OK")
except Exception as e:
    errors.append(f"❌ llm_router.py - {e}")

try:
    from app.core.retrieval import RetrievalService
    print("✅ retrieval.py - OK")
except Exception as e:
    errors.append(f"❌ retrieval.py - {e}")

try:
    from app.db.milvus_client import MilvusClient
    print("✅ milvus_client.py - OK")
except Exception as e:
    errors.append(f"❌ milvus_client.py - {e}")

# 输出结果
print("\n" + "="*50)
if errors:
    print("检查失败:")
    for err in errors:
        print(f"  {err}")
else:
    print("所有模块检查通过！")
