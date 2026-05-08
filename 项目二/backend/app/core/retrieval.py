"""
backend/app/core/retrieval.py — 检索服务（智能集成版）
======================================================
作用：封装完整的RAG流程，自动探测可用服务，智能判断何时启用增强功能
原理：
  1. 启动时自动探测以下服务的可用性：
     - Query理解（轻量规则引擎，零依赖 → 默认启用）
     - Redis缓存（尝试连接，能连上就启用）
     - BM25索引（从Milvus拉取文档构建，超过阈值则跳过）
     - CrossEncoder重排序（模型能加载就启用）
  2. 运行时智能判断：
     每个增强功能独立探测，独立降级，互不影响
  3. 故障自动降级：
     任一功能在执行中报错 → 仅该功能跳过，整个流程继续

功能自动探测逻辑（无需手动配置）：
  - Query理解：轻量规则引擎，无外部依赖，只要 jieba 安装就能用 → 自动启用
  - 缓存：启动时尝试连接 Redis → 能连上就启用，连不上就跳过
  - BM25混合检索：启动时从 Milvus 拉取文档构建 BM25 索引
    → 文档数 < HYBRID_MAX_DOCS 则启用，否则跳过（避免内存爆炸）
  - 重排序：启动时尝试加载 CrossEncoder 模型
    → 模型加载成功则启用，加载失败则跳过（不会阻塞启动）
"""

import hashlib
import logging
from typing import List, Dict, Any, Optional

from ..config import settings                            # 全局配置
from .embedding_service import get_embedding_service   # 文本转向量
from ..db.milvus_client import get_milvus_client       # Milvus向量检索
from .llm_router import get_llm_router                 # 智能分流（SGLang/Ollama自动选）
from ..exceptions import BusinessException, ErrorCode  # 自定义异常

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    作用：检索服务，封装完整的RAG流程
    原理：启动时自动探测所有可用服务，运行时根据探测结果智能启用/跳过
          Query → [缓存] → [Query理解] → [编码] → [检索(向量→混合)] → [重排序]
          → [LLM生成] → 返回
          所有功能"能用就用，不能用就跳过"，无需手动配置
    """

    def __init__(self):
        """
        作用：初始化检索服务，自动探测所有可用功能
        原理：
          1. 加载必须组件（Embedding / Milvus / LLM）
          2. 逐个探测可选组件，记录其可用性
          3. 不可用的组件在查询时自动跳过
        """
        # =========================================================
        # 必须组件（没有它们RAG无法运行）
        # =========================================================
        self.embedding_service = get_embedding_service()
        self.milvus_client = get_milvus_client()
        self.llm_router = get_llm_router()

        # =========================================================
        # 可选组件状态字典
        # =========================================================
        self.features = {
            "query_understanding": False,   # Query理解
            "cache": False,                 # Redis缓存
            "hybrid_retrieval": False,      # BM25混合检索
            "reranker": False,              # CrossEncoder重排序
        }

        # 缓存实例（只在探测成功时创建）
        self._redis_cache = None
        self.query_cache = None
        self.query_understanding = None
        self.hybrid_retriever = None
        self.reranker = None

        # =========================================================
        # 自动探测：Query理解（轻量规则引擎，零外部依赖）
        # =========================================================
        self._probe_query_understanding()

        # =========================================================
        # 自动探测：Redis缓存
        # =========================================================
        self._probe_cache()

        # =========================================================
        # 自动探测：BM25混合检索
        # =========================================================
        self._probe_hybrid_retrieval()

        # =========================================================
        # 自动探测：CrossEncoder重排序
        # =========================================================
        self._probe_reranker()

        # =========================================================
        # 汇总打印
        # =========================================================
        enabled = [name for name, ok in self.features.items() if ok]
        if enabled:
            logger.info(f"🔧 增强功能已启用: {', '.join(enabled)}")
        else:
            logger.info("🔧 无增强功能启用（仅基础RAG流程）")

    def _probe_query_understanding(self):
        """
        作用：自动探测Query理解功能是否可用
        原理：QueryUnderstandingService 是纯规则引擎（jieba + 正则），
              没有外部依赖，只要能 import jieba 就能用 → 直接启用
        逻辑：尝试导入 get_query_understanding_service，成功则创建实例
        """
        try:
            from .query_understanding import get_query_understanding_service
            self.query_understanding = get_query_understanding_service()
            self.features["query_understanding"] = True
            logger.info("✅ Query理解已启用（轻量规则引擎，零依赖）")
        except Exception as e:
            logger.warning(f"⚠️ Query理解不可用（不影响主流程）: {e}")

    def _probe_cache(self):
        """
        作用：自动探测Redis缓存是否可用
        原理：尝试连接Redis（超时3秒），能连上就启用
        """
        try:
            from ..db.redis_client import RedisCache
            redis_test = RedisCache(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD or None,
            )
            if redis_test.is_connected():
                self._redis_cache = redis_test
                self.features["cache"] = True
                logger.info(f"✅ Redis缓存已启用（{settings.REDIS_HOST}:{settings.REDIS_PORT}）")
            else:
                logger.info("ℹ️ Redis缓存未启用（Redis服务未运行）")
        except Exception as e:
            logger.info(f"ℹ️ Redis缓存未启用: {e}")

    def _probe_hybrid_retrieval(self):
        """
        作用：探测 Milvus 中文档数量，判断是否适合启用 BM25 混合检索
        原理：只检查文档数量，不实际构建 BM25 索引
              索引构建和实际检索在查询时延迟加载
        """
        try:
            stats = self.milvus_client.get_collection_stats()
            num_entities = stats.get("num_entities", 0)

            if num_entities == 0:
                logger.info("ℹ️ BM25混合检索跳过：Milvus集合为空")
                return

            if isinstance(num_entities, int) and num_entities > settings.HYBRID_MAX_DOCS:
                logger.info(
                    f"ℹ️ BM25混合检索跳过：文档数 {num_entities} > "
                    f"阈值 {settings.HYBRID_MAX_DOCS}"
                )
                return

            # 文档数量合理，标记为可用（延迟加载，查询时再构建索引）
            self.hybrid_retriever = None
            self.features["hybrid_retrieval"] = True
            logger.info(
                f"✅ BM25混合检索已就绪（延迟加载，{num_entities} 篇文档，首次查询时构建索引）"
            )

        except Exception as e:
            logger.info(f"ℹ️ BM25混合检索未启用: {e}")

    def _probe_reranker(self):
        """
        作用：自动探测CrossEncoder重排序是否可用
        原理：只检查本地模型文件是否存在，不实际加载模型
              实际加载延迟到第一次查询需要时
        逻辑：
          1. 检查 Modelscope / HuggingFace 缓存目录
          2. 存在则标记为"可用但不加载"
          3. 不存在则跳过，给出提示
          4. 实际加载在第一次调用时触发（见 _lazy_load_reranker）
        """
        try:
            import os
            cache_candidates = [
                os.path.expanduser("~/.cache/modelscope/hub/models/BAAI/bge-reranker-base"),
                os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-reranker-base"),
            ]

            self._reranker_path = None
            for cache_path in cache_candidates:
                if os.path.exists(cache_path):
                    self._reranker_path = cache_path
                    logger.info(f"发现重排序模型缓存: {cache_path}")
                    break

            if self._reranker_path is None:
                logger.info(
                    "ℹ️ CrossEncoder重排序未启用：未找到本地模型缓存"
                )
                return

            # 只标记可用，不实际加载模型
            self.reranker = None  # 占位，首次查询时才加载
            self.features["reranker"] = True
            logger.info(
                "✅ Reranker重排序已就绪（延迟加载，首次查询时加载模型）"
            )

        except Exception as e:
            logger.info(f"ℹ️ Reranker重排序探测异常（跳过）: {e}")

    def _lazy_load_reranker(self) -> bool:
        """
        作用：延迟加载重排序模型，仅在首次需要时加载
        原理：第一次调用时创建 RerankerService 实例（加载模型），
              后续直接复用已有实例
        返回：
            True=模型可用，False=加载失败
        """
        if self.reranker is not None:
            return True
        try:
            from .reranker import get_reranker_service
            reranker = get_reranker_service()
            if reranker.reranker.model is not None:
                self.reranker = reranker
                logger.info("✅ Reranker重排序模型加载完成（延迟加载）")
                return True
            else:
                logger.warning("Reranker模型加载失败，回退到原始排序")
                self.features["reranker"] = False
                return False
        except Exception as e:
            logger.warning(f"Reranker模型延迟加载失败: {e}")
            self.features["reranker"] = False
            return False

    def _generate_cache_key(self, question: str, content_type: Optional[str], top_k: int) -> str:
        """
        作用：生成缓存的key
        原理：对 question + content_type + top_k 做MD5哈希
        """
        raw = f"{question}|{content_type}|{top_k}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        作用：查询缓存
        原理：如果缓存命中，直接跳过检索和LLM生成
        """
        if not self.features["cache"] or not self._redis_cache:
            return None
        try:
            cached = self._redis_cache.get(cache_key)
            if cached:
                logger.info("⚡ 缓存命中，直接返回缓存结果")
                return cached.get("result") if isinstance(cached, dict) else cached
        except Exception as e:
            logger.warning(f"缓存查询异常（跳过）: {e}")
        return None

    def _write_cache(self, cache_key: str, result: Dict[str, Any]):
        """
        作用：写入缓存
        """
        if not self.features["cache"] or not self._redis_cache:
            return
        try:
            self._redis_cache.set(
                cache_key, result, ttl=settings.CACHE_TTL_SECONDS
            )
            logger.debug("结果已缓存")
        except Exception as e:
            logger.warning(f"缓存写入异常（跳过）: {e}")

    def query(
        self,
        question: str,
        top_k: int = settings.RETRIEVAL_TOP_K,
        content_type: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        作用：执行完整的RAG查询（智能集成版）
        原理：根据自动探测结果，智能启用/跳过各增强功能
        参数：
            question: 用户问题
            top_k: 检索返回多少条chunk
            content_type: 可选，按内容类型过滤
            history: 可选，对话历史
        返回：
            {"answer": "...", "sources": [...], "retrieval_results": [...]}
        """
        try:
            logger.info(f"开始查询: {question}")

            # =========================================================
            # 步骤0：缓存查询（智能判断）
            # =========================================================
            cache_key = self._generate_cache_key(question, content_type, top_k)
            cached = self._check_cache(cache_key)
            if cached:
                return cached

            # =========================================================
            # 步骤0.5：Query理解（智能判断）
            # =========================================================
            effective_question = question
            effective_history = history

            if self.features["query_understanding"] and self.query_understanding:
                try:
                    understanding = self.query_understanding.understand(
                        question, conversation_history=history
                    )
                    if understanding.needs_clarification:
                        logger.info(f"需要澄清: {understanding.clarification_prompt}")
                        return {
                            "answer": understanding.clarification_prompt or "请明确您的问题",
                            "sources": [],
                            "retrieval_results": [],
                        }
                    effective_question = understanding.normalized_query
                    if understanding.sub_queries and len(understanding.sub_queries) > 1:
                        effective_question = understanding.sub_queries[0]
                    logger.info(
                        f"意图: {understanding.intent.value} | "
                        f"实体: {[e.text for e in understanding.entities]}"
                    )
                except Exception as e:
                    logger.warning(f"Query理解异常（跳过）: {e}")

            # =========================================================
            # 步骤1：文本编码
            # =========================================================
            embedding = self.embedding_service.encode(effective_question)

            # =========================================================
            # 步骤2：向量检索
            # =========================================================
            vector_results = self.milvus_client.search(
                embedding=embedding,
                top_k=top_k,
                content_type=content_type
            )

            # =========================================================
            # 步骤2.5：混合检索（智能判断）——向量+BM25，RRF融合
            # =========================================================
            if self.features["hybrid_retrieval"] and self.hybrid_retriever:
                try:
                    retrieval_results = self.hybrid_retriever.search(
                        query=effective_question,
                        vector_results=[
                            (r, r.get("score", 0.0)) for r in vector_results
                        ],
                        top_k=top_k
                    )
                    retrieval_results = [
                        {**doc, "score": score} for doc, score in retrieval_results
                    ]
                except Exception as e:
                    logger.warning(f"混合检索异常（回退到纯向量检索）: {e}")
                    retrieval_results = vector_results
            else:
                retrieval_results = vector_results

            if not retrieval_results:
                logger.warning("未检索到相关内容")
                raise BusinessException(
                    ErrorCode.RETRIEVAL_EMPTY, "未找到与问题相关的信息"
                )

            logger.info(f"检索到 {len(retrieval_results)} 条结果")

            # =========================================================
            # 步骤2.8：重排序（智能判断）——CrossEncoder精排
            # 延迟加载：第一次需要时加载模型，后续复用
            # =========================================================
            if self.features["reranker"] and self._lazy_load_reranker():
                try:
                    rerank_docs = [
                        {**r, "content": r.get("text", "")}
                        for r in retrieval_results
                    ]
                    reranked = self.reranker.rerank_documents(
                        query=effective_question,
                        documents=rerank_docs,
                        top_k=top_k
                    )
                    retrieval_results = [
                        {
                            k: v for k, v in doc.items()
                            if k not in ("content", "rerank_score")
                        }
                        for doc in reranked
                    ]
                    for i, doc in enumerate(reranked):
                        if i < len(retrieval_results):
                            retrieval_results[i]["rerank_score"] = doc.get(
                                "rerank_score", 0.0
                            )
                except Exception as e:
                    logger.warning(f"重排序异常（回退到原始排序）: {e}")

            # =========================================================
            # 步骤3：构建上下文
            # =========================================================
            context = [r["text"] for r in retrieval_results]

            # =========================================================
            # 步骤4：智能生成答案
            # =========================================================
            answer = self.llm_router.generate(
                query=effective_question,
                context=context,
                history=effective_history
            )

            # =========================================================
            # 步骤5：组装结果
            # =========================================================
            sources = [
                {
                    "text": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                    "page": r["page_start"],
                    "section": r["section"],
                    "content_type": r["content_type"],
                    "score": round(r["score"], 4),
                }
                for r in retrieval_results
            ]

            result = {
                "answer": answer,
                "sources": sources,
                "retrieval_results": retrieval_results,
            }

            # =========================================================
            # 步骤6：写入缓存（智能判断）
            # =========================================================
            self._write_cache(cache_key, result)

            return result

        except BusinessException:
            raise
        except Exception as e:
            logger.error(f"查询失败: {e}")
            raise BusinessException(
                ErrorCode.LLM_GENERATION_FAILED, str(e)
            )

    def query_stream(
        self,
        question: str,
        top_k: int = settings.RETRIEVAL_TOP_K,
        content_type: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ):
        """
        作用：流式查询（逐步返回答案token）
        原理：和query()流程一致，但流式返回（不缓存流式结果）
        """
        try:
            logger.info(f"开始流式查询: {question}")

            # 步骤0.5：Query理解（智能判断）
            effective_question = question
            effective_history = history
            if self.features["query_understanding"] and self.query_understanding:
                try:
                    understanding = self.query_understanding.understand(
                        question, conversation_history=history
                    )
                    if understanding.needs_clarification:
                        yield understanding.clarification_prompt or "请明确您的问题"
                        return
                    effective_question = understanding.normalized_query
                    if understanding.sub_queries and len(understanding.sub_queries) > 1:
                        effective_question = understanding.sub_queries[0]
                except Exception:
                    pass

            # 步骤1：编码
            embedding = self.embedding_service.encode(effective_question)

            # 步骤2：检索
            vector_results = self.milvus_client.search(
                embedding=embedding, top_k=top_k, content_type=content_type
            )

            # 步骤2.5：混合检索
            if self.features["hybrid_retrieval"] and self.hybrid_retriever:
                try:
                    retrieval_results = self.hybrid_retriever.search(
                        query=effective_question,
                        vector_results=[
                            (r, r.get("score", 0.0)) for r in vector_results
                        ],
                        top_k=top_k
                    )
                    retrieval_results = [
                        {**doc, "score": score} for doc, score in retrieval_results
                    ]
                except Exception:
                    retrieval_results = vector_results
            else:
                retrieval_results = vector_results

            if not retrieval_results:
                yield "未找到与问题相关的信息"
                return

            # 步骤2.8：重排序
            if self.features["reranker"] and self._lazy_load_reranker():
                try:
                    rerank_docs = [
                        {**r, "content": r.get("text", "")}
                        for r in retrieval_results
                    ]
                    reranked = self.reranker.rerank_documents(
                        query=effective_question, documents=rerank_docs, top_k=top_k
                    )
                    retrieval_results = [
                        {k: v for k, v in doc.items()
                         if k not in ("content", "rerank_score")}
                        for doc in reranked
                    ]
                except Exception:
                    pass

            # 步骤3：上下文
            context = [r["text"] for r in retrieval_results]

            # 步骤4：流式生成
            yield from self.llm_router.generate_stream(
                query=effective_question, context=context, history=effective_history
            )

        except Exception as e:
            logger.error(f"流式查询失败: {e}")
            yield f"生成失败: {str(e)}"

    def get_feature_status(self) -> Dict[str, Dict[str, Any]]:
        """
        作用：获取各功能的当前状态，方便调试和监控
        返回：
            {
                "query_understanding": {"enabled": true/false, "reason": "..."},
                "cache": {"enabled": true/false, "reason": "..."},
                ...
            }
        """
        reasons = {
            "query_understanding": (
                "轻量规则引擎，已就绪" if self.features["query_understanding"]
                else "不可用（缺少jieba依赖）"
            ),
            "cache": (
                f"Redis已连接 ({settings.REDIS_HOST}:{settings.REDIS_PORT})"
                if self.features["cache"]
                else "不可用（Redis服务未运行）"
            ),
            "hybrid_retrieval": (
                "BM25索引已构建，支持向量+BM25融合" if self.features["hybrid_retrieval"]
                else "不可用（文档数过多或Milvus为空）"
            ),
            "reranker": (
                "CrossEncoder模型已加载" if self.features["reranker"]
                else "不可用（模型加载失败或未安装依赖）"
            ),
        }
        return {
            name: {"enabled": ok, "reason": reasons.get(name, "")}
            for name, ok in self.features.items()
        }


# 全局服务实例（单例模式）
_retrieval_service = None


def get_retrieval_service() -> RetrievalService:
    """
    作用：获取检索服务单例
    原理：第一次调用时创建（触发自动探测），后续直接返回已有实例
    """
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service
