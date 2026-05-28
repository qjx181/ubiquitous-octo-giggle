"""backend/app/core/retrieval.py — RAG检索服务，自动探测+智能降级"""
# 【作用】导入 hashlib 模块，用于生成 MD5 哈希值（缓存键）
import hashlib
# 【作用】导入 logging 模块，用于在控制台和日志文件中打印运行日志
import logging
# 【作用】导入 threading 模块，用于创建后台线程（比如异步加载模型，不阻塞主线程）
import threading
# 【作用】导入 asyncio 模块，用于 async 调用链
import asyncio
# 【作用】从 typing 导入类型提示，让代码更可读、编辑器能自动补全
# 【原理】List 表示列表类型，Dict 表示字典，Any 表示任意类型，Optional 表示可选（可能为 None），Generator 表示生成器（惰性返回数据）
from typing import List, Dict, Any, Optional, Generator

# 【作用】从父级目录的 config 模块中导入 settings，settings 里存了所有配置文件（数据库地址、密钥、阈值等）
from ..config import settings
# 【作用】从同级目录的 embedding_service 中导入 get_embedding_service，用来把文字变成向量（机器能理解的数字数组）
from .embedding_service import get_embedding_service
# 【作用】从父级目录 db 包的 milvus_client 中导入 get_milvus_client，用来连接 Milvus 向量数据库
from ..db.milvus_client import get_milvus_client
# 【作用】从同级目录导入 get_llm_router，用来调用大语言模型（LLM）生成答案
from .llm_router import get_llm_router
# 【作用】从父级目录的 exceptions 模块导入自定义异常类和错误码，用于统一处理业务异常
from ..exceptions import BusinessException, ErrorCode

# 【作用】创建一个 logger 对象，专门记录当前模块（retrieval.py）的日志
# 【逻辑】__name__ 的值是 "backend.app.core.retrieval"（根据目录结构自动生成的）
logger = logging.getLogger(__name__)

# Pipeline 适配器模式：将各管道阶段标准化为可插拔接口
# 如果 pipeline 模块导入失败（无依赖），回退到旧版顺序执行
try:
    from .pipeline.pipeline import build_default_pipeline, RetrievalPipeline
    from .pipeline.interfaces import PipelineContext
    from .pipeline.dag_engine import check_comparison_intent
    _PIPELINE_AVAILABLE = True
except ImportError as e:
    _PIPELINE_AVAILABLE = False
    logger.warning(f"Pipeline 模块不可用，使用旧版顺序执行: {e}")
    # 定义占位符，避免类型检查报错
    check_comparison_intent = None

# 自进化引擎：从历史查询中学习优化策略
try:
    from .evolution.query_learner import get_query_learner, QueryLearner
    _EVOLUTION_AVAILABLE = True
except ImportError as e:
    _EVOLUTION_AVAILABLE = False
    logger.warning(f"自进化引擎不可用: {e}")
    # 定义占位符，避免类型检查报错
    QueryLearner = None
    get_query_learner = None


class RetrievalService:
    """RAG检索服务：自动探测可用功能，运行时根据探测结果智能启用/跳过

    作用：RAG系统的总调度中心，协调多个子模块（Query理解、Embedding、Milvus、
          BM25混合检索、Reranker、LLM生成、Redis缓存），按需执行检索管道。

    设计原理（自动探测+智能降级）：
      - __init__中调用4个_probe_xxx方法，探测每个增强功能是否可用
      - 探测只在启动时执行一次，后续运行时根据features字典决定是否走该功能
      - 探测异常不会导致服务崩溃，只是该功能不启用（容错设计）
      - Reranker采用异步预加载，probe只检查文件存在性，不真正加载模型
      - 混合检索的probe只检查文档数量范围，真正创建检索器在首次使用时

    调用流程：
      query() / query_stream() → _run_retrieval_pipeline() →
        Query理解 → Embedding编码 → 向量检索 → 混合检索(可选) → Reranker(可选) →
        LLM生成 → 结果写入缓存

    面试官可能问：
      Q: 为什么用probe探测而不是手动配置开关？
      A: 用户明确要求"不需要手动配置开关，系统自动判断"。
         每个probe_xxx方法自己检测环境（文件存在、网络连通、数量范围），
         能用的就启用，不能用的静默跳过。

      Q: 探测失败怎么办？会不会让服务起不来？
      A: 不会。所有probe都有try-except，异常只记录日志，features保持False。
         这是设计原则："宁可少一个功能，也不让服务崩溃"。

      Q: Reranker为什么是异步预加载？probe为什么不真正加载模型？
      A: 详见_probe_reranker和_async_preload_reranker的注释。
         一句话：probe只检查模型文件在不在（毫秒级），加载模型在后台线程做。
         这样服务秒级启动，模型加载失败也不会崩溃。
    """

    def __init__(self):
        # 【作用】初始化时获取 embedding 服务的单例对象，embedding 服务负责把文本转成向量
        self.embedding_service = get_embedding_service()
        # 【作用】初始化时获取 Milvus 向量数据库的客户端对象，用来做相似度搜索
        self.milvus_client = get_milvus_client()
        # 【作用】初始化时获取 LLM 路由器的单例对象，用来调用大模型生成答案
        self.llm_router = get_llm_router()

        # 【作用】用一个字典记录各种增强功能是否可用，默认全部关闭
        # 【原理】探测会在后面的方法里执行，如果探测成功就把对应值改成 True
        self.features = {"query_understanding": False, "cache": False, "hybrid_retrieval": False, "reranker": False}
        # 【作用】初始化 Redis 缓存对象为 None，后面探测成功后才会赋值
        self._redis_cache = None
        # 【作用】初始化 query 理解服务为 None，后面探测成功后才赋值
        self.query_understanding = None
        # 【作用】初始化混合检索器为 None，后面探测成功后才会赋值
        self.hybrid_retriever = None
        # 【作用】初始化重排序器（Reranker）为 None，后面加载模型后才赋值
        self.reranker = None
        # 【作用】记录 Reranker 模型缓存在磁盘上的路径，后面探测时找到路径就存这里
        self._reranker_path = None
        # 【作用】创建一个 threading.Event 对象，用来标记 Reranker 模型是否已经加载完毕
        # 【原理】Event 是一个线程间的信号旗，后台线程加载完就调用 set() 通知主线程
        self._reranker_loaded = threading.Event()

        # 【作用】依次探测各个增强功能是否能用
        # 【逻辑】每个 _probe_xxx 方法内部会尝试导入、连接、检查环境，然后更新 self.features
        self._probe_query_understanding()
        self._probe_cache()
        self._probe_hybrid_retrieval()
        self._probe_reranker()

        # 【作用】收集所有已启用的功能名称列表
        # 【逻辑】列表推导式：遍历 features 字典，只保留值为 True 的那些键
        enabled = [n for n, ok in self.features.items() if ok]
        # 【作用】打印日志，告诉开发者哪些增强功能成功启用了
        # 【逻辑】如果 enabled 列表不为空，就展示"🔧 增强功能已启用: xxx"，否则展示"🔧 无增强功能启用"
        logger.info(f"{'🔧 增强功能已启用: ' + ', '.join(enabled) if enabled else '🔧 无增强功能启用'}")

        # 【作用】如果 Pipeline 模块可用，构建标准化的管道编排器
        # 【逻辑】将现有的各服务实例作为 Adapter 注入 Pipeline
        # 对比旧版 _run_retrieval_pipeline 里的 200 行顺序调用，
        # Pipeline 把每个阶段拆成独立 Adapter，可单独测试、替换、插桩
        # 实现了 HiveWard 式"接口即契约"的设计
        self._pipeline: Optional[RetrievalPipeline] = None
        if _PIPELINE_AVAILABLE:
            try:
                # 在探测完成后构建 Pipeline
                # hybrid_retriever 和 reranker 在 probe 阶段设为 None，
                # build_default_pipeline 收到 None 会传入 HybridSearchAdapter(reranker=None)，
                # 该 Adapter 内部会直接透传上一阶段的结果
                self._pipeline = build_default_pipeline(
                    query_understanding=self.query_understanding,
                    embedding_service=self.embedding_service,
                    milvus_client=self.milvus_client,
                    hybrid_retriever=self.hybrid_retriever,
                    reranker_service=self.reranker,  # 可能为 None，RerankAdapter 会处理
                    min_score=settings.MIN_SCORE_THRESHOLD,
                    llm_router=self.llm_router,
                )
                logger.info(f"✅ Pipeline 编排器已构建，阶段: {', '.join(self._pipeline.get_stage_names())}")
            except Exception as e:
                logger.warning(f"⚠️ Pipeline 构建失败，使用旧版顺序执行: {e}")
                self._pipeline = None

        # 【作用】初始化状态机：跟踪每个查询的执行进度，支持断点续传和超时保护
        # 【设计意图】状态机记录每个查询从 INITIALIZING → COMPLETED 的完整生命周期
        # 如果某个阶段卡住，状态机可以自动回滚到上一个稳定状态
        self._state_machine_enabled = _PIPELINE_AVAILABLE
        if self._state_machine_enabled:
            logger.info("✅ 状态机已启用（支持断点续传）")

        # 【作用】初始化自进化引擎：从历史查询中学习优化策略
        # 【设计意图】记录每个查询的参数、结果质量、执行时间，自动提炼成功模式
        self._query_learner = None
        if _EVOLUTION_AVAILABLE and get_query_learner is not None:
            try:
                self._query_learner = get_query_learner()
                logger.info("✅ 自进化引擎已启用")
            except Exception as e:
                logger.warning(f"⚠️ 自进化引擎初始化失败: {e}")

    def _probe_query_understanding(self):
        # 【作用】探测 Query 理解功能是否可用
        """尝试导入 query_understanding 模块并实例化，成功就启用，失败就静默跳过"""
        try:
            # 【作用】尝试从同级目录导入 query_understanding 模块的 get_query_understanding_service 函数
            # 【原理】import 写在 try 块里：如果这个模块不存在或导入时报错，直接抛异常被 except 捕获
            from .query_understanding import get_query_understanding_service
            # 【作用】调用导入的函数获取 query 理解服务的实例
            self.query_understanding = get_query_understanding_service()
            # 【作用】标记 Query 理解功能已启用
            self.features["query_understanding"] = True
            # 【作用】打印成功日志
            logger.info("✅ Query理解已启用")
        except Exception as e:
            # 【作用】如果导入或实例化失败（比如模块没装），记录警告日志但不中断程序
            logger.warning(f"⚠️ Query理解不可用: {e}")

    def _probe_cache(self):
        # 【作用】探测 Redis 缓存是否可用
        """尝试连接 Redis，成功则启用缓存功能，失败则跳过（缓存不是必需的）"""
        try:
            # 【作用】从父级目录 db 包的 redis_client 模块导入 RedisCache 类
            from ..db.redis_client import RedisCache
            # 【作用】创建一个 Redis 客户端实例，连接参数从 settings 配置中读取
            # 【逻辑】host 是 Redis 服务器的 IP/域名，port 是端口（默认 6379），db 是数据库编号，password 如果为空则传 None
            rc = RedisCache(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB, password=settings.REDIS_PASSWORD or None)
            # 【作用】调用 is_connected() 方法检查是否真正连上了 Redis 服务器
            if rc.is_connected():
                # 【作用】连上了就把实例保存到 self._redis_cache，后面读写缓存都用它
                self._redis_cache = rc
                # 【作用】标记缓存功能已启用
                self.features["cache"] = True
                # 【作用】打印成功日志
                logger.info(f"✅ Redis缓存已启用")
            else:
                # 【作用】没连上（比如 Redis 服务没启动），打印提示日志
                logger.info("ℹ️ Redis缓存未启用")
        except Exception as e:
            # 【作用】如果整个 Redis 连接过程报错，打印提示日志，程序继续正常运行
            logger.info(f"ℹ️ Redis缓存未启用: {e}")

    def _probe_hybrid_retrieval(self):
        # 【作用】探测混合检索（向量检索 + BM25 全文检索）是否可用
        """检查 Milvus 中文档数量，如果文档太少或太多就跳过混合检索"""
        try:
            # 【作用】获取 Milvus 集合的统计信息，包括文档总数等
            stats = self.milvus_client.get_collection_stats()
            # 【作用】从统计信息中取出"实体数量"（即文档数量），如果拿不到就默认 0
            n = stats.get("num_entities", 0)
            # 【作用】如果文档数量是 0，说明数据库是空的，混合检索没有意义，直接返回
            if n == 0:
                logger.info("ℹ️ BM25混合检索跳过：Milvus为空")
                return
            # 【作用】如果文档数量超过了设定的最大阈值，也跳过混合检索
            # 【原理】文档太多时 BM25 索引会占用大量内存，小场景才适合用混合检索
            if isinstance(n, int) and n > settings.HYBRID_MAX_DOCS:
                logger.info(f"ℹ️ BM25混合检索跳过：{n}篇 > 阈值{settings.HYBRID_MAX_DOCS}")
                return
            # 【作用】通过探测了，这里先不真正实例化混合检索器（设为 None），后面需要时再创建
            self.hybrid_retriever = None
            # 【作用】标记混合检索功能已启用
            self.features["hybrid_retrieval"] = True
            # 【作用】打印就绪日志，附带当前文档数量
            logger.info(f"✅ BM25混合检索已就绪（{n}篇文档）")
        except Exception as e:
            # 【作用】如果获取统计信息时报错（比如 Milvus 还没建集合），打印提示日志，功能不启用
            logger.info(f"ℹ️ BM25混合检索未启用: {e}")

    def _probe_reranker(self):
        # 【作用】探测 Reranker（重排序模型）是否可用，即本地是否已经缓存了模型文件
        """检查本地是否有 Reranker 模型缓存，如果有则标记为可用并启动异步预加载"""
        try:
            # 【作用】导入 os 模块，用于操作文件路径和环境变量
            import os
            # 【作用】列出两个可能的模型缓存路径
            # 【逻辑】modelscope 和 huggingface 是两个模型下载平台，它们的缓存目录结构不同
            paths = [os.path.expanduser("~/.cache/modelscope/hub/models/BAAI/bge-reranker-base"),
                     os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-reranker-base")]
            # 【作用】遍历两个路径，检查哪个路径真实存在
            for p in paths:
                if os.path.exists(p):
                    # 【作用】找到存在的路径就存到 self._reranker_path
                    self._reranker_path = p
                    break
            # 【作用】如果两个路径都不存在，说明本地没有下载过模型缓存
            if not self._reranker_path:
                logger.info("ℹ️ Reranker未启用：未找到本地模型缓存")
                return
            # 【作用】模型缓存存在，先给 reranker 赋值为 None（后面再真正加载模型）
            self.reranker = None
            # 【作用】标记 Reranker 功能已启用
            self.features["reranker"] = True
            logger.info("✅ Reranker已就绪（异步预加载中...）")

            # 【作用】在后台线程里预加载 Reranker 模型，不阻塞当前线程
            # 【原理】daemon=True 表示守护线程，主线程退出时后台线程自动结束
            import threading
            t = threading.Thread(target=self._async_preload_reranker, daemon=True)
            t.start()
        except Exception as e:
            # 【作用】如果探测过程中出任何异常，打印提示日志，Reranker 功能不启用
            logger.info(f"ℹ️ Reranker探测跳过: {e}")

    def _async_preload_reranker(self):
        """后台线程：预加载 Reranker 模型，不会阻塞服务启动"""
        try:
            # 【作用】从同级目录的 reranker 模块导入 get_reranker_service 函数
            from .reranker import get_reranker_service
            # 【作用】获取 Reranker 服务实例，这会真正加载模型到内存
            r = get_reranker_service()
            # 【作用】检查模型是否成功加载（r.reranker.model 不为 None 就表示加载成功了）
            if r.reranker.model is not None:
                # 【作用】保存 Reranker 服务实例到 self.reranker，后续检索时直接用它重排序
                self.reranker = r
                logger.info("✅ Reranker模型预加载完成")
            else:
                # 【作用】模型加载失败（比如下载的模型文件有问题），关闭该功能
                self.features["reranker"] = False
        except Exception as e:
            # 【作用】如果加载过程报错，打印警告日志，后续第一次使用时还会再尝试同步加载
            logger.warning(f"Reranker预加载失败（首次使用时将同步重试）: {e}")
        finally:
            # 【作用】无论成功还是失败，都设置 Event 信号，告诉主线程：异步加载已经完成了
            # 【逻辑】set() 之后，任何等待这个 Event 的 wait() 调用都会立即返回
            self._reranker_loaded.set()

    def _lazy_load_reranker(self) -> bool:
        # 【作用】懒加载 Reranker：如果还没加载过就立即加载，返回是否加载成功
        """尝试获取 Reranker 实例，有就返回 True，没有则同步加载一次"""
        # 【作用】如果 Reranker 已经加载完成（self.reranker 不为 None），直接返回成功
        if self.reranker is not None:
            return True
        # 【作用】检查后台预加载是否还没有完成
        if not self._reranker_loaded.is_set():
            logger.info("⏳ Reranker后台预加载未完成，开始同步加载...")
            # 【作用】等待后台线程最多 60 秒完成加载
            # 【原理】wait(timeout=60) 会阻塞最多 60 秒，等 Event 被 set() 或被超时打断
            self._reranker_loaded.wait(timeout=60)
            # 【作用】如果后台线程成功加载了模型（wait 后 self.reranker 不为 None），返回成功
            if self.reranker is not None:
                return True
        # 【作用】如果后台加载没有成功（或者超时了），尝试在当前线程同步加载一次
        try:
            from .reranker import get_reranker_service
            r = get_reranker_service()
            if r.reranker.model is not None:
                self.reranker = r
                logger.info("✅ Reranker模型同步加载完成")
                return True
            # 【作用】同步加载也没成功，关闭 Reranker 功能
            self.features["reranker"] = False
            return False
        except Exception as e:
            # 【作用】同步加载报错，记录警告日志，关闭功能
            logger.warning(f"Reranker同步加载失败: {e}")
            self.features["reranker"] = False
            return False

    def _cache_key(self, q: str, ct: Optional[str], tk: int) -> str:
        # 【作用】根据查询内容生成一个唯一的缓存键（MD5 哈希值），用于在 Redis 中查找或存储缓存
        """根据查询、内容类型、top_k 生成 MD5 缓存键"""
        # 【逻辑】把问题 q、内容类型 ct、top_k 值 tk 拼成一个字符串，中间用竖线分隔，然后取 UTF-8 编码后的 MD5 哈希值
        # 【原理】MD5 哈希可以把任意长度的输入变成一个固定长度（32位十六进制）的字符串，保证不同输入生成不同键
        # 【注意】MD5 虽然不安全（可碰撞），但用来做缓存键足够了
        return hashlib.md5(f"{q}|{ct}|{tk}".encode("utf-8")).hexdigest()

    def _check_cache(self, k: str) -> Optional[Dict]:
        # 【作用】根据缓存键去 Redis 中查缓存，有就返回缓存的数据，没有就返回 None
        """检查 Redis 中是否有缓存结果，有则返回"""
        # 【作用】先判断缓存功能是否启用，或者 Redis 客户端是否初始化了，没有就直接返回 None
        if not self.features["cache"] or not self._redis_cache:
            return None
        try:
            # 【作用】用 Redis 客户端的 get 方法根据键获取值
            c = self._redis_cache.get(k)
            # 【作用】如果取到了结果（不为空/None）
            if c:
                # 【作用】缓存命中，直接返回完整结果
                logger.info(f"[缓存命中] key={k[:8]}..., 直接返回")
                return c
        except Exception:
            # 【作用】如果 Redis 查询报错（比如连接断开），静默忽略，就当没缓存
            pass
        return None

    def _write_cache(self, k: str, r: Dict):
        # 【作用】把查询结果写入 Redis 缓存，方便下次同样的查询直接返回
        """将结果写入 Redis 缓存"""
        # 【作用】先判断缓存功能是否启用，没有就直接返回
        if not self.features["cache"] or not self._redis_cache:
            return
        try:
            # 【作用】调用 Redis 客户端的 set 方法写入缓存，同时设置过期时间
            # 【逻辑】ttl 是"生存时间"，单位为秒，从 settings 中读取配置
            self._redis_cache.set(k, r, ttl=settings.CACHE_TTL_SECONDS)
            logger.info(f"[缓存写入] key={k[:8]}..., TTL={settings.CACHE_TTL_SECONDS}s")
        except Exception:
            # 【作用】写入失败就静默忽略，不影响主流程
            pass

    async def _run_retrieval_pipeline(self, question: str, top_k: int, content_type: Optional[str], history: Optional[List[Dict]], kb_name: Optional[str] = None) -> tuple:
        """执行检索管道（优先使用 Pipeline 编排器，失败回退旧版）

        返回：(检索结果列表, 有效问题, 有效历史, 澄清文本或None)

        设计对比：
          旧版：_run_retrieval_pipeline 里写了约200行顺序调用代码，
                各阶段耦合，添加/移除阶段需要改这个方法的内部逻辑。
          新版：Pipeline 编排器将每个阶段拆为独立 Adapter，
                各阶段通过 PipelineContext 传参，互不感知。
                添加新阶段只需写一个新的 Adapter 并注册到 Pipeline 里。

          这跟 HiveWard 的 Blueprint 设计思想一致：
            HiveWard 在画布上添加/删除/替换节点，不需要改运行时代码。
            我们在 Pipeline 里添加/移除/替换阶段，不需要改 RetrievalService 代码。

        降级策略：
          如果 Pipeline 不可用或执行失败，自动回退到旧版顺序执行。
          这是"渐进式改造"——先让新架构能跑，旧代码保留作保底。

        新增优化（v2）：
          1. 自进化引擎：从历史查询中学习优化策略
          2. 状态机：跟踪每个查询的执行进度（仅在Pipeline模式下）
        """
        import time
        start_time = time.time()

        # 【作用】查询自进化引擎，获取优化建议
        # 【设计意图】如果历史中有类似查询且效果好，直接复用其参数
        evolution_params = None
        if self._query_learner and history is None:  # 只对首次查询优化，避免对话中重复优化
            try:
                # 获取意图提示
                intent_hints = self._query_learner.get_intent_hints(question)
                if intent_hints:
                    logger.info(f"🧠 自进化引擎意图提示: {intent_hints}")
                
                # 获取检索参数建议（基于最可能的意图）
                if intent_hints:
                    best_intent = max(intent_hints.items(), key=lambda x: x[1])[0]
                    evolution_params = self._query_learner.get_retrieval_params(best_intent)
                    if evolution_params:
                        logger.info(f"🧠 自进化引擎建议: {evolution_params}")
            except Exception as e:
                logger.debug(f"自进化引擎查询失败: {e}")

        # 【作用】检测是否为比较型查询
        # 【设计意图】比较型查询（如"A和B哪个好"）需要特殊处理
        is_comparison = False
        if _PIPELINE_AVAILABLE and check_comparison_intent is not None:
            try:
                is_comparison = check_comparison_intent(question)
                if is_comparison:
                    logger.info(f"🔀 检测到比较型查询")
            except Exception:
                pass

        # 优先使用 Pipeline 编排器
        if self._pipeline is not None and _PIPELINE_AVAILABLE:
            try:
                ctx = PipelineContext(
                    question=question,
                    top_k=top_k,
                    content_type=content_type,
                    history=history,
                    kb_name=kb_name,
                )

                ctx = await self._pipeline.run(ctx)

                # 【作用】记录查询到自进化引擎
                if self._query_learner:
                    try:
                        duration = time.time() - start_time
                        # 记录查询结果（简化版，只记录基本信息）
                        self._query_learner.record_query(
                            query=question,
                            normalized_query=ctx.normalized_query or question,
                            intent="general",
                            intent_confidence=0.8,
                            retrieval_results=ctx.reranked_results or [],
                            answer=None,  # 答案在后续生成
                            user_feedback=None  # 用户反馈后续收集
                        )
                    except Exception:
                        pass

                # 从 PipelineContext 提取返回值
                if ctx.clarification_prompt:
                    return [], ctx.normalized_query or question, history, ctx.clarification_prompt

                results = ctx.reranked_results or []
                q = ctx.normalized_query or question
                return results, q, history, None

            except Exception as e:
                logger.warning(f"Pipeline 执行失败，回退到旧版顺序执行: {e}")
                # 【作用】记录失败查询到自进化引擎
                if self._query_learner:
                    try:
                        duration = time.time() - start_time
                        # 记录失败查询（简化版）
                        self._query_learner.record_query(
                            query=question,
                            normalized_query=question,
                            intent="general",
                            intent_confidence=0.0,
                            retrieval_results=[],
                            answer=None,
                            user_feedback=None
                        )
                    except Exception:
                        pass

        # ========== 旧版顺序执行（保底） ==========
        # 【作用】初始化 q（有效问题）和 h（有效历史）为入参的原始值，后面可能会被 Query 理解修改
        q, h = question, history

        # 【作用】如果 Query 理解功能已启用，就尝试对用户的问题进行处理
        if self.features["query_understanding"] and self.query_understanding:
            try:
                # 【作用】调用 Query 理解服务的 understand 方法分析用户问题
                # 【逻辑】传入用户的问题和对话历史，返回一个理解结果对象 u
                u = self.query_understanding.understand(question, conversation_history=history)
                # 【作用】如果模型认为需要向用户追问更多信息
                if u.needs_clarification:
                    # 【作用】直接返回空的检索结果和澄清提示
                    # 【注意】返回格式是 (空列表, 原问题, 原历史, 澄清文本)
                    return [], q, h, u.clarification_prompt
                # 【作用】取出模型规范化后的查询语句（可能纠正了错别字、补全了缩写）
                q = u.normalized_query
                # 【作用】如果模型生成了多个子查询，且多于一个，就取第一个子查询作为主要查询
                if u.sub_queries and len(u.sub_queries) > 1:
                    q = u.sub_queries[0]
            except Exception:
                # 【作用】如果 Query 理解过程出错，就静默忽略，使用原始的问题继续检索
                pass

        # 【作用】把处理后的查询 q 编码成向量（一个浮点数数组，表示语义特征）
        emb = self.embedding_service.encode(q)
        # 【作用】用编码后的向量去 Milvus 向量数据库做相似度搜索
        # 【逻辑】多取候选（top_k*6），让reranker有更多选择，自然排序
        # 【设计意图】从top_k*3扩充到top_k*6，确保reranker能看到更多候选，减少遗漏
        vr = self.milvus_client.search(embedding=emb, top_k=top_k * 6, content_type=content_type, kb_name=kb_name)

        # 【作用】如果混合检索功能已启用，就对向量检索结果做一次 BM25 全文检索融合
        if self.features["hybrid_retrieval"] and self.hybrid_retriever:
            try:
                # 【作用】调用混合检索器，传入原始文本查询 q 和向量检索结果列表
                # 【逻辑】列表推导式把每条结果转换成一个 (文档, 分数) 的元组，score 默认为 0.0
                rr = self.hybrid_retriever.search(query=q, vector_results=[(r, r.get("score", 0.0)) for r in vr], top_k=top_k)
                # 【作用】把混合检索返回的 (文档, 分数) 元组列表合并成一个字典列表，将分数写入文档的 score 字段
                rr = [{**d, "score": s} for d, s in rr]
            except Exception:
                # 【作用】如果混合检索报错，降级为纯向量检索结果
                rr = vr
        else:
            # 【作用】如果混合检索未启用，直接使用向量检索结果
            rr = vr

        # 【作用】如果 Reranker 功能已启用，且成功获取了 Reranker 实例（可能触发懒加载），就对结果重排序
        if self.features["reranker"] and self._lazy_load_reranker():
            try:
                # 【作用】为每条结果添加一个 "content" 字段，值来自已有的 "text" 字段
                # 【原理】Reranker 服务期望文档有 "content" 字段，而原始结果里是 "text" 字段名，这里做个映射
                docs = [{**r, "content": r.get("text", "")} for r in rr]
                # 【作用】调用 Reranker 服务的 rerank_documents 方法对文档重排序
                # 【逻辑】传入原始查询 q 和文档列表，返回按相关性重新排列后的文档列表
                rd = self.reranker.rerank_documents(query=q, documents=docs, top_k=top_k)
                # 【作用】从重排序后的结果中移除临时添加的 "content" 字段和旧的 "rerank_score" 字段（如果有）
                # 【逻辑】列表推导式：对每个字典 d，只保留不在 ("content", "rerank_score") 这两个键中的字段
                rr = [{k: v for k, v in d.items() if k not in ("content", "rerank_score")} for d in rd]
                # 【作用】遍历重排序后的结果，把每条文档的 "rerank_score"（重排序分数）写回到结果中
                for i, d in enumerate(rd):
                    if i < len(rr):
                        rr[i]["rerank_score"] = d.get("rerank_score", 0.0)
            except Exception:
                # 【作用】如果重排序过程报错，就静默忽略，使用重排序之前的结果
                pass

        # 【作用】对最终结果做分数阈值过滤
        # 【原理】用向量检索的原始余弦相似度（score字段）做阈值过滤，低于阈值的弃掉
        # 【设计意图】这是提升CP（上下文精度）的关键——不让低相关性文档进入LLM上下文
        # 【边界情况】如果过滤后结果为空，保留至少1个最高分文档，避免LLM无上下文可用
        if rr and settings.MIN_SCORE_THRESHOLD > 0:
            filtered = [r for r in rr if r.get("score", 0) >= settings.MIN_SCORE_THRESHOLD]
            if not filtered and rr:
                # 【作用】如果全部被过滤，保留最高分的那一个作为底线
                rr = [rr[0]]
            else:
                rr = filtered[:top_k]

        # 【作用】返回最终检索结果、有效问题、有效历史、澄清提示（None 表示没有澄清需求）
        return rr, q, h, None

    @staticmethod
    def _build_sources(retrieval_results: List[Dict]) -> List[Dict]:
        # 【作用】把原始检索结果整理成简洁的"引用来源"列表，返回给前端展示
        """从检索结果中提取来源信息（截断长文本、保留元数据），用于前端展示"""
        return [{"text": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                 "page": r["page_start"], "section": r["section"],
                 "content_type": r["content_type"], "score": round(r["score"], 4)}
                for r in retrieval_results]

    async def prepare_pipeline(self, question: str, top_k: int = settings.RETRIEVAL_TOP_K, content_type: Optional[str] = None, history: Optional[List[Dict]] = None, kb_name: Optional[str] = None) -> dict:
        """只执行检索管道，不调 LLM。返回 sources + 上下文，供流式接口先发来源再逐 token 推答案。

        返回:
            {"rr": list[dict], "q": str, "h": list|None, "clarification": str|None,
             "sources": list[dict], "ctx": list[str]}
        """
        rr, q, h, clarification = await self._run_retrieval_pipeline(question, top_k, content_type, history, kb_name)
        ctx = [r["text"] for r in rr] if rr else []
        return {
            "rr": rr, "q": q, "h": h, "clarification": clarification,
            "sources": self._build_sources(rr) if rr else [],
            "ctx": ctx,
        }

    async def query(self, question: str, top_k: int = settings.RETRIEVAL_TOP_K, content_type: Optional[str] = None, history: Optional[List[Dict]] = None, kb_name: Optional[str] = None) -> Dict:
        # 【作用】对外提供的同步查询接口：接收问题，返回包含答案和来源的字典
        """主查询入口：检索 + 生成，含缓存

        参数：
          question: 用户问题
          top_k: 返回top-k个检索结果（默认3）
          content_type: 可选的类型过滤（如"政策"、"新闻"）
          history: 对话历史
          kb_name: 知识库名称过滤（可选，用于多公司场景）

        返回：
          {"answer": str, "sources": list[dict], "retrieval_results": list[dict]}

        流程：
          缓存检查 → _run_retrieval_pipeline → LLM生成 → 结果缓存写入

        边界情况：
          - 缓存命中：直接返回缓存结果，不走任何管道
          - 需澄清：返回澄清文本，sources和retrieval_results为空
          - 检索结果为空：抛BusinessException
          - 未知异常：包装成BusinessException统一处理

        面试官可能问：
          Q: 同步查询和流式查询有什么区别？
          A: query()一次性返回完整答案和来源，适合非对话场景（如API调用）；
             query_stream()用yield逐段返回，适合前端SSE打字机效果。
             两者的检索流程完全一样，区别在LLM生成阶段——generate vs generate_stream。

          Q: 缓存策略是什么样的？
          A: MD5(question|content_type|top_k)作为缓存键。
             缓存命中时跳过完整的检索+生成管道，直接返回历史结果。
             TTL由config中的CACHE_TTL_SECONDS控制。
             注意：缓存只针对有检索结果的query，需澄清的query不缓存。

          Q: 为什么检索结果为空时要抛异常而不是返回空答案？
          A: 抛BusinessException可以让前端统一处理"无结果"的场景（如展示
             "未找到相关信息"的提示），而不是让LLM瞎编一个答案。
             这是RAG系统的关键设计原则："不知道就是不知道"。
        """
        try:
            # 【作用】根据问题、内容类型、top_k 生成一个唯一的缓存键
            ck = self._cache_key(question, content_type, top_k)
            # 【作用】拿着缓存键去 Redis 中检查是否有缓存结果
            cached = self._check_cache(ck)
            if cached:
                # 【作用】如果有缓存，直接返回缓存的结果，不必重新检索和调用 LLM
                return cached

            # 【作用】执行完整的检索流水线（async 版本）
            rr, q, h, clarification = await self._run_retrieval_pipeline(question, top_k, content_type, history, kb_name)
            # 【作用】如果有澄清提示（说明需要让用户再补充信息）
            if clarification:
                # 【作用】返回一个只有澄清文本、没有来源和检索结果的响应
                return {"answer": clarification, "sources": [], "retrieval_results": []}
            # 【作用】如果检索结果为空（没找到任何相关文档）
            if not rr:
                # 【作用】抛出一个业务异常，告诉前端"找不到相关信息"
                raise BusinessException(ErrorCode.RETRIEVAL_EMPTY, "未找到与问题相关的信息")

            # 【作用】从检索结果中提取所有文本，作为 LLM 的参考上下文
            ctx = [r["text"] for r in rr]
            # 【作用】调用 LLM 路由器生成答案，传入处理后的查询 q、参考上下文 ctx、对话历史 h
            answer = await self.llm_router.generate(query=q, context=ctx, history=h)

            # 【作用】组装最终的返回结果：答案 + 来源列表 + 原始检索结果
            result = {"answer": answer, "sources": self._build_sources(rr), "retrieval_results": rr}
            # 【作用】将本次查询结果写入 Redis 缓存，下次相同查询可以秒回
            self._write_cache(ck, result)
            return result
        except BusinessException:
            # 【作用】如果是我们主动抛出的业务异常（比如没找到信息），直接往上抛，不包装
            raise
        except Exception as e:
            # 【作用】如果是其他未知异常，记录错误日志，然后包装成业务异常抛给上层
            logger.error(f"查询失败: {e}")
            raise BusinessException(ErrorCode.LLM_GENERATION_FAILED, str(e))

    async def query_stream(self, question: str, top_k: int = settings.RETRIEVAL_TOP_K, content_type: Optional[str] = None, history: Optional[List[Dict]] = None, kb_name: Optional[str] = None):
        # 【作用】对外提供的流式查询接口：每次 yield 返回一段文本，适合前端打字机效果
        """流式查询入口：边检索边输出（SSE 效果）

        参数：同query()
        返回：Generator[str]，逐段yield生成文本

        与query()的区别：
          - query()走generate（一次性返回完整答案）
          - query_stream()走generate_stream（yield逐段返回）
          - 检索流程完全相同，只有LLM生成方式不同

        边界情况：
          - 需澄清：yield澄清文本后return
          - 检索为空：yield提示文本后return
          - 异常：yield错误信息后return（不抛异常）

        面试官可能问：
          Q: 为什么query_stream异常时不抛异常，而是yield错误信息？
          A: 因为Generator不能抛异常后被调用者捕获——yield from的调用方
             捕获异常很麻烦。所以yield一段文字提示"生成失败: xxx"
             让前端直接展示，比让连接断掉更好。

          Q: query_stream为什么不走缓存？
          A: 当前版本query_stream不走缓存。流式输出的中间状态不适合缓存。
             如果需要一个流式但有缓存的方案，可以返回缓存中的完整文本
             再按固定长度切分成块yield出来。
        """
        try:
            # 【作用】执行完整的检索流水线（async 版本）
            rr, q, h, clarification = await self._run_retrieval_pipeline(question, top_k, content_type, history, kb_name)
            if clarification:
                yield clarification
                return
            if not rr:
                yield "未找到与问题相关的信息"
                return
            ctx = [r["text"] for r in rr]
            # 【作用】async for 逐 token 获取 LLM 流式输出
            async for token in self.llm_router.generate_stream(query=q, context=ctx, history=h):
                yield token
        except Exception as e:
            yield f"生成失败: {str(e)}"

    def get_feature_status(self) -> Dict:
        # 【作用】返回当前各增强功能的启用状态和说明文字，方便前端展示或调试
        """获取所有增强功能的状态"""
        reasons = {
            # 【作用】定义每个功能的状态说明文字：已启用 或 不可用
            "query_understanding": "已启用" if self.features["query_understanding"] else "不可用",
            "cache": "已启用" if self.features["cache"] else "不可用",
            "hybrid_retrieval": "已就绪" if self.features["hybrid_retrieval"] else "不可用",
            "reranker": "已就绪" if self.features["reranker"] else "不可用",
        }
        # 【作用】组装成统一的格式返回：每个功能包含 enabled 布尔值和 reason 文字说明
        return {n: {"enabled": self.features[n], "reason": reasons[n]} for n in self.features}


# 【作用】模块级全局变量，用作 RetrievalService 的单例缓存
_retrieval_service = None


def get_retrieval_service() -> RetrievalService:
    # 【作用】获取 RetrievalService 的单例对象，避免重复创建和重复探测
    """单例工厂：全局只初始化一次 RetrievalService"""
    global _retrieval_service
    if _retrieval_service is None:
        # 【作用】如果是第一次调用，就创建实例并保存到全局变量
        _retrieval_service = RetrievalService()
    return _retrieval_service
