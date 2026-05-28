"""
Milvus向量数据库客户端，封装连接管理、向量检索与统计功能。
"""

# 【作用】引入logging模块，用于记录程序运行日志
# 【逻辑】记录Milvus连接、检索等操作的状态，方便排查问题
import logging
# 【作用】引入类型注解，让函数签名更清晰
from typing import List, Dict, Any, Optional

# 【作用】从pymilvus导入MilvusClient（新版客户端）
# 【原理】MilvusClient是pymilvus 2.3+版本提供的简化客户端，替代了旧的Collection API
# 【优势】新版API更简洁，不需要手动管理连接池和集合对象
from pymilvus import MilvusClient

# 【作用】从配置模块读取Milvus连接参数（地址、端口、集合名等）
from ..config import settings

# 【作用】创建当前模块的日志记录器
logger = logging.getLogger(__name__)


class MilvusClientWrapper:
    """
    作用：封装 MilvusClient，提供连接管理、向量检索与统计查询。
    原理：内部使用pymilvus的MilvusClient，对外提供更简洁的接口。

    面试官可能问：
      Q: 为什么封装MilvusClient而不是直接使用？
      A: (1) 统一管理连接参数（从config读取，避免硬编码）
         (2) 启动时验证集合存在性，及早发现问题
         (3) 对外暴露更简洁的接口（search只需传向量和top_k）
         (4) 如果以后换向量数据库，只需要改这一个文件

      Q: 为什么用HTTP连接不用gRPC？
      A: Milvus 2.3+支持HTTP协议，比gRPC更容易调试（可用curl测试）。
         HTTP在WSL/Windows混合环境下更稳定（gRPC在WSL2中有时会有
         网络桥接问题）。生产环境建议用gRPC，性能更好。
    """

    def __init__(self):
        """
        作用：初始化客户端：读取配置、连接 Milvus 并验证集合存在。
        逻辑：
          1. 从配置读取集合名
          2. 调用_connect()建立连接
          3. 如果连接或验证失败，抛出异常阻止服务启动
        """
        # 【作用】记录要操作的Milvus集合名（如"prospectus"）
        # 【原理】一个Milvus可以包含多个集合，每个集合类似于关系数据库中的一张表
        self.collection_name = settings.MILVUS_COLLECTION
        # 【作用】先占位，_connect()里真正创建客户端
        self.client = None
        # 【作用】调用连接方法，建立与Milvus的连接并验证集合存在
        self._connect()

    def _connect(self):
        """
        作用：连接 Milvus 服务，创建客户端并检查集合是否已存在。
        逻辑：
          1. 构建Milvus的URI（http://host:port）
          2. 创建MilvusClient实例
          3. 检查需要的集合是否已存在
          4. 如果集合不存在，提示用户先运行离线导入脚本
        """
        try:
            # 【作用】构建Milvus连接地址
            # 【原理】Milvus新版支持HTTP协议连接（之前是gRPC）
            uri = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
            # 【作用】创建Milvus客户端实例（建立连接）
            self.client = MilvusClient(uri=uri)
            logger.info(f"已连接Milvus: {uri}")

            # 【作用】检查配置的集合是否在Milvus中存在
            # 【原理】has_collection返回True/False，查询Milvus的元数据
            if not self.client.has_collection(self.collection_name):
                # 【作用】集合不存在时记录错误并抛出异常
                # 【逻辑】集合不存在说明还没导入数据，提示用户先运行离线脚本
                logger.error(f"集合不存在: {self.collection_name}")
                raise RuntimeError(
                    f"Milvus集合 {self.collection_name} 不存在，请先运行离线导入脚本"
                )

            logger.info(f"集合已加载: {self.collection_name}")

        except Exception as e:
            # 【作用】捕获连接过程中所有异常
            logger.error(f"Milvus连接失败: {e}")
            raise RuntimeError(f"Milvus连接失败: {e}")

    def search(
        self,
        embedding: List[float],
        top_k: int = 10,
        content_type: Optional[str] = None,
        section: Optional[str] = None,
        kb_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        作用：向量检索——用问题向量在 Milvus 中查找最相似的 chunks。
        支持按内容类型、章节和知识库名称过滤。

        参数说明：
            embedding: 问题向量（1024维浮点数列表）
            top_k: 返回条数（默认10）
            content_type: 按内容类型过滤（"text" 或 "table"）
            section: 按章节过滤（如"财务会计信息"）
            kb_name: 按知识库名称过滤（如"招股说明书1"），用于多公司场景
        返回：
            检索结果列表，每个元素包含 id/chunk_id/text/page_start/
            page_end/section/content_type/table_index/score/kb_name
        异常：
            RuntimeError: 检索失败
        """
        try:
            # 【作用】准备搜索的过滤条件
            # 【原理】Milvus支持标量字段过滤，类似SQL的WHERE子句
            filter_expr = None
            filters = []

            # 【作用】如果指定了内容类型，添加过滤条件
            if content_type:
                # 【格式】content_type == "text" 或 content_type == "table"
                filters.append(f'content_type == "{content_type}"')

            # 【作用】如果指定了章节，添加过滤条件
            if section:
                # 【格式】section == "财务会计信息"
                filters.append(f'section == "{section}"')

            # 【作用】如果指定了知识库名称，添加过滤条件
            # 【原理】多公司场景下，按kb_name限定检索范围，避免跨公司文档混入
            if kb_name:
                filters.append(f'kb_name == "{kb_name}"')

            # 【作用】如果有过滤条件，用AND连接成完整的过滤表达式
            if filters:
                # 【格式】'content_type == "text" and section == "财务会计信息"'
                filter_expr = " and ".join(filters)

            # 【作用】设置向量检索参数
            # 【原理】metric_type="IP"表示内积距离（IP=Inner Product）
            # 因为BGE-M3向量经过normalize_embeddings=True归一化，内积=余弦相似度
            # nprobe=16是IVF索引的搜索参数，控制搜索簇的数量
            search_params = {
                "metric_type": "IP",   # 内积（余弦相似度）
                "params": {"nprobe": 16}  # 搜索探针数
            }

            # 【作用】执行向量检索
            # 【参数说明】
            #   collection_name: 要搜索的集合
            #   data: 查询向量（列表形式，可以传多个向量同时搜索）
            #   anns_field: 向量字段名（必须是FLAT/IVF_FLAT/HNSW等支持索引的字段）
            #   search_params: 搜索参数
            #   limit: 返回的最相似结果数
            #   filter: 标量字段的过滤表达式
            #   output_fields: 返回结果中包含的字段
            results = self.client.search(
                collection_name=self.collection_name,
                data=[embedding],           # 传入查询向量
                anns_field="embedding",     # 向量字段名
                search_params=search_params,
                limit=top_k,                # 返回Top-K结果
                filter=filter_expr,         # 标量过滤
                output_fields=[             # 返回哪些字段
                    "chunk_id", "text", "page_start", "page_end",
                    "section", "content_type", "table_index", "source_pdf", "kb_name"
                ]
            )

            # 【作用】解析搜索结果，转换成统一的字典格式
            return self._parse_results(results)

        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            raise RuntimeError(f"向量检索失败: {e}")

    def _parse_results(self, results) -> List[Dict[str, Any]]:
        """
        作用：解析 Milvus 新版 API 返回结果，提取 id/distance/entity 字段为字典列表。

        新版返回格式：[[{id, distance, entity: {chunk_id, text, ...}}]]
        内部将新版 distance 映射为对外接口的 score。
        """
        # 【作用】初始化空的结果列表
        parsed = []

        # 【作用】遍历搜索结果
        # 【格式】results的结构：[[{id, distance, entity:{chunk_id,text,...}}, ...]]
        # results[0]是第一个查询向量的结果列表（因为data=[embedding]只传了一个向量）
        for hit in results[0]:
            # 【作用】获取元数据字段
            # 【原理】entity是Milvus存储的非向量字段集合
            entity = hit.get("entity", {})

            # 【作用】将Milvus的返回格式转换成统一的字典格式
            # 【逻辑】每个结果包含检索到的文档信息和相似度分数
            parsed.append({
                "id": hit.get("id"),                    # Milvus内部的ID
                "chunk_id": entity.get("chunk_id"),     # 文档块的唯一ID
                "text": entity.get("text"),             # 文档块的文本内容
                "page_start": entity.get("page_start"),  # 起始页码
                "page_end": entity.get("page_end"),      # 结束页码
                "section": entity.get("section"),       # 所属章节名
                "content_type": entity.get("content_type"),  # 内容类型（text/table）
                "table_index": entity.get("table_index"),    # 如果表格数据，记录表格索引
                "source_pdf": entity.get("source_pdf", ""),  # v2.0: 来源PDF路径
                "kb_name": entity.get("kb_name", ""),        # v2.0: 知识库名称
                "score": hit.get("distance"),            # 相似度分数（内积距离）
            })

        return parsed

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        作用：获取集合统计信息（数据量、向量维度等），兼容多种 pymilvus 版本。

        返回：{"collection_name": str, "num_entities": int, "embedding_dim": int}
        """
        try:
            num_entities = 0

            # 【作用】方式1：优先用新版query_count API获取数据量
            # 【原理】pymilvus 2.4+提供了query_count方法直接返回文档总数
            try:
                num_entities = self.client.query_count(
                    collection_name=self.collection_name
                )
            except (AttributeError, Exception):
                # 【作用】如果query_count不存在（旧版pymilvus），用方式2
                pass

            # 【作用】方式2：如果方式1没拿到数据，用query方法测试连通性
            # 【逻辑】至少确认集合可以访问，标记为-1表示"存在但不确定数量"
            if num_entities == 0:
                try:
                    # 【作用】发一个最小查询，验证集合可访问
                    self.client.query(
                        collection_name=self.collection_name,
                        filter="id >= 0",
                        output_fields=["id"],
                        limit=1,
                    )
                    # 【作用】如果查询成功且集合存在，标记为-1
                    if self.client.has_collection(self.collection_name):
                        num_entities = -1
                except Exception:
                    # 【作用】如果两种方式都失败，保留0
                    pass

            # 【作用】返回统计信息
            return {
                "collection_name": self.collection_name,  # 集合名称
                "num_entities": num_entities,              # 文档总数（-1表示未知但可用）
                "embedding_dim": settings.EMBEDDING_DIM,   # 向量维度（来自配置）
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}


# 【作用】全局客户端实例（单例模式）
# 【原理】_milvus_client是模块级变量，确保整个进程只有一个Milvus连接
# 【逻辑】初始化None，第一次调用get_milvus_client()时创建
_milvus_client = None


def get_milvus_client() -> MilvusClientWrapper:
    """
    作用：获取 MilvusClientWrapper 单例（避免重复建立 gRPC 连接）。
    原理：全局变量_milvus_client，第一次调用时创建连接，后续直接复用。
    """
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClientWrapper()
    return _milvus_client
