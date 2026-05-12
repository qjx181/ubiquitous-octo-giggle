"""
backend/app/db/milvus_client.py — Milvus向量数据库客户端（单一职责）
=====================================================================
作用：封装与Milvus向量数据库的所有交互操作
原理：
  1. Milvus是一个高性能分布式向量数据库，支持十亿级向量的近实时检索
  2. 使用gRPC协议通信（比HTTP更高效，适合频繁的向量计算）
  3. 支持向量检索+标量过滤混合查询
  4. 使用新版MilvusClient API（取代旧版ORM-style Collection API）
     参考：https://milvus.io/docs/search.md
"""

import logging
from typing import List, Dict, Any, Optional

from pymilvus import MilvusClient  # 新版客户端（取代旧版 Collection）

from ..config import settings

logger = logging.getLogger(__name__)


class MilvusClientWrapper:
    """
    作用：Milvus向量数据库客户端
    原理：封装新版MilvusClient API，提供连接管理、向量检索、统计等功能
    使用示例：
        client = MilvusClientWrapper()
        results = client.search(embedding_vector, top_k=5)
        # results = [{"text": "...", "page_start": 15, "score": 0.87}, ...]
    """

    def __init__(self):
        """
        作用：初始化Milvus客户端
        原理：自动连接Milvus服务并加载指定的集合
        逻辑：
          1. 从config读取主机+端口+集合名
          2. 创建MilvusClient实例（自带连接管理，无需手动connect）
          3. 检查集合是否存在，不存在则报错
        """
        self.collection_name = settings.MILVUS_COLLECTION  # 集合名，如"prospectus"
        self.client = None  # 先占位，_connect()里赋值
        self._connect()

    def _connect(self):
        """
        作用：连接Milvus服务
        原理：
          1. MilvusClient初始化时自动建立gRPC连接
          2. has_collection()检查集合是否已创建
          3. 连接失败会抛异常，阻止程序启动（及早暴露问题）
        逻辑：
          1. 创建MilvusClient实例（传入uri）
          2. 检查集合是否存在
        """
        try:
            # 创建新版MilvusClient
            # 新版API直接用uri参数，格式：http://host:port
            uri = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
            self.client = MilvusClient(uri=uri)
            logger.info(f"已连接Milvus: {uri}")

            # 检查集合是否存在
            # 如果还没跑过离线导入，集合不存在，这里直接报错
            if not self.client.has_collection(self.collection_name):
                logger.error(f"集合不存在: {self.collection_name}")
                raise RuntimeError(f"Milvus集合 {self.collection_name} 不存在，请先运行离线导入脚本")

            logger.info(f"集合已加载: {self.collection_name}")

        except Exception as e:
            logger.error(f"Milvus连接失败: {e}")
            raise RuntimeError(f"Milvus连接失败: {e}")

    def search(
        self,
        embedding: List[float],
        top_k: int = 10,
        content_type: Optional[str] = None,
        section: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        作用：向量检索——找与问题向量最相似的chunks
        原理：
          1. 用问题向量去Milvus中计算所有向量与它的内积(IP)相似度
          2. 返回相似度最高的top_k个结果
          3. 支持标量过滤（如只搜"text"类型或某个章节）
        新版API说明：
          新版MilvusClient的search()参数与旧版Collection.search()不同：
          - 使用collection_name指定集合（而非先打开Collection对象）
          - data直接传向量列表（支持批量查询）
          - search_params里指定metric_type和索引参数
          - filter代替旧版的expr
          - output_fields返回需要的字段
          - limit指定返回条数
        参数：
            embedding: 问题向量（1024维浮点数列表）
            top_k: 返回多少条结果
            content_type: 过滤条件（"text"文本 / "table"表格）
            section: 按章节过滤
        返回：
            检索结果列表，每条的格式：
            {
                "id": 12345,                          # Milvus自动生成的ID
                "chunk_id": "chunk_0012",             # chunk编号
                "text": "2023年营业收入...",           # chunk原文
                "page_start": 15,                     # 起始页码
                "page_end": 16,                       # 结束页码
                "section": "财务会计信息",             # 所属章节
                "content_type": "text",                # 内容类型
                "table_index": None,                   # 表格编号
                "distance": 0.8721,                    # 相似度分数（新版返回distance）
            }
        异常：
            RuntimeError: 检索失败（如Milvus服务断开）
        """
        try:
            # 构建标量过滤条件（新版API用filter参数，语法同旧版expr）
            filter_expr = None
            filters = []

            if content_type:
                # 按内容类型过滤
                filters.append(f'content_type == "{content_type}"')

            if section:
                # 按章节过滤
                filters.append(f'section == "{section}"')

            if filters:
                filter_expr = " and ".join(filters)

            # 设置检索参数
            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 16}
            }

            # 执行向量检索（新版API）
            # 参数说明：
            #   collection_name: 集合名称
            #   data: 查询向量（列表，支持批量查询）
            #   anns_field: 在哪个向量字段上搜
            #   search_params: 检索参数
            #   limit: 返回前N条
            #   filter: 标量过滤表达式（代替旧版的expr）
            #   output_fields: 需要返回哪些字段
            results = self.client.search(
                collection_name=self.collection_name,
                data=[embedding],
                anns_field="embedding",
                search_params=search_params,
                limit=top_k,
                filter=filter_expr,
                output_fields=[
                    "chunk_id", "text", "page_start", "page_end",
                    "section", "content_type", "table_index"
                ]
            )

            # 解析新版API返回的结果
            # 新版返回格式：[[{id: ..., distance: ..., entity: {...}}, ...]]
            #   - id: 主键
            #   - distance: 距离分数
            #   - entity: 字典，包含output_fields中请求的字段
            return self._parse_results(results)

        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            raise RuntimeError(f"向量检索失败: {e}")

    def _parse_results(self, results) -> List[Dict[str, Any]]:
        """
        作用：将Milvus新版API返回的结果解析成结构化的Python字典
        原理：
          新版MilvusClient返回结果格式：
          [
            [  # 第一批查询的结果（我们一次只查一批）
              {
                "id": 12345,
                "distance": 0.8721,
                "entity": {
                  "chunk_id": "chunk_0012",
                  "text": "...",
                  ...
                }
              },
              ...
            ]
          ]
        逻辑：
          - results[0] 是第一批查询的结果
          - 每个元素是字典，包含id/distance/entity
          - entity中是output_fields的数据
        返回：
            解析后的字典列表
        """
        parsed = []

        # results[0] 是第一批查询的结果
        # 新版API每个结果是一个字典（而非旧版的hit对象）
        for hit in results[0]:
            entity = hit.get("entity", {})  # 获取entity字段
            parsed.append({
                "id": hit.get("id"),
                "chunk_id": entity.get("chunk_id"),
                "text": entity.get("text"),
                "page_start": entity.get("page_start"),
                "page_end": entity.get("page_end"),
                "section": entity.get("section"),
                "content_type": entity.get("content_type"),
                "table_index": entity.get("table_index"),
                "score": hit.get("distance"),  # 新版返回distance，保持对外接口用score
            })

        return parsed

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        作用：获取集合的统计信息（数据量、维度等）
        原理：调用 MilvusClient 获取统计，兼容多种 pymilvus 版本
        返回：
            {"collection_name": "prospectus", "num_entities": 1234, "embedding_dim": 1024}
        """
        try:
            num_entities = 0

            # 方法1：尝试 query_count（部分新版 pymilvus 支持）
            try:
                num_entities = self.client.query_count(
                    collection_name=self.collection_name
                )
            except (AttributeError, Exception):
                pass

            # 方法2：用 count 聚合查询（部分版本支持）
            if num_entities == 0:
                try:
                    count_result = self.client.query(
                        collection_name=self.collection_name,
                        filter="id >= 0",
                        output_fields=["count(*)"],
                        limit=1,
                    )
                    if count_result and len(count_result) > 0:
                        for item in count_result:
                            for v in item.values():
                                if isinstance(v, int):
                                    num_entities = v
                                    break
                except Exception:
                    pass

            # 方法3：直接查询所有 id 数量（兜底）
            if num_entities == 0:
                try:
                    all_data = self.client.query(
                        collection_name=self.collection_name,
                        filter="id >= 0",
                        output_fields=["id"],
                        limit=1,
                    )
                    # 如果至少能查到一条，说明有数据，但具体数量未知
                    # 用 has_collection 确认集合存在
                    if self.client.has_collection(self.collection_name):
                        num_entities = -1  # -1 表示"有数据但数量未知"
                except Exception:
                    pass

            return {
                "collection_name": self.collection_name,
                "num_entities": num_entities,
                "embedding_dim": settings.EMBEDDING_DIM,
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}


# 全局客户端实例（单例模式）
_milvus_client = None


def get_milvus_client() -> MilvusClientWrapper:
    """
    作用：获取Milvus客户端单例
    原理：第一次调用时连接Milvus，后续直接返回已有连接
          避免反复建立gRPC连接（浪费资源）
    返回：
        MilvusClientWrapper实例
    """
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClientWrapper()
    return _milvus_client
