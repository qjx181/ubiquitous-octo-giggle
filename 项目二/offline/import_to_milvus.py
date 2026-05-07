"""
import_to_milvus.py — 向量导入Milvus
作用：读取向量JSONL，补回文本，导入Milvus，支持REBUILD开关和幂等
"""

import json
import sys
from pathlib import Path

from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    EMBEDDINGS_JSONL_PATH, CHUNKS_JSONL_PATH,
    MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION,
    MILVUS_BATCH_SIZE, EMBEDDING_DIM, REBUILD_COLLECTION,
    RUN_TIMESTAMP, setup_logger,
)

logger = setup_logger("import_to_milvus")


def connect_milvus() -> bool:
    """
    作用：连接到本地Milvus服务
    原理：pymilvus通过gRPC协议连接Milvus的19530端口
    逻辑：connections.connect()，失败则记录错误
    返回：True连接成功，False连接失败
    """
    logger.info(f"连接Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    try:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        logger.info("Milvus连接成功")
        return True
    except Exception as e:
        logger.error(f"连接失败: {e}")
        return False


def create_collection() -> Collection:
    """
    作用：在Milvus中创建prospectus集合，定义字段结构
    原理：Milvus集合类似数据库表，需要提前定义字段类型
    逻辑：
      - id: 自增主键
      - chunk_id: 原始chunk编号
      - text: 文本内容（最大10000字符，防止超长chunk）
      - page_start/page_end: 页码范围（来源追踪）
      - section: 所属章节
      - source_pdf: 来源PDF路径
      - embedding: 1024维向量（用于相似度检索）
    返回：Collection对象
    """
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="page_start", dtype=DataType.INT64),
        FieldSchema(name="page_end", dtype=DataType.INT64),
        FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="source_pdf", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]
    schema = CollectionSchema(fields, description="招股说明书问答系统向量库")
    c = Collection(name=MILVUS_COLLECTION, schema=schema)
    logger.info(f"集合 '{MILVUS_COLLECTION}' 创建成功")
    return c


def create_index(collection):
    """
    作用：为向量字段创建索引，加速检索
    原理：IVF_FLAT是倒排文件索引，适合中小规模数据快速检索
    逻辑：用内积距离（IP）作为相似度度量
    """
    params = {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
    try:
        collection.create_index(field_name="embedding", index_params=params)
        logger.info("向量索引创建成功")
    except Exception as e:
        logger.warning(f"索引可能已存在: {e}")


def check_existing_data(collection) -> int:
    """
    作用：检查集合中已有数据量
    原理：幂等设计，已有数据则跳过导入
    逻辑：collection.load()后读取num_entities
    返回：数据条数
    """
    try:
        collection.load()
        return collection.num_entities
    except Exception:
        return 0


def check_collection_exists(name: str) -> bool:
    """作用：检查集合是否已存在"""
    return utility.has_collection(name)


def load_text_map() -> dict:
    """
    作用：从chunk JSONL加载文本映射（chunk_id → text）
    原理：向量文件已与文本解耦（不存text），导入Milvus时需要补回
    逻辑：读取chunk JSONL，建立chunk_id到text的字典映射
    返回：{chunk_id: text, ...}
    """
    m = {}
    p = Path(str(CHUNKS_JSONL_PATH))
    if not p.exists():
        logger.warning(f"chunk文件不存在: {CHUNKS_JSONL_PATH}")
        return m
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                m[d["chunk_id"]] = d["text"]
            except (json.JSONDecodeError, KeyError):
                continue
    logger.info(f"已加载{len(m)}条文本映射")
    return m


def load_data(collection):
    """
    作用：从向量JSONL读取数据，补回文本，批量导入Milvus
    原理：幂等设计——已有数据则跳过；向量文件已解耦——从chunk JSONL补text
    逻辑：
      1. 检查集合是否已有数据，有则跳过（幂等）
      2. 加载文本映射（补回text）
      3. 逐行解析向量JSONL，组装成列格式
      4. 校验所有向量维度一致
      5. 分批导入（每批MILVUS_BATCH_SIZE条）
    """
    existing = check_existing_data(collection)
    if existing > 0:
        logger.info(f"集合已有{existing}条数据，跳过导入（如需重建设REBUILD_COLLECTION=True）")
        return
    ip = Path(str(EMBEDDINGS_JSONL_PATH))
    if not ip.exists():
        logger.error(f"文件不存在: {EMBEDDINGS_JSONL_PATH}")
        return
    logger.info(f"加载向量数据: {EMBEDDINGS_JSONL_PATH}")
    text_map = load_text_map()
    cids, texts, pss, pes, secs, sps, embs = ([] for _ in range(7))
    with open(ip, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                cid = d["chunk_id"]
                cids.append(cid)
                texts.append(text_map.get(cid, ""))
                pss.append(d.get("page_start", 0))
                pes.append(d.get("page_end", 0))
                secs.append(d.get("section") or "")
                sps.append(d.get("source_pdf", ""))
                embs.append(d["embedding"])
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"第{ln}行解析失败: {e}")
    total = len(cids)
    logger.info(f"共加载{total}条")
    if not total:
        return
    # 校验：所有向量维度必须一致
    first_dim = len(embs[0])
    for i, e in enumerate(embs):
        if len(e) != first_dim:
            logger.error(f"第{i}条向量维度不一致: 期望{first_dim}, 实际{len(e)}")
            return
    logger.info(f"向量维度校验通过: {first_dim}维")
    for i in range(0, total, MILVUS_BATCH_SIZE):
        end = min(i + MILVUS_BATCH_SIZE, total)
        batch = [cids[i:end], texts[i:end], pss[i:end], pes[i:end], secs[i:end], sps[i:end], embs[i:end]]
        try:
            collection.insert(batch)
            logger.info(f"已导入{end}/{total}条")
        except Exception as e:
            logger.error(f"第{i}-{end}条导入失败: {e}")
    collection.flush()
    logger.info(f"导入完成，集合当前数据量: {collection.num_entities}")


def print_quality_report(collection):
    """
    作用：输出导入质量报告
    逻辑：显示集合名、数据量、向量维度
    """
    try:
        collection.load()
        logger.info("=" * 50)
        logger.info("  导入质量报告")
        logger.info("=" * 50)
        logger.info(f"集合: {MILVUS_COLLECTION}")
        logger.info(f"数据量: {collection.num_entities}条")
        logger.info(f"向量维度: {EMBEDDING_DIM}")
    except Exception as e:
        logger.warning(f"质量报告获取失败: {e}")


def main():
    """
    作用：程序入口，串联导入流程
    逻辑：连接Milvus→按REBUILD_COLLECTION决定重建或使用已有→导入→质量报告
    """
    logger.info("=" * 50)
    logger.info("  导入Milvus 开始")
    logger.info("=" * 50)
    if not connect_milvus():
        return
    if REBUILD_COLLECTION:
        if utility.has_collection(MILVUS_COLLECTION):
            logger.info("REBUILD_COLLECTION=True，删除旧集合...")
            utility.drop_collection(MILVUS_COLLECTION)
        c = create_collection()
        create_index(c)
    elif check_collection_exists(MILVUS_COLLECTION):
        logger.info(f"集合已存在，直接使用")
        c = Collection(MILVUS_COLLECTION)
    else:
        c = create_collection()
        create_index(c)
    load_data(c)
    print_quality_report(c)
    logger.info("🎉 导入完成")


if __name__ == "__main__":
    main()
