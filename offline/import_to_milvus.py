"""
import_to_milvus.py — 向量导入Milvus脚本（v2.0 多知识库支持）
===========================================
作用：读取向量JSONL，补回文本，导入Milvus向量数据库
原理：
  1. Milvus是高性能分布式向量数据库，支持十亿级向量检索
  2. 使用gRPC协议通信，比HTTP更高效
  3. 向量文件已与文本解耦，导入时需要从chunk JSONL补回text
  4. v2.0：遍历所有知识库，逐个导入到同一集合（实现"一起搜"）
  5. REBUILD开关：True时删除旧集合重新创建
输入：每个KB的 data/chunks/{kb_name}_带向量.jsonl + data/chunks/{kb_name}_分块.jsonl
输出：Milvus集合 prospectus（包含所有KB数据）
"""

import json
import sys
from pathlib import Path

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# =============================================================================
# 模块路径设置
# =============================================================================
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    get_kb_paths,
    KNOWLEDGE_BASES,
    MILVUS_HOST,
    MILVUS_PORT,
    MILVUS_COLLECTION,
    MILVUS_BATCH_SIZE,
    EMBEDDING_DIM,
    REBUILD_COLLECTION,
    RUN_TIMESTAMP,
    setup_logger,
)

logger = setup_logger("import_to_milvus")


# =============================================================================
# 连接函数
# =============================================================================
def connect_milvus() -> bool:
    logger.info(f"连接Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    try:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        logger.info("Milvus连接成功")
        return True
    except Exception as e:
        logger.error(f"连接失败: {e}")
        return False


# =============================================================================
# 集合创建函数
# =============================================================================
def create_collection() -> Collection:
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=20000),
        FieldSchema(name="page_start", dtype=DataType.INT64),
        FieldSchema(name="page_end", dtype=DataType.INT64),
        FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="source_pdf", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="table_index", dtype=DataType.INT64),
        FieldSchema(name="kb_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]
    schema = CollectionSchema(fields, description="招股说明书问答系统向量库（多知识库）")
    c = Collection(name=MILVUS_COLLECTION, schema=schema)
    logger.info(f"集合 '{MILVUS_COLLECTION}' 创建成功")
    return c


# =============================================================================
# 索引创建函数
# =============================================================================
def create_index(collection):
    params = {
        "metric_type": "IP",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    try:
        collection.create_index(field_name="embedding", index_params=params)
        logger.info("向量索引创建成功")
    except Exception as e:
        logger.warning(f"索引可能已存在: {e}")


# =============================================================================
# 加载单个KB的数据并导入
# =============================================================================
def load_kb_data(collection, kb_name: str):
    """加载单个知识库的数据并导入到Milvus集合

    参数：
        collection: Milvus Collection对象
        kb_name: 知识库名称
    """
    kb_paths = get_kb_paths(kb_name)
    embeddings_path = kb_paths["embeddings_jsonl_path"]
    chunks_path = kb_paths["chunks_jsonl_path"]

    # 检查向量文件
    ep = Path(str(embeddings_path))
    if not ep.exists():
        logger.warning(f"知识库 '{kb_name}' 向量文件不存在: {embeddings_path}，跳过")
        return

    # 加载文本映射
    text_map = {}
    cp = Path(str(chunks_path))
    if cp.exists():
        with open(cp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    text_map[d["chunk_id"]] = d["text"]
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info(f"  已加载 {len(text_map)} 条文本映射")
    else:
        logger.warning(f"  chunk文件不存在: {chunks_path}，导入时text字段将为空")

    # 解析向量JSONL，组装列式数据
    cids, texts, pss, pes, secs, sps, ctypes, tidxs, kbnames, embs = (
        [] for _ in range(10)
    )

    with open(ep, "r", encoding="utf-8") as f:
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
                ctypes.append(d.get("type", "text"))
                tidxs.append(d.get("table_index", 0))
                kbnames.append(kb_name)
                embs.append(d["embedding"])
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"  kb '{kb_name}' 第{ln}行解析失败: {e}")

    total = len(cids)
    logger.info(f"  知识库 '{kb_name}' 加载 {total} 条")

    if total == 0:
        return

    # 向量维度校验
    first_dim = len(embs[0])
    for i, e in enumerate(embs):
        if len(e) != first_dim:
            logger.error(f"    第{i}条向量维度不一致: 期望{first_dim}, 实际{len(e)}")
            return

    # 批量导入
    for i in range(0, total, MILVUS_BATCH_SIZE):
        end = min(i + MILVUS_BATCH_SIZE, total)
        batch = [
            cids[i:end],
            texts[i:end],
            pss[i:end],
            pes[i:end],
            secs[i:end],
            sps[i:end],
            ctypes[i:end],
            tidxs[i:end],
            kbnames[i:end],
            embs[i:end],
        ]
        try:
            collection.insert(batch)
            logger.info(f"  kb '{kb_name}' 已导入 {end}/{total} 条")
        except Exception as e:
            logger.error(f"  kb '{kb_name}' 第{i}-{end}条导入失败: {e}")

    collection.flush()
    logger.info(f"  知识库 '{kb_name}' 导入完成")


# =============================================================================
# 质量报告函数
# =============================================================================
def print_quality_report(collection):
    try:
        collection.load()
        logger.info("=" * 50)
        logger.info("  导入质量报告")
        logger.info("=" * 50)
        logger.info(f"集合: {MILVUS_COLLECTION}")
        logger.info(f"总数据量: {collection.num_entities}条")
        logger.info(f"向量维度: {EMBEDDING_DIM}")
        logger.info(f"知识库: {list(KNOWLEDGE_BASES.keys())}")
    except Exception as e:
        logger.warning(f"质量报告获取失败: {e}")


# =============================================================================
# 程序入口
# =============================================================================
def main():
    logger.info("=" * 50)
    logger.info("  导入Milvus 开始（多知识库模式）")
    logger.info("=" * 50)

    if not connect_milvus():
        return

    # 处理集合（重建或复用）
    if REBUILD_COLLECTION:
        if utility.has_collection(MILVUS_COLLECTION):
            logger.info("REBUILD_COLLECTION=True，删除旧集合...")
            utility.drop_collection(MILVUS_COLLECTION)
        c = create_collection()
        create_index(c)
    elif utility.has_collection(MILVUS_COLLECTION):
        logger.info("集合已存在，直接使用")
        c = Collection(MILVUS_COLLECTION)
    else:
        c = create_collection()
        create_index(c)

    # 遍历所有知识库，逐个导入
    for kb_name in KNOWLEDGE_BASES:
        logger.info(f"")
        logger.info(f"--- 导入知识库: {kb_name} ---")
        load_kb_data(c, kb_name)

    print_quality_report(c)
    logger.info("🎉 导入完成，所有知识库已就绪")


if __name__ == "__main__":
    main()
