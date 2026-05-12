# -*- coding: utf-8 -*-
"""
适配 embedding 字段的 Milvus 导入脚本
不删除原有数据，只增量导入
"""
import json
import uuid
from pathlib import Path
from pymilvus import MilvusClient, DataType

# ====================== 【你只需要改这里】 ======================
JSONL_PATH = r"C:\Users\qjx\Desktop\向量.jsonl"  # 你的文件路径
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "civil_code_rag"
VECTOR_DIM = 1024  # 和你向量维度一致
# ===============================================================

def load_jsonl_data(jsonl_path):
    """加载 JSONL 数据"""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def init_milvus_client():
    """连接 Milvus"""
    client = MilvusClient(uri=MILVUS_URI)
    return client

def create_collection_if_not_exist(client, collection_name, dim):
    """创建集合（不存在才创建，不删原有数据）"""
    if client.has_collection(collection_name):
        print(f"✅ 集合已存在：{collection_name}，不删除，直接增量导入")
        return

    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=True
    )

    schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=256, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="tags", datatype=DataType.VARCHAR, max_length=256)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )
    print(f"✅ 新建集合：{collection_name}")

def insert_to_milvus(client, collection_name, jsonl_data):
    """增量导入（适配你的 embedding 字段）"""
    insert_list = []

    for item in jsonl_data:
        insert_list.append({
            "id": str(uuid.uuid4()),
            "vector": item["embedding"],  # 适配你的 embedding 字段
            "text": item.get("content", item.get("text", "")),
            "source": item.get("source", "unknown"),
            "tags": ",".join(item.get("tags", []))
        })

    res = client.insert(collection_name=collection_name, data=insert_list)
    print(f"✅ 导入成功！本次插入数量：{res['insert_count']}")
    return res

if __name__ == "__main__":
    print("🔗 正在连接 Milvus...")
    client = init_milvus_client()

    print(f"📂 正在加载向量文件：{JSONL_PATH}")
    data = load_jsonl_data(JSONL_PATH)

    create_collection_if_not_exist(client, COLLECTION_NAME, VECTOR_DIM)
    insert_to_milvus(client, COLLECTION_NAME, data)

    print("\n🎉 全部完成！原有数据完全保留，已增量导入新向量")