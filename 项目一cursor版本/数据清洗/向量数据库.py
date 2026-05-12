import json
from pathlib import Path

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# =========================
# 配置区
# =========================
DATA_PATH = r"C:\Users\qjx\Desktop\2309b\专高阶段\专高六\项目一cursor版本\数据清洗\中华人民共和国民法典_chunks_with_vectors.jsonl"
COLLECTION_NAME = "civil_code_rag"

MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"

VECTOR_DIM = 1024
BATCH_SIZE = 1000


# =========================
# 基础工具
# =========================
def load_data(path):
    """
    支持 json 和 jsonl
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError("JSON 文件内容必须是数组列表格式")

    else:
        raise ValueError(f"不支持的文件格式: {suffix}，仅支持 .json / .jsonl")


def chunk_list(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def create_or_get_collection():
    """
    如果 collection 存在就直接返回；
    不存在则创建。
    """
    if utility.has_collection(COLLECTION_NAME):
        print(f"Collection `{COLLECTION_NAME}` 已存在，直接使用。")
        return Collection(COLLECTION_NAME)

    print(f"Collection `{COLLECTION_NAME}` 不存在，开始创建。")

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=100
        ),
        FieldSchema(
            name="article_no",
            dtype=DataType.VARCHAR,
            max_length=50
        ),
        FieldSchema(
            name="path",
            dtype=DataType.VARCHAR,
            max_length=500
        ),
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=65535
        ),
        FieldSchema(
            name="vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=VECTOR_DIM
        ),
    ]

    schema = CollectionSchema(
        fields=fields,
        description="中华人民共和国民法典 RAG 知识库"
    )

    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema
    )

    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {
            "M": 16,
            "efConstruction": 200
        }
    }

    collection.create_index(
        field_name="vector",
        index_params=index_params
    )

    print("Collection 创建完成。")
    return collection


def collection_has_data(collection):
    """
    判断 collection 是否为空
    """
    try:
        return collection.num_entities > 0
    except Exception:
        # 某些版本建议先 load 后再查
        collection.load()
        return collection.num_entities > 0


def insert_items(collection, items):
    ids = []
    article_nos = []
    paths = []
    texts = []
    vectors = []

    for item in items:
        vec = item.get("vector")
        if vec is None:
            continue

        ids.append(str(item.get("id", "")))
        article_nos.append(str(item.get("article_no", "")))
        paths.append(str(item.get("path", "")))
        texts.append(str(item.get("text", "")))
        vectors.append(vec)

    if not ids:
        return 0

    data = [ids, article_nos, paths, texts, vectors]
    collection.insert(data)
    collection.flush()
    return len(ids)


def import_data_if_needed(collection, items):
    """
    如果 collection 里已经有数据，就跳过；
    没数据才导入。
    """
    collection.load()
    if collection_has_data(collection):
        print(f"Collection `{COLLECTION_NAME}` 已有数据，跳过导入。")
        return

    print(f"Collection `{COLLECTION_NAME}` 为空，开始导入数据。")
    total_inserted = 0

    for batch in chunk_list(items, BATCH_SIZE):
        inserted = insert_items(collection, batch)
        total_inserted += inserted
        print(f"本批导入: {inserted} 条")

    collection.load()
    print(f"导入完成，总计导入: {total_inserted} 条")


# =========================
# 检索测试函数
# =========================
def test_search(collection, query_vector, top_k=5):
    """
    用传入的 query_vector 做一次测试检索
    """
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=top_k,
        output_fields=["id", "article_no", "path", "text"]
    )

    print("\n====== 检索测试结果 ======")
    for i, hit in enumerate(results[0], start=1):
        entity = hit.entity
        print(f"\n[{i}] score = {hit.score:.4f}")
        print(f"id: {entity.get('id')}")
        print(f"条号: {entity.get('article_no')}")
        print(f"路径: {entity.get('path')}")
        print(f"正文: {entity.get('text')[:200]}")
        print("-" * 80)


# =========================
# 主流程
# =========================
def main():
    print("Connecting to Milvus...")
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )

    print(f"Loading data from: {DATA_PATH}")
    items = load_data(DATA_PATH)
    print(f"Loaded {len(items)} records")

    collection = create_or_get_collection()

    # 如果 collection 为空则导入
    import_data_if_needed(collection, items)

    # 如果你想做检索测试，需要外部传入 query_vector
    # 这里先给一个提示
    print("\nMilvus collection is ready.")



if __name__ == "__main__":
    main()