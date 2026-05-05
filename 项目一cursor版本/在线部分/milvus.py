from pymilvus import connections, utility, Collection

MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"

COLLECTIONS = [
    "civil_code_rag",
    "hypertension_rag_chunks",
]

def print_collection_schema(name: str):
    print("=" * 80)
    print(f"Collection: {name}")

    if not utility.has_collection(name):
        print("状态: 不存在")
        return

    col = Collection(name)
    schema = col.schema

    print(f"状态: 存在")
    print(f"描述: {schema.description}")
    print(f"主键字段: {schema.primary_field.name if schema.primary_field else 'None'}")
    print(f"自动ID: {schema.auto_id}")
    print("字段列表:")

    for field in schema.fields:
        extra = []
        if field.is_primary:
            extra.append("PRIMARY")
        if field.auto_id:
            extra.append("AUTO_ID")
        if getattr(field, "dim", None):
            extra.append(f"dim={field.dim}")
        extra_text = f" ({', '.join(extra)})" if extra else ""
        print(f"  - {field.name}: {field.dtype}{extra_text}")

    try:
        indexes = col.indexes
        if indexes:
            print("索引:")
            for idx in indexes:
                print(f"  - field={idx.field_name}, index_name={idx.index_name}, params={idx.params}")
        else:
            print("索引: 无")
    except Exception as e:
        print(f"索引读取失败: {e}")

    try:
        print(f"实体数量: {col.num_entities}")
    except Exception as e:
        print(f"实体数量读取失败: {e}")

def main():
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    for name in COLLECTIONS:
        print_collection_schema(name)

if __name__ == "__main__":
    main()