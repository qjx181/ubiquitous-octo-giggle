"""
诊断脚本：对失败题目做无section_filter的原始Milvus检索
运行：D:\an\envs\project2\python.exe eval\diag_retrieval.py
"""
import sys, json
sys.path.insert(0, ".")

from pymilvus import MilvusClient
from app.config import settings
from app.core.embedding import get_embedding_client

# 5道全挂的题
FAILING_QUESTIONS = [
    "公司的股权结构是怎样的？",
    "公司的毛利率是多少？变化趋势如何？",
    "公司的供应商有哪些？供应商集中度如何？",
    "公司的技术标准和专利情况如何？",
    "公司最近三年的营业收入分别是多少？",
]

# 对照组：正常通过的题
PASSING_QUESTIONS = [
    "公司的主要业务是什么？",
    "公司的主要产品和技术有哪些？",
]

def main():
    # 连接 Milvus
    uri = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    client = MilvusClient(uri=uri)
    collection = settings.MILVUS_COLLECTION

    # embedding 客户端
    emb_client = get_embedding_client()

    all_questions = FAILING_QUESTIONS + PASSING_QUESTIONS

    for q in all_questions:
        print(f"\n{'='*60}")
        print(f"问题: {q}")
        print(f"{'='*60}")

        # 生成 embedding
        emb = emb_client.encode([q])[0]

        # 无过滤检索 top 5
        results = client.search(
            collection_name=collection,
            data=[emb],
            anns_field="embedding",
            search_params={"metric_type": "IP", "params": {"nprobe": 16}},
            limit=5,
            output_fields=["chunk_id", "text", "section", "content_type", "page_start", "page_end", "kb_name"],
        )

        if not results or not results[0]:
            print("  无检索结果!")
            continue

        for i, hit in enumerate(results[0]):
            entity = hit.get("entity", {})
            score = hit.get("distance", 0)
            text = (entity.get("text") or "")[:200]
            section = entity.get("section", "?")
            ctype = entity.get("content_type", "?")
            pages = f"{entity.get('page_start', '?')}-{entity.get('page_end', '?')}"
            kb = entity.get("kb_name", "?")
            print(f"\n  [{i+1}] score={score:.4f}  section=[{section}]  type={ctype}  pages={pages}  kb={kb}")
            print(f"      {text}...")

    print("\n\n诊断完成。")

if __name__ == "__main__":
    main()
