import hashlib
import json
import logging
from datetime import datetime

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility, connections

from config import MILVUS_HOST, MILVUS_PORT

logger = logging.getLogger(__name__)

CHAT_KB_COLLECTION = "chat_knowledge_base"


def _connect():
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)


def ensure_chat_knowledge_collection(dim: int) -> Collection:
    _connect()
    if utility.has_collection(CHAT_KB_COLLECTION):
        return Collection(CHAT_KB_COLLECTION)

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="style", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="query", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="retrieved_context", dtype=DataType.VARCHAR, max_length=16384),
        FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=16384),
        FieldSchema(name="references_json", dtype=DataType.VARCHAR, max_length=16384),
        FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields=fields, description="聊天检索内容与最终答案知识库")
    collection = Collection(CHAT_KB_COLLECTION, schema=schema)
    collection.create_index(
        field_name="embedding",
        index_params={"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}},
    )
    return collection


def _row_to_ref(row: dict, score: float) -> tuple[dict, float, str]:
    content = row.get("answer") or row.get("retrieved_context") or row.get("query") or ""
    item = {
        "id": row.get("id", ""),
        "title": row.get("query", "聊天知识"),
        "content": content,
        "text": content,
        "preview": content,
        "excerpt": content,
        "page": row.get("created_at", ""),
        "source_file": row.get("session_id", ""),
        "url": "",
    }
    return item, float(score), "chat_kb"


def save_chat_knowledge_to_milvus(
    model,
    *,
    user_id: str,
    session_id: str,
    role: str,
    style: str,
    query: str,
    retrieved_context: str,
    answer: str,
    references: list | None = None,
) -> int:
    try:
        text_for_embedding = "\n".join([query or "", retrieved_context or "", answer or ""]).strip()
        if not text_for_embedding:
            return 0
        embedding = model.encode([text_for_embedding], normalize_embeddings=True)[0]
        dim = len(embedding)
        collection = ensure_chat_knowledge_collection(dim)
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        doc_id = hashlib.sha1(f"{user_id}|{session_id}|{query[:120]}|{answer[:120]}|{created_at}".encode("utf-8")).hexdigest()
        row = {
            "id": doc_id,
            "user_id": user_id[:128],
            "session_id": session_id[:128],
            "role": role[:32],
            "style": style[:64],
            "query": query[:4096],
            "retrieved_context": retrieved_context[:16384],
            "answer": answer[:16384],
            "references_json": json.dumps(references or [], ensure_ascii=False)[:16384],
            "created_at": created_at,
            "embedding": embedding.tolist() if hasattr(embedding, "tolist") else list(embedding),
        }
        collection.insert([row])
        collection.flush()
        try:
            collection.load()
        except Exception:
            pass
        return 1
    except Exception:
        logger.exception("save chat knowledge failed")
        return 0


def search_chat_knowledge(query: str, model, top_k: int = 5):
    try:
        if not query.strip():
            return []
        _connect()
        if not utility.has_collection(CHAT_KB_COLLECTION):
            return []
        collection = Collection(CHAT_KB_COLLECTION)
        collection.load()
        q_vec = list(model.encode([query], normalize_embeddings=True)[0].tolist())
        results = collection.search(
            data=[q_vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {}},
            limit=top_k,
            output_fields=["query", "retrieved_context", "answer", "created_at", "session_id", "role", "style"],
        )
        hits = []
        if results and results[0]:
            for hit in results[0]:
                entity = hit.entity
                row = {
                    "id": getattr(entity, "id", ""),
                    "query": entity.get("query", ""),
                    "retrieved_context": entity.get("retrieved_context", ""),
                    "answer": entity.get("answer", ""),
                    "created_at": entity.get("created_at", ""),
                    "session_id": entity.get("session_id", ""),
                    "role": entity.get("role", ""),
                    "style": entity.get("style", ""),
                }
                hits.append(_row_to_ref(row, float(hit.score)))
        return hits
    except Exception:
        logger.exception("search chat knowledge failed")
        return []

