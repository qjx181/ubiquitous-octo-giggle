import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from time import perf_counter

from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer

from config import (
    MILVUS_HOST,
    MILVUS_PORT,
    BGE_M3_PATH,
    BM25_TOP_K,
    VECTOR_TOP_K,
    RERANK_TOP_K,
    DATA_JSONL,
    DOCTOR_DATA_JSONL,
    COLLECTION_NAME,
    DOCTOR_COLLECTION_NAME,
)
from data_loader import load_jsonl, build_bm25_index, tokenize, filter_bm25_items
from reranker import load_reranker, rerank_candidates
from web_fallback import is_professional_query, crawl_web_knowledge, web_collection_name
from services.knowledge_store import search_chat_knowledge

logger = logging.getLogger(__name__)
RETRIEVAL_EXECUTOR = ThreadPoolExecutor(max_workers=4)


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(BGE_M3_PATH)


@lru_cache(maxsize=1)
def get_reranker_model():
    from config import BGE_RERANK_PATH
    return load_reranker(BGE_RERANK_PATH)


@lru_cache(maxsize=2)
def get_collection(collection_name: str):
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    return Collection(collection_name)


@lru_cache(maxsize=8192)
def cached_encode(query: str):
    return tuple(get_embedding_model().encode([query], normalize_embeddings=True)[0].tolist())


@lru_cache(maxsize=4096)
def cached_semantic_search(query: str, jsonl_path: str, role: str):
    items = get_bm25_items(jsonl_path)
    bm25 = get_bm25_index(jsonl_path)
    if not items or bm25 is None:
        return []
    q_tokens = tokenize(query)
    scores = bm25.get_scores(q_tokens)
    ranked = sorted(zip(items, scores), key=lambda x: x[1], reverse=True)[:BM25_TOP_K]
    if not ranked:
        return []
    max_score = ranked[0][1]
    min_score = ranked[-1][1]
    denom = float(max_score - min_score) or 1.0
    results = []
    for item, score in ranked:
        norm_score = max(0.0, min(1.0, float((float(score) - float(min_score)) / denom)))
        if role == "律师":
            mapped = {"id": item.get("id", ""), "title": item.get("title", ""), "content": item.get("text", ""), "page": item.get("article_no", item.get("chunk_index", "")), "source_file": item.get("source", ""), "path": item.get("path", "")}
        else:
            mapped = {"id": item.get("id", ""), "title": item.get("title", ""), "content": item.get("content") or item.get("text") or "", "page": item.get("page", item.get("page_start", "")), "source_file": item.get("source_file", "")}
        results.append((mapped, norm_score, "semantic"))
    return results


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split())


def _item_dedupe_key(item: dict) -> str:
    content = _normalize_text(item.get("content") or item.get("text") or "")[:240]
    title = _normalize_text(item.get("title") or "")[:120]
    source = _normalize_text(item.get("source_file") or item.get("path") or "")[:160]
    return f"{source}|{title}|{content}"


@lru_cache(maxsize=4096)
def cached_vector_search(query: str, collection_name: str, role: str):
    collection = get_collection(collection_name)
    try:
        collection.load()
    except Exception:
        logger.exception("collection load failed collection=%s", collection_name)
        return []
    q_vec = list(cached_encode(query))
    if role == "律师":
        output_fields = ["id", "article_no", "path", "text"]
        candidate_fields = ["vector", "embedding"]
    else:
        output_fields = ["id", "title", "content", "page", "source_file"]
        candidate_fields = ["embedding", "vector"]
    for anns_field in candidate_fields:
        try:
            results = collection.search(data=[q_vec], anns_field=anns_field, param={"metric_type": "COSINE", "params": {}}, limit=min(VECTOR_TOP_K, 10), output_fields=output_fields)
            hits = []
            if results and results[0]:
                for hit in results[0]:
                    entity = hit.entity
                    if role == "律师":
                        text_value = entity.get("text", "")
                        item = {"id": entity.get("id", ""), "title": entity.get("article_no", ""), "content": text_value, "text": text_value, "preview": text_value, "excerpt": text_value, "page": entity.get("article_no", ""), "source_file": entity.get("path", ""), "path": entity.get("path", "")}
                    else:
                        content_value = entity.get("content", "")
                        item = {"id": entity.get("id", ""), "title": entity.get("title", ""), "content": content_value, "text": content_value, "preview": content_value, "excerpt": content_value, "page": entity.get("page", ""), "source_file": entity.get("source_file", "")}
                    hits.append((item, float(hit.score), "vector"))
            if hits:
                return hits
        except Exception:
            logger.exception("vector search failed collection=%s field=%s", collection_name, anns_field)
            continue
    return []


@lru_cache(maxsize=4096)
def cached_web_search(query: str, role: str):
    return tuple(crawl_web_knowledge(query, role) or [])


@lru_cache(maxsize=4096)
def cached_web_milvus_search(query: str, role: str):
    collection_name = web_collection_name(role)
    collection = get_collection(collection_name)
    try:
        collection.load()
    except Exception:
        logger.exception("web collection load failed collection=%s", collection_name)
        return []
    q_vec = list(cached_encode(query))
    output_fields = ["id", "title", "content", "page", "source_file", "url", "fetched_at"]
    try:
        results = collection.search(data=[q_vec], anns_field="embedding", param={"metric_type": "COSINE", "params": {}}, limit=WEB_TOP_K if "WEB_TOP_K" in globals() else 3, output_fields=output_fields)
        hits = []
        if results and results[0]:
            for hit in results[0]:
                entity = hit.entity
                content_value = entity.get("content", "")
                item = {
                    "id": entity.get("id", ""),
                    "title": entity.get("title", ""),
                    "content": content_value,
                    "text": content_value,
                    "preview": content_value,
                    "excerpt": content_value,
                    "page": entity.get("page", ""),
                    "source_file": entity.get("source_file", ""),
                    "url": entity.get("url", ""),
                    "fetched_at": entity.get("fetched_at", ""),
                }
                hits.append((item, float(hit.score), "web_memory"))
        return hits
    except Exception:
        logger.exception("web milvus search failed collection=%s", collection_name)
        return []


@lru_cache(maxsize=2)
def get_jsonl_items(jsonl_path: str):
    if not jsonl_path or not Path(jsonl_path).exists():
        return []
    return load_jsonl(jsonl_path)


@lru_cache(maxsize=2)
def get_bm25_index(jsonl_path: str):
    items = get_jsonl_items(jsonl_path)
    if not items:
        return None
    return build_bm25_index(items)


@lru_cache(maxsize=2)
def get_bm25_items(jsonl_path: str):
    items = get_jsonl_items(jsonl_path)
    return filter_bm25_items(items) if items else []


@lru_cache(maxsize=2)
def get_query_jsonl_path(role: str) -> str:
    if role == "律师":
        return DATA_JSONL
    if role == "医生":
        return DOCTOR_DATA_JSONL
    return ""


@lru_cache(maxsize=2)
def get_query_collection_name(role: str) -> str:
    if role == "律师":
        return COLLECTION_NAME
    if role == "医生":
        return DOCTOR_COLLECTION_NAME
    return ""


@lru_cache(maxsize=2)
def get_query_items(role: str):
    jsonl_path = get_query_jsonl_path(role)
    return get_jsonl_items(jsonl_path) if jsonl_path else []


def is_greeting_or_smalltalk(query: str) -> bool:
    text = (query or "").strip().lower()
    if not text:
        return True
    smalltalk_keywords = ["你好", "您好", "hello", "hi", "嗨", "早上好", "下午好", "晚上好", "谢谢", "再见", "拜拜"]
    return len(text) <= 12 and any(k in text for k in smalltalk_keywords)


def is_knowledge_query(query: str, role: str) -> bool:
    if is_greeting_or_smalltalk(query):
        return False
    jsonl_path = get_query_jsonl_path(role)
    if not jsonl_path or not Path(jsonl_path).exists():
        return False
    if role == "医生":
        medical_keywords = ["高血压", "血压", "收缩压", "舒张压", "降压", "降压药", "氨氯地平", "硝苯地平", "缬沙坦", "厄贝沙坦", "氯沙坦", "美托洛尔", "利尿剂", "acei", "arb", "钙拮抗剂", "心脑血管", "靶器官", "低盐", "限盐", "血压计", "动脉压"]
        return any(k in query.strip().lower() for k in medical_keywords)
    return is_professional_query(query, role)


def should_use_local_knowledge(query: str, role: str) -> bool:
    return is_knowledge_query(query, role)


def is_web_query(query: str, role: str) -> bool:
    if is_greeting_or_smalltalk(query):
        return False
    return is_professional_query(query, role)


def build_context(query: str, role: str):
    start_time = perf_counter()
    collection_name = get_query_collection_name(role)
    jsonl_path = get_query_jsonl_path(role)
    items = get_query_items(role)
    if not collection_name:
        return "", [], 0.0, 0, 0, 0, 0.0
    if is_greeting_or_smalltalk(query):
        return "", [], perf_counter() - start_time, len(items), 0, 0, 0.0
    if not (jsonl_path and Path(jsonl_path).exists()) and not utility.has_collection(collection_name):
        return "", [], perf_counter() - start_time, len(items), 0, 0, 0.0

    use_local = should_use_local_knowledge(query, role)
    futures = {
        "semantic": RETRIEVAL_EXECUTOR.submit(cached_semantic_search, query, jsonl_path, role) if use_local and jsonl_path else None,
        "vector": RETRIEVAL_EXECUTOR.submit(cached_vector_search, query, collection_name, role) if use_local else None,
        "web_memory": RETRIEVAL_EXECUTOR.submit(cached_web_milvus_search, query, role) if is_knowledge_query(query, role) else None,
        "web": RETRIEVAL_EXECUTOR.submit(cached_web_search, query, role) if is_web_query(query, role) else None,
        "chat_kb": RETRIEVAL_EXECUTOR.submit(search_chat_knowledge, query, get_embedding_model(), 5),
    }
    results = []
    for name, fut in futures.items():
        if fut is None:
            continue
        try:
            data = fut.result()
            if name == "web":
                results.extend([(item, 1.0, "web_memory") for item in data])
            else:
                results.extend(data)
        except Exception:
            logger.exception("retrieval task failed name=%s role=%s", name, role)

    semantic_count = len([r for r in results if r[2] == "semantic"])
    vector_count = len([r for r in results if r[2] in ("vector", "web_memory", "chat_kb")])
    if not results:
        return "", [], perf_counter() - start_time, len(items), semantic_count, vector_count, 0.0

    # 去重，避免相同片段重复进入重排和 prompt
    deduped = []
    seen = set()
    for item in results:
        key = _item_dedupe_key(item[0])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    # 先按基础分预筛，减少重排器负担，提高速度
    deduped.sort(key=lambda x: x[1], reverse=True)
    rerank_input = deduped[: max(RERANK_TOP_K * 3, 12)]
    if not rerank_input:
        return "", [], perf_counter() - start_time, len(items), semantic_count, vector_count, 0.0

    if len(rerank_input) == 1:
        ranked = [(rerank_input[0], rerank_input[0][1])]
    else:
        reranker = get_reranker_model()
        ranked = rerank_candidates(query, rerank_input, reranker, top_k=RERANK_TOP_K)

    context_parts, refs = [], []
    for idx, ((item, base_score, source), rr_score) in enumerate(ranked, start=1):
        title = item.get("title", "")
        content = item.get("content", "")
        page = item.get("page", "")
        source_file = item.get("source_file", "")
        short_content = _normalize_text(content)[:1200]
        context_parts.append(f"[{idx}] 类型：{source}\n来源：{source_file} 第{page}页\n标题：{title}\n内容：{short_content}")
        refs.append({"rank": idx, "id": str(item.get("id") or ""), "title": title, "page": page, "source_file": source_file, "base_score": base_score, "rerank_score": rr_score, "source": source, "source_label": source, "content": short_content, "text": short_content, "excerpt": short_content[:800], "preview": short_content[:500], "engine": item.get("engine", "")})
    best_score = max([base_score for (_, base_score, _) in results], default=0.0)
    return "\n\n".join(context_parts), refs, perf_counter() - start_time, len(items), semantic_count, vector_count, best_score
