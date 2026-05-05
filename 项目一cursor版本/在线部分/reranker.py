from sentence_transformers import CrossEncoder


def build_candidate_text(item):
    title = item.get("title") or item.get("article_no") or ""
    source = item.get("source_file") or item.get("path") or item.get("source") or ""
    text = item.get("content") or item.get("text") or ""
    return f"{source}\n{title}\n{text}".strip()


def rerank_candidates(query, candidates, reranker_model, top_k=5):
    docs = [build_candidate_text(item) for item, _, _ in candidates]
    pairs = [[query, doc] for doc in docs]
    scores = reranker_model.predict(pairs, convert_to_numpy=True).tolist()

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked[:top_k]


def load_reranker(model_path):
    # 使用 CrossEncoder 避免 FlagEmbedding 与 transformers 的接口兼容问题
    return CrossEncoder(model_path)
