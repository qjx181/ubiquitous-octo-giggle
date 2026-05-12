# data_loader.py 完整修复版
import json
from rank_bm25 import BM25Okapi


def load_jsonl(jsonl_path):
    items = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
    except Exception:
        pass
    return items


def tokenize(text: str):
    if not text or not isinstance(text, str):
        return []
    return text.strip().split()


def _get_bm25_text(item):
    return item.get("text") or item.get("content") or ""


def filter_bm25_items(items):
    """返回和 BM25 索引一一对应的有效文档，避免分数与原始 items 错位。"""
    return [item for item in items if tokenize(_get_bm25_text(item))]


def build_bm25_index(items):
    # 过滤空文本，彻底避免 rank_bm25 内部除零
    valid_items = filter_bm25_items(items)
    corpus = [tokenize(_get_bm25_text(item)) for item in valid_items]

    if not corpus:
        return None

    return BM25Okapi(corpus)