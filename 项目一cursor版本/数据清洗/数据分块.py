import json
import re
from pathlib import Path

INPUT_JSON = r"C:\Users\qjx\Desktop\中华人民共和国民法典_rag.json"
OUTPUT_JSONL = r"C:\Users\qjx\Desktop\中华人民共和国民法典_chunks.jsonl"
OUTPUT_JSON = r"C:\Users\qjx\Desktop\中华人民共和国民法典_chunks.json"


MAX_CHUNK_LEN = 350   # 超过这个长度就继续切
MIN_CHUNK_LEN = 80    # 太短的句子尽量合并


def split_sentences(text: str):
    """
    按中文标点切句，保留句末标点。
    """
    if not text:
        return []

    # 先按句末标点分割
    parts = re.split(r"([。！？；])", text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentence = (parts[i] + parts[i + 1]).strip()
        if sentence:
            sentences.append(sentence)

    # 处理最后一段没有标点的情况
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())

    return sentences


def merge_short_sentences(sentences, min_len=MIN_CHUNK_LEN):
    """
    合并过短句子，避免切块太碎。
    """
    merged = []
    buffer = ""

    for sent in sentences:
        if not buffer:
            buffer = sent
            continue

        if len(buffer) < min_len:
            buffer += sent
        else:
            merged.append(buffer)
            buffer = sent

    if buffer:
        merged.append(buffer)

    return merged


def split_long_text(text: str, max_len=MAX_CHUNK_LEN):
    """
    把过长正文切成多个语义子块。
    """
    if len(text) <= max_len:
        return [text.strip()]

    sentences = split_sentences(text)
    sentences = merge_short_sentences(sentences)

    chunks = []
    current = ""

    for sent in sentences:
        if not current:
            current = sent
            continue

        if len(current) + len(sent) <= max_len:
            current += sent
        else:
            chunks.append(current.strip())
            current = sent

    if current.strip():
        chunks.append(current.strip())

    return chunks


def build_path(item):
    parts = []
    if item.get("part"):
        parts.append(f"{item.get('part')} {item.get('part_title') or ''}".strip())
    if item.get("chapter"):
        parts.append(f"{item.get('chapter')} {item.get('chapter_title') or ''}".strip())
    if item.get("section"):
        parts.append(f"{item.get('section')} {item.get('section_title') or ''}".strip())
    if item.get("article_no"):
        parts.append(item.get("article_no"))
    return " > ".join(parts)


def main():
    input_path = Path(INPUT_JSON)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = []

    for idx, item in enumerate(data, start=1):
        text = (item.get("text") or "").strip()
        if not text:
            continue

        # 超长条文切分
        sub_chunks = split_long_text(text)

        total = len(sub_chunks)

        for sub_idx, sub_text in enumerate(sub_chunks, start=1):
            chunk_id = f"civilcode-{idx}-{sub_idx}"

            path = item.get("path") or build_path(item)
            embedding_text = f"{path}\n{sub_text}"

            chunks.append({
                "id": chunk_id,
                "source": item.get("source"),
                "title": item.get("title", "中华人民共和国民法典"),
                "part": item.get("part"),
                "part_title": item.get("part_title"),
                "chapter": item.get("chapter"),
                "chapter_title": item.get("chapter_title"),
                "section": item.get("section"),
                "section_title": item.get("section_title"),
                "article_no": item.get("article_no"),
                "article_no_arabic": item.get("article_no_arabic"),
                "chunk_index": sub_idx,
                "chunk_total": total,
                "text": sub_text,
                "path": path,
                "embedding_text": embedding_text,
                "chunk_type": "article_chunk",
            })

    # 输出 JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # 输出 JSONL
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for item in chunks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done. Total chunks: {len(chunks)}")
    print(f"JSON saved to: {OUTPUT_JSON}")
    print(f"JSONL saved to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()