import json
from pathlib import Path

INPUT_PATH = r"c:\Users\qjx\Desktop\高血压_clean.jsonl"
OUTPUT_PATH = r"c:\Users\qjx\Desktop\高血压_chunked.jsonl"

# 每块最大字符数
MAX_CHARS = 700
# 重叠字符数，便于 RAG 保留上下文
OVERLAP_CHARS = 120
# 最小块长度，太短的块会被丢弃
MIN_CHARS = 80


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")
    text = text.replace("\t", " ")
    text = "\n".join(line.strip() for line in text.splitlines())
    text = "\n".join(line for line in text.splitlines() if line)
    return text.strip()


def split_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP_CHARS):
    """
    按字符长度切块，保留重叠上下文。
    """
    text = normalize_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()

        if len(chunk) >= MIN_CHARS:
            chunks.append(chunk)

        if end >= n:
            break

        start = end - overlap
        if start < 0:
            start = 0

    return chunks


def main():
    input_file = Path(INPUT_PATH)
    output_file = Path(OUTPUT_PATH)

    if not input_file.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_file}")

    records = []
    with input_file.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"跳过第 {line_idx} 行，JSON 解析失败")
                continue

            title = item.get("title", "")
            content = item.get("content", "")
            source_file = item.get("source_file", "unknown")
            page = item.get("page", None)
            tags = item.get("tags", [])

            # 将标题和正文合并，便于语义分块
            full_text = ""
            if title and content:
                full_text = f"{title}\n{content}"
            elif content:
                full_text = content
            else:
                full_text = title

            chunks = split_text(full_text)

            for chunk_idx, chunk in enumerate(chunks, start=1):
                records.append({
                    "id": f"{Path(source_file).stem}_{page}_{chunk_idx}",
                    "source_file": source_file,
                    "page": page,
                    "chunk_index": chunk_idx,
                    "title": title,
                    "content": chunk,
                    "tags": tags,
                    "text_length": len(chunk),
                })

    with output_file.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"分块完成，共生成 {len(records)} 条 chunk。")
    print(f"输出文件: {output_file}")


if __name__ == "__main__":
    main()