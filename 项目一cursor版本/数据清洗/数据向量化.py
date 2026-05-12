import json
from pathlib import Path

import numpy as np
from FlagEmbedding import BGEM3FlagModel


DATA_DIR = Path(r"C:\Users\qjx\Desktop\2309b\专高阶段\专高六\项目一cursor版本\数据清洗")
MODEL_PATH = r"C:\Users\qjx\.cache\modelscope\hub\models\BAAI\bge-m3"

INPUT_JSONL = DATA_DIR / "中华人民共和国民法典_chunks.jsonl"
OUTPUT_JSONL = DATA_DIR / "中华人民共和国民法典_chunks_with_vectors.jsonl"

BATCH_SIZE = 16


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    print(f"Loading data from: {INPUT_JSONL}")
    data = load_jsonl(INPUT_JSONL)
    print(f"Loaded {len(data)} chunks")

    print(f"Loading BGE-M3 model from: {MODEL_PATH}")
    model = BGEM3FlagModel(
        MODEL_PATH,
        use_fp16=True
    )

    texts = [item["embedding_text"] for item in data]
    vectors = []

    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_no = i // BATCH_SIZE + 1
        print(f"Encoding batch {batch_no}/{total_batches}")

        result = model.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            max_length=8192,
            return_dense=True,
            return_sparse=False
        )

        dense_vecs = result["dense_vecs"]
        vectors.extend(dense_vecs)

    print("Attaching vectors to chunks...")
    for item, vec in zip(data, vectors):
        item["vector"] = np.asarray(vec, dtype=np.float32).tolist()

    print(f"Saving to: {OUTPUT_JSONL}")
    save_jsonl(OUTPUT_JSONL, data)

    print("Done.")


if __name__ == "__main__":
    main()