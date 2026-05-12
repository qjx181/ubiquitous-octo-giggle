import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer



# =========================
# 路径配置
# =========================
INPUT_JSONL = r"C:\Users\qjx\Desktop\rag_chunks.jsonl"
OUTPUT_JSONL = r"C:\Users\qjx\Desktop\向量.jsonl"
MODEL_PATH = r"C:\Users\qjx\.cache\modelscope\hub\models\BAAI\bge-m3"

import torch
print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
print("GPU设备名称:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无可用GPU")
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(records, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    # 校验输入文件
    input_path = Path(INPUT_JSONL)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    # ========== GPU 设置 ==========
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ 使用设备：{device}")
    # 打印GPU信息（验证）
    if device == "cuda":
        print(f"📌 GPU型号：{torch.cuda.get_device_name(0)}")

    # ========== 模型加载 ==========
    print("正在加载模型...")
    # 校验模型路径
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    # 加载BGE-M3模型（适配版本+设备）
    model = SentenceTransformer(
        MODEL_PATH,
        device=device,
        trust_remote_code=True  # 适配BGE-M3自定义模型
    )

    # ========== 数据处理 ==========
    print("正在读取清洗后的数据...")
    records = load_jsonl(INPUT_JSONL)
    if not records:
        raise ValueError("输入文件无有效数据！")

    # 拼接标题+内容
    texts = []
    for rec in records:
        title = rec.get("title", "")
        content = rec.get("content", "")
        text = f"{title}\n{content}".strip()
        texts.append(text)

    # ========== 文本向量化 ==========
    print(f"开始向量化，共 {len(texts)} 条...")
    embeddings = model.encode(
        texts,
        batch_size=16,  # 适配4060显存，可根据情况调整
        show_progress_bar=True,
        normalize_embeddings=True,
        device=device
    )

    # ========== 保存结果 ==========
    # 写入向量（确保浮点格式）
    for rec, emb in zip(records, embeddings):
        rec["embedding"] = np.asarray(emb, dtype=np.float32).tolist()

    print("正在保存结果...")
    save_jsonl(records, OUTPUT_JSONL)
    print(f"✅ 完成！输出文件：{OUTPUT_JSONL}")


if __name__ == "__main__":
    main()