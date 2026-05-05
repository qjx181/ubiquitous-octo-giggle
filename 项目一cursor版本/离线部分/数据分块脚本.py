# -*- coding: utf-8 -*-
"""
RAG 专用 —— 文本分块工具
功能：清洗 + 分段 + 分块 + 导出标准格式
适合：法律条文、合同、政策、手册、PDF 提取后的文本
"""
import json
import re
from pathlib import Path

# ====================== 【你只需要改这里】 ======================
INPUT_TEXT_PATH = r"C:\Users\qjx\Desktop\刑法清洗结果.jsonl"    # 你的文本文件
OUTPUT_CHUNK_PATH = r"C:\Users\qjx\Desktop\rag_chunks.jsonl"  # 输出分块结果

# RAG 分块参数（法律文档推荐值）
MAX_CHUNK_LENGTH = 1000    # 每块最大长度
CHUNK_OVERLAP = 150        # 块之间重叠（防止断句）
# =================================================================

# ====================== 清洗规则 ======================
CLEAN_RULES = [
    re.compile(r"[\u3000\xa0\t ]+"),                # 多余空格
    re.compile(r"\n{3,}"),                          # 多余换行
    re.compile(r"^(第\s*\d+\s*页|\d+)$", re.M),     # 页码
    re.compile(r"^(目录|版权|免责|附录|声明)$", re.M), # 页眉页脚
]

def clean_text(text: str) -> str:
    """清洗脏文本"""
    if not text:
        return ""
    text = text.replace("\r\n", "\n")
    for rule in CLEAN_RULES:
        text = rule.sub(" ", text)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)

# ====================== RAG 智能分块 ======================
def split_rag_chunks(text: str) -> list[str]:
    """
    智能分块：
    1. 优先按段落分
    2. 过长段落自动切分
    3. 自动添加重叠
    """
    text = clean_text(text)
    if not text:
        return []

    # 按段落分割
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        # 段落太长 → 强制分割
        if len(para) > MAX_CHUNK_LENGTH:
            if current:
                chunks.append(current.strip())
                current = ""
            # 长文本滑动窗口切分
            start = 0
            while start < len(para):
                end = min(start + MAX_CHUNK_LENGTH, len(para))
                chunk = para[start:end].strip()
                chunks.append(chunk)
                start = max(end - CHUNK_OVERLAP, start + 1)
            continue

        # 正常段落：尝试合并
        combined = current + "\n\n" + para if current else para
        if len(combined) <= MAX_CHUNK_LENGTH:
            current = combined
        else:
            if current:
                chunks.append(current.strip())
            current = para

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 50]  # 过滤太短的块

# ====================== 导出 RAG 标准格式 ======================
def save_rag_chunks(chunks: list[str], out_path: str | Path):
    """导出 JSONL，直接可用于向量库检索"""
    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True)

    rag_data = []
    for i, chunk in enumerate(chunks, 1):
        rag_data.append({
            "chunk_id": f"chunk_{i:04d}",
            "content": chunk,
            "text": chunk,
            "length": len(chunk),
            "source": Path(INPUT_TEXT_PATH).name,
            "tags": ["法律", "RAG知识库", "刑法"]
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for item in rag_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 分块完成！共生成 {len(rag_data)} 块")
    print(f"📁 已保存到：{out_path}")
    print("🧩 可直接导入：Chroma / FAISS / Milvus 等向量库")

# ====================== 主程序 ======================
if __name__ == "__main__":
    print("🔍 开始读取文本并进行 RAG 分块...")
    with open(INPUT_TEXT_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = split_rag_chunks(content)
    save_rag_chunks(chunks, OUTPUT_CHUNK_PATH)