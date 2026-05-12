# -*- coding: utf-8 -*-
"""
PDF 文本清洗 + 分块 + 导出 JSONL（法律文档专用）
PyCharm 直接运行，无需命令行！
"""
import json
import re
from pathlib import Path

import fitz  # pymupdf

# ====================== 【只需改这里！】 ======================
PDF_PATH = r"C:\Users\qjx\Desktop\2309b\专高阶段\专高六\项目一cursor版本\离线部分\中华人民共和国刑法_20201226.pdf"  # 你的 PDF 路径
OUTPUT_JSONL = r"C:\Users\qjx\Desktop\刑法清洗结果.jsonl"         # 输出路径
# ==============================================================

# 清洗规则
WHITESPACE_RE = re.compile(r"[\u3000\xa0\t ]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
PAGE_NO_RE = re.compile(r"^(第\s*\d+\s*页|\d+)$")
HEADER_FOOTER_RE = re.compile(r"^(目录|参考文献|附录|免责声明|版权所有|.*法律.*声明.*)$")


def clean_text(text: str) -> str:
    """清洗文本：去空格、去多余换行、去页码页眉"""
    if not isinstance(text, str):
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WHITESPACE_RE.sub(" ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)

    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if PAGE_NO_RE.fullmatch(line):
            continue
        if HEADER_FOOTER_RE.fullmatch(line):
            continue
        lines.append(line)

    return "\n".join(lines).strip()


def extract_pdf_text(pdf_path: str | Path) -> list[dict]:
    """提取 PDF 每页文本"""
    pages = []
    doc = fitz.open(str(pdf_path))
    for idx, page in enumerate(doc, 1):
        txt = page.get_text("text") or ""
        cleaned = clean_text(txt)
        pages.append({"page": idx, "content": cleaned})
    return pages


def split_chunks(text: str, max_len=800, overlap=80) -> list[str]:
    """智能分块"""
    text = clean_text(text)
    if not text:
        return []

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    buffer = ""

    for p in paras:
        if len(p) > max_len:
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
            i = 0
            while i < len(p):
                end = min(i + max_len, len(p))
                chunks.append(p[i:end].strip())
                i = max(end - overlap, i + 1)
            continue

        candidate = buffer + "\n\n" + p if buffer else p
        if len(candidate) <= max_len:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer.strip())
            buffer = p

    if buffer:
        chunks.append(buffer.strip())
    return [c for c in chunks if c]


def save_jsonl(pages: list[dict], out_path: str | Path):
    """保存为 JSONL"""
    items = []
    pdf_name = Path(PDF_PATH).stem

    for page in pages:
        p = page["page"]
        content = page["content"]
        if not content:
            continue

        for idx, chunk in enumerate(split_chunks(content), 1):
            items.append({
                "id": f"{pdf_name}_p{p}_c{idx}",
                "source": pdf_name,
                "page": p,
                "chunk": idx,
                "content": chunk,
                "text": chunk,
                "tags": ["法律", "刑法", "法规"]
            })

    with open(out_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 导出完成！共 {len(items)} 条")
    print(f"📁 文件保存至：{out_path}")


if __name__ == '__main__':
    print("🔍 开始提取 PDF 文本...")
    pages = extract_pdf_text(PDF_PATH)
    save_jsonl(pages, OUTPUT_JSONL)