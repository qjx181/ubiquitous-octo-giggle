import re
import json
from pathlib import Path

import fitz  # PyMuPDF


# =========================
# 直接写死路径
# =========================
INPUT_PDF = r"F:\qq\高血压.pdf"
OUTPUT_JSONL = r"F:\qq\高血压_clean.jsonl"


PAGE_NOISE_PATTERNS = [
    r"^\s*·\s*\d+\s*·\s*$",
    r"^\s*--\s*\d+\s+of\s+\d+\s*--\s*$",
    r"中华高血压杂志\(中英文\).*?$",
    r"ChinJHypertens.*?$",
    r"^\s*收稿日期:.*?$",
    r"^\s*责任编辑:.*?$",
]

REFERENCE_START_PATTERNS = [
    r"^\s*\[\d+\]",
    r"^\s*参考文献",
]

SECTION_TITLE_PATTERNS = [
    r"^\s*\d+\s+.+$",
    r"^\s*\d+\.\d+(\.\d+)?\s+.+$",
    r"^\s*要点\d+[A-Z]?\s+.+$",
    r"^\s*表\s*\d+\s+.+$",
    r"^\s*图\s*\d+\s+.+$",
]


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"[ \u2002\u2003\u2009]+", " ", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"(?<=[A-Za-z])\s+(?=[A-Za-z])", "", text)

    text = text.replace("mm Hg", "mmHg")
    text = text.replace("m g", "mg")
    text = text.replace("k g", "kg")
    text = text.replace("24 h", "24h")
    text = text.replace("2 h", "2h")
    text = text.replace("3 h", "3h")

    text = re.sub(r"(\d)\s*mmHg", r"\1mmHg", text)
    text = re.sub(r"(\d)\s*kg/m2", r"\1kg/m2", text)
    text = re.sub(r"(\d)\s*mL/\(min·1\.73\s*m2\)", r"\1mL/(min·1.73 m2)", text)
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    for pat in PAGE_NOISE_PATTERNS:
        if re.search(pat, s, flags=re.IGNORECASE):
            return True
    return False


def is_reference_start(line: str) -> bool:
    s = line.strip()
    for pat in REFERENCE_START_PATTERNS:
        if re.search(pat, s):
            return True
    return False


def is_section_title(line: str) -> bool:
    s = line.strip()
    if len(s) > 120:
        return False
    for pat in SECTION_TITLE_PATTERNS:
        if re.match(pat, s):
            return True
    return False


def merge_broken_lines(lines):
    merged = []
    buffer = ""

    def flush():
        nonlocal buffer
        if buffer.strip():
            merged.append(buffer.strip())
        buffer = ""

    for raw in lines:
        line = raw.strip()
        if not line:
            flush()
            continue

        if is_noise_line(line):
            continue

        if is_section_title(line):
            flush()
            merged.append(line)
            continue

        if is_reference_start(line):
            flush()
            merged.append("__REFERENCE_START__")
            merged.append(line)
            continue

        if not buffer:
            buffer = line
        else:
            if re.search(r"[。！？；：]$", buffer):
                flush()
                buffer = line
            else:
                buffer += line

    flush()
    return merged


def split_into_chunks(lines, max_chars=900, min_chars=120):
    chunks = []
    current_title = ""
    current_text = []

    def push_chunk():
        nonlocal current_text
        text = "\n".join(current_text).strip()
        if len(text) >= min_chars:
            chunks.append({
                "title": current_title,
                "content": text
            })
        current_text = []

    for line in lines:
        if line == "__REFERENCE_START__":
            push_chunk()
            chunks.append({
                "title": "参考文献",
                "content": "参考文献部分已截断，默认不纳入正文知识库。"
            })
            current_title = ""
            continue

        if is_section_title(line):
            push_chunk()
            current_title = line.strip()
            current_text.append(line.strip())
            continue

        if not current_text:
            current_text.append(line.strip())
            continue

        candidate = "\n".join(current_text + [line.strip()])
        if len(candidate) > max_chars:
            push_chunk()
            if current_title:
                current_text.append(current_title)
            current_text.append(line.strip())
        else:
            current_text.append(line.strip())

    push_chunk()
    return chunks


def extract_pdf_pages(pdf_path: str):
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text = normalize_text(text)
        pages.append({
            "page": i,
            "text": text
        })

    return pages


def infer_tags(title: str, content: str):
    text = f"{title} {content}"
    tags = []

    tag_rules = {
        "高血压": ["高血压", "血压", "降压"],
        "诊断": ["诊断", "分级", "分层", "测量", "评估"],
        "治疗": ["治疗", "降压药", "药物", "生活方式"],
        "危险因素": ["危险因素", "风险", "肥胖", "吸烟", "饮酒", "钠盐"],
        "老年": ["老年", "高龄"],
        "儿童": ["儿童", "青少年"],
        "妊娠": ["妊娠", "孕妇", "子痫前期"],
        "并发症": ["卒中", "冠心病", "心力衰竭", "肾脏", "认知障碍"],
        "继发性高血压": ["继发性", "原发性醛固酮增多症", "肾动脉狭窄", "睡眠呼吸暂停"],
    }

    for tag, kws in tag_rules.items():
        if any(kw in text for kw in kws):
            tags.append(tag)

    return tags


def clean_pdf(pdf_path: str, output_jsonl: str, keep_references=False):
    pages = extract_pdf_pages(pdf_path)

    all_records = []
    global_chunk_id = 1

    for page_item in pages:
        page_no = page_item["page"]
        raw_text = page_item["text"]

        lines = raw_text.splitlines()
        merged_lines = merge_broken_lines(lines)
        chunks = split_into_chunks(merged_lines)

        for idx, chunk in enumerate(chunks, start=1):
            title = chunk["title"]
            content = chunk["content"]

            if not keep_references and title == "参考文献":
                continue

            record = {
                "id": f"hypertension_{global_chunk_id:04d}",
                "source_file": Path(pdf_path).name,
                "page": page_no,
                "chunk_index": idx,
                "title": title,
                "content": content,
                "tags": infer_tags(title, content),
            }
            all_records.append(record)
            global_chunk_id += 1

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(all_records)


if __name__ == "__main__":
    if not Path(INPUT_PDF).exists():
        raise FileNotFoundError(f"找不到输入文件: {INPUT_PDF}")

    count = clean_pdf(
        pdf_path=INPUT_PDF,
        output_jsonl=OUTPUT_JSONL,
        keep_references=False
    )

    print(f"清洗完成，共导出 {count} 条知识块。")
    print(f"输出文件: {OUTPUT_JSONL}")