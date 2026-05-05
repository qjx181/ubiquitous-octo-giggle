import re
import json
from pathlib import Path
import pdfplumber

PDF_PATH = r"C:\Users\qjx\Desktop\中华人民共和国民法典.pdf"
OUTPUT_JSON = r"C:\Users\qjx\Desktop\中华人民共和国民法典_rag.json"
OUTPUT_JSONL = r"C:\Users\qjx\Desktop\中华人民共和国民法典_rag.jsonl"


def normalize_page_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\u3000", " ")
    text = text.replace("\t", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ ]{2,}", " ", text)

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # 去页码
        if re.fullmatch(r"－\d+－", line):
            continue
        if re.fullmatch(r"--\s*\d+\s+of\s+\d+\s*--", line):
            continue

        # 去孤立页眉页脚
        if line in {"目 录", "目录"}:
            continue

        lines.append(line)

    return "\n".join(lines)


def merge_broken_lines(text: str) -> str:
    """
    尽量把 PDF 断行恢复成连续段落
    """
    if not text:
        return ""

    lines = [x.strip() for x in text.splitlines() if x.strip()]
    merged = []

    for line in lines:
        if not merged:
            merged.append(line)
            continue

        prev = merged[-1]

        # 这些情况一般要拼接
        if not re.search(r"[。！？；：]$", prev):
            merged[-1] = prev + line
        else:
            merged.append(line)

    return "\n".join(merged)


def extract_article_blocks(text: str):
    """
    从全文中抽取条文块，支持条文跨行
    """
    # 统一一些空格格式
    text = text.replace("第一　条", "第一条")
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" 。", "。").replace(" ，", "，").replace(" ；", "；").replace(" ：", "：")

    # 先把“第X条”作为切分锚点
    pattern = re.compile(r"(第[一二三四五六七八九十百零〇两\d]+条)")
    matches = list(pattern.finditer(text))

    blocks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        blocks.append(block)

    return blocks


def parse_levels(block: str):
    """
    从一个条文块里识别层级标题
    """
    # 默认值
    part = part_title = chapter = chapter_title = section = section_title = None

    # 找最近的层级标题
    part_match = re.search(r"(第[一二三四五六七八九十百零〇两]+编)\s*([^\s第]+)?", block)
    chapter_match = re.search(r"(第[一二三四五六七八九十百零〇两]+章)\s*([^\s第]+)?", block)
    section_match = re.search(r"(第[一二三四五六七八九十百零〇两]+节)\s*([^\s第]+)?", block)
    article_match = re.search(r"(第[一二三四五六七八九十百零〇两\d]+条)\s*(.*)", block)

    if part_match:
        part = part_match.group(1)
        part_title = (part_match.group(2) or "").strip() or None

    if chapter_match:
        chapter = chapter_match.group(1)
        chapter_title = (chapter_match.group(2) or "").strip() or None

    if section_match:
        section = section_match.group(1)
        section_title = (section_match.group(2) or "").strip() or None

    article_no = None
    article_text = None

    if article_match:
        article_no = article_match.group(1)
        article_text = article_match.group(2).strip()

    return part, part_title, chapter, chapter_title, section, section_title, article_no, article_text


def build_path(part, part_title, chapter, chapter_title, section, section_title, article_no):
    items = []
    if part:
        items.append(f"{part}{' ' + part_title if part_title else ''}".strip())
    if chapter:
        items.append(f"{chapter}{' ' + chapter_title if chapter_title else ''}".strip())
    if section:
        items.append(f"{section}{' ' + section_title if section_title else ''}".strip())
    if article_no:
        items.append(article_no)
    return " > ".join(items)


def chinese_to_int(s: str):
    if not s:
        return None
    s = s.replace("第", "").replace("条", "")
    s = s.replace("〇", "零").replace("两", "二")

    if s.isdigit():
        return int(s)

    num_map = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    unit_map = {"十": 10, "百": 100, "千": 1000, "万": 10000}

    result = 0
    section = 0
    number = 0

    for ch in s:
        if ch in num_map:
            number = num_map[ch]
        elif ch in unit_map:
            unit = unit_map[ch]
            if number == 0:
                number = 1
            section += number * unit
            number = 0
        else:
            return None

    result += section + number
    return result if result > 0 else None


def main():
    all_pages = []

    with pdfplumber.open(PDF_PATH) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""
            cleaned = normalize_page_text(raw_text)
            if cleaned:
                all_pages.append((idx, cleaned))

    # 先合并全书文本
    full_text = "\n".join(text for _, text in all_pages)
    full_text = merge_broken_lines(full_text)

    # 抽取条文块
    blocks = extract_article_blocks(full_text)

    # 过滤掉目录块、明显不是正文的块
    articles = []
    current_part = current_part_title = None
    current_chapter = current_chapter_title = None
    current_section = current_section_title = None

    for block in blocks:
        # 层级更新
        part_match = re.search(r"(第[一二三四五六七八九十百零〇两]+编)\s*([^\s第]+)?", block)
        if part_match:
            current_part = part_match.group(1)
            current_part_title = (part_match.group(2) or "").strip() or None

        chapter_match = re.search(r"(第[一二三四五六七八九十百零〇两]+章)\s*([^\s第]+)?", block)
        if chapter_match:
            current_chapter = chapter_match.group(1)
            current_chapter_title = (chapter_match.group(2) or "").strip() or None

        section_match = re.search(r"(第[一二三四五六七八九十百零〇两]+节)\s*([^\s第]+)?", block)
        if section_match:
            current_section = section_match.group(1)
            current_section_title = (section_match.group(2) or "").strip() or None

        article_match = re.search(r"(第[一二三四五六七八九十百零〇两\d]+条)\s*(.*)", block)
        if not article_match:
            continue

        article_no = article_match.group(1)
        article_text = article_match.group(2).strip()

        # 排除目录中的“第一条”之类噪声：正文一般会有明显句子内容
        if len(article_text) < 5:
            continue

        # 清理一些残留噪声
        article_text = re.sub(r"\s+", " ", article_text).strip()
        article_text = article_text.replace(" ，", "，").replace(" 。", "。").replace(" ；", "；").replace(" ：", "：")
        article_text = article_text.replace("（ ", "（").replace(" ）", "）")

        item = {
            "id": f"civilcode-{len(articles) + 1}",
            "source": Path(PDF_PATH).name,
            "title": "中华人民共和国民法典",
            "part": current_part,
            "part_title": current_part_title,
            "chapter": current_chapter,
            "chapter_title": current_chapter_title,
            "section": current_section,
            "section_title": current_section_title,
            "article_no": article_no,
            "article_no_arabic": chinese_to_int(article_no),
            "text": article_text,
            "path": build_path(current_part, current_part_title, current_chapter, current_chapter_title, current_section, current_section_title, article_no),
            "chunk_type": "article",
            "embedding_text": f"{build_path(current_part, current_part_title, current_chapter, current_chapter_title, current_section, current_section_title, article_no)}\n{article_text}",
        }

        articles.append(item)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for item in articles:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done. Articles: {len(articles)}")
    print(f"JSON saved to: {OUTPUT_JSON}")
    print(f"JSONL saved to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()