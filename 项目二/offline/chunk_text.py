"""
chunk_text.py — 文本分块脚本
作用：按章节分组切分chunk，实现句子边界级overlap
"""

import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    PARSED_JSON_PATH, CHUNKS_JSONL_PATH, CHUNKS_DIR,
    CHUNK_MAX_CHARS, CHUNK_OVERLAP, MIN_LINE_LENGTH,
    PDF_HEADER_KEYWORDS, PDF_PAGE_NUMBER_PATTERN,
    RUN_TIMESTAMP, setup_logger,
)

logger = setup_logger("chunk_text")


def check_file_exists(file_path: str) -> bool:
    """作用：检查输入文件是否存在"""
    p = Path(file_path)
    if not p.exists():
        logger.error(f"文件不存在: {file_path}")
        return False
    return True


def backup_previous_output(output_path: str):
    """作用：备份旧输出文件，用于回滚"""
    p = Path(output_path)
    if p.exists():
        bn = f"{p.stem}_备份_{RUN_TIMESTAMP}{p.suffix}"
        shutil.copy2(p, p.parent / bn)
        logger.info(f"已备份旧文件 -> {p.parent / bn}")


def load_parsed_pages(json_path: str) -> list[dict]:
    """作用：加载 parse_pdf.py 输出的 JSON 文件"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_lines(text: str) -> str:
    """作用：把 PDF 中断开的行合并成连续段落"""
    lines = text.split("\n")
    merged, buf = [], ""
    for line in lines:
        s = line.strip()
        if not s:
            if buf:
                merged.append(buf)
                buf = ""
            continue
        if re.match(PDF_PAGE_NUMBER_PATTERN, s):
            continue
        if re.match(r"^\d+$", s) and len(s) <= 4:
            continue
        is_h = False
        for kw in PDF_HEADER_KEYWORDS:
            if kw in s and len(s) < 60:
                is_h = True
                break
        if is_h:
            continue
        if buf:
            end_chars = "。；：！？）\""
            if not re.search(rf"[{end_chars}]$", buf) or len(s) < MIN_LINE_LENGTH:
                buf += s
                continue
            if re.match(r"^[一二三四五六七八九十]", s):
                buf += s
                continue
            merged.append(buf)
        buf = s
    if buf:
        merged.append(buf)
    return "\n\n".join(merged)


def split_into_paragraphs(text: str) -> list[str]:
    """作用：按空行拆分成段落，去掉太短的"""
    raw = re.split(r"\n{2,}", text)
    return [p.strip() for p in raw if len(p.strip()) >= 10]


def split_long_paragraph(para: str, max_chars: int) -> list[str]:
    """作用：长段落按句子边界切分"""
    if len(para) <= max_chars:
        return [para]
    sentences = re.split(r"([。！？；\n])", para)
    combined, buf = [], ""
    for s in sentences:
        buf += s
        if s in ("。", "！", "？", "；", "\n") or len(buf) >= max_chars * 0.8:
            if buf.strip() and len(buf.strip()) > 10:
                combined.append(buf.strip())
            buf = ""
    if buf.strip() and len(buf.strip()) > 10:
        combined.append(buf.strip())
    result = []
    for seg in combined:
        if len(seg) > max_chars:
            for i in range(0, len(seg), max_chars):
                c = seg[i:i + max_chars].strip()
                if c and len(c) > 10:
                    result.append(c)
        else:
            result.append(seg)
    return result


def build_chunks_by_section(pages: list[dict]) -> list[dict]:
    """作用：按章节分组，在每组内切分chunk，实现句子边界级overlap"""
    if not pages:
        return []
    chunks, cid = [], 0
    sections = []
    cur_sec, cur_pages = None, []
    for p in pages:
        sec = p.get("section") or "未知"
        if sec != cur_sec:
            if cur_pages:
                sections.append((cur_sec, cur_pages))
            cur_sec, cur_pages = sec, []
        cur_pages.append(p)
    if cur_pages:
        sections.append((cur_sec, cur_pages))
    logger.info(f"检测到 {len(sections)} 个章节分组")
    for sec_name, sec_pages in sections:
        full_text = ""
        pr = (sec_pages[0]["page_num"], sec_pages[-1]["page_num"])
        for p in sec_pages:
            mt = merge_lines(p["text"])
            full_text = (full_text + "\n\n" + mt) if full_text else mt
        paragraphs = split_into_paragraphs(full_text)
        buf_text = ""
        for para in paragraphs:
            if buf_text:
                cand = buf_text + "\n" + para
                if len(cand) <= CHUNK_MAX_CHARS:
                    buf_text = cand
                    continue
                chunks.append({
                    "chunk_id": f"chunk_{cid:04d}",
                    "text": buf_text,
                    "page_start": pr[0],
                    "page_end": pr[1],
                    "section": sec_name,
                    "source_pdf": str(pages[0].get("source_pdf", "")),
                    "char_count": len(buf_text),
                })
                cid += 1
                overlap = buf_text
                if len(overlap) > CHUNK_OVERLAP:
                    cut = len(overlap) - CHUNK_OVERLAP
                    se = overlap.rfind("。", 0, cut)
                    overlap = overlap[se + 1:] if se > 0 else overlap[-CHUNK_OVERLAP:]
                buf_text = overlap + "\n" + para
                continue
            if len(para) <= CHUNK_MAX_CHARS:
                buf_text = para
            else:
                for sub in split_long_paragraph(para, CHUNK_MAX_CHARS):
                    chunks.append({
                        "chunk_id": f"chunk_{cid:04d}",
                        "text": sub,
                        "page_start": pr[0],
                        "page_end": pr[1],
                        "section": sec_name,
                        "source_pdf": str(pages[0].get("source_pdf", "")),
                        "char_count": len(sub),
                    })
                    cid += 1
        if buf_text:
            chunks.append({
                "chunk_id": f"chunk_{cid:04d}",
                "text": buf_text,
                "page_start": pr[0],
                "page_end": pr[1],
                "section": sec_name,
                "source_pdf": str(pages[0].get("source_pdf", "")),
                "char_count": len(buf_text),
            })
    return chunks


def print_quality_report(chunks: list[dict]):
    """作用：输出分块质量报告（数量、大小分布、overlap统计）"""
    if not chunks:
        logger.warning("无chunk数据")
        return
    total = len(chunks)
    chars = sum(c["char_count"] for c in chunks)
    avg = chars / total
    mn = min(c["char_count"] for c in chunks)
    mx = max(c["char_count"] for c in chunks)
    sec_cnt = {}
    for c in chunks:
        s = c["section"] or "未知"
        sec_cnt[s] = sec_cnt.get(s, 0) + 1
    oc = 0
    for i in range(1, len(chunks)):
        if chunks[i].get("section") == chunks[i-1].get("section"):
            if chunks[i-1]["text"][-CHUNK_OVERLAP:].strip() in chunks[i]["text"]:
                oc += 1
    logger.info("=" * 50)
    logger.info("  分块质量报告")
    logger.info("=" * 50)
    logger.info(f"总chunk数: {total}, 总字符: {chars:,}, 平均: {avg:.0f}, 最短/最长: {mn}/{mx}")
    logger.info(f"同章节含overlap相邻对: {oc}")
    for s, c in sorted(sec_cnt.items(), key=lambda x: -x[1]):
        logger.info(f"  {s[:25]:25s}: {c:4d}个 ({c/total*100:.1f}%)")


def main():
    """作用：程序入口，加载JSON→按章节分块→质量报告→保存"""
    logger.info("=" * 50)
    logger.info("  文本分块 开始")
    logger.info("=" * 50)
    if not check_file_exists(str(PARSED_JSON_PATH)):
        return
    backup_previous_output(str(CHUNKS_JSONL_PATH))
    logger.info(f"加载: {PARSED_JSON_PATH}")
    pages = load_parsed_pages(str(PARSED_JSON_PATH))
    if not pages:
        logger.error("未加载到数据")
        return
    logger.info(f"加载了 {len(pages)} 页")
    logger.info("按章节分块...")
    chunks = build_chunks_by_section(pages)
    if not chunks:
        logger.error("分块失败")
        return
    logger.info(f"生成 {len(chunks)} 个 chunks")
    print_quality_report(chunks)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_JSONL_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    logger.info(f"已保存: {CHUNKS_JSONL_PATH}")
    logger.info("🎉 分块完成")


if __name__ == "__main__":
    main()
