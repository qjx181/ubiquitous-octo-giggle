"""
parse_pdf.py — PDF解析脚本
作用：读取招股说明书PDF，提取每页文字，按章节整理后保存为JSON
"""

import json
import re
import shutil
import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    PDF_PATH, PARSED_JSON_PATH, PARSED_DIR,
    PDF_HEADER_KEYWORDS, PDF_PAGE_NUMBER_PATTERN,
    QUALITY_MIN_CHUNKS, QUALITY_MAX_EMPTY_RATIO,
    RUN_TIMESTAMP, setup_logger,
)

logger = setup_logger("parse_pdf")


def check_file_exists(file_path: str) -> bool:
    """
    作用：检查输入文件是否存在
    原理：如果PDF文件不存在，后续解析无法进行，所以提前拦截
    逻辑：用Path.exists()判断，不存在则记录错误日志并返回False
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"文件不存在: {file_path}")
        return False
    return True


def backup_previous_output(output_path: str):
    """
    作用：备份旧输出文件，用于回滚
    原理：每次运行可能覆盖旧结果，备份后如果新结果有问题可以恢复
    逻辑：检查文件是否存在，存在则复制为 原文件名_备份_时间戳.后缀
    """
    path = Path(output_path)
    if path.exists():
        backup_name = f"{path.stem}_备份_{RUN_TIMESTAMP}{path.suffix}"
        backup_path = path.parent / backup_name
        shutil.copy2(path, backup_path)
        logger.info(f"已备份旧文件 -> {backup_path}")


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    作用：把PDF文件逐页读出来，提取每一页的文字
    原理：PyMuPDF（fitz）可以直接读取PDF每一页的文本内容
    逻辑：
      1. fitz.open() 打开PDF
      2. 循环每一页，page.get_text() 提取文字
      3. 跳过空白页，记录页码和字符数
      4. 单页失败跳过，不中断整体流程
    返回：[{"page_num", "source_pdf", "text", "char_count"}, ...]
    """
    pages = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"无法打开PDF文件: {e}")
        return pages
    total_pages = len(doc)
    logger.info(f"PDF 共 {total_pages} 页")
    for page_index in range(total_pages):
        try:
            page = doc[page_index]
            text = page.get_text()
            if text and text.strip():
                pages.append({
                    "page_num": page_index + 1,
                    "source_pdf": str(pdf_path),
                    "text": text.strip(),
                    "char_count": len(text.strip()),
                })
        except Exception as e:
            logger.warning(f"第 {page_index + 1} 页读取失败，已跳过: {e}")
            continue
    doc.close()
    logger.info(f"成功读取 {len(pages)} 页有效内容（共 {total_pages} 页）")
    return pages


def guess_section(text: str) -> str | None:
    """
    作用：判断当前页属于招股说明书的哪个章节
    原理：章节标题通常在页面前几行，PDF页眉（公司名、页码）需要跳过
    逻辑：
      1. 把文本按换行拆成行列表
      2. 只看前10行（章节标题不会出现在页面中间）
      3. 跳过页眉行（从config.PDF_HEADER_KEYWORDS读取关键词）
      4. 匹配"第X节 XXX"或"第X章 XXX"的编号标题
      5. 匹配"重大事项提示""释义"等无编号标题
      6. 都没找到返回None
    返回：章节名字符串，没找到返回None
    """
    lines = text.split("\n")
    for line in lines[:10]:
        line = line.strip()
        if not line:
            continue
        is_header = False
        for kw in PDF_HEADER_KEYWORDS:
            if kw in line:
                is_header = True
                break
        if is_header:
            continue
        if re.match(PDF_PAGE_NUMBER_PATTERN, line):
            continue
        m = re.match(r"^(第[一二三四五六七八九十]+[节章]\s*\S+)", line)
        if m:
            return m.group(1)
        for kw in ["目  录", "目录", "声 明", "声明",
                    "重大事项提示", "释 义", "释义", "概 览", "概览"]:
            if kw in line and len(line) < 30:
                return kw
    return None


def clean_text(text: str) -> str:
    """
    作用：清理文本，去掉多余的空行
    原理：PDF提取的文字段落之间可能有多个换行，压缩后更紧凑
    逻辑：用正则把3个以上连续换行替换成2个换行
    """
    return re.sub(r"\n{3,}", "\n\n", text)


def print_quality_report(pages: list[dict]):
    """
    作用：输出解析质量报告
    原理：让用户知道解析结果是否正常，有数据异常时报警
    逻辑：统计总页数、总字符、章节分布，对比阈值输出警告
    """
    if not pages:
        logger.warning("无数据，跳过质量报告")
        return
    total_chars = sum(p["char_count"] for p in pages)
    avg_chars = total_chars / len(pages) if pages else 0
    secs = [p["section"] for p in pages if p["section"]]
    unique = list(dict.fromkeys(secs))
    logger.info("=" * 50)
    logger.info("  质量报告")
    logger.info("=" * 50)
    logger.info(f"总页数: {len(pages)}, 总字符: {total_chars:,}, 平均: {avg_chars:.0f}字符")
    logger.info(f"检测到章节: {len(unique)} 个")
    for s in unique:
        cnt = sum(1 for p in pages if p["section"] == s)
        logger.info(f"  - {s}: {cnt} 页")
    if len(pages) < QUALITY_MIN_CHUNKS:
        logger.warning(f"⚠️ 页数偏少: {len(pages)} < {QUALITY_MIN_CHUNKS}")
    empty = sum(1 for p in pages if not p.get("text", "").strip())
    ratio = empty / len(pages) if pages else 0
    if ratio > QUALITY_MAX_EMPTY_RATIO:
        logger.warning(f"⚠️ 空白页比例过高: {ratio:.1%} > {QUALITY_MAX_EMPTY_RATIO:.0%}")


def main():
    """
    作用：程序入口，串联整个PDF解析流程
    逻辑：校验输入→备份旧输出→读取PDF→清理+章节识别→质量报告→保存JSON
    """
    logger.info("=" * 50)
    logger.info("  PDF 解析 开始")
    logger.info("=" * 50)
    if not check_file_exists(str(PDF_PATH)):
        return
    backup_previous_output(str(PARSED_JSON_PATH))
    logger.info(f"读取PDF: {PDF_PATH}")
    pages = extract_text_from_pdf(str(PDF_PATH))
    if not pages:
        logger.error("未读取到任何有效内容")
        return
    success, fail = 0, 0
    current_section = None
    for page in pages:
        try:
            page["text"] = clean_text(page["text"])
            d = guess_section(page["text"])
            if d:
                current_section = d
            page["section"] = current_section
            success += 1
        except Exception as e:
            logger.warning(f"第 {page['page_num']} 页处理出错: {e}")
            page["section"] = current_section
            fail += 1
    logger.info(f"处理完成: 成功 {success} 页, 跳过 {fail} 页")
    print_quality_report(pages)
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PARSED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存: {PARSED_JSON_PATH}")
    logger.info("🎉 PDF 解析完成")


if __name__ == "__main__":
    main()
