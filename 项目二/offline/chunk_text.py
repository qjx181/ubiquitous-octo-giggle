"""
chunk_text.py — 文本分块脚本
==============================
作用：按章节分组切分chunk，实现句子边界级overlap，表格单独处理
原理：
  1. 招股说明书有明确章节结构，按章节分组可保持语义连贯性
  2. 每个chunk大小适中（800字符），便于Embedding模型处理
  3. 相邻chunk之间有150字符重叠，避免信息断裂
  4. 表格是结构化数据，需要保留完整表头，单独分块策略
输入：data/parsed/招股说明书_原始文本.json
输出：data/chunks/招股说明书_分块.jsonl
"""

import json  # JSON序列化/反序列化
import re  # 正则表达式，用于文本分割
import shutil  # 文件操作工具，用于备份
import sys  # 系统相关功能
from pathlib import Path  # 面向对象的文件路径处理

# =============================================================================
# 模块路径设置
# =============================================================================
# 将项目根目录添加到Python模块搜索路径
# 作用：确保能导入项目根目录下的config.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 从项目根目录导入配置
from config import (
    PARSED_JSON_PATH,  # 解析结果输入路径
    CHUNKS_JSONL_PATH,  # 分块结果输出路径
    CHUNKS_DIR,  # 分块结果目录
    CHUNK_MAX_CHARS,  # 每个chunk最大字符数
    CHUNK_OVERLAP,  # chunk之间重叠字符数
    MIN_LINE_LENGTH,  # 最小行长度阈值
    PDF_HEADER_KEYWORDS,  # 页眉关键词（用于过滤）
    PDF_PAGE_NUMBER_PATTERN,  # 页码正则（用于过滤）
    RUN_TIMESTAMP,  # 运行时间戳（用于备份文件名）
    setup_logger,  # 日志设置函数
)

# 创建日志记录器，名称为"chunk_text"
# 日志将同时输出到控制台和文件 logs/chunk_text_时间戳.log
logger = setup_logger("chunk_text")


# =============================================================================
# 文件检查函数
# =============================================================================
def check_file_exists(file_path: str) -> bool:
    """
    作用：检查输入文件是否存在
    
    原理：分块依赖于解析结果，如果输入文件不存在则无法继续
    
    参数：
      file_path: 文件路径字符串
    
    返回：
      True: 文件存在
      False: 文件不存在（同时记录错误日志）
    """
    p = Path(file_path)
    if not p.exists():
        logger.error(f"文件不存在: {file_path}")
        return False
    return True


# =============================================================================
# 备份函数
# =============================================================================
def backup_previous_output(output_path: str):
    """
    作用：备份旧输出文件，用于回滚
    
    原理：每次运行可能覆盖旧结果，备份后如果新结果有问题可以恢复
    
    参数：
      output_path: 输出文件路径
    
    备份命名规则：
      原文件名_备份_时间戳.后缀
    """
    p = Path(output_path)
    if p.exists():
        bn = f"{p.stem}_备份_{RUN_TIMESTAMP}{p.suffix}"
        shutil.copy2(p, p.parent / bn)
        logger.info(f"已备份旧文件 -> {p.parent / bn}")


# =============================================================================
# 数据加载函数
# =============================================================================
def load_parsed_pages(json_path: str) -> list[dict]:
    """
    作用：加载 parse_pdf.py 输出的 JSON 文件
    
    原理：
      - parse_pdf.py 输出的是标准JSON格式（整个文件是一个JSON数组）
      - 每个元素是一个页面字典，包含page_num、text、section等字段
    
    参数：
      json_path: JSON文件路径
    
    返回：
      页面字典列表
    
    示例返回：
      [
        {"page_num": 1, "text": "...", "section": "第一节 释义", "type": "text"},
        {"page_num": 2, "text": "...", "section": "第一节 释义", "type": "table", ...}
      ]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# 行合并函数
# =============================================================================
def merge_lines(text: str) -> str:
    """
    作用：把 PDF 中断开的行合并成连续段落
    
    原理：
      - PDF提取的文本中，一个段落可能被拆分成多行（因为排版）
      - 需要将属于同一段落的行合并，恢复连续文本
      - 同时过滤页眉、页码等无关内容
    
    参数：
      text: 原始文本（可能包含断行）
    
    返回：
      合并后的连续段落文本
    
    合并规则：
      1. 空行表示段落结束
      2. 页码行跳过
      3. 页眉行跳过
      4. 如果上一行不以句末标点结尾，或当前行很短，则合并
      5. 如果当前行以中文数字开头（如"一、"），则新起一段
    """
    lines = text.split("\n")  # 按换行拆分成行列表
    merged, buf = [], ""  # merged存储合并后的段落，buf缓存当前段落
    
    for line in lines:
        s = line.strip()  # 去除首尾空白
        if not s:
            # 空行表示段落结束
            if buf:
                merged.append(buf)  # 保存当前段落
                buf = ""  # 清空缓存
            continue
        
        # ---------------------------------------------------------------
        # 过滤页码行：匹配 "1-1-1" 格式
        # ---------------------------------------------------------------
        if re.match(PDF_PAGE_NUMBER_PATTERN, s):
            continue
        
        # ---------------------------------------------------------------
        # 过滤纯数字行（页码）：长度<=4的纯数字
        # ---------------------------------------------------------------
        if re.match(r"^\d+$", s) and len(s) <= 4:
            continue
        
        # ---------------------------------------------------------------
        # 过滤页眉行：包含页眉关键词且长度<60
        # ---------------------------------------------------------------
        is_h = False
        for kw in PDF_HEADER_KEYWORDS:
            if kw in s and len(s) < 60:
                is_h = True
                break
        if is_h:
            continue
        
        # ---------------------------------------------------------------
        # 行合并逻辑
        # ---------------------------------------------------------------
        if buf:
            # 句末标点：如果上一行以这些标点结尾，表示句子结束
            end_chars = "。；：！？）\""
            
            # 情况1：上一行不以句末标点结尾，或当前行很短（<30字符）
            # 说明当前行可能是上一行的延续
            if not re.search(rf"[{end_chars}]$", buf) or len(s) < MIN_LINE_LENGTH:
                buf += s  # 合并到当前段落
                continue
            
            # 情况2：当前行以中文数字开头（如"一、"、"二、"）
            # 说明是新段落的开始
            if re.match(r"^[一二三四五六七八九十]", s):
                buf += s
                continue
            
            # 情况3：上一行以句末标点结尾，且当前行不短
            # 说明是新段落，保存当前段落
            merged.append(buf)
        
        # 开始新段落
        buf = s
    
    # 处理最后一个段落
    if buf:
        merged.append(buf)
    
    # 用两个换行连接段落，保持段落间距
    return "\n\n".join(merged)


# =============================================================================
# 段落拆分函数
# =============================================================================
def split_into_paragraphs(text: str) -> list[str]:
    """
    作用：按空行拆分成段落，去掉太短的
    
    原理：
      - 连续两个换行表示段落分隔
      - 过滤掉长度<10的段落（可能是噪声）
    
    参数：
      text: 合并后的连续文本
    
    返回：
      段落字符串列表
    """
    # 正则 \n{2,} 匹配2个及以上的连续换行
    raw = re.split(r"\n{2,}", text)
    # 过滤：去除首尾空白，且长度>=10
    return [p.strip() for p in raw if len(p.strip()) >= 10]


# =============================================================================
# 长段落切分函数
# =============================================================================
def split_long_paragraph(para: str, max_chars: int) -> list[str]:
    """
    作用：长段落按句子边界切分
    
    原理：
      - 如果段落长度<=max_chars，直接返回
      - 否则按句子边界（。！？；）切分
      - 尽量保持句子完整，避免在句子中间切断
    
    参数：
      para: 段落文本
      max_chars: 最大字符数
    
    返回：
      切分后的子段落列表
    
    切分策略：
      1. 按句子边界切分
      2. 如果某个句子仍然很长，强制按max_chars切分
    """
    # 如果段落不长，直接返回
    if len(para) <= max_chars:
        return [para]
    
    # 按句子边界切分
    # 正则 ([。！？；\n]) 捕获句末标点，保留在结果中
    sentences = re.split(r"([。！？；\n])", para)
    
    combined, buf = [], ""  # combined存储结果，buf缓存当前子段落
    
    for s in sentences:
        buf += s  # 添加当前句子到缓存
        
        # 如果当前句子以句末标点结尾，或缓存已接近max_chars的80%
        if s in ("。", "！", "？", "；", "\n") or len(buf) >= max_chars * 0.8:
            if buf.strip() and len(buf.strip()) > 10:
                combined.append(buf.strip())  # 保存子段落
            buf = ""  # 清空缓存
    
    # 处理剩余内容
    if buf.strip() and len(buf.strip()) > 10:
        combined.append(buf.strip())
    
    # ---------------------------------------------------------------
    # 二次切分：如果某个子段落仍然超过max_chars，强制切分
    # ---------------------------------------------------------------
    result = []
    for seg in combined:
        if len(seg) > max_chars:
            # 强制按max_chars切分
            for i in range(0, len(seg), max_chars):
                c = seg[i:i + max_chars].strip()
                if c and len(c) > 10:
                    result.append(c)
        else:
            result.append(seg)
    
    return result


# =============================================================================
# 表格分块函数
# =============================================================================
def chunk_table(table_text: str, table_info: dict, max_chars: int = 2000) -> list[dict]:
    """
    作用：将大表格切分成多个小表格chunk，保留表头
    
    原理：
      - 表格chunk必须保留完整表头，否则无法理解数据含义
      - 按数据行切分，每个chunk包含表头+部分数据行
      - 控制每个chunk的大小不超过max_chars
    
    参数：
      table_text: Markdown格式的表格文本
      table_info: 表格元数据（page_num、table_index、section等）
      max_chars: 每个chunk最大字符数
    
    返回：
      表格chunk字典列表
    
    表格结构：
      第1行：[表格 N]（标题）
      第2行：表头（如"项目 | 2023年"）
      第3行：分隔线（--- | ---）
      第4行+：数据行
    """
    lines = table_text.split("\n")
    if len(lines) < 3:
        return []  # 数据不足，无法构成有效表格
    
    # 表头部分（前3行：标题 + 表头行 + 分隔线）
    header_lines = lines[:3]
    header_text = "\n".join(header_lines)
    data_lines = lines[3:]  # 数据行
    
    chunks = []  # 存储切分后的chunk
    buf_lines = []  # 缓存当前chunk的数据行
    cid = 0  # chunk序号
    
    for line in data_lines:
        # 预估添加当前行后的总大小
        test_text = header_text + "\n" + "\n".join(buf_lines + [line])
        
        # 如果超过max_chars且缓存不为空，保存当前chunk
        if len(test_text) > max_chars and buf_lines:
            chunk_text = header_text + "\n" + "\n".join(buf_lines)
            chunks.append({
                "chunk_id": f"table_{table_info['page_num']}_{table_info['table_index']}_{cid:02d}",
                "text": chunk_text,
                "type": "table",  # 类型标记
                "page_start": table_info["page_num"],  # 表格所在页码
                "page_end": table_info["page_num"],
                "table_index": table_info.get("table_index", 1),  # 表格序号
                "section": table_info.get("section", "未知"),  # 所属章节
                "source_pdf": table_info.get("source_pdf", ""),  # 来源PDF
                "char_count": len(chunk_text),  # 字符数
            })
            cid += 1
            buf_lines = [line]  # 开始新chunk，保留当前行
        else:
            buf_lines.append(line)  # 添加到缓存
    
    # 处理剩余行
    if buf_lines:
        chunk_text = header_text + "\n" + "\n".join(buf_lines)
        chunks.append({
            "chunk_id": f"table_{table_info['page_num']}_{table_info['table_index']}_{cid:02d}",
            "text": chunk_text,
            "type": "table",
            "page_start": table_info["page_num"],
            "page_end": table_info["page_num"],
            "table_index": table_info.get("table_index", 1),
            "section": table_info.get("section", "未知"),
            "source_pdf": table_info.get("source_pdf", ""),
            "char_count": len(chunk_text),
        })
    
    return chunks


# =============================================================================
# 主分块函数
# =============================================================================
def build_chunks_by_section(pages: list[dict]) -> list[dict]:
    """
    作用：按章节分组，在每组内切分chunk，实现句子边界级overlap，表格单独处理
    
    原理：
      1. 按章节分组：同一章节的内容放在一起
      2. 文本分块：按段落和句子边界切分，保持语义完整
      3. 表格分块：保留表头，按数据行切分
      4. Overlap：相邻chunk共享150字符，避免信息断裂
    
    参数：
      pages: 解析后的页面列表（来自parse_pdf.py的输出）
    
    返回：
      chunk字典列表，每个chunk包含：
        - chunk_id: 唯一标识
        - text: 文本内容
        - type: "text" 或 "table"
        - page_start/page_end: 页码范围
        - section: 所属章节
        - source_pdf: 来源PDF
        - char_count: 字符数
    
    分块策略：
      文本：
        - 按段落切分
        - 段落过长按句子边界切分
        - 相邻chunk有150字符重叠
      表格：
        - 保留完整表头
        - 按数据行切分
        - 每个chunk独立可理解
    """
    if not pages:
        return []
    
    chunks, cid = [], 0  # chunks存储结果，cid是chunk序号计数器
    sections = []  # 存储章节分组
    
    # -------------------------------------------------------------------------
    # 步骤1：按章节分组
    # -------------------------------------------------------------------------
    cur_sec, cur_pages = None, []  # 当前章节名和页面列表
    
    for p in pages:
        sec = p.get("section") or "未知"  # 获取章节名，默认为"未知"
        if sec != cur_sec:
            # 章节变化，保存上一组
            if cur_pages:
                sections.append((cur_sec, cur_pages))
            cur_sec, cur_pages = sec, []  # 开始新组
        cur_pages.append(p)
    
    # 保存最后一组
    if cur_pages:
        sections.append((cur_sec, cur_pages))
    
    logger.info(f"检测到 {len(sections)} 个章节分组")
    
    # -------------------------------------------------------------------------
    # 步骤2：对每个章节进行分块
    # -------------------------------------------------------------------------
    for sec_name, sec_pages in sections:
        # 分离文本页和表格页
        text_pages = [p for p in sec_pages if p.get("type") == "text"]
        table_pages = [p for p in sec_pages if p.get("type") == "table"]
        
        # =================================================================
        # 2.1 处理普通文本
        # =================================================================
        if text_pages:
            # 合并所有文本页的内容
            full_text = ""
            # 页码范围：从第一页到最后一页
            pr = (text_pages[0]["page_num"], text_pages[-1]["page_num"])
            
            for p in text_pages:
                mt = merge_lines(p["text"])  # 合并断行
                full_text = (full_text + "\n\n" + mt) if full_text else mt
            
            # 拆分成段落
            paragraphs = split_into_paragraphs(full_text)
            
            # 段落合并成chunk
            buf_text = ""  # 缓存当前chunk的文本
            
            for para in paragraphs:
                if buf_text:
                    # 尝试将当前段落添加到缓存
                    cand = buf_text + "\n" + para
                    
                    if len(cand) <= CHUNK_MAX_CHARS:
                        # 未超过限制，继续累积
                        buf_text = cand
                        continue
                    
                    # 超过限制，保存当前chunk
                    chunks.append({
                        "chunk_id": f"chunk_{cid:04d}",
                        "text": buf_text,
                        "type": "text",
                        "page_start": pr[0],
                        "page_end": pr[1],
                        "section": sec_name,
                        "source_pdf": str(pages[0].get("source_pdf", "")),
                        "char_count": len(buf_text),
                    })
                    cid += 1
                    
                    # -------------------------------------------------------
                    # Overlap处理：保留上一chunk末尾的150字符
                    # -------------------------------------------------------
                    overlap = buf_text
                    if len(overlap) > CHUNK_OVERLAP:
                        # 从后往前找最后一个"。"，确保在句子边界切分
                        cut = len(overlap) - CHUNK_OVERLAP
                        se = overlap.rfind("。", 0, cut)
                        if se > 0:
                            overlap = overlap[se + 1:]  # 保留句号后的内容
                        else:
                            overlap = overlap[-CHUNK_OVERLAP:]  # 强制切分
                    
                    # 新chunk从overlap开始，加上当前段落
                    buf_text = overlap + "\n" + para
                    continue
                
                # buf_text为空，开始新chunk
                if len(para) <= CHUNK_MAX_CHARS:
                    buf_text = para
                else:
                    # 段落过长，需要切分
                    for sub in split_long_paragraph(para, CHUNK_MAX_CHARS):
                        chunks.append({
                            "chunk_id": f"chunk_{cid:04d}",
                            "text": sub,
                            "type": "text",
                            "page_start": pr[0],
                            "page_end": pr[1],
                            "section": sec_name,
                            "source_pdf": str(pages[0].get("source_pdf", "")),
                            "char_count": len(sub),
                        })
                        cid += 1
            
            # 处理最后一个chunk
            if buf_text:
                chunks.append({
                    "chunk_id": f"chunk_{cid:04d}",
                    "text": buf_text,
                    "type": "text",
                    "page_start": pr[0],
                    "page_end": pr[1],
                    "section": sec_name,
                    "source_pdf": str(pages[0].get("source_pdf", "")),
                    "char_count": len(buf_text),
                })
                cid += 1
        
        # =================================================================
        # 2.2 处理表格
        # =================================================================
        for table_page in table_pages:
            table_chunks = chunk_table(
                table_page["text"],
                table_page,
                max_chars=CHUNK_MAX_CHARS
            )
            for tc in table_chunks:
                tc["chunk_id"] = f"chunk_{cid:04d}"
                chunks.append(tc)
                cid += 1
    
    return chunks


# =============================================================================
# 质量报告函数
# =============================================================================
def print_quality_report(chunks: list[dict]):
    """
    作用：输出分块质量报告（数量、大小分布、overlap统计）
    
    原理：
      - 统计chunk数量、总字符数、平均大小
      - 检查大小分布，发现异常（过大或过小的chunk）
      - 统计overlap效果
    
    参数：
      chunks: 分块后的chunk列表
    """
    if not chunks:
        logger.warning("无chunk数据")
        return
    
    total = len(chunks)  # 总chunk数
    chars = sum(c["char_count"] for c in chunks)  # 总字符数
    avg = chars / total  # 平均字符数
    mn = min(c["char_count"] for c in chunks)  # 最小字符数
    mx = max(c["char_count"] for c in chunks)  # 最大字符数
    
    # 统计每个章节的chunk数量
    sec_cnt = {}
    for c in chunks:
        s = c["section"] or "未知"
        sec_cnt[s] = sec_cnt.get(s, 0) + 1
    
    # 统计overlap效果
    oc = 0  # 含overlap的相邻对数
    for i in range(1, len(chunks)):
        # 检查相邻chunk是否同章节
        if chunks[i].get("section") == chunks[i-1].get("section"):
            # 检查是否有overlap（上一chunk末尾出现在当前chunk中）
            if chunks[i-1]["text"][-CHUNK_OVERLAP:].strip() in chunks[i]["text"]:
                oc += 1
    
    # 输出报告
    logger.info("=" * 50)
    logger.info("  分块质量报告")
    logger.info("=" * 50)
    logger.info(f"总chunk数: {total}, 总字符: {chars:,}, 平均: {avg:.0f}, 最短/最长: {mn}/{mx}")
    logger.info(f"同章节含overlap相邻对: {oc}")
    for s, c in sorted(sec_cnt.items(), key=lambda x: -x[1]):
        logger.info(f"  {s[:25]:25s}: {c:4d}个 ({c/total*100:.1f}%)")


# =============================================================================
# 程序入口
# =============================================================================
def main():
    """
    作用：程序入口，加载JSON→按章节分块→质量报告→保存
    
    执行流程：
      1. 检查输入文件
      2. 备份旧输出
      3. 加载解析结果
      4. 按章节分块
      5. 输出质量报告
      6. 保存为JSONL
    """
    logger.info("=" * 50)
    logger.info("  文本分块 开始")
    logger.info("=" * 50)
    
    # 步骤1：检查输入文件
    if not check_file_exists(str(PARSED_JSON_PATH)):
        return
    
    # 步骤2：备份旧输出
    backup_previous_output(str(CHUNKS_JSONL_PATH))
    
    # 步骤3：加载解析结果
    logger.info(f"加载: {PARSED_JSON_PATH}")
    pages = load_parsed_pages(str(PARSED_JSON_PATH))
    if not pages:
        logger.error("未加载到数据")
        return
    logger.info(f"加载了 {len(pages)} 页")
    
    # 步骤4：按章节分块
    logger.info("按章节分块...")
    chunks = build_chunks_by_section(pages)
    if not chunks:
        logger.error("分块失败")
        return
    logger.info(f"生成 {len(chunks)} 个 chunks")
    
    # 步骤5：输出质量报告
    print_quality_report(chunks)
    
    # 步骤6：保存为JSONL
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_JSONL_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            # JSONL格式：每行一个JSON对象
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    
    logger.info(f"已保存: {CHUNKS_JSONL_PATH}")
    logger.info("🎉 分块完成")


# =============================================================================
# 脚本执行入口
# =============================================================================
if __name__ == "__main__":
    main()
