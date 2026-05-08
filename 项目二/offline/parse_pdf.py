"""
parse_pdf.py — PDF解析脚本
==============================
作用：读取招股说明书PDF，提取每页文字和表格，按章节整理后保存为JSON
原理：
  1. PyMuPDF（fitz）是高性能的PDF处理库，支持文本提取和表格检测
  2. PDF是流式文档格式，需要逐页读取，每页独立处理
  3. 招股说明书有明确章节结构，需要识别章节标题进行分组
  4. 表格是结构化数据，需要特殊处理以保留行列关系
输入：data/招股说明书1.pdf
输出：data/parsed/招股说明书_原始文本.json
"""

import json  # JSON序列化/反序列化，用于保存解析结果
import re  # 正则表达式，用于文本匹配和清理
import shutil  # 文件操作工具，用于备份旧文件
import sys  # 系统相关功能，用于修改模块搜索路径
from pathlib import Path  # 面向对象的文件路径处理

import fitz  # PyMuPDF库，PDF处理的核心依赖

# =============================================================================
# 模块路径设置
# =============================================================================
# __file__ 是当前文件(parse_pdf.py)的路径
# .resolve() 转为绝对路径
# .parent.parent 获取上上级目录（即项目根目录）
# sys.path.insert(0, ...) 将项目根目录添加到Python模块搜索路径的最前面
# 作用：确保能导入项目根目录下的config.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 从项目根目录导入配置
from config import (
    PDF_PATH,  # PDF源文件路径
    PARSED_JSON_PATH,  # 解析结果输出路径
    PARSED_DIR,  # 解析结果目录
    PDF_HEADER_KEYWORDS,  # 页眉关键词列表（用于跳过页眉）
    PDF_PAGE_NUMBER_PATTERN,  # 页码正则表达式（用于跳过页码）
    QUALITY_MIN_CHUNKS,  # 质量检查：最小页数阈值
    QUALITY_MAX_EMPTY_RATIO,  # 质量检查：最大空白页比例阈值
    RUN_TIMESTAMP,  # 运行时间戳（用于备份文件名）
    setup_logger,  # 日志设置函数
)

# 创建日志记录器，名称为"parse_pdf"
# 日志将同时输出到控制台和文件 logs/parse_pdf_时间戳.log
logger = setup_logger("parse_pdf")


# =============================================================================
# 文件检查函数
# =============================================================================
def check_file_exists(file_path: str) -> bool:
    """
    作用：检查输入文件是否存在
    
    原理：
      - 如果PDF文件不存在，后续解析无法进行，所以提前拦截
      - 使用Path.exists()判断，比os.path.exists()更面向对象
    
    参数：
      file_path: 文件路径字符串
    
    返回：
      True: 文件存在
      False: 文件不存在（同时记录错误日志）
    
    示例：
      if not check_file_exists("data/xxx.pdf"):
          return  # 提前退出，避免后续报错
    """
    path = Path(file_path)  # 转为Path对象
    if not path.exists():  # 检查文件是否存在
        logger.error(f"文件不存在: {file_path}")
        return False
    return True


# =============================================================================
# 备份函数
# =============================================================================
def backup_previous_output(output_path: str):
    """
    作用：备份旧输出文件，用于回滚
    
    原理：
      - 每次运行可能覆盖旧结果，备份后如果新结果有问题可以恢复
      - 使用shutil.copy2()保留文件的元数据（如修改时间）
    
    参数：
      output_path: 输出文件路径
    
    备份命名规则：
      原文件名_备份_时间戳.后缀
      例如：招股说明书_原始文本_备份_20240507_152030.json
    
    示例：
      backup_previous_output("data/parsed/output.json")
      # 如果output.json存在，会复制为 output_备份_20240507_152030.json
    """
    path = Path(output_path)
    if path.exists():  # 只有文件存在时才备份
        # 构造备份文件名：原文件名_备份_时间戳.后缀
        backup_name = f"{path.stem}_备份_{RUN_TIMESTAMP}{path.suffix}"
        backup_path = path.parent / backup_name
        shutil.copy2(path, backup_path)  # copy2保留文件元数据
        logger.info(f"已备份旧文件 -> {backup_path}")


# =============================================================================
# 表格格式化函数
# =============================================================================
def format_table_as_text(table_data: list[list[str]], table_index: int) -> str:
    """
    作用：将表格数据格式化为易读的Markdown文本
    
    原理：
      - Markdown表格格式被广泛支持，LLM也能很好理解
      - 保留表头和行列对应关系，避免信息丢失
      - 用 | 分隔列，用 --- 分隔表头和数据
    
    参数：
      table_data: 二维数组，每个元素是单元格文本
                  例如：[["姓名", "年龄"], ["张三", "25"]]
      table_index: 表格序号（从0开始），用于生成标题
    
    返回：
      Markdown格式的表格文本
    
    示例：
      输入：[["项目", "2023年"], ["营收", "100万"]]
      输出：
        [表格 1]
        项目 | 2023年
        --- | ---
        营收 | 100万
    """
    # 防御性编程：检查数据有效性
    if not table_data or not table_data[0]:
        return ""  # 空数据返回空字符串
    
    lines = []  # 存储每一行文本
    lines.append(f"[表格 {table_index + 1}]")  # 表格标题，如"[表格 1]"
    
    # -------------------------------------------------------------------------
    # 表头行（第一行）
    # -------------------------------------------------------------------------
    header = table_data[0]  # 第一行作为表头
    # 将每个单元格转为字符串并去除空白，用" | "连接
    # 例如：["项目", "2023年"] → "项目 | 2023年"
    lines.append(" | ".join(str(cell).strip() if cell else "" for cell in header))
    
    # 分隔线：每个列对应一个"---"
    # 例如：2列 → "--- | ---"
    lines.append(" | ".join(["---"] * len(header)))
    
    # -------------------------------------------------------------------------
    # 数据行（第二行开始）
    # -------------------------------------------------------------------------
    for row in table_data[1:]:
        # 过滤空行：如果所有单元格都是空的，跳过
        if any(cell.strip() for cell in row if cell):
            # 格式化每一行数据
            lines.append(" | ".join(str(cell).strip() if cell else "" for cell in row))
    
    # 用换行符连接所有行
    return "\n".join(lines)


# =============================================================================
# 表格提取函数
# =============================================================================
def extract_tables_from_page(page: fitz.Page, page_num: int) -> list[dict]:
    """
    作用：从PDF页面中提取表格
    
    原理：
      - PyMuPDF的find_tables()使用机器学习模型检测页面中的表格区域
      - 自动识别表格的边框、单元格位置
      - extract()方法提取每个单元格的文本内容
    
    参数：
      page: fitz.Page对象，PDF的一页
      page_num: 页码（从1开始），用于日志和输出
    
    返回：
      表格字典列表，每个字典包含：
        - type: "table"（类型标记）
        - page_num: 页码
        - table_index: 表格序号
        - text: Markdown格式的表格文本
        - row_count: 行数
        - col_count: 列数
        - char_count: 字符数
    
    异常处理：
      - 单表格提取失败：记录警告，继续处理其他表格
      - 整页表格检测失败：记录警告，返回空列表
    """
    tables = []  # 存储提取到的表格
    try:
        # find_tables() 返回表格查找器对象
        # 内部使用算法检测页面中的表格结构
        tab_finder = page.find_tables()
        
        if tab_finder.tables:  # 如果检测到表格
            logger.info(f"第 {page_num} 页发现 {len(tab_finder.tables)} 个表格")
            
            # 遍历每个检测到的表格
            for idx, table in enumerate(tab_finder.tables):
                try:
                    # extract() 提取表格数据，返回二维数组
                    # 每个元素是对应单元格的文本字符串
                    table_data = table.extract()
                    
                    # 检查数据有效性：非空且至少有一行
                    if table_data and len(table_data) > 0:
                        # 将二维数组格式化为Markdown文本
                        table_text = format_table_as_text(table_data, idx)
                        
                        # 再次检查格式化后的文本非空
                        if table_text:
                            # 构建表格信息字典
                            tables.append({
                                "type": "table",  # 类型标记，区别于普通文本
                                "page_num": page_num,  # 表格所在页码
                                "table_index": idx + 1,  # 表格序号（从1开始）
                                "text": table_text,  # Markdown格式的表格文本
                                "row_count": len(table_data),  # 数据行数（含表头）
                                "col_count": len(table_data[0]) if table_data[0] else 0,  # 列数
                                "char_count": len(table_text),  # 总字符数（用于统计）
                            })
                except Exception as e:
                    # 单个表格提取失败，记录警告但不中断整体流程
                    logger.warning(f"第 {page_num} 页表格 {idx + 1} 提取失败: {e}")
                    continue  # 继续处理下一个表格
    except Exception as e:
        # 整页表格检测失败（如页面结构异常）
        logger.warning(f"第 {page_num} 页表格检测失败: {e}")
    
    return tables


# =============================================================================
# 主提取函数
# =============================================================================
def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    作用：把PDF文件逐页读出来，提取每一页的文字和表格
    
    原理：
      1. fitz.open() 打开PDF文件，返回文档对象
      2. 循环每一页：
         a. page.get_text() 提取普通文字（所有文本元素）
         b. page.find_tables() 检测并提取表格结构
      3. 跳过空白页（无内容的页）
      4. 单页失败跳过，不中断整体流程（容错设计）
    
    参数：
      pdf_path: PDF文件路径
    
    返回：
      内容块列表，每个元素是字典：
        文本块：{"page_num", "source_pdf", "text", "char_count", "type": "text"}
        表格块：{"page_num", "source_pdf", "text", "char_count", "type": "table", ...}
    
    设计要点：
      - 一页可能产生多个内容块（1个文本 + N个表格）
      - 文本和表格分开存储，便于后续差异化处理
      - 异常处理：单页失败不影响其他页
    """
    pages = []  # 存储所有内容块
    total_tables = 0  # 统计总表格数（用于日志）
    
    # -------------------------------------------------------------------------
    # 打开PDF文件
    # -------------------------------------------------------------------------
    try:
        doc = fitz.open(pdf_path)  # 打开PDF，返回Document对象
    except Exception as e:
        logger.error(f"无法打开PDF文件: {e}")
        return pages  # 返回空列表
    
    total_pages = len(doc)  # 获取总页数
    logger.info(f"PDF 共 {total_pages} 页")
    
    # -------------------------------------------------------------------------
    # 逐页处理
    # -------------------------------------------------------------------------
    for page_index in range(total_pages):
        try:
            page = doc[page_index]  # 获取当前页对象
            page_num = page_index + 1  # 页码从1开始（用户友好）
            
            # -----------------------------------------------------------------
            # 1. 提取普通文本
            # -----------------------------------------------------------------
            # page.get_text() 提取页面所有文本，返回字符串
            # 包括段落、标题、页眉页脚等所有文本元素
            text = page.get_text()
            
            # 检查文本非空（跳过空白页）
            if text and text.strip():
                pages.append({
                    "page_num": page_num,  # 页码
                    "source_pdf": str(pdf_path),  # 来源PDF路径（用于溯源）
                    "text": text.strip(),  # 去除首尾空白
                    "char_count": len(text.strip()),  # 字符数（用于质量检查）
                    "type": "text",  # 类型标记：普通文本
                })
            
            # -----------------------------------------------------------------
            # 2. 提取表格（如果页面有表格）
            # -----------------------------------------------------------------
            tables = extract_tables_from_page(page, page_num)
            if tables:  # 如果提取到表格
                pages.extend(tables)  # 将表格添加到内容块列表
                total_tables += len(tables)  # 统计表格数量
                
        except Exception as e:
            # 单页处理失败，记录警告但继续处理下一页
            logger.warning(f"第 {page_index + 1} 页读取失败，已跳过: {e}")
            continue  # 继续下一页
    
    # 关闭PDF文件，释放资源
    doc.close()
    
    # 输出统计信息
    logger.info(f"成功读取 {len(pages)} 个内容块（共 {total_pages} 页，{total_tables} 个表格）")
    return pages


# =============================================================================
# 章节识别函数
# =============================================================================
def guess_section(text: str) -> str | None:
    """
    作用：判断当前页属于招股说明书的哪个章节
    
    原理：
      - 招股说明书的章节标题通常在页面顶部
      - 需要跳过页眉（公司名、页码）等干扰信息
      - 章节标题有两种格式：
        1. 编号格式："第X节 XXX" 或 "第X章 XXX"
        2. 无编号格式："重大事项提示"、"释义"等
    
    参数：
      text: 页面文本内容
    
    返回：
      章节名字符串（如"第一节 释义"），没找到返回None
    
    处理逻辑：
      1. 只看前10行（章节标题不会出现在页面中间）
      2. 跳过页眉行（匹配PDF_HEADER_KEYWORDS）
      3. 跳过页码行（匹配PDF_PAGE_NUMBER_PATTERN）
      4. 匹配编号章节标题
      5. 匹配无编号章节标题
    """
    # 将文本按换行拆分成行列表
    lines = text.split("\n")
    
    # 只看前10行（章节标题通常在页面顶部）
    for line in lines[:10]:
        line = line.strip()  # 去除首尾空白
        if not line:
            continue  # 跳过空行
        
        # ---------------------------------------------------------------
        # 步骤1：跳过页眉行
        # ---------------------------------------------------------------
        is_header = False
        for kw in PDF_HEADER_KEYWORDS:
            if kw in line:
                is_header = True  # 标记为页眉
                break
        if is_header:
            continue  # 跳过页眉行
        
        # ---------------------------------------------------------------
        # 步骤2：跳过页码行
        # ---------------------------------------------------------------
        # 页码格式如 "1-1-1"，用正则匹配
        if re.match(PDF_PAGE_NUMBER_PATTERN, line):
            continue  # 跳过页码行
        
        # ---------------------------------------------------------------
        # 步骤3：匹配编号章节标题
        # ---------------------------------------------------------------
        # 正则表达式：^第[一二三四五六七八九十]+[节章]\s*\S+
        # 含义：以"第"开头，跟中文数字，跟"节"或"章"，跟任意非空白字符
        # 例如："第一节 释义"、"第二章 概览"
        m = re.match(r"^(第[一二三四五六七八九十]+[节章]\s*\S+)", line)
        if m:
            return m.group(1)  # 返回匹配到的章节名
        
        # ---------------------------------------------------------------
        # 步骤4：匹配无编号章节标题
        # ---------------------------------------------------------------
        # 招股说明书中的特殊章节，没有编号
        for kw in ["目  录", "目录", "声 明", "声明",
                    "重大事项提示", "释 义", "释义", "概 览", "概览"]:
            # 检查是否包含关键词，且长度小于30（避免匹配到正文）
            if kw in line and len(line) < 30:
                return kw  # 返回章节名
    
    # 前10行都没找到章节标题
    return None


# =============================================================================
# 文本清理函数
# =============================================================================
def clean_text(text: str) -> str:
    """
    作用：清理文本，去掉多余的空行
    
    原理：
      - PDF提取的文字段落之间可能有多个换行（原始格式残留）
      - 3个以上连续换行压缩为2个，保持段落间距一致
      - 不影响内容，只优化格式
    
    参数：
      text: 原始文本
    
    返回：
      清理后的文本
    
    示例：
      输入："第一段\n\n\n\n第二段"（4个换行）
      输出："第一段\n\n第二段"（2个换行）
    """
    # 正则表达式：\n{3,} 匹配3个及以上的连续换行
    # 替换为 \n\n（2个换行，即一个空行）
    return re.sub(r"\n{3,}", "\n\n", text)


# =============================================================================
# 质量报告函数
# =============================================================================
def print_quality_report(pages: list[dict]):
    """
    作用：输出解析质量报告
    
    原理：
      - 让用户快速了解解析结果是否正常
      - 对比预设阈值，异常时输出警告
      - 帮助发现PDF质量问题（如扫描件、加密PDF）
    
    统计指标：
      - 总页数、总字符数、平均每页字符数
      - 检测到的章节数量和分布
      - 空白页比例（检测解析异常）
    
    参数：
      pages: 解析后的内容块列表
    """
    if not pages:
        logger.warning("无数据，跳过质量报告")
        return
    
    # 计算总字符数
    total_chars = sum(p["char_count"] for p in pages)
    # 计算平均每页字符数
    avg_chars = total_chars / len(pages) if pages else 0
    
    # 提取所有章节名（去重，保持顺序）
    secs = [p["section"] for p in pages if p["section"]]
    unique = list(dict.fromkeys(secs))  # dict.fromkeys()保持插入顺序去重
    
    # 输出报告标题
    logger.info("=" * 50)
    logger.info("  质量报告")
    logger.info("=" * 50)
    
    # 输出基本统计
    logger.info(f"总页数: {len(pages)}, 总字符: {total_chars:,}, 平均: {avg_chars:.0f}字符")
    logger.info(f"检测到章节: {len(unique)} 个")
    
    # 输出每个章节的页数分布
    for s in unique:
        cnt = sum(1 for p in pages if p["section"] == s)
        logger.info(f"  - {s}: {cnt} 页")
    
    # -------------------------------------------------------------------------
    # 质量检查1：页数是否过少
    # -------------------------------------------------------------------------
    if len(pages) < QUALITY_MIN_CHUNKS:
        # 页数少于阈值，可能是PDF读取失败或文件损坏
        logger.warning(f"⚠️ 页数偏少: {len(pages)} < {QUALITY_MIN_CHUNKS}")
    
    # -------------------------------------------------------------------------
    # 质量检查2：空白页比例是否过高
    # -------------------------------------------------------------------------
    # 统计空白页数量（text为空或只有空白字符）
    empty = sum(1 for p in pages if not p.get("text", "").strip())
    ratio = empty / len(pages) if pages else 0
    if ratio > QUALITY_MAX_EMPTY_RATIO:
        # 空白页过多，可能是扫描件（图片PDF）或加密PDF
        logger.warning(f"⚠️ 空白页比例过高: {ratio:.1%} > {QUALITY_MAX_EMPTY_RATIO:.0%}")


# =============================================================================
# 程序入口
# =============================================================================
def main():
    """
    作用：程序入口，串联整个PDF解析流程
    
    执行流程：
      1. 校验输入文件是否存在
      2. 备份旧输出文件（如果存在）
      3. 读取PDF，提取文本和表格
      4. 清理文本，识别章节
      5. 输出质量报告
      6. 保存结果到JSON文件
    
    异常处理：
      - 每个步骤失败都会记录日志，尽量不影响后续步骤
      - 关键步骤（如文件不存在）会提前退出
    """
    # 输出开始标志
    logger.info("=" * 50)
    logger.info("  PDF 解析 开始")
    logger.info("=" * 50)
    
    # -------------------------------------------------------------------------
    # 步骤1：检查输入文件
    # -------------------------------------------------------------------------
    if not check_file_exists(str(PDF_PATH)):
        return  # 文件不存在，提前退出
    
    # -------------------------------------------------------------------------
    # 步骤2：备份旧输出
    # -------------------------------------------------------------------------
    backup_previous_output(str(PARSED_JSON_PATH))
    
    # -------------------------------------------------------------------------
    # 步骤3：读取PDF
    # -------------------------------------------------------------------------
    logger.info(f"读取PDF: {PDF_PATH}")
    pages = extract_text_from_pdf(str(PDF_PATH))
    
    if not pages:
        logger.error("未读取到任何有效内容")
        return  # 没有数据，提前退出
    
    # -------------------------------------------------------------------------
    # 步骤4：清理文本 + 章节识别
    # -------------------------------------------------------------------------
    success, fail = 0, 0  # 成功/失败计数
    current_section = None  # 当前章节名（用于传播）
    
    for page in pages:
        try:
            # 清理文本（压缩多余空行）
            page["text"] = clean_text(page["text"])
            
            # 尝试识别章节
            d = guess_section(page["text"])
            if d:
                current_section = d  # 更新当前章节
            
            # 为当前页设置章节（如果识别失败，沿用上一页的章节）
            page["section"] = current_section
            success += 1
        except Exception as e:
            # 单页处理失败，记录日志但继续
            logger.warning(f"第 {page['page_num']} 页处理出错: {e}")
            page["section"] = current_section  # 沿用上一页章节
            fail += 1
    
    logger.info(f"处理完成: 成功 {success} 页, 跳过 {fail} 页")
    
    # -------------------------------------------------------------------------
    # 步骤5：输出质量报告
    # -------------------------------------------------------------------------
    print_quality_report(pages)
    
    # -------------------------------------------------------------------------
    # 步骤6：保存结果
    # -------------------------------------------------------------------------
    # 创建输出目录（如果不存在）
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # 保存为JSON文件
    # ensure_ascii=False: 保留中文字符，不转义为\uXXXX
    # indent=2: 使用2个空格缩进，便于阅读
    with open(PARSED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    
    logger.info(f"已保存: {PARSED_JSON_PATH}")
    logger.info("🎉 PDF 解析完成")


# =============================================================================
# 脚本执行入口
# =============================================================================
# __name__ == "__main__" 确保只在直接运行此脚本时执行main()
# 如果作为模块被导入，不会自动执行
if __name__ == "__main__":
    main()
