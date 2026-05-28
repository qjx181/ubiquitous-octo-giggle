"""
parse_pdf_mineru.py — MinerU版PDF解析脚本
=============================================
作用：使用MinerU的pipeline后端解析招股说明书PDF，提取文字内容
      输出格式与原始parse_pdf.py兼容（list[dict]），下游chunk_text.py等无需改动

原理：
  1. MinerU的pipeline后端比PyMuPDF(fitz)多了版面分析功能：
     - 自动检测多栏排版，按正确阅读顺序排列
     - 自动丢弃页眉、页脚、页码（通过page_header/page_footer/page_number类型标记）
     - 自动识别公式（LaTeX输出）
     - 自动提取表格（HTML格式）
     - 扫描件自动切换OCR
  2. 使用CONTENT_LIST_V2输出格式，按页分组的结构化内容
  3. 每页的内容块转成与原始parse_pdf.py一致的 dict 格式

对比原始方案（parse_pdf.py）：
  原始：fitz.get_text()直接提取文本 → 含页眉页码 → 下游chunk_text.py手动过滤
  MinerU：pipeline后端自动做完版面分析+内容提取+去噪 → 直接得到干净文本

输入：data/招股说明书1.pdf
输出：data/parsed/招股说明书_原始文本.json（与原parse_pdf.py输出路径一致）

依赖：
  - mineru>=3.1.0（含pipeline后端）
  - pipeline模型：opendatalab/PDF-Extract-Kit-1.0
  - 运行前设置环境变量：MINERU_MODEL_SOURCE=modelscope（或huggingface）
"""

import json
import shutil
import sys
import os
from pathlib import Path

# =============================================================================
# 模块路径设置
# =============================================================================
# 将项目根目录添加到Python模块搜索路径，确保能导入config.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    get_kb_paths,  # v2.0: 获取指定KB的文件路径
    PARSED_DIR,
    RUN_TIMESTAMP,
    setup_logger,
)

from mineru.cli.common import do_parse, read_fn
from mineru.utils.enum_class import MakeMode

# 创建日志记录器
logger = setup_logger("parse_pdf_mineru")


# =============================================================================
# 常量
# =============================================================================
# MinerU content_list_v2中需要丢弃的块类型（非正文元素）
# page_header: 页眉（如"武汉兴图新科电子股份有限公司  招股意向书"）
# page_footer: 页脚
# page_number: 页码（如"1-1-1"）
DISCARD_BLOCK_TYPES = {"page_header", "page_footer", "page_number"}


# =============================================================================
# 备份函数
# =============================================================================
def backup_previous_output(output_path: str) -> None:
    """
    作用：备份旧输出文件，用于回滚
    原理：每次运行可能覆盖旧结果，备份后如果新结果有问题可以恢复
    参数：output_path — 输出文件路径
    """
    path = Path(output_path)
    if path.exists():
        backup_name = f"{path.stem}_备份_{RUN_TIMESTAMP}{path.suffix}"
        backup_path = path.parent / backup_name
        shutil.copy2(path, backup_path)
        logger.info(f"已备份旧文件 -> {backup_path}")


# =============================================================================
# MinerU内容块文本提取
# =============================================================================
def _extract_text_recursive(content) -> str:
    """
    作用：递归提取MinerU内容块中的纯文本

    原理：
      MinerU的CONTENT_LIST_V2格式采用嵌套结构，例如段落块：
        {"type":"paragraph", "content":{"paragraph_content":[{"type":"text","content":"..."}]}}
      需要递归遍历所有层级的 "content" 字段提取字符串

    参数：
      content — 任何类型的嵌套数据（dict、list、str）

    返回：
      提取到的纯文本字符串，层级间以空格连接

    边界情况：
      - content为空/None：返回空字符串
      - content直接是字符串：直接返回
      - content为dict：遍历'content'/'paragraph_content'/'title_content'/'item_content'等键
      - content为list：递归处理每个元素
    """
    texts = []

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        # 直接有content字段
        if "content" in content:
            texts.append(_extract_text_recursive(content["content"]))
        # 段落内容：paragraph_content → [{type, content}, ...]
        if "paragraph_content" in content:
            for item in content["paragraph_content"]:
                texts.append(_extract_text_recursive(item))
        # 标题内容：title_content → [{type, content}, ...]
        if "title_content" in content:
            for item in content["title_content"]:
                texts.append(_extract_text_recursive(item))
        # 列表项内容：item_content → [{type, content}, ...]
        if "item_content" in content:
            for item in content["item_content"]:
                texts.append(_extract_text_recursive(item))
        # 列表项：list_items → [{item_type, item_content}, ...]
        if "list_items" in content:
            for item in content["list_items"]:
                if "item_content" in item:
                    for sub in item["item_content"]:
                        texts.append(_extract_text_recursive(sub))

    elif isinstance(content, list):
        for item in content:
            texts.append(_extract_text_recursive(item))

    return " ".join(filter(None, texts))


def block_to_text(block: dict) -> str | None:
    """
    作用：将MinerU的单个内容块转换为纯文本

    参数：
      block — MinerU内容块dict，格式如：
        {"type":"paragraph", "content":{...}, "bbox":[x0,y0,x1,y1]}

    返回：
      纯文本字符串；如果块应该被丢弃（页眉/页脚/页码）则返回None

    块类型处理策略：
      - page_header/page_footer/page_number → 丢弃（返回None）
      - table → 返回HTML格式（如有）或None（图片表格）
      - image/chart/seal → 跳过（返回None，无可用文本）
      - paragraph/title/list/index → 递归提取文本
    """
    block_type = block.get("type", "")
    content = block.get("content", {})

    # 丢弃非正文元素
    if block_type in DISCARD_BLOCK_TYPES:
        return None

    # 表格：有HTML则返回HTML，否则跳过
    if block_type == "table":
        html = content.get("html", "")
        if html:
            return html
        return None

    # 图片/图表/印章：无法提取文字
    if block_type in ("image", "chart", "seal"):
        return None

    # 正文元素：段落、标题、列表、索引等
    text = _extract_text_recursive(content)
    return text.strip() if text else None


def extract_page_text_from_content_list_v2(
    content_list_v2: list,
    source_pdf: str,
) -> list[dict]:
    """
    作用：将MinerU的CONTENT_LIST_V2输出转换为项目二的标准解析格式

    MinerU输出格式：
      [
        [block_1_1, block_1_2, ...],  # 第1页的内容块列表
        [block_2_1, block_2_2, ...],  # 第2页的内容块列表
        ...
      ]
      每个block：
        {"type": "paragraph"|"title"|"table"|...,
         "content": {...},
         "bbox": [x0, y0, x1, y1]}

    标准输出格式：
      [
        {"page_num": 1, "source_pdf": "...", "text": "...",
         "char_count": N, "type": "text"},
        ...
      ]

    参数：
      content_list_v2 — MinerU输出的按页分组的内容列表
      source_pdf — PDF源文件路径

    返回：
      标准格式的解析结果列表

    设计说明：
      - 每页的所有有效内容块合并为一个"text"条目（保持与原始格式兼容）
      - 页眉/页脚/页码被MinerU自动识别并丢弃
      - 空页跳过不添加记录
      - char_count用于质量检查和统计

    面试官可能问：
      Q: 为什么每页只输出一个text条目，而不是每个内容块独立？
      A: 原始parse_pdf.py的输出格式是"每页一条"，下游chunk_text.py按此格式
         设计。保持兼容性可以减少改动范围。内容块的分割由chunk阶段完成。

      Q: 图片类的表格（html为空）怎么处理？
      A: 跳过。图片表格没有可提取的文本内容，需要OCR才能提取。
         如果要处理图片表格，需要启用pipeline的table功能，或使用hybrid后端。

      Q: 丢弃块类型的判断依据是什么？
      A: MinerU的版面分析模型会为每个内容块标注类型。page_header/page_footer/
         page_number是模型自动判断的，不需要手动配置规则。
    """
    pages = []

    for page_idx, page_contents in enumerate(content_list_v2):
        page_num = page_idx + 1  # 页码从1开始（用户友好）

        if not page_contents:
            logger.debug(f"第 {page_num} 页无有效内容，跳过")
            continue

        # 收集本页所有有效文本
        page_texts = []
        for block in page_contents:
            text = block_to_text(block)
            if text:
                page_texts.append(text)

        if not page_texts:
            continue

        # 合并为本页的完整文本
        combined = "\n".join(page_texts)

        pages.append(
            {
                "page_num": page_num,
                "source_pdf": source_pdf,
                "text": combined,
                "char_count": len(combined),
                "type": "text",
            }
        )

    return pages


# =============================================================================
# MinerU解析主函数
# =============================================================================
def parse_pdf_with_mineru(
    pdf_path: str,
    output_dir: str,
) -> list[dict]:
    """
    作用：使用MinerU pipeline后端解析PDF，返回标准格式的内容列表

    原理：
      1. 调用do_parse()执行MinerU解析，输出到指定目录
      2. 从输出目录读取CONTENT_LIST_V2格式的结果文件
      3. 转换为本项目标准格式

    参数：
      pdf_path — PDF文件的绝对路径
      output_dir — MinerU输出目录

    返回：
      标准格式的解析结果列表
      每条记录包含：page_num, source_pdf, text, char_count, type

    API调用链：
      read_fn(pdf_path) → PDF字节
      → do_parse(output_dir, pdf_bytes, ...) → MinerU核心解析
        → _process_pipeline() → pipeline后端处理
          → MineruPipelineModel() → 加载模型(Layout+OCR+Formula+Table)
          → pipeline_doc_analyze_streaming() → 滑动窗口推理
          → union_make(pdf_info, CONTENT_LIST_V2) → 生成结构化JSON
      → 读取 *_content_list_v2.json → 格式转换

    注意事项：
      - formula_enable=True时加载UniMERNet公式识别模型（~773MB）
      - table_enable=True时加载StructEqTable表格模型（~1.75GB）
      - 环境变量MINERU_MODEL_SOURCE控制模型下载源
      - 首次运行会自动下载缺失的模型文件

    边界情况：
      - 解析失败时抛出异常，由调用方处理
      - 空PDF（0页）返回空列表
      - 扫描件PDF会自动启用OCR
    """
    pdf_name = Path(pdf_path).stem  # 文件名（不含扩展名）
    parse_method = "auto"  # auto：自动检测扫描件/文字版

    logger.info(f"开始MinerU解析: {pdf_path}")

    # -------------------------------------------------------------------------
    # 步骤1：读取PDF文件为字节数据
    # -------------------------------------------------------------------------
    pdf_bytes = read_fn(Path(pdf_path))

    # -------------------------------------------------------------------------
    # 步骤2：调用MinerU解析
    # -------------------------------------------------------------------------
    # do_parse参数说明：
    #   output_dir — 输出目录
    #   pdf_file_names — PDF文件名列表（不含路径）
    #   pdf_bytes_list — PDF字节数据列表
    #   p_lang_list — 语言列表，["ch"]表示中文
    #   backend="pipeline" — 使用pipeline后端（不需要VLM模型）
    #   parse_method="auto" — 自动检测扫描件/文字版
    #   formula_enable=True — 启用公式识别（LaTeX输出）
    #   table_enable=True — 启用表格识别（HTML输出）
    #   其余f_dump_*参数控制输出文件类型（为减少磁盘占用只保留content_list）
    do_parse(
        output_dir=output_dir,
        pdf_file_names=[pdf_name],
        pdf_bytes_list=[pdf_bytes],
        p_lang_list=["ch"],
        backend="pipeline",
        parse_method=parse_method,
        formula_enable=True,
        table_enable=True,
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_dump_md=False,
        f_dump_middle_json=False,
        f_dump_model_output=False,
        f_dump_orig_pdf=False,
        f_dump_content_list=True,
        f_make_md_mode=MakeMode.CONTENT_LIST_V2,
    )

    # -------------------------------------------------------------------------
    # 步骤3：读取解析结果
    # -------------------------------------------------------------------------
    # MinerU输出目录结构：
    #   output_dir/{pdf_name}/{parse_method}/{pdf_name}_content_list_v2.json
    result_dir = Path(output_dir) / pdf_name / parse_method
    result_file = result_dir / f"{pdf_name}_content_list_v2.json"

    if not result_file.exists():
        logger.error(f"MinerU解析结果文件不存在: {result_file}")
        # 回退：尝试读取content_list（V1版本）
        result_file = result_dir / f"{pdf_name}_content_list.json"
        if not result_file.exists():
            raise FileNotFoundError(f"MinerU解析未生成结果文件: {result_file}")

    logger.info(f"读取解析结果: {result_file}")
    with open(result_file, "r", encoding="utf-8") as f:
        content_list_v2 = json.load(f)

    # -------------------------------------------------------------------------
    # 步骤4：转换为标准格式
    # -------------------------------------------------------------------------
    source_pdf = str(Path(pdf_path).resolve())
    pages = extract_page_text_from_content_list_v2(content_list_v2, source_pdf)

    # -------------------------------------------------------------------------
    # 统计信息
    # -------------------------------------------------------------------------
    total_blocks = sum(len(page) for page in content_list_v2)
    discarded = sum(
        1 for page in content_list_v2 for b in page
        if b.get("type") in DISCARD_BLOCK_TYPES
    )

    logger.info(
        f"MinerU解析完成: {len(pages)} 个内容块 "
        f"（{pdf_name}，共 {len(content_list_v2)} 页，"
        f"{total_blocks} 个原始块，已丢弃 {discarded} 个非正文块）"
    )

    return pages


# =============================================================================
# 主入口
# =============================================================================
if __name__ == "__main__":
    """
    执行流程：
      1. 检查PDF文件是否存在
      2. 备份旧解析结果
      3. 使用MinerU解析PDF
      4. 保存解析结果为JSON（兼容原格式）
      5. 输出统计信息

    使用方式：
      # 先设置模型下载源
      export MINERU_MODEL_SOURCE=modelscope  # 或 huggingface
      export MINERU_PDF_RENDER_THREADS=1     # WSL环境需要

      # 运行解析
      python offline/parse_pdf_mineru.py

    与其他脚本的配合：
      - 输出文件被 chunk_text.py 读取进行分块
      - 分块结果被 generate_embeddings.py 向量化
      - 向量被导入 Milvus 供在线服务检索

    注意事项：
      - 首次运行会自动下载pipeline模型（约2-3GB）
    注意事项：
      - 如果遇到BrokenProcessPool错误，设置MINERU_PDF_RENDER_THREADS=1
    """
    # v2.0: 确定知识库名称
    import os, sys as _sys
    kb_name = _sys.argv[1] if len(_sys.argv) > 1 else os.environ.get("KB_NAME", "招股说明书2")
    kb_paths = get_kb_paths(kb_name)
    pdf_path = str(kb_paths["pdf_path"])
    parsed_json_path = kb_paths["parsed_json_path"]

    if not Path(pdf_path).exists():
        logger.error(f"PDF文件不存在: {pdf_path}")
        sys.exit(1)

    # 解析输出目录
    output_dir = str(PARSED_DIR)

    # 备份旧输出
    backup_previous_output(str(parsed_json_path))

    # 执行MinerU解析
    pages = parse_pdf_with_mineru(pdf_path, output_dir)

    # 保存结果
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(parsed_json_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    logger.info(f"解析结果已保存: {parsed_json_path} ({len(pages)} 条)")
    print(f"\nMinerU解析完成！共 {len(pages)} 个内容块")
    print(f"输出文件: {parsed_json_path}")
