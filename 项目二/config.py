"""
config.py — 项目配置文件
=========================
作用：集中管理整个项目的配置参数，包括路径、常量、日志设置
原理：所有脚本都导入此文件，确保配置统一，避免硬编码分散在各处
"""

import logging  # Python标准日志库，用于记录程序运行信息
from pathlib import Path  # 面向对象的文件路径处理，比字符串路径更安全
from datetime import datetime  # 日期时间处理，用于生成时间戳

# =============================================================================
# 项目根目录
# =============================================================================
# __file__ 是当前文件(config.py)的路径
# .resolve() 转换为绝对路径，并解析符号链接
# .parent 获取父目录，即项目根目录
PROJECT_DIR = Path(__file__).resolve().parent

# =============================================================================
# 日志配置
# =============================================================================
# 日志文件存放目录：项目根目录/logs/
LOG_DIR = PROJECT_DIR / "logs"

# 日志级别：INFO 表示记录 INFO、WARNING、ERROR、CRITICAL 级别的日志
# DEBUG < INFO < WARNING < ERROR < CRITICAL
LOG_LEVEL = logging.INFO

# 日志格式：
# %(asctime)s - 时间戳，如 2024-01-15 10:30:45,123
# %(levelname)s - 日志级别，如 INFO、ERROR
# %(name)s - 日志器名称，如 parse_pdf、chunk_text
# %(message)s - 具体的日志内容
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"

# 运行时间戳，格式：年月日_时分秒
# 用途：为每次运行生成唯一标识，用于备份文件名、日志文件名
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# =============================================================================
# 数据目录和文件路径
# =============================================================================
# 数据根目录：项目根目录/data/
DATA_DIR = PROJECT_DIR / "data"

# PDF源文件路径：data/招股说明书1.pdf
PDF_PATH = DATA_DIR / "招股说明书1.pdf"

# 解析结果目录：data/parsed/
PARSED_DIR = DATA_DIR / "parsed"

# 解析结果文件：data/parsed/招股说明书_原始文本.json
# 存储 parse_pdf.py 的输出，每页一个JSON对象
PARSED_JSON_PATH = PARSED_DIR / "招股说明书_原始文本.json"

# 分块结果目录：data/chunks/
CHUNKS_DIR = DATA_DIR / "chunks"

# 分块结果文件：data/chunks/招股说明书_分块.jsonl
# JSONL格式：每行一个JSON对象，适合流式处理大文件
CHUNKS_JSONL_PATH = CHUNKS_DIR / "招股说明书_分块.jsonl"

# 向量结果文件：data/chunks/招股说明书_带向量.jsonl
# 存储 generate_embeddings.py 的输出，包含向量数据
EMBEDDINGS_JSONL_PATH = CHUNKS_DIR / "招股说明书_带向量.jsonl"

# =============================================================================
# 文本分块参数
# =============================================================================
# 每个chunk的最大字符数
# 设置800是因为：BGE-M3模型对长文本处理效果好，但过长会截断
# 800字符约等于200-300个中文词，能保留完整语义
CHUNK_MAX_CHARS = 800

# chunk之间的重叠字符数
# 原理：相邻chunk共享150字符，确保语义连续性，避免信息断裂
# 例如：chunk1包含第1-800字符，chunk2包含第650-1450字符（重叠150）
CHUNK_OVERLAP = 150

# 最小行长度：小于此长度的行被视为无效内容（如页码、页眉）
MIN_LINE_LENGTH = 30

# =============================================================================
# PDF解析参数
# =============================================================================
# PDF页眉关键词列表
# 用途：识别并跳过页眉行，避免干扰章节识别
# "武汉兴图新科"是公司名，"招股意向书"/"招股说明书"是文档类型
PDF_HEADER_KEYWORDS = ["武汉兴图新科", "招股意向书", "招股说明书"]

# 页码正则表达式：匹配 "1-1-1" 格式的页码
# 用途：识别并跳过页码行
PDF_PAGE_NUMBER_PATTERN = r"^\d+-\d+-\d+$"

# =============================================================================
# Embedding模型配置
# =============================================================================
# BGE-M3模型本地路径
# BGE-M3是北京智源研究院开源的多语言Embedding模型
# 特点：支持100+语言，1024维向量，语义理解能力强
EMBEDDING_MODEL_PATH = "C:/Users/qjx/.cache/modelscope/hub/models/BAAI/bge-m3"

# 批量编码大小：每次处理32个文本
# 原理：GPU显存有限，批量处理提高效率，但过大可能OOM（显存溢出）
EMBEDDING_BATCH_SIZE = 32

# 向量维度：BGE-M3输出1024维浮点数向量
# 维度越高，语义表达能力越强，但存储和计算成本也越高
EMBEDDING_DIM = 1024

# =============================================================================
# Milvus向量数据库配置
# =============================================================================
# Milvus服务主机地址
# 127.0.0.1 表示本地运行，如果是远程服务器则改为对应IP
MILVUS_HOST = "127.0.0.1"

# Milvus服务端口：19530是Milvus的默认gRPC端口
# gRPC是高性能的远程过程调用协议，比HTTP更快
MILVUS_PORT = "19530"

# Milvus集合名称：类似数据库的表名
# 一个集合存储一种类型的向量数据
MILVUS_COLLECTION = "prospectus"

# 批量导入大小：每次导入100条记录
# 原理：批量导入减少网络往返，提高效率，但过大可能超时
MILVUS_BATCH_SIZE = 100

# 是否重建集合：True表示删除旧数据重新导入，False表示增量导入
# 注意：重建会丢失所有已有数据！
REBUILD_COLLECTION = False

# =============================================================================
# 质量检查阈值
# =============================================================================
# 最小chunk数：如果解析后chunk数少于100，提示警告
# 用途：检测解析是否异常（如PDF读取失败）
QUALITY_MIN_CHUNKS = 100

# 最大空白页比例：空白页超过5%时提示警告
# 空白页可能表示PDF解析问题或扫描件质量问题
QUALITY_MAX_EMPTY_RATIO = 0.05

# 最小向量化比例：向量化成功率低于95%时提示警告
# 可能原因：模型加载失败、显存不足、文本过长被截断
QUALITY_MIN_EMBEDDING_RATIO = 0.95


# =============================================================================
# 日志设置函数
# =============================================================================
def setup_logger(name: str) -> logging.Logger:
    """
    作用：创建并配置一个日志记录器
    
    原理：
      1. Python的logging模块采用"日志器-处理器-格式化器"三层架构
      2. 日志器(Logger)：负责产生日志记录
      3. 处理器(Handler)：负责输出日志到不同目的地（控制台、文件）
      4. 格式化器(Formatter)：定义日志的输出格式
    
    参数：
      name: 日志器名称，通常用模块名（如 "parse_pdf"）
    
    返回：
      配置好的日志器对象
    
    示例：
      logger = setup_logger("parse_pdf")
      logger.info("开始解析PDF")  # 输出到控制台和文件
    """
    # 获取或创建指定名称的日志器
    # 如果已存在同名日志器，直接返回（避免重复配置）
    logger = logging.getLogger(name)
    
    # 设置日志级别：只记录INFO及以上级别的日志
    logger.setLevel(LOG_LEVEL)
    
    # 禁止日志向上传播到父日志器
    # 原理：Python日志器是层级结构（root → 子），propagate=False防止重复输出
    logger.propagate = False
    
    # 如果日志器已有处理器（说明已配置过），直接返回
    # 避免重复添加处理器导致日志重复输出
    if logger.handlers:
        return logger
    
    # -------------------------------------------------------------------------
    # 处理器1：控制台输出（实时查看）
    # -------------------------------------------------------------------------
    console = logging.StreamHandler()
    console.setLevel(LOG_LEVEL)
    console.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console)
    
    # -------------------------------------------------------------------------
    # 处理器2：文件输出（持久保存）
    # -------------------------------------------------------------------------
    # 创建日志目录（如果不存在）
    # parents=True：递归创建父目录
    # exist_ok=True：目录已存在时不报错
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # 日志文件名格式：模块名_时间戳.log
    # 例如：parse_pdf_20240507_152030.log
    log_file = LOG_DIR / f"{name}_{RUN_TIMESTAMP}.log"
    
    # 创建文件处理器，追加模式，UTF-8编码（支持中文）
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(LOG_LEVEL)
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(fh)
    
    return logger
