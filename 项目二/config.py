"""
config.py — 项目配置文件
=========================
"""

import logging
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parent

LOG_DIR = PROJECT_DIR / "logs"
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

DATA_DIR = PROJECT_DIR / "data"
PDF_PATH = DATA_DIR / "招股说明书1.pdf"
PARSED_DIR = DATA_DIR / "parsed"
PARSED_JSON_PATH = PARSED_DIR / "招股说明书_原始文本.json"
CHUNKS_DIR = DATA_DIR / "chunks"
CHUNKS_JSONL_PATH = CHUNKS_DIR / "招股说明书_分块.jsonl"
EMBEDDINGS_JSONL_PATH = CHUNKS_DIR / "招股说明书_带向量.jsonl"

CHUNK_MAX_CHARS = 800
CHUNK_OVERLAP = 150
MIN_LINE_LENGTH = 30

PDF_HEADER_KEYWORDS = ["武汉兴图新科", "招股意向书", "招股说明书"]
PDF_PAGE_NUMBER_PATTERN = r"^\d+-\d+-\d+$"

EMBEDDING_MODEL_PATH = "C:/Users/qjx/.cache/modelscope/hub/models/BAAI/bge-m3"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_DIM = 1024

MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
MILVUS_COLLECTION = "prospectus"
MILVUS_BATCH_SIZE = 100
REBUILD_COLLECTION = False

QUALITY_MIN_CHUNKS = 100
QUALITY_MAX_EMPTY_RATIO = 0.05
QUALITY_MIN_EMBEDDING_RATIO = 0.95


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False
    if logger.handlers:
        return logger
    console = logging.StreamHandler()
    console.setLevel(LOG_LEVEL)
    console.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{name}_{RUN_TIMESTAMP}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(LOG_LEVEL)
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(fh)
    return logger
