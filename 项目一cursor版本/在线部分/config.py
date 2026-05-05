import os


def _env_path(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


DATA_JSONL = _env_path(
    "DATA_JSONL",
    r"C:\Users\qjx\Desktop\2309b\专高阶段\专高六\项目一cursor版本\数据清洗\中华人民共和国民法典_chunks_with_vectors.jsonl",
)
DOCTOR_DATA_JSONL = _env_path(
    "DOCTOR_DATA_JSONL",
    r"C:\Users\qjx\Desktop\2309b\专高阶段\专高六\项目一cursor版本\离线部分\高血压_chunked.jsonl",
)

# 模型路径支持环境变量配置，默认保留本机缓存路径
BGE_M3_PATH = _env_path(
    "BGE_M3_PATH",
    r"C:\Users\qjx\.cache\modelscope\hub\models\BAAI\bge-m3",
)
BGE_RERANK_PATH = _env_path(
    "BGE_RERANK_PATH",
    r"C:\Users\qjx\.cache\modelscope\hub\models\BAAI\bge-reranker-base",
)

MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1").strip()
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530").strip()
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "civil_code_rag").strip()
DOCTOR_COLLECTION_NAME = os.getenv("DOCTOR_COLLECTION_NAME", "hypertension_rag_chunks").strip()

BM25_TOP_K = int(os.getenv("BM25_TOP_K", "20"))
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "20"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.15"))

LLM_API_BASE = os.getenv("LLM_API_BASE", "http://127.0.0.1:11434").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "Ollama").strip()
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen2.5:0.5b").strip()
FRONTEND_API_BASE = os.getenv("FRONTEND_API_BASE", "http://127.0.0.1:8000").strip()
LLM_WAIT_SECONDS = float(os.getenv("LLM_WAIT_SECONDS", "1.5"))

# Web fallback / crawl 配置
WEB_TOP_K = int(os.getenv("WEB_TOP_K", "3"))
WEB_TIMEOUT_SECONDS = float(os.getenv("WEB_TIMEOUT_SECONDS", "10"))
# 并行与重试控制
SEARCH_RETRIES = int(os.getenv("WEB_SEARCH_RETRIES", "3"))
BACKOFF_BASE = float(os.getenv("WEB_BACKOFF_BASE", "0.5"))
PARALLEL_SEARCH_TIMEOUT = float(os.getenv("WEB_PARALLEL_SEARCH_TIMEOUT", "8"))
WEB_BACKOFF_JITTER = float(os.getenv("WEB_BACKOFF_JITTER", "0.2"))

# User-Agent rotation and optional proxy
WEB_USER_AGENTS = os.getenv("WEB_USER_AGENTS", "").split("||") if os.getenv("WEB_USER_AGENTS") else []
# 单个代理配置，例如 http://user:pass@host:port 或 socks5://...
WEB_PROXY = os.getenv("WEB_PROXY", "").strip()
