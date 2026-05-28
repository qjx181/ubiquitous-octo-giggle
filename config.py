"""项目二配置 — 本地Embedding + 本地LLM"""

# Milvus连接
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_NAME = "zhidu_chunks"

# Embedding模型
EMBEDDING_MODEL_PATH = "C:/Users/qjx/.cache/modelscope/hub/models/BAAI/bge-m3"

# 向量维度
EMBEDDING_DIMENSION = 1024

# 文本分块
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# 文本解析
PARSER_BACKEND = "mineru"

# LLM服务
LLM_BASE_URL = "http://127.0.0.1:11434"
LLM_MODEL_NAME = "qwen2.5:7b"

# Milvus文本字段最大长度
MILVUS_TEXT_MAX_LENGTH = 10000
