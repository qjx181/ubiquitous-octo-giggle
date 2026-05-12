"""
backend/app/config.py — 后端配置管理
=====================================
作用：使用 Pydantic Settings 管理配置，支持环境变量和 .env 文件
原理：
  1. Pydantic 提供类型安全的配置管理
  2. 支持从环境变量、.env 文件加载配置
  3. 多环境分离：dev / staging / prod
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    应用配置类
    
    所有配置项都有默认值，同时支持通过环境变量覆盖
    环境变量名与字段名相同（大小写敏感）
    """
    
    # =============================================================================
    # 应用基础配置
    # =============================================================================
    APP_NAME: str = "Prospectus QA Enterprise"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    ENV: str = Field(default="development", env="ENV")
    
    # =============================================================================
    # 服务器配置
    # =============================================================================
    HOST: str = "0.0.0.0"  # 监听所有网络接口
    PORT: int = 8000       # 服务端口
    WORKERS: int = 4       # 工作进程数
    
    # =============================================================================
    # 安全配置
    # =============================================================================
    SECRET_KEY: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"
    
    # =============================================================================
    # Milvus 向量数据库配置
    # =============================================================================
    MILVUS_HOST: str = Field(default="localhost", env="MILVUS_HOST")
    MILVUS_PORT: int = Field(default=19530, env="MILVUS_PORT")
    MILVUS_COLLECTION: str = "prospectus"
    MILVUS_USER: str = Field(default="", env="MILVUS_USER")
    MILVUS_PASSWORD: str = Field(default="", env="MILVUS_PASSWORD")
    
    # =============================================================================
    # Redis 缓存配置
    # =============================================================================
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = Field(default="", env="REDIS_PASSWORD")
    
    # =============================================================================
    # LLM 智能调度配置（SGLang + Ollama 组合）
    # =============================================================================
    # 启用智能调度：根据任务复杂度自动选择引擎
    #   - 简单任务（事实性问答）→ Ollama（轻量模型，省显存）
    #   - 复杂任务（分析/推理）→ SGLang（大模型，高质量）
    LLM_ROUTER_ENABLED: bool = Field(default=True, env="LLM_ROUTER_ENABLED")

    # SGLang 配置（大模型，处理复杂任务）
    SGLANG_HOST: str = Field(default="http://localhost:30000", env="SGLANG_HOST")
    # 注意：model 名必须与 SGLang 服务端 /v1/models 返回的 id 一致
    # 当前实际运行的是 Qwen2.5-1.5B-Instruct
    SGLANG_MODEL: str = Field(default="Qwen2.5-1.5B-Instruct", env="SGLANG_MODEL")

    # Ollama 配置（轻量模型，处理简单任务）
    # qwen2.5:0.5b 正在后台下载，完成后 Ollama 即可用
    OLLAMA_HOST: str = Field(default="http://localhost:11434", env="OLLAMA_HOST")
    OLLAMA_MODEL: str = Field(default="qwen2.5:0.5b", env="OLLAMA_MODEL")

    # 复杂度判断阈值
    LLM_COMPLEX_THRESHOLD: int = Field(default=100, env="LLM_COMPLEX_THRESHOLD")

    # 第三方 API 配置（需要 API Key）
    API_HOST: str = Field(default="", env="API_HOST")
    API_MODEL: str = Field(default="", env="API_MODEL")
    API_KEY: str = Field(default="", env="API_KEY")

    # LLM 通用参数
    LLM_TIMEOUT: int = 120     # LLM调用超时（秒）
    LLM_MAX_RETRIES: int = 3   # 最大重试次数
    # 注意：SGLang Qwen2.5-1.5B-Instruct 最大上下文只有 4096 tokens
    # max_tokens + 输入 tokens 不能超过 4096
    LLM_MAX_TOKENS: int = 512  # 最大生成token数（设小一点留给输入上下文空间）
    LLM_TEMPERATURE: float = 0.7  # 温度参数（创造性 vs 确定性）
    
    # =============================================================================
    # Embedding 配置
    # =============================================================================
    EMBEDDING_MODEL_PATH: str = Field(
        default="C:/Users/qjx/.cache/modelscope/hub/models/BAAI/bge-m3",
        env="EMBEDDING_MODEL_PATH"
    )
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DIM: int = 1024
    
    # =============================================================================
    # 检索配置
    # =============================================================================
    RETRIEVAL_TOP_K: int = 5       # 向量检索返回数量（Qwen1.5B只有4096上下文，设小一点）
    RERANKER_TOP_K: int = 5        # 重排序后返回数量
    BM25_WEIGHT: float = 0.3       # BM25检索权重
    VECTOR_WEIGHT: float = 0.7     # 向量检索权重
    
    # =============================================================================
    # 缓存配置
    # =============================================================================
    CACHE_TTL_SECONDS: int = 3600   # 缓存过期时间（1小时）
    CACHE_MAX_SIZE: int = 10000     # 最大缓存条目数
    
    # =============================================================================
    # 日志配置
    # =============================================================================
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = "json"
    
    class Config:
        """Pydantic配置"""
        env_file = ".env"              # 从 .env 文件加载
        env_file_encoding = "utf-8"    # 编码格式
        case_sensitive = True          # 环境变量大小写敏感


@lru_cache()
def get_settings() -> Settings:
    """
    获取配置单例
    
    使用 lru_cache 装饰器确保只创建一次 Settings 实例
    避免重复读取环境变量和 .env 文件
    
    返回：
        Settings 配置对象
    """
    return Settings()


# 全局配置实例
settings = get_settings()
