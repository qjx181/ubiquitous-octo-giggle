"""
backend/app/config.py — 后端配置管理
=====================================
作用：集中管理所有配置项，统一从环境变量或.env文件读取
原理：
  1. 使用Pydantic Settings管理配置，提供类型安全
  2. 所有配置都有默认值，同时支持通过环境变量覆盖
  3. 支持.env文件，方便不同环境切换配置
  4. 使用lru_cache缓存Settings实例，避免重复读取配置

好处：
  - 硬编码集中管理，改配置只改一个地方
  - 环境变量覆盖，部署时不同环境用不同配置
  - 类型安全，写错类型会在启动时立刻报错（而非运行时）
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    作用：应用配置类，所有配置项集中在这里
    原理：继承BaseSettings，自动从环境变量读取值
          字段名 = 环境变量名（大小写敏感）
          比如 DEBUG=True 会从环境变量 DEBUG 读取
    使用方式：
        settings = get_settings()
        print(settings.MILVUS_HOST)  # 输出 localhost
    """

    # =============================================================================
    # 应用基础配置
    # =============================================================================
    APP_NAME: str = "Prospectus QA Enterprise"   # 应用名称
    APP_VERSION: str = "2.0.0"                   # 应用版本
    DEBUG: bool = Field(default=False, env="DEBUG")  # 调试模式（开启后代码修改自动重启）
    ENV: str = Field(default="development", env="ENV")  # 运行环境（development/staging/production）

    # =============================================================================
    # 服务器配置
    # =============================================================================
    HOST: str = "0.0.0.0"   # 监听所有网络接口（0.0.0.0 = 局域网内其他设备也能访问）
    PORT: int = 8000        # 服务端口
    WORKERS: int = 4        # 工作进程数（Linux下多worker提升并发）

    # =============================================================================
    # 安全配置
    # =============================================================================
    SECRET_KEY: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30  # 访问令牌过期时间（分钟）
    ALGORITHM: str = "HS256"               # JWT加密算法

    # =============================================================================
    # Milvus 向量数据库配置
    # =============================================================================
    MILVUS_HOST: str = Field(default="localhost", env="MILVUS_HOST")   # Milvus服务地址
    MILVUS_PORT: int = Field(default=19530, env="MILVUS_PORT")         # Milvus gRPC端口
    MILVUS_COLLECTION: str = "prospectus"                                # 集合名（必须和离线导入一致）
    MILVUS_USER: str = Field(default="", env="MILVUS_USER")             # Milvus用户名（生产环境）
    MILVUS_PASSWORD: str = Field(default="", env="MILVUS_PASSWORD")     # Milvus密码

    # =============================================================================
    # Redis 缓存配置（可选，自动探测可用性）
    # =============================================================================
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = Field(default="", env="REDIS_PASSWORD")

    # =============================================================================
    # LLM 智能调度配置（SGLang + Ollama 组合）
    # =============================================================================
    # 启用智能调度：根据任务复杂度自动选择引擎
    #   简单任务（事实性问答）→ Ollama（轻量模型，省显存）
    #   复杂任务（分析/推理）→ SGLang（大模型，高质量）
    LLM_ROUTER_ENABLED: bool = Field(default=True, env="LLM_ROUTER_ENABLED")

    # SGLang 配置（大模型，处理复杂任务）
    SGLANG_HOST: str = Field(default="http://localhost:30000", env="SGLANG_HOST")
    SGLANG_MODEL: str = Field(default="Qwen2.5-1.5B-Instruct", env="SGLANG_MODEL")

    # Ollama 配置（轻量模型，处理简单任务）
    OLLAMA_HOST: str = Field(default="http://localhost:11434", env="OLLAMA_HOST")
    OLLAMA_MODEL: str = Field(default="qwen2.5:0.5b", env="OLLAMA_MODEL")

    # 复杂度判断阈值
    LLM_COMPLEX_THRESHOLD: int = Field(default=100, env="LLM_COMPLEX_THRESHOLD")

    # 第三方 API 配置（需要API Key）
    API_HOST: str = Field(default="", env="API_HOST")
    API_MODEL: str = Field(default="", env="API_MODEL")
    API_KEY: str = Field(default="", env="API_KEY")

    # LLM 通用参数
    LLM_TIMEOUT: int = 120      # LLM调用超时（秒），120秒后放弃
    LLM_MAX_RETRIES: int = 3    # 调用失败时的最大重试次数
    LLM_MAX_TOKENS: int = 512   # 最大生成token数
    LLM_TEMPERATURE: float = 0.7  # 温度参数（0~1，0=确定性最高，1=创造性最高）

    # =============================================================================
    # Embedding 配置
    # =============================================================================
    EMBEDDING_MODEL_PATH: str = Field(
        default="C:/Users/qjx/.cache/modelscope/hub/models/BAAI/bge-m3",
        env="EMBEDDING_MODEL_PATH"
    )
    EMBEDDING_BATCH_SIZE: int = 32   # 批量编码时一次处理多少条
    EMBEDDING_DIM: int = 1024        # 向量维度（BGE-M3输出1024维）

    # =============================================================================
    # 检索配置
    # =============================================================================
    RETRIEVAL_TOP_K: int = 3        # 向量检索返回数量
    RERANKER_TOP_K: int = 3         # 重排序后返回数量
    BM25_WEIGHT: float = 0.3        # BM25检索权重（当前未启用混合检索）
    VECTOR_WEIGHT: float = 0.7      # 向量检索权重

    # =============================================================================
    # 缓存配置
    # =============================================================================
    CACHE_TTL_SECONDS: int = 3600    # 缓存过期时间（1小时）
    CACHE_MAX_SIZE: int = 10000      # 最大缓存条目数

    # =============================================================================
    # 日志配置
    # =============================================================================
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")  # 日志级别
    LOG_FORMAT: str = "json"          # 日志格式（json适合日志收集系统）

    # =============================================================================
    # 混合检索 BM25 配置
    # =============================================================================
    # BM25索引加载的最大文档数（超出此数量则跳过BM25）
    HYBRID_MAX_DOCS: int = Field(default=10000, env="HYBRID_MAX_DOCS")

    class Config:
        """
        作用：Pydantic Settings的配置类
        env_file = ".env"：从.env文件读取环境变量
        case_sensitive = True：字段名大小写敏感（比如HOST和host是不同变量）
        """
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    作用：获取配置单例
    原理：lru_cache装饰器缓存函数返回值
          第一次调用时创建Settings()实例（读取.env）
          后续调用直接返回缓存的结果
          避免反复读取文件，提高性能
    返回：
        Settings 配置对象
    """
    return Settings()


# 全局配置实例
# 作用：模块级变量，其他地方 from ..config import settings 直接使用
settings = get_settings()
