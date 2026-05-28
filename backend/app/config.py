"""
backend/app/config.py — 配置管理
===================================
作用：集中管理整个项目的所有配置参数
原理：使用pydantic-settings库，从环境变量或.env文件读取配置
逻辑：
  1. 定义Settings类，每个字段都有默认值
  2. 可以用.env文件覆盖默认值
  3. 使用lru_cache缓存配置实例，避免重复读取
"""

# 【作用】从functools导入lru_cache装饰器，用于缓存函数返回值
# 【原理】lru_cache是"最近最少使用缓存"，被装饰的函数只执行一次，后续调用直接返回缓存结果
# 【逻辑】get_settings()被多次调用时，每次都创建Settings对象太浪费，用缓存只创建一次
from functools import lru_cache

# 【作用】从pydantic导入Field类，用于定义字段的详细属性（默认值、环境变量名等）
# 【原理】Field可以给字段添加额外的元数据，比如指定从哪个环境变量读取值
from pydantic import Field

# 【作用】从pydantic-settings导入BaseSettings，这是pydantic的配置管理基类
# 【原理】BaseSettings会自动从环境变量和.env文件读取配置值，覆盖代码中的默认值
# 【优势】不需要手动解析.env文件，框架自动搞定
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    作用：项目配置类，集中管理所有可配置参数
    原理：继承BaseSettings后，系统自动从环境变量或.env文件读取值
    使用方式：settings = get_settings()
              然后 settings.EMBEDDING_MODEL_PATH 就能拿到配置值

    配置覆盖优先级（高→低）：
      1. 环境变量（最高）
      2. .env文件
      3. 代码中的默认值（最低）

    面试官可能问：
      Q: pydantic-settings 和 python-dotenv 有什么区别？
      A: python-dotenv只解析.env文件，需要手动 os.getenv() 读取。
         pydantic-settings 是 pydantic 生态的一部分，自动从环境变量
         和 .env 读取配置值，并与类型校验、字段描述等整合。
         项目中在线服务用 pydantic-settings（需要灵活配置），
         离线管道用普通常量（固定参数不需要动态覆盖）。

      Q: 为什么离线config.py和在线config.py分开？
      A: 离线管道是批处理脚本，参数固定（路径、chunk大小等），
         用普通常量更轻量。在线服务需要支持 Docker 部署（通过环境变量
         配置数据库地址等），用 pydantic-settings 更合适。
         职责分离：离线配置关注"本地文件路径"，在线配置关注"服务连接参数"。

      Q: @lru_cache() 在 get_settings() 上有什么用？
      A: get_settings() 被多次调用时（每个模块导入时都调一次），
         lru_cache 确保 Settings() 只创建一次，后续返回缓存实例。
         避免重复读取 .env 文件和校验配置值。这是轻量级的单例模式。

      Q: EMBEDDING_MODEL_PATH 为什么是 Windows 路径？
      A: 开发环境在 Windows 上。生产部署时可以通过环境变量
         EMBEDDING_MODEL_PATH 覆盖为 Linux 路径。
    """

    # ==========================================
    # 应用基本信息
    # ==========================================

    # 【作用】应用名称，显示在API文档标题上
    APP_NAME: str = "Prospectus QA Enterprise"
    # 【作用】应用版本号
    APP_VERSION: str = "2.0.0"
    # 【作用】调试模式开关
    # 【原理】Field(env="DEBUG")表示可以从环境变量DEBUG读取，不设置时用默认值False
    DEBUG: bool = Field(default=False, env="DEBUG")
    # 【作用】运行环境（development/production/testing）
    ENV: str = Field(default="development", env="ENV")

    # ==========================================
    # 服务地址
    # ==========================================

    # 【作用】服务监听地址，0.0.0.0表示监听所有网络接口
    # 【原理】0.0.0.0允许外部设备访问，若用127.0.0.1只能本机访问
    HOST: str = "0.0.0.0"
    # 【作用】服务监听端口
    PORT: int = 8000
    # 【作用】worker进程数（多进程处理请求）
    WORKERS: int = 4

    # ==========================================
    # JWT认证配置（用于用户登录验证）
    # ==========================================

    # 【作用】JWT加密密钥，用于签发和验证Token
    # 【注意】生产环境必须修改这个密钥，否则任何人都可以伪造Token
    SECRET_KEY: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    # 【作用】Access Token过期时间（分钟）
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    # 【作用】JWT加密算法
    ALGORITHM: str = "HS256"

    # ==========================================
    # Milvus向量数据库配置
    # ==========================================

    # 【作用】Milvus服务地址
    MILVUS_HOST: str = Field(default="localhost", env="MILVUS_HOST")
    # 【作用】Milvus服务端口（gRPC默认19530）
    MILVUS_PORT: int = Field(default=19530, env="MILVUS_PORT")
    # 【作用】Milvus集合名（相当于关系数据库中的表名）
    MILVUS_COLLECTION: str = "prospectus"
    # 【作用】Milvus用户名（如果启用了认证）
    MILVUS_USER: str = Field(default="", env="MILVUS_USER")
    # 【作用】Milvus密码
    MILVUS_PASSWORD: str = Field(default="", env="MILVUS_PASSWORD")

    # ==========================================
    # Redis缓存配置
    # ==========================================

    # 【作用】Redis服务地址
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    # 【作用】Redis服务端口
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    # 【作用】Redis数据库编号（0-15），默认用0号库
    REDIS_DB: int = 0
    # 【作用】Redis密码（如果没有密码就留空）
    REDIS_PASSWORD: str = Field(default="", env="REDIS_PASSWORD")

    # ==========================================
    # LLM配置（智能路由 + 第三方API）
    # ==========================================

    # 【作用】是否启用LLM路由功能
    # 【原理】启用后根据问题复杂度自动选择SGLang或Ollama
    LLM_ROUTER_ENABLED: bool = Field(default=True, env="LLM_ROUTER_ENABLED")
    # 【作用】SGLang服务地址
    SGLANG_HOST: str = Field(default="http://localhost:30000", env="SGLANG_HOST")
    # 【作用】SGLang上部署的模型名称
    SGLANG_MODEL: str = Field(default="Qwen2.5-1.5B-Instruct", env="SGLANG_MODEL")
    # 【作用】Ollama服务地址
    OLLAMA_HOST: str = Field(default="http://localhost:11434", env="OLLAMA_HOST")
    # 【作用】Ollama上使用的模型名称
    OLLAMA_MODEL: str = Field(default="qwen2.5:0.5b", env="OLLAMA_MODEL")
    # 【作用】复杂任务判断阈值（字符数）
    # 【原理】问题字符串长度超过这个值就判定为复杂任务，交给大模型处理
    LLM_COMPLEX_THRESHOLD: int = Field(default=100, env="LLM_COMPLEX_THRESHOLD")
    # 【作用】第三方API的地址（如硅基流动、DeepSeek）
    API_HOST: str = Field(default="", env="API_HOST")
    # 【作用】第三方API的模型名
    API_MODEL: str = Field(default="", env="API_MODEL")
    # 【作用】第三方API的密钥
    API_KEY: str = Field(default="", env="API_KEY")

    # 【作用】DeepSeek API配置
    # 【原理】DeepSeek 是国产大模型，性价比高，适合复杂推理任务
    DEEPSEEK_HOST: str = Field(default="https://api.deepseek.com/v1", env="DEEPSEEK_HOST")
    DEEPSEEK_MODEL: str = Field(default="deepseek-chat", env="DEEPSEEK_MODEL")
    DEEPSEEK_KEY: str = Field(default="", env="DEEPSEEK_KEY")
    # 【作用】LLM API调用的超时时间（秒），超过这个时间没有响应就重试
    LLM_TIMEOUT: int = 120
    # 【作用】LLM调用失败的最大重试次数
    LLM_MAX_RETRIES: int = 3
    # 【作用】LLM生成的最大token数量（限制回答长度）
    LLM_MAX_TOKENS: int = 512
    # 【作用】LLM生成温度（0=确定性最高，1=最随机，默认0.7平衡）
    # 【原理】温度控制模型输出的随机性：越低越保守，越高越有创意
    LLM_TEMPERATURE: float = 0.7

    # ==========================================
    # Embedding模型配置
    # ==========================================

    # 【作用】BGE-M3模型的本地路径
    # 【原理】因为无法直接访问HuggingFace，所以使用Modelscope下载的本地缓存
    # 【注意】这是Windows路径格式，需要在Windows环境中使用
    EMBEDDING_MODEL_PATH: str = Field(
        default="C:/Users/qjx/.cache/modelscope/hub/models/BAAI/bge-m3",
        env="EMBEDDING_MODEL_PATH"
    )
    # 【作用】批量编码时每批处理的文本数量
    # 【原理】batch_size越大GPU利用率越高，但太大可能显存溢出
    EMBEDDING_BATCH_SIZE: int = 32
    # 【作用】BGE-M3模型输出的向量维度
    # 【原理】BGE-M3输出1024维向量，Milvus集合的向量字段维度必须与此一致
    EMBEDDING_DIM: int = 1024

    # ==========================================
    # 检索配置
    # ==========================================

    # 【作用】向量检索返回的结果数量（Top-K）
    RETRIEVAL_TOP_K: int = 3
    # 【作用】重排序后最终返回的结果数量
    RERANKER_TOP_K: int = 3
    # 【作用】BM25检索在混合检索中的权重（0.3表示占30%）
    BM25_WEIGHT: float = 0.3
    # 【作用】向量检索在混合检索中的权重（0.7表示占70%）
    VECTOR_WEIGHT: float = 0.7
    # 【作用】检索结果的最低相似度分数阈值，低于此值的片段会被过滤掉
    # 【原理】向量检索返回的余弦相似度在0~1之间，低于阈值说明语义不相关
    # 【设计意图】过滤掉低相关性的检索结果，提高Context Precision
    MIN_SCORE_THRESHOLD: float = Field(default=0.3, env="MIN_SCORE_THRESHOLD")

    # ==========================================
    # 缓存配置
    # ==========================================

    # 【作用】缓存过期时间（秒），3600秒=1小时
    CACHE_TTL_SECONDS: int = 3600
    # 【作用】缓存最大条目数，超过此数量会淘汰旧缓存
    CACHE_MAX_SIZE: int = 10000

    # ==========================================
    # 日志配置
    # ==========================================

    # 【作用】日志级别（DEBUG/INFO/WARNING/ERROR），可以从环境变量LOG_LEVEL设置
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    # 【作用】日志输出格式（json格式便于logstash等日志收集系统解析）
    LOG_FORMAT: str = "json"

    # ==========================================
    # 混合检索阈值
    # ==========================================

    # 【作用】BM25混合检索的文档数量上限
    # 【原理】小规模语料库BM25效果好，文档过多时BM25索引构建变慢，超过此值自动禁用
    HYBRID_MAX_DOCS: int = Field(default=10000, env="HYBRID_MAX_DOCS")

    # ==========================================
    # Pydantic内部配置
    # ==========================================

    class Config:
        # 【作用】指定.env文件的路径（默认在项目根目录）
        env_file = ".env"
        # 【作用】指定.env文件的编码格式
        env_file_encoding = "utf-8"
        # 【作用】字段名大小写敏感（APP_NAME和app_name视为不同字段）
        case_sensitive = True


# 【作用】缓存装饰器：get_settings()只执行一次，后续直接返回缓存结果
# 【原理】lru_cache()创建一个缓存字典，键是参数，值是返回值
# 因为get_settings()没有参数，所以结果永久有效
@lru_cache()
def get_settings() -> Settings:
    """
    作用：获取配置实例（单例）
    原理：lru_cache确保只创建一次Settings对象
    返回：Settings实例
    """
    return Settings()


# 【作用】导出全局配置实例，其他模块通过 from ..config import settings 使用
# 【逻辑】模块加载时立即调用get_settings()创建配置实例
settings = get_settings()
