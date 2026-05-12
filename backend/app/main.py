"""
<<<<<<< HEAD
backend/app/main.py — FastAPI入口
==================================
作用：创建FastAPI应用，注册路由和中间件
原理：
  1. FastAPI是高性能的异步Web框架
  2. 自动API文档（Swagger UI / ReDoc）
  3. Pydantic模型自动校验请求/响应
  4. 异步支持，高并发处理
=======
backend/app/main.py — FastAPI应用入口
=======================================
作用：创建和配置FastAPI应用实例，注册所有路由和中间件
原理：
  1. FastAPI是一个高性能异步Web框架，基于Starlette
  2. 自动生成API文档（Swagger UI + ReDoc）
  3. 中间件的注册顺序影响执行顺序——后注册的先执行
  4. 通过python -m app.main 启动（不要直接python main.py）
>>>>>>> 6d37b33 (提交信息)
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

<<<<<<< HEAD
# 使用相对导入，不需要PYTHONPATH
=======
# 使用相对导入（这样才支持 python -m app.main）
>>>>>>> 6d37b33 (提交信息)
from .config import settings
from .api.v1.chat import router as chat_router
from .middleware.logging_middleware import LoggingMiddleware
from .middleware.rate_limit import RateLimitMiddleware

# 配置日志
<<<<<<< HEAD
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
=======
# 作用：设置全局日志格式，所有模块共用
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),  # 日志级别，默认INFO
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"  # 时间 [级别] 模块名 - 消息
>>>>>>> 6d37b33 (提交信息)
)

logger = logging.getLogger(__name__)


# =============================================================================
# 创建FastAPI应用
# =============================================================================

<<<<<<< HEAD
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="招股说明书智能问答系统API",
    docs_url="/docs",      # Swagger UI路径
    redoc_url="/redoc",    # ReDoc路径
=======
# 创建FastAPI实例
# 原理：FastAPI是一个类，实例化时需要传入应用信息
# docs_url和redoc_url是自动文档的路径
app = FastAPI(
    title=settings.APP_NAME,          # 应用名: "Prospectus QA Enterprise"
    version=settings.APP_VERSION,      # 版本: "2.0.0"
    description="招股说明书智能问答系统API",  # 描述
    docs_url="/docs",     # Swagger UI 文档路径（http://localhost:8000/docs）
    redoc_url="/redoc",   # ReDoc 文档路径（http://localhost:8000/redoc）
>>>>>>> 6d37b33 (提交信息)
)


# =============================================================================
<<<<<<< HEAD
# 注册中间件（注意：后注册的先执行）
# =============================================================================

# 限流中间件：限制请求频率（每秒100请求）
app.add_middleware(
    RateLimitMiddleware,
    max_requests=100,
    window_seconds=60,
)

# 日志中间件：记录请求信息
app.add_middleware(LoggingMiddleware)

# CORS中间件：允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 允许所有来源（生产环境应限制）
    allow_credentials=True,
    allow_methods=["*"],           # 允许所有HTTP方法
=======
# 注册中间件（注意：后注册的先执行——栈结构）
# =============================================================================
# 原理：中间件像洋葱，请求从外到内经过，响应从内到外经过
#       后注册的中间件在最外层，最先处理请求，最后处理响应
# 执行顺序：CORS → Logging → RateLimit → 实际路由处理

# 限流中间件（最内层，最后处理请求，最先处理响应）
app.add_middleware(
    RateLimitMiddleware,
    max_requests=100,       # 每个IP每分钟最多100次请求
    window_seconds=60,      # 时间窗口60秒
)

# 日志中间件（中间层）
app.add_middleware(LoggingMiddleware)

# CORS中间件（最外层，最先处理请求）
# 作用：允许跨域请求，让前端页面（即使在不同端口/域名）能调用API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 允许所有来源（生产环境应限制为具体域名）
    allow_credentials=True,         # 允许携带Cookie
    allow_methods=["*"],           # 允许所有HTTP方法（GET/POST/PUT/DELETE...）
>>>>>>> 6d37b33 (提交信息)
    allow_headers=["*"],           # 允许所有请求头
)


# =============================================================================
# 注册路由
# =============================================================================

<<<<<<< HEAD
# 对话API
app.include_router(
    chat_router,
    prefix="/api/v1",    # API版本前缀
    tags=["v1"]
=======
# 对话API路由
# 所有 /api/v1/chat/ 开头的请求都由 chat_router 处理
app.include_router(
    chat_router,
    prefix="/api/v1",    # API版本前缀，方便后续升级v2
    tags=["v1"]          # Swagger文档中的标签分组
>>>>>>> 6d37b33 (提交信息)
)


# =============================================================================
# 健康检查
# =============================================================================

@app.get("/health")
async def health_check():
    """
<<<<<<< HEAD
    健康检查接口
    
    用于监控服务状态
    
    返回：
        服务状态信息
=======
    作用：健康检查接口，用于监控服务是否正常运行
    原理：不依赖任何服务（即使Milvus/LLM挂了，这个接口也能返回）
          监控系统定期GET /health，如果返回非200则认为服务异常
    返回：
        {
            "status": "healthy",
            "version": "2.0.0",
            "environment": "development"
        }
>>>>>>> 6d37b33 (提交信息)
    """
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "environment": settings.ENV,
    }


# =============================================================================
# 启动事件
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """
<<<<<<< HEAD
    应用启动时执行
    
    用于初始化资源（如数据库连接、模型加载等）
=======
    作用：应用启动时自动执行
    原理：FastAPI在服务启动完成后、接收请求前调用
          用于初始化资源、打印启动信息
>>>>>>> 6d37b33 (提交信息)
    """
    logger.info(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} 启动")
    logger.info(f"环境: {settings.ENV}")
    logger.info(f"文档地址: http://{settings.HOST}:{settings.PORT}/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """
<<<<<<< HEAD
    应用关闭时执行
    
    用于释放资源
=======
    作用：应用关闭时自动执行
    原理：FastAPI在收到关闭信号后、进程退出前调用
          用于释放资源、关闭连接
>>>>>>> 6d37b33 (提交信息)
    """
    logger.info("👋 服务关闭")


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
<<<<<<< HEAD
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=1,  # Windows 上 workers 必须设为 1
        reload=settings.DEBUG,  # 调试模式自动重载
=======
    """
    作用：通过 python -m app.main 启动时的入口
    原理：
      1. uvicorn.run()启动ASGI服务器
      2. workers=1：Windows上必须设为1（Windows不支持多worker）
      3. reload=True：调试模式，代码修改后自动重启
    注意：
      要在 backend/ 目录下通过 python -m app.main 运行
      不要在 app/ 目录下 python main.py，否则相对导入会报错
    """
    import uvicorn

    uvicorn.run(
        "app.main:app",           # ASGI应用路径
        host=settings.HOST,        # 监听地址（0.0.0.0 = 所有网卡）
        port=settings.PORT,        # 端口（默认8000）
        workers=1,                 # Windows上必须为1
        reload=settings.DEBUG,     # 调试模式自动热重载
>>>>>>> 6d37b33 (提交信息)
    )
