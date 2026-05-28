"""backend/app/main.py — FastAPI应用入口"""

import logging
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="RAG智能问答系统API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from .api.v1.chat import router as chat_router
app.include_router(chat_router, prefix="/api/v1", tags=["v1"])


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.APP_VERSION, "environment": settings.ENV}


@app.on_event("startup")
async def startup_event():
    logger.info(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} 启动")
    logger.info(f"环境: {settings.ENV} | 文档: http://{settings.HOST}:{settings.PORT}/docs")

    # 同步预加载Embedding + RetrievalService + LLM Router
    # 启动慢几秒，但第一个请求秒回
    import concurrent.futures
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        def _preload_all():
            # 1. Embedding模型（耗时最长）
            from .core.embedding_service import get_embedding_service
            get_embedding_service()
            logger.info("✅ Embedding模型预加载完成")
            # 2. RetrievalService（含Milvus连接 + LLM Router初始化）
            from .core.retrieval import get_retrieval_service
            get_retrieval_service()
            logger.info("✅ RetrievalService预加载完成")
        await loop.run_in_executor(pool, _preload_all)

    logger.info("🎯 所有服务预加载完成，首次请求将秒回")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("👋 服务关闭")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, workers=1, reload=settings.DEBUG)
