"""
backend/app/middleware/logging_middleware.py — 请求日志中间件
===============================================================
作用：记录每个HTTP请求的详细信息（谁、什么时间、请求了什么、花了多久）
原理：
  1. FastAPI中间件像一个"钩子"，在请求处理前后执行
  2. 请求进来时记录开始时间和请求信息
  3. 请求处理完成后记录状态码和耗时
  4. 在响应头中添加X-Process-Time，方便前端或调试工具查看耗时

好处：
  - 快速定位慢请求（耗时超过几秒的请求需要优化）
  - 监控异常状态码（5xx表示服务器错误）
  - 统计各接口的调用频率
"""

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    作用：记录HTTP请求的日志
    原理：在请求开始和结束时分别记录日志
      开始时：记录请求方法、路径、客户端IP
      结束时：记录状态码、处理耗时
    """

    async def dispatch(self, request: Request, call_next):
        """
        作用：处理每个请求，记录日志
        参数：
            request: HTTP请求对象
            call_next: 下一个处理函数（中间件或路由）
        返回：
            HTTP响应对象（添加了X-Process-Time响应头）
        """
        # 记录请求开始时间（精确到秒）
        start_time = time.time()

        # 获取客户端IP
        client_host = request.client.host if request.client else "unknown"

        # 记录请求信息
        # 作用：知道谁在什么时间请求了什么接口
        logger.info(
            f"请求开始 | {request.method} {request.url.path} | "
            f"客户端: {client_host}"
        )

        # 调用下一个处理函数（执行实际的请求处理）
        # await说明：中间件是异步的，不阻塞其他请求
        response = await call_next(request)

        # 计算处理耗时
        process_time = time.time() - start_time

        # 记录响应信息
        # 作用：知道这个请求处理了多久，状态码是否正常
        logger.info(
            f"请求完成 | {request.method} {request.url.path} | "
            f"状态码: {response.status_code} | "
            f"耗时: {process_time:.3f}s"
        )

        # 添加响应头，显示处理时间
        # 原理：响应头中的X-前缀是自定义头的惯例
        # 前端或调试工具可以看到这个请求花了多久
        response.headers["X-Process-Time"] = str(process_time)

        return response
