"""
backend/app/middleware/rate_limit.py — 限流中间件
=================================================
<<<<<<< HEAD
职责：限制每个IP的请求频率，防止滥用

功能：
  1. 基于IP地址进行限流
  2. 使用内存字典记录请求次数
  3. 超过限制返回429状态码

原理：
  滑动窗口算法：在固定时间窗口内统计请求次数
  超过阈值则拒绝请求，保护后端服务

注意：
  生产环境建议使用Redis实现分布式限流
  当前为单机内存实现，重启后计数器清零
=======
作用：限制每个IP的请求频率，防止恶意攻击或意外高并发导致服务崩溃
原理：
  1. 滑动窗口算法——记录每个IP的请求时间戳
  2. 每次请求时，清除窗口外的旧时间戳
  3. 如果窗口内的请求数超过阈值，返回429状态码
  4. 使用内存字典存储（生产环境建议换成Redis，支持分布式限流）

好处：
  - 保护后端服务不被突发流量冲垮
  - 防止恶意用户大量刷API
  - 滑动窗口比固定窗口更平滑（不会在边界瞬间集中爆发）
>>>>>>> 6d37b33 (提交信息)
"""

import logging
import time
from typing import Dict, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
<<<<<<< HEAD
    限流中间件

    限制每个IP在单位时间内的请求次数

=======
    作用：限流中间件
    原理：每个IP独立计数，超过阈值返回HTTP 429 Too Many Requests
>>>>>>> 6d37b33 (提交信息)
    参数：
        max_requests: 时间窗口内最大请求数（默认100）
        window_seconds: 时间窗口大小（默认60秒）
    """

    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        """
<<<<<<< HEAD
        初始化限流中间件

        参数：
            app: FastAPI应用
            max_requests: 最大请求数
            window_seconds: 时间窗口（秒）
        """
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # 存储每个IP的请求记录: {ip: [(timestamp1, count), (timestamp2, count), ...]}
=======
        作用：初始化限流中间件
        参数：
            app: FastAPI应用实例
            max_requests: 最大请求数（100次/分钟）
            window_seconds: 时间窗口（60秒）
        """
        super().__init__(app)
        self.max_requests = max_requests          # 每分钟最多100次
        self.window_seconds = window_seconds      # 时间窗口
        # 存储每个IP的请求时间戳
        # 格式：{ip: [timestamp1, timestamp2, ...]}
>>>>>>> 6d37b33 (提交信息)
        self.requests: Dict[str, list] = {}

    async def dispatch(self, request: Request, call_next):
        """
<<<<<<< HEAD
        处理请求，检查是否超过限流阈值

        参数：
            request: HTTP请求对象
            call_next: 下一个处理函数

        返回：
            HTTP响应对象
        """
        # 获取客户端IP
        client_ip = request.client.host if request.client else "unknown"

        # 检查是否超过限流
=======
        作用：处理每个请求，检查是否超过限流阈值
        逻辑：
          1. 获取客户端IP
          2. 检查该IP是否超过限流
          3. 超过则返回429，否则记录请求并继续
        参数：
            request: HTTP请求对象
            call_next: 下一个中间件或路由处理函数
        返回：
            HTTP响应对象（正常处理或429限流响应）
        """
        # 获取客户端IP
        # 注意：如果经过反向代理，request.client.host可能是代理IP
        # 生产环境应通过X-Forwarded-For头获取真实IP
        client_ip = request.client.host if request.client else "unknown"

        # 检查是否超过限流阈值
>>>>>>> 6d37b33 (提交信息)
        if self._is_rate_limited(client_ip):
            logger.warning(f"限流触发 | IP: {client_ip} | 路径: {request.url.path}")
            return JSONResponse(
                status_code=429,
                content={
                    "code": 10003,
                    "message": "请求过于频繁，请稍后再试",
                    "detail": f"每{self.window_seconds}秒最多{self.max_requests}次请求",
                }
            )

<<<<<<< HEAD
        # 记录本次请求
        self._record_request(client_ip)

        # 继续处理请求
=======
        # 记录本次请求的时间戳
        self._record_request(client_ip)

        # 继续处理请求（调用下一个中间件或路由）
>>>>>>> 6d37b33 (提交信息)
        return await call_next(request)

    def _is_rate_limited(self, client_ip: str) -> bool:
        """
<<<<<<< HEAD
        检查IP是否超过限流阈值

        参数：
            client_ip: 客户端IP地址

        返回：
            True表示超过限流，False表示未超过
        """
        current_time = time.time()
        window_start = current_time - self.window_seconds

        # 获取该IP的请求记录
        ip_requests = self.requests.get(client_ip, [])

        # 清理过期的请求记录（滑动窗口）
        ip_requests = [t for t in ip_requests if t > window_start]
        self.requests[client_ip] = ip_requests

        # 检查请求次数
=======
        作用：检查某个IP是否超过了限流阈值
        原理：滑动窗口算法
          1. 获取当前时间戳
          2. 计算窗口起始时间 = 当前时间 - 窗口大小
          3. 从该IP的请求记录中过滤出窗口内的请求
          4. 如果窗口内请求数 ≥ 阈值 → 限流
        参数：
            client_ip: 客户端IP地址
        返回：
            True=超过限流，False=未超过
        """
        current_time = time.time()  # 当前时间戳（秒）
        window_start = current_time - self.window_seconds  # 窗口起始点

        # 获取该IP的历史请求记录
        ip_requests = self.requests.get(client_ip, [])

        # 滑动窗口：只保留窗口内的时间戳
        # 原理：清除过期记录，保持列表长度等于窗口内实际请求数
        ip_requests = [t for t in ip_requests if t > window_start]
        self.requests[client_ip] = ip_requests

        # 检查请求次数是否超过阈值
>>>>>>> 6d37b33 (提交信息)
        return len(ip_requests) >= self.max_requests

    def _record_request(self, client_ip: str):
        """
<<<<<<< HEAD
        记录请求时间戳

=======
        作用：记录一次请求的时间戳
>>>>>>> 6d37b33 (提交信息)
        参数：
            client_ip: 客户端IP地址
        """
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        self.requests[client_ip].append(time.time())
