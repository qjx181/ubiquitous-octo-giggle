"""
rag_mcp_server.py — 项目二 RAG 系统的 MCP Server 包装

用法（Stdio 模式，课堂演示首选）:
    python rag_mcp_server.py

用法（HTTP 模式，远程访问）:
    python rag_mcp_server.py --http

连接到 Hermes Agent（在 config.yaml 中添加）:
    mcp_servers:
      rag:
        command: "D:\\an\\envs\\project2\\python.exe"
        args: ["C:\\path\\to\\rag_mcp_server.py"]
        timeout: 180

依赖:
    pip install mcp

原理:
    将项目二的 RetrievalService（Query理解→Embedding→检索→Reranker→LLM生成）
    包装为 MCP Tool。任何支持 MCP 的 AI 客户端可以通过标准协议调用 RAG 能力，
    不需要手动对接 REST API。
"""

import asyncio
import json
import sys
import os
import logging
import threading
from contextlib import contextmanager
from typing import Any

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("rag-mcp")


@contextmanager
def _QuietStdout():
    """临时将 sys.stdout 重定向到 sys.stderr，保护 MCP 管道"""
    original_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = original_stdout


def load_rag_service():
    """
    加载项目二的 RAG 服务。

    支持两种运行模式：
    1. 在项目二目录下直接运行（Windows conda 环境）
    2. 指定路径加载

    自动探测项目二的后端路径，支持 /mnt/c/（WSL）和 C:\\（Windows）两种路径格式。
    """
    # 探测项目二 backend 目录
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), "..", "..", "backend"),
        # 常见部署路径
        r"C:\Users\qjx\Desktop\github\项目二 ---- 工单2\backend",
        "/mnt/c/Users/qjx/Desktop/github/项目二 ---- 工单2/backend",
    ]

    backend_path = None
    for p in candidate_paths:
        abs_p = os.path.abspath(p)
        if os.path.isdir(abs_p):
            backend_path = abs_p
            break

    if not backend_path:
        raise RuntimeError(
            "找不到项目二 backend 目录。请指定环境变量 RAG_BACKEND_PATH"
        )

    sys.path.insert(0, backend_path)
    logger.info(f"📂 加载项目二后端: {backend_path}")

    # 动态导入（项目二不在当前 Python 路径中）
    from app.core.retrieval import get_retrieval_service
    from app.config import settings

    service = get_retrieval_service()

    # 输出启动状态
    enabled = [n for n, ok in service.features.items() if ok]
    logger.info(f"🔧 增强功能: {', '.join(enabled) if enabled else '无'}")
    logger.info(f"🔗 Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
    logger.info(f"🤖 LLM Router: SGLang={service.llm_router.sglang_available}, "
                f"DeepSeek={service.llm_router.deepseek_available}, "
                f"Ollama={service.llm_router.ollama_available}")

    return service


def format_sources(sources: list) -> str:
    """格式化引用来源为可读文本"""
    parts = []
    for i, s in enumerate(sources, 1):
        parts.append(
            f"[{i}] {s.get('text', '')[:200]}..."
            f"（章节: {s.get('section', '无')}, "
            f"页码: {s.get('page', '无')}, "
            f"相关度: {s.get('score', 0):.3f}）"
        )
    return "\n\n".join(parts)


async def handle_rag_query(service, arguments: dict) -> dict:
    """rag_query 工具的执行逻辑"""
    question = arguments["question"]
    top_k = arguments.get("top_k", 3)
    kb_name = arguments.get("kb_name")

    logger.info(f"[rag_query] question={question[:60]}... top_k={top_k} kb_name={kb_name}")

    result = await service.query(
        question=question,
        top_k=top_k,
        kb_name=kb_name,
    )

    answer = result.get("answer", "")
    sources = result.get("sources", [])

    # 拼接完整输出：答案 + 来源
    output = answer
    if sources:
        output += "\n\n---\n📎 参考来源:\n" + format_sources(sources)

    return {
        "content": [{"type": "text", "text": output}],
        "metadata": {
            "source_count": len(sources),
            "top_k": top_k,
        },
    }


async def handle_rag_search(service, arguments: dict) -> dict:
    """rag_search 工具的执行逻辑（仅检索不生成）"""
    query = arguments["query"]
    top_k = arguments.get("top_k", 5)
    kb_name = arguments.get("kb_name")

    logger.info(f"[rag_search] query={query[:60]}... top_k={top_k}")

    # 走检索管道的生成前阶段
    from app.core.pipeline.interfaces import PipelineContext

    ctx = PipelineContext(
        question=query,
        top_k=top_k,
        kb_name=kb_name,
    )

    if service._pipeline:
        # 只运行到 reranker 阶段，跳过 LLM 生成
        ctx = await service._pipeline.run(ctx)
        results = ctx.reranked_results or ctx.vector_results or []
    else:
        # 旧版管道
        from app.core.embedding_service import get_embedding_service
        emb = get_embedding_service()
        query_vec = emb.encode(query)
        results = service.milvus_client.search(
            embedding=query_vec, top_k=top_k
        )

    output = []
    for i, r in enumerate(results, 1):
        output.append(
            f"[{i}] {r.get('text', '')[:300]}...\n"
            f"    📄 {r.get('section', '无')} 第{r.get('page', '无')}页 "
            f"(score: {r.get('score', 0):.3f})"
        )

    return {
        "content": [{"type": "text", "text": "\n\n".join(output) if output else "未检索到相关内容"}],
        "metadata": {"result_count": len(results), "top_k": top_k},
    }


# =============================================================================
# 主入口
# =============================================================================

def main_stdio():
    """Stdio 模式：通过 stdin/stdout 与父进程通信"""
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent

    logger.info("🚀 启动 MCP Server (Stdio 模式)")

    # 先创建 server 实例，再在后台加载模型
    # 避免 BGE-M3 加载（~30秒）阻塞 MCP 握手导致超时
    server = Server("rag-prospectus")
    service = None
    service_ready = threading.Event()

    def _init_service():
        """在后台线程中初始化服务（不阻塞 MCP 握手）"""
        nonlocal service
        with _QuietStdout():
            service = load_rag_service()
        service_ready.set()
        logger.info("✅ RAG 服务初始化完成")

    threading.Thread(target=_init_service, daemon=True).start()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="rag_query",
                description="基于招股说明书知识库回答用户问题。"
                            "支持多公司知识库过滤（通过 kb_name 参数）。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "用户的问题，如'公司去年的营收是多少'",
                        },
                        "top_k": {
                            "type": "number",
                            "description": "检索参考的文档片段数量",
                            "default": 3,
                        },
                        "kb_name": {
                            "type": "string",
                            "description": "知识库名称过滤（可选），如'招股说明书1'",
                        },
                    },
                    "required": ["question"],
                },
            ),
            Tool(
                name="rag_search",
                description="仅检索文档片段，不生成答案。"
                            "返回匹配问题的原始文档段落及来源信息。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索查询语句",
                        },
                        "top_k": {
                            "type": "number",
                            "description": "返回的文档片段数量",
                            "default": 5,
                        },
                        "kb_name": {
                            "type": "string",
                            "description": "知识库名称过滤（可选）",
                        },
                    },
                    "required": ["query"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        # 等待后台服务初始化完成（不阻塞事件循环）
        if not service_ready.is_set():
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, service_ready.wait)

        if name == "rag_query":
            result = await handle_rag_query(service, arguments)
        elif name == "rag_search":
            result = await handle_rag_search(service, arguments)
        else:
            raise ValueError(f"未知工具: {name}")

        texts = []
        for c in result["content"]:
            texts.append(TextContent(type=c["type"], text=c["text"]))
        return texts

    async def _run():
        async with stdio_server() as (read, write):
            await server.run(read, write,
                             server.create_initialization_options())

    asyncio.run(_run())


def main_http():
    """HTTP 模式：以 Web 服务形式运行，支持远程客户端"""
    from mcp.server import Server
    from mcp.server.http import http_server
    from mcp.types import Tool, TextContent

    logger.info("🚀 启动 MCP Server (HTTP 模式)")

    # HTTP 模式需要先初始化服务再启动 Web 服务器
    # 由于 mcp 库的 HTTP 支持在不同版本中 API 不同，
    # 建议参考官方文档: https://github.com/modelcontextprotocol/python-sdk
    logger.warning("HTTP 模式需要 mcp>=1.5.0，请运行 pip install --upgrade mcp")
    logger.info("Stdio 模式更稳定，建议课堂演示使用 Stdio")
    raise NotImplementedError(
        "HTTP 模式待实现。请使用 Stdio 模式: python rag_mcp_server.py"
    )


if __name__ == "__main__":
    if "--http" in sys.argv:
        main_http()
    else:
        main_stdio()
