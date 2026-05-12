"""
backend/app/api/v1/chat.py — 对话API
=====================================
作用：对外提供问答接口，是用户与RAG系统交互的入口
原理：
  1. 用户发送问题 → FastAPI接收请求 → 调用RetrievalService执行RAG流程
  2. 支持普通问答（一次性返回）和流式问答（逐字返回，打字机效果）
  3. 使用Pydantic模型做请求/响应的格式校验
  4. 业务异常通过HTTPException返回，FastAPI自动转成标准错误响应
"""

from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...config import settings
from ...core.retrieval import get_retrieval_service
from ...exceptions import BusinessException

# 创建路由实例
# 作用：定义一个API路由组，所有接口都挂在 /api/v1/chat/ 下
# prefix="/chat" 会和 main.py 里的 prefix="/api/v1" 拼接成 /api/v1/chat/
router = APIRouter(prefix="/chat", tags=["对话"])


# =============================================================================
# 请求模型
# =============================================================================

class ChatRequest(BaseModel):
    """
    作用：定义API请求的数据格式和校验规则
    原理：
      - 继承Pydantic的BaseModel，自动做类型校验
      - 如果客户端传了不符合类型的数据，FastAPI自动返回422错误
      - json_schema_extra提供示例数据，Swagger文档里会显示
    字段：
        question: 用户问题（必填，str类型）
        top_k: 检索结果数量（可选，默认5）
        content_type: 内容类型过滤（可选，"text"或"table"）
        history: 对话历史（可选，用于多轮对话）
    """
    question: str                    # 用户问题（必填，不能为空）
    top_k: int = settings.RETRIEVAL_TOP_K                   # 检索结果数量（从config读取）
    content_type: Optional[str] = None  # 内容类型过滤（可选）
    history: Optional[List[Dict[str, str]]] = None  # 对话历史列表 [{"role":"user","content":"..."}]

    class Config:
        # json_schema_extra作用：在Swagger文档中显示示例请求
        json_schema_extra = {
            "example": {
                "question": "公司的营业收入是多少？",
                "top_k": 5,
                "content_type": None,
                "history": None
            }
        }


# =============================================================================
# 响应模型
# =============================================================================

class Source(BaseModel):
    """
    作用：定义引用来源的数据格式
    字段：
        text: 引用原文片段
        page: 页码
        section: 章节名
        content_type: 内容类型
        score: 相似度分数
    """
    text: str           # 引用文本片段（截取前200字）
    page: int           # 起始页码
    section: str        # 所属章节
    content_type: str   # 内容类型（text/table）
    score: float        # 相似度分数（0~1，越高越相关）


class ChatResponse(BaseModel):
    """
    作用：定义API响应的数据格式
    字段：
        answer: LLM生成的答案文本
        sources: 引用来源列表（前端可以展示"依据第XX页"）
    """
    answer: str         # 生成的答案
    sources: List[Source]  # 引用来源列表


# =============================================================================
# API端点
# =============================================================================

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    作用：问答接口（/api/v1/chat/  POST请求）
    原理：
      执行完整的RAG流程：
        1. 调用get_retrieval_service()获取检索服务单例
        2. service.query()执行五步流程：编码→检索→建上下文→生成→组装
        3. 返回答案+引用来源
    参数：
        request: ChatRequest（FastAPI自动将请求体JSON解析为ChatRequest对象）
    返回：
        ChatResponse，包含answer和sources
    异常：
        BusinessException → HTTPException（统一错误响应格式）
        未知异常 → 500 Internal Server Error
    示例请求：
        POST /api/v1/chat/
        {"question": "公司的营业收入是多少？", "top_k": 5}
    示例响应：
        {
            "answer": "根据招股说明书，公司2023年营业收入为1000万元...",
            "sources": [{"text": "2023年营业收入...", "page": 15, ...}]
        }
    """
    try:
        # 获取检索服务单例（第一次调用会初始化所有组件）
        # 原理：get_retrieval_service() → 创建RetrievalService
        #   → 创建EmbeddingService（加载BGE-M3模型）
        #   → 创建MilvusClient（连接Milvus加载集合）
        #   → 创建LLMRouter（检测SGLang/Ollama）
        service = get_retrieval_service()

        # 执行查询
        # 原理：五步RAG流程
        #   问题→编码→向量检索→LLM智能生成→返回
        result = service.query(
            question=request.question,
            top_k=request.top_k,
            content_type=request.content_type,
            history=request.history
        )

        # 组装响应
        # 原理：把字典结果转换成Pydantic模型
        # FastAPI会自动将Pydantic模型序列化为JSON响应
        return ChatResponse(
            answer=result["answer"],
            sources=[
                Source(**source)   # 把字典解包成Source对象
                for source in result["sources"]
            ]
        )

    except BusinessException as e:
        # 业务异常（如"没找到相关信息"）
        # 直接抛出，FastAPI会自动捕获并转为结构化HTTP响应
        raise HTTPException(
            status_code=e.status_code,
            detail=e.detail
        )
    except Exception as e:
        # 未知异常（兜底，防止未捕获异常导致服务崩溃）
        raise HTTPException(
            status_code=500,
            detail=f"服务器内部错误: {str(e)}"
        )


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    作用：流式问答接口（/api/v1/chat/stream  POST请求）
    原理：
      和chat()一样走RAG流程，但最后一步调用query_stream
      用SSE（Server-Sent Events，服务器推送事件）格式逐步返回
      前端可以实现"打字机逐字输出"效果
    参数：
        request: ChatRequest
    返回：
        SSE流，每行一个token
    示例请求：
        POST /api/v1/chat/stream
        {"question": "公司的营业收入是多少？"}
    示例响应（SSE流）：
        data: 根据
        data: 招股
        data: 说明书
        data: ...
        data: [DONE]
    """
    from fastapi.responses import StreamingResponse

    try:
        service = get_retrieval_service()

        # 流式生成器
        # 作用：逐步从service.query_stream获取token
        #       每次yield一个SSE格式的data行
        async def generate():
            # 调用service的流式查询
            for token in service.query_stream(
                question=request.question,
                top_k=request.top_k,
                content_type=request.content_type,
                history=request.history
            ):
                # SSE格式：每行以 "data: " 开头
                yield f"data: {token}\n\n"
            # 结束标志
            yield "data: [DONE]\n\n"

        # StreamingResponse：将生成器包装成HTTP流式响应
        # media_type="text/event-stream" 是SSE的标准MIME类型
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"流式生成失败: {str(e)}"
        )
