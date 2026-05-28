"""backend/app/api/v1/chat.py — 对话API

作用：对外提供问答接口，是用户与RAG系统交互的入口
原理：
  1. 用户发送问题 → FastAPI接收请求 → 调用RetrievalService执行RAG流程
  2. 支持普通问答（一次性返回）和流式问答（逐字返回，打字机效果）
  3. 使用Pydantic模型做请求/响应的格式校验
  4. 业务异常通过HTTPException返回，FastAPI自动转成标准错误响应

两个接口：
  - POST /api/v1/chat/       → 普通问答，返回完整答案+来源
  - POST /api/v1/chat/stream → 流式问答，返回SSE事件流（打字机效果）

面试官可能问：
  Q: 为什么普通问答和流式问答要分成两个接口？
  A: 返回格式完全不同。普通接口返回完整JSON，包含answer和sources。
     流式接口返回SSE文本流，无法同时返回sources（sources在流开始前就已确定）。
     如果既要流式又要sources，可以在流开始时先发一段JSON包含sources，
     再逐token发answer内容——但会显著增加前端解析复杂度。

  Q: sources（引用来源）是怎么返回的？
  A: sources = [Source(**source) for source in result["sources"]]。
     每个Source包含text(原文)、page(页码)、section(章节)、score(分数)。
     这些在检索阶段（_run_retrieval_pipeline）就已经确定了，LLM只负责生成答案。

  Q: SSE流式输出的[DONE]标记是做什么的？
  A: 前端无法知道LLM什么时候生成完毕（因为没有Content-Length）。
     [DONE]标记让前端明确知道流已结束，可以关闭loading状态、停止接收。
     这是OpenAI SSE格式的业界惯例。

  Q: 流式接口异常时已经发出去的token怎么办？
  A: 这是SSE的固有问题——无法"撤回"已发送的数据。
     如果异常发生在前几个token内，用户体验影响较小；
     如果发生在接近结尾处，用户看到完整回答但实际未完成。
     改进方案：异常时发一条特殊SSE消息（如data: [ERROR]），前端收到后展示"回答被截断"。
"""

# 【作用】导入类型提示相关工具，让代码更清晰、IDE能给出更好的补全
# 【原理】typing 是 Python 标准库，List/Dict/Any/Optional 用于声明变量和函数参数的类型
# 【逻辑】Optional[str] 表示这个参数可以是 str 类型，也可以是 None
from typing import List, Dict, Any, Optional

# 【作用】从 FastAPI 框架中导入路由器和 HTTP 异常类
# 【原理】APIRouter 用来组织一组相关的 API 路由，HTTPException 用于主动抛出 HTTP 错误
# 【逻辑】APIRouter 可以挂载到主应用的某个路径前缀下（比如 /chat）
from fastapi import APIRouter, HTTPException
# 【作用】从 Pydantic 库导入 BaseModel，用于定义数据模型
# 【原理】Pydantic 会自动做类型校验和序列化/反序列化
# 【逻辑】我们定义的请求和响应类都继承 BaseModel，FastAPI 会自动校验请求体是否符合模型
from pydantic import BaseModel

# 【作用】导入配置模块，读取 settings 中的配置项（如检索数量 top_k）
# 【原理】settings 是一个全局配置对象，通常在 config.py 中从环境变量或配置文件读取
# 【逻辑】用相对路径从当前包的上一级（...）导入 .../config.py 中的 settings
from ...config import settings
# 【作用】导入获取检索服务实例的工厂函数
# 【原理】get_retrieval_service() 返回一个 RetrievalService 对象，封装了 RAG 的核心逻辑
# 【逻辑】每次请求都调用这个函数获得一个实例，避免全局单例带来的问题
from ...core.retrieval import get_retrieval_service
# 【作用】导入自定义的业务异常类，用于区分业务错误和系统错误
# 【原理】BusinessException 携带 status_code 和 detail，方便统一处理
# 【逻辑】捕获到 BusinessException 时，转成 FastAPI 的 HTTPException，让框架帮我们返回标准错误响应
from ...exceptions import BusinessException
from ...db.redis_client import get_session_manager

# 【作用】创建一个 FastAPI 路由器实例，所有以 /chat 开头的路由都归它管理
# 【原理】APIRouter(prefix="/chat") 给所有注册的路由自动加上 /chat 前缀
# 【逻辑】tags=["对话"] 在 Swagger 文档界面上把接口分组到"对话"标签下，方便查看
router = APIRouter(prefix="/chat", tags=["对话"])


# 【作用】定义对话请求的数据模型，用户在 POST 请求体中发送的 JSON 必须符合这个结构
# 【原理】继承 BaseModel，Pydantic 会自动校验请求体中的每个字段类型
# 【逻辑】FastAPI 收到 POST 请求时，自动把 JSON 解析成 ChatRequest 对象
class ChatRequest(BaseModel):
    """对话请求模型

    字段：
        question: 用户问题（必填）
        session_id: 会话ID（可选，用于Redis会话管理）
        top_k: 检索结果数量（默认从config读取）
        content_type: 内容类型过滤（可选）
        kb_name: 知识库名称过滤（可选）
    """
    # 【作用】用户提问的文本，必填字段，类型必须是 str 字符串
    # 【原理】Pydantic 会校验请求体中必须有 "question" 这个 key，且值必须是字符串
    # 【逻辑】如果请求体中没有 question，FastAPI 会自动返回 422 校验错误
    question: str
    # 【作用】会话ID，用于Redis会话管理
    # 【原理】可选字段，传了就用Redis管理历史，不传就不使用会话
    # 【逻辑】前端可以生成一个唯一ID（如UUID）作为session_id
    session_id: Optional[str] = None
    # 【作用】检索时返回的最相关文档数量，从配置中读取默认值
    # 【原理】int 类型，设置默认值为 settings.RETRIEVAL_TOP_K（通常在 config.py 中定义）
    # 【逻辑】用户可以不传这个字段，不传就使用系统默认值
    top_k: int = settings.RETRIEVAL_TOP_K
    # 【作用】可选的文档类型过滤条件，比如只搜索"年报"或"公告"
    # 【原理】Optional[str] 表示要么是字符串，要么是 None（不指定过滤）
    # 【逻辑】传 None 表示不过滤，检索所有类型的内容
    content_type: Optional[str] = None
    # 【作用】知识库名称过滤，用于多公司场景下限定检索范围
    # 【原理】可选字段，传入如"招股说明书1"则只检索该知识库的文档
    # 【逻辑】不传表示检索所有知识库，传入则过滤到指定kb_name
    kb_name: Optional[str] = None

    # 【作用】Pydantic V2 的配置内部类，用来添加额外的模型配置
    # 【原理】json_schema_extra 用于在生成的 OpenAPI 文档中添加示例数据
    # 【逻辑】这只是给前端/Swagger 文档看的示例，不影响实际校验逻辑
    class Config:
        json_schema_extra = {
            "example": {
                "question": "公司的营业收入是多少？",
                "session_id": "user-123-session-456",
                "top_k": 5,
                "content_type": None
            }
        }


# 【作用】定义引用来源的数据模型，表示答案是从哪些文档片段中生成的
# 【原理】继承 BaseModel，Pydantic 自动校验和序列化
# 【逻辑】每个 Source 对象对应一条检索到的文档片段
class Source(BaseModel):
    """引用来源数据模型

    字段：
        text: 引用原文片段
        page: 页码
        section: 章节名
        content_type: 内容类型
        score: 相似度分数
    """
    # 【作用】检索到的原文片段文本内容（字符串类型）
    # 【原理】必填字段，从检索结果中提取的文档原文
    # 【逻辑】这是最终展示给用户看的引文内容
    text: str
    # 【作用】该片段在源文档中的页码（整数类型）
    # 【原理】int 类型，如果是 PDF 文件就是 PDF 页码
    # 【逻辑】帮助用户快速定位到原文的具体位置
    page: int
    # 【作用】该片段所属的章节名称（字符串类型）
    # 【原理】从文档结构中提取的章节标题
    # 【逻辑】比如"第三章 财务数据"或"Appendix A"
    section: str
    # 【作用】文档的内容类型标签，比如"年报"、"公告"、"研报"
    # 【原理】区分不同来源，方便用户判断信息的可信度
    # 【逻辑】与 ChatRequest 中的 content_type 过滤条件对应
    content_type: str
    # 【作用】该片段与用户问题的语义相似度分数（浮点数类型）
    # 【原理】float 类型，分数越高表示与问题越相关
    # 【逻辑】通常范围是 0~1，用来排序检索结果的相关性
    score: float


# 【作用】定义对话响应的数据模型，返回给前端的数据结构
# 【原理】继承 BaseModel，FastAPI 会把返回值自动序列化成 JSON
# 【逻辑】返回给用户最终答案和引用来源列表
class ChatResponse(BaseModel):
    """对话响应模型

    字段：
        answer: LLM生成的答案文本
        sources: 引用来源列表
    """
    # 【作用】LLM 大模型生成的答案文本
    # 【原理】str 类型，是 RAG 流程最后一步由大语言模型产出的回答
    # 【逻辑】这个字符串会直接展示给用户看
    answer: str
    # 【作用】引用来源列表，每条记录包含原文片段、页码、章节等信息
    # 【原理】List[Source] 表示这是一个列表，列表里每个元素是 Source 类型
    # 【逻辑】用户可以看到答案是从哪些文档中来的，方便验证和溯源
    sources: List[Source]


# 【作用】用装饰器注册一个 POST 请求路由，路径为 /（完整路径是 /api/v1/chat/）
# 【原理】@router.post("/") 注册 POST 方法，response_model=ChatResponse 声明返回的数据模型
# 【逻辑】FastAPI 接收到 POST /api/v1/chat/ 请求时，自动调用下面的 chat 函数
@router.post("/", response_model=ChatResponse)
# 【作用】定义异步函数 chat，处理用户的普通问答请求（一次性返回完整答案）
# 【原理】async def 表示这是一个异步函数，内部可以用 await 等待 IO 操作
# 【逻辑】参数 request: ChatRequest 告诉 FastAPI 把请求体自动解析成 ChatRequest 对象
async def chat(request: ChatRequest):
    """问答接口（/api/v1/chat/  POST请求）

    执行完整的RAG流程（编码→检索→建上下文→生成→组装）。
    返回答案及引用来源。

    异常：
        BusinessException → HTTPException
        未知异常 → 500 Internal Server Error
    """
    # 【作用】用 try 块包住可能出错的业务逻辑，统一做异常处理
    # 【原理】try/except 是 Python 异常处理机制，捕获 try 块中抛出的任何异常
    # 【逻辑】这样写的好处是：所有异常都从 except 块统一处理，代码更干净
    try:
        # 【作用】调用工厂函数获取检索服务的实例对象
        # 【原理】get_retrieval_service() 内部初始化了向量数据库客户端、LLM 客户端等依赖
        # 【逻辑】每次请求都获取新实例，避免在并发场景下共享状态的线程安全问题
        service = get_retrieval_service()

        # 【作用】短期记忆：从Redis获取历史记录（如果传了session_id）
        # 【原理】session_id相同 → 同一个用户会话 → 从Redis读取之前的对话历史
        # 【逻辑】传了session_id就从Redis读历史，不传则为空（无上下文）
        history = []
        if request.session_id:
            sm = get_session_manager()
            redis_history = sm.get_history(request.session_id, limit=5)
            if redis_history:
                # Redis有历史 → 转成LLM需要的格式 [{role, content}, ...]
                history = [{"role": h["role"], "content": h["content"]} for h in redis_history]

        result = await service.query(
            question=request.question,
            top_k=request.top_k,
            content_type=request.content_type,
            history=history,
            kb_name=request.kb_name
        )

        # 【作用】短期记忆：把本轮问答保存到Redis
        # 【原理】add_to_history内部自动截断，保留最近20轮（40条）
        # 【逻辑】先存user问题，再存assistant答案；session不存在会自动创建
        if request.session_id:
            sm = get_session_manager()
            sm.add_to_history(request.session_id, "user", request.question)
            sm.add_to_history(request.session_id, "assistant", result["answer"])

        return ChatResponse(
            answer=result["answer"],
            sources=[Source(**source) for source in result["sources"]]
        )
    # 【作用】专门捕获业务异常，返回对应的 HTTP 状态码和错误信息
    # 【原理】except BusinessException as e 捕获自定义的 BusinessException 类型异常
    # 【逻辑】业务异常包含特定的 status_code（如 404、400），不是统一的 500 错误
    except BusinessException as e:
        # 【作用】将业务异常转换成 FastAPI 的 HTTP 异常，框架会自动处理成标准 JSON 错误响应
        # 【原理】HTTPException 是 FastAPI 内置的异常类，传入状态码和详情
        # 【逻辑】FastAPI 中间件会捕获这个异常，返回 {"detail": "xxx"} 格式的 JSON 响应
        raise HTTPException(
            status_code=e.status_code,  # 使用业务异常中定义的状态码（如 404）
            detail=e.detail             # 使用业务异常中定义的错误详情
        )
    # 【作用】捕获所有未知的系统异常，返回 500 内部服务器错误
    # 【原理】except Exception as e 捕获 Python 所有内置异常类型的基类
    # 【逻辑】兜底处理：任何意料之外的异常都被转换成 500 错误，避免泄露内部信息给用户
    except Exception as e:
        # 【作用】返回 HTTP 500 状态码和错误信息
        # 【原理】同样抛出 HTTPException，状态码硬编码为 500
        # 【逻辑】把异常信息作为字符串拼接到错误详情中，方便前端显示或调试
        raise HTTPException(
            status_code=500,                               # 固定返回 500 内部服务器错误
            detail=f"服务器内部错误: {str(e)}"                # 把异常对象的字符串形式拼接到错误信息中
        )


# 【作用】用装饰器注册一个 POST 请求路由，路径为 /stream（完整路径是 /api/v1/chat/stream）
# 【原理】@router.post("/stream") 注册 POST 方法的子路由
# 【逻辑】没有指定 response_model，因为流式返回不是一次性 JSON，而是 SSE 格式的文本流
@router.post("/stream")
# 【作用】定义异步函数 chat_stream，处理流式问答请求（逐字返回，打字机效果）
# 【原理】async def 异步函数，内部使用生成器逐步产生 token
# 【逻辑】与 chat 函数不同，这个端点返回的是 SSE 流，不是一次性 JSON
async def chat_stream(request: ChatRequest):
    """流式问答接口（/api/v1/chat/stream  POST请求）

    使用SSE（Server-Sent Events）格式逐步返回token，
    前端可实现打字机逐字输出效果。
    """
    # 【作用】延迟导入 StreamingResponse，只在需要流式响应时才加载
    # 【原理】from fastapi.responses import StreamingResponse 导入 SSE 响应类
    # 【逻辑】函数内部 import 是 Python 的延迟导入技巧，减少模块启动时的加载时间
    from fastapi.responses import StreamingResponse

    # 【作用】用 try 块包住流式生成逻辑，统一捕获异常
    try:
        # 【作用】获取检索服务实例，与普通接口相同
        service = get_retrieval_service()

        # 【作用】短期记忆：从Redis获取历史（与chat()逻辑一致）
        history = []
        if request.session_id:
            sm = get_session_manager()
            redis_history = sm.get_history(request.session_id, limit=5)
            if redis_history:
                history = [{"role": h["role"], "content": h["content"]} for h in redis_history]

        async def generate():
            pipe = await service.prepare_pipeline(
                question=request.question,
                top_k=request.top_k,
                content_type=request.content_type,
                history=history,
                kb_name=request.kb_name
            )

            # 【作用】需澄清 → 直接发澄清文本
            if pipe["clarification"]:
                yield f"data: {pipe['clarification']}\n\n"
                yield "data: [DONE]\n\n"
                return

            # 【作用】无检索结果 → 发提示文本
            if not pipe["rr"]:
                yield "data: 未找到与问题相关的信息\n\n"
                yield "data: [DONE]\n\n"
                return

            # 【作用】先发 sources（JSON 格式），前端收到后渲染引用来源
            import json
            sources_json = json.dumps({"type": "sources", "data": pipe["sources"]}, ensure_ascii=False)
            yield f"data: {sources_json}\n\n"

            # 【作用】再逐 token 流式推送 LLM 生成的答案，同时收集完整文本
            full_answer = ""
            async for token in service.llm_router.generate_stream(query=pipe["q"], context=pipe["ctx"], history=pipe["h"]):
                full_answer += token
                yield f"data: {token}\n\n"
            # 【作用】流式结束后，保存问答到Redis
            if request.session_id:
                sm = get_session_manager()
                sm.add_to_history(request.session_id, "user", request.question)
                sm.add_to_history(request.session_id, "assistant", full_answer)
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
    # 【作用】捕获流式生成过程中所有可能的异常
    # 【原理】except Exception as e 捕获所有异常类型（兜底处理）
    # 【逻辑】流式接口只做简单异常处理，统一返回 500 错误
    except Exception as e:
        # 【作用】抛出 HTTP 500 错误，包含具体的失败原因
        # 【原理】与普通接口的异常处理方式相同
        # 【逻辑】注意：流式响应中一旦发生异常，之前发给客户端的 token 可能已经无法撤回
        raise HTTPException(
            status_code=500,                              # 固定返回 500 内部服务器错误
            detail=f"流式生成失败: {str(e)}"                # 把异常信息拼接到错误详情中
        )
