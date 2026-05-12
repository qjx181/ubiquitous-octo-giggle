import logging
from ipaddress import ip_address, ip_network

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from memory import get_session_role
from services.retrieval import build_context, get_query_collection_name, get_query_jsonl_path

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/debug", tags=["debug"])
PRIVATE_NETWORKS = [
    ip_network("127.0.0.0/8"),
    ip_network("10.0.0.0/8"),
    ip_network("172.16.0.0/12"),
    ip_network("192.168.0.0/16"),
]


class RagDebugResponse(BaseModel):
    user_id: str
    session_id: str
    role: str
    collection_name: str
    jsonl_path: str
    bm25_item_count: int
    bm25_hit_count: int
    vector_hit_count: int
    merged_hit_count: int
    has_rag_context: bool
    rag_status: str
    retrieval_seconds: float
    message: str


def _ensure_intranet(request: Request):
    client_host = request.client.host if request.client else ""
    try:
        ip = ip_address(client_host)
        if not any(ip in net for net in PRIVATE_NETWORKS):
            raise HTTPException(status_code=403, detail="仅允许内网访问")
    except ValueError:
        raise HTTPException(status_code=403, detail="仅允许内网访问")


@router.get("/rag", response_model=RagDebugResponse)
def debug_rag(user_id: str, session_id: str, query: str, role: str | None = None, request: Request = None):
    if request is None:
        raise HTTPException(status_code=400, detail="缺少请求上下文")
    _ensure_intranet(request)
    user_id = user_id.strip()
    session_id = session_id.strip()
    query = query.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id不能为空")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id不能为空")
    if not query:
        raise HTTPException(status_code=400, detail="query不能为空")

    session_role = get_session_role(user_id, session_id)
    role_to_use = (role or session_role).strip()
    collection_name = get_query_collection_name(role_to_use)
    jsonl_path = get_query_jsonl_path(role_to_use)
    rag_context, refs, retrieval_seconds, bm25_item_count, bm25_hit_count, vector_hit_count, best_score = build_context(query, role_to_use)
    return RagDebugResponse(
        user_id=user_id,
        session_id=session_id,
        role=role_to_use,
        collection_name=collection_name,
        jsonl_path=jsonl_path,
        bm25_item_count=bm25_item_count,
        bm25_hit_count=bm25_hit_count,
        vector_hit_count=vector_hit_count,
        merged_hit_count=len(refs),
        has_rag_context=bool(rag_context),
        rag_status="已命中RAG" if rag_context else "未命中RAG",
        retrieval_seconds=round(retrieval_seconds, 3),
        message="调试信息获取成功",
    )
