import logging
import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from auth import decode_access_token, get_token_from_request
from memory import append_message, list_sessions, create_session as memory_create_session
from role_manager import get_role_prompt, get_role_style_prompt

logger = logging.getLogger(__name__)
router = APIRouter(tags=["session"])


class SessionCreateRequest(BaseModel):
    role: str
    style: str | None = None


class SessionCreateResponse(BaseModel):
    session_id: str
    role: str
    style: str
    message: str


class HistoryListResponse(BaseModel):
    sessions: list[str]
    message: str


@router.post("/session/create", response_model=SessionCreateResponse)
def create_session(req: SessionCreateRequest, request: Request):
    token = get_token_from_request(request)
    if not token:
        raise HTTPException(status_code=401, detail="未登录")
    username = decode_access_token(token)
    role = req.role.strip()
    style = (req.style or "默认").strip() or "默认"
    if not role:
        raise HTTPException(status_code=400, detail="role不能为空")
    role_prompt = get_role_prompt(role)
    if not role_prompt:
        raise HTTPException(status_code=400, detail="角色不存在或角色提示词为空")
    session_id = uuid.uuid4().hex
    style_prompt = get_role_style_prompt(role, style)
    memory_create_session(username, session_id, role, style)
    append_message(username, session_id, "system", style_prompt)
    return SessionCreateResponse(session_id=session_id, role=role, style=style, message="会话创建成功")


@router.get("/sessions", response_model=HistoryListResponse)
def get_user_sessions(request: Request):
    token = get_token_from_request(request)
    if not token:
        raise HTTPException(status_code=401, detail="未登录")
    username = decode_access_token(token)
    sessions = list_sessions(username)
    return HistoryListResponse(sessions=sessions, message="获取成功")