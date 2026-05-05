import logging

from fastapi import APIRouter, HTTPException, Request, Response

from auth import (
    RegisterRequest,
    LoginRequest,
    AuthResponse,
    MeResponse,
    create_user,
    get_user,
    verify_password,
    create_access_token,
    decode_access_token,
    get_token_from_request,
    update_last_login,
)
from memory import list_sessions

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse)
def register(req: RegisterRequest):
    create_user(req.username, req.password)
    token = create_access_token(req.username.strip())
    return AuthResponse(username=req.username.strip(), token=token, message="注册成功")


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest, response: Response):
    user = get_user(req.username)
    if not user:
        raise HTTPException(status_code=400, detail="用户名或密码错误")
    if not verify_password(req.password, user["password_hash"], user["salt"]):
        raise HTTPException(status_code=400, detail="用户名或密码错误")
    token = create_access_token(user["username"])
    response.set_cookie("access_token", token, httponly=True, samesite="lax")
    update_last_login(user["username"])
    return AuthResponse(username=user["username"], token=token, message="登录成功")


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie("access_token")
    return {"message": "已退出登录"}


@router.get("/me", response_model=MeResponse)
def auth_me(request: Request):
    token = get_token_from_request(request)
    if not token:
        raise HTTPException(status_code=401, detail="未登录")
    username = decode_access_token(token)
    return MeResponse(username=username, message="已登录")


@router.get("/sessions")
def auth_sessions(request: Request):
    token = get_token_from_request(request)
    if not token:
        raise HTTPException(status_code=401, detail="未登录")
    username = decode_access_token(token)
    return {"sessions": list_sessions(username), "message": "获取成功"}
