import logging

from memory import create_session as memory_create_session, set_session_role, get_session_role, list_sessions

logger = logging.getLogger(__name__)

ALLOWED_ROLES = {"律师", "医生"}
ALLOWED_STYLES = {"严谨专业", "温和耐心", "简洁干练", "亲切易懂", "默认"}


def init_new_session(user_id: str, session_id: str, role: str, style: str = "默认"):
    if not user_id or not session_id or not role:
        return {"user_id": user_id, "session_id": session_id, "role": role, "error": "参数不能为空"}
    if role not in ALLOWED_ROLES:
        return {"user_id": user_id, "session_id": session_id, "role": role, "error": "无效的角色类型"}
    if style not in ALLOWED_STYLES:
        style = "默认"
    try:
        memory_create_session(user_id, session_id, role, style)
        return {"user_id": user_id, "session_id": session_id, "role": role, "error": None}
    except Exception:
        logger.exception("创建会话失败 user_id=%s session_id=%s role=%s", user_id, session_id, role)
        return {"user_id": user_id, "session_id": session_id, "role": role, "error": "创建会话失败"}


def change_session_role(user_id: str, session_id: str, role: str):
    if not user_id or not session_id or not role:
        return {"user_id": user_id, "session_id": session_id, "role": role, "error": "参数不能为空"}
    if role not in ALLOWED_ROLES:
        return {"user_id": user_id, "session_id": session_id, "role": role, "error": "无效的角色类型"}
    try:
        set_session_role(user_id, session_id, role)
        return {"user_id": user_id, "session_id": session_id, "role": role, "error": None}
    except Exception:
        logger.exception("切换角色失败 user_id=%s session_id=%s role=%s", user_id, session_id, role)
        return {"user_id": user_id, "session_id": session_id, "role": role, "error": "切换角色失败"}
