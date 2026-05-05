import json
import logging
import time
from typing import List, Dict, Optional
import redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

# ==================== 配置 ====================
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
REDIS_DB = 0
SESSION_TTL_SECONDS = 60 * 30  # 30分钟过期
MAX_HISTORY_MESSAGES = 10      # 最大保留消息数
SUMMARY_TRIGGER_MESSAGES = 8   # 超过该轮数后开始压缩摘要
ALLOWED_ROLES = {"律师", "医生"}  # 支持的角色
ALLOWED_STYLES = {"严谨专业", "温和耐心", "简洁干练", "亲切易懂", "默认"}

# ==================== 存储引擎（自动降级） ====================
# Redis 客户端
redis_client: Optional[redis.Redis] = None
redis_available = False

# 内存降级存储（Redis不可用时使用）
# 结构：{ "user:session": {"messages": [], "expire": timestamp, "summary": str} }
MEMORY_HISTORY: Dict[str, Dict] = {}
# 结构：{ "user:session": {"role": role, "style": style, "expire": timestamp} }
MEMORY_ROLE: Dict[str, Dict] = {}
# 结构：{ user_id: set(session_id) }
MEMORY_SESSIONS: Dict[str, set] = {}

MESSAGE_ROLES = {"system", "user", "assistant"}

# 初始化 Redis 连接
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
        socket_timeout=2  # 超时防止卡死
    )
    redis_client.ping()
    redis_available = True
    logger.info("Redis 连接成功，使用 Redis 存储")
except RedisError:
    redis_available = False
    logger.warning("Redis 未启动，自动切换为内存存储（重启服务数据丢失）")

# ==================== 私有 Key 生成 ====================
def _history_key(user_id: str, session_id: str) -> str:
    return f"chat:history:{user_id}:{session_id}"

def _role_key(user_id: str, session_id: str) -> str:
    return f"chat:role:{user_id}:{session_id}"

def _sessions_key(user_id: str) -> str:
    return f"chat:sessions:{user_id}"

def _mem_key(user_id: str, session_id: str) -> str:
    return f"{user_id}:{session_id}"

# ==================== 内存过期清理 ====================
def _clean_expired_memory():
    """清理内存中过期的会话数据"""
    now = time.time()
    expired_keys = [k for k, v in MEMORY_HISTORY.items() if v["expire"] < now]
    for key in expired_keys:
        MEMORY_HISTORY.pop(key, None)
        MEMORY_ROLE.pop(key, None)

    for user_id, sessions in list(MEMORY_SESSIONS.items()):
        alive_sessions = set()
        for session_id in sessions:
            key = _mem_key(user_id, session_id)
            history_item = MEMORY_HISTORY.get(key)
            role_item = MEMORY_ROLE.get(key)
            if history_item or role_item:
                alive_sessions.add(session_id)
        if alive_sessions:
            MEMORY_SESSIONS[user_id] = alive_sessions
        else:
            MEMORY_SESSIONS.pop(user_id, None)

# ==================== 公开接口（完全兼容原有代码） ====================
def get_history(user_id: str, session_id: str) -> List[Dict]:
    """获取对话历史"""
    if not user_id or not session_id:
        return []

    # Redis 逻辑
    if redis_available:
        try:
            data = redis_client.get(_history_key(user_id, session_id))
            if not data:
                return []
            payload = json.loads(data)
            if isinstance(payload, dict):
                messages = payload.get("messages", [])
                return messages if isinstance(messages, list) else []
            return payload if isinstance(payload, list) else []
        except (RedisError, json.JSONDecodeError):
            return []

    # 内存逻辑
    _clean_expired_memory()
    return MEMORY_HISTORY.get(_mem_key(user_id, session_id), {}).get("messages", [])

def save_history(user_id: str, session_id: str, history: List[Dict]):
    """保存对话历史（自动截断）"""
    if not user_id or not session_id:
        return

    trimmed = history[-MAX_HISTORY_MESSAGES:]
    summary = MEMORY_HISTORY.get(_mem_key(user_id, session_id), {}).get("summary", "")

    # Redis 逻辑
    if redis_available:
        try:
            payload = {"messages": trimmed, "summary": summary}
            redis_client.set(
                _history_key(user_id, session_id),
                json.dumps(payload, ensure_ascii=False),
                ex=SESSION_TTL_SECONDS
            )
        except RedisError:
            pass
        return

    # 内存逻辑
    _clean_expired_memory()
    MEMORY_HISTORY[_mem_key(user_id, session_id)] = {
        "messages": trimmed,
        "summary": summary,
        "expire": time.time() + SESSION_TTL_SECONDS
    }

def append_message(user_id: str, session_id: str, role: str, content: str):
    """追加消息到历史"""
    if not user_id or not session_id or not content:
        return
    if role not in MESSAGE_ROLES:
        return

    history = get_history(user_id, session_id)
    history.append({"role": role, "content": content.strip()})
    save_history(user_id, session_id, history)

def clear_history(user_id: str, session_id: str):
    """清空指定会话历史"""
    if not user_id or not session_id:
        return

    if redis_available:
        try:
            redis_client.delete(_history_key(user_id, session_id))
        except RedisError:
            pass
    else:
        MEMORY_HISTORY.pop(_mem_key(user_id, session_id), None)


def get_history_summary(user_id: str, session_id: str) -> str:
    if not user_id or not session_id:
        return ""
    if redis_available:
        try:
            data = redis_client.get(_history_key(user_id, session_id))
            if not data:
                return ""
            payload = json.loads(data)
            return payload.get("summary", "") if isinstance(payload, dict) else ""
        except (RedisError, json.JSONDecodeError):
            return ""
    _clean_expired_memory()
    return MEMORY_HISTORY.get(_mem_key(user_id, session_id), {}).get("summary", "")


def set_history_summary(user_id: str, session_id: str, summary: str):
    if not user_id or not session_id:
        return
    if redis_available:
        try:
            current = get_history(user_id, session_id)
            payload = {"messages": current, "summary": summary}
            redis_client.set(
                _history_key(user_id, session_id),
                json.dumps(payload, ensure_ascii=False),
                ex=SESSION_TTL_SECONDS
            )
        except RedisError:
            pass
        return
    _clean_expired_memory()
    item = MEMORY_HISTORY.get(_mem_key(user_id, session_id), {"messages": [], "expire": time.time() + SESSION_TTL_SECONDS})
    item["summary"] = summary
    MEMORY_HISTORY[_mem_key(user_id, session_id)] = item

def set_session_role(user_id: str, session_id: str, role: str, style: str = "默认"):
    """设置会话角色，并维护会话列表"""
    if not user_id or not session_id or role not in ALLOWED_ROLES:
        return
    if style not in ALLOWED_STYLES:
        style = "默认"

    # Redis 逻辑
    if redis_available:
        try:
            redis_client.set(
                _role_key(user_id, session_id),
                role,
                ex=SESSION_TTL_SECONDS
            )
            redis_client.sadd(_sessions_key(user_id), session_id)
            redis_client.expire(_sessions_key(user_id), SESSION_TTL_SECONDS)
        except RedisError:
            pass
        return

    # 内存逻辑
    _clean_expired_memory()
    MEMORY_ROLE[_mem_key(user_id, session_id)] = {
        "role": role,
        "style": style,
        "expire": time.time() + SESSION_TTL_SECONDS
    }
    # 维护会话列表
    if user_id not in MEMORY_SESSIONS:
        MEMORY_SESSIONS[user_id] = set()
    MEMORY_SESSIONS[user_id].add(session_id)

def get_session_role(user_id: str, session_id: str) -> str:
    """获取会话绑定的角色"""
    if not user_id or not session_id:
        return "默认助手"

    # Redis 逻辑
    if redis_available:
        try:
            role = redis_client.get(_role_key(user_id, session_id))
            return role if role in ALLOWED_ROLES else "默认助手"
        except RedisError:
            return "默认助手"

    # 内存逻辑
    _clean_expired_memory()
    return MEMORY_ROLE.get(_mem_key(user_id, session_id), {}).get("role", "默认助手")

def list_sessions(user_id: str) -> List[str]:
    """获取用户的所有会话ID"""
    if not user_id:
        return []

    if redis_available:
        try:
            return list(redis_client.smembers(_sessions_key(user_id)))
        except RedisError:
            return []

    _clean_expired_memory()
    return list(MEMORY_SESSIONS.get(user_id, set()))

def create_session(user_id: str, session_id: str, role: str, style: str = "默认"):
    """创建会话：绑定角色 + 初始化空历史"""
    set_session_role(user_id, session_id, role, style)
    save_history(user_id, session_id, [])