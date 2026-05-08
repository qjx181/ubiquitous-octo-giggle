# -*- coding: utf-8 -*-
"""
Redis缓存模块
功能：会话管理、Query缓存、结果缓存

模块职责：
1. RedisCache - Redis基础操作封装
2. SessionManager - 会话管理器，管理用户会话状态
3. QueryCache - Query结果缓存，避免重复计算
4. CacheManager - 统一的缓存管理接口

缓存策略：
- L1缓存：内存（应用层，本模块暂未实现）
- L2缓存：Redis（分布式）

为什么需要缓存？
1. 减少重复计算 - 相同的Query直接返回缓存结果
2. 提升响应速度 - 缓存命中时响应更快
3. 减轻后端压力 - 减少对LLM和数据库的调用
"""

# 导入JSON处理
import json
# 导入哈希算法，用于生成缓存key
import hashlib
# 导入类型注解
from typing import Optional, Dict, Any, List
# 导入时间处理
from datetime import datetime, timedelta
# 导入Redis客户端
import redis
# 导入Redis异常
from redis.exceptions import RedisError


class RedisCache:
    """
    Redis缓存管理器
    
    封装Redis的基础操作，提供：
    - get/set/delete等基础方法
    - 自动JSON序列化/反序列化
    - 连接管理
    - 异常处理
    """
    
    def __init__(
        self,
        host: str = "localhost",      # Redis主机地址
        port: int = 6379,              # Redis端口
        db: int = 0,                  # 数据库编号
        password: Optional[str] = None,  # 密码
        default_ttl: int = 3600      # 默认过期时间：1小时
    ):
        """
        初始化Redis连接
        
        Args:
            host: Redis服务器地址
            port: Redis服务器端口
            db: Redis数据库编号
            password: Redis密码（可选）
            default_ttl: 默认过期时间（秒）
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.default_ttl = default_ttl
        
        # Redis客户端（延迟初始化）
        self._client: Optional[redis.Redis] = None
        # 建立连接
        self._connect()
        
    def _connect(self):
        """
        建立Redis连接
        
        包含连接测试，确保连接可用
        """
        try:
            # 创建Redis客户端
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,    # 自动将bytes转为str
                socket_connect_timeout=5,  # 连接超时5秒
                socket_timeout=5,         # 操作超时5秒
                health_check_interval=30  # 健康检查间隔30秒
            )
            # 测试连接
            self._client.ping()
        except RedisError as e:
            # 连接失败时打印警告，但不抛出异常
            print(f"Warning: Redis connection failed: {e}")
            self._client = None
            
    def is_connected(self) -> bool:
        """
        检查Redis连接状态
        
        Returns:
            是否已连接
        """
        if self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except RedisError:
            return False
            
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值（反序列化后）或None（不存在或已过期）
        """
        if not self.is_connected():
            return None
            
        try:
            value = self._client.get(key)
            if value:
                # JSON反序列化
                return json.loads(value)
            return None
        except (RedisError, json.JSONDecodeError) as e:
            print(f"Redis get error: {e}")
            return None
            
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        nx: bool = False  # NX: Only set if Not eXists
    ) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值（会被JSON序列化）
            ttl: 过期时间（秒），None使用默认值
            nx: 是否仅在key不存在时设置
            
        Returns:
            是否设置成功
        """
        if not self.is_connected():
            return False
            
        try:
            # 使用默认TTL或指定的TTL
            ttl = ttl or self.default_ttl
            # JSON序列化
            serialized = json.dumps(value, ensure_ascii=False)
            
            # 设置值并指定过期时间
            return self._client.setex(key, ttl, serialized) is not None
        except (RedisError, TypeError) as e:
            print(f"Redis set error: {e}")
            return False
            
    def delete(self, key: str) -> bool:
        """
        删除缓存
        
        Args:
            key: 缓存键
            
        Returns:
            是否删除成功
        """
        if not self.is_connected():
            return False
            
        try:
            # 删除返回被删除的key数量
            return self._client.delete(key) > 0
        except RedisError as e:
            print(f"Redis delete error: {e}")
            return False
            
    def exists(self, key: str) -> bool:
        """
        检查key是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            是否存在
        """
        if not self.is_connected():
            return False
        try:
            return self._client.exists(key) > 0
        except RedisError:
            return False
            
    def ttl(self, key: str) -> int:
        """
        获取key的剩余生存时间
        
        Args:
            key: 缓存键
            
        Returns:
            剩余秒数，-2表示key不存在，-1表示key无过期时间
        """
        if not self.is_connected():
            return -2
        try:
            return self._client.ttl(key)
        except RedisError:
            return -2


class SessionManager:
    """
    会话管理器
    
    管理用户会话状态，包括：
    - 会话创建和销毁
    - 对话历史记录
    - 用户偏好设置
    - 会话过期管理
    
    应用场景：
    - 多轮对话：保存上下文，实现连续对话
    - 用户状态：保存用户的查询偏好
    - 临时数据：保存中间计算结果
    """
    
    def __init__(self, redis_cache: RedisCache, session_ttl: int = 86400):
        """
        初始化会话管理器
        
        Args:
            redis_cache: Redis缓存实例
            session_ttl: 会话过期时间（秒），默认24小时
        """
        self.redis = redis_cache
        self.session_ttl = session_ttl
        
    def _get_session_key(self, session_id: str) -> str:
        """
        生成会话在Redis中的key
        
        格式：session:{session_id}
        
        Args:
            session_id: 会话ID
            
        Returns:
            Redis key
        """
        return f"session:{session_id}"
        
    def create_session(self, session_id: str, user_id: Optional[str] = None) -> Dict:
        """
        创建新会话
        
        Args:
            session_id: 会话ID
            user_id: 用户ID（可选）
            
        Returns:
            会话数据字典
        """
        # 构建会话数据
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),  # ISO格式时间
            "last_activity": datetime.now().isoformat(),
            "conversation_history": [],  # 对话历史
            "preferences": {},            # 用户偏好
            "metadata": {}               # 额外元数据
        }
        
        # 保存到Redis
        key = self._get_session_key(session_id)
        self.redis.set(key, session_data, ttl=self.session_ttl)
        
        return session_data
        
    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        获取会话数据
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话数据或None
        """
        key = self._get_session_key(session_id)
        return self.redis.get(key)
        
    def update_session(self, session_id: str, updates: Dict) -> bool:
        """
        更新会话数据
        
        Args:
            session_id: 会话ID
            updates: 要更新的字段字典
            
        Returns:
            是否更新成功
        """
        # 获取现有会话
        session = self.get_session(session_id)
        if not session:
            return False
            
        # 合并更新
        session.update(updates)
        # 更新最后活跃时间
        session["last_activity"] = datetime.now().isoformat()
        
        # 保存
        key = self._get_session_key(session_id)
        return self.redis.set(key, session, ttl=self.session_ttl)
        
    def add_to_history(self, session_id: str, role: str, content: str) -> bool:
        """
        添加对话历史
        
        Args:
            session_id: 会话ID
            role: 角色（user/assistant）
            content: 对话内容
            
        Returns:
            是否添加成功
        """
        # 第1步：获取会话数据，若不存在则自动创建新会话
        session = self.get_session(session_id)
        if not session:
            session = self.create_session(session_id)

        # 第2步：取出已有的对话历史列表（初始为空列表）
        history = session.get("conversation_history", [])

        # 第3步：将新消息追加到历史末尾（包含角色、内容、时间戳）
        history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        # 第4步：截断逻辑 —— 保留最近20轮对话（每轮user+assistant共2条，故上限40条）
        if len(history) > 40:
            history = history[-40:]
            
        return self.update_session(session_id, {"conversation_history": history})
        
    def get_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """
        获取对话历史
        
        Args:
            session_id: 会话ID
            limit: 返回最近多少轮（每轮包含user和assistant）
            
        Returns:
            对话历史列表
        """
        session = self.get_session(session_id)
        if not session:
            return []

        history = session.get("conversation_history", [])
        # limit * 2 原因：每轮对话包含 user 和 assistant 两条记录
        # 用负索引切片取末尾 limit*2 条，等价于最近 limit 轮完整对话
        return history[-limit*2:]
        
    def delete_session(self, session_id: str) -> bool:
        """
        删除会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否删除成功
        """
        key = self._get_session_key(session_id)
        return self.redis.delete(key)
        
    def clear_history(self, session_id: str) -> bool:
        """
        清空对话历史
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否清空成功
        """
        return self.update_session(session_id, {"conversation_history": []})


class QueryCache:
    """
    Query结果缓存
    
    缓存Query的检索和生成结果，
    避免对相同Query重复进行：
    - 向量检索
    - BM25检索
    - LLM生成
    
    缓存策略：
    - 基于Query内容生成hash作为key
    - 可选的上下文隔离
    - TTL过期机制
    """
    
    def __init__(self, redis_cache: RedisCache, default_ttl: int = 1800):
        """
        初始化Query缓存
        
        Args:
            redis_cache: Redis缓存实例
            default_ttl: 默认过期时间（秒），默认30分钟
        """
        self.redis = redis_cache
        self.default_ttl = default_ttl
        
    def _generate_cache_key(self, query: str, context: Optional[str] = None) -> str:
        """
        生成缓存key

        使用Query的MD5哈希作为key，
        确保相同Query得到相同的key

        Args:
            query: 查询字符串
            context: 上下文标识（如会话ID）

        Returns:
            缓存key
        """
        # 标准化：去除首尾空白并统一为小写，消除因空格/大小写导致的key不一致
        normalized = query.strip().lower()
        # 若传入了上下文标识（如会话ID），将其拼入待哈希字符串以隔离不同上下文的缓存
        if context:
            normalized = f"{context}:{normalized}"

        # 计算MD5哈希：将标准化后的字符串编码为utf-8字节流，再取哈希值
        hash_obj = hashlib.md5(normalized.encode('utf-8'))
        # 返回带 "query:" 前缀的十六进制摘要作为Redis key
        return f"query:{hash_obj.hexdigest()}"
        
    def get_cached_result(
        self, 
        query: str, 
        context: Optional[str] = None
    ) -> Optional[Dict]:
        """
        获取缓存的Query结果

        Args:
            query: 查询字符串
            context: 上下文标识

        Returns:
            缓存的结果或None
        """
        # 通过相同哈希算法生成key，保证相同query+context命中同一缓存条目
        key = self._generate_cache_key(query, context)
        # 委托RedisCache.get从Redis读取并自动反序列化JSON
        # 若key不存在或已过期，返回None
        return self.redis.get(key)
        
    def cache_result(
        self, 
        query: str, 
        result: Dict, 
        context: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        缓存Query结果

        Args:
            query: 查询字符串
            result: 结果数据
            context: 上下文标识
            ttl: 过期时间（秒）

        Returns:
            是否缓存成功
        """
        # 生成与get_cached_result一致的缓存key
        key = self._generate_cache_key(query, context)

        # 包装缓存数据，添加元信息
        cache_data = {
            "query": query,           # 原始Query（用于调试/溯源）
            "result": result,         # 查询结果（向量检索/BM25/LLM生成结果）
            "cached_at": datetime.now().isoformat(),  # 缓存写入时间戳
            # 预计算过期时间，方便外部逻辑直接判断而无需查询TTL
            "expires_at": (datetime.now() + timedelta(seconds=ttl or self.default_ttl)).isoformat()
        }

        # 写入Redis：自动JSON序列化并设置TTL过期（过期后Redis自动删除）
        # self.redis.set 内部调用 Redis SETEX 实现原子写入+过期
        return self.redis.set(key, cache_data, ttl=ttl)
        
    def invalidate_cache(self, query: str, context: Optional[str] = None) -> bool:
        """
        使缓存失效
        
        Args:
            query: 查询字符串
            context: 上下文标识
            
        Returns:
            是否删除成功
        """
        key = self._generate_cache_key(query, context)
        return self.redis.delete(key)
        
    def get_cache_stats(self) -> Dict:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "connected": self.redis.is_connected(),
            "default_ttl": self.default_ttl
        }


class CacheManager:
    """
    缓存管理器
    
    统一管理所有缓存，提供：
    - Redis连接管理
    - 会话管理
    - Query缓存
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None
    ):
        """
        初始化缓存管理器
        
        Args:
            redis_host: Redis主机
            redis_port: Redis端口
            redis_db: Redis数据库
            redis_password: Redis密码
        """
        # 创建Redis缓存实例
        self.redis = RedisCache(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password
        )
        
        # 创建会话管理器
        self.session = SessionManager(self.redis)
        # 创建Query缓存
        self.query = QueryCache(self.redis)
        
    def health_check(self) -> Dict:
        """
        健康检查
        
        Returns:
            健康状态字典
        """
        return {
            "redis_connected": self.redis.is_connected(),
            "session_manager": "active",
            "query_cache": "active"
        }


# 全局缓存管理器实例（单例）
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """
    获取缓存管理器实例（单例）
    
    Returns:
        CacheManager实例
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


# RedisCache / SessionManager / QueryCache 独立单例获取函数
# 作用：让其他模块可以直接获取 RedisCache 或 SessionManager，无需经过 CacheManager
_redis_cache_instance = None
_session_manager_instance = None
_query_cache_instance = None


def get_redis_cache() -> RedisCache:
    """
    作用：获取 RedisCache 单例
    返回：RedisCache 实例
    """
    global _redis_cache_instance
    if _redis_cache_instance is None:
        _redis_cache_instance = RedisCache()
    return _redis_cache_instance


def get_session_manager() -> SessionManager:
    """
    作用：获取 SessionManager 单例
    原理：基于 RedisCache 实例构建
    返回：SessionManager 实例
    """
    global _session_manager_instance
    if _session_manager_instance is None:
        _session_manager_instance = SessionManager(get_redis_cache())
    return _session_manager_instance


def get_query_cache() -> QueryCache:
    """
    作用：获取 QueryCache 单例
    原理：基于 RedisCache 实例构建
    返回：QueryCache 实例
    """
    global _query_cache_instance
    if _query_cache_instance is None:
        _query_cache_instance = QueryCache(get_redis_cache())
    return _query_cache_instance
