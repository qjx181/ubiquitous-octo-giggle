# -*- coding: utf-8 -*-
"""
认证授权模块
功能：JWT认证、权限控制、用户管理

模块职责：
1. JWTAuth - JWT认证管理器，负责Token的生成、验证、刷新
2. PermissionChecker - 权限检查器，基于RBAC模型进行权限控制
3. UserManager - 用户管理器，负责用户注册、认证、查询

安全特性：
- 密码使用bcrypt加密
- Token有过期时间
- Token黑名单机制（可撤销）
- RBAC权限控制
"""

# 导入JWT库，用于Token生成和验证
import jwt
# 导入时间处理
from datetime import datetime, timedelta
# 导入类型注解
from typing import Optional, Dict, List
# 导入枚举
from enum import Enum
# 导入数据类
from dataclasses import dataclass
# 导入密码加密
from passlib.context import CryptContext


class UserRole(Enum):
    """
    用户角色枚举
    
    定义系统中的用户角色及其权限级别：
    - ADMIN: 管理员，拥有最高权限
    - USER: 普通用户，拥有基本权限
    - GUEST: 访客，权限受限
    """
    ADMIN = "admin"           # 管理员：所有权限
    USER = "user"             # 普通用户：聊天、查看
    GUEST = "guest"           # 访客：仅聊天


@dataclass
class User:
    """
    用户数据类
    
    存储用户的基本信息
    """
    id: str                          # 用户唯一ID
    username: str                     # 用户名
    email: str                        # 邮箱
    role: UserRole                    # 用户角色
    is_active: bool = True           # 是否激活
    created_at: Optional[str] = None # 创建时间


class JWTAuth:
    """
    JWT认证管理器
    
    负责：
    1. 密码哈希和验证
    2. Access Token生成和验证
    3. Refresh Token生成和验证
    4. Token撤销（黑名单）
    
    JWT结构：
    - Header: 包含算法信息
    - Payload: 包含用户信息和声明
    - Signature: 签名，确保Token不被篡改
    """
    
    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire: int = 30,  # Access Token有效期：30分钟
        refresh_token_expire: int = 7    # Refresh Token有效期：7天
    ):
        """
        初始化JWT认证管理器
        
        Args:
            secret_key: JWT签名密钥（非常重要，必须保密）
            algorithm: 加密算法，默认HS256
            access_token_expire: Access Token过期时间（分钟）
            refresh_token_expire: Refresh Token过期时间（天）
        """
        self.secret_key = secret_key                    # 签名密钥
        self.algorithm = algorithm                    # 加密算法
        self.access_token_expire = access_token_expire  # Access Token有效期
        self.refresh_token_expire = refresh_token_expire  # Refresh Token有效期
        
        # 创建密码加密上下文
        # bcrypt：自适应哈希算法，可抵抗彩虹表攻击和GPU加速攻击
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # Token黑名单集合，用于存储已撤销的Token
        self.token_blacklist: set = set()
        
    def hash_password(self, password: str) -> str:
        """
        哈希密码
        
        使用bcrypt算法对密码进行哈希，
        哈希过程是不可逆的，保证密码安全存储。
        
        Args:
            password: 明文密码
            
        Returns:
            哈希后的密码字符串
        """
        return self.pwd_context.hash(password)
        
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        验证密码
        
        将用户输入的明文密码与存储的哈希密码进行比对。
        
        Args:
            plain_password: 用户输入的明文密码
            hashed_password: 存储的哈希密码
            
        Returns:
            密码是否匹配
        """
        return self.pwd_context.verify(plain_password, hashed_password)
        
    def create_access_token(
        self,
        user_id: str,
        username: str,
        role: UserRole,
        extra_claims: Optional[Dict] = None
    ) -> str:
        """
        创建Access Token
        
        Access Token用于API访问认证，包含用户身份信息。
        
        Args:
            user_id: 用户ID
            username: 用户名
            role: 用户角色
            extra_claims: 额外的声明（可选）
            
        Returns:
            JWT Token字符串
        """
        now = datetime.utcnow()  # 当前时间
        expires = now + timedelta(minutes=self.access_token_expire)  # 过期时间
        
        # 构建Token载荷
        payload = {
            "sub": user_id,                     # Subject：用户ID
            "username": username,                # 用户名
            "role": role.value,                  # 角色
            "iat": now,                          # Issued At：签发时间
            "exp": expires,                      # Expiration：过期时间
            "type": "access"                     # Token类型
        }
        
        # 添加额外声明
        if extra_claims:
            payload.update(extra_claims)
            
        # 生成JWT Token
        # jwt.encode(payload, secret_key, algorithm)
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
    def create_refresh_token(self, user_id: str) -> str:
        """
        创建Refresh Token
        
        Refresh Token用于获取新的Access Token，
        有效期较长，但权限较小。
        
        Args:
            user_id: 用户ID
            
        Returns:
            JWT Token字符串
        """
        now = datetime.utcnow()
        expires = now + timedelta(days=self.refresh_token_expire)
        
        payload = {
            "sub": user_id,             # 用户ID
            "iat": now,                 # 签发时间
            "exp": expires,             # 过期时间
            "type": "refresh"           # Token类型：refresh
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
    def verify_token(self, token: str) -> Optional[Dict]:
        """
        验证Token
        
        验证Token的签名和有效期
        
        Args:
            token: JWT Token字符串
            
        Returns:
            Token载荷（验证成功）或None（验证失败）
        """
        # 检查Token是否在黑名单中
        if token in self.token_blacklist:
            return None
            
        try:
            # 解码并验证Token
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            # Token过期
            return None
        except jwt.InvalidTokenError:
            # Token无效（签名错误、格式错误等）
            return None
            
    def revoke_token(self, token: str):
        """
        撤销Token
        
        将Token加入黑名单，已撤销的Token无法通过验证。
        用于：
        - 用户登出
        - 管理员强制下线
        - 检测到异常访问
        
        Args:
            token: JWT Token字符串
        """
        self.token_blacklist.add(token)
        
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """
        使用Refresh Token刷新Access Token
        
        Args:
            refresh_token: Refresh Token
            
        Returns:
            新的Access Token或None（刷新失败）
        """
        # 验证Refresh Token
        payload = self.verify_token(refresh_token)
        if not payload:
            return None
            
        # 检查Token类型
        if payload.get("type") != "refresh":
            return None
            
        # 获取用户ID
        user_id = payload.get("sub")
        
        # 创建新的Access Token
        # 注意：实际应用中应从数据库获取完整的用户信息
        return self.create_access_token(
            user_id=user_id,
            username="",  # 从数据库获取
            role=UserRole.USER  # 从数据库获取
        )


class PermissionChecker:
    """
    权限检查器
    
    基于RBAC（Role-Based Access Control）模型进行权限控制。
    
    RBAC原理：
    - 用户被分配角色（如Admin、User）
    - 角色拥有权限（如chat:read、admin:write）
    - 通过检查用户的角色是否拥有所需权限来判断访问权限
    
    权限命名规范：
    - 资源:操作 的格式
    - 如 chat:read 表示读取聊天记录的权限
    """
    
    # 权限定义表：key为权限名，value为拥有该权限的角色列表
    PERMISSIONS = {
        # 聊天权限：所有人都可以读取，登录用户可以写入
        "chat:read": [UserRole.ADMIN, UserRole.USER, UserRole.GUEST],
        "chat:write": [UserRole.ADMIN, UserRole.USER],
        
        # 管理权限：只有管理员可以操作
        "admin:read": [UserRole.ADMIN],
        "admin:write": [UserRole.ADMIN],
        
        # 用户管理权限
        "user:read": [UserRole.ADMIN, UserRole.USER],
        "user:write": [UserRole.ADMIN, UserRole.USER],
    }
    
    @classmethod
    def has_permission(cls, role: UserRole, permission: str) -> bool:
        """
        检查角色是否拥有指定权限
        
        Args:
            role: 用户角色
            permission: 权限标识
            
        Returns:
            是否拥有权限
        """
        # 从权限表中获取该权限允许的角色列表
        allowed_roles = cls.PERMISSIONS.get(permission, [])
        # 检查当前角色是否在允许列表中
        return role in allowed_roles
        
    @classmethod
    def check_permission(cls, user: User, permission: str) -> bool:
        """
        检查用户是否拥有指定权限
        
        Args:
            user: 用户对象
            permission: 权限标识
            
        Returns:
            是否拥有权限
        """
        # 检查用户是否激活
        if not user.is_active:
            return False
        # 检查权限
        return cls.has_permission(user.role, permission)


class UserManager:
    """
    用户管理器
    
    负责用户的创建、认证、查询等操作。
    
    注意：当前实现使用内存存储，实际生产环境应使用数据库。
    """
    
    def __init__(self):
        """初始化用户管理器"""
        # 用户存储（实际应使用数据库）
        # key: user_id, value: 用户数据字典
        self._users: Dict[str, Dict] = {}
        # JWT认证实例
        self._jwt_auth = JWTAuth(secret_key="your-secret-key-change-in-production")
        
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.USER
    ) -> Optional[User]:
        """
        创建新用户
        
        Args:
            username: 用户名
            email: 邮箱
            password: 密码（会进行哈希存储）
            role: 用户角色，默认普通用户
            
        Returns:
            用户对象（创建成功）或None（用户名已存在）
        """
        # 检查用户名是否已存在
        for user in self._users.values():
            if user["username"] == username:
                return None
                
        # 生成唯一用户ID
        import uuid
        user_id = str(uuid.uuid4())
        
        # 构建用户数据
        user_data = {
            "id": user_id,
            "username": username,
            "email": email,
            "password_hash": self._jwt_auth.hash_password(password),  # 哈希密码
            "role": role,
            "is_active": True,
            "created_at": datetime.now().isoformat()
        }
        
        # 存储用户
        self._users[user_id] = user_data
        
        # 返回用户对象
        return User(
            id=user_id,
            username=username,
            email=email,
            role=role,
            is_active=True,
            created_at=user_data["created_at"]
        )
        
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """
        用户认证
        
        验证用户名和密码，返回用户对象
        
        Args:
            username: 用户名
            password: 明文密码
            
        Returns:
            用户对象（认证成功）或None（认证失败）
        """
        # 查找用户
        user_data = None
        for u in self._users.values():
            if u["username"] == username:
                user_data = u
                break
                
        # 用户不存在
        if not user_data:
            return None
            
        # 验证密码
        if not self._jwt_auth.verify_password(password, user_data["password_hash"]):
            return None
            
        # 检查用户是否激活
        if not user_data["is_active"]:
            return None
            
        return User(
            id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            role=user_data["role"],
            is_active=user_data["is_active"],
            created_at=user_data["created_at"]
        )
        
    def get_user(self, user_id: str) -> Optional[User]:
        """
        获取用户
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户对象或None
        """
        user_data = self._users.get(user_id)
        if not user_data:
            return None
            
        return User(
            id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            role=user_data["role"],
            is_active=user_data["is_active"],
            created_at=user_data["created_at"]
        )
        
    def get_jwt_auth(self) -> JWTAuth:
        """获取JWT认证实例"""
        return self._jwt_auth


# 全局用户管理器实例（单例）
_user_manager = None

def get_user_manager() -> UserManager:
    """
    获取用户管理器实例（单例）
    
    Returns:
        UserManager实例
    """
    global _user_manager
    if _user_manager is None:
        _user_manager = UserManager()
    return _user_manager
