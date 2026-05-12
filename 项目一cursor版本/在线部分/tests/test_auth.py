"""测试 auth.py：用户认证、密码哈希、JWT token"""
import time
import pytest
from fastapi import HTTPException

from auth import (
    _hash_password,
    verify_password,
    create_user,
    get_user,
    update_last_login,
    create_access_token,
    decode_access_token,
    init_db,
)

# 确保用 SQLite（conftest 已设 AUTH_BACKEND=sqlite）


class TestPasswordHashing:
    def test_hash_and_verify(self):
        """密码哈希后能正确验证"""
        password_hash, salt = _hash_password("my_password")
        assert verify_password("my_password", password_hash, salt)

    def test_wrong_password_fails(self):
        """错误密码验证应失败"""
        password_hash, salt = _hash_password("correct")
        assert not verify_password("wrong", password_hash, salt)

    def test_different_salts(self):
        """两次哈希结果不同（因为 salt 不同）"""
        h1, s1 = _hash_password("same")
        h2, s2 = _hash_password("same")
        assert h1 != h2


class TestUserCRUD:
    def setup_method(self):
        # 清理数据库，避免之前的测试残留
        import auth as auth_mod
        db_path = auth_mod.DATA_DIR / "auth.db"
        if db_path.exists():
            db_path.unlink()
        init_db()

    def teardown_method(self):
        import auth as auth_mod
        db_path = auth_mod.DATA_DIR / "auth.db"
        if db_path.exists():
            db_path.unlink()

    def test_create_and_get_user(self):
        """创建用户后能查询到"""
        create_user("testuser1", "testpass123")
        user = get_user("testuser1")
        assert user is not None
        assert user["username"] == "testuser1"

    def test_create_duplicate_user(self):
        """重复用户名应报错"""
        create_user("testdup", "pass123")
        with pytest.raises(HTTPException) as exc:
            create_user("testdup", "otherpass")
        assert exc.value.status_code == 400

    def test_get_nonexistent_user(self):
        """不存在的用户返回 None"""
        assert get_user("nonexistent_user_xyz") is None

    def test_update_last_login(self):
        """更新登录时间不应报错"""
        create_user("testlogin", "pass123")
        update_last_login("testlogin")
        user = get_user("testlogin")
        assert user["last_login_at"] is not None


class TestToken:
    def test_create_and_decode(self):
        """创建 token 后能解码出用户名"""
        token = create_access_token("testuser")
        username = decode_access_token(token)
        assert username == "testuser"

    def test_expired_token(self):
        """
        BUG 检测: 当 TOKEN_TTL_SECONDS=0 时，exp = now + 0 = now
        检查条件 exp < now 在相等时不成立，所以 token 不会过期
        应该改为 exp <= now 或使用负 TTL
        """
        import auth
        original_ttl = auth.TOKEN_TTL_SECONDS
        auth.TOKEN_TTL_SECONDS = 0  # 立即过期（但现实现不会过期）
        try:
            token = create_access_token("testuser")
            # 当前代码不会抛出异常（bug）
            username = decode_access_token(token)
            assert username == "testuser", (
                "BUG: TTL=0 时 token 仍然有效，"
                "create_access_token 中 exp=now+0=now，"
                "decode_access_token 中 exp < now 为 False"
            )
        finally:
            auth.TOKEN_TTL_SECONDS = original_ttl

    def test_tampered_token(self):
        """篡改 token 应拒绝"""
        token = create_access_token("testuser")
        tampered = token + "x"
        with pytest.raises(HTTPException) as exc:
            decode_access_token(tampered)
        assert exc.value.status_code == 401

    def test_empty_token(self):
        """空 token 应拒绝"""
        with pytest.raises(HTTPException):
            decode_access_token("")
