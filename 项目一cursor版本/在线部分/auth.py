import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Optional

import pymysql
from fastapi import HTTPException, Request
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
SQLITE_DB_PATH = DATA_DIR / "auth.db"
SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "dev-secret-change-me")
TOKEN_TTL_SECONDS = int(os.getenv("AUTH_TOKEN_TTL_SECONDS", str(60 * 60 * 24 * 7)))
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "2309b")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "123456")
MYSQL_CHARSET = os.getenv("MYSQL_CHARSET", "utf8mb4")

USE_MYSQL = os.getenv("AUTH_BACKEND", "mysql").lower() != "sqlite"
_active_backend = "mysql" if USE_MYSQL else "sqlite"


def _sqlite_conn():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _mysql_conn():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset=MYSQL_CHARSET,
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor,
    )


def _use_mysql() -> bool:
    return _active_backend == "mysql"


def _init_sqlite_db():
    with _sqlite_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                last_login_at INTEGER
            )
            """
        )
        conn.commit()


def init_db():
    global _active_backend
    if USE_MYSQL:
        try:
            with _mysql_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS users (
                            id BIGINT PRIMARY KEY AUTO_INCREMENT,
                            username VARCHAR(255) UNIQUE NOT NULL,
                            password_hash VARCHAR(255) NOT NULL,
                            salt VARCHAR(255) NOT NULL,
                            created_at BIGINT NOT NULL,
                            last_login_at BIGINT NULL
                        ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
                        """
                    )
                conn.commit()
            _active_backend = "mysql"
            return
        except Exception as exc:
            print(f"⚠️ MySQL 初始化失败，自动切换为 SQLite：{exc}")
            _active_backend = "sqlite"

    _init_sqlite_db()


def _hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return base64.urlsafe_b64encode(digest).decode("utf-8"), salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    calc, _ = _hash_password(password, salt)
    return hmac.compare_digest(calc, password_hash)


def create_user(username: str, password: str):
    username = username.strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")
    password_hash, salt = _hash_password(password)
    now = int(time.time())
    if _use_mysql():
        try:
            conn = _mysql_conn()
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO users (username, password_hash, salt, created_at) VALUES (%s, %s, %s, %s)",
                        (username, password_hash, salt, now),
                    )
                conn.commit()
        except pymysql.err.IntegrityError:
            raise HTTPException(status_code=400, detail="用户名已存在")
        return

    try:
        with _sqlite_conn() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
                (username, password_hash, salt, now),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="用户名已存在")


def get_user(username: str):
    username = username.strip()
    if _use_mysql():
        conn = _mysql_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users WHERE username = %s", (username,))
                row = cur.fetchone()
                return row

    with _sqlite_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None


def update_last_login(username: str):
    username = username.strip()
    if _use_mysql():
        conn = _mysql_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE users SET last_login_at = %s WHERE username = %s", (int(time.time()), username))
            conn.commit()
        return

    with _sqlite_conn() as conn:
        conn.execute("UPDATE users SET last_login_at = ? WHERE username = ?", (int(time.time()), username))
        conn.commit()


def _sign(data: bytes) -> str:
    return base64.urlsafe_b64encode(hmac.new(SECRET_KEY.encode("utf-8"), data, hashlib.sha256).digest()).decode("utf-8")


def create_access_token(username: str) -> str:
    payload = {"sub": username, "exp": int(time.time()) + TOKEN_TTL_SECONDS}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    body = base64.urlsafe_b64encode(raw).decode("utf-8")
    sig = _sign(raw)
    return f"{body}.{sig}"


def decode_access_token(token: str) -> str:
    try:
        body, sig = token.split(".", 1)
        raw = base64.urlsafe_b64decode(body.encode("utf-8"))
        expected = _sign(raw)
        if not hmac.compare_digest(expected, sig):
            raise HTTPException(status_code=401, detail="无效的登录凭证")
        payload = json.loads(raw.decode("utf-8"))
        if int(payload.get("exp", 0)) < int(time.time()):
            raise HTTPException(status_code=401, detail="登录已过期")
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="无效的登录凭证")
        return username
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="无效的登录凭证")


def get_token_from_request(request: Request) -> Optional[str]:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth.split(" ", 1)[1].strip()
    token = request.cookies.get("access_token")
    return token.strip() if token else None


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    username: str
    token: str
    message: str


class MeResponse(BaseModel):
    username: str
    message: str
