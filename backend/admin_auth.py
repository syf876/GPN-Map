"""Server-side administrator sessions. Credentials never appear in task state or responses."""
import hashlib
import hmac
import secrets
import sqlite3
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from backend.deployment_security import _setting_value

COOKIE_NAME = "hexi_admin_session"
LOCKED_MESSAGE = "实时生成仅向管理员开放，可切换演示案例查看效果"
SESSION = ContextVar("administrator_session", default="")
router = APIRouter(prefix="/api/auth", tags=["administrator"])


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1)
    return "scrypt$16384$8$1$" + salt.hex() + "$" + digest.hex()


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, n, r, p, salt, expected = encoded.split("$")
        if scheme != "scrypt" or (int(n), int(r), int(p)) != (16384, 8, 1):
            return False
        digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1)
        return hmac.compare_digest(digest, bytes.fromhex(expected))
    except (ValueError, TypeError):
        return False


def _secret() -> str:
    return _setting_value("ADMIN_SESSION_SECRET", "")


@contextmanager
def _db():
    path = Path(_setting_value("ADMIN_SESSION_DB", str(Path(__file__).resolve().parents[1] / "artifacts" / "admin_sessions.sqlite3")))
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10)
    conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, expires REAL NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS attempts (ip TEXT NOT NULL, at REAL NOT NULL)")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _signature(value: str) -> str:
    return hmac.new(_secret().encode(), value.encode(), hashlib.sha256).hexdigest()


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def session_valid(token: str) -> bool:
    if not token or len(token) > 256 or len(_secret()) < 32:
        return False
    try:
        session_id, signature = token.split(".", 1)
        if not hmac.compare_digest(signature, _signature(session_id)):
            return False
        with _db() as db:
            row = db.execute("SELECT expires FROM sessions WHERE id=?", (_digest(token),)).fetchone()
        return bool(row and row[0] > time.time())
    except (ValueError, TypeError, sqlite3.Error):
        return False


def require_generation_access(kind: str = "generation") -> None:
    if _setting_value("ENABLE_PAID_APIS", "false").lower() != "true":
        raise HTTPException(status_code=503, detail="外部付费 API 默认禁用，请由管理员配置后启用")
    # Recheck the revocable session immediately before every provider submission.
    if not session_valid(SESSION.get()):
        raise HTTPException(status_code=403, detail=LOCKED_MESSAGE)


@contextmanager
def generation_session(token: str):
    marker = SESSION.set(token)
    try:
        yield
    finally:
        SESSION.reset(marker)


def csrf_token(token: str) -> str:
    return _signature("csrf:" + token)


def allowed_origins() -> list[str]:
    return [x.strip().rstrip("/") for x in _setting_value("APP_ALLOWED_ORIGINS", "http://127.0.0.1:8080,http://localhost:8080").split(",") if x.strip()]


def check_origin(request: Request) -> None:
    origin = request.headers.get("origin", "").rstrip("/")
    if origin and origin not in allowed_origins():
        raise HTTPException(status_code=403, detail="请求来源不允许")


def authenticated_write(request: Request) -> str:
    check_origin(request)
    token = request.cookies.get(COOKIE_NAME, "")
    if not session_valid(token):
        raise HTTPException(status_code=403, detail=LOCKED_MESSAGE)
    if not hmac.compare_digest(request.headers.get("X-CSRF-Token", "").encode(), csrf_token(token).encode()):
        raise HTTPException(status_code=403, detail="会话校验失败，请刷新后重试")
    return token


def _cookie_options():
    return dict(httponly=True, secure=_setting_value("AUTH_COOKIE_SECURE", "true").lower() not in {"false", "0"}, samesite="strict", path="/")


class LoginRequest(BaseModel):
    password: str = Field(..., min_length=1, max_length=1024)


@router.get("/session")
def session_status(request: Request, response: Response):
    response.headers["Cache-Control"] = "no-store"
    token = request.cookies.get(COOKIE_NAME, "")
    valid = session_valid(token)
    return {"is_admin": valid, "csrf_token": csrf_token(token) if valid else ""}


@router.post("/login")
def login(body: LoginRequest, request: Request, response: Response):
    check_origin(request)
    encoded = _setting_value("ADMIN_PASSWORD_HASH", "")
    if not encoded or len(_secret()) < 32:
        raise HTTPException(status_code=503, detail="管理员登录尚未配置")
    ip = request.client.host if request.client else "unknown"
    if _setting_value("AUTH_TRUST_PROXY", "false").lower() == "true" and ip in {"127.0.0.1", "::1"}:
        ip = request.headers.get("X-Forwarded-For", ip).split(",")[0].strip()
    now = time.time()
    window = max(60, int(_setting_value("ADMIN_LOGIN_WINDOW_SECONDS", "900")))
    limit = max(1, int(_setting_value("ADMIN_LOGIN_MAX_ATTEMPTS", "5")))
    with _db() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("DELETE FROM attempts WHERE at<?", (now-window,))
        count = db.execute("SELECT COUNT(*) FROM attempts WHERE ip=?", (_digest(ip),)).fetchone()[0]
        if count >= limit:
            raise HTTPException(status_code=429, detail="登录尝试过多，请稍后再试", headers={"Retry-After": str(window)})
        db.execute("INSERT INTO attempts VALUES (?,?)", (_digest(ip), now))
    if not verify_password(body.password, encoded):
        raise HTTPException(status_code=401, detail="管理员密码错误")
    token = secrets.token_urlsafe(32)
    token += "." + _signature(token)
    ttl = min(86400, max(60, int(_setting_value("ADMIN_SESSION_TTL_SECONDS", "3600"))))
    with _db() as db:
        db.execute("DELETE FROM sessions WHERE expires<?", (now,))
        db.execute("INSERT INTO sessions VALUES (?,?)", (_digest(token), now+ttl))
        db.execute("DELETE FROM attempts WHERE ip=?", (_digest(ip),))
    response.set_cookie(COOKIE_NAME, token, max_age=ttl, **_cookie_options())
    response.headers["Cache-Control"] = "no-store"
    return {"is_admin": True, "csrf_token": csrf_token(token)}


@router.post("/logout")
def logout(request: Request, response: Response):
    token = authenticated_write(request)
    with _db() as db:
        db.execute("DELETE FROM sessions WHERE id=?", (_digest(token),))
    response.delete_cookie(COOKIE_NAME, **_cookie_options())
    response.headers["Cache-Control"] = "no-store"
    return {"is_admin": False, "csrf_token": ""}
