import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from auth import init_db
from routes.auth import router as auth_router
from routes.chat import router as chat_router
from routes.debug import router as debug_router
from routes.session import router as session_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
INDEX_HTML = TEMPLATES_DIR / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("service startup completed")
    yield
    logger.info("service shutdown completed")


app = FastAPI(title="多角色智能对话系统", version="完整版", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 显式、稳定地挂载所有路由
app.include_router(auth_router)
app.include_router(session_router)
app.include_router(chat_router)
app.include_router(debug_router)


@app.get("/", response_class=HTMLResponse)
def homepage():
    if INDEX_HTML.is_file():
        return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>前端首页文件不存在</h3>", status_code=500)


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "服务运行正常"}


def _collect_routes():
    routes = []
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = sorted(list(getattr(route, "methods", []) or []))
        if path:
            routes.append({"path": path, "methods": methods})
    return {"routes": routes}


@app.get("/__routes")
def debug_routes():
    return _collect_routes()


@app.get("/_routes")
def debug_routes_single_underscore():
    return _collect_routes()


@app.get("/routes")
def debug_routes_plain():
    return _collect_routes()


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    use_tunnel = os.getenv("USE_TUNNEL", "0").strip().lower() in {"1", "true", "yes", "y"}
    ngrok_token = os.getenv("NGROK_AUTHTOKEN", "").strip()

    if use_tunnel and ngrok_token:
        try:
            from tunnel import start_tunnel

            tunnel = start_tunnel()
            logger.info("public url: %s", getattr(tunnel, "public_url", ""))
        except Exception:
            logger.exception("start tunnel failed")
    else:
        logger.info("tunnel disabled or NGROK_AUTHTOKEN missing, skip public url")

    uvicorn.run("main:app", host=host, port=port, reload=False)
