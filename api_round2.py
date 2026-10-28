from importlib import import_module
from pathlib import Path
import posixpath

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.env_loader import load_root_env


load_root_env()

from backend.admin_auth import COOKIE_NAME, SESSION, allowed_origins, router as auth_router
from backend.demo_catalog import router as demo_router, find_case, saved_result
from backend.route.api import router as api_router


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="输入一句诗")
    imagery_threshold: float = Field(0.5, ge=0.0, le=1.0)


app = FastAPI(title="Hexi Poetry Round2 Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(demo_router)
app.include_router(api_router)


@app.middleware("http")
async def bind_admin_session(request: Request, call_next):
    marker = SESSION.set(request.cookies.get(COOKIE_NAME, ""))
    try:
        if posixpath.normpath(request.url.path.replace("\\", "/")).startswith("/static/generated/"):
            from backend.admin_auth import session_valid
            if not session_valid(SESSION.get()):
                from fastapi.responses import JSONResponse
                return JSONResponse({"detail": "管理员登录后可读取生成资源"}, status_code=403)
        return await call_next(request)
    finally:
        SESSION.reset(marker)

app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "frontend" / "static")), name="static")


def _load_orchestration_attr(attr_name: str):
    module_path = "backend.service.orchestration"
    try:
        module = import_module(module_path)
    except (ModuleNotFoundError, ImportError, OSError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"optional service unavailable: {module_path}: {exc}",
        ) from exc
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"optional service unavailable: {module_path}.{attr_name}",
        ) from exc


@app.get("/health")
def health():
    get_predictor = _load_orchestration_attr("get_predictor")
    predictor = get_predictor()
    return {
        "status": "ok",
        "semantic_backend": type(predictor).__name__,
        "device": predictor.device,
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    from backend.deployment_security import generation_access_allowed
    case = find_case(req.text)
    if case and not generation_access_allowed("image"):
        return saved_result(case)["prediction"]
    try:
        get_predictor = _load_orchestration_attr("get_predictor")
        return get_predictor().predict(
            req.text,
            imagery_threshold=req.imagery_threshold,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/predict/map")
def predict_map(req: PredictRequest):
    from backend.deployment_security import generation_access_allowed
    case = find_case(req.text)
    if case and not generation_access_allowed("image"):
        return saved_result(case)["map"]
    try:
        get_predictor = _load_orchestration_attr("get_predictor")
        build_map_prediction = _load_orchestration_attr("build_map_prediction")
        prediction = get_predictor().predict(
            req.text,
            imagery_threshold=req.imagery_threshold,
        )
        return build_map_prediction(prediction)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
