import os
import time
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .model_manager import ModelManager
from .dependencies import app_state, get_model_manager
from .routes import prediction, models, explanation, websocket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting CalmSense API...")

    models_dir = os.environ.get("MODELS_DIR", "./models")
    default_model = os.environ.get("DEFAULT_MODEL", None)

    app_state.model_manager = ModelManager(
        models_dir=models_dir, default_model=default_model, lazy_load=True
    )

    if default_model:
        success, load_time = app_state.model_manager.load_model(default_model)
        if success:
            logger.info(f"Loaded default model: {default_model} in {load_time:.2f}ms")
        else:
            logger.warning(f"Failed to load default model: {default_model}")

    logger.info("CalmSense API started successfully")
    yield
    logger.info("Shutting down CalmSense API...")


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60
        self._hits: dict = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        cutoff = now - self.window_seconds

        hits = self._hits[client_ip]
        self._hits[client_ip] = [t for t in hits if t > cutoff]

        if len(self._hits[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too many requests",
                    "retry_after_seconds": self.window_seconds,
                },
            )

        self._hits[client_ip].append(now)
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), camera=(), microphone=()"
        )
        return response


def create_app(
    title: str = "CalmSense Stress Detection API",
    version: str = "1.0.0",
    models_dir: Optional[str] = None,
    default_model: Optional[str] = None,
    cors_origins: Optional[list] = None,
) -> FastAPI:
    application = FastAPI(
        title=title,
        version=version,
        description="""
        ## CalmSense Stress Detection API

        Real-time stress detection using physiological signals.

        ### Features
        - **Prediction**: Single and batch stress predictions
        - **Explanation**: SHAP, LIME, Grad-CAM explanations
        - **Model Management**: Load, list, and switch models
        - **WebSocket**: Real-time streaming predictions
        - **Clinical Interpretation**: Physiologically-grounded insights

        ### Models
        - Classical ML: Random Forest, XGBoost, LightGBM, CatBoost
        - Deep Learning: CNN, LSTM, Transformer, Cross-Modal Attention

        ### Quick Start
        1. Check available models: `GET /api/v1/models`
        2. Make prediction: `POST /api/v1/predict`
        3. Get explanation: `POST /api/v1/explain`
        """,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    if cors_origins:
        origins = cors_origins
    else:
        env_origins = os.environ.get("CORS_ORIGINS")
        if env_origins:
            origins = [o.strip() for o in env_origins.split(",")]
        else:
            origins = DEFAULT_CORS_ORIGINS
            logger.warning(
                "CORS_ORIGINS not set. Using default origins. "
                "Set CORS_ORIGINS environment variable for production."
            )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
    )

    application.add_middleware(SecurityHeadersMiddleware)

    rate_limit = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120"))
    application.add_middleware(RateLimitMiddleware, requests_per_minute=rate_limit)

    @application.middleware("http")
    async def add_timing_header(request: Request, call_next):
        start_time = time.time()
        request_id = app_state.increment_request_count()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        response.headers["X-Request-ID"] = str(request_id)
        return response

    application.include_router(prediction.router, prefix="/api/v1", tags=["Prediction"])
    application.include_router(models.router, prefix="/api/v1", tags=["Models"])
    application.include_router(
        explanation.router, prefix="/api/v1", tags=["Explanation"]
    )
    application.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])

    @application.get("/", tags=["Root"])
    async def root():
        return {
            "name": title,
            "version": version,
            "status": "running",
            "docs": "/docs",
            "health": "/health",
        }

    @application.get("/health", tags=["Health"])
    async def health_check():
        uptime = (datetime.now(timezone.utc) - app_state.start_time).total_seconds()
        return {
            "status": "healthy",
            "version": version,
            "models_loaded": (
                len(app_state.model_manager.models) if app_state.model_manager else 0
            ),
            "gpu_available": TORCH_AVAILABLE and torch.cuda.is_available(),
            "uptime_seconds": uptime,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @application.get("/health/detailed", tags=["Health"])
    async def detailed_health():
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
        except ImportError:
            memory_mb = 0.0
            cpu_percent = 0.0

        uptime = (datetime.now(timezone.utc) - app_state.start_time).total_seconds()

        gpu_memory_mb = None
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024

        uptime_minutes = uptime / 60.0
        requests_per_minute = (
            app_state.request_count / uptime_minutes if uptime_minutes > 0 else 0.0
        )

        return {
            "status": "healthy",
            "version": version,
            "models_loaded": (
                len(app_state.model_manager.models) if app_state.model_manager else 0
            ),
            "gpu_available": TORCH_AVAILABLE and torch.cuda.is_available(),
            "uptime_seconds": uptime,
            "memory_usage_mb": memory_mb,
            "cpu_percent": cpu_percent,
            "gpu_memory_mb": gpu_memory_mb,
            "active_connections": 0,
            "requests_per_minute": requests_per_minute,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @application.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.warning(f"ValueError on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=400,
            content={
                "error": "Bad Request",
                "detail": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    @application.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)

        is_debug = os.environ.get("DEBUG", "false").lower() == "true"
        detail = str(exc) if is_debug else "An unexpected error occurred"

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": detail,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    return application


app = create_app()


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    import uvicorn

    uvicorn.run("api.main:app", host=host, port=port, reload=reload)


__all__ = ["app", "create_app", "get_model_manager", "app_state", "run_server"]


if __name__ == "__main__":
    run_server()
