"""
FastAPI application for Coffee Bean Analyzer.

Endpoints:
  POST /api/predict  — Upload image → dual ONNX inference (roast + defect)
  GET  /api/history  — Recent prediction history
  GET  /api/stats    — Aggregate statistics
  GET  /health       — Health check
  GET  /             — Serve frontend SPA

Security:
  - CORS locked to configured origins
  - Rate limiting on /api/predict (30/min per IP)
  - API docs disabled in production
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import get_settings
from app.database import close_db, get_history, get_stats, init_db, log_prediction
from app.storage import init_storage, upload_image
from model.predict import ONNXPredictor

# ── Class definitions (duplicated here to avoid importing torch) ────────
ROAST_CLASSES = ["Dark", "Green", "Light", "Medium"]
DEFECT_CLASSES = [
    "Broken", "Cut", "Dry Cherry", "Fade", "Floater",
    "Full Black", "Full Sour", "Fungus Damage", "Husk",
    "Immature", "Parchment", "Partial Black", "Partial Sour",
    "Severe Insect Damage", "Shell", "Slight Insect Damage", "Withered",
]

# ── Global model references ────────────────────────────────────────────
roast_predictor: Optional[ONNXPredictor] = None
defect_predictor: Optional[ONNXPredictor] = None

# ── Rate limiter ────────────────────────────────────────────────────────
settings = get_settings()
limiter = Limiter(key_func=get_remote_address, default_limits=[])


# ── Lifespan ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models + init DB. Shutdown: cleanup."""
    global roast_predictor, defect_predictor

    print("🚀 Starting Coffee Bean Analyzer...")

    # Initialize database
    print("  📊 Initializing database...")
    await init_db()

    # Initialize cloud storage
    print("  ☁️  Initializing storage...")
    await init_storage()

    # Load ONNX models
    print("  🧠 Loading ONNX models...")
    try:
        roast_predictor = ONNXPredictor(
            model_path=settings.ROAST_MODEL_PATH,
            class_names=ROAST_CLASSES,
        )
        print(f"     ✅ Roast model loaded: {roast_predictor}")
    except FileNotFoundError:
        print(f"     ⚠️  Roast model not found at {settings.ROAST_MODEL_PATH}")

    try:
        defect_predictor = ONNXPredictor(
            model_path=settings.DEFECT_MODEL_PATH,
            class_names=DEFECT_CLASSES,
        )
        print(f"     ✅ Defect model loaded: {defect_predictor}")
    except FileNotFoundError:
        print(f"     ⚠️  Defect model not found at {settings.DEFECT_MODEL_PATH}")

    print("✅ Application ready!\n")
    yield

    # Shutdown
    print("\n🛑 Shutting down...")
    await close_db()


# ── App ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Coffee Bean Analyzer API",
    description="Classify coffee beans by roast level and detect physical defects",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
    openapi_url="/openapi.json" if settings.ENVIRONMENT == "development" else None,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Allowed file types ──────────────────────────────────────────────────
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# ── Routes ──────────────────────────────────────────────────────────────


@app.get("/")
async def serve_frontend():
    """Serve the frontend SPA."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse(
        {"message": "Coffee Bean Analyzer API", "docs": "/docs"},
        status_code=200,
    )


@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run."""
    return {
        "status": "healthy",
        "models": {
            "roast": roast_predictor is not None and roast_predictor.is_loaded(),
            "defect": defect_predictor is not None and defect_predictor.is_loaded(),
        },
    }


@app.post("/api/predict")
@limiter.limit(settings.RATE_LIMIT)
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Upload a coffee bean image for classification.

    Returns roast level prediction and defect detection results
    with confidence scores and probability distributions.
    """
    # Validate file type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )

    # Read and validate file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)} MB",
        )

    # Check at least one model is loaded
    if roast_predictor is None and defect_predictor is None:
        raise HTTPException(
            status_code=503,
            detail="No models are currently loaded. Please check server logs.",
        )

    # Load image
    try:
        import io
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Run inference
    start_time = time.perf_counter()
    result = {}

    if roast_predictor is not None:
        roast_result = roast_predictor.predict(image)
        result["roast"] = {
            "prediction": roast_result["prediction"],
            "confidence": roast_result["confidence"],
            "probabilities": roast_result["probabilities"],
        }

    if defect_predictor is not None:
        defect_result = defect_predictor.predict(image)
        result["defect"] = {
            "prediction": defect_result["prediction"],
            "confidence": defect_result["confidence"],
            "probabilities": defect_result["probabilities"],
        }

    total_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    result["inference_time_ms"] = total_time_ms

    # Upload image to cloud storage (non-blocking, best-effort)
    image_url = None
    try:
        image_url = await upload_image(image, file.filename)
        result["image_url"] = image_url
    except Exception:
        pass  # Don't fail the prediction if upload fails

    # Log prediction to database (best-effort)
    try:
        if "roast" in result:
            await log_prediction(
                filename=file.filename,
                analysis_type="roast",
                predicted_class=result["roast"]["prediction"],
                confidence=result["roast"]["confidence"],
                probabilities=result["roast"]["probabilities"],
                inference_time_ms=total_time_ms,
                image_url=image_url,
            )
        if "defect" in result:
            await log_prediction(
                filename=file.filename,
                analysis_type="defect",
                predicted_class=result["defect"]["prediction"],
                confidence=result["defect"]["confidence"],
                probabilities=result["defect"]["probabilities"],
                inference_time_ms=total_time_ms,
                image_url=image_url,
            )
    except Exception as e:
        print(f"  ⚠️  Failed to log prediction: {e}")

    return result


@app.get("/api/history")
async def history(limit: int = 20, offset: int = 0):
    """Get recent prediction history."""
    if limit > 100:
        limit = 100
    records = await get_history(limit=limit, offset=offset)
    return {"predictions": records, "count": len(records)}


@app.get("/api/stats")
async def stats():
    """Get aggregate prediction statistics."""
    return await get_stats()
