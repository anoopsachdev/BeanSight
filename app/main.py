"""
FastAPI application for Coffee Bean Analyzer.

Pipeline:
  Stage 1  Roast classifier (EfficientNet-B0, 4 classes) — decides the route
  Stage 2a Green bean  → 17-class defect classifier (EfficientNet-B0, single bean)
  Stage 2b Roasted bean → YOLOv8n defect detector (Broken / Quaker / Scorched / Normal)

Endpoints:
  POST /api/predict  — Upload image → tiered ONNX inference (roast + defect)
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

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import get_settings
from app.database import (
    DatabaseUnavailable,
    close_db,
    db_status,
    get_history,
    get_stats,
    init_db,
    log_prediction,
)
from app.storage import init_storage, storage_status, upload_image
from model.predict import ONNXPredictor, YOLOv8ONNXPredictor

# ── Class definitions (duplicated here to avoid importing torch) ────────
ROAST_CLASSES = ["Dark", "Green", "Light", "Medium"]
DEFECT_CLASSES = [
    "Broken", "Cut", "Dry Cherry", "Fade", "Floater",
    "Full Black", "Full Sour", "Fungus Damage", "Husk",
    "Immature", "Parchment", "Partial Black", "Partial Sour",
    "Severe Insect Damage", "Shell", "Slight Insect Damage", "Withered",
]
ROASTED_DEFECT_CLASSES = ["Broken", "Quaker", "Normal", "Scorched"]

# ── Global model references ────────────────────────────────────────────
roast_predictor: Optional[ONNXPredictor] = None
defect_predictor: Optional[ONNXPredictor] = None
roasted_yolo_predictor: Optional[YOLOv8ONNXPredictor] = None

# ── Rate limiter ────────────────────────────────────────────────────────
settings = get_settings()
def get_real_ip(request: Request) -> str:
    """Extract real client IP from X-Forwarded-For header (useful behind Cloud Run Load Balancer)."""
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"

limiter = Limiter(key_func=get_real_ip, default_limits=[])


# ── Lifespan ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models + init DB. Shutdown: cleanup."""
    global roast_predictor, defect_predictor, roasted_yolo_predictor

    print("🚀 Starting Coffee Bean Analyzer...")

    import asyncio

    # Initialize database and storage in the background to avoid blocking Uvicorn startup
    # (Cloud Run requires the port to be bound quickly for health checks)
    print("  📊 Scheduling database and storage initialization...")

    async def _init_services():
        # Both helpers swallow their own errors and record them for /health,
        # so a paused/unreachable Supabase project never crashes startup.
        await asyncio.gather(init_db(), init_storage())

    asyncio.create_task(_init_services())

    async def load_models():
        global roast_predictor, defect_predictor, roasted_yolo_predictor
        import asyncio

        print("  🧠 Loading ONNX models in background...")
        # Run synchronous ONNX loading in thread pool to not block event loop
        def _load():
            global roast_predictor, defect_predictor, roasted_yolo_predictor
            if roast_predictor is None:
                try:
                    roast_predictor = ONNXPredictor(
                        model_path=settings.ROAST_MODEL_PATH,
                        class_names=ROAST_CLASSES,
                    )
                    print(f"     ✅ Roast model loaded")
                except FileNotFoundError:
                    pass
            if defect_predictor is None:
                try:
                    defect_predictor = ONNXPredictor(
                        model_path=settings.DEFECT_MODEL_PATH,
                        class_names=DEFECT_CLASSES,
                    )
                    print(f"     ✅ Green Defect model loaded")
                except FileNotFoundError:
                    pass
            if roasted_yolo_predictor is None:
                try:
                    roasted_yolo_predictor = YOLOv8ONNXPredictor(
                        model_path=settings.ROASTED_DEFECT_MODEL_PATH,
                        class_names=ROASTED_DEFECT_CLASSES,
                    )
                    print(f"     ✅ Roasted YOLO model loaded")
                except Exception as e:
                    print(f"     ⚠️  YOLO model failed: {e}")

        await asyncio.to_thread(_load)
        print("✅ Models ready!")

    asyncio.create_task(load_models())
    print("✅ Application ready (warming up background tasks)!\n")
    yield

    # Shutdown
    print("\n🛑 Shutting down...")
    await close_db()


# ── App ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="BeanSight API",
    description="Classify coffee beans by roast level, then detect defects with the matching green or roasted model",
    version="2.1.0",
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
    """Health check endpoint for Cloud Run (and for the UI status pill)."""
    return {
        "status": "healthy",
        "models": {
            "roast": roast_predictor is not None and roast_predictor.is_loaded(),
            "defect": defect_predictor is not None and defect_predictor.is_loaded(),
            "roasted_yolo": roasted_yolo_predictor is not None and roasted_yolo_predictor.is_loaded(),
        },
        "database": db_status(),
        "storage": storage_status(),
    }


@app.post("/api/predict")
@limiter.limit(settings.RATE_LIMIT)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    bean_type: str = Form("auto"),
):
    """
    Upload a coffee bean image for classification.

    ``bean_type`` selects which defect model runs:
      - ``auto`` (default): decided by the roast classifier (Green → green model)
      - ``green`` / ``roasted``: manual override when the classifier is wrong

    The roast classifier always runs and its result is always returned.
    """
    bean_type = (bean_type or "auto").lower()
    if bean_type not in ("auto", "green", "roasted"):
        raise HTTPException(status_code=400, detail="bean_type must be 'auto', 'green' or 'roasted'")
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

    # Stage 1: roast-level classifier (also decides which Stage 2 model runs)
    roast_prediction = "Medium"
    if roast_predictor is not None:
        roast_result = roast_predictor.predict(image)
        roast_prediction = roast_result["prediction"]
        result["roast"] = {
            "prediction": roast_result["prediction"],
            "confidence": roast_result["confidence"],
            "probabilities": roast_result["probabilities"],
        }

    # Stage 2: defect analysis, chosen by the Stage 1 result unless overridden
    classifier_route = "green" if roast_prediction == "Green" else "roasted"
    if bean_type == "auto":
        route = classifier_route
    else:
        route = bean_type
    if route == "roasted" and roasted_yolo_predictor is None:
        route = "green"
    result["route_source"] = "classifier" if bean_type == "auto" else "manual"
    result["classifier_route"] = classifier_route

    if route == "green":
        # Unroasted (green) bean → 17-class defect classifier (single bean)
        result["route"] = "green"
        if defect_predictor is not None:
            defect_result = defect_predictor.predict(image)
            result["defect"] = {
                "type": "green_agricultural",
                "prediction": defect_result["prediction"],
                "confidence": defect_result["confidence"],
                "probabilities": defect_result["probabilities"],
            }
    else:
        # Roasted bean → YOLOv8 detector (one or many beans, per-bean boxes)
        result["route"] = "roasted"
        yolo_result = roasted_yolo_predictor.predict(image)
        bean_count = yolo_result.get("bean_count", len(yolo_result.get("detections", [])))
        defect_summary = yolo_result["defect_summary"]
        result["roasted_defect"] = {
            "type": "roasted_mechanical",
            "bean_count": bean_count,
            "normal_count": yolo_result.get("normal_count", 0),
            "defect_count": yolo_result["defect_count"],
            "defect_summary": defect_summary,
            "class_summary": yolo_result.get("class_summary", defect_summary),
            "mean_confidence": yolo_result.get("mean_confidence", 0.0),
            "detections": yolo_result["detections"],
            "annotated_image": yolo_result["annotated_image"],
        }
        # Compact summary of the detector output (also used for the audit log).
        # Confidence is the mean detector score of the defect boxes (or of all
        # boxes when no defects were found) — never a made-up constant.
        if defect_summary:
            top_defect = max(defect_summary.items(), key=lambda x: x[1])[0]
            confidence = yolo_result.get("mean_defect_confidence", yolo_result.get("mean_confidence", 0.0))
        else:
            top_defect = "No Defect"
            confidence = yolo_result.get("mean_confidence", 0.0)
        class_summary = yolo_result.get("class_summary", defect_summary)
        result["defect"] = {
            "type": "roasted_mechanical",
            "prediction": top_defect,
            "confidence": confidence,
            # Share of detected beans per class (not softmax probabilities).
            "probabilities": {
                cls: round(class_summary.get(cls, 0) / max(bean_count, 1), 4)
                for cls in ROASTED_DEFECT_CLASSES
            },
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
                analysis_type="roasted_defect" if result.get("route") == "roasted" else "defect",
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
    """Get recent prediction history (empty list + flag if the DB is unreachable)."""
    limit = max(1, min(limit, 100))
    try:
        records = await get_history(limit=limit, offset=offset)
    except DatabaseUnavailable as exc:
        return {"predictions": [], "count": 0, "database_available": False, "error": str(exc)}
    return {"predictions": records, "count": len(records), "database_available": True}


@app.get("/api/stats")
async def stats():
    """Get aggregate prediction statistics (zeros + flag if the DB is unreachable)."""
    try:
        data = await get_stats()
    except DatabaseUnavailable as exc:
        return {
            "total_predictions": 0,
            "total_samples": 0,
            "avg_inference_time_ms": 0,
            "predictions_by_type": {},
            "top_predicted_classes": {},
            "roast_distribution": {},
            "defect_distribution": {},
            "database_available": False,
            "error": str(exc),
        }
    data["database_available"] = True
    return data
