import logging
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.analyze import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("deepfake-api")

app = FastAPI(
    title="Forensic.AI Backend",
    description="Multimodal deepfake detection API — Visual 3D CNN + Audio 1D CNN + Cross-Modal Attention Fusion",
    version="2.0.0"
)

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ROUTES
# =========================
app.include_router(router)


# =========================
# STARTUP — Model Registry
# =========================
@app.on_event("startup")
async def startup_model_registry():
    """
    Run the Model Registry at startup to load all trained weights.
    This ensures weight loading happens once at boot, not per-request.
    """
    try:
        from models.cnn_model import model as visual_model
        from models.audio_model import audio_model
        from services.fusion import cross_modal_attention_model
        from models.model_registry import load_all_models

        load_all_models(visual_model, audio_model, cross_modal_attention_model)
    except Exception as e:
        logger.error(f"⚠️  Model registry startup error: {e}", exc_info=True)
        logger.warning("   → Backend will start but inference may use random weights.")


# =========================
# ROOT ROUTE
# =========================
@app.get("/")
def root():
    from models.model_registry import get_registry_status, is_trained
    status = get_registry_status()
    return {
        "status": "backend running",
        "models_trained": is_trained(),
        "visual_weights": status.get("visual_path") or "random init",
        "audio_weights":  status.get("audio_path")  or "random init",
        "fusion_weights": status.get("fusion_path")  or "random init",
    }


# =========================
# MODEL STATUS ROUTE
# =========================
@app.get("/model-status")
def model_status():
    """Returns detailed status of all loaded models."""
    from models.model_registry import get_registry_status, is_trained
    status = get_registry_status()
    manifest = status.get("manifest") or {}
    return {
        "is_fully_trained": is_trained(),
        "visual_loaded":  status.get("visual_loaded", False),
        "audio_loaded":   status.get("audio_loaded", False),
        "fusion_loaded":  status.get("fusion_loaded", False),
        "visual_path":    status.get("visual_path"),
        "audio_path":     status.get("audio_path"),
        "fusion_path":    status.get("fusion_path"),
        "training_info": {
            "best_val_auc": manifest.get("best_val_auc"),
            "best_epoch":   manifest.get("best_epoch"),
            "timestamp":    manifest.get("timestamp"),
            "visual_dim":   manifest.get("visual_dim", 128),
            "audio_dim":    manifest.get("audio_dim", 128),
            "embed_dim":    manifest.get("embed_dim", 128),
        }
    }