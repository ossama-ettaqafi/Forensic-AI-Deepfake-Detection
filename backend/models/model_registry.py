"""
model_registry.py
─────────────────────────────────────────────────────────────────────────────
Central Model Registry for Forensic.AI Backend.

Loads all three trained model weights at startup in the correct order,
reads the training_manifest.json (if available) to know exact architecture
dimensions, validates shapes with dummy forward passes, and exposes status.

Expected files (copied from Kaggle /kaggle/working/ output):
  backend/models/efficientnet_backbone.pth          → Visual3DCNN weights
  backend/models/audio_backbone.pth                 → AudioSpectrogramCNN weights
  backend/models/deepfake_detector_cross_modality.pt → CrossModalAttentionFusion weights
  backend/models/training_manifest.json             → Shape metadata (optional)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import logging
import torch

logger = logging.getLogger("deepfake-api")

# ─── Candidate search paths (relative to CWD or absolute from project root)
_VISUAL_PATHS = [
    "models/visual_model.pth",
    "weights/visual_model.pth",
    "backend/models/visual_model.pth",
    "backend/weights/visual_model.pth",
    "models/efficientnet_backbone.pth",
    "weights/efficientnet_backbone.pth",
    "backend/models/efficientnet_backbone.pth",
    "backend/weights/efficientnet_backbone.pth",
]
_AUDIO_PATHS = [
    "models/audio_model.pth",
    "weights/audio_model.pth",
    "backend/models/audio_model.pth",
    "backend/weights/audio_model.pth",
    "models/audio_backbone.pth",
    "weights/audio_backbone.pth",
    "backend/models/audio_backbone.pth",
    "backend/weights/audio_backbone.pth",
]
_FUSION_PATHS = [
    "models/fusion_model.pth",
    "weights/fusion_model.pth",
    "backend/models/fusion_model.pth",
    "backend/weights/fusion_model.pth",
    "models/deepfake_detector_cross_modality.pt",
    "weights/deepfake_detector_cross_modality.pt",
    "backend/models/deepfake_detector_cross_modality.pt",
    "backend/weights/deepfake_detector_cross_modality.pt",
]
_MANIFEST_PATHS = [
    "models/config.json",
    "weights/config.json",
    "backend/models/config.json",
    "backend/weights/config.json",
    "models/training_manifest.json",
    "weights/training_manifest.json",
    "backend/models/training_manifest.json",
    "backend/weights/training_manifest.json",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_file(candidates: list) -> str | None:
    """Return the first existing path from a list of candidates."""
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _load_manifest() -> dict:
    """
    Load training_manifest.json if available.
    Returns default dims if not found.
    """
    path = _find_file(_MANIFEST_PATHS)
    defaults = {
        "visual_dim": 128,
        "audio_dim":  128,
        "embed_dim":  128,
        "num_heads":  4,
    }
    if path is None:
        logger.info("ℹ️  No training_manifest.json found — using default architecture dims.")
        return defaults
    try:
        with open(path) as f:
            manifest = json.load(f)
        # Merge with defaults for any missing keys
        for k, v in defaults.items():
            manifest.setdefault(k, v)
        logger.info(f"📋 Loaded training manifest from {path}")
        logger.info(f"   Visual Dim: {manifest['visual_dim']} | Audio Dim: {manifest['audio_dim']}"
                    f" | Embed Dim: {manifest['embed_dim']} | Heads: {manifest['num_heads']}")
        if "best_val_auc" in manifest:
            logger.info(f"   Best Val AUC: {manifest['best_val_auc']:.4f} @ epoch {manifest.get('best_epoch', '?')}")
        return manifest
    except Exception as e:
        logger.warning(f"⚠️  Failed to read manifest ({e}). Using default dims.")
        return defaults


def _remap_state_dict(state: dict, prefix_to_remove: str = "") -> dict:
    """
    Handles state dicts that were saved from a sub-module (e.g. model.visual_branch).
    If keys have a prefix like 'visual_branch.' strip it.
    Also handles the inverse case where keys are missing expected prefixes.
    """
    if not state:
        return state
    first_key = next(iter(state))
    if prefix_to_remove and first_key.startswith(prefix_to_remove):
        return {k[len(prefix_to_remove):]: v for k, v in state.items()}
    return state


def _validate_model(model, name: str, dummy_inputs: tuple, expected_shape: tuple) -> bool:
    """Run a dummy forward pass and verify output shape."""
    try:
        model.eval()
        with torch.no_grad():
            out = model(*[x.to(device) for x in dummy_inputs])
            # Handle tuple outputs (e.g. fusion returns logits, attn_weights)
            if isinstance(out, tuple):
                out = out[0]
        assert out.shape == expected_shape, (
            f"{name} output shape mismatch: expected {expected_shape}, got {out.shape}"
        )
        logger.info(f"   ✅ {name} shape validation passed: {out.shape}")
        return True
    except Exception as e:
        logger.warning(f"   ⚠️  {name} shape validation failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Registry state
# ─────────────────────────────────────────────────────────────────────────────

_registry_status = {
    "visual_loaded":  False,
    "audio_loaded":   False,
    "fusion_loaded":  False,
    "manifest_loaded": False,
    "visual_path":    None,
    "audio_path":     None,
    "fusion_path":    None,
    "manifest":       None,
}


def get_registry_status() -> dict:
    """Return the current loading status of all models."""
    return dict(_registry_status)


def is_trained() -> bool:
    """True if all three model weight files were successfully loaded."""
    return (
        _registry_status["visual_loaded"]
        and _registry_status["audio_loaded"]
        and _registry_status["fusion_loaded"]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main loader — called at backend startup
# ─────────────────────────────────────────────────────────────────────────────

def load_all_models(visual_model, audio_model, fusion_model):
    """
    Load trained weights into the three pre-instantiated model objects.
    
    Parameters
    ----------
    visual_model : Visual3DCNN      — already instantiated and on device
    audio_model  : AudioSpectrogramCNN — already instantiated and on device
    fusion_model : CrossModalAttentionFusion — already instantiated and on device
    
    All models are modified in-place.
    """
    logger.info("=" * 65)
    logger.info("  🔬 Forensic.AI — Model Registry Startup")
    logger.info("=" * 65)

    # 1. Read manifest (may contain exact dims from training run)
    manifest = _load_manifest()
    _registry_status["manifest"] = manifest
    _registry_status["manifest_loaded"] = True

    # 2. Load Visual3DCNN weights
    visual_path = _find_file(_VISUAL_PATHS)
    if visual_path:
        try:
            state = torch.load(visual_path, map_location=device, weights_only=True)
            # State dicts saved directly from sub-module have clean keys
            # (e.g., 'conv1.weight'). Load with strict=False for safety.
            visual_model.load_state_dict(state, strict=False)
            _registry_status["visual_loaded"] = True
            _registry_status["visual_path"] = visual_path
            logger.info(f"  ✅ Visual3DCNN weights loaded from: {visual_path}")
        except Exception as e:
            logger.warning(f"  ⚠️  Visual weights load failed ({e}) — using random init")
    else:
        logger.warning("  ℹ️  Visual backbone weights not found — using random init")
        logger.warning("      → Copy efficientnet_backbone.pth from Kaggle to backend/models/")

    # 3. Load AudioSpectrogramCNN weights
    audio_path = _find_file(_AUDIO_PATHS)
    if audio_path:
        try:
            state = torch.load(audio_path, map_location=device, weights_only=True)
            audio_model.load_state_dict(state, strict=False)
            _registry_status["audio_loaded"] = True
            _registry_status["audio_path"] = audio_path
            logger.info(f"  ✅ AudioSpectrogramCNN weights loaded from: {audio_path}")
        except Exception as e:
            logger.warning(f"  ⚠️  Audio weights load failed ({e}) — using random init")
    else:
        logger.warning("  ℹ️  Audio backbone weights not found — using random init")
        logger.warning("      → Copy audio_backbone.pth from Kaggle to backend/models/")

    # 4. Load CrossModalAttentionFusion weights
    fusion_path = _find_file(_FUSION_PATHS)
    if fusion_path:
        try:
            state = torch.load(fusion_path, map_location=device, weights_only=True)
            fusion_model.load_state_dict(state, strict=False)
            _registry_status["fusion_loaded"] = True
            _registry_status["fusion_path"] = fusion_path
            logger.info(f"  ✅ CrossModalAttentionFusion weights loaded from: {fusion_path}")
        except Exception as e:
            logger.warning(f"  ⚠️  Fusion weights load failed ({e}) — using random init")
    else:
        logger.warning("  ℹ️  Fusion layer weights not found — using random init")
        logger.warning("      → Copy deepfake_detector_cross_modality.pt from Kaggle to backend/models/")

    # 5. Shape validation
    logger.info("\n  🔍 Running shape validation...")

    visual_model.eval()
    audio_model.eval()
    fusion_model.eval()

    _validate_model(
        visual_model, "Visual3DCNN",
        dummy_inputs=(torch.zeros(1, 3, 12, 112, 112),),
        expected_shape=(1, 2)
    )
    _validate_model(
        audio_model, "AudioSpectrogramCNN",
        dummy_inputs=(torch.zeros(1, 40, 79),),
        expected_shape=(1, 2)
    )
    # Fusion requires both feature sequences
    with torch.no_grad():
        vis_seq = visual_model.extract_sequence_features(torch.zeros(1, 3, 12, 112, 112).to(device))
        aud_seq = audio_model.extract_sequence_features(torch.zeros(1, 40, 79).to(device))
    _validate_model(
        fusion_model, "CrossModalAttentionFusion",
        dummy_inputs=(vis_seq, aud_seq),
        expected_shape=(1, 2)
    )

    # 6. Final status report
    logger.info("\n" + "=" * 65)
    if is_trained():
        logger.info("  🚀 ALL MODELS LOADED WITH TRAINED WEIGHTS — Production Ready")
    else:
        loaded = [
            k.replace("_loaded", "").title()
            for k, v in _registry_status.items()
            if k.endswith("_loaded") and k != "manifest_loaded" and v
        ]
        missing = [
            k.replace("_loaded", "").title()
            for k, v in _registry_status.items()
            if k.endswith("_loaded") and k != "manifest_loaded" and not v
        ]
        if loaded:
            logger.info(f"  ✅ Loaded: {', '.join(loaded)}")
        if missing:
            logger.warning(f"  ⚠️  Random init: {', '.join(missing)}")
        logger.warning("     → Train on Kaggle and copy weights to backend/models/")
    logger.info("=" * 65 + "\n")
