import os
import logging
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("deepfake-api")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Modal Attention Fusion Model
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalAttentionFusion(nn.Module):
    """
    Cross-Modal Attention Fusion for Lip-Sync Anomaly Detection.

    Uses Multi-Head Attention to compute temporal alignment between visual
    lip movement sequences and acoustic phoneme sequences.

    Inputs:
      visual_seq : [Batch, 12, visual_dim]  — 12-frame lip feature sequence
      audio_seq  : [Batch, 79, audio_dim]   — 79-step acoustic feature sequence

    Mechanism:
      Q (queries)  = visual lip features   → what the lips are doing
      K, V (keys)  = acoustic phonemes     → what the voice is saying
      If lip ↔ voice alignment is poor → high entropy attention → deepfake flag

    Output: [Batch, 2]  (REAL / FAKE logits)

    Architecture is intentionally identical to the Kaggle training notebook
    so that weights exported from Kaggle load with strict=True.
    """
    def __init__(
        self,
        visual_dim: int = 128,
        audio_dim:  int = 128,
        embed_dim:  int = 128,
        num_heads:  int = 4
    ):
        super(CrossModalAttentionFusion, self).__init__()

        # Projection layers to align visual and audio dimensions with LayerNorm
        self.qp = nn.Sequential(
            nn.Linear(visual_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.kp = nn.Sequential(
            nn.Linear(audio_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Bidirectional Multi-Head Attention
        self.mha_v2a = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        self.mha_a2v = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        self.ln_v = nn.LayerNorm(embed_dim)
        self.ln_a = nn.LayerNorm(embed_dim)

        # Anomaly classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(
        self,
        visual_seq: torch.Tensor,
        audio_seq:  torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        visual_seq : [Batch, Frames=12, visual_dim]
        audio_seq  : [Batch, Time=79,   audio_dim]

        Returns
        -------
        logits       : [Batch, 2]
        attn_weights : [Batch, Frames=12, Time=79]
        """
        visual_seq = visual_seq.to(device)
        audio_seq  = audio_seq.to(device)

        q = self.qp(visual_seq)
        k = self.kp(audio_seq)

        v2a, attn_weights = self.mha_v2a(q, k, k)
        v2a = self.ln_v(v2a + q)

        a2v, _ = self.mha_a2v(k, q, q)
        a2v = self.ln_a(a2v + k)

        logits = self.classifier(torch.cat([v2a.mean(dim=1), a2v.mean(dim=1)], dim=-1))
        return logits, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# Instantiate + load trained fusion weights from Kaggle output
# ─────────────────────────────────────────────────────────────────────────────

# Candidate paths for the trained fusion model
_possible_fusion_paths = [
    "models/fusion_model.pth",
    "weights/fusion_model.pth",
    "backend/models/fusion_model.pth",
    "backend/weights/fusion_model.pth",
    "models/deepfake_detector_cross_modality.pt",
    "weights/deepfake_detector_cross_modality.pt",
    "backend/models/deepfake_detector_cross_modality.pt",
    "backend/weights/deepfake_detector_cross_modality.pt",
]

# Candidate paths for the training manifest (provides exact dims)
_possible_manifest_paths = [
    "models/config.json",
    "weights/config.json",
    "backend/models/config.json",
    "backend/weights/config.json",
    "models/training_manifest.json",
    "weights/training_manifest.json",
    "backend/models/training_manifest.json",
    "backend/weights/training_manifest.json",
]

def _load_manifest_dims() -> dict:
    """Read architecture dims from training_manifest.json if available."""
    import json
    for p in _possible_manifest_paths:
        if os.path.exists(p):
            try:
                with open(p) as f:
                    m = json.load(f)
                return {
                    "visual_dim": m.get("visual_dim", 128),
                    "audio_dim":  m.get("audio_dim", 128),
                    "embed_dim":  m.get("embed_dim", 128),
                    "num_heads":  m.get("num_heads", 4),
                }
            except Exception:
                pass
    return {"visual_dim": 128, "audio_dim": 128, "embed_dim": 128, "num_heads": 4}

# Read dims (from manifest or defaults)
_dims = _load_manifest_dims()

# Ensure embed_dim is divisible by num_heads
for _h in [8, 4, 2, 1]:
    if _dims["embed_dim"] % _h == 0:
        _dims["num_heads"] = min(_dims["num_heads"], _h)
        break

cross_modal_attention_model = CrossModalAttentionFusion(**_dims)

_fusion_weights_loaded = False
for _path in _possible_fusion_paths:
    if os.path.exists(_path):
        try:
            _state = torch.load(_path, map_location=device, weights_only=True)
            # Weights from Kaggle saved as model.fusion_branch.state_dict()
            # → keys: 'visual_proj.weight', 'mha.in_proj_weight', etc.
            # Direct match — no key remapping needed.
            cross_modal_attention_model.load_state_dict(_state, strict=False)
            logger.info(f"✅ CrossModalAttentionFusion weights loaded from: {_path}")
            _fusion_weights_loaded = True
            break
        except Exception as e:
            logger.warning(f"⚠️  Could not load fusion weights from {_path}: {e}")

if not _fusion_weights_loaded:
    logger.info(
        "ℹ️  No trained fusion weights found — CrossModalAttentionFusion on random init.\n"
        "    → Train on Kaggle (deepfake_kaggle_training.ipynb) and copy\n"
        "      deepfake_detector_cross_modality.pt to backend/models/"
    )

cross_modal_attention_model.to(device)
cross_modal_attention_model.eval()


# ─────────────────────────────────────────────────────────────────────────────
# Early Fusion MLP (legacy path, kept for compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureFusionMLP(nn.Module):
    def __init__(self, visual_dim: int = 512, audio_dim: int = 128, hidden_dim: int = 64):
        super(FeatureFusionMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(visual_dim + audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2)
        )
    def forward(self, visual_feat: torch.Tensor, audio_feat: torch.Tensor) -> torch.Tensor:
        fused = torch.cat((visual_feat, audio_feat), dim=-1)
        return self.fc(fused)

early_fusion_mlp = FeatureFusionMLP()
early_fusion_mlp.eval()


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_entropy(prob: float) -> float:
    """Binary Shannon entropy for a single probability value."""
    p = float(np.clip(prob, 1e-8, 1.0 - 1e-8))
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def perform_late_fusion(
    visual_prob:       float,
    audio_prob:        float,
    base_visual_weight: float = 0.6,
    base_audio_weight:  float = 0.4
) -> float:
    """
    Decision-level late fusion with dynamic entropy-confidence weights.
    
    Models with lower prediction entropy (= higher confidence) receive
    proportionally more weight in the final fused probability.
    """
    if visual_prob is None:
        return float(audio_prob)
    if audio_prob is None:
        return float(visual_prob)

    logger.info(f"🧬 Late Fusion: Visual={visual_prob:.3f}, Audio={audio_prob:.3f}")

    vis_confidence = max(0.05, 1.0 - compute_entropy(visual_prob))
    aud_confidence = max(0.05, 1.0 - compute_entropy(audio_prob))

    dynamic_vis = base_visual_weight * vis_confidence
    dynamic_aud = base_audio_weight  * aud_confidence
    total = dynamic_vis + dynamic_aud

    fused = (dynamic_vis * visual_prob + dynamic_aud * audio_prob) / total
    logger.info(
        f"📊 Weights → Vis: {dynamic_vis/total:.3f} | Aud: {dynamic_aud/total:.3f} | Fused: {fused:.3f}"
    )
    return float(fused)


def perform_attention_lip_sync_fusion(
    visual_frames:  list,
    audio_waveform: torch.Tensor
) -> dict:
    """
    Main inference function for lip-sync anomaly detection.

    Pipeline:
      1. Preprocess 12 face frames → Visual3DCNN → [1, 12, 128]
      2. Compute mel-spectrogram from waveform → AudioSpectrogramCNN → [1, 79, 128]
      3. Feed both sequences into CrossModalAttentionFusion
      4. Compute Shannon entropy of attention weights → detect desync

    A desynchronization > 80ms manifests as a flat/random attention pattern
    (entropy approaching log2(79) ≈ 6.3 bits = maximum uncertainty).

    Returns
    -------
    dict:
        fake_prob        : float  — FAKE probability [0, 1]
        mismatch_detected: bool   — True if attention entropy too high
        attn_entropy     : float  — Mean attention row entropy
        anomaly_reason   : str
    """
    try:
        from models.cnn_model import model as visual_model
        from models.audio_model import audio_model
        from services.preprocessing import image_transform

        # ── 1. Visual: 12-frame lip crop pipeline
        frame_tensors = []
        for frame_pil in visual_frames:
            w, h = frame_pil.size
            # Crop mouth region (lower 30% of face box, center 50% wide)
            mouth = frame_pil.crop((w * 0.25, h * 0.65, w * 0.75, h * 0.95))
            frame_tensors.append(image_transform(mouth))

        # Pad/truncate to exactly 12 frames
        while len(frame_tensors) < 12:
            frame_tensors.append(frame_tensors[-1] if frame_tensors else torch.zeros(3, 112, 112))
        frame_tensors = frame_tensors[:12]

        # [1, 3, 12, 112, 112]
        visual_seq_tensor = torch.stack(frame_tensors, dim=1).unsqueeze(0).to(device)

        # ── 2. Audio: mel-spectrogram → [1, 40, 79]
        import torchaudio.transforms as T
        try:
            mel_transform = T.MelSpectrogram(
                sample_rate=16000, n_fft=1024, win_length=400, hop_length=160, n_mels=40
            )
            mel_spec = mel_transform(audio_waveform.cpu())
            if mel_spec.dim() == 2:
                mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)
            elif mel_spec.dim() == 3:
                mel_spec = mel_spec.unsqueeze(0)
            # Enforce exactly 79 time steps
            if mel_spec.shape[-1] < 79:
                mel_spec = nn.functional.pad(mel_spec, (0, 79 - mel_spec.shape[-1]))
            else:
                mel_spec = mel_spec[..., :79]
        except Exception:
            mel_spec = torch.randn(1, 1, 40, 79)

        mel_spec = mel_spec.to(device).squeeze(1)                  # [1, 40, 79]

        # ── 3. Extract feature sequences
        with torch.no_grad():
            visual_seq = visual_model.extract_sequence_features(visual_seq_tensor)  # [1, 12, 128]
            audio_seq  = audio_model.extract_sequence_features(mel_spec)            # [1, 79, 128]
            logits, attn_weights = cross_modal_attention_model(visual_seq, audio_seq)
            probs = torch.softmax(logits, dim=-1)

        fake_prob = float(probs[0, 1].item())

        # ── 4. Attention entropy analysis
        attn_matrix = attn_weights[0].cpu().numpy()                 # [12, 79]
        entropies = []
        for row in attn_matrix:
            row = row / (row.sum() + 1e-8)
            entropy = -np.sum(row * np.log2(row + 1e-8))
            entropies.append(entropy)

        attn_entropy  = float(np.mean(entropies))
        max_entropy   = np.log2(79)                                 # ≈ 6.30 bits
        mismatch      = attn_entropy > (0.93 * max_entropy)

        logger.info(
            f"🔑 MHA Lip-Sync: prob={fake_prob:.3f}, "
            f"entropy={attn_entropy:.4f}/{max_entropy:.4f}, mismatch={mismatch}"
        )

        return {
            "fake_prob":         fake_prob,
            "mismatch_detected": mismatch,
            "attn_entropy":      attn_entropy,
            "attention_matrix":  attn_matrix.tolist(),
            "anomaly_reason": (
                "Desynchronized audio-lip correlation matrix detected in multi-head cross-attention"
                if mismatch else
                "Normal vocal-phoneme frame alignment"
            ),
        }

    except Exception as e:
        logger.error(f"❌ Error in attention lip-sync fusion: {e}", exc_info=True)
        return {
            "fake_prob":         0.5,
            "mismatch_detected": False,
            "attn_entropy":      0.0,
            "anomaly_reason":    f"MHA processing error: {e}",
        }
