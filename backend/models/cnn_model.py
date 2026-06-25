import os
import logging
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

logger = logging.getLogger("deepfake-api")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VisualEfficientNetSequence(nn.Module):
    """
    EfficientNet-B4 Visual Sequence Feature Extractor for spatio-temporal deepfake forensic analysis.
    
    Inputs:
      x : [Batch, Channels=3, Depth=12, Height=112, Width=112] (video sequence of 12 frames)
    
    Outputs:
      logits : [Batch, 2] (REAL / FAKE logits via classification head)
      
    Feature extractor path:
      extract_sequence_features() → [Batch, 12, 128]
      → feeds CrossModalAttentionFusion in backend/services/fusion.py
    """
    def __init__(self, model_name: str = 'efficientnet-b4', visual_dim: int = 128):
        super(VisualEfficientNetSequence, self).__init__()
        
        logger.info(f"🧬 Initializing Visual Model with {model_name} backbone...")
        # Load architecture configuration without downloading ImageNet weights
        self.backbone = EfficientNet.from_name(model_name)
        num_ftrs = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()  # Remove original classification head
        
        # Projection layer to align features with multi-modal embed_dim (128)
        self.proj = nn.Sequential(
            nn.Linear(num_ftrs, visual_dim),
            nn.BatchNorm1d(visual_dim),
            nn.ReLU()
        )
        
        # Temporal attention layer
        self.temporal_attn = nn.Linear(visual_dim, 1)
        
        # Binary classification fallback head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(visual_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def extract_sequence_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts features for each frame in the sequence using the EfficientNet backbone.
        
        Input shape:  [Batch, 3, 12, 112, 112] (or [Batch, 3, 112, 112] frame)
        Output shape: [Batch, 12, visual_dim] (or [Batch, 1, visual_dim])
        """
        x = x.to(device)
        if len(x.shape) == 4:
            x = x.unsqueeze(2)  # [B, C, 1, H, W]
            
        batch_size, channels, depth, height, width = x.shape
        
        # Reshape to [Batch * Depth, Channels, Height, Width] to run frame-by-frame
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [Batch, Depth, Channels, Height, Width]
        x = x.view(batch_size * depth, channels, height, width)
        
        # Extract features
        features = self.backbone(x)  # [Batch * Depth, 1792]
        features = self.proj(features)  # [Batch * Depth, visual_dim]
        
        # Reshape back to [Batch, Depth, visual_dim]
        features = features.view(batch_size, depth, -1)  # [Batch, 12, visual_dim]
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass for binary classification.
        """
        f = self.extract_sequence_features(x)  # [Batch, 12, visual_dim]
        w = torch.softmax(self.temporal_attn(f), dim=1)
        return self.classifier((f * w).sum(dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Instantiate + load weights
# ─────────────────────────────────────────────────────────────────────────────

model = VisualEfficientNetSequence()

_possible_paths = [
    "models/visual_model.pth",
    "weights/visual_model.pth",
    "backend/models/visual_model.pth",
    "backend/weights/visual_model.pth",
    "models/efficientnet_backbone.pth",
    "weights/efficientnet_backbone.pth",
    "backend/models/efficientnet_backbone.pth",
    "backend/weights/efficientnet_backbone.pth",
]

_weights_loaded = False
for _path in _possible_paths:
    if os.path.exists(_path):
        try:
            _state = torch.load(_path, map_location=device, weights_only=True)
            model.load_state_dict(_state, strict=False)
            logger.info(f"✅ VisualEfficientNetSequence weights loaded from: {_path}")
            _weights_loaded = True
            break
        except Exception as e:
            logger.warning(f"⚠️  Could not load visual weights from {_path}: {e}")

if not _weights_loaded:
    logger.info(
        "ℹ️  No trained visual weights found — VisualEfficientNetSequence running on random init.\n"
        "    → Train on Kaggle and copy visual_model.pth to backend/models/"
    )

model.to(device)
model.eval()