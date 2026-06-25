import sys
import os
import torch
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    print("🧪 Importing fusion modules...")
    from services.fusion import (
        CrossModalAttentionFusion,
        cross_modal_attention_model,
        perform_attention_lip_sync_fusion
    )
    print("✅ Successfully imported Cross-Modal fusion modules!")
    
    print("🧪 Verifying model forward pass with mock sequences...")
    # Mock visual sequence: [Batch=1, Frames=8, Visual_Dim=128]
    # Mock audio sequence: [Batch=1, Time=8, Audio_Dim=128]
    visual_seq = torch.randn(1, 8, 128)
    audio_seq = torch.randn(1, 8, 128)
    
    logits, attn_weights = cross_modal_attention_model(visual_seq, audio_seq)
    print(f"✅ Forward pass completed! Logits shape: {logits.shape}, Attention weights shape: {attn_weights.shape}")
    
    print("🧪 Verifying interface function perform_attention_lip_sync_fusion...")
    # Mock visual PIL frames (8 frames of size 128x128)
    mock_frames = [Image.new("RGB", (128, 128), color=(100, 150, 200)) for _ in range(8)]
    # Mock audio waveform
    mock_waveform = torch.randn(1, 16000 * 2) # 2 seconds of audio at 16kHz
    
    res = perform_attention_lip_sync_fusion(mock_frames, mock_waveform)
    print(f"✅ Lip-sync fusion completed successfully!")
    print(f"📊 Results: {res}")
    
    print("\n🎉 ALL CROSS-MODAL MODEL TESTS PASSED SUCCESSFULLY!")
    sys.exit(0)
    
except Exception as e:
    print(f"❌ Test Failed: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
