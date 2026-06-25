# 🧬 Kaggle Neural Network Weights Registry

This directory serves as the centralized repository for your custom pre-trained deep learning weights imported from **Kaggle** (or local training sessions). 

Both the **Visual (EfficientNet)** and **Auditory (Spectrogram CNN)** modalities are designed to scan this folder automatically upon startup.

---

## 📂 Expected Weights & Filenames

Please place your `.pth` or `.pt` weight files here using the exact filenames listed below:

### 1. 👁️ Visual Deepfake Detection Backbone
*   **Target Filename:** `efficientnet_backbone.pth`
*   **Model Architecture:** PyTorch `torchvision.models.efficientnet_b0` (with custom 2-class classifier output).
*   **Tensor Output Shape:** `[Batch, 2]` logits where class `0` represents `REAL` and class `1` represents `FAKE`.

### 2. 🔊 Auditory Deepfake Detection Backbone
*   **Target Filename:** `audio_backbone.pth`
*   **Model Architecture:** 2D Spectrogram Convolutional Neural Network (defined in [audio_model.py](file:///d:/Deepfake%20Detection/backend/models/audio_model.py)).
*   **Tensor Input Shape:** `[Batch, 1, Mel_Bins, Time_Frames]` 
*   **Tensor Output Shape:** `[Batch, 2]` logits where class `0` represents `REAL` and class `1` represents `FAKE`.

---

## 💡 Automated Scanning Mechanics

The backend deep learning initialization routines look for weights in the following order:
1.  `backend/weights/` (This folder - Recommended for all new models)
2.  `backend/models/`
3.  Fallback to unweighted parameter initializations with **deterministic analytical routines** (so that your application remains fully functional even if weights are not yet present).

### ⚡ Troubleshooting Load Mappings
When saving your PyTorch models from Kaggle (which likely use GPUs/CUDA), ensure you save the state dict:
```python
# On Kaggle
torch.save(model.state_dict(), 'efficientnet_backbone.pth')
```

The server automatically handles mapping the tensors back to the appropriate local device (either CPU or CUDA/GPU) using:
```python
state = torch.load(path, map_location=device)
```
