<div align="center">

# 🔬 Forensic.AI — Deepfake Integrity Matrix

**A multimodal AI system for audio-visual deepfake detection**  
*Visual 3D CNN · Audio 1D CNN · Cross-Modal Attention Fusion · LLM Explainability*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Developed by **ETTAQAFI Ossama** · Supervised by **Pr. KARTIT Ali**

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Detection Pipelines](#-detection-pipelines)
- [Performance](#-performance)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Running the App](#-running-the-app)
- [API Reference](#-api-reference)
- [Tech Stack](#-tech-stack)

---

## 🧠 Overview

**Forensic.AI** is a full-stack deepfake detection platform that fuses visual and acoustic modalities to identify manipulated media with high accuracy. The system ingests a video file, extracts synchronized facial keyframes and Mel-spectrogram audio features, then fuses them via a **Multi-Head Cross-Attention (MHA)** mechanism to detect lip-sync mismatches and other AI-generated artifacts.

A secondary **LLM Explainability** layer (powered by Gemini 1.5 Pro) generates structured forensic reports in plain English, making complex AI decisions interpretable for non-technical stakeholders.

---

## 🏗 Architecture

```
Video Input
    │
    ├── Frame Extraction (MTCNN Face Detection) ──► Visual EfficientNet-B4
    │       12 synchronized facial keyframes              │
    │                                                     ▼
    │                                             Visual Feature Embeddings
    │                                             [Batch, 12, 128]
    │
    └── Audio Extraction (librosa) ──────────────► Audio 1D CNN
            79-frame Mel-spectrogram                     │
                                                         ▼
                                                 Audio Feature Embeddings
                                                 [Batch, 79, 128]
                                                         │
                                    ┌────────────────────┘
                                    │
                          Cross-Modal Attention Fusion (MHA)
                                    │
                                    ▼
                           Binary Classification
                           REAL / FAKE + Confidence
                                    │
                                    ▼
                           LLM Forensic Report
                           (Gemini 1.5 Pro API)
```

---

## ✨ Features

| Feature | Description |
|---|---|
| **Multimodal Fusion** | Simultaneously analyzes visual frames and audio waveforms |
| **Face Detection** | MTCNN-based facial crop and mouth region isolation |
| **Temporal Analysis** | Processes 12 synchronized keyframes per video |
| **Cross-Modal Attention** | Detects lip-sync mismatches via MHA fusion |
| **Early Exit** | Skips fusion at 92% confidence for faster inference |
| **LLM Reports** | Natural language forensic explanations via Gemini API |
| **Interactive UI** | Real-time analysis dashboard with animated results |
| **Model Registry** | Pre-loads all weights at server startup for zero-latency inference |

---

## 🔍 Detection Pipelines

### 🎯 Cross-Modal CNN Fusion *(Active)*

The primary production pipeline. Processes synchronized 12-frame facial sequences alongside 79-frame Mel-spectrogram audio tensors and fuses them via Multi-Head Cross-Attention.

- **Visual backbone**: `EfficientNet-B4` → projection to 128-dim embeddings
- **Audio backbone**: `1D CNN` → Mel-spectrogram feature extraction
- **Fusion**: `CrossModalAttentionFusion` (Multi-Head Attention, 4 heads, 128 dim)
- **Trained on**: DFDC & FakeAVCeleb datasets

### 🔬 Episodic Meta-Learning *(Under Development)*

Utilizes Few-Shot Test-Time Adaptation (TTA-MAML) to dynamically adjust classifier weights, improving robustness against unseen generative models and diverse camera qualities.

### 🤖 Cognitive LLM Agent *(Under Development)*

Combines localized computer vision metrics with biometrics, coordinated by an LLM reasoning agent to provide natural language explainability and cognitive fact-checking, powered by the Google Gemini 1.5 Pro API.

---

## 📊 Performance

Trained using **5-Fold StratifiedGroupKFold** with identity-disjoint splits on 400 videos.

| Metric | Score |
|---|---|
| **Mean AUC** | **0.9205** (95% CI: 0.874 – 0.959) |
| **Mean F1** | **0.9531** |
| **Mean Accuracy** | **0.9225** |
| **Best Fold AUC** | 0.9918 |

### Ablation Study

| Configuration | AUC |
|---|---|
| Visual Only | 0.9368 |
| Audio Only | 0.6744 |
| Late Fusion | 0.9396 |
| **Cross-Modal Attention (Full)** | **0.9317** |

> Architecture: `VACMA v3` · Protocol: StratifiedGroupKFold identity-disjoint · Bootstraps: 2000 · Timestamp: 2026-06-14

---

## 📁 Project Structure

```
Deepfake Detection/
│
├── backend/                     # FastAPI Python backend
│   ├── main.py                  # App entry point, CORS, model registry startup
│   ├── requirements.txt         # Python dependencies
│   ├── Dockerfile               # Backend container definition
│   │
│   ├── models/                  # Model definitions & trained weights
│   │   ├── cnn_model.py         # VisualEfficientNetSequence (EfficientNet-B4)
│   │   ├── audio_model.py       # 1D CNN for Mel-spectrogram features
│   │   ├── meta_learning.py     # Episodic meta-learning adapter
│   │   ├── model_registry.py    # Centralized weight loader
│   │   ├── best_visual.pth      # Trained visual model weights
│   │   ├── best_audio.pth       # Trained audio model weights
│   │   └── best_fusion.pth      # Trained fusion model weights
│   │
│   ├── routes/
│   │   └── analyze.py           # /analyze endpoint — full inference pipeline
│   │
│   ├── services/
│   │   ├── fusion.py            # CrossModalAttentionFusion (MHA)
│   │   ├── preprocessing.py     # Image & audio preprocessing
│   │   ├── video_processing.py  # Frame extraction & audio separation
│   │   ├── llm_explainability.py # Gemini API + fallback forensic report
│   │   └── inference.py         # Inference helpers
│   │
│   └── utils/
│       └── early_exit.py        # Confidence-based early exit scheduler
│
├── frontend/                    # React + Vite frontend
│   ├── index.html               # App shell
│   ├── package.json             # Node.js dependencies
│   ├── vite.config.js           # Vite configuration
│   ├── Dockerfile               # Frontend container definition
│   │
│   └── src/
│       ├── App.jsx              # Root layout — sidebar + workspace
│       ├── style.css            # Global design system & component styles
│       │
│       └── components/
│           ├── AnalysisSection.jsx  # Upload, inference trigger, results tab
│           ├── ResultCard.jsx       # Detailed forensic results display
│           ├── UploadBox.jsx        # Drag-and-drop video upload
│           ├── BackendSection.jsx   # "How it works" sidebar accordion
│           └── ...                  # Additional UI components
│
├── start.bat                    # One-click launcher (Windows)
├── requirements.txt             # Root-level Python dependencies
└── pyproject.toml               # Project metadata & tool configuration
```

---

## ✅ Prerequisites

### Backend
- **Python** 3.10+
- **CUDA** (optional, recommended for GPU inference)
- **FFmpeg** installed and accessible in `PATH` (required for audio extraction)

### Frontend
- **Node.js** 18+ and **npm**

### Optional
- **Google Gemini API Key** — for LLM-powered forensic report generation (falls back to a structured local report if not set)

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd "Deepfake Detection"
```

### 2. Backend setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

> **Note:** Install PyTorch with CUDA support for GPU inference:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

### 3. Frontend setup

```bash
cd frontend
npm install
```

### 4. (Optional) Configure Gemini API Key

Set the environment variable before starting the backend:

```bash
# Windows PowerShell
$env:GEMINI_API_KEY = "your-api-key-here"

# macOS / Linux
export GEMINI_API_KEY="your-api-key-here"
```

If not set, the system automatically falls back to a local structured forensic report template.

---

## 🚀 Running the App

### Option A — One-click (Windows)

```batch
start.bat
```

This launches both the backend and frontend in separate terminal windows automatically.

### Option B — Manual

**Terminal 1 — Backend:**
```bash
cd backend
venv\Scripts\activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

### Access the App

| Service | URL |
|---|---|
| **Frontend UI** | http://localhost:5173 |
| **Backend API** | http://localhost:8000 |
| **API Docs (Swagger)** | http://localhost:8000/docs |

---

## 📡 API Reference

### `GET /`
Returns backend health status and model loading state.

**Response:**
```json
{
  "status": "backend running",
  "models_trained": true,
  "visual_weights": "models/best_visual.pth",
  "audio_weights": "models/best_audio.pth",
  "fusion_weights": "models/best_fusion.pth"
}
```

---

### `GET /model-status`
Returns detailed status of all loaded model components.

**Response:**
```json
{
  "is_fully_trained": true,
  "visual_loaded": true,
  "audio_loaded": true,
  "fusion_loaded": true,
  "training_info": {
    "best_val_auc": 0.9918,
    "best_epoch": 42,
    "timestamp": "2026-06-14 01:40:29"
  }
}
```

---

### `POST /analyze`
Analyzes a video file for deepfake content.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | `File` | ✅ | Video file to analyze (`.mp4`, `.avi`, `.mov`, etc.) |
| `mode` | `string` | ✅ | Detection mode: `cross_modal_cnn` |
| `prompt` | `string` | ❌ | Custom Gemini analysis prompt |
| `enable_early_exit` | `bool` | ❌ | Enable confidence-based early exit (default: `true`) |

**Response:**
```json
{
  "prediction": "FAKE",
  "confidence": 0.94,
  "visual_score": 0.91,
  "audio_score": 0.88,
  "early_exit_triggered": false,
  "visual_anomalies": ["Temporal inconsistency in periorbital region"],
  "audio_anomalies": ["Phase discontinuity at 2.3kHz"],
  "forensic_report": "## Forensic Analysis Report\n..."
}
```

---

## 🛠 Tech Stack

### Backend
| Library | Purpose |
|---|---|
| **FastAPI** | REST API framework |
| **PyTorch** | Deep learning inference |
| **EfficientNet-PyTorch** | Visual backbone (EfficientNet-B4) |
| **facenet-pytorch** | MTCNN face detection |
| **OpenCV** | Video frame extraction |
| **torchaudio** | Audio processing & Mel-spectrograms |
| **timm** | Additional model utilities |
| **google-generativeai** | Gemini API for LLM reports |

### Frontend
| Library | Purpose |
|---|---|
| **React 19** | UI framework |
| **Vite** | Build tool & dev server |
| **Framer Motion** | Animations & transitions |
| **Axios** | HTTP client for API calls |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ by **ETTAQAFI Ossama** · Master's Final Year Project (PFE) · 2026

</div>
