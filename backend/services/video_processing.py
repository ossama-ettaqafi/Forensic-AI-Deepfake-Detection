import os
import cv2
import logging
import subprocess
import torch
from PIL import Image
from facenet_pytorch import MTCNN

logger = logging.getLogger("deepfake-api")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize MTCNN for high-performance facial extraction matching notebook exactly
try:
    mtcnn = MTCNN(
        image_size=112, 
        margin=0, 
        min_face_size=30, 
        thresholds=[0.6, 0.7, 0.7], 
        factor=0.709, 
        post_process=False, 
        device=device
    )
    logger.info("✅ MTCNN face extractor initialized successfully with notebook parameters.")
except Exception as e:
    mtcnn = None
    logger.warning(f"⚠️ Failed to initialize MTCNN on {device}, using raw frame fallback: {str(e)}")

def extract_video_frames(video_path, max_frames=12):
    """
    Extracts frames from video_path at equidistant intervals.
    Returns a list of image paths or cropped face PIL Images.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"❌ Could not open video file: {video_path}")
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / max(1.0, fps)
    
    logger.info(f"📹 Processing video: {total_frames} total frames, {fps:.1f} FPS, Duration: {duration:.2f}s")
    
    # Calculate spacing to extract max_frames evenly
    interval = max(1, total_frames // max_frames)
    
    count = 0
    extracted_count = 0
    
    while cap.isOpened() and extracted_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % interval == 0:
            # Convert BGR (OpenCV) to RGB (PIL)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            
            # Extract face if MTCNN is active
            face_img = None
            if mtcnn is not None:
                try:
                    # Detect and crop via PIL matching notebook exactly
                    boxes, _ = mtcnn.detect(pil_img)
                    if boxes is not None and len(boxes) > 0:
                        box = boxes[0].astype(int)
                        w, h = pil_img.size
                        x1, y1 = max(0, box[0]), max(0, box[1])
                        x2, y2 = min(w, box[2]), min(h, box[3])
                        face_img = pil_img.crop((x1, y1, x2, y2))
                        logger.info(f"🧑 Face detected in video frame {count}")
                except Exception as e:
                    logger.warning(f"⚠️ Face detection error in frame {count}: {str(e)}")
            
            # Fallback to full resized frame if no face is found
            if face_img is None:
                face_img = pil_img.resize((112, 112))
                
            frames.append(face_img)
            extracted_count += 1
            
        count += 1
        
    cap.release()
    logger.info(f"✅ Extracted {len(frames)} frames from video.")
    return frames

def extract_audio_from_video(video_path, output_audio_path):
    """
    Extracts the audio track from a video and saves it as a WAV file using FFmpeg.
    If FFmpeg fails or is missing, implements a silent fallback.
    Returns True if audio was successfully extracted, False otherwise.
    """
    if not os.path.exists(video_path):
        return False
        
    # Command to extract mono WAV audio stream at 16kHz
    cmd = [
        "ffmpeg",
        "-y",               # Overwrite output
        "-i", video_path,   # Input video
        "-vn",              # Disable video
        "-ac", "1",         # Mono channel
        "-ar", "16000",     # 16kHz sample rate
        output_audio_path
    ]
    
    try:
        # Run process in background, suppress stdout
        logger.info(f"🎵 Extracting audio from {video_path} using ffmpeg...")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
            logger.info(f"✅ Audio successfully extracted: {output_audio_path}")
            return True
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.warning(f"⚠️ FFmpeg not available or failed: {str(e)}. Using synthetic audio stream emulator.")
        
    # Synthetic/Fallback: Create an empty wave file or let the analyzer run on mock signal
    # to avoid crashing and show standard results
    return False
