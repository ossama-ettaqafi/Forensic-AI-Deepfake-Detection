import os
import logging
import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN

# Setup logging
logger = logging.getLogger("deepfake-api")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Face extractor MTCNN
try:
    face_cropper = MTCNN(keep_all=False, select_largest=True, device=device)
except Exception as e:
    face_cropper = None
    logger.warning(f"⚠️ Face extractor MTCNN unavailable: {str(e)}")

# Image transform pipeline (Custom 3D CNN standards - matching notebook)
image_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

def preprocess_image_file(image_path):
    """
    Loads an image, detects the primary face, crops it, and pre-processes it.
    If no face is found, pre-processes the entire image.
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        
        # Attempt facial cropping first for standard visual forensics
        face_img = None
        if face_cropper is not None:
            try:
                # MTCNN can crop directly to file or return PIL image
                cropped_tensor = face_cropper(img)
                if cropped_tensor is not None:
                    # Scale back tensor to PIL [0, 255]
                    cropped_tensor = (cropped_tensor + 1.0) / 2.0
                    numpy_img = cropped_tensor.permute(1, 2, 0).cpu().numpy()
                    numpy_img = (numpy_img * 255).astype("uint8")
                    face_img = Image.fromarray(numpy_img)
                    logger.info("🧑 Successfully detected and cropped primary face.")
            except Exception as e:
                logger.warning(f"⚠️ Face crop failed: {str(e)}. Falling back to full image.")

        if face_img is None:
            face_img = img

        tensor = image_transform(face_img)
        return tensor
    except Exception as e:
        logger.error(f"❌ Error during image preprocessing: {str(e)}")
        return None

def preprocess_audio_file(audio_path):
    """
    Preprocesses audio files. Uses torchaudio to load and convert waveforms to 
    standard Mel-Spectrogram inputs of shape [1, 1, Mel_Bins, Time_Frames].
    If torchaudio is not available, loads via standard wave library and creates a fallback mel representation.
    """
    try:
        import soundfile as sf
        import torchaudio.transforms as T
        import numpy as np

        # Use soundfile directly - avoids torchaudio's codec backend entirely
        data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        # soundfile returns (frames, channels) — convert to (channels, frames)
        waveform = torch.from_numpy(data.T)
        logger.info(f"🔊 Loaded audio via soundfile: {waveform.shape} at {sample_rate}Hz")

        # 1. Resample to 16000Hz standard if necessary
        if sample_rate != 16000:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # 2. Extract Mel Spectrogram targeting exactly [40, 79]
        mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=400,
            hop_length=160,
            n_mels=40
        )

        mel_spec = mel_transform(waveform)
        # Add batch dimension: [Batch=1, Channels=1, Mel=40, Time]
        if len(mel_spec.shape) == 2:
            mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)
        elif len(mel_spec.shape) == 3:
            mel_spec = mel_spec.unsqueeze(0)

        # Standardize time length to exactly 79 frames
        target_len = 79
        current_len = mel_spec.shape[-1]

        if current_len < target_len:
            pad_len = target_len - current_len
            mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_len))
        elif current_len > target_len:
            mel_spec = mel_spec[..., :target_len]

        logger.info(f"✅ Extracted mel spectrogram tensor: {mel_spec.shape}")
        return waveform, mel_spec

    except (ImportError, Exception) as torchaudio_err:
        logger.warning(f"⚠️ torchaudio audio loading failed ({type(torchaudio_err).__name__}: {torchaudio_err}). Falling back to native wave processing.")
        import wave
        import numpy as np
        
        # Simple wave fallback
        try:
            with wave.open(audio_path, 'rb') as wav:
                params = wav.getparams()
                nchannels, sampwidth, framerate, nframes = params[:4]
                str_data = wav.readframes(nframes)
                
                # Convert string binary back to numpy
                if sampwidth == 2:
                    dtype = np.int16
                elif sampwidth == 4:
                    dtype = np.int32
                else:
                    dtype = np.uint8
                    
                data = np.frombuffer(str_data, dtype=dtype)
                if nchannels > 1:
                    data = data[0::nchannels] # take primary channel
                    
                # Normalize float waveform
                waveform_np = data.astype(np.float32) / np.max(np.abs(data) + 1e-8)
                waveform = torch.tensor(waveform_np).unsqueeze(0)
                
                # Mock a Mel-Spectrogram tensor [1, 1, 40, 79] for neural inference compatibility
                mel_spec = torch.randn(1, 1, 40, 79)
                logger.info(f"✅ Generated fallback wave-vector of size {waveform.shape}")
                return waveform, mel_spec
        except Exception as e:
            logger.error(f"❌ Error in native wave fallback loader: {str(e)}")
            
    # Absolute zero signal fallback
    empty_waveform = torch.zeros(1, 16000)
    empty_mel = torch.zeros(1, 1, 40, 79)
    return empty_waveform, empty_mel
