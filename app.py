import os

# Set HF token BEFORE any huggingface imports (transformers imports huggingface_hub internally)
# This must be the first thing to override any Colab secrets
HF_TOKEN = ""
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

import torch
import librosa
import tempfile
import numpy as np
import gradio as gr
import soundfile as sf
from transformers import AutoModel, pipeline

# Only use spaces.GPU decorator on HuggingFace Spaces, not on Colab
try:
    import spaces
    gpu_decorator = spaces.GPU
except (ImportError, Exception):
    # On Colab/local, no decorator needed - GPU is already available
    gpu_decorator = lambda fn: fn

# Load Whisper for automatic transcription (optional, for when user doesn't provide ref text)
whisper_pipe = None
def get_whisper():
    global whisper_pipe
    if whisper_pipe is None:
        print("[INFO] Loading Whisper model for auto-transcription...")
        whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("[INFO] Whisper loaded successfully")
    return whisper_pipe

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    try:
        whisper = get_whisper()
        result = whisper(audio_path, generate_kwargs={"language": "ta"})  # Tamil
        return result["text"]
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
        return None

@gpu_decorator
def synthesize_speech(text, ref_audio, ref_text):
    print(f"[DEBUG] synthesize_speech called with text length: {len(text) if text else 0}")
    print(f"[DEBUG] ref_audio type: {type(ref_audio)}, ref_text length: {len(ref_text) if ref_text else 0}")
    
    if not text or text.strip() == "":
        print("[ERROR] No text provided")
        raise gr.Error("Please provide text to synthesize.")
    if len(text) > 500:
        print("[ERROR] Text too long")
        raise gr.Error("Text too long (max 500 characters).")
    if ref_audio is None:
        print("[ERROR] Missing reference audio")
        raise gr.Error("Please provide a reference audio.")
    
    # Ensure valid reference audio input
    if isinstance(ref_audio, tuple) and len(ref_audio) == 2:
        sample_rate, audio_data = ref_audio
        print(f"[DEBUG] Audio sample rate: {sample_rate}, shape: {audio_data.shape if hasattr(audio_data, 'shape') else 'unknown'}")
    else:
        print(f"[ERROR] Invalid ref_audio format: {type(ref_audio)}")
        raise gr.Error("Invalid reference audio input.")

    temp_audio = None
    try:
        # Convert to numpy array
        audio_data = np.array(audio_data)
        print(f"[DEBUG] Raw audio shape: {audio_data.shape}, dtype: {audio_data.dtype}, min: {audio_data.min()}, max: {audio_data.max()}")
        
        # Handle stereo to mono conversion
        if len(audio_data.shape) > 1:
            if audio_data.shape[0] == 2:  # (2, samples) format
                audio_data = audio_data.mean(axis=0)
            elif audio_data.shape[1] == 2:  # (samples, 2) format
                audio_data = audio_data.mean(axis=1)
            print(f"[DEBUG] Converted to mono: {audio_data.shape}")
        
        # Normalize to float32 in range [-1, 1]
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        else:
            audio_data = audio_data.astype(np.float32)
            # If values are outside [-1, 1], normalize
            max_val = np.abs(audio_data).max()
            if max_val > 1.0:
                audio_data = audio_data / max_val
        
        print(f"[DEBUG] Normalized audio: min={audio_data.min():.3f}, max={audio_data.max():.3f}")
        
        # Resample to 24 kHz for the model
        if sample_rate != 24000:
            print(f"[DEBUG] Resampling from {sample_rate} to 24000 Hz")
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=24000)
            sample_rate = 24000
        
        # Limit audio length (5-15 seconds is ideal for voice cloning)
        max_samples = 15 * sample_rate  # 15 seconds max
        if len(audio_data) > max_samples:
            print(f"[DEBUG] Trimming audio from {len(audio_data)/sample_rate:.1f}s to 15s")
            audio_data = audio_data[:max_samples]
        
        # Save temp audio (needed for transcription and model inference)
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_audio.name, audio_data, samplerate=sample_rate, format="WAV", subtype="PCM_16")
        temp_audio.flush()
        print(f"[DEBUG] Temp audio saved to: {temp_audio.name}")
        
        # Auto-transcribe if ref_text is empty
        if not ref_text or ref_text.strip() == "":
            print("[INFO] No reference text provided, auto-transcribing with Whisper...")
            ref_text = transcribe_audio(temp_audio.name)
            if ref_text:
                print(f"[INFO] Auto-transcribed: {ref_text[:100]}...")
            else:
                raise gr.Error("Failed to auto-transcribe. Please provide the reference text manually.")
        
        print(f"[DEBUG] Final ref audio: {len(audio_data)/sample_rate:.1f} seconds")

        print("[DEBUG] Running model inference...")
        with torch.inference_mode():
            audio = model(text, ref_audio_path=temp_audio.name, ref_text=ref_text)
        print(f"[DEBUG] Model output shape: {audio.shape if hasattr(audio, 'shape') else 'unknown'}, dtype: {audio.dtype if hasattr(audio, 'dtype') else 'unknown'}")

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        print(f"[DEBUG] Returning audio with sample rate 24000")
        return (24000, audio)
    except Exception as exc:
        print(f"[ERROR] Synthesis failed: {exc}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Synthesis failed: {exc}")
    finally:
        if temp_audio and os.path.exists(temp_audio.name):
            try:
                os.remove(temp_audio.name)
            except OSError:
                pass


# Load TTS model
repo_id = "ai4bharat/IndicF5"

print(f"Loading model: {repo_id}")
print(f"Using token: {HF_TOKEN[:10]}..." if HF_TOKEN else "No token provided")

# Test token access using direct HTTP request
import requests
print("Verifying token access to gated files...")
test_url = f"https://huggingface.co/{repo_id}/resolve/main/checkpoints/vocab.txt"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}
resp = requests.head(test_url, headers=headers, allow_redirects=True)
if resp.status_code == 200:
    print(f"✓ Token verified - can access gated files")
else:
    print(f"❌ Token verification failed: HTTP {resp.status_code}")
    print(f"   The token does NOT have access to this repository.")
    raise SystemExit("Token does not have access. Please check the token.")

try:
    # Login to Hugging Face - this sets up authentication globally
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("Successfully authenticated with Hugging Face")
    except Exception as login_err:
        print(f"Login attempt note: {login_err}")
    
    # Load model with token
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True, token=HF_TOKEN)
    print("Model loaded successfully!")
except OSError as exc:
    error_msg = str(exc)
    if "403" in error_msg or "gated repo" in error_msg.lower() or "not in the authorized list" in error_msg.lower():
        raise OSError(
            f"\n{'='*60}\n"
            "ACCESS DENIED: Your Hugging Face account doesn't have access to this gated model.\n\n"
            "To fix this:\n"
            "1. Visit: https://huggingface.co/ai4bharat/IndicF5\n"
            "2. Click 'Request access' or 'Agree and access repository'\n"
            "3. Wait for approval (may take time)\n"
            "4. Once approved, run this script again\n\n"
            f"Original error: {error_msg}\n"
            f"{'='*60}"
        )
    else:
        raise OSError(
            f"Failed to load model: {error_msg}\n"
            "If the repository is gated, make sure:\n"
            "1. Your HF_TOKEN has access to the model\n"
            "2. You've requested access at https://huggingface.co/ai4bharat/IndicF5"
        )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device", device)
model = model.to(device)

# Define Gradio interface with layout adjustments
with gr.Blocks() as iface:
    gr.Markdown(
        """
        # **Voice Cloning TTS for Indian Languages**

        Supports **11 Indian languages**:  
        **Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu.**  
        
        Generate speech using a reference prompt audio. The reference text is **optional** - if not provided, it will be auto-transcribed using Whisper.
        """
    )
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Text to Synthesize", placeholder="Enter the text you want the cloned voice to speak...", lines=3)
            ref_audio_input = gr.Audio(type="numpy", label="Reference Voice Audio (5-15 sec recommended)")
            ref_text_input = gr.Textbox(label="Reference Text (Optional - auto-transcribed if empty)", placeholder="What is being said in the reference audio? Leave empty for auto-transcription.", lines=2)
            submit_btn = gr.Button("🎤 Generate Speech", variant="primary")
        
        with gr.Column():
            output_audio = gr.Audio(label="Generated Speech", type="numpy")

    submit_btn.click(
        synthesize_speech,
        inputs=[text_input, ref_audio_input, ref_text_input],
        outputs=[output_audio],
    )

# Limit concurrency to avoid GPU overload and make app queueable/public
iface.queue(max_size=4, default_concurrency_limit=1)

# Close any existing Gradio instances
try:
    gr.close_all()
except:
    pass

# Check if running in Colab
def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

# Launch with share=True for Colab (required for public access)
iface.launch(
    server_name="0.0.0.0",
    server_port=None,  # Let Gradio find an available port
    share=is_colab(),  # Auto-share on Colab
    show_error=True,
)
