import os
import spaces
import torch
import librosa
import tempfile
import numpy as np
import gradio as gr
import soundfile as sf
from transformers import AutoModel

@spaces.GPU
def synthesize_speech(text, ref_audio, ref_text):
    if not text or text.strip() == "":
        return "Error: Please provide text to synthesize."
    if len(text) > 500:
        return "Error: Text too long (max 500 characters)."
    if ref_audio is None or ref_text.strip() == "":
        return "Error: Please provide a reference audio and its corresponding text."
    
    # Ensure valid reference audio input
    if isinstance(ref_audio, tuple) and len(ref_audio) == 2:
        sample_rate, audio_data = ref_audio
    else:
        return "Error: Invalid reference audio input."

    temp_audio = None
    try:
        # Normalize input audio to 24 kHz for the model
        if sample_rate != 24000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=24000)
            sample_rate = 24000

        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_audio.name, audio_data, samplerate=sample_rate, format="WAV")
        temp_audio.flush()

        with torch.inference_mode():
            audio = model(text, ref_audio_path=temp_audio.name, ref_text=ref_text)

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        return 24000, audio
    except Exception as exc:  # broad catch to return user-friendly errors
        return f"Error during synthesis: {exc}"
    finally:
        if temp_audio and os.path.exists(temp_audio.name):
            try:
                os.remove(temp_audio.name)
            except OSError:
                pass


# Load TTS model
repo_id = "ai4bharat/IndicF5"
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
try:
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True, token=hf_token)
except OSError as exc:
    raise OSError(
        "Failed to load model. If the repository is gated, set an access token in "
        "HF_TOKEN or HUGGINGFACEHUB_API_TOKEN environment variables. "
        "Original error: "
        f"{exc}"
    )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device", device)
model = model.to(device)

# Define Gradio interface with layout adjustments
with gr.Blocks() as iface:
    gr.Markdown(
        """
        # **Voice Cloning TTS for Indian Languages**

        supports **11 Indian languages**:  
        **Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu.**  
        
        Generate speech using a reference prompt audio and its corresponding text.
        """
    )
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Text to Synthesize", placeholder="Enter the text to convert to speech...", lines=3)
            ref_audio_input = gr.Audio(type="numpy", label="Reference Prompt Audio")
            ref_text_input = gr.Textbox(label="Text in Reference Prompt Audio", placeholder="Enter the transcript of the reference audio...", lines=2)
            submit_btn = gr.Button("🎤 Generate Speech", variant="primary")
        
        with gr.Column():
            output_audio = gr.Audio(label="Generated Speech", type="numpy")

    submit_btn.click(
        synthesize_speech,
        inputs=[text_input, ref_audio_input, ref_text_input],
        outputs=[output_audio],
    )

# Limit concurrency to avoid GPU overload and make app queueable/public
iface.queue(max_size=4, concurrency_count=1)


def should_share():
    # Allow env toggle or auto-share when running inside Colab
    if os.environ.get("GRADIO_SHARE", "0") == "1":
        return True
    return "COLAB_RELEASE" in os.environ or "COLAB_GPU" in os.environ


iface.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
    share=should_share(),
    show_error=True,
)
