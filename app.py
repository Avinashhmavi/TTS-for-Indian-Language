import io
import spaces
import torch
import librosa
import requests
import tempfile
import numpy as np
import gradio as gr
import soundfile as sf
from transformers import AutoModel

# Function to load reference audio from URL
def load_audio_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        audio_data, sample_rate = sf.read(io.BytesIO(response.content))
        return sample_rate, audio_data
    return None, None

@spaces.GPU
def synthesize_speech(text, ref_audio, ref_text):
    if ref_audio is None or ref_text.strip() == "":
        return "Error: Please provide a reference audio and its corresponding text."
    
    # Ensure valid reference audio input
    if isinstance(ref_audio, tuple) and len(ref_audio) == 2:
        sample_rate, audio_data = ref_audio
    else:
        return "Error: Invalid reference audio input."
    
    # Save reference audio directly without resampling
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, audio_data, samplerate=sample_rate, format='WAV')
        temp_audio.flush()
    
    audio = model(text, ref_audio_path=temp_audio.name, ref_text=ref_text)
    # Normalize output and save
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    return 24000, audio


# Load TTS model
repo_id = "ai4bharat/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
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

    submit_btn.click(synthesize_speech, inputs=[text_input, ref_audio_input, ref_text_input], outputs=[output_audio])


iface.launch()
