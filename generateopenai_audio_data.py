from openai import OpenAI
from os import getenv
import soundfile as sf
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import sys

# Initialize the OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
)

# Function to load and preprocess audio
def load_audio(file_path, sample_rate=16000):
    # Load audio using librosa
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio, sr

# Function to encode audio using a pre-trained model (e.g., Wav2Vec2)
def encode_audio(audio, processor, model):
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Calculate the mean across the time dimension (dimension 1)
    audio_encoding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return audio_encoding

# Function to query LFM-40B model with encoded audio
def query_model(client, model_name, audio_encoding, sr):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": """
                Analyze the following audio data.  The data represents the mean activations from a Wav2Vec2 model, a pre-trained model used for speech recognition.  These activations are derived from the internal representations of the model after processing the raw audio waveform.  Higher values generally indicate stronger activation of certain features learned by the model.  The sampling rate of the original audio is also provided.  Try to interpret the activations in the context of speech recognition and provide insights about the potential content of the audio.
                """
            },
            {
                "role": "user",
                "content": f"Audio Encoding: {audio_encoding}\nSampling Rate: {sr}"
            }
        ]
    )
    return completion.choices[0].message.content

# Example evaluation pipeline
def evaluate_audio_analysis(file_path):
    # Load audio file
    audio, sr = load_audio(file_path)
    
    # Use a pre-trained audio model like Wav2Vec2 to encode the audio
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    audio_encoding = encode_audio(audio, processor, model)
    
    # Query LFM-40B with the audio encoding for analysis
    analysis = query_model(client, "liquid/lfm-40b", audio_encoding, sr)
    
    print("Analysis Result:", analysis)

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generateopenai_audio_data.py <audio_file_path>")
        sys.exit(1)
    audio_file_path = sys.argv[1]
    evaluate_audio_analysis(audio_file_path)

