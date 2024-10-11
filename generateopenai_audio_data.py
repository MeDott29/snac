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

# Function to encode audio using a pre-trained model (e.g., Wav2Vec2) and extract features
def encode_audio(audio, processor, model, sr):
    inputs = processor(audio, return_tensors="pt", sampling_rate=sr, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state.squeeze()
    
    # Min-max scaling
    min_val = torch.min(last_hidden_state)
    max_val = torch.max(last_hidden_state)
    scaled_hidden_state = (last_hidden_state - min_val) / (max_val - min_val)
    audio_encoding = scaled_hidden_state.mean(dim=0).tolist()

    # Extract features using Librosa
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    rmse = np.mean(librosa.feature.rms(y=audio))

    # Append features to the encoding
    audio_encoding.extend([zcr, spectral_centroid, rmse])

    return audio_encoding

# Function to query LFM-40B model with encoded audio and features
def query_model(client, model_name, audio_encoding, sr):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": f"""
                Analyze the following audio data. The data includes the mean activations from a Wav2Vec2 model, a pre-trained model used for speech recognition, along with some additional audio features.  Higher Wav2Vec2 activations generally indicate stronger activation of certain learned features.  The additional features are: Zero Crossing Rate (ZCR), Spectral Centroid, and Root Mean Square Energy (RMSE).  ZCR indicates the rate of zero crossings in the audio waveform.  Spectral Centroid represents the "center of mass" of the audio spectrum.  RMSE measures the average energy of the audio signal.  The sampling rate of the original audio is also provided.  Try to interpret the data and provide insights about the potential content of the audio.  The audio might contain speech, noise, music, or other sounds.  Consider all possibilities and provide a balanced analysis.
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
    audio_encoding = encode_audio(audio, processor, model, sr)
    
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
