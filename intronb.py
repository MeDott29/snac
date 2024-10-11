# Install necessary libraries
# !pip install snac numpy scipy librosa soundfile

# Import necessary libraries
import torch
import numpy as np
import librosa
import soundfile as sf
from snac import SNAC
import matplotlib.pyplot as plt
from scipy import signal

# Generate complex waveform
def generate_complex_waveform(duration=5, sr=32000):
    """Generates a complex waveform with multiple frequencies and waveforms."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = 0
    for freq in [100, 200, 500, 1000, 2000]:  # Example frequencies
        waveform += 0.2 * signal.square(2 * np.pi * freq * t)
        waveform += 0.2 * np.sin(2 * np.pi * freq * t)
        waveform += 0.2 * signal.sawtooth(2 * np.pi * freq * t)
        waveform += 0.2 * signal.triangle(2 * np.pi * freq * t)
    waveform = waveform / np.max(np.abs(waveform))  # Normalize
    return waveform, sr

# Generate waveform and save to file
try:
    waveform, sr = generate_complex_waveform()
    sf.write('complex_waveform.wav', waveform, sr)
    print("Waveform generated and saved successfully.")
except Exception as e:
    print(f"Error generating or saving waveform: {e}")


# Load SNAC model
try:
    model = SNAC.from_pretrained("hubertsiuzdak/snac_32khz").eval().cuda()
    print("SNAC model loaded successfully.")
except Exception as e:
    print(f"Error loading SNAC model: {e}")

# Load audio
try:
    audio, sr = librosa.load('complex_waveform.wav', sr=32000, mono=True)
    audio = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).float().cuda()
    print("Audio loaded successfully.")
except FileNotFoundError:
    print("Error: complex_waveform.wav not found.  Make sure the waveform generation was successful.")
except Exception as e:
    print(f"Error loading audio: {e}")


# Encode and decode
try:
    with torch.no_grad():
        codes = model.encode(audio)
        audio_hat = model.decode(codes)
    print("Encoding and decoding successful.")
except Exception as e:
    print(f"Error during encoding/decoding: {e}")

# Convert to numpy and save
try:
    audio_hat_np = audio_hat.squeeze().cpu().numpy()
    sf.write('reconstructed_audio.wav', audio_hat_np, sr)
    print("Reconstructed audio saved successfully.")
except Exception as e:
    print(f"Error saving reconstructed audio: {e}")

# Normalize audio after loading to handle potential amplitude variations
audio_hat_np = audio_hat_np / np.max(np.abs(audio_hat_np))

# Display waveforms
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(waveform)
plt.title('Original Waveform')
plt.subplot(2, 1, 2)
plt.plot(audio_hat_np)
plt.title('Reconstructed Waveform')
plt.tight_layout()
plt.show()


# Play audio in notebook (using IPython.display.Audio)
from IPython.display import Audio
try:
    display(Audio('complex_waveform.wav'))
    display(Audio('reconstructed_audio.wav'))
except Exception as e:
    print(f"Error playing audio: {e}")