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
import os

# Generate complex waveform
def generate_complex_waveform(duration=2, sr=32000):
    """Generates a complex waveform with multiple frequencies and waveforms."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = 0
    for freq in [100, 200, 500, 1000, 2000]:  # Example frequencies
        waveform += 0.2 * signal.square(2 * np.pi * freq * t)
        waveform += 0.2 * np.sin(2 * np.pi * freq * t)
        waveform += 0.2 * signal.sawtooth(2 * np.pi * freq * t)
        waveform += 0.2 * signal.sawtooth(2 * np.pi * freq * t, width=0.5) #Corrected triangle wave generation
    waveform = waveform / np.max(np.abs(waveform))  # Normalize
    return waveform, sr

# Generate waveform and save to file
try:
    waveform, sr = generate_complex_waveform()
    sf.write('complex_waveform.wav', waveform, sr)
    print("Waveform generated and saved successfully.")
except FileNotFoundError as e:
    print(f"Error: File not found: {e}")
except Exception as e:
    print(f"Error generating or saving waveform: {e}")
    exit(1) #Exit if waveform generation fails

