import numpy as np
import librosa
import soundfile as sf
from scipy.signal import pink_noise

# Generate white noise
def generate_white_noise(duration=2, sr=32000):
    """Generates white noise."""
    return np.random.normal(size=int(sr * duration)), sr

# Generate pink noise (efficient version)
def generate_pink_noise(duration=2, sr=32000):
    """Generates pink noise using scipy."""
    return pink_noise(int(sr * duration)), sr

# Generate sine wave with varying amplitudes
def generate_varying_amplitude_sine_wave(duration=2, sr=32000):
    """Generates a sine wave with varying amplitudes."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = 100  # Base frequency
    amplitude = np.sin(2 * np.pi * freq * t) + 0.5  # Varying amplitude
    return amplitude * np.sin(2 * np.pi * freq * t), sr

# Generate a combination of sine waves with varying amplitudes and frequencies
def generate_complex_sine_wave(duration=2, sr=32000):
    """Generates a complex sine wave with varying amplitudes and frequencies."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freqs = [100, 200, 500, 1000, 2000]  # Frequencies
    amplitudes = [0.2, 0.3, 0.4, 0.5, 0.6]  # Amplitudes
    waveform = np.zeros_like(t)
    for freq, amplitude in zip(freqs, amplitudes):
        waveform += amplitude * np.sin(2 * np.pi * freq * t)
    return waveform, sr

# Save waveforms to files. Includes error handling.
def save_waveform(waveform, sr, filename):
    """Saves a waveform to a file."""
    try:
        sf.write(filename, waveform, sr)
        print(f"{filename} generated and saved successfully.")
    except Exception as e:
        print(f"Error saving {filename}: {e}")

# Example usage
if __name__ == "__main__":
    duration = 2
    sr = 32000

    # Generate and save waveforms
    waveforms = {
        "white_noise": generate_white_noise,
        "pink_noise": generate_pink_noise,
        "varying_amplitude_sine_wave": generate_varying_amplitude_sine_wave,
        "complex_sine_wave": generate_complex_sine_wave,
    }

    for name, func in waveforms.items():
        waveform, sr = func(duration, sr)
        save_waveform(waveform, sr, f"{name}.wav")

