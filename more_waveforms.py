import numpy as np
import librosa
import soundfile as sf

# Generate white noise
def generate_white_noise(duration=2, sr=32000):
    """Generates white noise."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    noise = np.random.normal(size=t.shape)
    return noise, sr

# Generate pink noise using a filter
def generate_pink_noise(duration=2, sr=32000):
    """Generates pink noise."""
    n_samples = int(sr * duration)
    white_noise = np.random.randn(n_samples)
    
    # Create a pink noise filter
    f = np.arange(1, n_samples // 2 + 1)
    a = 1 / (f ** 1.4)  # Pink noise spectrum
    pink_filter = np.zeros_like(f, dtype=np.complex)
    for i in range(len(f)):
        pink_filter[i] = white_noise[i] * a[i]
    
    # Apply the filter to the white noise
    pink_noise = np.real(np.fft.ifft(pink_filter))
    return pink_noise[:n_samples], sr

# Generate sine wave with varying amplitudes
def generate_varying_amplitude_sine_wave(duration=2, sr=32000):
    """Generates a sine wave with varying amplitudes."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = 100  # Base frequency
    amplitude = np.sin(2 * np.pi * freq * t) + 0.5  # Varying amplitude
    waveform = amplitude * np.sin(2 * np.pi * freq * t)
    return waveform, sr

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

# Save waveforms to files
def save_waveform(waveform, sr, filename):
    """Saves a waveform to a file."""
    sf.write(filename, waveform, sr)

# Example usage
if __name__ == "__main__":
    duration = 2
    sr = 32000

    # Generate and save white noise
    white_noise, _ = generate_white_noise(duration, sr)
    save_waveform(white_noise, sr, 'white_noise.wav')
    print("White noise generated and saved successfully.")

    # Generate and save pink noise
    pink_noise, _ = generate_pink_noise(duration, sr)
    save_waveform(pink_noise, sr, 'pink_noise.wav')
    print("Pink noise generated and saved successfully.")

    # Generate and save sine wave with varying amplitudes
    varying_amplitude_sine_wave, _ = generate_varying_amplitude_sine_wave(duration, sr)
    save_waveform(varying_amplitude_sine_wave, sr, 'varying_amplitude_sine_wave.wav')
    print("Varying amplitude sine wave generated and saved successfully.")

    # Generate and save complex sine wave
    complex_sine_wave, _ = generate_complex_sine_wave(duration, sr)
    save_waveform(complex_sine_wave, sr, 'complex_sine_wave.wav')
    print("Complex sine wave generated and saved successfully.")
