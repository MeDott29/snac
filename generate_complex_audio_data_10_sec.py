import numpy as np
import torch
from snac import SNAC
from scipy.io import wavfile
import json
from tqdm import tqdm

def generate_waveform(duration, sample_rate, freq, waveform_type, amplitude=1.0):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    if waveform_type == 'sine':
        return amplitude * np.sin(2 * np.pi * freq * t)
    elif waveform_type == 'square':
        return amplitude * np.sign(np.sin(2 * np.pi * freq * t))
    elif waveform_type == 'sawtooth':
        return amplitude * (2 * (freq * t - np.floor(0.5 + freq * t)))
    elif waveform_type == 'triangle':
        return amplitude * (2 * np.abs(2 * (freq * t - np.floor(0.5 + freq * t))) - 1)

def generate_complex_audio(duration, sample_rate, waveforms):
    complex_audio = np.zeros(int(sample_rate * duration))
    for waveform in waveforms:
        complex_audio += generate_waveform(duration, sample_rate, **waveform)
    return complex_audio / len(waveforms)  # Normalize

def encode_with_snac(audio, sample_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SNAC.from_pretrained("hubertsiuzdak/snac_44khz").eval().to(device)
    audio_tensor = torch.tensor(audio).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        codes = model.encode(audio_tensor)
    return [code.cpu().numpy().tolist() for code in codes]

def main():
    sample_rate = 44100
    duration = 5  # seconds
    num_samples = 10
    data = []

    for i in tqdm(range(num_samples)):
        waveforms = [
            {'freq': np.random.uniform(100, 1000), 'waveform_type': np.random.choice(['sine', 'square', 'sawtooth', 'triangle']), 'amplitude': np.random.uniform(0.5, 1.0)}
            for _ in range(np.random.randint(2, 5))
        ]
        
        complex_audio = generate_complex_audio(duration, sample_rate, waveforms)
        snac_codes = encode_with_snac(complex_audio, sample_rate)
        
        sample_data = {
            'waveforms': waveforms,
            'snac_codes': snac_codes
        }
        data.append(sample_data)
        
        # Save audio as WAV for verification (optional)
        wavfile.write(f"complex_audio_{i}.wav", sample_rate, (complex_audio * 32767).astype(np.int16))

    # Save the data to a JSON file
    with open('complex_audio_data.json', 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    main()
