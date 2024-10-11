import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Define time vector
sr = 32000  # Sample rate
duration = 1  # Duration in seconds
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

# Generate waveforms using scipy.signal for accurate shapes
waveforms = {
    "sawtooth": signal.sawtooth(2 * np.pi * 100 * t),
    "square": signal.square(2 * np.pi * 100 * t),
    "triangle": signal.sawtooth(2 * np.pi * 100 * t, width=0.5),
    "sine": np.sin(2 * np.pi * 100 * t),
}

# Plot waveforms
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for i, (name, waveform) in enumerate(waveforms.items()):
    axs[i // 2, i % 2].plot(t, waveform)
    axs[i // 2, i % 2].set_title(name)
    axs[i // 2, i % 2].set_xlabel("Time (s)")
    axs[i // 2, i % 2].set_ylabel("Amplitude")

plt.tight_layout()
plt.show()
