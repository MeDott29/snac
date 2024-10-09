import torch
from snac import SNAC
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

# Determine the device (CUDA if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Generate a simple sine wave
sample_rate = 44100  # Sample rate in Hz
duration = 3  # Duration of audio in seconds
num_samples = sample_rate * duration
time_axis = torch.linspace(0, duration, num_samples, device=device) # Move tensor creation to device
sine_wave = torch.sin(2 * torch.pi * 440 * time_axis).unsqueeze(0).unsqueeze(0)

# Load the SNAC model onto the selected device
try:
    model = SNAC.from_pretrained("hubertsiuzdak/snac_44khz").eval().to(device)
    logging.info("SNAC model loaded successfully.")
except Exception as e:
    logging.error(f"An error occurred while loading the SNAC model: {e}")
    exit(1)

# Encode the audio data
try:
    with torch.no_grad():
        codes = model.encode(sine_wave)
    logging.info("SNAC encoding complete.")
    for i, code in enumerate(codes):
        logging.info(f"Code {i+1}: shape={code.shape}, first 20 values={code[0, :20].tolist()}")
except RuntimeError as e: # Catch CUDA-specific errors
    logging.error(f"A CUDA error occurred during encoding: {e}. Please check your CUDA setup.")
    exit(1)
except Exception as e:
    logging.error(f"An unexpected error occurred during encoding: {e}")
    exit(1)

logging.info("SNAC code generation complete.")

