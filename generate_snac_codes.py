import torch
from snac import SNAC
import pickle
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    logging.warning("CUDA not available, using CPU.")

# Placeholder audio data (replace with your actual audio)
sample_rate = 44100
duration = 3
num_samples = sample_rate * duration
time = torch.linspace(0, duration, num_samples)
sine_wave = torch.sin(2 * torch.pi * 440 * time).unsqueeze(0).unsqueeze(0)
audio_data = sine_wave.to(device)


# Load the SNAC model
try:
    model = SNAC.from_pretrained("hubertsiuzdak/snac_44khz").eval().to(device)
except Exception as e:
    logging.error(f"Error loading SNAC model: {e}")
    exit(1)

# Cache file for storing generated codes
cache_file = "snac_codes.pkl"

# Check if cached codes exist
if os.path.exists(cache_file):
    try:
        with open(cache_file, "rb") as file:
            codes = pickle.load(file)
        logging.info("Loaded cached SNAC codes.")
    except (EOFError, pickle.UnpicklingError) as e:  # More specific exception handling
        logging.warning(f"Error loading cached SNAC codes: {e}. Regenerating codes.")
        codes = None
    except Exception as e:
        logging.exception(f"An unexpected error occurred while loading cached SNAC codes: {e}")
        codes = None
else:
    codes = None

# Encode the audio data if not cached
if codes is None:
    try:
        with torch.no_grad(): # More efficient than torch.inference_mode()
            codes = model.encode(audio_data)
        logging.info("Generated SNAC codes:")
        for i, code in enumerate(codes):
            logging.info(f"  Code {i+1} shape: {code.shape}, first 20 values: {code[0, :20].tolist()}")

        # Cache the generated codes
        with open(cache_file, "wb") as file:
            pickle.dump(codes, file)
        logging.info("Cached SNAC codes for future use.")
    except RuntimeError as e: # Catch CUDA errors specifically
        logging.error(f"CUDA error during encoding: {e}. Check your CUDA setup.")
        exit(1)
    except Exception as e:
        logging.exception(f"An unexpected error occurred during encoding: {e}")
        exit(1)

print("SNAC code generation complete.")

