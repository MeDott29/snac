import torch
from snac import SNAC
import pickle
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

# Placeholder audio data (replace with your actual audio)
sample_rate = 44100
duration = 3
num_samples = sample_rate * duration
time_axis = torch.linspace(0, duration, num_samples)
sine_wave = torch.sin(2 * torch.pi * 440 * time_axis).unsqueeze(0).unsqueeze(0)
audio_data = sine_wave

# Check if CUDA is available and move data to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_data = audio_data.to(device)
if device.type == "cpu":
    logging.warning("CUDA not available, using CPU. This may significantly slow down processing.")

# Cache file for storing generated codes
cache_file = "snac_codes.pkl"

# Check if cached codes exist and load them
if os.path.exists(cache_file):
    try:
        with open(cache_file, "rb") as file:
            codes = pickle.load(file)
        logging.info("Loaded cached SNAC codes.")
        # Print a summary of the cached code shapes
        for i, code in enumerate(codes):
            print(f"Cached Code {i+1}: shape={code.shape}")
    except (EOFError, pickle.UnpicklingError) as e:
        logging.warning(f"Error loading cached SNAC codes: {e}. Regenerating codes.")
        codes = None
    except Exception as e:
        logging.exception(f"An unexpected error occurred while loading cached SNAC codes: {e}")
        codes = None
else:
    codes = None

# Load the SNAC model if codes are not cached
if codes is None:
    try:
        model = SNAC.from_pretrained("hubertsiuzdak/snac_44khz").eval().to(device)
        logging.info("SNAC model loaded.")
    except Exception as e:
        logging.error(f"Error loading SNAC model: {e}")
        exit(1)

# Encode the audio data if not cached
if codes is None:
    try:
        with torch.no_grad():
            codes = model.encode(audio_data)
        logging.info("Generated SNAC codes:")
        for i, code in enumerate(codes):
            logging.info(f"Code {i+1}: shape={code.shape}, first 20 values={code[0, :20].tolist()}")

        # Cache the generated codes
        with open(cache_file, "wb") as file:
            pickle.dump(codes, file)
        logging.info("Cached SNAC codes for future use.")
    except RuntimeError as e:
        logging.error(f"CUDA error during encoding: {e}. Check your CUDA setup.")
        exit(1)
    except Exception as e:
        logging.exception(f"An unexpected error occurred during encoding: {e}")
        exit(1)

logging.info("SNAC code generation complete.")
