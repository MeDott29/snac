from openai import OpenAI, OpenAIError
from os import getenv, path
import torch
from snac import SNAC
import re
import pickle
import json
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

# Generate simple waveforms
sample_rate = 44100  # Sample rate in Hz
duration = 3  # Updated duration of audio to 3 seconds
num_samples = sample_rate * duration

# Sine wave
time = torch.linspace(0, duration, num_samples)
sine_wave = torch.sin(2 * torch.pi * 440 * time)  # Frequency of 440 Hz
sine_wave = sine_wave.unsqueeze(0).unsqueeze(0)  # Added to ensure correct shape

# Square wave
square_wave = torch.where(torch.sin(2 * torch.pi * 220 * time) > 0, 1, -1)  # Frequency of 220 Hz
square_wave = square_wave.unsqueeze(0).unsqueeze(0)  # Added to ensure correct shape

# Stack the waveforms for processing
audio_data = torch.cat([sine_wave, square_wave], dim=0)

# Check if CUDA is available and move data to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_data = audio_data.to(device)
if device.type == "cpu":
    logging.warning("CUDA not available, using CPU. This may significantly slow down processing.")


# Load the SNAC model
try:
    model = SNAC.from_pretrained("hubertsiuzdak/snac_44khz").eval().to(device)
except Exception as e:
    logging.error(f"Error loading SNAC model: {e}")
    exit(1)


# Cache file for storing generated codes
cache_file = "snac_codes.pkl"

# Check if cached codes exist and load them
codes = None
try:
    start_time = time.time()
    with open(cache_file, "rb") as file:
        codes = pickle.load(file)
    end_time = time.time()
    logging.info(f"Loaded cached SNAC codes in {end_time - start_time:.4f} seconds.")
except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
    logging.warning(f"Could not load cached SNAC codes: {e}. Regenerating codes.")
except Exception as e:
    logging.exception(f"An unexpected error occurred while loading cached SNAC codes: {e}")

# Encode the audio data if not cached
if codes is None:
    try:
        start_time = time.time()
        with torch.inference_mode():
            codes = model.encode(audio_data)
        end_time = time.time()
        logging.info(f"Generated SNAC codes in {end_time - start_time:.4f} seconds:")  # Print header
        for i, code in enumerate(codes):
            logging.info(f"  Code {i+1} shape: {code.shape}, first 20 values: {code[0, :20].tolist()}")

        # Cache the generated codes
        with open(cache_file, "wb") as file:
            pickle.dump(codes, file)
        logging.info("Cached SNAC codes for future use.")
    except Exception as e:
        logging.error(f"Error encoding audio: {e}")
        exit(1)

# Update the system message to include information about the waveform and SNAC tokens
system_message = """
You are now an expert in generating well-formatted SNAC tokens that will be decoded into audio data. Your primary task is to generate accurate and structured SNAC token sequences in JSON format.

## Audio Data Description

The audio consists of two channels: a 440Hz sine wave and a 220Hz square wave, both 3 seconds long, sampled at 44100 Hz.  The SNAC encoder has produced codes at multiple resolutions.

## Output Format
You should output a JSON array where each element is a dictionary representing a code. Each dictionary should have a "code_index" key (an integer), a "shape" key (a list of integers), and a "data" key (a list of numbers).

Here is an example of the expected JSON format:
```json
[
  {
    "code_index": 1,
    "shape": [2, 44],
    "data": [1919, 2942, 1962, 1962, 1962, 1962, 1962, 1962, 1962, 1962, 1962, 1962, 1962, 1962, 1962, 1962, 1962, 1962, 1962, 1962]
  },
  {
    "code_index": 2,
    "shape": [2, 88],
    "data": [126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126]
  },
  {
    "code_index": 3,
    "shape": [2, 176],
    "data": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
  },
  {
    "code_index": 4,
    "shape": [2, 352],
    "data": [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
  }
]
```
"""

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
)

# Format SNAC codes for display.  This is now structured for easier JSON parsing by the LLM.
formatted_codes = []
for i, code in enumerate(codes):
    formatted_codes.append({
        "code_index": i + 1,
        "shape": list(code.shape),
        "data": code[0, :20].tolist() # Sending more data for better context.
    })

# Display SNAC codes to the LLM
messages = [
    {"role": "system", "content": system_message},
    {
        "role": "user",
        "content": "Please provide the SNAC codes in the exact JSON format as shown in the example. Do not include any additional explanations or steps."
    }
]

try:
    start_time = time.time()
    completion = client.chat.completions.create(
        model="liquid/lfm-40b",
        messages=messages,
        timeout=60  # Set a timeout of 60 seconds
    )
    end_time = time.time()
    llm_response = completion.choices[0].message.content
    logging.info(f"LLM Response generated in {end_time - start_time:.4f} seconds:\n{llm_response}")

    try:
        # Pre-validation: Check if the response looks like valid JSON before parsing.
        if not llm_response.strip().startswith('[') or not llm_response.strip().endswith(']'):
            raise ValueError("LLM response does not appear to be a JSON array.")

        llm_codes = json.loads(llm_response)
        # Validate the structure of the JSON response
        for code in llm_codes:
            if not isinstance(code, dict) or "shape" not in code or "data" not in code or "code_index" not in code:
                raise ValueError("Invalid JSON format from LLM: Missing keys.")
            if not isinstance(code["shape"], list) or not all(isinstance(x, int) for x in code["shape"]):
                raise ValueError("Invalid 'shape' format in LLM response.")
            if not isinstance(code["data"], list) or not all(isinstance(x, (int, float)) for x in code["data"]):
                raise ValueError("Invalid 'data' format in LLM response.")
            if not isinstance(code["code_index"], int):
                raise ValueError("Invalid 'code_index' format in LLM response.")

        logging.info("LLM codes successfully parsed and validated.")
        token_summary = llm_codes # Use the validated JSON data

    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON response from LLM: {e}")
        token_summary = None
    except ValueError as e:
        logging.error(f"Error validating LLM response: {e}")
        token_summary = None
    except Exception as e:
        logging.exception(f"An unexpected error occurred during LLM response processing: {e}")
        token_summary = None

except OpenAIError as e:
    logging.error(f"OpenAI API Error: {e}")
except Exception as e:
    logging.exception(f"An unexpected error occurred: {e}")

print(token_summary)
