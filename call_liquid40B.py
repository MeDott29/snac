from openai import OpenAI, OpenAIError
from os import getenv, path
import torch
from snac import SNAC
import re
import pickle
import json

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

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU.")

# Load the SNAC model
try:
    model = SNAC.from_pretrained("hubertsiuzdak/snac_44khz").eval().to(device)
except Exception as e:
    print(f"Error loading SNAC model: {e}")
    exit(1)


# Cache file for storing generated codes
cache_file = "snac_codes.pkl"

# Check if cached codes exist
if path.exists(cache_file):
    try:
        with open(cache_file, "rb") as file:
            codes = pickle.load(file)
        print("Loaded cached SNAC codes.")
    except Exception as e:
        print(f"Error loading cached SNAC codes: {e}. Regenerating codes.")
        codes = None
else:
    codes = None

# Encode the audio data if not cached
if codes is None:
    try:
        with torch.inference_mode():
            codes = model.encode(audio_data.to(device))
        print("Generated SNAC codes:")  # Print header
        for i, code in enumerate(codes):
            print(f"  Code {i+1} shape: {code.shape}, first 10 values: {code[0, :10].tolist()}")

        # Cache the generated codes
        with open(cache_file, "wb") as file:
            pickle.dump(codes, file)
        print("Cached SNAC codes for future use.")
    except Exception as e:
        print(f"Error encoding audio: {e}")
        exit(1)

# Update the system message to include information about the waveform and SNAC tokens
system_message = """
You are now an expert in generating well-formatted SNAC tokens that will be decoded into audio data. Your primary task is to generate accurate and structured SNAC token sequences in JSON format.

## Audio Data Description

The audio consists of two channels: a 440Hz sine wave and a 220Hz square wave, both 3 seconds long, sampled at 44100 Hz.

## Output Format
You should output a JSON array where each element is a dictionary representing a code. Each dictionary should have a "code_index" key (an integer), a "shape" key (a list of integers), and a "data" key (a list of numbers).

Here is an example of the expected JSON format:
```json
[
  {
    "code_index": 1,
    "shape": [2, 44],
    "data": [1919, 2942, 1962, 1962, 1962, 1962, 1962, 1962, 1962, 1962]
  },
  {
    "code_index": 2,
    "shape": [2, 88],
    "data": [126, 126, 126, 126, 126, 126, 126, 126, 126, 126]
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
        "data": code[0, :10].tolist() # Only sending a subset of the data for brevity.
    })

# Display SNAC codes to the LLM
messages = [
    {"role": "system", "content": system_message},
    {
        "role": "user",
        "content": f"Generate codes that encode audio data into the following SNAC codes in JSON format:\n\n{json.dumps(formatted_codes, indent=2)}"
    }
]

try:
    completion = client.chat.completions.create(
        model="liquid/lfm-40b",
        messages=messages,
    )
    llm_response = completion.choices[0].message.content
    print(f"LLM Response:\n{llm_response}")

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

        print("LLM codes successfully parsed and validated.")
        token_summary = llm_codes # Use the validated JSON data

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from LLM: {e}")
        token_summary = None
    except ValueError as e:
        print(f"Error validating LLM response: {e}")
        token_summary = None
    except Exception as e:
        print(f"An unexpected error occurred during LLM response processing: {e}")
        token_summary = None

except OpenAIError as e:
    print(f"OpenAI API Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print(token_summary)
