from openai import OpenAI, OpenAIError
from os import getenv, path
import torch
from snac import SNAC
import re
import pickle

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
        print("Generated SNAC codes:", codes)  # Print the generated codes

        # Cache the generated codes
        with open(cache_file, "wb") as file:
            pickle.dump(codes, file)
        print("Cached SNAC codes for future use.")
    except Exception as e:
        print(f"Error encoding audio  {e}")
        exit(1)

# Update the system message to include information about the waveform and SNAC tokens
system_message = """
You are now an expert in generating well-formatted SNAC tokens the will be decoded into audio data. Your primary task is to generate accurate and structured SNAC token sequences. You will receive a description of the audio data and example SNAC tokens.

## Audio Data Description

The audio consists of two channels: a 440Hz sine wave and a 220Hz square wave, both 3 seconds long, sampled at 44100 Hz.

## Example SNAC Tokens


## Task

## Output Format
"""

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
)

# Display SNAC tokens to the LLM
messages = [
    {"role": "system", "content": system_message},
    {
        "role": "user",
        "content": "Describe the audio content represented by the example SNAC tokens.",
    }
]

try:
    completion = client.chat.completions.create(
        model="liquid/lfm-40b",
        messages=messages,
    )
    llm_response = completion.choices[0].message.content
    print(f"LLM Response:\n{llm_response}")

    # Extract relevant information from LLM response using regular expressions (This part is not needed anymore)
    token_summary = llm_response

except OpenAIError as e:
    print(f"OpenAI API Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print(token_summary)
