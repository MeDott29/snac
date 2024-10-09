from openai import OpenAI, OpenAIError
from os import getenv
import torch
from snac import SNAC
import re

# Generate simple waveforms - this part remains unchanged for now, but the output is used differently
sample_rate = 44100  # Sample rate in Hz
duration = 5  # Duration of audio in seconds
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

# Load the SNAC model (this part is not used in this modified version)
# model = SNAC.from_pretrained("hubertsiuzdak/snac_44khz").eval().to(device)

# Encode the audio data (this part is not used in this modified version)
# with torch.inference_mode():
#     codes = model.encode(audio_data.to(device))

# Placeholder for dynamic token summary - will be updated after LLM response
token_summary = ""

# Add the system message here - Modified to describe the audio, not provide code
system_message = """
You are now an expert in generating well-formatted SNAC token descriptions for audio data. Your primary task is to assist the user by providing accurate and structured descriptions of hypothetical SNAC token sequences. You will receive a description of the audio data.

## SNAC Token Description Generation

Your role is to describe what SNAC tokens for the provided audio data *would* look like.  You will *not* generate fake tokens. Here's a detailed breakdown of the process:

1. Audio Input: The user will provide a description of the audio data. The audio consists of two channels: a 440Hz sine wave and a 220Hz square wave, both 5 seconds long, sampled at 44100 Hz.

2. Hypothetical Tokenization Process: Imagine analyzing this audio signal at different temporal resolutions to create SNAC tokens, capturing both coarse and fine details.

3. Variable-Length Sequences: Imagine organizing tokens into sequences of variable lengths, where each sequence corresponds to a specific temporal resolution.

4. Output Format: Describe the characteristics of the *hypothetical* generated SNAC tokens, including the number of sequences and the length of the longest sequence. Provide the number of tokens in each sequence.  Assume a reasonable tokenization process.

## Formatting Guidelines

- Describe the number of sequences generated.
- Indicate the length of the longest sequence.
- Provide a concise summary of the *hypothetical* token generation process, including the number of tokens in each sequence. Use a structured format like this:
  ```
  Summary of Hypothetical SNAC Tokens:
  Sequence 1: <number of tokens> tokens
  Sequence 2: <number of tokens> tokens
  ...
  Sequence N: <number of tokens> tokens
  Longest Sequence: <number of tokens> tokens
  ```

Focus on delivering accurate and properly formatted descriptions of hypothetical SNAC tokens based on the provided audio data description.
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
        "content": "Describe the hypothetical SNAC tokens for the described audio data. Provide a summary of the hypothetical tokens as described in the system message.",
    }
]

try:
    completion = client.chat.completions.create(
        model="liquid/lfm-40b",
        messages=messages,
    )
    llm_response = completion.choices[0].message.content
    print(f"LLM Response:\n{llm_response}")  # Added for debugging

    # Extract relevant information from LLM response using regular expressions
    sequence_lengths = []
    matches = re.findall(r"Sequence\s*(\d+)\s*:\s*(\d+)\s*tokens", llm_response)
    if matches:
        sequence_lengths = [int(length) for _, length in matches]  # Extract sequence lengths
        num_sequences = len(sequence_lengths)
        longest_sequence = max(sequence_lengths)
        token_summary = f"Generated {num_sequences} sequences of SNAC tokens. Sequence lengths: {sequence_lengths}. Longest sequence: {longest_sequence} tokens."
    else:
        token_summary = "Could not extract sequence information from LLM response."

except OpenAIError as e:
    print(f"OpenAI API Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


print(token_summary)
