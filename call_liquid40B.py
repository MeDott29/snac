from openai import OpenAI, OpenAIError
from os import getenv
import torch
from snac import SNAC

# Generate a sample audio waveform
sample_rate = 44100  # Sample rate in Hz
duration = 5  # Duration of audio in seconds
num_samples = sample_rate * duration
audio_data = torch.rand(1, 1, num_samples)

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU.")


# Load the SNAC model
model = SNAC.from_pretrained("hubertsiuzdak/snac_44khz").eval().to(device)

# Encode the audio data
with torch.inference_mode():
    codes = model.encode(audio_data.to(device))

# Placeholder for dynamic token summary - will be updated after LLM response
token_summary = ""


# Add the system message here
system_message = """
You are now an expert in generating well-formatted SNAC tokens for audio data. Your primary task is to assist the user by providing accurate and structured token sequences.  You will receive a summary of the token generation process, not the raw tokens themselves due to size limitations.

## SNAC Token Generation

Your role is to generate SNAC tokens based on the provided audio data. Here's a detailed breakdown of the process:

1. Audio Input: The user will provide audio data, typically in the form of a waveform or spectrogram.

2. Tokenization Process: Analyze the audio signal at different temporal resolutions to create SNAC tokens, capturing both coarse and fine details.

3. Variable-Length Sequences: Organize tokens into sequences of variable lengths, where each sequence corresponds to a specific temporal resolution.

4. Output Format:  Describe the characteristics of the generated SNAC tokens, including the number of sequences and the length of the longest sequence.  Provide the number of tokens in each sequence.

## Formatting Guidelines

- Describe the number of sequences generated.
- Indicate the length of the longest sequence.
- Provide a concise summary of the token generation process, including the number of tokens in each sequence.  Use a structured format like this:
  ```
  Summary of Generated SNAC Tokens:
  Sequence 1: <number of tokens> tokens
  Sequence 2: <number of tokens> tokens
  ...
  Sequence N: <number of tokens> tokens
  Longest Sequence: <number of tokens> tokens
  ```

Focus on delivering accurate and properly formatted SNAC tokens.
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
        "content": "Generate SNAC tokens for the provided audio data.  Provide a summary of the generated tokens as described in the system message.",
    }
]

try:
    completion = client.chat.completions.create(
        model="liquid/lfm-40b",
        messages=messages,
    )
    llm_response = completion.choices[0].message.content
    print(f"LLM Response:\n{llm_response}") #Added for debugging

    #Extract relevant information from LLM response (this part needs refinement based on the actual LLM output format)
    #This is a placeholder and needs to be adapted to your LLM's response structure.
    num_sequences = 0
    longest_sequence = 0
    sequence_lengths = []
    lines = llm_response.strip().split('\n')
    for line in lines:
        if "Sequence" in line and ":" in line:
            parts = line.split(":")
            try:
                num_tokens = int(parts[1].strip().split()[0])
                sequence_lengths.append(num_tokens)
            except (ValueError, IndexError):
                print(f"Warning: Could not parse sequence length from line: {line}")
        elif "Longest Sequence" in line and ":" in line:
            parts = line.split(":")
            try:
                longest_sequence = int(parts[1].strip().split()[0])
            except (ValueError, IndexError):
                print(f"Warning: Could not parse longest sequence length from line: {line}")

    num_sequences = len(sequence_lengths)
    token_summary = f"Generated {num_sequences} sequences of SNAC tokens. Sequence lengths: {sequence_lengths}. Longest sequence: {longest_sequence} tokens."

except OpenAIError as e:
    print(f"OpenAI API Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


print(token_summary)
