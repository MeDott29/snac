from openai import OpenAI
from os import getenv
import torch
from snac import SNAC

# Generate a sample audio waveform
sample_rate = 44100  # Sample rate in Hz
duration = 5  # Duration of audio in seconds
num_samples = sample_rate * duration
audio_data = torch.rand(1, 1, num_samples)

# Load the SNAC model
model = SNAC.from_pretrained("hubertsiuzdak/snac_44khz").eval().cuda()

# Encode the audio data
with torch.inference_mode():
    codes = model.encode(audio_data.cuda())

# Summarize SNAC tokens for LLM
token_summary = f"Generated {len(codes)} sequences of SNAC tokens.  Sequence lengths vary, with the longest sequence having {max(len(code) for code in codes)} tokens."


# Add the system message here
system_message = """
You are now an expert in generating well-formatted SNAC tokens for audio data. Your primary task is to assist the user by providing accurate and structured token sequences.  You will receive a summary of the token generation process, not the raw tokens themselves due to size limitations.

## SNAC Token Generation

Your role is to generate SNAC tokens based on the provided audio data. Here's a detailed breakdown of the process:

1. Audio Input: The user will provide audio data, typically in the form of a waveform or spectrogram.

2. Tokenization Process: Analyze the audio signal at different temporal resolutions to create SNAC tokens, capturing both coarse and fine details.

3. Variable-Length Sequences: Organize tokens into sequences of variable lengths, where each sequence corresponds to a specific temporal resolution.

4. Output Format:  Describe the characteristics of the generated SNAC tokens, including the number of sequences and the length of the longest sequence.

## Formatting Guidelines

- Describe the number of sequences generated.
- Indicate the length of the longest sequence.
- Provide a concise summary of the token generation process.

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
        "content": "Here's an example of a summary of properly formatted SNAC tokens generated from audio ",
    },
    {
        "role": "assistant",
        "content": token_summary,
    },
    {
        "role": "user",
        "content": "Generate SNAC tokens for the provided audio data.  Provide a summary of the generated tokens as described in the system message.",
    }
]

completion = client.chat.completions.create(
    model="liquid/lfm-40b",
    messages=messages,
)

print(completion.choices[0].message.content)
