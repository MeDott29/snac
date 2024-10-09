from openai import OpenAI
from os import getenv
import json

# Load the generated data
with open('complex_audio_data.json', 'r') as f:
    audio_data = json.load(f)

# Create a system message to teach Liquid-40B about SNAC codes
system_message = """You are an AI trained to understand and generate SNAC (Multi-Scale Neural Audio Codec) codes for complex audio waveforms. Given a description of audio waveforms, you should be able to predict the corresponding SNAC codes.

SNAC encodes audio into hierarchical tokens at different temporal resolutions. The codes are represented as lists of integer arrays, where each array corresponds to a different temporal scale.

Here are some examples of waveform descriptions and their corresponding SNAC codes:

"""

# Add examples from our generated data
for i, sample in enumerate(audio_data[:5]):  # Use the first 5 samples as examples
    system_message += f"Example {i + 1}:\n"
    system_message += f"Waveforms: {sample['waveforms']}\n"
    system_message += f"SNAC Codes: {sample['snac_codes']}\n\n"

system_message += "Based on these examples, try to understand the relationship between the waveform parameters and the resulting SNAC codes. When given new waveform descriptions, attempt to generate plausible SNAC codes that would represent the described audio."

# gets API Key from environment variable OPENAI_API_KEY
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=getenv("OPENROUTER_API_KEY"),
)

completion = client.chat.completions.create(
  model="liquid/lfm-40b",
  messages=[
    {"role": "system", "content": system_message},
user_prompt = """Generate SNAC codes for an audio sample with the following waveforms: 
[{'freq': 440, 'waveform_type': 'sine', 'amplitude': 0.8}, 
 {'freq': 880, 'waveform_type': 'square', 'amplitude': 0.6}]

Remember, SNAC codes should be a list of integer arrays. Each array represents a different temporal scale. 
Don't explain the process, just provide the codes in the format shown in the examples."""

# Update the messages in the API call
messages=[
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
]
print(completion.choices[0].message.content)
