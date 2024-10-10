import json
from openai import OpenAI, OpenAIError
from os import getenv
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

# JSONL file path
jsonl_file = "snac_code_generation_log.jsonl"

# Load the JSON data
try:
    with open("complex_audio_data_one_period.json", "r") as f:
        audio_data = json.load(f)
    logging.info("Complex audio data loaded successfully.")
except (FileNotFoundError, json.JSONDecodeError) as e:
    logging.error(f"Error loading audio data: {e}")
    exit(1)

# Construct the system message
# You are an AI trained to understand and generate SNAC codes for audio. Given audio parameters, create the corresponding SNAC codes. SNAC codes are hierarchical and multi-resolution.  They are represented as a JSON array where each element is a list of integers representing codes at a specific temporal resolution. 
system_message = """you imagine audio from approximate calculations and then encode it"""

# Add the examples to the system message.  Limit the number of examples to avoid exceeding token limits.
num_examples = min(5, len(audio_data))
messages = []
for i in range(num_examples):
    waveforms = audio_data[i]["waveforms"]
    snac_codes = audio_data[i]["snac_codes"]
    user_message = f"Example {i+1}:Waveforms: {waveforms}"
    assistant_message = f"SNAC Codes: {snac_codes}"
    messages.append({"role": "user", "content": user_message})
    messages.append({"role": "assistant", "content": assistant_message})
    print(messages)

# Define the user prompt.  Use an example from the dataset not included in the system message examples.
test_index = num_examples # Use the next example for testing
test_waveforms = audio_data[test_index]["waveforms"]
user_prompt = f"Generate SNAC codes for the following waveforms:\n{json.dumps(test_waveforms)}"

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=getenv("OPENROUTER_API_KEY"),
)

try:
    response = client.chat.completions.create(
        model="liquid/lfm-40b",  # Or whichever model you want to use
        messages=[
            {"role": "system", "content": system_message},
            *messages,
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        max_tokens=10000  # Adjust as needed
    )
    generated_codes = response.choices[0].message.content
    print(f"Generated Codes:\n{generated_codes}")

    # Append data to JSONL file
    log_data = {
        "system_message": system_message,
        "user_prompt": user_prompt,
        "generated_codes": generated_codes,
        "test_waveforms": test_waveforms  # Include the input waveforms
    }
    with open(jsonl_file, "a") as f:
        f.write(json.dumps(log_data) + "\n")
    logging.info(f"Data appended to {jsonl_file}")

except OpenAIError as e:
    logging.error(f"OpenAI API error: {e}")

logging.info("SNAC code generation with LLM complete.")