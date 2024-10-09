import pickle
from openai import OpenAI
from os import getenv
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

# Load the cached SNAC codes
try:
    with open("snac_codes.pkl", "rb") as f:
        codes = pickle.load(f)
    logging.info("SNAC codes loaded from cache.")
except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
    logging.error(f"Error loading SNAC codes from cache: {e}.  Check that snac_codes.pkl exists and is a valid pickle file.")
    exit(1)
except Exception as e:
    logging.exception(f"An unexpected error occurred while loading the SNAC codes: {e}")
    exit(1)


# Flatten the codes into a single list
flat_codes = [value for code in codes for value in code[0].tolist()]

# Get model's maximum token limit (if possible, otherwise use a default)
max_tokens = 32000  # Default value


# Check for excessively long code lists and truncate if necessary
if len(flat_codes) > max_tokens:
    original_length = len(flat_codes)
    truncated_length = max_tokens
    percentage_truncated = (1 - (truncated_length / original_length)) * 100
    flat_codes = flat_codes[:max_tokens]
    logging.warning(f"SNAC code list truncated from {original_length} to {truncated_length} tokens ({percentage_truncated:.2f}% truncated).")

# Convert the (potentially truncated) flat codes list to a string representation
codes_str = " ".join(map(str, flat_codes))

# gets API Key from environment variable OPENROUTER_API_KEY
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=getenv("OPENROUTER_API_KEY"),
)

try:
    completion = client.chat.completions.create(
      model="liquid/lfm-40b",
      messages=[
        {
          "role": "user",
          "content": f"Here are some SNAC codes representing a sine wave (truncated to {len(flat_codes)} tokens for model compatibility): {codes_str}\n\nThese codes represent a single sine wave.  Can you generate additional SNAC codes that, when combined with these, create a richer, more complex soundscape?  Focus on creating a musically pleasing and coherent result by considering the patterns and relationships within the provided data.  The output should be a JSON array of numerical values representing the additional SNAC codes."
        }
      ],
      response_format={"type": "json_object"}
    )
    print(json.dumps(completion.choices[0].message.content, indent=2)) #Print formatted JSON
except Exception as e:
    logging.exception(f"An error occurred during the API call: {e}")
    exit(1)
