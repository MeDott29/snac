import pickle
from openai import OpenAI
from os import getenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

# Load the cached SNAC codes
try:
    with open("snac_codes.pkl", "rb") as f:
        codes = pickle.load(f)
    logging.info("SNAC codes loaded from cache.")
except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
    logging.error(f"Error loading SNAC codes from cache: {e}")
    exit(1)


# Flatten the codes into a single list
flat_codes = [value for code in codes for value in code[0].tolist()]

# Get model's maximum token limit (if possible, otherwise use a default)
max_tokens = 2000  # Default value
try:
    # Attempt to get the model's token limit (this part might require adjustments depending on the OpenAI API)
    #  This section is commented out because it requires specific API calls that are not provided in the prompt.
    # response = client.models.retrieve(id="liquid/lfm-40b")
    # max_tokens = response.max_tokens
    # logging.info(f"Retrieved max_tokens from model: {max_tokens}")
except Exception as e:
    logging.warning(f"Could not retrieve max_tokens from model: {e}. Using default value of {max_tokens}.")


# Check for excessively long code lists and truncate if necessary
if len(flat_codes) > max_tokens:
    original_length = len(flat_codes)
    truncated_length = max_tokens
    percentage_truncated = (1 - (truncated_length / original_length)) * 100
    flat_codes = flat_codes[:max_tokens]
    logging.warning(f"SNAC code list truncated from {original_length} to {truncated_length} tokens ({percentage_truncated:.2f}% truncated).")

# Convert the (potentially truncated) flat codes list to a string representation
codes_str = " ".join(map(str, flat_codes))

# gets API Key from environment variable OPENAI_API_KEY
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
          "content": f"Here are some SNAC codes for a sine wave (truncated to {len(flat_codes)} for model compatibility): {codes_str}\n\nCan you create codes that layer in another wave, treating these codes like a book that you understand?  Focus on the patterns and relationships within the provided data to generate a coherent and musically pleasing result."
        }
      ]
    )
    print(completion.choices[0].message.content)
except Exception as e:
    logging.error(f"An error occurred during the API call: {e}")
    exit(1)

