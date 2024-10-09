import pickle
from openai import OpenAI
from os import getenv

# Load the cached SNAC codes
with open("snac_codes.pkl", "rb") as f:
    codes = pickle.load(f)

# Flatten the codes into a single list
flat_codes = [value for code in codes for value in code[0].tolist()]

# Check for excessively long code lists and truncate if necessary
max_tokens = 2000  # Adjust this based on the model's token limit and desired level of detail
if len(flat_codes) > max_tokens:
    flat_codes = flat_codes[:max_tokens]
    print("Warning: SNAC code list truncated due to length exceeding model token limit.")

# Convert the (potentially truncated) flat codes list to a string representation
codes_str = " ".join(map(str, flat_codes))

# gets API Key from environment variable OPENAI_API_KEY
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=getenv("OPENROUTER_API_KEY"),
)

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

