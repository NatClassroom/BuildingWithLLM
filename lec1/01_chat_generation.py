# TOPIC 1: Chat Generation
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents={
        "role": "user",
        "parts": [{"text": "Explain how AI works in a few words"}],
    },
)

print(response.text)


## Adding "system instruction" to the request
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents={
        "role": "user",
        "parts": [{"text": "Explain how AI works in a few words"}],
    },
    config={"system_instruction": "Answer like you are a dog."},
)

print(response.text)
