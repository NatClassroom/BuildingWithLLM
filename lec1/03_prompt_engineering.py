# TOPIC 3: Prompt engineering
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

few_shot_contents = [
    {
        "role": "user",
        "parts": [
            {
                "text": (
                    "You are a cuisine classifier. "
                    "Given the name of a dish, respond with the most likely cuisine."
                )
            }
        ],
    },
    {
        "role": "user",
        "parts": [{"text": "Dish: Sushi"}],
    },
    {
        "role": "model",
        "parts": [{"text": "Cuisine: Japanese"}],
    },
    {
        "role": "user",
        "parts": [{"text": "Dish: Tacos al pastor"}],
    },
    {
        "role": "model",
        "parts": [{"text": "Cuisine: Mexican"}],
    },
    {
        "role": "user",
        "parts": [{"text": "Dish: Butter chicken"}],
    },
]

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=few_shot_contents,
)

print(response.text)
