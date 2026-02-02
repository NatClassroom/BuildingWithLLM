# TOPIC: Automatic Function Calling
# The SDK handles everything: calling your function and sending results back to the model

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# Step 1: Define your function (the "tool" the model can use)
def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: The city name, e.g. "Tokyo"

    Returns:
        Weather information for the city.
    """
    # In real app, this would call a weather API
    fake_weather = {
        "tokyo": {"temp": 22, "condition": "sunny"},
        "london": {"temp": 15, "condition": "rainy"},
        "new york": {"temp": 18, "condition": "cloudy"},
    }
    return fake_weather.get(city.lower(), {"temp": 20, "condition": "unknown"})


# Step 2: Pass the function as a tool to the model
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What's the weather like in Tokyo?",
    config=types.GenerateContentConfig(
        tools=[get_weather],  # Just pass the function directly!
    ),
)

# The SDK automatically:
# 1. Lets the model decide to call get_weather("Tokyo")
# 2. Executes get_weather("Tokyo") for you
# 3. Sends the result back to the model
# 4. Returns the final response

print(response.text)
