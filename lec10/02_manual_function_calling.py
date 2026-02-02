# TOPIC: Manual Function Calling
# Understanding what happens under the hood

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# Step 1: Define your function
def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    fake_weather = {
        "tokyo": {"temp": 22, "condition": "sunny"},
        "london": {"temp": 15, "condition": "rainy"},
    }
    return fake_weather.get(city.lower(), {"temp": 20, "condition": "unknown"})


# Step 2: Declare the function schema for the model
weather_tool = types.Tool(
    function_declarations=[
        {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        }
    ]
)

# Step 3: Send request to model
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What's the weather in Tokyo?",
    config=types.GenerateContentConfig(tools=[weather_tool]),
)

# Step 4: Check if model wants to call a function
function_call = response.candidates[0].content.parts[0].function_call
print(f"Model wants to call: {function_call.name}")
print(f"With arguments: {function_call.args}")

# Step 5: Execute the function ourselves
result = get_weather(**function_call.args)
print(f"Function returned: {result}")

# Step 6: Send the result back to continue the conversation
final_response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Content(role="user", parts=[types.Part(text="What's the weather in Tokyo?")]),
        response.candidates[0].content,  # Model's function call
        types.Content(
            role="user",
            parts=[types.Part.from_function_response(name="get_weather", response=result)],
        ),
    ],
    config=types.GenerateContentConfig(tools=[weather_tool]),
)

print(f"\nFinal answer: {final_response.text}")
