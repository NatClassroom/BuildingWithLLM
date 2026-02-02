# TOPIC: Multiple Tools
# The model picks the right tool based on user intent

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# Define multiple tools the model can choose from
def get_weather(city: str) -> dict:
    """Get current weather for a city.

    Args:
        city: The city name
    """
    return {"city": city, "temp": 22, "condition": "sunny"}


def calculate(expression: str) -> dict:
    """Evaluate a math expression.

    Args:
        expression: Math expression like "2 + 2" or "100 * 0.15"
    """
    try:
        result = eval(expression)  # Note: use safer parser in production!
        return {"expression": expression, "result": result}
    except Exception:
        return {"error": "Invalid expression"}


def search_web(query: str) -> dict:
    """Search the web for information.

    Args:
        query: Search query string
    """
    # Fake search results
    return {"query": query, "results": ["Result 1", "Result 2", "Result 3"]}


# Test different queries - the model picks the appropriate tool
queries = [
    "What's 15% of 200?",
    "What's the weather in Paris?",
    "Find information about Python programming",
]

for query in queries:
    print(f"\n{'='*50}")
    print(f"Query: {query}")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query,
        config=types.GenerateContentConfig(
            tools=[get_weather, calculate, search_web],
        ),
    )

    print(f"Response: {response.text}")
