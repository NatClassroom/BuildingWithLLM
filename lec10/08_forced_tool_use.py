# TOPIC: Controlling Tool Use
# Force the model to use tools, or let it decide

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_stock_price(symbol: str) -> dict:
    """Get current stock price.

    Args:
        symbol: Stock ticker symbol
    """
    # Fake prices
    prices = {"AAPL": 185.50, "GOOGL": 140.25, "MSFT": 378.90}
    return {"symbol": symbol, "price": prices.get(symbol.upper(), 100.00)}


tools = [get_stock_price]

# 1. AUTO mode (default) - Model decides whether to use tools
print("=" * 50)
print("MODE: AUTO (model decides)")
print("=" * 50)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What's 2 + 2?",  # Doesn't need the tool
    config=types.GenerateContentConfig(
        tools=tools,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="AUTO")
        ),
    ),
)
print(f"Query: What's 2 + 2?")
print(f"Response: {response.text}")

# 2. ANY mode - Force model to use at least one tool
print("\n" + "=" * 50)
print("MODE: ANY (must use a tool)")
print("=" * 50)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Tell me about Apple stock",
    config=types.GenerateContentConfig(
        tools=tools,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="ANY")
        ),
    ),
)
print(f"Query: Tell me about Apple stock")
print(f"Response: {response.text}")

# 3. NONE mode - Disable tool use entirely
print("\n" + "=" * 50)
print("MODE: NONE (tools disabled)")
print("=" * 50)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What's the price of AAPL?",  # Would normally use tool
    config=types.GenerateContentConfig(
        tools=tools,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="NONE")
        ),
    ),
)
print(f"Query: What's the price of AAPL?")
print(f"Response: {response.text}")
