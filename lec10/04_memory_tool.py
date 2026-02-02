# TOPIC: Memory Tool
# Give the agent ability to remember information across the conversation

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Simple in-memory storage (in production: use a database)
memory = {}


def save_memory(key: str, value: str) -> dict:
    """Save information to memory.

    Args:
        key: The key to store the value under
        value: The value to store
    """
    memory[key] = value
    return {"status": "saved", "key": key}


def recall_memory(key: str) -> dict:
    """Recall information from memory.

    Args:
        key: The key to look up
    """
    if key in memory:
        return {"found": True, "value": memory[key]}
    return {"found": False, "message": f"No memory found for '{key}'"}


def list_memories() -> dict:
    """List all stored memories."""
    return {"memories": list(memory.keys())}


# Conversation with memory
tools = [save_memory, recall_memory, list_memories]
config = types.GenerateContentConfig(tools=tools)

queries = [
    "Remember that my favorite color is blue",
    "Remember that my dog's name is Max",
    "What memories do you have about me?",
    "What is my favorite color?",
]

for query in queries:
    print(f"\nUser: {query}")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query,
        config=config,
    )
    print(f"Agent: {response.text}")

print(f"\n[Debug] Memory state: {memory}")
