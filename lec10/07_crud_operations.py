# TOPIC: CRUD Operations
# Give the agent ability to Create, Read, Update, Delete data

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Simple todo database
todos = {}
next_id = 1


def create_todo(title: str, priority: str = "medium") -> dict:
    """Create a new todo item.

    Args:
        title: The todo title
        priority: Priority level (low, medium, high)
    """
    global next_id
    todos[next_id] = {"id": next_id, "title": title, "priority": priority, "done": False}
    result = {"created": todos[next_id]}
    next_id += 1
    return result


def list_todos() -> dict:
    """List all todo items."""
    return {"todos": list(todos.values())}


def update_todo(id: int, done: bool = None, title: str = None, priority: str = None) -> dict:
    """Update a todo item.

    Args:
        id: The todo ID to update
        done: Mark as done/undone
        title: New title
        priority: New priority
    """
    if id not in todos:
        return {"error": f"Todo {id} not found"}
    if done is not None:
        todos[id]["done"] = done
    if title is not None:
        todos[id]["title"] = title
    if priority is not None:
        todos[id]["priority"] = priority
    return {"updated": todos[id]}


def delete_todo(id: int) -> dict:
    """Delete a todo item.

    Args:
        id: The todo ID to delete
    """
    if id not in todos:
        return {"error": f"Todo {id} not found"}
    deleted = todos.pop(id)
    return {"deleted": deleted}


# Natural language todo management
tools = [create_todo, list_todos, update_todo, delete_todo]
config = types.GenerateContentConfig(tools=tools)

queries = [
    "Add a high priority todo: Finish the project report",
    "Add a todo: Buy groceries",
    "Add a low priority todo: Clean the garage",
    "Show me all my todos",
    "Mark the first todo as done",
    "Delete the grocery todo",
    "What todos do I have left?",
]

for query in queries:
    print(f"\nUser: {query}")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query,
        config=config,
    )
    print(f"Agent: {response.text}")

print(f"\n[Debug] Final database state: {todos}")
