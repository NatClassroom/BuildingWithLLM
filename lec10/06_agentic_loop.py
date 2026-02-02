# TOPIC: Agentic Loop
# Agent keeps calling tools until the task is complete (manual control)

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# Simulated tools for a research agent
def search(query: str) -> dict:
    """Search for information.

    Args:
        query: Search query
    """
    # Fake search results
    results = {
        "python creator": "Python was created by Guido van Rossum in 1991.",
        "python features": "Python features: dynamic typing, garbage collection, multi-paradigm.",
        "guido van rossum": "Guido van Rossum is a Dutch programmer, born in 1956.",
    }
    for key, value in results.items():
        if key in query.lower():
            return {"result": value}
    return {"result": "No results found"}


def take_notes(note: str) -> dict:
    """Save a note for later.

    Args:
        note: The note to save
    """
    notes.append(note)
    return {"status": "saved", "total_notes": len(notes)}


def finish(summary: str) -> dict:
    """Signal that research is complete.

    Args:
        summary: Final summary of findings
    """
    return {"done": True, "summary": summary}


# Storage
notes = []

# Tool declarations with auto-calling disabled so we control the loop
tools = [search, take_notes, finish]
config = types.GenerateContentConfig(
    tools=tools,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

# Initial request
messages = [
    types.Content(
        role="user",
        parts=[types.Part(text="Research who created Python and summarize your findings. Use search, take notes, then finish.")],
    )
]

print("Starting agentic loop...\n")

# Agentic loop - keep going until agent calls "finish"
for step in range(10):  # Max 10 steps
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages,
        config=config,
    )

    # Check if model wants to call a function
    part = response.candidates[0].content.parts[0]

    if hasattr(part, "function_call") and part.function_call:
        fc = part.function_call
        print(f"Step {step + 1}: Calling {fc.name}({fc.args})")

        # Execute the function
        func = {"search": search, "take_notes": take_notes, "finish": finish}[fc.name]
        result = func(**fc.args)
        print(f"         Result: {result}\n")

        # Add to conversation
        messages.append(response.candidates[0].content)
        messages.append(
            types.Content(
                role="user",
                parts=[types.Part.from_function_response(name=fc.name, response=result)],
            )
        )

        # Check if done
        if fc.name == "finish":
            print("Agent finished!")
            print(f"Notes collected: {notes}")
            break
    else:
        print(f"Final response: {response.text}")
        break
