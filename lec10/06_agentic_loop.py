# TOPIC: Agentic Loop with Finish Reason
# Use finish_reason to control when the agent stops

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


# Storage
notes = []

# Tool declarations with auto-calling disabled so we control the loop
tools = [search, take_notes]
config = types.GenerateContentConfig(
    tools=tools,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

# Initial request
messages = [
    types.Content(
        role="user",
        parts=[types.Part(text="Research who created Python. Search for info, take notes, then give me a summary.")],
    )
]

print("Starting agentic loop...\n")

# Agentic loop - use finish_reason to know when to stop
for step in range(10):  # Max 10 steps as safety limit
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages,
        config=config,
    )

    candidate = response.candidates[0]
    finish_reason = candidate.finish_reason

    print(f"Step {step + 1}: finish_reason = {finish_reason}")

    # Check finish_reason to decide what to do
    # STOP = model is done, no more tool calls
    # Other reasons (like MAX_TOKENS, SAFETY) might need handling too

    if finish_reason == "STOP":
        # Model finished - check if it called a tool or gave final answer
        part = candidate.content.parts[0]

        if hasattr(part, "function_call") and part.function_call:
            # Model wants to call a function
            fc = part.function_call
            print(f"         Tool call: {fc.name}({fc.args})")

            # Execute the function
            func = {"search": search, "take_notes": take_notes}[fc.name]
            result = func(**fc.args)
            print(f"         Result: {result}\n")

            # Add to conversation and continue
            messages.append(candidate.content)
            messages.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(name=fc.name, response=result)],
                )
            )
        else:
            # Model gave a text response - we're done!
            print(f"\n{'='*50}")
            print("AGENT COMPLETE (finish_reason=STOP, no tool call)")
            print(f"{'='*50}")
            print(f"Final response: {response.text}")
            print(f"\nNotes collected: {notes}")
            break
    else:
        # Handle other finish reasons
        print(f"Unexpected finish_reason: {finish_reason}")
        print(f"Response: {response.text}")
        break
