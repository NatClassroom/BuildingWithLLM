# TOPIC: Agentic Loop with Finish Reason
# Use finish_reason to control when the agent stops

# KEY DIFFERENCE: OpenAI vs Gemini
# ================================
# OpenAI:  finish_reason = "tool_calls" → model wants to call a tool
#          finish_reason = "stop"       → model is done
#
# Gemini:  finish_reason = "STOP" for BOTH cases!
#          You must check if response contains function_call parts
#          - Has function_call → execute tool, continue loop
#          - No function_call  → model is done, exit loop

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


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


notes = []

tools = [search, take_notes]
config = types.GenerateContentConfig(
    tools=tools,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

messages = [
    types.Content(
        role="user",
        parts=[types.Part(text="Research who created Python. Search for info, take notes, then give me a summary.")],
    )
]

print("Starting agentic loop...\n")

for step in range(10):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages,
        config=config,
    )

    candidate = response.candidates[0]
    finish_reason = candidate.finish_reason

    # Gemini finish_reason values:
    # - STOP: Normal completion (could be text OR tool call!)
    # - MAX_TOKENS: Hit token limit
    # - SAFETY: Blocked for safety
    # - MALFORMED_FUNCTION_CALL: Tool call couldn't be parsed

    print(f"Step {step + 1}: finish_reason = {finish_reason}")

    if finish_reason == "STOP":
        # STOP means model finished, but we need to check WHAT it produced
        part = candidate.content.parts[0]
        has_function_call = hasattr(part, "function_call") and part.function_call

        if has_function_call:
            # Model wants to call a tool (unlike OpenAI, no special finish_reason)
            fc = part.function_call
            print(f"         → Tool call: {fc.name}({fc.args})")

            func = {"search": search, "take_notes": take_notes}[fc.name]
            result = func(**fc.args)
            print(f"         → Result: {result}\n")

            messages.append(candidate.content)
            messages.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(name=fc.name, response=result)],
                )
            )
        else:
            # Model produced text response - we're done
            print(f"         → Text response (no tool call)\n")
            print(f"{'='*50}")
            print(f"AGENT COMPLETE")
            print(f"{'='*50}")
            print(f"Response: {response.text}")
            print(f"\nNotes: {notes}")
            break

    elif finish_reason == "MALFORMED_FUNCTION_CALL":
        print("Error: Model tried to call a function but it was malformed")
        break

    else:
        print(f"Stopped due to: {finish_reason}")
        break
