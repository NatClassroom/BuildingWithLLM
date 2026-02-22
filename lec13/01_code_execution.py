# TOPIC: Agentic Code Execution
# Give the LLM a "run_python" tool so it can write and execute code.
# This is the most powerful affordance: a Turing-complete tool.
#
# Pattern: generate code → execute → observe output → iterate
#
# Run: uv run python lec13/01_code_execution.py

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import subprocess
import tempfile

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def run_python(code: str) -> dict:
    """Execute Python code and return the output.

    Args:
        code: Python code to execute
    """
    # Write code to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=10,  # Kill after 10 seconds
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "TIMEOUT: Code took too long", "returncode": -1}
    finally:
        os.unlink(tmp_path)


# Agent loop with code execution
tools = [run_python]
config = types.GenerateContentConfig(
    system_instruction=(
        "You are a Python coding agent. When asked a question, write Python code "
        "to solve it using the run_python tool. Examine the output and iterate if "
        "there are errors. When done, provide a clear final answer."
    ),
    tools=tools,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

if __name__ == "__main__":
    query = "Calculate the first 20 Fibonacci numbers, then find which ones are also prime numbers."

    messages = [
        types.Content(role="user", parts=[types.Part(text=query)])
    ]

    print(f"User: {query}\n")
    print("=" * 60)

    for step in range(10):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages,
            config=config,
        )

        candidate = response.candidates[0]
        parts = candidate.content.parts

        # Check for function calls (Gemini can return multiple parts)
        function_calls = [p for p in parts if hasattr(p, "function_call") and p.function_call]

        if function_calls:
            fc = function_calls[0].function_call
            print(f"\nStep {step + 1}: Agent writes code")
            print(f"--- code ---")
            print(fc.args.get("code", ""))
            print(f"--- end code ---\n")

            # Execute the code
            result = run_python(**fc.args)

            if result["stdout"]:
                print(f"stdout: {result['stdout']}")
            if result["stderr"]:
                print(f"stderr: {result['stderr']}")

            # Feed result back to model
            messages.append(candidate.content)
            messages.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(name=fc.name, response=result)],
                )
            )
        else:
            # Text response — agent is done
            print(f"\n{'=' * 60}")
            print(f"AGENT COMPLETE (after {step + 1} steps)")
            print(f"{'=' * 60}")
            print(f"\n{response.text}")
            break
