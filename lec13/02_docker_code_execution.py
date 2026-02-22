# TOPIC: Sandboxed Code Execution with Docker
# In 01 we used bare subprocess — no real isolation.
# Here we run generated code inside a Docker container with:
#   - No network access  (--network none)
#   - Memory limit       (--memory 128m)
#   - CPU limit          (--cpus 0.5)
#   - Read-only FS       (--read-only)
#   - Auto-cleanup       (--rm)
#
# The LLM-generated code can't escape, can't phone home, can't eat resources.
#
# Requires: Docker installed and running
# Run: uv run python lec13/02_docker_code_execution.py

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import subprocess

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

DOCKER_IMAGE = "python:3.13-slim"


def run_python_sandboxed(code: str) -> dict:
    """Execute Python code in a sandboxed Docker container.

    The container has no network access, limited memory/CPU,
    and a read-only filesystem. This is safe to run untrusted code.

    Args:
        code: Python code to execute
    """
    try:
        result = subprocess.run(
            [
                "docker", "run",
                "--rm",                     # Remove container after exit
                "-i",                       # Accept stdin
                "--network", "none",        # No network access
                "--memory", "128m",         # Max 128MB RAM
                "--cpus", "0.5",            # Max half a CPU core
                "--read-only",              # Read-only filesystem
                "--tmpfs", "/tmp:size=10m", # Small writable /tmp
                DOCKER_IMAGE,
                "python", "-c", code,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "TIMEOUT: Container killed after 30s", "returncode": -1}
    except FileNotFoundError:
        return {"stdout": "", "stderr": "ERROR: Docker not found. Is Docker installed and running?", "returncode": -1}


# Agent config
tools = [run_python_sandboxed]
config = types.GenerateContentConfig(
    system_instruction=(
        "You are a Python coding agent. Write Python code to answer questions "
        "using the run_python_sandboxed tool. The code runs in a sandboxed "
        "container with no network and limited resources. Only the Python "
        "standard library is available. Iterate if there are errors."
    ),
    tools=tools,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

if __name__ == "__main__":
    # Verify Docker is available
    docker_check = subprocess.run(["docker", "info"], capture_output=True)
    if docker_check.returncode != 0:
        print("ERROR: Docker is not running. Please start Docker first.")
        exit(1)

    print(f"Using Docker image: {DOCKER_IMAGE}")

    # Pull image if needed
    subprocess.run(["docker", "pull", "-q", DOCKER_IMAGE], capture_output=True)

    query = "Write code to find all prime numbers under 100, group them by their last digit, and show the count for each group."

    messages = [
        types.Content(role="user", parts=[types.Part(text=query)])
    ]

    print(f"\nUser: {query}\n")
    print("=" * 60)

    for step in range(10):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages,
            config=config,
        )

        candidate = response.candidates[0]
        parts = candidate.content.parts
        function_calls = [p for p in parts if hasattr(p, "function_call") and p.function_call]

        if function_calls:
            fc = function_calls[0].function_call
            print(f"\nStep {step + 1}: Agent writes code (runs in Docker container)")
            print(f"--- code ---")
            print(fc.args.get("code", ""))
            print(f"--- end code ---\n")

            result = run_python_sandboxed(**fc.args)

            if result["stdout"]:
                print(f"stdout: {result['stdout']}")
            if result["stderr"]:
                print(f"stderr: {result['stderr']}")

            messages.append(candidate.content)
            messages.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(name=fc.name, response=result)],
                )
            )
        else:
            print(f"\n{'=' * 60}")
            print(f"AGENT COMPLETE (after {step + 1} steps)")
            print(f"{'=' * 60}")
            print(f"\n{response.text}")
            break
