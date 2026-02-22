# TOPIC: Composable Skills
# A "Skill" packages a system prompt + tools + config into a reusable unit.
# Instead of building one monolithic agent, you build focused skills
# and route user requests to the right one.
#
# Skill = system prompt + tools + execution logic
#
# Run: uv run python lec13/04_skill.py

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import subprocess
import tempfile
import json

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ---------------------------------------------------------------------------
# Tools (shared or skill-specific)
# ---------------------------------------------------------------------------

def run_python(code: str) -> dict:
    """Execute Python code and return the output.

    Args:
        code: Python code to execute
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["python", tmp_path], capture_output=True, text=True, timeout=10
        )
        return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "TIMEOUT", "returncode": -1}
    finally:
        os.unlink(tmp_path)


def explain(explanation: str) -> dict:
    """Provide a clear explanation to the user.

    Args:
        explanation: The explanation text
    """
    return {"status": "delivered"}


# ---------------------------------------------------------------------------
# Skill definitions
# ---------------------------------------------------------------------------

class Skill:
    """A reusable agent capability: system prompt + tools + config."""

    def __init__(self, name: str, description: str, system_prompt: str, tools: list):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.tools = tools
        self.config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def __repr__(self):
        return f"Skill({self.name!r})"


# Skill 1: Data Analysis
data_analysis_skill = Skill(
    name="data_analysis",
    description="Analyze data, compute statistics, create tables. Uses Python code execution.",
    system_prompt=(
        "You are a data analysis expert. When given a question about data, "
        "write Python code to compute the answer. Use pandas, statistics, or "
        "basic Python. Always print results clearly. Iterate if code has errors."
    ),
    tools=[run_python],
)

# Skill 2: Code Explainer
code_explainer_skill = Skill(
    name="code_explainer",
    description="Explain code step by step. Runs code to verify behavior if needed.",
    system_prompt=(
        "You are a patient code teacher. When given code, explain it step by step. "
        "If helpful, run the code to demonstrate its behavior. Use the explain tool "
        "to deliver your explanation, and run_python to demonstrate."
    ),
    tools=[run_python, explain],
)

# ---------------------------------------------------------------------------
# Skill registry & router
# ---------------------------------------------------------------------------

ALL_SKILLS = [data_analysis_skill, code_explainer_skill]


def route_to_skill(user_query: str) -> Skill:
    """Use the LLM to pick the best skill for the query."""
    skill_descriptions = "\n".join(
        f"- {s.name}: {s.description}" for s in ALL_SKILLS
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Pick the best skill for this query. Reply with ONLY the skill name.\n\nSkills:\n{skill_descriptions}\n\nQuery: {user_query}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {"skill": {"type": "string"}},
                "required": ["skill"],
            },
        ),
    )

    chosen = json.loads(response.text)["skill"]
    for skill in ALL_SKILLS:
        if skill.name == chosen:
            print(f"Router → selected skill: {skill.name}")
            return skill

    # Default fallback
    print(f"Router → defaulting to: {ALL_SKILLS[0].name}")
    return ALL_SKILLS[0]


def execute_skill(skill: Skill, user_query: str) -> str:
    """Run the agentic loop using the selected skill."""
    tool_map = {}
    for tool in skill.tools:
        tool_map[tool.__name__] = tool

    messages = [
        types.Content(role="user", parts=[types.Part(text=user_query)])
    ]

    for step in range(10):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages,
            config=skill.config,
        )

        candidate = response.candidates[0]
        parts = candidate.content.parts
        function_calls = [p for p in parts if hasattr(p, "function_call") and p.function_call]

        if function_calls:
            fc = function_calls[0].function_call
            print(f"  Step {step + 1}: {fc.name}({dict(fc.args)})")

            func = tool_map[fc.name]
            result = func(**fc.args)

            messages.append(candidate.content)
            messages.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(name=fc.name, response=result)],
                )
            )
        else:
            return response.text

    return "Max steps reached."


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    queries = [
        "Given the list [23, 45, 12, 67, 34, 89, 2], compute the mean, median, and standard deviation.",
        "Explain what this code does step by step: sorted(set(x**2 for x in range(10) if x % 2 == 0))",
    ]

    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"User: {query}\n")

        # Step 1: Route to the right skill
        skill = route_to_skill(query)

        # Step 2: Execute with that skill
        answer = execute_skill(skill, query)

        print(f"\nAnswer:\n{answer}")
