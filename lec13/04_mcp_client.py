# TOPIC: MCP Client — Connecting LLM Agent to MCP Tools
# This script connects to the MCP server (03_mcp_server.py),
# discovers its tools at runtime, and lets a Gemini agent use them.
#
# Key idea: tools are NOT hardcoded — they're discovered via MCP protocol.
# Any MCP server can be plugged in without changing client code.
#
# Run: uv run python lec13/04_mcp_client.py

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def mcp_tools_to_gemini(mcp_tools: list) -> types.Tool:
    """Convert MCP tool definitions to Gemini function declarations."""
    declarations = []
    for tool in mcp_tools:
        # Build properties from MCP input schema
        properties = {}
        required = []
        if tool.inputSchema and "properties" in tool.inputSchema:
            for name, prop in tool.inputSchema["properties"].items():
                properties[name] = {
                    "type": prop.get("type", "string"),
                    "description": prop.get("description", ""),
                }
            required = tool.inputSchema.get("required", [])

        declarations.append({
            "name": tool.name,
            "description": tool.description or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        })
    return types.Tool(function_declarations=declarations)


async def run_agent():
    """Connect to MCP server, discover tools, run Gemini agent."""

    # Step 1: Connect to MCP server
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "lec13/03_mcp_server.py", "--stdio"],
    )

    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        # Step 2: Discover tools (no hardcoding!)
        tool_list = await session.list_tools()
        print("Discovered MCP tools:")
        for tool in tool_list.tools:
            print(f"  - {tool.name}: {tool.description}")
        print()

        # Step 3: Convert MCP tools → Gemini format
        gemini_tools = mcp_tools_to_gemini(tool_list.tools)
        config = types.GenerateContentConfig(
            tools=[gemini_tools],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        # Step 4: Agentic loop — Gemini decides which MCP tools to call
        query = "Read the file 'pyproject.toml', count the words in it, and search for any lines mentioning 'google'."

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

            function_calls = [p for p in parts if hasattr(p, "function_call") and p.function_call]

            if function_calls:
                fc = function_calls[0].function_call
                print(f"\nStep {step + 1}: Calling MCP tool '{fc.name}' with {dict(fc.args)}")

                # Step 5: Call tool through MCP protocol (not directly!)
                mcp_result = await session.call_tool(fc.name, dict(fc.args))
                result_text = mcp_result.content[0].text if mcp_result.content else "No result"

                print(f"  Result: {result_text[:200]}{'...' if len(result_text) > 200 else ''}")

                messages.append(candidate.content)
                messages.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_function_response(
                            name=fc.name,
                            response={"result": result_text},
                        )],
                    )
                )
            else:
                print(f"\n{'=' * 60}")
                print(f"AGENT COMPLETE (after {step + 1} steps)")
                print(f"{'=' * 60}")
                print(f"\n{response.text}")
                break


if __name__ == "__main__":
    asyncio.run(run_agent())
