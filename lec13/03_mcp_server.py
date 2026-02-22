# TOPIC: MCP Server — Model Context Protocol
# MCP is a standard protocol for exposing tools to LLMs.
# Instead of hardcoding tools, you expose them via MCP and
# any MCP-compatible client can discover and use them.
#
# This file creates an MCP server with 3 tools and demonstrates
# two ways to run it:
#   1. stdio transport  — clients launch this as a subprocess
#   2. SSE transport    — runs as an HTTP server, clients connect over the network
#
# Run demo:    uv run python lec13/03_mcp_server.py
# Run as HTTP: uv run python lec13/03_mcp_server.py --sse
#
# Setup: uv add "mcp>=1.0.0"

from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("DemoTools")


@mcp.tool()
def word_count(text: str) -> str:
    """Count the number of words, characters, and lines in the given text.

    Args:
        text: The text to analyze
    """
    words = len(text.split())
    chars = len(text)
    lines = text.count("\n") + 1
    return f"Words: {words}, Characters: {chars}, Lines: {lines}"


@mcp.tool()
def read_file(path: str) -> str:
    """Read the contents of a file.

    Args:
        path: Path to the file to read
    """
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


@mcp.tool()
def search_in_text(text: str, query: str) -> str:
    """Search for a query string in text and return matching lines.

    Args:
        text: The text to search in
        query: The string to search for
    """
    matches = []
    for i, line in enumerate(text.splitlines(), 1):
        if query.lower() in line.lower():
            matches.append(f"  Line {i}: {line.strip()}")

    if matches:
        return f"Found {len(matches)} match(es):\n" + "\n".join(matches)
    return "No matches found."


# ---------------------------------------------------------------------------
# Demo: connect to our own server and call tools via MCP protocol
# ---------------------------------------------------------------------------

async def demo():
    """Start server as subprocess, connect as MCP client, call each tool."""
    import asyncio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from contextlib import AsyncExitStack

    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "lec13/03_mcp_server.py", "--stdio"],
    )

    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        # 1. Discover tools
        tool_list = await session.list_tools()
        print("Discovered tools via MCP:")
        for tool in tool_list.tools:
            print(f"  - {tool.name}: {tool.description.splitlines()[0]}")
        print()

        # 2. Call each tool through the MCP protocol
        print("=" * 60)
        print("Calling tools via MCP protocol")
        print("=" * 60)

        # Tool 1: word_count
        print("\n[word_count]")
        result = await session.call_tool("word_count", {"text": "Hello world from MCP!\nThis is line two."})
        print(f"  Input: 'Hello world from MCP!\\nThis is line two.'")
        print(f"  Result: {result.content[0].text}")

        # Tool 2: read_file
        print("\n[read_file]")
        result = await session.call_tool("read_file", {"path": "pyproject.toml"})
        content = result.content[0].text
        print(f"  Input: 'pyproject.toml'")
        print(f"  Result: ({len(content)} chars) {content[:80]}...")

        # Tool 3: search_in_text
        print("\n[search_in_text]")
        result = await session.call_tool("search_in_text", {"text": content, "query": "google"})
        print(f"  Input: text=<pyproject.toml>, query='google'")
        print(f"  Result: {result.content[0].text}")

    print("\n" + "=" * 60)
    print("All tools called successfully via MCP protocol!")


if __name__ == "__main__":
    import sys

    if "--stdio" in sys.argv:
        # Mode 1: stdio transport (used by MCP clients like 03_mcp_client.py)
        mcp.run(transport="stdio")

    elif "--sse" in sys.argv:
        # Mode 2: SSE transport (HTTP server — accessible over the network)
        # Clients can connect at http://localhost:8000/sse
        print("Starting MCP server on http://localhost:8000/sse")
        print("Any MCP client can now connect over HTTP.")
        print("Press Ctrl+C to stop.\n")
        mcp.run(transport="sse")

    else:
        # Default: run interactive demo
        import asyncio
        asyncio.run(demo())
