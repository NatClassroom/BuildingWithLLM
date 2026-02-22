# TOPIC: MCP Server — Model Context Protocol
# MCP is a standard protocol for exposing tools to LLMs.
# Instead of hardcoding tools, you expose them via MCP and
# any MCP-compatible client can discover and use them.
#
# This file creates an MCP server with 3 tools.
# Run it as a standalone process, then connect from 03_mcp_client.py.
#
# Setup: uv add "mcp>=1.0.0"
# Run:   uv run python lec13/02_mcp_server.py

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


if __name__ == "__main__":
    # Run server over stdio (the standard MCP transport)
    # Clients launch this process and communicate via stdin/stdout
    print("Starting MCP server (stdio transport)...")
    mcp.run(transport="stdio")
