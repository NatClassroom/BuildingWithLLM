# Lecture 13: Agentic Code Execution, MCP & Skills

Three patterns for building powerful, modular agents.

## Files

| File | Topic | Description |
|------|-------|-------------|
| `01_code_execution.py` | **Code Execution** | Agent that writes and runs Python code |
| `02_mcp_server.py` | **MCP Server** | Expose tools via Model Context Protocol |
| `03_mcp_client.py` | **MCP Client** | Discover and use MCP tools from a Gemini agent |
| `04_skill.py` | **Skills** | Composable skill routing (system prompt + tools) |

## Key Concepts

### Code Execution — The Ultimate Tool
```
User query → LLM generates code → Execute → Observe output → Iterate → Final answer
```
A `run_python` tool gives the agent Turing-complete capabilities. It can solve any computable problem by writing code.

### MCP — Model Context Protocol
```
MCP Server (exposes tools) ←stdio→ MCP Client (discovers & calls tools) → LLM Agent
```
MCP standardizes how tools are exposed. Instead of hardcoding tool definitions, clients discover them at runtime. Any MCP server works with any MCP client.

### Skills — Composable Agent Capabilities
```
User query → Router (picks skill) → Skill (system prompt + tools + loop) → Answer
```
A Skill packages: system prompt + tools + config. Build focused skills, then route queries to the right one.

## Running the Examples

```bash
# Code execution agent
uv run python lec13/01_code_execution.py

# MCP: server runs as subprocess, client launches it automatically
uv run python lec13/03_mcp_client.py

# Skills with routing
uv run python lec13/04_skill.py
```
