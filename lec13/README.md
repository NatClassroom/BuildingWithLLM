# Lecture 13: Agentic Code Execution, MCP & Skills

Three patterns for building powerful, modular agents.

## Files

| File | Topic | Description |
|------|-------|-------------|
| `01_code_execution.py` | **Code Execution** | Agent that writes and runs Python code (subprocess) |
| `02_docker_code_execution.py` | **Sandboxed Execution** | Same agent, but code runs in a Docker container |
| `03_mcp_server.py` | **MCP Server** | Expose tools via Model Context Protocol |
| `04_mcp_client.py` | **MCP Client** | Discover and use MCP tools from a Gemini agent |
| `05_skill.py` | **Skills** | Composable skill routing (system prompt + tools) |

## Key Concepts

### Code Execution — The Ultimate Tool
```
User query → LLM generates code → Execute → Observe output → Iterate → Final answer
```
A `run_python` tool gives the agent Turing-complete capabilities. It can solve any computable problem by writing code.

### Sandboxing — Don't Trust LLM-Generated Code
| Approach | Isolation | Setup |
|----------|-----------|-------|
| `subprocess` | None (time limit only) | Zero |
| **Docker** | Network, memory, CPU, filesystem | Docker installed |
| gVisor / Firecracker | VM-level | More complex |
| E2B, Modal, Daytona | Hosted sandbox API | API key |

`02_docker_code_execution.py` demonstrates Docker sandboxing with:
- `--network none` — no internet access
- `--memory 128m` — capped RAM
- `--cpus 0.5` — capped CPU
- `--read-only` — immutable filesystem

### MCP — Model Context Protocol
```
MCP Server (exposes tools) ←stdio/sse→ MCP Client (discovers & calls tools) → LLM Agent
```
MCP standardizes how tools are exposed. Instead of hardcoding tool definitions, clients discover them at runtime. Any MCP server works with any MCP client.

### Skills — Composable Agent Capabilities
```
User query → Router (picks skill) → Skill (system prompt + tools + loop) → Answer
```
A Skill packages: system prompt + tools + config. Build focused skills, then route queries to the right one.

## Running the Examples

```bash
# Code execution agent (bare subprocess)
uv run python lec13/01_code_execution.py

# Code execution agent (Docker sandbox)
uv run python lec13/02_docker_code_execution.py

# MCP server demo (calls tools via MCP protocol)
uv run python lec13/03_mcp_server.py

# MCP client + Gemini agent
uv run python lec13/04_mcp_client.py

# Skills with routing
uv run python lec13/05_skill.py
```
