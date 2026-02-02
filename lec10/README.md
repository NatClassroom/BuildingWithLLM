# Lecture 10: Function Calling

Function calling allows LLMs to use external tools and APIs to perform actions beyond text generation.

## How It Works

```
User Query → Model → [Decides to call function] → Your Code Executes → Result → Model → Final Response
```

1. You define functions (tools) the model can use
2. Model analyzes the user's request
3. Model decides which function to call and with what arguments
4. Your code executes the function
5. Result is sent back to the model
6. Model generates the final response using the result

## Files

### Basics
| File | Description |
|------|-------------|
| `01_auto_function_calling.py` | Simplest approach - SDK handles everything |
| `02_manual_function_calling.py` | Step-by-step to understand the mechanics |
| `03_multiple_tools.py` | Model picks the right tool for each query |

### Agent Affordances
| File | Affordance | Description |
|------|------------|-------------|
| `04_memory_tool.py` | **Memory** | Agent can save/recall information |
| `05_chart_creation.py` | **Visualization** | Agent can create bar, pie, line charts |
| `06_agentic_loop.py` | **Multi-step** | Agent keeps calling tools until done |
| `07_crud_operations.py` | **Data Management** | Agent can Create, Read, Update, Delete |
| `08_forced_tool_use.py` | **Control Modes** | Force or disable tool use |

## Running the Examples

```bash
uv run python lec10/01_auto_function_calling.py
uv run python lec10/02_manual_function_calling.py
# ... etc
```

## Key Concepts

### Tool Definition
```python
def my_tool(param: str) -> dict:
    """Description shown to the model.

    Args:
        param: Description of this parameter
    """
    return {"result": "..."}
```

### Affordances You Can Give Agents

| Affordance | What It Enables |
|------------|-----------------|
| **Memory** | Remember user preferences, conversation history |
| **Visualization** | Create charts, graphs, diagrams |
| **Web Search** | Access current information |
| **Database** | Persistent storage, CRUD operations |
| **File System** | Read/write files |
| **APIs** | Send emails, create calendar events, etc. |

### Tool Use Modes

| Mode | Behavior |
|------|----------|
| `AUTO` | Model decides whether to use tools (default) |
| `ANY` | Model must use at least one tool |
| `NONE` | Tools disabled, text response only |

### Auto vs Manual Function Calling

- **Auto**: SDK handles execution loop automatically
- **Manual**: You control each step (useful for logging, validation, complex flows)
