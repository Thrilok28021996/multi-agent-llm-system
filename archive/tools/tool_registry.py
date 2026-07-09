"""
Tool registry with descriptions and schemas for LLM tool use.
These descriptions are injected into agent system prompts.
"""
from typing import Dict

TOOL_SCHEMAS: Dict[str, Dict] = {
    "bash": {
        "description": "Execute a shell command or Python code. Use for running scripts, tests, calculations, checking output.",
        "parameters": {
            "command": {"type": "string", "description": "The shell command to execute (must start with: python, python3, pip, pytest, git, npm, node, ls, cat, echo, grep, find, curl, wget, cargo, make)"}
        },
        "returns": "stdout output or error message",
        "example": '{"tool": "bash", "command": "python -c \'print(2+2)\'"}'
    },
    "read_file": {
        "description": "Read the contents of a file.",
        "parameters": {
            "path": {"type": "string", "description": "Absolute or relative file path"},
            "offset": {"type": "integer", "description": "Line number to start from (optional, default 0)"},
            "limit": {"type": "integer", "description": "Maximum lines to read (optional)"}
        },
        "returns": "File contents as text",
        "example": '{"tool": "read_file", "path": "main.py"}'
    },
    "write_file": {
        "description": "Write content to a file (creates or overwrites).",
        "parameters": {
            "path": {"type": "string", "description": "File path to write to"},
            "content": {"type": "string", "description": "Content to write"}
        },
        "returns": "Success or error message",
        "example": '{"tool": "write_file", "path": "app.py", "content": "print(\'hello\')"}'
    },
    "search_files": {
        "description": "Search file contents using regex pattern.",
        "parameters": {
            "pattern": {"type": "string", "description": "Regex pattern to search for"},
            "path": {"type": "string", "description": "Directory or file to search in (default: current dir)"},
            "file_pattern": {"type": "string", "description": "File glob pattern like '*.py' (optional)"}
        },
        "returns": "List of matching file:line:content results",
        "example": '{"tool": "search_files", "pattern": "def main", "path": ".", "file_pattern": "*.py"}'
    },
    "list_files": {
        "description": "List files matching a glob pattern.",
        "parameters": {
            "pattern": {"type": "string", "description": "Glob pattern like '**/*.py' or 'src/**'"}
        },
        "returns": "List of matching file paths",
        "example": '{"tool": "list_files", "pattern": "**/*.py"}'
    },
    "web_search": {
        "description": "Search the web for current information, documentation, or solutions.",
        "parameters": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "description": "Maximum results to return (default: 5)"}
        },
        "returns": "List of {title, url, snippet} results",
        "example": '{"tool": "web_search", "query": "FastAPI JWT authentication example", "max_results": 5}'
    },
    "web_fetch": {
        "description": "Fetch content from a URL (documentation, API specs, examples).",
        "parameters": {
            "url": {"type": "string", "description": "URL to fetch"},
            "extract_mode": {"type": "string", "description": "text or markdown (default: text)"}
        },
        "returns": "Page content as text",
        "example": '{"tool": "web_fetch", "url": "https://fastapi.tiangolo.com/tutorial/", "extract_mode": "text"}'
    },
    "edit_file": {
        "description": "Edit a specific section of a file by replacing old text with new text.",
        "parameters": {
            "path": {"type": "string", "description": "File path"},
            "old_text": {"type": "string", "description": "Exact text to replace (must match uniquely)"},
            "new_text": {"type": "string", "description": "Replacement text"}
        },
        "returns": "Success or error",
        "example": '{"tool": "edit_file", "path": "app.py", "old_text": "def old():", "new_text": "def new():"}'
    },
}


def get_tools_for_role(role: str) -> Dict[str, Dict]:
    """Return appropriate tools for each agent role."""
    base_tools = ["read_file", "search_files", "list_files"]

    role_tools = {
        "developer": list(TOOL_SCHEMAS.keys()),  # All tools
        "qa_engineer": ["bash", "read_file", "search_files", "list_files", "write_file"],
        "researcher": ["web_search", "web_fetch", "read_file", "search_files"],
        "cto": ["read_file", "search_files", "list_files", "web_search"],
        "ceo": ["web_search", "read_file"],
        "devops_engineer": ["bash", "read_file", "write_file", "search_files", "list_files"],
        "security_engineer": ["bash", "read_file", "search_files", "list_files", "web_search"],
        "product_manager": ["web_search", "web_fetch", "read_file"],
        "data_analyst": ["bash", "read_file", "web_search"],
    }

    tools = role_tools.get(role, base_tools)
    return {name: TOOL_SCHEMAS[name] for name in tools if name in TOOL_SCHEMAS}


def format_tools_for_prompt(tools: Dict[str, Dict]) -> str:
    """Format tool schemas into a system prompt section."""
    if not tools:
        return ""

    lines = [
        "\n## AVAILABLE TOOLS",
        "You can use tools by including tool calls in your response using this exact format:",
        "",
        "<tool_use>",
        '{"tool": "tool_name", "param1": "value1", "param2": "value2"}',
        "</tool_use>",
        "",
        "You will see the result, then continue. Use tools when you need to:",
        "- Run code to verify it works",
        "- Read files to understand existing code",
        "- Search for solutions or documentation",
        "- Write or modify files",
        "",
        "### TOOLS:",
    ]

    for name, schema in tools.items():
        lines.append(f"\n**{name}**: {schema['description']}")
        param_str = ', '.join(f'{k} ({v["type"]})' for k, v in schema['parameters'].items())
        lines.append(f"  Parameters: {param_str}")
        lines.append(f"  Example: {schema['example']}")

    lines.append("\nIMPORTANT: Only call tools when needed. For final answers, respond without tool calls.")
    return "\n".join(lines)
