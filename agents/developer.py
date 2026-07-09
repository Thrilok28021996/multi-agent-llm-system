import json
from openai import OpenAI
from pathlib import Path

# --- OpenAI Responses API tool schema (NOT input_schema) ---
tools = [
    {
        "type": "function",
        "name": "read_file",
        "description": "Read the full contents of a text file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative or absolute file path",
                }
            },
            "required": ["path"],
        },
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Write (overwrite) text content to a file at the given path. Creates parent dirs if needed.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
]


def read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return f"ERROR: {path} does not exist"
    return p.read_text(encoding="utf-8")


def write_file(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {path}"


TOOL_IMPL = {"read_file": read_file, "write_file": write_file}


def developer(tasks: str, max_turns: int = 10) -> str:
    client = OpenAI(base_url="http://localhost:1234/v1/", api_key="lmstudio")

    response = client.responses.create(
        model="aisha/qwen3.5-4b-nothink",
        instructions="You are a developer. You will do the tasks assigned by the orchestrator.",
        input=tasks,
        tools=tools,
        temperature=0.1,
    )
    print("response", response.output)

    for _ in range(max_turns):
        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]

        if not function_calls:
            return response.output_text

        tool_outputs = []
        for call in function_calls:
            fn = TOOL_IMPL[call.name]
            args = json.loads(call.arguments)
            try:
                result = fn(**args)
            except Exception as e:
                result = f"ERROR: {e}"
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": str(result),
                }
            )

        response = client.responses.create(
            model="aisha/qwen3.5-4b-nothink",
            previous_response_id=response.id,
            input=tool_outputs,
            tools=tools,
            temperature=0.1,
        )

    return "Max turns reached without final answer."
