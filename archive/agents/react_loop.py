"""
ReAct (Reason + Act) tool use loop.
Enables LLM to iteratively call tools and reason about their results.

Loop:
1. LLM generates response (may contain tool calls)
2. Parser extracts tool calls
3. Tools execute, results collected
4. Results injected into message history
5. LLM generates next response
6. Repeat until no tool calls or max_iterations reached
"""
import json
import re
import asyncio
from typing import Any, Dict, List, Tuple


class ToolCallParser:
    """Parses <tool_use>...</tool_use> blocks from LLM responses."""

    TOOL_USE_PATTERN = re.compile(
        r'<tool_use>\s*([\s\S]*?)\s*</tool_use>',
        re.IGNORECASE
    )

    def extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Extract all tool calls from LLM response text."""
        calls = []
        for match in self.TOOL_USE_PATTERN.finditer(text):
            raw = match.group(1).strip()
            try:
                call = json.loads(raw)
                if "tool" in call:
                    calls.append(call)
            except json.JSONDecodeError:
                # Try to salvage partial JSON
                tool_match = re.search(r'"tool"\s*:\s*"([^"]+)"', raw)
                if tool_match:
                    calls.append({"tool": tool_match.group(1), "_raw": raw})
        return calls

    def strip_tool_calls(self, text: str) -> str:
        """Remove tool call blocks from text, keeping surrounding content."""
        return self.TOOL_USE_PATTERN.sub("", text).strip()

    def has_tool_calls(self, text: str) -> bool:
        return bool(self.TOOL_USE_PATTERN.search(text))


class ToolExecutor:
    """Executes tool calls using the agent's existing tool infrastructure."""

    def __init__(self, agent):
        self.agent = agent

    async def execute(self, tool_call: Dict[str, Any]) -> str:
        """Execute a single tool call and return result as string."""
        tool_name = tool_call.get("tool", "")
        params = {k: v for k, v in tool_call.items() if k not in ("tool", "_raw")}

        try:
            result = await self._dispatch(tool_name, params)
            return f"Tool '{tool_name}' succeeded:\n{result}"
        except Exception as e:
            return f"Tool '{tool_name}' failed: {str(e)}"

    async def _dispatch(self, tool_name: str, params: Dict) -> str:
        """Dispatch to the right tool method on the agent."""
        agent = self.agent

        if tool_name == "bash":
            command = params.get("command", "")
            if not command:
                return "Error: 'command' parameter required"
            if hasattr(agent, 'bash_execute'):
                result = agent.bash_execute(command, timeout=30)
                return str(result)
            # Fallback to use_tool
            result = agent.use_tool("execute_command", command=command)
            return result.get("output", result.get("error", str(result)))

        elif tool_name == "read_file":
            path = params.get("path", "")
            offset = params.get("offset", 0)
            limit = params.get("limit", 200)
            if hasattr(agent, 'read_file'):
                return agent.read_file(path, offset=offset, limit=limit)
            result = agent.use_tool("read_file", path=path)
            return result.get("content", str(result))

        elif tool_name == "write_file":
            path = params.get("path", "")
            content = params.get("content", "")
            if hasattr(agent, 'write_file'):
                agent.write_file(path, content)
                return f"Written {len(content)} chars to {path}"
            result = agent.use_tool("write_file", path=path, content=content)
            return "Written successfully" if result.get("success") else str(result)

        elif tool_name == "edit_file":
            path = params.get("path", "")
            old_text = params.get("old_text", "")
            new_text = params.get("new_text", "")
            if hasattr(agent, 'edit_file'):
                agent.edit_file(path, old_text, new_text)
                return f"Edited {path}"
            result = agent.use_tool("edit_file", path=path, old_content=old_text, new_content=new_text)
            return "Edited successfully" if result.get("success") else str(result)

        elif tool_name == "search_files":
            pattern = params.get("pattern", "")
            path = params.get("path", ".")
            file_pattern = params.get("file_pattern", "")
            if hasattr(agent, 'grep_search'):
                result = agent.grep_search(pattern, path=path, glob_pattern=file_pattern or None)
                return str(result)[:2000]
            result = agent.use_tool("search_files", pattern=pattern, path=path)
            return str(result.get("results", result))[:2000]

        elif tool_name == "list_files":
            pattern = params.get("pattern", "**/*")
            if hasattr(agent, 'glob_files'):
                result = agent.glob_files(pattern)
                return "\n".join(str(r) for r in result[:50])
            result = agent.use_tool("list_directory", path=".")
            return str(result)[:2000]

        elif tool_name == "web_search":
            query = params.get("query", "")
            max_results = params.get("max_results", 5)
            if hasattr(agent, 'web_search_async'):
                results = await agent.web_search_async(query, max_results)
                formatted = []
                for r in (results or [])[:max_results]:
                    if isinstance(r, dict):
                        formatted.append(
                            f"- {r.get('title', '')}: {r.get('snippet', '')}\n  URL: {r.get('url', '')}"
                        )
                return "\n".join(formatted) if formatted else "No results found"
            elif hasattr(agent, 'web_search'):
                try:
                    results = await asyncio.to_thread(agent.web_search, query, max_results)
                    formatted = []
                    for r in (results or [])[:max_results]:
                        if isinstance(r, dict):
                            formatted.append(
                                f"- {r.get('title', '')}: {r.get('snippet', '')}\n  URL: {r.get('url', '')}"
                            )
                    return "\n".join(formatted) if formatted else "No results found"
                except Exception as e:
                    return f"Web search error: {e}"
            return "Web search not available"

        elif tool_name == "web_fetch":
            url = params.get("url", "")
            mode = params.get("extract_mode", "text")
            if hasattr(agent, 'web_fetch_async'):
                content = await agent.web_fetch_async(url)
                return str(content)[:3000]
            elif hasattr(agent, 'web_fetch'):
                try:
                    content = await asyncio.to_thread(agent.web_fetch, url)
                    return str(content)[:3000]
                except Exception as e:
                    return f"Web fetch error: {e}"
            return "Web fetch not available"

        else:
            return (
                f"Unknown tool: '{tool_name}'. "
                "Available: bash, read_file, write_file, edit_file, search_files, list_files, web_search, web_fetch"
            )


class ReActLoop:
    """
    ReAct reasoning + action loop.
    Enables LLM to call tools iteratively and reason about results.
    """

    def __init__(self, max_iterations: int = 8, max_tool_calls_per_iter: int = 3):
        self.max_iterations = max_iterations
        self.max_tool_calls_per_iter = max_tool_calls_per_iter
        self.parser = ToolCallParser()

    async def run(
        self,
        agent,
        initial_messages: List[Dict],
        tools_enabled: bool = True,
    ) -> Tuple[str, List[Dict]]:
        """
        Run the ReAct loop.
        Returns (final_response, full_message_history).
        """
        if not agent.model_spec:
            return f"ReActLoop error: agent '{getattr(agent, 'name', agent)}' has no model_spec configured.", initial_messages

        from config.llm_client import get_llm_client

        messages = list(initial_messages)
        final_response = ""
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            # Call LLM
            try:
                # Enforce token budget before LLM call
                trimmed = agent._trim_messages_to_budget(messages)
                response_text, input_tok, output_tok = get_llm_client().chat(
                    agent.model_spec,
                    trimmed,
                    temperature=agent.config.temperature,
                    max_tokens=agent.config.max_tokens,
                )
            except Exception as e:
                return f"LLM error: {e}", messages

            # Check for tool calls
            if not tools_enabled or not self.parser.has_tool_calls(response_text):
                # No tool calls — this is the final response
                final_response = response_text
                messages.append({"role": "assistant", "content": response_text})
                break

            # Extract tool calls
            tool_calls = self.parser.extract_tool_calls(response_text)

            # Add assistant message (full response including tool call blocks)
            messages.append({
                "role": "assistant",
                "content": response_text
            })

            # Execute tools and collect results
            executor = ToolExecutor(agent)
            tool_results = []

            for call in tool_calls[:self.max_tool_calls_per_iter]:
                tool_name = call.get("tool", "unknown")
                result = await executor.execute(call)
                tool_results.append(f"[Tool: {tool_name}]\n{result}")

            # Feed results back to LLM
            results_text = "\n\n".join(tool_results)
            messages.append({
                "role": "user",
                "content": (
                    f"<tool_results>\n{results_text}\n</tool_results>\n\n"
                    "Continue based on these results. If you have everything you need, "
                    "provide your final answer without any tool calls."
                )
            })

            # If approaching iteration limit, prompt for final answer
            if iteration >= self.max_iterations - 1:
                messages.append({
                    "role": "user",
                    "content": "Please provide your final answer now without using any more tools."
                })

        return final_response, messages
