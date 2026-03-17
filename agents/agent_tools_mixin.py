"""
Agent Tools Mixin - Provides all 13 Claude Code tools to agents

This mixin class provides convenient methods for agents to use all Claude Code tools:
1. Read - Read files with line numbers
2. Write - Write files safely
3. Edit - Surgical file edits
4. MultiEdit - Batch edits with rollback
5. Glob - Find files by pattern
6. Grep - Search file contents
7. Bash - Execute commands
8. WebFetch - Fetch URLs
9. WebSearch - Search the web
10. Todo - Manage tasks
11. Notebook - Edit Jupyter notebooks
12. Task - Background task execution
13. LSP - Code intelligence

Usage:
    class MyAgent(BaseAgent, AgentToolsMixin):
        def some_method(self):
            # Use tools directly
            content = self.read_file("path/to/file.py")
            self.write_file("output.py", content)
            files = self.glob_files("**/*.py")
            matches = self.grep_search("def main", path="src/")
"""

from typing import List, Dict, Any, Optional
import asyncio


class AgentToolsMixin:
    """
    Mixin providing all 13 Claude Code tools to agents.

    This class should be mixed with BaseAgent or any agent class
    that has self.tools (UnifiedTools instance).
    """

    # =========================================================================
    # FILE OPERATIONS
    # =========================================================================

    def read_file(
        self,
        file_path: str,
        offset: int = 0,
        limit: Optional[int] = None
    ) -> str:
        """
        Read a file with line numbers.

        Args:
            file_path: Path to the file
            offset: Line number to start from (0-indexed)
            limit: Maximum number of lines to read

        Returns:
            File content with line numbers

        Example:
            content = agent.read_file("app.py", offset=0, limit=100)
        """
        if not hasattr(self, 'tools'):
            raise AttributeError("Agent must have 'tools' attribute (UnifiedTools instance)")

        result = self.tools.read.read(file_path, offset, limit)
        if result.success:
            return result.data
        else:
            raise FileNotFoundError(f"Could not read {file_path}: {result.error}")

    def write_file(
        self,
        file_path: str,
        content: str,
        create_dirs: bool = True
    ) -> bool:
        """
        Write content to a file.

        Args:
            file_path: Path to the file
            content: Content to write
            create_dirs: Create parent directories if they don't exist

        Returns:
            True if successful

        Example:
            agent.write_file("output/result.py", code_content)
        """
        if not hasattr(self, 'tools'):
            raise AttributeError("Agent must have 'tools' attribute")

        result = self.tools.write.write(file_path, content)
        if not result.success:
            raise IOError(f"Could not write {file_path}: {result.error}")
        return True

    def edit_file(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False
    ) -> bool:
        """
        Make surgical edits to a file.

        Args:
            file_path: Path to the file
            old_string: String to replace (must be unique unless replace_all=True)
            new_string: Replacement string
            replace_all: Replace all occurrences

        Returns:
            True if successful

        Example:
            agent.edit_file("app.py", "def old()", "def new()")
        """
        if not hasattr(self, 'tools'):
            raise AttributeError("Agent must have 'tools' attribute")

        result = self.tools.edit.edit(file_path, old_string, new_string, replace_all)
        if not result.success:
            raise ValueError(f"Could not edit {file_path}: {result.error}")
        return True

    def multi_edit_file(
        self,
        file_path: str,
        edits: List[Dict[str, str]]
    ) -> bool:
        """
        Make multiple edits to a file atomically.

        Args:
            file_path: Path to the file
            edits: List of edit dicts with 'old_string' and 'new_string'

        Returns:
            True if successful (all edits applied or all rolled back)

        Example:
            agent.multi_edit_file("app.py", [
                {"old_string": "v1", "new_string": "v2"},
                {"old_string": "old", "new_string": "new"}
            ])
        """
        if not hasattr(self, 'tools'):
            raise AttributeError("Agent must have 'tools' attribute")

        result = self.tools.multi_edit.multi_edit(file_path, edits)
        if not result.success:
            raise ValueError(f"Could not multi-edit {file_path}: {result.error}")
        return True

    # =========================================================================
    # FILE SEARCH
    # =========================================================================

    def glob_files(
        self,
        pattern: str,
        path: Optional[str] = None
    ) -> List[str]:
        """
        Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "**/*.py", "src/**/*.ts")
            path: Directory to search in (default: workspace root)

        Returns:
            List of matching file paths (sorted by modification time)

        Example:
            py_files = agent.glob_files("**/*.py")
            test_files = agent.glob_files("tests/**/test_*.py")
        """
        if not hasattr(self, 'tools'):
            raise AttributeError("Agent must have 'tools' attribute")

        result = self.tools.glob.glob(pattern, path)
        if result.success:
            return result.data
        else:
            return []

    def grep_search(
        self,
        pattern: str,
        path: Optional[str] = None,
        file_type: Optional[str] = None,
        glob_pattern: Optional[str] = None,
        case_insensitive: bool = False,
        context_lines: int = 0
    ) -> Dict[str, Any]:
        """
        Search for pattern in files using regex.

        Args:
            pattern: Regex pattern to search for
            path: Directory to search in
            file_type: File type filter (e.g., "py", "js", "ts")
            glob_pattern: Glob pattern for files (e.g., "*.py")
            case_insensitive: Case-insensitive search
            context_lines: Number of context lines to show

        Returns:
            Dict with matches (format depends on output_mode)

        Example:
            # Find function definitions
            matches = agent.grep_search(r"def \\w+", file_type="py")

            # Find TODO comments with context
            todos = agent.grep_search("TODO", context_lines=2)
        """
        if not hasattr(self, 'tools'):
            raise AttributeError("Agent must have 'tools' attribute")

        result = self.tools.grep.grep(
            pattern=pattern,
            path=path,
            type=file_type,
            glob=glob_pattern,
            case_insensitive=case_insensitive,
            context_before=context_lines,
            context_after=context_lines
        )

        if result.success:
            return result.data
        else:
            return {"matches": [], "error": result.error}

    # =========================================================================
    # COMMAND EXECUTION
    # =========================================================================

    def bash_execute(
        self,
        command: str,
        timeout: int = 120,
        description: Optional[str] = None
    ) -> str:
        """
        Execute a bash command.

        Args:
            command: Command to execute
            timeout: Timeout in seconds
            description: Human-readable description

        Returns:
            Command output

        Example:
            output = agent.bash_execute("python --version")
            output = agent.bash_execute("pytest tests/", timeout=300)
        """
        if not hasattr(self, 'tools'):
            raise AttributeError("Agent must have 'tools' attribute")

        result = self.tools.bash.execute(command, timeout)
        if result.success:
            return result.data
        else:
            raise RuntimeError(f"Command failed: {result.error}")

    # =========================================================================
    # WEB OPERATIONS
    # =========================================================================

    async def web_fetch_async(
        self,
        url: str,
        prompt: Optional[str] = None
    ) -> str:
        """
        Fetch content from a URL (async).

        Args:
            url: URL to fetch
            prompt: Optional prompt for content extraction

        Returns:
            Fetched content

        Example:
            content = await agent.web_fetch_async("https://example.com")
            data = await agent.web_fetch_async("https://api.github.com/users/octocat", "Extract user info")
        """
        if not hasattr(self, 'tools'):
            raise AttributeError("Agent must have 'tools' attribute")

        result = await self.tools.web_fetch.fetch(url, prompt)
        if result.success:
            return result.data
        else:
            raise ConnectionError(f"Could not fetch {url}: {result.error}")

    def web_fetch(
        self,
        url: str,
        prompt: Optional[str] = None
    ) -> str:
        """
        Fetch content from a URL (sync wrapper).

        Args:
            url: URL to fetch
            prompt: Optional prompt for content extraction

        Returns:
            Fetched content

        Example:
            content = agent.web_fetch("https://example.com")
        """
        return asyncio.run(self.web_fetch_async(url, prompt))

    async def web_search_async(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, str]]:
        """
        Search the web using DuckDuckGo (async).

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of search results with title, url, snippet

        Example:
            results = await agent.web_search_async("Python async programming")
        """
        if not hasattr(self, 'tools'):
            raise AttributeError("Agent must have 'tools' attribute")

        result = await self.tools.web_search.search(query, max_results)
        if result.success:
            return result.data
        else:
            return []

    def web_search(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, str]]:
        """
        Search the web using DuckDuckGo (sync wrapper).

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of search results

        Example:
            results = agent.web_search("FastAPI tutorial")
        """
        return asyncio.run(self.web_search_async(query, max_results))

    # =========================================================================
    # TASK MANAGEMENT
    # =========================================================================

    def todo_create(
        self,
        task: str,
        priority: str = "medium"
    ) -> int:
        """
        Create a TODO task.

        Args:
            task: Task description
            priority: Priority level (low, medium, high)

        Returns:
            Task ID

        Example:
            task_id = agent.todo_create("Implement authentication", priority="high")
        """
        if not hasattr(self, 'tools'):
            raise AttributeError("Agent must have 'tools' attribute")

        result = self.tools.todo.create(task, priority)
        if result.success:
            return result.data['id']
        else:
            raise ValueError(f"Could not create task: {result.error}")

    def todo_list(self) -> List[Dict[str, Any]]:
        """
        List all TODO tasks.

        Returns:
            List of tasks with id, description, status, priority

        Example:
            tasks = agent.todo_list()
            pending_tasks = [t for t in tasks if t['status'] == 'pending']
        """
        if not hasattr(self, 'tools'):
            raise AttributeError("Agent must have 'tools' attribute")

        result = self.tools.todo.list()
        if result.success:
            return result.data
        else:
            return []

    def todo_complete(self, task_id: int) -> bool:
        """
        Mark a TODO task as complete.

        Args:
            task_id: Task ID to complete

        Returns:
            True if successful

        Example:
            agent.todo_complete(1)
        """
        if not hasattr(self, 'tools'):
            raise AttributeError("Agent must have 'tools' attribute")

        result = self.tools.todo.complete(task_id)
        return result.success

    # =========================================================================
    # NOTEBOOK OPERATIONS
    # =========================================================================

    def notebook_edit(
        self,
        notebook_path: str,
        cell_index: int,
        new_source: str,
        cell_type: str = "code"
    ) -> bool:
        """
        Edit a Jupyter notebook cell.

        Args:
            notebook_path: Path to .ipynb file
            cell_index: Cell index (0-based)
            new_source: New cell content
            cell_type: Cell type ("code" or "markdown")

        Returns:
            True if successful

        Example:
            agent.notebook_edit("analysis.ipynb", 0, "import pandas as pd")
        """
        if not hasattr(self, 'tools'):
            raise AttributeError("Agent must have 'tools' attribute")

        result = self.tools.notebook.edit(notebook_path, cell_index, new_source, cell_type)
        if not result.success:
            raise ValueError(f"Could not edit notebook: {result.error}")
        return True

    # =========================================================================
    # CODE INTELLIGENCE (LSP)
    # =========================================================================

    def lsp_definition(
        self,
        file_path: str,
        line: int,
        character: int
    ) -> Dict[str, Any]:
        """
        Get definition location for a symbol.

        Args:
            file_path: Source file
            line: Line number (0-based)
            character: Character position (0-based)

        Returns:
            Definition location(s)

        Example:
            definition = agent.lsp_definition("app.py", 10, 5)
        """
        if not hasattr(self, 'tools'):
            raise AttributeError("Agent must have 'tools' attribute")

        result = self.tools.lsp.definition(file_path, line, character)
        if result.success:
            return result.data
        else:
            return {}

    def lsp_references(
        self,
        file_path: str,
        line: int,
        character: int
    ) -> List[Dict[str, Any]]:
        """
        Find all references to a symbol.

        Args:
            file_path: Source file
            line: Line number (0-based)
            character: Character position (0-based)

        Returns:
            List of reference locations

        Example:
            refs = agent.lsp_references("app.py", 10, 5)
        """
        if not hasattr(self, 'tools'):
            raise AttributeError("Agent must have 'tools' attribute")

        result = self.tools.lsp.references(file_path, line, character)
        if result.success:
            return result.data
        else:
            return []

    def lsp_hover(
        self,
        file_path: str,
        line: int,
        character: int
    ) -> str:
        """
        Get hover information (docs, type info) for a symbol.

        Args:
            file_path: Source file
            line: Line number (0-based)
            character: Character position (0-based)

        Returns:
            Hover information text

        Example:
            info = agent.lsp_hover("app.py", 10, 5)
        """
        if not hasattr(self, 'tools'):
            raise AttributeError("Agent must have 'tools' attribute")

        result = self.tools.lsp.hover(file_path, line, character)
        if result.success:
            return result.data.get('contents', '')
        else:
            return ""

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def file_exists(self, file_path: str) -> bool:
        """Check if a file exists."""
        try:
            self.read_file(file_path, limit=1)
            return True
        except FileNotFoundError:
            return False

    def read_multiple_files(self, file_paths: List[str]) -> Dict[str, str]:
        """Read multiple files at once."""
        results = {}
        for path in file_paths:
            try:
                results[path] = self.read_file(path)
            except FileNotFoundError:
                results[path] = None
        return results

    def write_multiple_files(self, files: Dict[str, str]) -> Dict[str, bool]:
        """Write multiple files at once."""
        results = {}
        for path, content in files.items():
            try:
                results[path] = self.write_file(path, content)
            except IOError:
                results[path] = False
        return results

    def find_and_replace(
        self,
        pattern: str,
        replacement: str,
        file_pattern: str = "**/*.py"
    ) -> List[str]:
        """
        Find pattern in files and replace it.

        Args:
            pattern: Regex pattern to find
            replacement: Replacement string
            file_pattern: Glob pattern for files

        Returns:
            List of modified files

        Example:
            modified = agent.find_and_replace(
                r"old_function",
                "new_function",
                "**/*.py"
            )
        """
        # Find files matching pattern
        files = self.glob_files(file_pattern)

        # Find matches in those files
        matches = self.grep_search(pattern, glob_pattern=file_pattern)

        modified_files = []
        for file_path in files:
            try:
                content = self.read_file(file_path)
                if pattern in content:
                    new_content = content.replace(pattern, replacement)
                    self.write_file(file_path, new_content)
                    modified_files.append(file_path)
            except (FileNotFoundError, IOError):
                continue

        return modified_files


# Convenience function to check if an object has all tools
def has_all_tools(agent) -> bool:
    """Check if an agent has all 13 Claude Code tools available."""
    required_attrs = [
        'read_file', 'write_file', 'edit_file', 'multi_edit_file',
        'glob_files', 'grep_search', 'bash_execute',
        'web_fetch', 'web_search',
        'todo_create', 'todo_list', 'todo_complete',
        'notebook_edit',
        'lsp_definition', 'lsp_references', 'lsp_hover'
    ]
    return all(hasattr(agent, attr) for attr in required_attrs)
