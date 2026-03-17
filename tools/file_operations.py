"""File operations tool for creating, reading, updating files and directories."""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FileChange:
    """Represents a change made to a file."""

    path: str
    operation: str  # create, update, delete, rename
    timestamp: datetime
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    description: str = ""


class FileOperations:
    """
    File system operations tool for agents.
    Provides capabilities similar to Claude Code for file manipulation.
    """

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = Path(workspace_root).resolve()
        self.change_history: List[FileChange] = []

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to workspace root."""
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace_root / p
        return p.resolve()

    def _validate_path(self, path: Path) -> bool:
        """
        Ensure path is within workspace (security check).

        This method protects against:
        - Path traversal attacks (../../etc/passwd)
        - Symlink attacks (symlinks pointing outside workspace)

        Args:
            path: The path to validate

        Returns:
            True if path is safe, False otherwise
        """
        try:
            # Resolve the path to handle any .. or symlinks
            resolved = path.resolve()

            # Use os.path.realpath to follow ALL symlinks
            # This is important because Path.resolve() may not follow all symlinks
            # on some systems
            real_path = os.path.realpath(str(resolved))
            real_workspace = os.path.realpath(str(self.workspace_root))

            # Check if the real path starts with the workspace root
            if not real_path.startswith(real_workspace + os.sep) and real_path != real_workspace:
                return False

            # Additional check: make sure no path component is a symlink
            # pointing outside the workspace
            current = resolved
            while current != self.workspace_root and current != current.parent:
                if current.is_symlink():
                    link_target = os.path.realpath(str(current))
                    if not link_target.startswith(real_workspace + os.sep):
                        return False
                current = current.parent

            return True
        except (ValueError, OSError):
            return False

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a filename to prevent path traversal.

        Args:
            filename: The filename to sanitize

        Returns:
            Sanitized filename
        """
        # Remove any path separators
        filename = filename.replace('/', '_').replace('\\', '_')

        # Remove any null bytes
        filename = filename.replace('\x00', '')

        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')

        # Limit length
        if len(filename) > 255:
            filename = filename[:255]

        return filename

    def _log_change(self, change: FileChange) -> None:
        """Log a file change for history/undo."""
        self.change_history.append(change)

    # ============================================================
    #  READ OPERATIONS
    # ============================================================

    def read_file(self, path: str) -> Dict[str, Any]:
        """
        Read contents of a file.

        Returns:
            Dict with 'success', 'content', and 'error' keys
        """
        file_path = self._resolve_path(path)

        if not self._validate_path(file_path):
            return {"success": False, "content": None, "error": "Path outside workspace"}

        if not file_path.exists():
            return {"success": False, "content": None, "error": f"File not found: {path}"}

        if not file_path.is_file():
            return {"success": False, "content": None, "error": f"Not a file: {path}"}

        try:
            content = file_path.read_text(encoding="utf-8")
            return {
                "success": True,
                "content": content,
                "path": str(file_path),
                "size": len(content),
                "lines": content.count("\n") + 1
            }
        except Exception as e:
            return {"success": False, "content": None, "error": str(e)}

    def list_directory(self, path: str = ".", recursive: bool = False) -> Dict[str, Any]:
        """
        List contents of a directory.

        Args:
            path: Directory path
            recursive: If True, list all nested contents
        """
        dir_path = self._resolve_path(path)

        if not self._validate_path(dir_path):
            return {"success": False, "items": [], "error": "Path outside workspace"}

        if not dir_path.exists():
            return {"success": False, "items": [], "error": f"Directory not found: {path}"}

        if not dir_path.is_dir():
            return {"success": False, "items": [], "error": f"Not a directory: {path}"}

        try:
            items = []
            if recursive:
                for item in dir_path.rglob("*"):
                    rel_path = item.relative_to(dir_path)
                    items.append({
                        "name": str(rel_path),
                        "type": "dir" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else 0
                    })
            else:
                for item in dir_path.iterdir():
                    items.append({
                        "name": item.name,
                        "type": "dir" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else 0
                    })

            return {"success": True, "items": items, "path": str(dir_path)}
        except Exception as e:
            return {"success": False, "items": [], "error": str(e)}

    def search_files(self, pattern: str, path: str = ".") -> Dict[str, Any]:
        """
        Search for files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "*.py", "**/*.js")
            path: Starting directory
        """
        dir_path = self._resolve_path(path)

        if not self._validate_path(dir_path):
            return {"success": False, "matches": [], "error": "Path outside workspace"}

        try:
            matches = list(dir_path.glob(pattern))
            return {
                "success": True,
                "matches": [str(m.relative_to(self.workspace_root)) for m in matches],
                "count": len(matches)
            }
        except Exception as e:
            return {"success": False, "matches": [], "error": str(e)}

    def grep_content(self, pattern: str, path: str = ".", file_pattern: str = "*") -> Dict[str, Any]:
        """
        Search for text pattern in files.

        Args:
            pattern: Text to search for
            path: Directory to search in
            file_pattern: File glob pattern to limit search
        """
        dir_path = self._resolve_path(path)
        results = []

        try:
            for file_path in dir_path.rglob(file_pattern):
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        lines = content.split("\n")
                        for i, line in enumerate(lines, 1):
                            if pattern.lower() in line.lower():
                                results.append({
                                    "file": str(file_path.relative_to(self.workspace_root)),
                                    "line_number": i,
                                    "line": line.strip()
                                })
                    except (UnicodeDecodeError, PermissionError):
                        continue

            return {"success": True, "results": results, "count": len(results)}
        except Exception as e:
            return {"success": False, "results": [], "error": str(e)}

    # ============================================================
    #  WRITE OPERATIONS
    # ============================================================

    def create_file(self, path: str, content: str = "", description: str = "") -> Dict[str, Any]:
        """
        Create a new file with content.

        Args:
            path: File path to create
            content: Initial content
            description: Description of why this file was created
        """
        file_path = self._resolve_path(path)

        if not self._validate_path(file_path):
            return {"success": False, "error": "Path outside workspace"}

        if file_path.exists():
            return {"success": False, "error": f"File already exists: {path}"}

        try:
            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

            self._log_change(FileChange(
                path=str(file_path),
                operation="create",
                timestamp=datetime.now(),
                new_content=content,
                description=description
            ))

            return {
                "success": True,
                "path": str(file_path),
                "size": len(content),
                "message": f"Created file: {path}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def update_file(self, path: str, content: str, description: str = "") -> Dict[str, Any]:
        """
        Update/overwrite a file's content.

        Args:
            path: File path to update
            content: New content
            description: Description of the change
        """
        file_path = self._resolve_path(path)

        if not self._validate_path(file_path):
            return {"success": False, "error": "Path outside workspace"}

        if not file_path.exists():
            return {"success": False, "error": f"File not found: {path}"}

        try:
            old_content = file_path.read_text(encoding="utf-8")
            file_path.write_text(content, encoding="utf-8")

            self._log_change(FileChange(
                path=str(file_path),
                operation="update",
                timestamp=datetime.now(),
                old_content=old_content,
                new_content=content,
                description=description
            ))

            return {
                "success": True,
                "path": str(file_path),
                "old_size": len(old_content),
                "new_size": len(content),
                "message": f"Updated file: {path}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Edit a file by replacing specific text (like Claude Code's Edit tool).

        Args:
            path: File path
            old_text: Text to find and replace
            new_text: Replacement text
            description: Description of the edit
        """
        file_path = self._resolve_path(path)

        if not self._validate_path(file_path):
            return {"success": False, "error": "Path outside workspace"}

        if not file_path.exists():
            return {"success": False, "error": f"File not found: {path}"}

        try:
            old_content = file_path.read_text(encoding="utf-8")

            if old_text not in old_content:
                return {"success": False, "error": "Text to replace not found in file"}

            # Count occurrences
            occurrences = old_content.count(old_text)
            if occurrences > 1:
                return {
                    "success": False,
                    "error": f"Multiple occurrences ({occurrences}) found. Please provide more context."
                }

            new_content = old_content.replace(old_text, new_text)
            file_path.write_text(new_content, encoding="utf-8")

            self._log_change(FileChange(
                path=str(file_path),
                operation="edit",
                timestamp=datetime.now(),
                old_content=old_content,
                new_content=new_content,
                description=description
            ))

            return {
                "success": True,
                "path": str(file_path),
                "message": f"Edited file: {path}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def append_to_file(self, path: str, content: str, description: str = "") -> Dict[str, Any]:
        """Append content to an existing file."""
        file_path = self._resolve_path(path)

        if not self._validate_path(file_path):
            return {"success": False, "error": "Path outside workspace"}

        try:
            old_content = ""
            if file_path.exists():
                old_content = file_path.read_text(encoding="utf-8")

            new_content = old_content + content
            file_path.write_text(new_content, encoding="utf-8")

            self._log_change(FileChange(
                path=str(file_path),
                operation="append",
                timestamp=datetime.now(),
                old_content=old_content,
                new_content=new_content,
                description=description
            ))

            return {"success": True, "path": str(file_path), "message": f"Appended to: {path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ============================================================
    #  DIRECTORY OPERATIONS
    # ============================================================

    def create_directory(self, path: str, description: str = "") -> Dict[str, Any]:
        """Create a new directory (and parent directories if needed)."""
        dir_path = self._resolve_path(path)

        if not self._validate_path(dir_path):
            return {"success": False, "error": "Path outside workspace"}

        try:
            dir_path.mkdir(parents=True, exist_ok=True)

            self._log_change(FileChange(
                path=str(dir_path),
                operation="create_dir",
                timestamp=datetime.now(),
                description=description
            ))

            return {"success": True, "path": str(dir_path), "message": f"Created directory: {path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete(self, path: str, description: str = "") -> Dict[str, Any]:
        """Delete a file or directory."""
        target_path = self._resolve_path(path)

        if not self._validate_path(target_path):
            return {"success": False, "error": "Path outside workspace"}

        if not target_path.exists():
            return {"success": False, "error": f"Path not found: {path}"}

        try:
            old_content = None
            if target_path.is_file():
                old_content = target_path.read_text(encoding="utf-8")
                target_path.unlink()
            else:
                shutil.rmtree(target_path)

            self._log_change(FileChange(
                path=str(target_path),
                operation="delete",
                timestamp=datetime.now(),
                old_content=old_content,
                description=description
            ))

            return {"success": True, "path": str(target_path), "message": f"Deleted: {path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def rename(self, old_path: str, new_path: str, description: str = "") -> Dict[str, Any]:
        """Rename/move a file or directory."""
        old = self._resolve_path(old_path)
        new = self._resolve_path(new_path)

        if not self._validate_path(old) or not self._validate_path(new):
            return {"success": False, "error": "Path outside workspace"}

        if not old.exists():
            return {"success": False, "error": f"Source not found: {old_path}"}

        if new.exists():
            return {"success": False, "error": f"Destination exists: {new_path}"}

        try:
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(new))

            self._log_change(FileChange(
                path=str(old),
                operation="rename",
                timestamp=datetime.now(),
                description=f"Renamed to {new_path}. {description}"
            ))

            return {
                "success": True,
                "old_path": str(old),
                "new_path": str(new),
                "message": f"Renamed: {old_path} -> {new_path}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ============================================================
    #  UTILITY
    # ============================================================

    def get_file_info(self, path: str) -> Dict[str, Any]:
        """Get detailed information about a file or directory."""
        target_path = self._resolve_path(path)

        if not target_path.exists():
            return {"success": False, "error": f"Path not found: {path}"}

        try:
            stat = target_path.stat()
            info = {
                "success": True,
                "path": str(target_path),
                "name": target_path.name,
                "type": "dir" if target_path.is_dir() else "file",
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            }

            if target_path.is_file():
                info["extension"] = target_path.suffix
                info["lines"] = target_path.read_text(encoding="utf-8").count("\n") + 1

            return info
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_change_history(self) -> List[Dict[str, Any]]:
        """Get history of all file changes."""
        return [
            {
                "path": c.path,
                "operation": c.operation,
                "timestamp": c.timestamp.isoformat(),
                "description": c.description
            }
            for c in self.change_history
        ]

    def export_workspace_structure(self) -> str:
        """Export the workspace structure as a tree string."""
        def build_tree(path: Path, prefix: str = "") -> str:
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
            tree = ""
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                tree += f"{prefix}{connector}{item.name}\n"
                if item.is_dir():
                    extension = "    " if is_last else "│   "
                    tree += build_tree(item, prefix + extension)
            return tree

        return f"{self.workspace_root.name}/\n" + build_tree(self.workspace_root)

    # ============================================================
    #  ALIASES
    # ============================================================

    def write_file(self, path: str, content: str, description: str = "") -> Dict[str, Any]:
        """Write to a file (create or update). Alias for create/update operations."""
        file_path = self._resolve_path(path)
        if file_path.exists():
            return self.update_file(path, content, description)
        else:
            return self.create_file(path, content, description)

    def search_in_files(self, pattern: str, path: str = ".", file_pattern: str = "*") -> Dict[str, Any]:
        """Search for pattern in files. Alias for grep_content."""
        return self.grep_content(pattern, path, file_pattern)
