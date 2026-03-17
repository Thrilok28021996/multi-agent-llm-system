"""
Enhanced LSP Tool for Company AGI.

Provides Claude Code-style LSP operations with:
- Real LSP server support when available
- Graceful regex fallback
- All standard LSP operations
- Unified result format
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .lsp_server import (
    LSPManager,
    LSPClient,
    LSPLocation,
    LSPSymbol,
    LSPCallHierarchyItem,
    get_lsp_manager,
)


@dataclass
class LSPResult:
    """Unified result from LSP operations."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    source: str = "regex"  # "lsp" or "regex"
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedLSPTool:
    """
    Enhanced LSP tool with real server support and regex fallback.

    Operations:
    - goToDefinition: Find where a symbol is defined
    - findReferences: Find all references to a symbol
    - hover: Get hover information (docs, type info)
    - documentSymbol: Get all symbols in a document
    - workspaceSymbol: Search for symbols across workspace
    - goToImplementation: Find implementations of interface/abstract
    - prepareCallHierarchy: Get call hierarchy item at position
    - incomingCalls: Find all callers of a function
    - outgoingCalls: Find all functions called by a function
    """

    # Regex patterns for fallback
    PATTERNS = {
        "python": {
            "function": r"^\s*(?:async\s+)?def\s+(\w+)",
            "class": r"^\s*class\s+(\w+)",
            "variable": r"^(\w+)\s*(?::\s*\w+)?\s*=",
            "import": r"^(?:from\s+\S+\s+)?import\s+(.+)",
        },
        "javascript": {
            "function": r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(?.*?\)?\s*=>)",
            "class": r"class\s+(\w+)",
            "variable": r"(?:const|let|var)\s+(\w+)\s*=",
            "import": r"import\s+(?:\{[^}]+\}|\w+)",
        },
        "typescript": {
            "function": r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*(?::\s*\w+)?\s*=\s*(?:async\s*)?\(?.*?\)?\s*=>)",
            "class": r"class\s+(\w+)",
            "interface": r"interface\s+(\w+)",
            "type": r"type\s+(\w+)\s*=",
            "variable": r"(?:const|let|var)\s+(\w+)\s*(?::\s*\w+)?\s*=",
        },
        "go": {
            "function": r"func\s+(?:\([^)]+\)\s+)?(\w+)",
            "struct": r"type\s+(\w+)\s+struct",
            "interface": r"type\s+(\w+)\s+interface",
            "variable": r"(?:var|const)\s+(\w+)",
        },
        "rust": {
            "function": r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)",
            "struct": r"(?:pub\s+)?struct\s+(\w+)",
            "enum": r"(?:pub\s+)?enum\s+(\w+)",
            "trait": r"(?:pub\s+)?trait\s+(\w+)",
            "impl": r"impl(?:<[^>]+>)?\s+(\w+)",
        },
    }

    EXTENSION_TO_LANGUAGE = {
        ".py": "python",
        ".pyi": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
    }

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = Path(workspace_root).resolve()
        self._lsp_manager: Optional[LSPManager] = None

    async def _get_lsp_manager(self) -> LSPManager:
        """Get or create LSP manager."""
        if self._lsp_manager is None:
            self._lsp_manager = get_lsp_manager(str(self.workspace_root))
        return self._lsp_manager

    async def _get_client(self, file_path: str) -> Optional[LSPClient]:
        """Get LSP client for a file."""
        manager = await self._get_lsp_manager()
        return await manager.get_client_for_file(file_path)

    def _get_language(self, file_path: str) -> str:
        """Get language from file extension."""
        ext = Path(file_path).suffix.lower()
        return self.EXTENSION_TO_LANGUAGE.get(ext, "python")

    def _resolve_path(self, file_path: str) -> Path:
        """Resolve file path."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        return path.resolve()

    # ============================================================
    #  Main Operations
    # ============================================================

    async def go_to_definition(
        self,
        file_path: str,
        line: int,
        character: int
    ) -> LSPResult:
        """Find where a symbol is defined."""
        path = self._resolve_path(file_path)
        if not path.exists():
            return LSPResult(success=False, error=f"File not found: {file_path}")

        # Try real LSP first
        client = await self._get_client(str(path))
        if client and client._initialized:
            try:
                locations = await client.go_to_definition(str(path), line, character)
                if locations:
                    return LSPResult(
                        success=True,
                        data=[self._location_to_dict(loc) for loc in locations],
                        source="lsp",
                    )
            except Exception:
                pass

        # Fallback to regex
        return await self._regex_find_definition(path, line, character)

    async def find_references(
        self,
        file_path: str,
        line: int,
        character: int,
        include_declaration: bool = True
    ) -> LSPResult:
        """Find all references to a symbol."""
        path = self._resolve_path(file_path)
        if not path.exists():
            return LSPResult(success=False, error=f"File not found: {file_path}")

        # Try real LSP first
        client = await self._get_client(str(path))
        if client and client._initialized:
            try:
                locations = await client.find_references(
                    str(path), line, character, include_declaration
                )
                if locations:
                    return LSPResult(
                        success=True,
                        data=[self._location_to_dict(loc) for loc in locations],
                        source="lsp",
                    )
            except Exception:
                pass

        # Fallback to regex
        return await self._regex_find_references(path, line, character)

    async def hover(
        self,
        file_path: str,
        line: int,
        character: int
    ) -> LSPResult:
        """Get hover information (documentation, type info)."""
        path = self._resolve_path(file_path)
        if not path.exists():
            return LSPResult(success=False, error=f"File not found: {file_path}")

        # Try real LSP first
        client = await self._get_client(str(path))
        if client and client._initialized:
            try:
                result = await client.hover(str(path), line, character)
                if result:
                    return LSPResult(
                        success=True,
                        data={"contents": result.contents},
                        source="lsp",
                    )
            except Exception:
                pass

        # Fallback: extract symbol and provide basic info
        return await self._regex_hover(path, line, character)

    async def document_symbol(self, file_path: str) -> LSPResult:
        """Get all symbols in a document."""
        path = self._resolve_path(file_path)
        if not path.exists():
            return LSPResult(success=False, error=f"File not found: {file_path}")

        # Try real LSP first
        client = await self._get_client(str(path))
        if client and client._initialized:
            try:
                symbols = await client.document_symbols(str(path))
                if symbols:
                    return LSPResult(
                        success=True,
                        data=[self._symbol_to_dict(sym) for sym in symbols],
                        source="lsp",
                    )
            except Exception:
                pass

        # Fallback to regex
        return await self._regex_document_symbols(path)

    async def workspace_symbol(self, query: str) -> LSPResult:
        """Search for symbols across the workspace."""
        # Try real LSP first
        manager = await self._get_lsp_manager()
        for client in manager.clients.values():
            if client._initialized:
                try:
                    symbols = await client.workspace_symbols(query)
                    if symbols:
                        return LSPResult(
                            success=True,
                            data=[self._symbol_to_dict(sym) for sym in symbols],
                            source="lsp",
                        )
                except Exception:
                    pass

        # Fallback to regex search
        return await self._regex_workspace_symbols(query)

    async def go_to_implementation(
        self,
        file_path: str,
        line: int,
        character: int
    ) -> LSPResult:
        """Find implementations of an interface or abstract method."""
        path = self._resolve_path(file_path)
        if not path.exists():
            return LSPResult(success=False, error=f"File not found: {file_path}")

        # Try real LSP first
        client = await self._get_client(str(path))
        if client and client._initialized:
            try:
                locations = await client.go_to_implementation(str(path), line, character)
                if locations:
                    return LSPResult(
                        success=True,
                        data=[self._location_to_dict(loc) for loc in locations],
                        source="lsp",
                    )
            except Exception:
                pass

        # No good regex fallback for implementations
        return LSPResult(
            success=False,
            error="LSP server not available for implementation lookup",
            source="regex",
        )

    async def prepare_call_hierarchy(
        self,
        file_path: str,
        line: int,
        character: int
    ) -> LSPResult:
        """Get call hierarchy item at a position."""
        path = self._resolve_path(file_path)
        if not path.exists():
            return LSPResult(success=False, error=f"File not found: {file_path}")

        # Try real LSP first
        client = await self._get_client(str(path))
        if client and client._initialized:
            try:
                items = await client.prepare_call_hierarchy(str(path), line, character)
                if items:
                    return LSPResult(
                        success=True,
                        data=[self._call_hierarchy_to_dict(item) for item in items],
                        source="lsp",
                    )
            except Exception:
                pass

        # Basic regex fallback - find function at position
        return await self._regex_call_hierarchy(path, line, character)

    async def incoming_calls(
        self,
        file_path: str,
        line: int,
        character: int
    ) -> LSPResult:
        """Find all callers of the function at position."""
        # First prepare call hierarchy
        prep_result = await self.prepare_call_hierarchy(file_path, line, character)
        if not prep_result.success or not prep_result.data:
            return LSPResult(
                success=False,
                error="Could not find function at position",
            )

        if prep_result.source == "lsp":
            path = self._resolve_path(file_path)
            client = await self._get_client(str(path))
            if client and client._initialized:
                try:
                    # Reconstruct call hierarchy item
                    item_data = prep_result.data[0]
                    item = LSPCallHierarchyItem(
                        name=item_data["name"],
                        kind=item_data["kind"],
                        uri=item_data["uri"],
                        range=self._dict_to_range(item_data["range"]),
                        selection_range=self._dict_to_range(item_data["selection_range"]),
                    )
                    calls = await client.incoming_calls(item)
                    return LSPResult(
                        success=True,
                        data=[{
                            "from": self._call_hierarchy_to_dict(call[0]),
                            "from_ranges": [self._range_to_dict(r) for r in call[1]]
                        } for call in calls],
                        source="lsp",
                    )
                except Exception:
                    pass

        # Regex fallback - search for function name usage
        return await self._regex_find_callers(prep_result.data[0]["name"])

    async def outgoing_calls(
        self,
        file_path: str,
        line: int,
        character: int
    ) -> LSPResult:
        """Find all functions called by the function at position."""
        # First prepare call hierarchy
        prep_result = await self.prepare_call_hierarchy(file_path, line, character)
        if not prep_result.success or not prep_result.data:
            return LSPResult(
                success=False,
                error="Could not find function at position",
            )

        if prep_result.source == "lsp":
            path = self._resolve_path(file_path)
            client = await self._get_client(str(path))
            if client and client._initialized:
                try:
                    item_data = prep_result.data[0]
                    item = LSPCallHierarchyItem(
                        name=item_data["name"],
                        kind=item_data["kind"],
                        uri=item_data["uri"],
                        range=self._dict_to_range(item_data["range"]),
                        selection_range=self._dict_to_range(item_data["selection_range"]),
                    )
                    calls = await client.outgoing_calls(item)
                    return LSPResult(
                        success=True,
                        data=[{
                            "to": self._call_hierarchy_to_dict(call[0]),
                            "from_ranges": [self._range_to_dict(r) for r in call[1]]
                        } for call in calls],
                        source="lsp",
                    )
                except Exception:
                    pass

        # No good regex fallback for outgoing calls
        return LSPResult(
            success=False,
            error="LSP server not available for outgoing call analysis",
            source="regex",
        )

    # ============================================================
    #  Regex Fallback Methods
    # ============================================================

    async def _regex_find_definition(
        self,
        path: Path,
        line: int,
        character: int
    ) -> LSPResult:
        """Find definition using regex."""
        try:
            content = path.read_text(encoding='utf-8')
            lines = content.splitlines()

            if line < 1 or line > len(lines):
                return LSPResult(success=False, error="Line out of range")

            # Extract symbol at position
            target_line = lines[line - 1]
            symbol = self._extract_symbol_at_position(target_line, character)

            if not symbol:
                return LSPResult(success=False, error="No symbol at position")

            # Search for definition
            definitions = []
            _ = self._get_language(str(path))  # Language detection for future use

            # Build definition patterns
            def_patterns = [
                rf"^\s*(?:async\s+)?def\s+{re.escape(symbol)}\s*\(",
                rf"^\s*class\s+{re.escape(symbol)}[\s:(]",
                rf"(?:const|let|var|function)\s+{re.escape(symbol)}\s*[=:(]",
                rf"interface\s+{re.escape(symbol)}\s*[\{{<]",
                rf"type\s+{re.escape(symbol)}\s*=",
                rf"func\s+(?:\([^)]+\)\s+)?{re.escape(symbol)}\s*\(",
                rf"struct\s+{re.escape(symbol)}\s*\{{",
            ]
            combined = "|".join(f"({p})" for p in def_patterns)

            # Search in current file first
            for i, file_line in enumerate(lines, 1):
                if re.search(combined, file_line):
                    definitions.append({
                        "file": str(path.relative_to(self.workspace_root)),
                        "line": i,
                        "character": 0,
                        "content": file_line.strip()
                    })

            # Search in workspace
            for fp in self.workspace_root.rglob("*"):
                if fp == path or not fp.is_file():
                    continue
                if fp.suffix not in ['.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs']:
                    continue

                try:
                    file_content = fp.read_text(encoding='utf-8', errors='ignore')
                    for i, file_line in enumerate(file_content.splitlines(), 1):
                        if re.search(combined, file_line):
                            definitions.append({
                                "file": str(fp.relative_to(self.workspace_root)),
                                "line": i,
                                "character": 0,
                                "content": file_line.strip()
                            })
                except Exception:
                    continue

            return LSPResult(
                success=True,
                data=definitions,
                source="regex",
                metadata={"symbol": symbol}
            )

        except Exception as e:
            return LSPResult(success=False, error=str(e))

    async def _regex_find_references(
        self,
        path: Path,
        line: int,
        character: int
    ) -> LSPResult:
        """Find references using regex."""
        try:
            content = path.read_text(encoding='utf-8')
            lines = content.splitlines()

            if line < 1 or line > len(lines):
                return LSPResult(success=False, error="Line out of range")

            target_line = lines[line - 1]
            symbol = self._extract_symbol_at_position(target_line, character)

            if not symbol:
                return LSPResult(success=False, error="No symbol at position")

            pattern = rf"\b{re.escape(symbol)}\b"
            references = []

            for fp in self.workspace_root.rglob("*"):
                if not fp.is_file():
                    continue
                if fp.suffix not in ['.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs']:
                    continue

                try:
                    file_content = fp.read_text(encoding='utf-8', errors='ignore')
                    for i, file_line in enumerate(file_content.splitlines(), 1):
                        if re.search(pattern, file_line):
                            references.append({
                                "file": str(fp.relative_to(self.workspace_root)),
                                "line": i,
                                "character": 0,
                                "content": file_line.strip()
                            })
                except Exception:
                    continue

            return LSPResult(
                success=True,
                data=references,
                source="regex",
                metadata={"symbol": symbol}
            )

        except Exception as e:
            return LSPResult(success=False, error=str(e))

    async def _regex_hover(
        self,
        path: Path,
        line: int,
        character: int
    ) -> LSPResult:
        """Get basic hover info using regex."""
        try:
            content = path.read_text(encoding='utf-8')
            lines = content.splitlines()

            if line < 1 or line > len(lines):
                return LSPResult(success=False, error="Line out of range")

            target_line = lines[line - 1]
            symbol = self._extract_symbol_at_position(target_line, character)

            if not symbol:
                return LSPResult(success=False, error="No symbol at position")

            # Try to find definition and extract docstring
            language = self._get_language(str(path))

            # Look for docstring in Python
            if language == "python":
                for i, file_line in enumerate(lines):
                    if re.search(rf"^\s*def\s+{re.escape(symbol)}\s*\(", file_line):
                        # Look for docstring
                        if i + 1 < len(lines):
                            next_lines = lines[i + 1:i + 10]
                            docstring = self._extract_python_docstring(next_lines)
                            if docstring:
                                return LSPResult(
                                    success=True,
                                    data={"contents": f"**{symbol}**\n\n{docstring}"},
                                    source="regex",
                                )

            return LSPResult(
                success=True,
                data={"contents": f"Symbol: **{symbol}**"},
                source="regex",
            )

        except Exception as e:
            return LSPResult(success=False, error=str(e))

    async def _regex_document_symbols(self, path: Path) -> LSPResult:
        """Get document symbols using regex."""
        try:
            content = path.read_text(encoding='utf-8')
            lines = content.splitlines()
            language = self._get_language(str(path))
            patterns = self.PATTERNS.get(language, self.PATTERNS["python"])

            symbols = []
            for line_num, line in enumerate(lines, 1):
                for symbol_type, pattern in patterns.items():
                    match = re.search(pattern, line)
                    if match:
                        name = next((g for g in match.groups() if g), None)
                        if name:
                            symbols.append({
                                "name": name,
                                "kind": self._symbol_type_to_kind(symbol_type),
                                "kind_name": symbol_type,
                                "line": line_num,
                                "character": match.start(),
                                "preview": line.strip()
                            })

            return LSPResult(
                success=True,
                data=symbols,
                source="regex",
                metadata={"language": language}
            )

        except Exception as e:
            return LSPResult(success=False, error=str(e))

    async def _regex_workspace_symbols(self, query: str) -> LSPResult:
        """Search workspace for symbols matching query."""
        try:
            symbols = []
            query_lower = query.lower()

            for fp in self.workspace_root.rglob("*"):
                if not fp.is_file():
                    continue
                if fp.suffix not in ['.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs']:
                    continue

                try:
                    content = fp.read_text(encoding='utf-8', errors='ignore')
                    language = self._get_language(str(fp))
                    patterns = self.PATTERNS.get(language, self.PATTERNS["python"])

                    for line_num, line in enumerate(content.splitlines(), 1):
                        for symbol_type, pattern in patterns.items():
                            match = re.search(pattern, line)
                            if match:
                                name = next((g for g in match.groups() if g), None)
                                if name and query_lower in name.lower():
                                    symbols.append({
                                        "name": name,
                                        "kind": self._symbol_type_to_kind(symbol_type),
                                        "kind_name": symbol_type,
                                        "file": str(fp.relative_to(self.workspace_root)),
                                        "line": line_num,
                                    })
                except Exception:
                    continue

            return LSPResult(
                success=True,
                data=symbols[:100],  # Limit results
                source="regex",
                metadata={"query": query}
            )

        except Exception as e:
            return LSPResult(success=False, error=str(e))

    async def _regex_call_hierarchy(
        self,
        path: Path,
        line: int,
        character: int  # noqa: ARG002 - Required for interface
    ) -> LSPResult:
        """Get basic call hierarchy info using regex."""
        try:
            content = path.read_text(encoding='utf-8')
            lines = content.splitlines()

            if line < 1 or line > len(lines):
                return LSPResult(success=False, error="Line out of range")

            # Find function at this line
            language = self._get_language(str(path))
            func_patterns = {
                "python": r"^\s*(?:async\s+)?def\s+(\w+)",
                "javascript": r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=)",
                "typescript": r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=)",
                "go": r"func\s+(?:\([^)]+\)\s+)?(\w+)",
                "rust": r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)",
            }

            pattern = func_patterns.get(language, func_patterns["python"])

            # Search around the line for a function
            for i in range(max(0, line - 5), min(len(lines), line + 1)):
                match = re.search(pattern, lines[i])
                if match:
                    name = next((g for g in match.groups() if g), None)
                    if name:
                        return LSPResult(
                            success=True,
                            data=[{
                                "name": name,
                                "kind": 12,  # Function
                                "uri": f"file://{path}",
                                "range": {
                                    "start": {"line": i, "character": 0},
                                    "end": {"line": i, "character": len(lines[i])}
                                },
                                "selection_range": {
                                    "start": {"line": i, "character": match.start()},
                                    "end": {"line": i, "character": match.end()}
                                },
                            }],
                            source="regex",
                        )

            return LSPResult(success=False, error="No function found at position")

        except Exception as e:
            return LSPResult(success=False, error=str(e))

    async def _regex_find_callers(self, function_name: str) -> LSPResult:
        """Find callers of a function using regex."""
        try:
            # Look for function calls
            pattern = rf"\b{re.escape(function_name)}\s*\("
            callers = []

            for fp in self.workspace_root.rglob("*"):
                if not fp.is_file():
                    continue
                if fp.suffix not in ['.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs']:
                    continue

                try:
                    content = fp.read_text(encoding='utf-8', errors='ignore')
                    for i, line in enumerate(content.splitlines(), 1):
                        if re.search(pattern, line):
                            # Skip definition lines
                            if re.search(rf"def\s+{re.escape(function_name)}", line):
                                continue
                            if re.search(rf"function\s+{re.escape(function_name)}", line):
                                continue

                            callers.append({
                                "file": str(fp.relative_to(self.workspace_root)),
                                "line": i,
                                "content": line.strip()
                            })
                except Exception:
                    continue

            return LSPResult(
                success=True,
                data=callers,
                source="regex",
                metadata={"function": function_name}
            )

        except Exception as e:
            return LSPResult(success=False, error=str(e))

    # ============================================================
    #  Helper Methods
    # ============================================================

    def _extract_symbol_at_position(self, line: str, character: int) -> Optional[str]:
        """Extract the symbol at a given character position."""
        if character < 1 or character > len(line):
            return None

        # Find word boundaries
        start = character - 1
        end = character - 1

        while start > 0 and (line[start - 1].isalnum() or line[start - 1] == '_'):
            start -= 1

        while end < len(line) and (line[end].isalnum() or line[end] == '_'):
            end += 1

        symbol = line[start:end]
        return symbol if symbol else None

    def _extract_python_docstring(self, lines: List[str]) -> Optional[str]:
        """Extract Python docstring from lines after function def."""
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                quote = stripped[:3]
                if stripped.endswith(quote) and len(stripped) > 6:
                    return stripped[3:-3]
                # Multi-line docstring
                docstring_lines = [stripped[3:]]
                for j in range(i + 1, len(lines)):
                    next_line = lines[j]
                    if quote in next_line:
                        docstring_lines.append(next_line.split(quote)[0])
                        return '\n'.join(docstring_lines)
                    docstring_lines.append(next_line.strip())
                break
            elif stripped and not stripped.startswith('#'):
                break
        return None

    def _symbol_type_to_kind(self, symbol_type: str) -> int:
        """Convert symbol type string to LSP SymbolKind."""
        return {
            "function": 12,
            "class": 5,
            "interface": 11,
            "type": 26,
            "variable": 13,
            "constant": 14,
            "struct": 23,
            "enum": 10,
            "trait": 11,
            "impl": 5,
            "import": 2,
        }.get(symbol_type, 13)

    def _location_to_dict(self, loc: LSPLocation) -> Dict[str, Any]:
        """Convert LSPLocation to dict."""
        return {
            "file": loc.file_path,
            "line": loc.range.start.line + 1,
            "character": loc.range.start.character + 1,
        }

    def _symbol_to_dict(self, sym: LSPSymbol) -> Dict[str, Any]:
        """Convert LSPSymbol to dict."""
        result = {
            "name": sym.name,
            "kind": sym.kind,
            "kind_name": sym.kind_name,
        }
        if sym.location:
            result["file"] = sym.location.file_path
            result["line"] = sym.location.range.start.line + 1
        if sym.container_name:
            result["container"] = sym.container_name
        if sym.detail:
            result["detail"] = sym.detail
        return result

    def _call_hierarchy_to_dict(self, item: LSPCallHierarchyItem) -> Dict[str, Any]:
        """Convert LSPCallHierarchyItem to dict."""
        return {
            "name": item.name,
            "kind": item.kind,
            "uri": item.uri,
            "range": self._range_to_dict(item.range),
            "selection_range": self._range_to_dict(item.selection_range),
            "detail": item.detail,
        }

    def _range_to_dict(self, range_obj) -> Dict[str, Any]:
        """Convert LSPRange to dict."""
        return {
            "start": {"line": range_obj.start.line, "character": range_obj.start.character},
            "end": {"line": range_obj.end.line, "character": range_obj.end.character},
        }

    def _dict_to_range(self, d: Dict[str, Any]):
        """Convert dict to LSPRange."""
        from .lsp_server import LSPRange, LSPPosition
        return LSPRange(
            start=LSPPosition(d["start"]["line"], d["start"]["character"]),
            end=LSPPosition(d["end"]["line"], d["end"]["character"]),
        )

    async def cleanup(self) -> None:
        """Clean up LSP connections."""
        if self._lsp_manager:
            await self._lsp_manager.stop_all()
