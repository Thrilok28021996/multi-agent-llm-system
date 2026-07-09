"""
Enhanced LSP Integration for Company AGI.

Provides real Language Server Protocol support with:
- Multiple language server connections
- Go to definition, find references, hover, etc.
- Document symbols, workspace symbols
- Call hierarchy (incoming/outgoing calls)
- Diagnostics and code actions
- Graceful fallback to regex when server unavailable
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LSPServerType(Enum):
    """Supported LSP server types."""
    PYTHON = "python"  # pylsp, pyright
    TYPESCRIPT = "typescript"  # typescript-language-server
    JAVASCRIPT = "javascript"  # typescript-language-server
    GO = "go"  # gopls
    RUST = "rust"  # rust-analyzer
    JAVA = "java"  # jdtls
    C_CPP = "c_cpp"  # clangd


@dataclass
class LSPPosition:
    """Position in a document (0-indexed)."""
    line: int
    character: int

    def to_dict(self) -> Dict[str, int]:
        return {"line": self.line, "character": self.character}

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "LSPPosition":
        return cls(line=data["line"], character=data["character"])


@dataclass
class LSPRange:
    """Range in a document."""
    start: LSPPosition
    end: LSPPosition

    def to_dict(self) -> Dict[str, Any]:
        return {"start": self.start.to_dict(), "end": self.end.to_dict()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LSPRange":
        return cls(
            start=LSPPosition.from_dict(data["start"]),
            end=LSPPosition.from_dict(data["end"])
        )


@dataclass
class LSPLocation:
    """Location in a file."""
    uri: str
    range: LSPRange

    def to_dict(self) -> Dict[str, Any]:
        return {"uri": self.uri, "range": self.range.to_dict()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LSPLocation":
        return cls(
            uri=data["uri"],
            range=LSPRange.from_dict(data["range"])
        )

    @property
    def file_path(self) -> str:
        """Convert URI to file path."""
        if self.uri.startswith("file://"):
            return self.uri[7:]
        return self.uri


@dataclass
class LSPSymbol:
    """Symbol information."""
    name: str
    kind: int  # SymbolKind enum value
    location: Optional[LSPLocation] = None
    container_name: Optional[str] = None
    detail: Optional[str] = None

    SYMBOL_KINDS = {
        1: "File", 2: "Module", 3: "Namespace", 4: "Package",
        5: "Class", 6: "Method", 7: "Property", 8: "Field",
        9: "Constructor", 10: "Enum", 11: "Interface", 12: "Function",
        13: "Variable", 14: "Constant", 15: "String", 16: "Number",
        17: "Boolean", 18: "Array", 19: "Object", 20: "Key",
        21: "Null", 22: "EnumMember", 23: "Struct", 24: "Event",
        25: "Operator", 26: "TypeParameter"
    }

    @property
    def kind_name(self) -> str:
        return self.SYMBOL_KINDS.get(self.kind, "Unknown")


@dataclass
class LSPDiagnostic:
    """Diagnostic information (error, warning, etc.)."""
    range: LSPRange
    message: str
    severity: int  # 1=Error, 2=Warning, 3=Info, 4=Hint
    source: Optional[str] = None
    code: Optional[str] = None

    SEVERITY_NAMES = {1: "Error", 2: "Warning", 3: "Info", 4: "Hint"}

    @property
    def severity_name(self) -> str:
        return self.SEVERITY_NAMES.get(self.severity, "Unknown")


@dataclass
class LSPHoverResult:
    """Hover information."""
    contents: str
    range: Optional[LSPRange] = None


@dataclass
class LSPCompletionItem:
    """Completion suggestion."""
    label: str
    kind: int
    detail: Optional[str] = None
    documentation: Optional[str] = None
    insert_text: Optional[str] = None


@dataclass
class LSPCallHierarchyItem:
    """Call hierarchy item."""
    name: str
    kind: int
    uri: str
    range: LSPRange
    selection_range: LSPRange
    detail: Optional[str] = None


@dataclass
class LSPServerConfig:
    """Configuration for an LSP server."""
    server_type: LSPServerType
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    initialization_options: Dict[str, Any] = field(default_factory=dict)
    root_path: Optional[str] = None
    file_extensions: List[str] = field(default_factory=list)


# Default server configurations
DEFAULT_LSP_CONFIGS: Dict[LSPServerType, LSPServerConfig] = {
    LSPServerType.PYTHON: LSPServerConfig(
        server_type=LSPServerType.PYTHON,
        command="pylsp",
        args=[],
        file_extensions=[".py", ".pyi"],
        initialization_options={
            "pylsp": {
                "plugins": {
                    "pycodestyle": {"enabled": False},
                    "mccabe": {"enabled": False},
                    "pyflakes": {"enabled": True},
                    "pylint": {"enabled": False},
                }
            }
        }
    ),
    LSPServerType.TYPESCRIPT: LSPServerConfig(
        server_type=LSPServerType.TYPESCRIPT,
        command="typescript-language-server",
        args=["--stdio"],
        file_extensions=[".ts", ".tsx", ".js", ".jsx"],
    ),
    LSPServerType.GO: LSPServerConfig(
        server_type=LSPServerType.GO,
        command="gopls",
        args=["serve"],
        file_extensions=[".go"],
    ),
    LSPServerType.RUST: LSPServerConfig(
        server_type=LSPServerType.RUST,
        command="rust-analyzer",
        args=[],
        file_extensions=[".rs"],
    ),
}


class LSPClient:
    """Client for communicating with an LSP server."""

    def __init__(self, config: LSPServerConfig, workspace_root: str):
        self.config = config
        self.workspace_root = Path(workspace_root).resolve()
        self.process: Optional[asyncio.subprocess.Process] = None
        self._request_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._initialized = False
        self._open_documents: Dict[str, int] = {}  # uri -> version

    async def start(self) -> bool:
        """Start the LSP server."""
        try:
            env = os.environ.copy()
            env.update(self.config.env)

            self.process = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # Start reader task
            self._reader_task = asyncio.create_task(self._read_messages())

            # Initialize
            init_result = await self._initialize()
            if init_result:
                self._initialized = True
                await self._send_notification("initialized", {})

            return self._initialized

        except FileNotFoundError:
            return False
        except Exception as e:
            logger.warning("Failed to start LSP server: %s", e)
            return False

    async def stop(self) -> None:
        """Stop the LSP server."""
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self.process:
            try:
                await self._send_request("shutdown", {})
                await self._send_notification("exit", {})
            except Exception:
                pass

            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()

        self._initialized = False

    async def _initialize(self) -> bool:
        """Send initialize request."""
        result = await self._send_request("initialize", {
            "processId": os.getpid(),
            "rootUri": f"file://{self.workspace_root}",
            "rootPath": str(self.workspace_root),
            "capabilities": {
                "textDocument": {
                    "synchronization": {"dynamicRegistration": False},
                    "completion": {"dynamicRegistration": False},
                    "hover": {"dynamicRegistration": False},
                    "definition": {"dynamicRegistration": False},
                    "references": {"dynamicRegistration": False},
                    "documentSymbol": {"dynamicRegistration": False},
                    "codeAction": {"dynamicRegistration": False},
                    "formatting": {"dynamicRegistration": False},
                },
                "workspace": {
                    "symbol": {"dynamicRegistration": False},
                    "workspaceFolders": True,
                }
            },
            "initializationOptions": self.config.initialization_options,
            "workspaceFolders": [
                {"uri": f"file://{self.workspace_root}", "name": self.workspace_root.name}
            ]
        })

        return result is not None

    async def _send_request(self, method: str, params: Dict[str, Any]) -> Optional[Any]:
        """Send a JSON-RPC request and wait for response."""
        if not self.process or not self.process.stdin:
            return None

        self._request_id += 1
        request_id = self._request_id

        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        try:
            await self._send_message(message)
            result = await asyncio.wait_for(future, timeout=30)
            return result
        except asyncio.TimeoutError:
            del self._pending_requests[request_id]
            return None
        except Exception:
            if request_id in self._pending_requests:
                del self._pending_requests[request_id]
            return None

    async def _send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        await self._send_message(message)

    async def _send_message(self, message: Dict[str, Any]) -> None:
        """Send a message to the server."""
        if not self.process or not self.process.stdin:
            return

        content = json.dumps(message)
        header = f"Content-Length: {len(content)}\r\n\r\n"
        self.process.stdin.write(header.encode() + content.encode())
        await self.process.stdin.drain()

    async def _read_messages(self) -> None:
        """Read messages from the server."""
        if not self.process or not self.process.stdout:
            return

        buffer = b""
        while True:
            try:
                chunk = await self.process.stdout.read(4096)
                if not chunk:
                    break

                buffer += chunk

                while True:
                    # Parse header
                    header_end = buffer.find(b"\r\n\r\n")
                    if header_end == -1:
                        break

                    header = buffer[:header_end].decode()
                    content_length = 0
                    for line in header.split("\r\n"):
                        if line.lower().startswith("content-length:"):
                            content_length = int(line.split(":")[1].strip())
                            break

                    if content_length == 0:
                        break

                    content_start = header_end + 4
                    content_end = content_start + content_length

                    if len(buffer) < content_end:
                        break

                    content = buffer[content_start:content_end]
                    buffer = buffer[content_end:]

                    try:
                        message = json.loads(content.decode())
                        self._handle_message(message)
                    except json.JSONDecodeError:
                        pass

            except asyncio.CancelledError:
                break
            except Exception:
                continue

    def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle a received message."""
        if "id" in message and message["id"] in self._pending_requests:
            future = self._pending_requests.pop(message["id"])
            if "error" in message:
                future.set_result(None)
            else:
                future.set_result(message.get("result"))

    # ============================================================
    #  LSP Operations
    # ============================================================

    async def open_document(self, file_path: str) -> bool:
        """Open a document in the LSP server."""
        uri = f"file://{Path(file_path).resolve()}"

        if uri in self._open_documents:
            return True

        try:
            content = Path(file_path).read_text(encoding='utf-8')
            language_id = self._get_language_id(file_path)

            await self._send_notification("textDocument/didOpen", {
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id,
                    "version": 1,
                    "text": content,
                }
            })

            self._open_documents[uri] = 1
            return True

        except Exception:
            return False

    async def close_document(self, file_path: str) -> None:
        """Close a document in the LSP server."""
        uri = f"file://{Path(file_path).resolve()}"

        if uri not in self._open_documents:
            return

        await self._send_notification("textDocument/didClose", {
            "textDocument": {"uri": uri}
        })

        del self._open_documents[uri]

    async def go_to_definition(
        self,
        file_path: str,
        line: int,
        character: int
    ) -> List[LSPLocation]:
        """Go to definition of symbol at position."""
        await self.open_document(file_path)

        uri = f"file://{Path(file_path).resolve()}"
        result = await self._send_request("textDocument/definition", {
            "textDocument": {"uri": uri},
            "position": {"line": line - 1, "character": character - 1}  # Convert to 0-indexed
        })

        if not result:
            return []

        locations = []
        if isinstance(result, dict):
            result = [result]

        for loc in result:
            if "uri" in loc and "range" in loc:
                locations.append(LSPLocation.from_dict(loc))
            elif "targetUri" in loc:  # LocationLink
                locations.append(LSPLocation(
                    uri=loc["targetUri"],
                    range=LSPRange.from_dict(loc["targetRange"])
                ))

        return locations

    async def find_references(
        self,
        file_path: str,
        line: int,
        character: int,
        include_declaration: bool = True
    ) -> List[LSPLocation]:
        """Find all references to symbol at position."""
        await self.open_document(file_path)

        uri = f"file://{Path(file_path).resolve()}"
        result = await self._send_request("textDocument/references", {
            "textDocument": {"uri": uri},
            "position": {"line": line - 1, "character": character - 1},
            "context": {"includeDeclaration": include_declaration}
        })

        if not result:
            return []

        return [LSPLocation.from_dict(loc) for loc in result]

    async def hover(
        self,
        file_path: str,
        line: int,
        character: int
    ) -> Optional[LSPHoverResult]:
        """Get hover information at position."""
        await self.open_document(file_path)

        uri = f"file://{Path(file_path).resolve()}"
        result = await self._send_request("textDocument/hover", {
            "textDocument": {"uri": uri},
            "position": {"line": line - 1, "character": character - 1}
        })

        if not result or "contents" not in result:
            return None

        contents = result["contents"]
        if isinstance(contents, dict):
            contents = contents.get("value", str(contents))
        elif isinstance(contents, list):
            contents = "\n".join(
                c.get("value", str(c)) if isinstance(c, dict) else str(c)
                for c in contents
            )

        return LSPHoverResult(
            contents=str(contents),
            range=LSPRange.from_dict(result["range"]) if "range" in result else None
        )

    async def document_symbols(self, file_path: str) -> List[LSPSymbol]:
        """Get all symbols in a document."""
        await self.open_document(file_path)

        uri = f"file://{Path(file_path).resolve()}"
        result = await self._send_request("textDocument/documentSymbol", {
            "textDocument": {"uri": uri}
        })

        if not result:
            return []

        return self._parse_symbols(result, uri)

    async def workspace_symbols(self, query: str) -> List[LSPSymbol]:
        """Search for symbols in the workspace."""
        result = await self._send_request("workspace/symbol", {
            "query": query
        })

        if not result:
            return []

        symbols = []
        for sym in result:
            location = None
            if "location" in sym:
                location = LSPLocation.from_dict(sym["location"])

            symbols.append(LSPSymbol(
                name=sym["name"],
                kind=sym["kind"],
                location=location,
                container_name=sym.get("containerName"),
            ))

        return symbols

    async def get_diagnostics(self, file_path: str) -> List[LSPDiagnostic]:
        """Get diagnostics for a document."""
        # Note: Diagnostics are usually pushed via notifications
        # This is a placeholder for pull-based diagnostics
        await self.open_document(file_path)
        # Would need to implement diagnostic tracking from notifications
        return []

    async def prepare_call_hierarchy(
        self,
        file_path: str,
        line: int,
        character: int
    ) -> List[LSPCallHierarchyItem]:
        """Prepare call hierarchy at position."""
        await self.open_document(file_path)

        uri = f"file://{Path(file_path).resolve()}"
        result = await self._send_request("textDocument/prepareCallHierarchy", {
            "textDocument": {"uri": uri},
            "position": {"line": line - 1, "character": character - 1}
        })

        if not result:
            return []

        items = []
        for item in result:
            items.append(LSPCallHierarchyItem(
                name=item["name"],
                kind=item["kind"],
                uri=item["uri"],
                range=LSPRange.from_dict(item["range"]),
                selection_range=LSPRange.from_dict(item["selectionRange"]),
                detail=item.get("detail"),
            ))

        return items

    async def incoming_calls(
        self,
        item: LSPCallHierarchyItem
    ) -> List[Tuple[LSPCallHierarchyItem, List[LSPRange]]]:
        """Get incoming calls for a call hierarchy item."""
        result = await self._send_request("callHierarchy/incomingCalls", {
            "item": {
                "name": item.name,
                "kind": item.kind,
                "uri": item.uri,
                "range": item.range.to_dict(),
                "selectionRange": item.selection_range.to_dict(),
            }
        })

        if not result:
            return []

        calls = []
        for call in result:
            from_item = LSPCallHierarchyItem(
                name=call["from"]["name"],
                kind=call["from"]["kind"],
                uri=call["from"]["uri"],
                range=LSPRange.from_dict(call["from"]["range"]),
                selection_range=LSPRange.from_dict(call["from"]["selectionRange"]),
                detail=call["from"].get("detail"),
            )
            from_ranges = [LSPRange.from_dict(r) for r in call.get("fromRanges", [])]
            calls.append((from_item, from_ranges))

        return calls

    async def outgoing_calls(
        self,
        item: LSPCallHierarchyItem
    ) -> List[Tuple[LSPCallHierarchyItem, List[LSPRange]]]:
        """Get outgoing calls for a call hierarchy item."""
        result = await self._send_request("callHierarchy/outgoingCalls", {
            "item": {
                "name": item.name,
                "kind": item.kind,
                "uri": item.uri,
                "range": item.range.to_dict(),
                "selectionRange": item.selection_range.to_dict(),
            }
        })

        if not result:
            return []

        calls = []
        for call in result:
            to_item = LSPCallHierarchyItem(
                name=call["to"]["name"],
                kind=call["to"]["kind"],
                uri=call["to"]["uri"],
                range=LSPRange.from_dict(call["to"]["range"]),
                selection_range=LSPRange.from_dict(call["to"]["selectionRange"]),
                detail=call["to"].get("detail"),
            )
            from_ranges = [LSPRange.from_dict(r) for r in call.get("fromRanges", [])]
            calls.append((to_item, from_ranges))

        return calls

    async def go_to_implementation(
        self,
        file_path: str,
        line: int,
        character: int
    ) -> List[LSPLocation]:
        """Go to implementation of interface/abstract method."""
        await self.open_document(file_path)

        uri = f"file://{Path(file_path).resolve()}"
        result = await self._send_request("textDocument/implementation", {
            "textDocument": {"uri": uri},
            "position": {"line": line - 1, "character": character - 1}
        })

        if not result:
            return []

        if isinstance(result, dict):
            result = [result]

        return [LSPLocation.from_dict(loc) for loc in result]

    def _parse_symbols(self, symbols: List[Dict], uri: str) -> List[LSPSymbol]:
        """Parse symbol results (handles both flat and nested)."""
        result = []

        for sym in symbols:
            location = None
            if "location" in sym:
                location = LSPLocation.from_dict(sym["location"])
            elif "range" in sym:
                location = LSPLocation(uri=uri, range=LSPRange.from_dict(sym["range"]))

            result.append(LSPSymbol(
                name=sym["name"],
                kind=sym["kind"],
                location=location,
                detail=sym.get("detail"),
            ))

            # Handle nested children (DocumentSymbol)
            if "children" in sym:
                result.extend(self._parse_symbols(sym["children"], uri))

        return result

    def _get_language_id(self, file_path: str) -> str:
        """Get language ID from file extension."""
        ext = Path(file_path).suffix.lower()
        return {
            ".py": "python",
            ".pyi": "python",
            ".js": "javascript",
            ".jsx": "javascriptreact",
            ".ts": "typescript",
            ".tsx": "typescriptreact",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
        }.get(ext, "plaintext")


class LSPManager:
    """Manages multiple LSP server connections."""

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = Path(workspace_root).resolve()
        self.clients: Dict[LSPServerType, LSPClient] = {}
        self._extension_to_type: Dict[str, LSPServerType] = {}

        # Build extension mapping
        for server_type, config in DEFAULT_LSP_CONFIGS.items():
            for ext in config.file_extensions:
                self._extension_to_type[ext] = server_type

    async def get_client_for_file(self, file_path: str) -> Optional[LSPClient]:
        """Get or create an LSP client for a file."""
        ext = Path(file_path).suffix.lower()
        server_type = self._extension_to_type.get(ext)

        if not server_type:
            return None

        if server_type not in self.clients:
            config = DEFAULT_LSP_CONFIGS.get(server_type)
            if not config:
                return None

            client = LSPClient(config, str(self.workspace_root))
            if await client.start():
                self.clients[server_type] = client
            else:
                return None

        return self.clients.get(server_type)

    async def stop_all(self) -> None:
        """Stop all LSP clients."""
        for client in self.clients.values():
            await client.stop()
        self.clients.clear()

    def is_server_available(self, server_type: LSPServerType) -> bool:
        """Check if an LSP server is available on the system."""
        config = DEFAULT_LSP_CONFIGS.get(server_type)
        if not config:
            return False

        import shutil
        return shutil.which(config.command) is not None


# Global LSP manager
_global_lsp_manager: Optional[LSPManager] = None


def get_lsp_manager(workspace_root: str = ".") -> LSPManager:
    """Get or create the global LSP manager."""
    global _global_lsp_manager
    if _global_lsp_manager is None:
        _global_lsp_manager = LSPManager(workspace_root)
    return _global_lsp_manager


async def reset_lsp_manager() -> None:
    """Reset the global LSP manager."""
    global _global_lsp_manager
    if _global_lsp_manager:
        await _global_lsp_manager.stop_all()
    _global_lsp_manager = None
