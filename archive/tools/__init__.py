"""Tools module for Company AGI - Unified toolkit with Claude Code-style capabilities."""

from .unified_tools import (
    UnifiedTools,
    ToolResult,
    ReadTool,
    EditTool,
    MultiEditTool,
    WriteTool,
    GlobTool,
    GrepTool,
    BashTool,
    WebFetchTool,
    WebSearchTool,
    TodoTool,
    TodoItem,
    TodoStatus,
    NotebookTool,
    TaskTool,
    BackgroundTask,
    LSPTool,
)

from .mcp import (
    MCPManager,
    MCPServerConfig,
    MCPTransport,
    MCPTool,
    MCPResource,
    MCPResult,
    MCPConnection,
    MCPToolWrapper,
    get_mcp_manager,
)

from .lsp_server import (
    LSPManager,
    LSPClient,
    LSPServerType,
    LSPPosition,
    LSPRange,
    LSPLocation,
    LSPSymbol,
    LSPDiagnostic,
    LSPHoverResult,
    LSPCompletionItem,
    LSPCallHierarchyItem,
    LSPServerConfig,
    get_lsp_manager,
)

from .enhanced_lsp import (
    EnhancedLSPTool,
    LSPResult as EnhancedLSPResult,
)

from .file_operations import FileOperations
from .command_executor import CommandExecutor
from .test_runner import TestRunner, TestResult, run_project_tests
from .code_formatter import CodeFormatter, FormatResult, format_project, format_file

from .git_tools import (
    GitTools,
    GitChange,
    GitChangeType,
    GitCommit,
    GitBranch,
    GitResult,
    PRInfo,
    get_git_tools,
    reset_git_tools,
)

from .image_tools import (
    ImageTools,
    ImageData,
    ImageMetadata,
    ImageFormat,
    ImageResult,
    get_image_tools,
    reset_image_tools,
)

from .pdf_tools import (
    PDFTools,
    PDFData,
    PDFPage,
    PDFMetadata,
    PDFResult,
    PDFExtractionMode,
    get_pdf_tools,
    reset_pdf_tools,
)

__all__ = [
    # Unified tools (primary toolkit)
    "UnifiedTools",
    "ToolResult",
    "ReadTool",
    "EditTool",
    "MultiEditTool",
    "WriteTool",
    "GlobTool",
    "GrepTool",
    "BashTool",
    "WebFetchTool",
    "WebSearchTool",
    "TodoTool",
    "TodoItem",
    "TodoStatus",
    "NotebookTool",
    "TaskTool",
    "BackgroundTask",
    "LSPTool",
    # MCP (Model Context Protocol)
    "MCPManager",
    "MCPServerConfig",
    "MCPTransport",
    "MCPTool",
    "MCPResource",
    "MCPResult",
    "MCPConnection",
    "MCPToolWrapper",
    "get_mcp_manager",
    # Enhanced LSP (Language Server Protocol)
    "LSPManager",
    "LSPClient",
    "LSPServerType",
    "LSPPosition",
    "LSPRange",
    "LSPLocation",
    "LSPSymbol",
    "LSPDiagnostic",
    "LSPHoverResult",
    "LSPCompletionItem",
    "LSPCallHierarchyItem",
    "LSPServerConfig",
    "get_lsp_manager",
    "EnhancedLSPTool",
    "EnhancedLSPResult",
    # Legacy tools (for compatibility)
    "FileOperations",
    "CommandExecutor",
    "TestRunner",
    "TestResult",
    "run_project_tests",
    "CodeFormatter",
    "FormatResult",
    "format_project",
    "format_file",
    # Git tools
    "GitTools",
    "GitChange",
    "GitChangeType",
    "GitCommit",
    "GitBranch",
    "GitResult",
    "PRInfo",
    "get_git_tools",
    "reset_git_tools",
    # Image tools
    "ImageTools",
    "ImageData",
    "ImageMetadata",
    "ImageFormat",
    "ImageResult",
    "get_image_tools",
    "reset_image_tools",
    # PDF tools
    "PDFTools",
    "PDFData",
    "PDFPage",
    "PDFMetadata",
    "PDFResult",
    "PDFExtractionMode",
    "get_pdf_tools",
    "reset_pdf_tools",
]
