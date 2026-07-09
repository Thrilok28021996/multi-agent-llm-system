"""Enhanced Code Parser - Supports multiple Claude Code-style formats with robust error handling."""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class FileMarkerFormat(Enum):
    """Supported file marker formats."""
    EQUALS = "=== FILE: {path} ==="  # Format 1
    MARKDOWN_BOLD = "**File Path:** `{path}`"  # Format 2
    HEADING = "#### File: {path}"  # Format 3
    CLAUDE_CODE = "File: {path}"  # Format 4 (Claude Code style)
    BACKTICKS = "`{path}`"  # Format 5 (just backticks)
    COLON = "{path}:"  # Format 6 (path with colon)
    COMMENT_STYLE = "# File: {path}"  # Format 7 (comment style)


@dataclass
class ParsedFile:
    """Represents a parsed file with metadata."""
    path: str
    content: str
    language: Optional[str] = None
    line_start: int = 0
    line_end: int = 0
    confidence: float = 1.0  # 0-1, how confident we are about the parse
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass
class ParseError:
    """Represents a parsing error."""
    line_number: int
    message: str
    context: str
    severity: str = "warning"  # warning, error, critical


class EnhancedCodeParser:
    """
    Enhanced code parser supporting multiple formats with robust error handling.

    Supports all Claude Code-style file markers and gracefully handles malformed input.
    """

    # File marker patterns (in order of specificity)
    FILE_PATTERNS = [
        # Format 1: === FILE: path/to/file.py ===
        (r'^={3,}\s*FILE:\s*(.+?)\s*={3,}$', 'equals'),

        # Format 2: **File Path:** `path/to/file.py`
        (r'\*\*\s*File\s+Path:\s*\*\*\s*`([^`]+)`', 'markdown_bold'),

        # Format 3: #### File: path/to/file.py
        (r'^#{1,6}\s+File:\s*`?([^`\n]+?)`?\s*$', 'heading'),

        # Format 4: File: path/to/file.py (Claude Code style)
        (r'^File:\s*`?([^`\n]+?)`?\s*$', 'claude_code'),

        # Format 5: <file>path/to/file.py</file> (XML style)
        (r'<file>([^<]+)</file>', 'xml'),

        # Format 6: // File: path/to/file.py (comment style)
        (r'^(?://|#|\*)\s*File:\s*(.+?)$', 'comment'),

        # Format 7: [file: path/to/file.py]
        (r'\[file:\s*([^\]]+)\]', 'bracket'),

        # Format 8: path/to/file.py: (at start of line with colon)
        (r'^([a-zA-Z0-9_/\-\.]+\.[a-z]{1,5}):\s*$', 'path_colon'),
    ]

    # Code block markers
    CODE_BLOCK_START = [
        (r'^```(\w+)?', 'fenced'),  # ```python
        (r'^~~~(\w+)?', 'tildes'),  # ~~~python
        (r'^    ', 'indented'),     # 4-space indent
        (r'^\t', 'tab_indented'),   # tab indent
    ]

    CODE_BLOCK_END = [
        r'^```',
        r'^~~~',
        r'^(?!\s{4}|\t)',  # not indented
    ]

    def __init__(self, strict_mode: bool = False):
        """
        Initialize parser.

        Args:
            strict_mode: If True, fail on any parsing errors. If False, collect warnings.
        """
        self.strict_mode = strict_mode
        self.errors: List[ParseError] = []
        self.warnings: List[ParseError] = []

    def parse(self, response: str) -> Dict[str, ParsedFile]:
        """
        Parse code files from LLM response.

        Args:
            response: LLM response containing code files

        Returns:
            Dictionary mapping file paths to ParsedFile objects

        Raises:
            ValueError: If strict_mode is True and critical errors are found
        """
        self.errors = []
        self.warnings = []

        files = {}
        current_file = None
        current_content = []
        current_language = None
        in_code_block = False
        code_block_type = None
        line_start = 0

        lines = response.split("\n")

        for line_num, line in enumerate(lines, start=1):
            # Try to match file marker
            file_match, marker_type = self._match_file_marker(line)

            if file_match:
                # Save previous file
                if current_file:
                    parsed_file = self._create_parsed_file(
                        current_file,
                        current_content,
                        current_language,
                        line_start,
                        line_num - 1
                    )
                    # Store marker type in metadata for debugging
                    parsed_file.metadata = {'marker_type': marker_type}
                    files[current_file] = parsed_file

                # Start new file
                current_file = file_match.strip()
                current_content = []
                current_language = self._detect_language(current_file)
                in_code_block = False
                code_block_type = None
                line_start = line_num

                # Validate file path
                if not self._is_valid_path(current_file):
                    self._add_warning(
                        line_num,
                        f"Suspicious file path: {current_file}",
                        line
                    )

                continue

            # Handle code block markers
            if not in_code_block:
                block_start, lang, block_type = self._match_code_block_start(line)
                if block_start:
                    in_code_block = True
                    code_block_type = block_type
                    if lang:
                        current_language = lang
                    continue
            else:
                if self._match_code_block_end(line, code_block_type):
                    in_code_block = False
                    code_block_type = None
                    continue

            # Collect content
            if current_file:
                if in_code_block:
                    # Inside code block - preserve all content
                    current_content.append(line)
                elif not line.strip().startswith('#') or code_block_type == 'indented':
                    # Outside code block - skip comments unless in indented block
                    if line.strip():  # Non-empty line
                        # Check if this looks like code (has valid syntax patterns)
                        if self._looks_like_code(line):
                            current_content.append(line)

        # Save last file
        if current_file:
            files[current_file] = self._create_parsed_file(
                current_file,
                current_content,
                current_language,
                line_start,
                len(lines)
            )

        # Check for errors
        if self.strict_mode and self.errors:
            error_msgs = [f"Line {e.line_number}: {e.message}" for e in self.errors]
            raise ValueError(f"Parsing errors:\n" + "\n".join(error_msgs))

        if not files:
            self._add_warning(0, "No files found in response", response)

        return files

    def _match_file_marker(self, line: str) -> Tuple[Optional[str], Optional[str]]:
        """Try to match a file marker pattern."""
        for pattern, marker_type in self.FILE_PATTERNS:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1).strip(), marker_type
        return None, None

    def _match_code_block_start(self, line: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Check if line starts a code block."""
        for pattern, block_type in self.CODE_BLOCK_START:
            match = re.match(pattern, line)
            if match:
                lang = match.group(1) if match.lastindex else None
                return True, lang, block_type
        return False, None, None

    def _match_code_block_end(self, line: str, block_type: Optional[str]) -> bool:
        """Check if line ends a code block."""
        if not block_type:
            return False

        if block_type in ['fenced', 'tildes']:
            return bool(re.match(r'^```|^~~~', line))
        elif block_type in ['indented', 'tab_indented']:
            # End indented block when we hit non-indented line
            return not (line.startswith('    ') or line.startswith('\t'))

        return False

    def _looks_like_code(self, line: str) -> bool:
        """Heuristic to determine if a line looks like code."""
        # Code indicators
        code_patterns = [
            r'^\s*(def|class|import|from|if|for|while|return|const|let|var|function)\b',
            r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*[=\(]',  # assignment or function call
            r'^\s*[}\])]',  # closing brackets
            r'^\s*[{]',  # opening brace
            r'^\s*//',  # comment
            r'^\s*#',   # comment
            r'^\s*\*',  # multi-line comment
            r'^\s*@',   # decorator
            r'^\s*<',   # HTML/XML tag
        ]

        return any(re.match(pattern, line) for pattern in code_patterns)

    def _detect_language(self, filepath: str) -> Optional[str]:
        """Detect programming language from file extension."""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'jsx',
            '.tsx': 'tsx',
            '.go': 'go',
            '.rs': 'rust',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.cs': 'csharp',
            '.rb': 'ruby',
            '.php': 'php',
            '.sh': 'bash',
            '.sql': 'sql',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.xml': 'xml',
            '.md': 'markdown',
        }

        for ext, lang in ext_map.items():
            if filepath.endswith(ext):
                return lang

        return None

    def _is_valid_path(self, path: str) -> bool:
        """Validate that a path looks reasonable."""
        # Check for suspicious patterns
        suspicious = [
            r'\.\./\.\.',  # path traversal
            r'^/',         # absolute path (usually not wanted)
            r'\\',         # Windows paths mixed with Unix
            r'\0',         # null bytes
            r'[<>"|?*]',   # invalid filename chars
        ]

        for pattern in suspicious:
            if re.search(pattern, path):
                return False

        # Must have valid file extension
        if not re.search(r'\.[a-z0-9]{1,5}$', path, re.IGNORECASE):
            return False

        return True

    def _create_parsed_file(
        self,
        path: str,
        content_lines: List[str],
        language: Optional[str],
        line_start: int,
        line_end: int
    ) -> ParsedFile:
        """Create ParsedFile object with content cleanup."""
        # Join content and clean up
        content = "\n".join(content_lines)

        # Remove excessive blank lines
        content = re.sub(r'\n{3,}', '\n\n', content)

        # Trim leading/trailing whitespace
        content = content.strip()

        # Calculate confidence based on content quality
        confidence = self._calculate_confidence(content, language)

        return ParsedFile(
            path=path,
            content=content,
            language=language,
            line_start=line_start,
            line_end=line_end,
            confidence=confidence
        )

    def _calculate_confidence(self, content: str, language: Optional[str]) -> float:
        """Calculate confidence score for parsed content."""
        if not content:
            return 0.0

        score = 1.0

        # Reduce score if no language detected
        if not language:
            score -= 0.2

        # Reduce score if content is very short
        if len(content) < 20:
            score -= 0.3

        # Reduce score if content has no code patterns
        if not any(self._looks_like_code(line) for line in content.split('\n')):
            score -= 0.3

        return max(0.0, score)

    def _add_warning(self, line_number: int, message: str, context: str):
        """Add a parsing warning."""
        self.warnings.append(ParseError(
            line_number=line_number,
            message=message,
            context=context,
            severity="warning"
        ))

    def _add_error(self, line_number: int, message: str, context: str):
        """Add a parsing error."""
        self.errors.append(ParseError(
            line_number=line_number,
            message=message,
            context=context,
            severity="error"
        ))

    def get_parse_report(self) -> str:
        """Get a human-readable parse report."""
        lines = []

        if self.errors:
            lines.append("=== ERRORS ===")
            for err in self.errors:
                lines.append(f"Line {err.line_number}: {err.message}")
                lines.append(f"  Context: {err.context}")

        if self.warnings:
            lines.append("\n=== WARNINGS ===")
            for warn in self.warnings:
                lines.append(f"Line {warn.line_number}: {warn.message}")

        return "\n".join(lines) if lines else "No issues found"


def extract_single_code_block(response: str, language: Optional[str] = None) -> str:
    """
    Extract a single code block from response (for simple cases).

    Args:
        response: LLM response
        language: Expected language (optional)

    Returns:
        Extracted code content
    """
    parser = EnhancedCodeParser()

    # Try to find fenced code block
    pattern = r'```(?:' + (language or r'\w*') + r')?\n(.*?)```'
    match = re.search(pattern, response, re.DOTALL)

    if match:
        return match.group(1).strip()

    # Try indented code block
    lines = response.split('\n')
    code_lines = []
    in_code = False

    for line in lines:
        if line.startswith('    ') or line.startswith('\t'):
            code_lines.append(line[4:] if line.startswith('    ') else line[1:])
            in_code = True
        elif in_code and not line.strip():
            code_lines.append('')
        elif in_code:
            break

    if code_lines:
        return '\n'.join(code_lines).strip()

    # No code block found, return as-is
    return response.strip()


# Backward compatibility function
def parse_code_files(response: str, strict: bool = False) -> Dict[str, str]:
    """
    Parse code files from LLM response (backward compatible).

    Args:
        response: LLM response containing files
        strict: If True, fail on parsing errors

    Returns:
        Dictionary mapping file paths to content strings
    """
    parser = EnhancedCodeParser(strict_mode=strict)
    parsed_files = parser.parse(response)

    # Convert ParsedFile objects to simple strings
    return {path: pf.content for path, pf in parsed_files.items()}
