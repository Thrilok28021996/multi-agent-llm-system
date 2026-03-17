"""Enhanced Review System - Claude Code-style code review with inline suggestions."""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class IssueSeverity(Enum):
    """Issue severity levels."""
    CRITICAL = "critical"  # Security vulnerabilities, data loss risks
    HIGH = "high"          # Bugs that break functionality
    MEDIUM = "medium"      # Quality issues, performance problems
    LOW = "low"            # Style issues, minor improvements
    INFO = "info"          # Suggestions, best practices


class IssueCategory(Enum):
    """Issue categories."""
    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    STYLE = "style"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    ARCHITECTURE = "architecture"


@dataclass
class CodeSuggestion:
    """Represents an inline code suggestion (Claude Code style)."""
    file_path: str
    line_start: int
    line_end: int
    original_code: str
    suggested_code: str
    explanation: str
    severity: IssueSeverity
    category: IssueCategory

    def to_diff(self) -> str:
        """Generate a unified diff format."""
        lines = []
        lines.append(f"--- {self.file_path}:{self.line_start}")
        lines.append(f"+++ {self.file_path}:{self.line_start}")
        lines.append("@@ Changes @@")

        for line in self.original_code.split('\n'):
            lines.append(f"- {line}")

        for line in self.suggested_code.split('\n'):
            lines.append(f"+ {line}")

        return '\n'.join(lines)

    def to_claude_code_format(self) -> str:
        """Format as Claude Code-style suggestion."""
        severity_icons = {
            IssueSeverity.CRITICAL: "🚨",
            IssueSeverity.HIGH: "❗",
            IssueSeverity.MEDIUM: "⚠️",
            IssueSeverity.LOW: "💡",
            IssueSeverity.INFO: "ℹ️"
        }

        icon = severity_icons.get(self.severity, "•")

        return f"""
{icon} {self.severity.value.upper()} - {self.category.value} in {self.file_path}:{self.line_start}

**Issue:** {self.explanation}

**Current code:**
```
{self.original_code}
```

**Suggested change:**
```
{self.suggested_code}
```
"""


@dataclass
class ReviewContext:
    """Context for code review."""
    file_path: str
    code: str
    language: Optional[str] = None
    purpose: Optional[str] = None  # What this code is supposed to do
    requirements: List[str] = field(default_factory=list)  # Specific requirements to check
    related_files: List[str] = field(default_factory=list)  # Related code context
    user_query: Optional[str] = None  # Original user request

    # Parsed code structure
    functions: List[Dict[str, Any]] = field(default_factory=list)
    classes: List[Dict[str, Any]] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)

    def analyze_structure(self):
        """Parse code structure for better review."""
        if not self.language:
            self.language = self._detect_language()

        if self.language == 'python':
            self._parse_python_structure()
        elif self.language in ['javascript', 'typescript']:
            self._parse_js_structure()

    def _detect_language(self) -> str:
        """Detect language from file extension."""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.go': 'go',
            '.rs': 'rust',
            '.java': 'java',
        }

        for ext, lang in ext_map.items():
            if self.file_path.endswith(ext):
                return lang

        return 'unknown'

    def _parse_python_structure(self):
        """Parse Python code structure."""
        lines = self.code.split('\n')

        for i, line in enumerate(lines, start=1):
            # Find imports
            if re.match(r'^\s*(?:import|from)\s+', line):
                self.imports.append(line.strip())

            # Find function definitions
            func_match = re.match(r'^\s*def\s+(\w+)\s*\(([^)]*)\)', line)
            if func_match:
                self.functions.append({
                    'name': func_match.group(1),
                    'params': func_match.group(2),
                    'line': i
                })

            # Find class definitions
            class_match = re.match(r'^\s*class\s+(\w+)', line)
            if class_match:
                self.classes.append({
                    'name': class_match.group(1),
                    'line': i
                })

    def _parse_js_structure(self):
        """Parse JavaScript/TypeScript structure."""
        lines = self.code.split('\n')

        for i, line in enumerate(lines, start=1):
            # Find imports
            if re.match(r'^\s*(?:import|require|from)\s+', line):
                self.imports.append(line.strip())

            # Find function definitions
            func_match = re.match(r'^\s*(?:function|const|let|var)\s+(\w+)\s*[=\(]', line)
            if func_match:
                self.functions.append({
                    'name': func_match.group(1),
                    'line': i
                })

            # Find class definitions
            class_match = re.match(r'^\s*class\s+(\w+)', line)
            if class_match:
                self.classes.append({
                    'name': class_match.group(1),
                    'line': i
                })


@dataclass
class ReviewResult:
    """Result of a code review."""
    context: ReviewContext
    suggestions: List[CodeSuggestion] = field(default_factory=list)
    overall_assessment: str = ""
    meets_requirements: bool = True
    critical_issues_count: int = 0
    high_issues_count: int = 0
    medium_issues_count: int = 0
    low_issues_count: int = 0

    def add_suggestion(self, suggestion: CodeSuggestion):
        """Add a suggestion and update counters."""
        self.suggestions.append(suggestion)

        if suggestion.severity == IssueSeverity.CRITICAL:
            self.critical_issues_count += 1
            self.meets_requirements = False
        elif suggestion.severity == IssueSeverity.HIGH:
            self.high_issues_count += 1
        elif suggestion.severity == IssueSeverity.MEDIUM:
            self.medium_issues_count += 1
        elif suggestion.severity == IssueSeverity.LOW:
            self.low_issues_count += 1

    def get_summary(self) -> str:
        """Get review summary."""
        total = len(self.suggestions)

        if total == 0:
            return "✅ No issues found. Code looks good!"

        summary_lines = [
            f"📊 Review Summary for {self.context.file_path}",
            f"{'='*60}",
            f"Total Issues: {total}",
        ]

        if self.critical_issues_count > 0:
            summary_lines.append(f"  🚨 Critical: {self.critical_issues_count}")
        if self.high_issues_count > 0:
            summary_lines.append(f"  ❗ High: {self.high_issues_count}")
        if self.medium_issues_count > 0:
            summary_lines.append(f"  ⚠️  Medium: {self.medium_issues_count}")
        if self.low_issues_count > 0:
            summary_lines.append(f"  💡 Low: {self.low_issues_count}")

        summary_lines.append("")

        if not self.meets_requirements:
            summary_lines.append("❌ Critical issues found - code needs fixes before deployment")
        elif self.high_issues_count > 0:
            summary_lines.append("⚠️  High priority issues - recommend fixing before merge")
        else:
            summary_lines.append("✅ No critical issues - minor improvements suggested")

        return '\n'.join(summary_lines)

    def to_claude_code_format(self) -> str:
        """Format full review in Claude Code style."""
        output = [self.get_summary(), ""]

        if self.overall_assessment:
            output.append("## Overall Assessment")
            output.append(self.overall_assessment)
            output.append("")

        if self.suggestions:
            output.append("## Detailed Suggestions")
            output.append("")

            # Group by severity
            by_severity = {}
            for sev in IssueSeverity:
                by_severity[sev] = [s for s in self.suggestions if s.severity == sev]

            for severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH,
                           IssueSeverity.MEDIUM, IssueSeverity.LOW, IssueSeverity.INFO]:
                issues = by_severity.get(severity, [])
                if issues:
                    output.append(f"### {severity.value.upper()} Priority")
                    output.append("")
                    for suggestion in issues:
                        output.append(suggestion.to_claude_code_format())
                        output.append("")

        return '\n'.join(output)


class EnhancedReviewSystem:
    """
    Enhanced review system with context-aware analysis.

    Properly reviews queries and context before validation.
    """

    def __init__(self):
        self.review_cache: Dict[str, ReviewResult] = {}

    def create_context(
        self,
        file_path: str,
        code: str,
        user_query: Optional[str] = None,
        purpose: Optional[str] = None,
        requirements: Optional[List[str]] = None,
        related_files: Optional[List[str]] = None
    ) -> ReviewContext:
        """
        Create review context with proper understanding of the task.

        This is the first step - understand what we're reviewing and why.
        """
        context = ReviewContext(
            file_path=file_path,
            code=code,
            user_query=user_query,
            purpose=purpose,
            requirements=requirements or [],
            related_files=related_files or []
        )

        # Analyze code structure
        context.analyze_structure()

        return context

    def review_with_context(
        self,
        context: ReviewContext,
        focus_areas: Optional[List[str]] = None
    ) -> ReviewResult:
        """
        Perform context-aware code review.

        Reviews the query and purpose first, then validates against requirements.
        """
        result = ReviewResult(context=context)

        # Step 1: Understand the query and context
        self._analyze_query_context(context, result)

        # Step 2: Check against requirements
        if context.requirements:
            self._validate_requirements(context, result)

        # Step 3: Standard code review
        if not focus_areas:
            focus_areas = ['bugs', 'security', 'quality', 'performance']

        if 'bugs' in focus_areas:
            self._check_bugs(context, result)

        if 'security' in focus_areas:
            self._check_security(context, result)

        if 'quality' in focus_areas:
            self._check_quality(context, result)

        if 'performance' in focus_areas:
            self._check_performance(context, result)

        # Step 4: Generate overall assessment
        result.overall_assessment = self._generate_assessment(context, result)

        return result

    def _analyze_query_context(self, context: ReviewContext, result: ReviewResult):
        """Analyze the user query and purpose to understand intent."""
        if not context.user_query and not context.purpose:
            return

        # Check if code structure matches the stated purpose
        if context.purpose:
            # Example: If purpose says "authentication", check for auth-related code
            purpose_lower = context.purpose.lower()
            code_lower = context.code.lower()

            # Basic keyword matching
            if 'auth' in purpose_lower:
                if 'password' not in code_lower and 'token' not in code_lower:
                    result.add_suggestion(CodeSuggestion(
                        file_path=context.file_path,
                        line_start=1,
                        line_end=1,
                        original_code="# Missing authentication implementation",
                        suggested_code="# Add authentication: passwords, tokens, or OAuth",
                        explanation="Code purpose states authentication but no auth-related code found",
                        severity=IssueSeverity.HIGH,
                        category=IssueCategory.BUG
                    ))

            if 'database' in purpose_lower or 'sql' in purpose_lower:
                if 'sql' not in code_lower and 'query' not in code_lower:
                    result.add_suggestion(CodeSuggestion(
                        file_path=context.file_path,
                        line_start=1,
                        line_end=1,
                        original_code="# Missing database operations",
                        suggested_code="# Add database queries and connection handling",
                        explanation="Code purpose mentions database but no DB operations found",
                        severity=IssueSeverity.MEDIUM,
                        category=IssueCategory.BUG
                    ))

    def _validate_requirements(self, context: ReviewContext, result: ReviewResult):
        """Validate code against specific requirements."""
        for req in context.requirements:
            req_lower = req.lower()
            code_lower = context.code.lower()

            # Check if requirement keywords appear in code
            req_keywords = re.findall(r'\w+', req_lower)
            important_keywords = [k for k in req_keywords if len(k) > 4]

            found_keywords = sum(1 for k in important_keywords if k in code_lower)

            if len(important_keywords) > 0 and found_keywords == 0:
                result.add_suggestion(CodeSuggestion(
                    file_path=context.file_path,
                    line_start=1,
                    line_end=1,
                    original_code="# Requirement not met",
                    suggested_code=f"# Implement: {req}",
                    explanation=f"Requirement appears unmet: {req}",
                    severity=IssueSeverity.HIGH,
                    category=IssueCategory.BUG
                ))

    def _check_bugs(self, context: ReviewContext, result: ReviewResult):
        """Check for common bugs."""
        lines = context.code.split('\n')

        for i, line in enumerate(lines, start=1):
            # Check for common Python bugs
            if context.language == 'python':
                # Mutable default arguments
                if re.search(r'def\s+\w+\([^)]*=\s*\[\]', line) or \
                   re.search(r'def\s+\w+\([^)]*=\s*\{\}', line):
                    result.add_suggestion(CodeSuggestion(
                        file_path=context.file_path,
                        line_start=i,
                        line_end=i,
                        original_code=line.strip(),
                        suggested_code=line.replace('=[]', '=None').replace('={}', '=None'),
                        explanation="Mutable default arguments can cause bugs. Use None and create inside function.",
                        severity=IssueSeverity.HIGH,
                        category=IssueCategory.BUG
                    ))

                # Bare except
                if re.match(r'^\s*except\s*:\s*$', line):
                    result.add_suggestion(CodeSuggestion(
                        file_path=context.file_path,
                        line_start=i,
                        line_end=i,
                        original_code=line.strip(),
                        suggested_code="except Exception as e:",
                        explanation="Bare except catches all exceptions including system exits. Be specific.",
                        severity=IssueSeverity.MEDIUM,
                        category=IssueCategory.BUG
                    ))

    def _check_security(self, context: ReviewContext, result: ReviewResult):
        """Check for security issues."""
        lines = context.code.split('\n')

        for i, line in enumerate(lines, start=1):
            # SQL injection
            if 'execute' in line.lower() and ('+' in line or '{' in line or '%' in line):
                if 'cursor' in line.lower() or 'sql' in line.lower():
                    result.add_suggestion(CodeSuggestion(
                        file_path=context.file_path,
                        line_start=i,
                        line_end=i,
                        original_code=line.strip(),
                        suggested_code="# Use parameterized queries: cursor.execute(sql, (param1, param2))",
                        explanation="Potential SQL injection - use parameterized queries instead of string formatting",
                        severity=IssueSeverity.CRITICAL,
                        category=IssueCategory.SECURITY
                    ))

            # Hardcoded secrets
            secret_patterns = [
                (r'password\s*=\s*["\'][\w]+["\']', "password"),
                (r'api[_-]?key\s*=\s*["\'][\w]+["\']', "API key"),
                (r'secret\s*=\s*["\'][\w]+["\']', "secret"),
                (r'token\s*=\s*["\'][\w]{20,}["\']', "token"),
            ]

            for pattern, secret_type in secret_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    result.add_suggestion(CodeSuggestion(
                        file_path=context.file_path,
                        line_start=i,
                        line_end=i,
                        original_code=line.strip(),
                        suggested_code=f"# Load from environment: {secret_type} = os.getenv('{secret_type.upper()}')",
                        explanation=f"Hardcoded {secret_type} - use environment variables instead",
                        severity=IssueSeverity.CRITICAL,
                        category=IssueCategory.SECURITY
                    ))

    def _check_quality(self, context: ReviewContext, result: ReviewResult):
        """Check code quality."""
        lines = context.code.split('\n')

        for i, line in enumerate(lines, start=1):
            # Long lines
            if len(line) > 120:
                result.add_suggestion(CodeSuggestion(
                    file_path=context.file_path,
                    line_start=i,
                    line_end=i,
                    original_code=line,
                    suggested_code="# Break into multiple lines",
                    explanation="Line exceeds 120 characters - consider breaking it up for readability",
                    severity=IssueSeverity.LOW,
                    category=IssueCategory.STYLE
                ))

            # Print statements in production code
            if re.match(r'^\s*print\s*\(', line):
                result.add_suggestion(CodeSuggestion(
                    file_path=context.file_path,
                    line_start=i,
                    line_end=i,
                    original_code=line.strip(),
                    suggested_code="# Use logging: logger.info(...) or logger.debug(...)",
                    explanation="Use logging instead of print for production code",
                    severity=IssueSeverity.LOW,
                    category=IssueCategory.QUALITY
                ))

    def _check_performance(self, context: ReviewContext, result: ReviewResult):
        """Check for performance issues."""
        code = context.code

        # Nested loops with large data
        if code.count('for ') >= 3:
            result.add_suggestion(CodeSuggestion(
                file_path=context.file_path,
                line_start=1,
                line_end=1,
                original_code="# Multiple nested loops detected",
                suggested_code="# Consider optimization: use sets, dictionaries, or vectorization",
                explanation="Multiple nested loops detected - may have O(n²) or O(n³) complexity",
                severity=IssueSeverity.MEDIUM,
                category=IssueCategory.PERFORMANCE
            ))

    def _generate_assessment(self, context: ReviewContext, result: ReviewResult) -> str:
        """Generate overall assessment."""
        lines = []

        if context.user_query:
            lines.append(f"**Original Request:** {context.user_query}")
            lines.append("")

        if context.purpose:
            lines.append(f"**Code Purpose:** {context.purpose}")
            lines.append("")

        lines.append(f"**Language:** {context.language or 'unknown'}")
        lines.append(f"**File:** {context.file_path}")
        lines.append(f"**Lines of Code:** {len(context.code.split(chr(10)))}")
        lines.append("")

        if context.functions:
            lines.append(f"**Functions Found:** {', '.join(f['name'] for f in context.functions[:5])}")

        if context.classes:
            lines.append(f"**Classes Found:** {', '.join(c['name'] for c in context.classes[:5])}")

        lines.append("")

        if result.meets_requirements:
            lines.append("✅ Code structure appears to meet stated requirements")
        else:
            lines.append("❌ Code may not fully meet requirements - critical issues found")

        return '\n'.join(lines)


# Convenience function
def quick_review(
    file_path: str,
    code: str,
    user_query: Optional[str] = None,
    requirements: Optional[List[str]] = None
) -> str:
    """
    Quick review with Claude Code-style output.

    Args:
        file_path: Path to the file being reviewed
        code: Code content to review
        user_query: Original user request (for context)
        requirements: List of requirements to validate

    Returns:
        Formatted review report
    """
    system = EnhancedReviewSystem()

    context = system.create_context(
        file_path=file_path,
        code=code,
        user_query=user_query,
        requirements=requirements
    )

    result = system.review_with_context(context)

    return result.to_claude_code_format()
