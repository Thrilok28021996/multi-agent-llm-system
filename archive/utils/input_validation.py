"""Input validation and sanitization utilities for LLM prompts."""

import re


class InputValidationError(Exception):
    """Exception raised for input validation errors."""
    pass


# Maximum lengths for different input types
MAX_PROBLEM_DESCRIPTION_LENGTH = 10000
MAX_PROMPT_LENGTH = 50000
MAX_CODE_LENGTH = 100000

# Patterns that might indicate prompt injection attempts
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above)\s+(instructions?|prompts?)",
    r"disregard\s+(all\s+)?(previous|above)",
    r"forget\s+(everything|all)",
    r"you\s+are\s+now\s+a",
    r"new\s+instructions?:",
    r"system\s*:\s*",
    r"\[INST\]",
    r"\[/INST\]",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"<\|system\|>",
    r"<\|user\|>",
    r"<\|assistant\|>",
]


def sanitize_prompt_input(
    text: str,
    max_length: int = MAX_PROMPT_LENGTH,
    allow_code_blocks: bool = True,
    check_injection: bool = True
) -> str:
    """
    Sanitize user input before including in LLM prompts.

    Args:
        text: The input text to sanitize
        max_length: Maximum allowed length
        allow_code_blocks: Whether to allow code blocks in input
        check_injection: Whether to check for prompt injection patterns

    Returns:
        Sanitized text

    Raises:
        InputValidationError: If input contains potential injection attempts
    """
    if not text:
        return ""

    # Remove null bytes
    text = text.replace('\x00', '')

    # Normalize unicode
    text = text.encode('utf-8', errors='ignore').decode('utf-8')

    # Truncate to max length
    if len(text) > max_length:
        text = text[:max_length] + "... [truncated]"

    # Check for potential prompt injection
    if check_injection:
        text_lower = text.lower()
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                # Don't raise an error, just flag and sanitize
                text = f"[User input - sanitized]: {text}"
                break

    # Escape special markdown/formatting that could confuse the model
    if not allow_code_blocks:
        # Escape code block markers
        text = text.replace('```', '\\`\\`\\`')

    return text


def validate_problem_description(description: str) -> str:
    """
    Validate and sanitize a problem description.

    Args:
        description: The problem description to validate

    Returns:
        Validated and sanitized description

    Raises:
        InputValidationError: If description is invalid
    """
    if not description:
        raise InputValidationError("Problem description cannot be empty")

    if len(description) < 10:
        raise InputValidationError("Problem description is too short (min 10 characters)")

    description = sanitize_prompt_input(
        description,
        max_length=MAX_PROBLEM_DESCRIPTION_LENGTH,
        allow_code_blocks=True,
        check_injection=True
    )

    return description


def truncate_to_token_limit(
    text: str,
    max_tokens: int,
    chars_per_token: float = 4.0
) -> str:
    """
    Truncate text to fit within an estimated token limit.

    This is an approximation - actual token count depends on the tokenizer.
    On average, 1 token ≈ 4 characters for English text.

    Args:
        text: The text to truncate
        max_tokens: Maximum number of tokens
        chars_per_token: Estimated characters per token

    Returns:
        Truncated text
    """
    if not text:
        return ""

    max_chars = int(max_tokens * chars_per_token)

    if len(text) <= max_chars:
        return text

    # Truncate and try to end at a sentence or word boundary
    truncated = text[:max_chars]

    # Try to find a good truncation point
    # Look for sentence end
    last_sentence = max(
        truncated.rfind('. '),
        truncated.rfind('.\n'),
        truncated.rfind('? '),
        truncated.rfind('! ')
    )

    if last_sentence > max_chars * 0.8:  # Don't truncate too much
        truncated = truncated[:last_sentence + 1]
    else:
        # Fall back to word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.9:
            truncated = truncated[:last_space]

    return truncated + "... [truncated]"


def sanitize_code_input(code: str, language: str = "python") -> str:
    """
    Sanitize code input to prevent injection in code-related prompts.

    Args:
        code: The code to sanitize
        language: The programming language

    Returns:
        Sanitized code
    """
    if not code:
        return ""

    # Remove null bytes
    code = code.replace('\x00', '')

    # Truncate if too long
    if len(code) > MAX_CODE_LENGTH:
        code = code[:MAX_CODE_LENGTH] + "\n# ... [code truncated]"

    return code


def validate_file_path(path: str) -> str:
    """
    Validate a file path input.

    Args:
        path: The file path to validate

    Returns:
        Validated path

    Raises:
        InputValidationError: If path is invalid
    """
    if not path:
        raise InputValidationError("File path cannot be empty")

    # Remove null bytes
    path = path.replace('\x00', '')

    # Check for path traversal attempts
    if '..' in path:
        raise InputValidationError("Path traversal not allowed")

    # Check for suspicious patterns
    if path.startswith('/etc/') or path.startswith('/root/'):
        raise InputValidationError("Access to system directories not allowed")

    # Limit length
    if len(path) > 1000:
        raise InputValidationError("File path too long")

    return path


def estimate_token_count(text: str, chars_per_token: float = 4.0) -> int:
    """
    Estimate the token count for a piece of text.

    Args:
        text: The text to estimate
        chars_per_token: Estimated characters per token

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    return int(len(text) / chars_per_token)
