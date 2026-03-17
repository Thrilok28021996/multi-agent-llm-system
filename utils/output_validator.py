"""
Structured output validator and enforcer.
Validates LLM outputs against expected formats and re-prompts if invalid.
Works with: JSON schemas, key-value formats, verdict strings, code blocks.
"""
import json
import re
from typing import Any, Tuple


class OutputValidator:
    """
    Validates LLM output against a schema/format constraint.
    If invalid, generates a correction prompt.
    """

    def validate_json(self, text: str, required_keys: list = None) -> Tuple[bool, Any, str]:
        """
        Extract and validate JSON from LLM output.
        Returns (is_valid, parsed_data, error_message)
        """
        # Try to extract JSON block
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try bare JSON
            json_match = re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', text)
            json_str = json_match.group() if json_match else text.strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON: {e}"

        if required_keys:
            missing = [k for k in required_keys if k not in data]
            if missing:
                return False, data, f"Missing required keys: {missing}"

        return True, data, ""

    def validate_verdict(self, text: str, valid_verdicts: list) -> Tuple[bool, str, str]:
        """
        Extract verdict from LLM text.
        Returns (is_valid, verdict, error)
        """
        text_upper = text.upper()
        for verdict in valid_verdicts:
            if verdict.upper() in text_upper:
                return True, verdict, ""
        return False, "", f"No valid verdict found. Expected one of: {valid_verdicts}"

    def validate_code_block(self, text: str, language: str = None) -> Tuple[bool, str, str]:
        """Extract code from markdown code blocks."""
        pattern = rf'```{language or ""}\s*([\s\S]*?)\s*```' if language else r'```(?:\w+)?\s*([\s\S]*?)\s*```'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return True, match.group(1), ""
        return False, "", "No code block found in response"

    def get_correction_prompt(self, original_prompt: str, response: str, error: str, format_hint: str = "") -> str:
        """Generate a correction prompt when output format is invalid."""
        return (
            f"Your previous response had a format error: {error}\n\n"
            f"Original task: {original_prompt[:300]}\n\n"
            f"Your response was:\n{response[:500]}\n\n"
            f"Please provide a corrected response{f' in the format: {format_hint}' if format_hint else ''}.\n"
            "Ensure your response strictly follows the required format."
        )
