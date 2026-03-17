"""
Streaming Output for Company AGI.

Provides Claude Code-style streaming with:
- Real-time response display
- Partial response handling
- Progress indicators
- Async streaming support
"""

import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional


class StreamEventType(Enum):
    """Types of streaming events."""
    START = "start"
    TOKEN = "token"
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_END = "tool_use_end"
    THINKING_START = "thinking_start"
    THINKING_END = "thinking_end"
    ERROR = "error"
    END = "end"


@dataclass
class StreamEvent:
    """A streaming event."""
    event_type: StreamEventType
    data: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamStats:
    """Statistics for a streaming session."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    tokens_streamed: int = 0
    chunks_received: int = 0
    tool_calls: int = 0
    errors: int = 0

    @property
    def duration_seconds(self) -> float:
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    @property
    def tokens_per_second(self) -> float:
        duration = self.duration_seconds
        if duration > 0:
            return self.tokens_streamed / duration
        return 0.0


class StreamBuffer:
    """Buffer for accumulating streamed content."""

    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.content: str = ""
        self.chunks: List[str] = []

    def append(self, chunk: str) -> None:
        """Append a chunk to the buffer."""
        self.chunks.append(chunk)
        self.content += chunk
        if len(self.content) > self.max_size:
            # Trim from the beginning
            self.content = self.content[-self.max_size:]

    def get_content(self) -> str:
        """Get accumulated content."""
        return self.content

    def get_last_n_chars(self, n: int) -> str:
        """Get last n characters."""
        return self.content[-n:] if len(self.content) >= n else self.content

    def clear(self) -> None:
        """Clear the buffer."""
        self.content = ""
        self.chunks = []


class ProgressIndicator:
    """Animated progress indicator for streaming."""

    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    DOTS_FRAMES = [".", "..", "...", ""]

    def __init__(
        self,
        style: str = "spinner",
        message: str = "Thinking",
        output: Any = None,
    ):
        self.style = style
        self.message = message
        self.output = output or sys.stdout
        self.frame_index = 0
        self.is_running = False
        self._task: Optional[asyncio.Task[None]] = None

    def _get_frame(self) -> str:
        """Get current animation frame."""
        if self.style == "spinner":
            frames = self.SPINNER_FRAMES
        else:
            frames = self.DOTS_FRAMES
        return frames[self.frame_index % len(frames)]

    async def _animate(self) -> None:
        """Animation loop."""
        while self.is_running:
            frame = self._get_frame()
            self.output.write(f"\r{frame} {self.message}")
            self.output.flush()
            self.frame_index += 1
            await asyncio.sleep(0.1)

    def start(self) -> None:
        """Start the progress indicator."""
        self.is_running = True
        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._animate())
        except RuntimeError:
            # No running loop, use sync version
            pass

    def stop(self, clear: bool = True) -> None:
        """Stop the progress indicator."""
        self.is_running = False
        if self._task:
            self._task.cancel()
        if clear:
            # Clear the line
            self.output.write("\r" + " " * (len(self.message) + 5) + "\r")
            self.output.flush()

    def update_message(self, message: str) -> None:
        """Update the progress message."""
        self.message = message


class StreamingOutput:
    """
    Handles streaming output display.

    Features:
    - Real-time token display
    - Tool use indicators
    - Thinking mode display
    - Progress tracking
    """

    def __init__(
        self,
        output: Any = None,
        show_progress: bool = True,
        show_stats: bool = True,
        color_enabled: bool = True,
        buffer_size: int = 100000,
    ):
        self.output = output or sys.stdout
        self.show_progress = show_progress
        self.show_stats = show_stats
        self.color_enabled = color_enabled

        self.buffer = StreamBuffer(max_size=buffer_size)
        self.stats = StreamStats()
        self.progress = ProgressIndicator(output=self.output)

        self._callbacks: List[Callable[[StreamEvent], None]] = []
        self._is_streaming = False
        self._current_tool: Optional[str] = None
        self._in_thinking = False

    def _colorize(self, text: str, color: str) -> str:
        """Add color to text if enabled."""
        if not self.color_enabled:
            return text

        colors = {
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "magenta": "\033[95m",
            "cyan": "\033[96m",
            "white": "\033[97m",
            "reset": "\033[0m",
            "dim": "\033[2m",
            "bold": "\033[1m",
        }
        color_code = colors.get(color, "")
        reset = colors["reset"]
        return f"{color_code}{text}{reset}"

    def add_callback(self, callback: Callable[[StreamEvent], None]) -> None:
        """Add a callback for stream events."""
        self._callbacks.append(callback)

    def _emit_event(self, event: StreamEvent) -> None:
        """Emit an event to all callbacks."""
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception:
                pass

    async def stream_tokens(
        self,
        token_iterator: AsyncIterator[str],
        prefix: str = "",
    ) -> str:
        """Stream tokens from an async iterator."""
        self._is_streaming = True
        self.stats = StreamStats()
        self.buffer.clear()

        self._emit_event(StreamEvent(event_type=StreamEventType.START))

        if prefix:
            self.output.write(prefix)
            self.output.flush()

        try:
            async for token in token_iterator:
                self.buffer.append(token)
                self.stats.tokens_streamed += 1
                self.stats.chunks_received += 1

                self.output.write(token)
                self.output.flush()

                self._emit_event(StreamEvent(
                    event_type=StreamEventType.TOKEN,
                    data=token,
                ))

        except Exception as e:
            self.stats.errors += 1
            self._emit_event(StreamEvent(
                event_type=StreamEventType.ERROR,
                data=str(e),
            ))
            raise
        finally:
            self._is_streaming = False
            self.stats.end_time = datetime.now()
            self._emit_event(StreamEvent(event_type=StreamEventType.END))

        self.output.write("\n")
        self.output.flush()

        if self.show_stats:
            self._show_stats()

        return self.buffer.get_content()

    def stream_chunk(self, chunk: str) -> None:
        """Stream a single chunk (sync version)."""
        self.buffer.append(chunk)
        self.stats.tokens_streamed += len(chunk.split())
        self.stats.chunks_received += 1

        self.output.write(chunk)
        self.output.flush()

        self._emit_event(StreamEvent(
            event_type=StreamEventType.TOKEN,
            data=chunk,
        ))

    def start_tool_use(self, tool_name: str, tool_input: Optional[Dict] = None) -> None:
        """Indicate start of tool use."""
        self._current_tool = tool_name
        self.stats.tool_calls += 1

        # Show tool indicator
        indicator = self._colorize(f"\n⚡ Using {tool_name}", "cyan")
        if tool_input:
            # Show abbreviated input
            input_str = str(tool_input)
            indicator += self._colorize(f" {input_str}", "dim")
        indicator += "\n"

        self.output.write(indicator)
        self.output.flush()

        self._emit_event(StreamEvent(
            event_type=StreamEventType.TOOL_USE_START,
            data={"tool": tool_name, "input": tool_input},
        ))

    def end_tool_use(self, result: Optional[str] = None, success: bool = True) -> None:
        """Indicate end of tool use."""
        icon = "✓" if success else "✗"
        color = "green" if success else "red"

        indicator = self._colorize(f"  {icon} ", color)
        if result:
            indicator += self._colorize(result, "dim")
        indicator += "\n"

        self.output.write(indicator)
        self.output.flush()

        self._emit_event(StreamEvent(
            event_type=StreamEventType.TOOL_USE_END,
            data={"tool": self._current_tool, "result": result, "success": success},
        ))

        self._current_tool = None

    def start_thinking(self, message: str = "Thinking") -> None:
        """Start thinking indicator."""
        self._in_thinking = True

        indicator = self._colorize(f"\n💭 {message}...\n", "magenta")
        self.output.write(indicator)
        self.output.flush()

        self._emit_event(StreamEvent(
            event_type=StreamEventType.THINKING_START,
            data=message,
        ))

    def end_thinking(self) -> None:
        """End thinking indicator."""
        self._in_thinking = False

        self._emit_event(StreamEvent(
            event_type=StreamEventType.THINKING_END,
        ))

    def show_thinking_content(self, content: str) -> None:
        """Show thinking content (dimmed)."""
        if self._in_thinking:
            dimmed = self._colorize(content, "dim")
            self.output.write(dimmed)
            self.output.flush()

    def _show_stats(self) -> None:
        """Show streaming statistics."""
        stats_line = self._colorize(
            f"\n[{self.stats.tokens_streamed} tokens, "
            f"{self.stats.duration_seconds:.1f}s, "
            f"{self.stats.tokens_per_second:.1f} tok/s]",
            "dim"
        )
        self.output.write(stats_line + "\n")
        self.output.flush()

    def newline(self) -> None:
        """Output a newline."""
        self.output.write("\n")
        self.output.flush()

    def write(self, text: str) -> None:
        """Write text directly."""
        self.output.write(text)
        self.output.flush()

    def write_error(self, error: str) -> None:
        """Write an error message."""
        error_text = self._colorize(f"Error: {error}", "red")
        self.output.write(error_text + "\n")
        self.output.flush()

    def write_success(self, message: str) -> None:
        """Write a success message."""
        success_text = self._colorize(f"✓ {message}", "green")
        self.output.write(success_text + "\n")
        self.output.flush()

    def write_warning(self, message: str) -> None:
        """Write a warning message."""
        warning_text = self._colorize(f"⚠ {message}", "yellow")
        self.output.write(warning_text + "\n")
        self.output.flush()

    def write_info(self, message: str) -> None:
        """Write an info message."""
        info_text = self._colorize(f"ℹ {message}", "blue")
        self.output.write(info_text + "\n")
        self.output.flush()

    def get_stats(self) -> StreamStats:
        """Get streaming statistics."""
        return self.stats

    def clear_line(self) -> None:
        """Clear the current line."""
        self.output.write("\r" + " " * 80 + "\r")
        self.output.flush()


class MockTokenIterator:
    """Mock token iterator for testing."""

    def __init__(self, text: str, delay: float = 0.02):
        self.text = text
        self.delay = delay
        self.words = text.split()
        self.index = 0

    def __aiter__(self) -> "MockTokenIterator":
        return self

    async def __anext__(self) -> str:
        if self.index >= len(self.words):
            raise StopAsyncIteration

        word = self.words[self.index]
        self.index += 1

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        # Add space except for last word
        return word + (" " if self.index < len(self.words) else "")


# Singleton instance
_streaming_output: Optional[StreamingOutput] = None


def get_streaming_output(
    show_progress: bool = True,
    show_stats: bool = True,
    color_enabled: bool = True,
) -> StreamingOutput:
    """Get or create the global streaming output."""
    global _streaming_output
    if _streaming_output is None:
        _streaming_output = StreamingOutput(
            show_progress=show_progress,
            show_stats=show_stats,
            color_enabled=color_enabled,
        )
    return _streaming_output


def reset_streaming_output() -> None:
    """Reset the global streaming output."""
    global _streaming_output
    _streaming_output = None
