"""
Memory Usage Monitoring and Warnings for Company AGI.

Provides memory management:
- System memory monitoring
- Context memory tracking per agent
- Warning thresholds with callbacks
- Auto-cleanup triggers
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class MemoryLevel(Enum):
    """Memory usage levels (numeric for comparison)."""
    NORMAL = 0          # < 60%
    WARNING = 1         # 60-75%
    HIGH = 2            # 75-85%
    CRITICAL = 3        # 85-95%
    EMERGENCY = 4       # > 95%


@dataclass
class MemoryThresholds:
    """Memory threshold configuration."""
    warning_pct: float = 60.0
    high_pct: float = 75.0
    critical_pct: float = 85.0
    emergency_pct: float = 95.0
    auto_compact_pct: float = 80.0
    max_context_tokens: int = 8192

    def get_level(self, usage_pct: float) -> MemoryLevel:
        """Get memory level for usage percentage."""
        if usage_pct >= self.emergency_pct:
            return MemoryLevel.EMERGENCY
        if usage_pct >= self.critical_pct:
            return MemoryLevel.CRITICAL
        if usage_pct >= self.high_pct:
            return MemoryLevel.HIGH
        if usage_pct >= self.warning_pct:
            return MemoryLevel.WARNING
        return MemoryLevel.NORMAL


@dataclass
class SystemMemoryInfo:
    """System memory information."""
    total_bytes: int = 0
    available_bytes: int = 0
    used_bytes: int = 0
    used_pct: float = 0.0
    level: MemoryLevel = MemoryLevel.NORMAL
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_gb": self.total_bytes / (1024**3),
            "available_gb": self.available_bytes / (1024**3),
            "used_gb": self.used_bytes / (1024**3),
            "used_pct": self.used_pct,
            "level": self.level.value,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentMemoryInfo:
    """Memory info for a single agent."""
    agent_name: str
    current_tokens: int = 0
    max_tokens: int = 8192
    message_count: int = 0
    experience_count: int = 0
    pattern_count: int = 0
    last_compaction: Optional[str] = None

    @property
    def usage_pct(self) -> float:
        """Calculate usage percentage."""
        if self.max_tokens == 0:
            return 0.0
        return (self.current_tokens / self.max_tokens) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "current_tokens": self.current_tokens,
            "max_tokens": self.max_tokens,
            "usage_pct": self.usage_pct,
            "message_count": self.message_count,
            "experience_count": self.experience_count,
            "pattern_count": self.pattern_count,
            "last_compaction": self.last_compaction,
        }


@dataclass
class MemoryWarning:
    """A memory warning."""
    timestamp: str
    level: MemoryLevel
    message: str
    agent: Optional[str] = None
    usage_pct: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "level": self.level.value,
            "message": self.message,
            "agent": self.agent,
            "usage_pct": self.usage_pct,
            "details": self.details,
        }


# Type for warning callbacks
WarningCallback = Callable[[MemoryWarning], None]
CompactCallback = Callable[[str], None]  # agent_name


class MemoryMonitor:
    """
    Memory monitoring and warning system.

    Features:
    - System memory monitoring
    - Per-agent context tracking
    - Configurable thresholds
    - Warning callbacks
    - Auto-compact triggers
    """

    def __init__(
        self,
        thresholds: Optional[MemoryThresholds] = None,
        check_interval_seconds: float = 30.0,
        on_warning: Optional[WarningCallback] = None,
        on_compact_needed: Optional[CompactCallback] = None,
    ):
        self.thresholds = thresholds or MemoryThresholds()
        self.check_interval_seconds = check_interval_seconds
        self.on_warning = on_warning
        self.on_compact_needed = on_compact_needed

        self._agent_memory: Dict[str, AgentMemoryInfo] = {}
        self._warnings: List[MemoryWarning] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.disabled = False  # When True, skip background thread + psutil

        # Track last warning level per agent to avoid spam
        self._last_warned: Dict[str, MemoryLevel] = {}

    def _get_system_memory(self) -> SystemMemoryInfo:
        """Get current system memory info."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            used_pct = mem.percent
            level = self.thresholds.get_level(used_pct)

            return SystemMemoryInfo(
                total_bytes=mem.total,
                available_bytes=mem.available,
                used_bytes=mem.used,
                used_pct=used_pct,
                level=level,
            )
        except ImportError:
            # psutil not available - return unknown
            return SystemMemoryInfo(level=MemoryLevel.NORMAL)

    def register_agent(
        self,
        agent_name: str,
        max_tokens: int = 8192,
    ) -> None:
        """Register an agent for memory tracking."""
        with self._lock:
            if agent_name not in self._agent_memory:
                self._agent_memory[agent_name] = AgentMemoryInfo(
                    agent_name=agent_name,
                    max_tokens=max_tokens,
                )

    def update_agent_memory(
        self,
        agent_name: str,
        current_tokens: int,
        message_count: Optional[int] = None,
        experience_count: Optional[int] = None,
        pattern_count: Optional[int] = None,
    ) -> Optional[MemoryWarning]:
        """Update agent memory usage and check thresholds."""
        if self.disabled:
            return None
        with self._lock:
            if agent_name not in self._agent_memory:
                self.register_agent(agent_name)

            info = self._agent_memory[agent_name]
            info.current_tokens = current_tokens

            if message_count is not None:
                info.message_count = message_count
            if experience_count is not None:
                info.experience_count = experience_count
            if pattern_count is not None:
                info.pattern_count = pattern_count

            usage_pct = info.usage_pct
            level = self.thresholds.get_level(usage_pct)

            # Check if we need to warn
            warning = None
            last_level = self._last_warned.get(agent_name, MemoryLevel.NORMAL)

            if level.value > last_level.value:
                warning = self._create_warning(
                    level=level,
                    message=f"Agent {agent_name} memory at {usage_pct:.1f}%",
                    agent=agent_name,
                    usage_pct=usage_pct,
                    details=info.to_dict(),
                )
                self._last_warned[agent_name] = level

                # Trigger compact if needed
                if usage_pct >= self.thresholds.auto_compact_pct and self.on_compact_needed:
                    self.on_compact_needed(agent_name)

            elif level.value < last_level.value:
                # Level decreased - reset tracking
                self._last_warned[agent_name] = level

            return warning

    def record_compaction(self, agent_name: str) -> None:
        """Record that an agent performed compaction."""
        with self._lock:
            if agent_name in self._agent_memory:
                self._agent_memory[agent_name].last_compaction = datetime.now().isoformat()
                # Reset warning level after compaction
                self._last_warned[agent_name] = MemoryLevel.NORMAL

    def _create_warning(
        self,
        level: MemoryLevel,
        message: str,
        agent: Optional[str] = None,
        usage_pct: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ) -> MemoryWarning:
        """Create and record a warning."""
        warning = MemoryWarning(
            timestamp=datetime.now().isoformat(),
            level=level,
            message=message,
            agent=agent,
            usage_pct=usage_pct,
            details=details or {},
        )

        self._warnings.append(warning)

        # Keep only recent warnings
        if len(self._warnings) > 100:
            self._warnings = self._warnings[-100:]

        # Call callback
        if self.on_warning:
            self.on_warning(warning)

        return warning

    def check_system_memory(self) -> Optional[MemoryWarning]:
        """Check system memory and create warning if needed."""
        if self.disabled:
            return None
        info = self._get_system_memory()

        if info.level in [MemoryLevel.CRITICAL, MemoryLevel.EMERGENCY]:
            return self._create_warning(
                level=info.level,
                message=f"System memory {info.level.value}: {info.used_pct:.1f}% used",
                usage_pct=info.used_pct,
                details=info.to_dict(),
            )

        if info.level == MemoryLevel.HIGH:
            return self._create_warning(
                level=info.level,
                message=f"System memory high: {info.used_pct:.1f}% used",
                usage_pct=info.used_pct,
                details=info.to_dict(),
            )

        return None

    def start_monitoring(self) -> None:
        """Start background memory monitoring."""
        if self._running or self.disabled:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop_monitoring(self) -> None:
        """Stop background memory monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            # Check system memory
            self.check_system_memory()

            # Check all agents
            with self._lock:
                for agent_name, info in self._agent_memory.items():
                    usage_pct = info.usage_pct
                    level = self.thresholds.get_level(usage_pct)

                    if level.value >= MemoryLevel.HIGH.value:
                        last_level = self._last_warned.get(agent_name, MemoryLevel.NORMAL)
                        if level.value > last_level.value:
                            self._create_warning(
                                level=level,
                                message=f"Agent {agent_name} memory at {usage_pct:.1f}%",
                                agent=agent_name,
                                usage_pct=usage_pct,
                                details=info.to_dict(),
                            )
                            self._last_warned[agent_name] = level

                            # Trigger compact if needed
                            if usage_pct >= self.thresholds.auto_compact_pct and self.on_compact_needed:
                                self.on_compact_needed(agent_name)

            time.sleep(self.check_interval_seconds)

    def get_agent_memory(self, agent_name: str) -> Optional[AgentMemoryInfo]:
        """Get memory info for an agent."""
        return self._agent_memory.get(agent_name)

    def get_all_agent_memory(self) -> Dict[str, AgentMemoryInfo]:
        """Get memory info for all agents."""
        return dict(self._agent_memory)

    def get_system_memory(self) -> SystemMemoryInfo:
        """Get current system memory info."""
        return self._get_system_memory()

    def get_recent_warnings(self, count: int = 10) -> List[MemoryWarning]:
        """Get recent warnings."""
        return self._warnings[-count:]

    def get_dashboard(self, color: bool = True) -> str:
        """Get memory dashboard as formatted string."""
        colors = {
            "header": "\033[1;36m",
            "normal": "\033[32m",
            "warning": "\033[33m",
            "high": "\033[38;5;208m",  # Orange
            "critical": "\033[31m",
            "emergency": "\033[1;31m",
            "reset": "\033[0m",
            "bold": "\033[1m",
        }

        if not color:
            colors = {k: "" for k in colors}

        c = colors
        lines = [
            f"{c['bold']}Memory Dashboard{c['reset']}",
            "=" * 50,
        ]

        # System memory
        sys_mem = self._get_system_memory()
        level_color = colors.get(sys_mem.level.name.lower(), "")
        lines.extend([
            f"\n{c['header']}System Memory:{c['reset']}",
            f"  Total:     {sys_mem.total_bytes / (1024**3):.1f} GB",
            f"  Available: {sys_mem.available_bytes / (1024**3):.1f} GB",
            f"  Used:      {level_color}{sys_mem.used_pct:.1f}%{c['reset']} ({sys_mem.level.name.lower()})",
        ])

        # Agent memory
        if self._agent_memory:
            lines.append(f"\n{c['header']}Agent Context Memory:{c['reset']}")
            for name, info in sorted(self._agent_memory.items()):
                level = self.thresholds.get_level(info.usage_pct)
                level_color = colors.get(level.name.lower(), "")

                bar_width = 20
                filled = int((info.usage_pct / 100.0) * bar_width)
                bar = "=" * filled + " " * (bar_width - filled)

                lines.append(
                    f"  {name:15} [{level_color}{bar}{c['reset']}] "
                    f"{level_color}{info.usage_pct:5.1f}%{c['reset']} "
                    f"({info.current_tokens:,}/{info.max_tokens:,} tokens)"
                )

                if level.value >= MemoryLevel.HIGH.value:
                    if info.last_compaction:
                        lines.append(f"                  Last compact: {info.last_compaction}")
                    else:
                        lines.append(f"                  {c['warning']}Consider compacting{c['reset']}")

        # Recent warnings
        recent_warnings = self.get_recent_warnings(3)
        if recent_warnings:
            lines.append(f"\n{c['header']}Recent Warnings:{c['reset']}")
            for warning in recent_warnings:
                level_color = colors.get(warning.level.name.lower(), "")
                lines.append(
                    f"  [{warning.timestamp[:19]}] {level_color}{warning.level.name}{c['reset']}: {warning.message}"
                )

        return "\n".join(lines)


# Global memory monitor instance
_memory_monitor: Optional[MemoryMonitor] = None


def get_memory_monitor(
    auto_start: bool = False,
) -> MemoryMonitor:
    """Get or create the global memory monitor."""
    global _memory_monitor
    if _memory_monitor is None:
        _memory_monitor = MemoryMonitor()
        if auto_start:
            _memory_monitor.start_monitoring()
    return _memory_monitor


def reset_memory_monitor() -> None:
    """Reset the global memory monitor."""
    global _memory_monitor
    if _memory_monitor:
        _memory_monitor.stop_monitoring()
    _memory_monitor = None
