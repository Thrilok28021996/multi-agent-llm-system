"""Shared memory system for company-wide knowledge base."""

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ui.console import console
from utils.file_lock import atomic_write_json, safe_read_json


class MemoryType(Enum):
    """Types of memories that can be stored."""

    PROBLEM = "problem"  # Discovered problems
    SOLUTION = "solution"  # Implemented solutions
    DECISION = "decision"  # Company decisions
    INSIGHT = "insight"  # Research insights
    MEETING = "meeting"  # Meeting notes
    TASK = "task"  # Tasks and their status
    ARTIFACT = "artifact"  # Generated artifacts (code, docs)


@dataclass
class Memory:
    """A single memory item."""

    id: str
    type: MemoryType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    tags: List[str] = field(default_factory=list)
    importance: float = 0.5  # 0.0 to 1.0
    related_ids: List[str] = field(default_factory=list)


class SharedMemory:
    """
    Company-wide shared memory system.
    Acts as a knowledge base accessible by all agents.
    """

    def __init__(self, persist_dir: str = "./output/memory"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.memories: Dict[str, Memory] = {}
        self._load_from_disk()

    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for a memory."""
        timestamp = datetime.now().isoformat()
        hash_input = f"{content}{timestamp}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:12]

    def _load_from_disk(self) -> None:
        """Load memories from persistent storage with file locking."""
        memory_file = self.persist_dir / "shared_memory.json"
        if memory_file.exists():
            try:
                # Use safe_read_json with file locking to prevent race conditions
                data = safe_read_json(memory_file, default=[])
                if data:
                    for item in data:
                        item["type"] = MemoryType(item["type"])
                        item["created_at"] = datetime.fromisoformat(item["created_at"])
                        memory = Memory(**item)
                        self.memories[memory.id] = memory
            except Exception as e:
                console.warning(f"Could not load memory: {e}")

    def _save_to_disk(self) -> None:
        """Save memories to persistent storage with file locking and atomic writes."""
        memory_file = self.persist_dir / "shared_memory.json"
        data = []
        for memory in self.memories.values():
            item = asdict(memory)
            item["type"] = memory.type.value
            item["created_at"] = memory.created_at.isoformat()
            data.append(item)
        # Use atomic_write_json with file locking to prevent race conditions
        # and ensure data integrity even if process crashes during write
        atomic_write_json(memory_file, data, indent=2)

    # ============================================================
    #  STORE OPERATIONS
    # ============================================================

    def store(
        self,
        content: str,
        memory_type: MemoryType,
        created_by: str = "",
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        importance: float = 0.5,
        related_ids: List[str] = None
    ) -> Memory:
        """
        Store a new memory.

        Args:
            content: The content of the memory
            memory_type: Type of memory (problem, solution, decision, etc.)
            created_by: Agent that created this memory
            tags: Tags for categorization
            metadata: Additional metadata
            importance: Importance score (0.0 to 1.0)
            related_ids: IDs of related memories

        Returns:
            The created Memory object
        """
        memory_id = self._generate_id(content)
        memory = Memory(
            id=memory_id,
            type=memory_type,
            content=content,
            metadata=metadata or {},
            created_by=created_by,
            tags=tags or [],
            importance=importance,
            related_ids=related_ids or []
        )
        self.memories[memory_id] = memory
        self._save_to_disk()
        return memory

    def store_problem(
        self,
        description: str,
        source: str,
        severity: str = "medium",
        domain: str = "",
        created_by: str = "researcher"
    ) -> Memory:
        """Store a discovered problem."""
        return self.store(
            content=description,
            memory_type=MemoryType.PROBLEM,
            created_by=created_by,
            tags=[domain, severity] if domain else [severity],
            metadata={
                "source": source,
                "severity": severity,
                "domain": domain,
                "status": "new"
            },
            importance={"low": 0.3, "medium": 0.5, "high": 0.8, "critical": 1.0}.get(severity, 0.5)
        )

    def store_solution(
        self,
        problem_id: str,
        solution_description: str,
        implementation_path: str = "",
        created_by: str = "developer"
    ) -> Memory:
        """Store a solution for a problem."""
        return self.store(
            content=solution_description,
            memory_type=MemoryType.SOLUTION,
            created_by=created_by,
            metadata={
                "problem_id": problem_id,
                "implementation_path": implementation_path,
                "status": "implemented"
            },
            related_ids=[problem_id],
            importance=0.7
        )

    def store_decision(
        self,
        decision: str,
        reasoning: str,
        made_by: str,
        participants: List[str] = None
    ) -> Memory:
        """Store a company decision."""
        return self.store(
            content=decision,
            memory_type=MemoryType.DECISION,
            created_by=made_by,
            metadata={
                "reasoning": reasoning,
                "participants": participants or [made_by],
                "status": "approved"
            },
            importance=0.8
        )

    def store_meeting(
        self,
        topic: str,
        participants: List[str],
        summary: str,
        action_items: List[str] = None
    ) -> Memory:
        """Store meeting notes."""
        return self.store(
            content=summary,
            memory_type=MemoryType.MEETING,
            created_by="orchestrator",
            tags=["meeting"],
            metadata={
                "topic": topic,
                "participants": participants,
                "action_items": action_items or []
            },
            importance=0.6
        )

    # ============================================================
    #  RETRIEVE OPERATIONS
    # ============================================================

    def get(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID."""
        return self.memories.get(memory_id)

    def get_by_type(self, memory_type: MemoryType) -> List[Memory]:
        """Get all memories of a specific type."""
        return [m for m in self.memories.values() if m.type == memory_type]

    def get_problems(self, status: str = None) -> List[Memory]:
        """Get all problem memories, optionally filtered by status."""
        problems = self.get_by_type(MemoryType.PROBLEM)
        if status:
            problems = [p for p in problems if p.metadata.get("status") == status]
        return sorted(problems, key=lambda m: m.importance, reverse=True)

    def get_recent(self, limit: int = 10) -> List[Memory]:
        """Get the most recent memories."""
        sorted_memories = sorted(
            self.memories.values(),
            key=lambda m: m.created_at,
            reverse=True
        )
        return sorted_memories[:limit]

    def search(
        self,
        query: str,
        memory_type: MemoryType = None,
        tags: List[str] = None
    ) -> List[Memory]:
        """
        Search memories by content.

        Args:
            query: Search query
            memory_type: Filter by type
            tags: Filter by tags
        """
        results = []
        query_lower = query.lower()

        for memory in self.memories.values():
            # Type filter
            if memory_type and memory.type != memory_type:
                continue

            # Tag filter
            if tags and not any(t in memory.tags for t in tags):
                continue

            # Content search
            if query_lower in memory.content.lower():
                results.append(memory)

        return sorted(results, key=lambda m: m.importance, reverse=True)

    def get_related(self, memory_id: str) -> List[Memory]:
        """Get memories related to a specific memory."""
        memory = self.get(memory_id)
        if not memory:
            return []

        related = []
        for related_id in memory.related_ids:
            related_memory = self.get(related_id)
            if related_memory:
                related.append(related_memory)

        # Also find memories that reference this one
        for m in self.memories.values():
            if memory_id in m.related_ids and m.id not in [r.id for r in related]:
                related.append(m)

        return related

    # ============================================================
    #  UPDATE OPERATIONS
    # ============================================================

    def update_status(self, memory_id: str, status: str) -> bool:
        """Update the status of a memory."""
        memory = self.get(memory_id)
        if not memory:
            return False
        memory.metadata["status"] = status
        self._save_to_disk()
        return True

    def add_related(self, memory_id: str, related_id: str) -> bool:
        """Add a relationship between memories."""
        memory = self.get(memory_id)
        if not memory:
            return False
        if related_id not in memory.related_ids:
            memory.related_ids.append(related_id)
            self._save_to_disk()
        return True

    def add_tag(self, memory_id: str, tag: str) -> bool:
        """Add a tag to a memory."""
        memory = self.get(memory_id)
        if not memory:
            return False
        if tag not in memory.tags:
            memory.tags.append(tag)
            self._save_to_disk()
        return True

    # ============================================================
    #  DELETE OPERATIONS
    # ============================================================

    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        if memory_id in self.memories:
            del self.memories[memory_id]
            self._save_to_disk()
            return True
        return False

    def clear_all(self) -> None:
        """Clear all memories (use with caution)."""
        self.memories.clear()
        self._save_to_disk()

    # ============================================================
    #  ANALYTICS
    # ============================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        type_counts = {}
        for memory in self.memories.values():
            type_name = memory.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        return {
            "total_memories": len(self.memories),
            "by_type": type_counts,
            "avg_importance": sum(m.importance for m in self.memories.values()) / max(len(self.memories), 1),
            "unique_creators": len(set(m.created_by for m in self.memories.values())),
            "unique_tags": len(set(t for m in self.memories.values() for t in m.tags))
        }

    def export_summary(self) -> str:
        """Export a human-readable summary of memories."""
        lines = ["=" * 50, "COMPANY MEMORY SUMMARY", "=" * 50, ""]

        stats = self.get_stats()
        lines.append(f"Total Memories: {stats['total_memories']}")
        lines.append(f"Average Importance: {stats['avg_importance']:.2f}")
        lines.append("")

        for memory_type in MemoryType:
            memories = self.get_by_type(memory_type)
            if memories:
                lines.append(f"\n{memory_type.value.upper()} ({len(memories)}):")
                lines.append("-" * 30)
                for m in memories[:5]:  # Show top 5
                    lines.append(f"  [{m.id}] {m.content}")

        return "\n".join(lines)
