"""Learning system for agents to improve from past experiences."""

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ui.console import console


@dataclass
class Lesson:
    """A lesson learned from experience."""
    id: str
    category: str  # problem_solving, communication, coding, decision_making
    content: str
    source_context: str
    outcome: str  # success, failure, partial
    importance: float = 0.5
    times_applied: int = 0
    times_successful: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


@dataclass
class Pattern:
    """A recognized pattern from experiences."""
    id: str
    pattern_type: str  # approach, mistake, success_factor
    description: str
    examples: List[str] = field(default_factory=list)
    frequency: int = 1
    confidence: float = 0.5


class AgentLearning:
    """
    Learning system that helps agents improve from past experiences.
    Extracts lessons, recognizes patterns, and applies learning to new situations.
    """

    def __init__(self, agent_name: str, persist_dir: str = "./output/learning"):
        self.agent_name = agent_name
        self.persist_dir = Path(persist_dir) / agent_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Storage
        self.lessons: Dict[str, Lesson] = {}
        self.patterns: Dict[str, Pattern] = {}
        self.interaction_history: List[Dict[str, Any]] = []

        # Statistics
        self.stats = {
            "total_interactions": 0,
            "successful_outcomes": 0,
            "lessons_applied": 0,
            "patterns_recognized": 0
        }

        self._load_from_disk()

    def _generate_id(self, content: str) -> str:
        """Generate unique ID."""
        return hashlib.sha256(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:12]

    def _load_from_disk(self) -> None:
        """Load learning data from disk."""
        lessons_file = self.persist_dir / "lessons.json"
        patterns_file = self.persist_dir / "patterns.json"
        stats_file = self.persist_dir / "stats.json"

        if lessons_file.exists():
            try:
                data = json.loads(lessons_file.read_text())
                for item in data:
                    item["created_at"] = datetime.fromisoformat(item["created_at"])
                    lesson = Lesson(**item)
                    self.lessons[lesson.id] = lesson
            except Exception as e:
                console.warning(f"Could not load lessons: {e}")

        if patterns_file.exists():
            try:
                data = json.loads(patterns_file.read_text())
                for item in data:
                    pattern = Pattern(**item)
                    self.patterns[pattern.id] = pattern
            except Exception as e:
                console.warning(f"Could not load patterns: {e}")

        if stats_file.exists():
            try:
                self.stats = json.loads(stats_file.read_text())
            except Exception as e:
                console.warning(f"Could not load learning stats: {e}")

    def _save_to_disk(self) -> None:
        """Save learning data to disk."""
        # Save lessons
        lessons_data = []
        for lesson in self.lessons.values():
            item = asdict(lesson)
            item["created_at"] = lesson.created_at.isoformat()
            lessons_data.append(item)
        (self.persist_dir / "lessons.json").write_text(json.dumps(lessons_data, indent=2))

        # Save patterns
        patterns_data = [asdict(p) for p in self.patterns.values()]
        (self.persist_dir / "patterns.json").write_text(json.dumps(patterns_data, indent=2))

        # Save stats
        (self.persist_dir / "stats.json").write_text(json.dumps(self.stats, indent=2))

    # ============================================================
    #  RECORDING EXPERIENCES
    # ============================================================

    def record_interaction(
        self,
        task_type: str,
        input_context: str,
        action_taken: str,
        outcome: str,
        success: bool,
        metadata: Dict[str, Any] = None  # type: ignore
    ) -> None:
        """
        Record an interaction for learning.

        Args:
            task_type: Type of task (e.g., "code_review", "problem_analysis")
            input_context: What was the input/situation
            action_taken: What action was taken
            outcome: What was the result
            success: Was it successful
            metadata: Additional information
        """
        interaction = {
            "task_type": task_type,
            "input_context": input_context,
            "action_taken": action_taken,
            "outcome": outcome,
            "success": success,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }

        self.interaction_history.append(interaction)
        self.stats["total_interactions"] += 1
        if success:
            self.stats["successful_outcomes"] += 1

        # Keep history bounded
        if len(self.interaction_history) > 1000:
            self.interaction_history = self.interaction_history[-500:]

        # Auto-extract lessons from significant interactions
        if success or (not success and len(self.interaction_history) % 5 == 0):
            self._extract_lesson_from_interaction(interaction)

        self._save_to_disk()

    def _extract_lesson_from_interaction(self, interaction: Dict[str, Any]) -> None:
        """Automatically extract lessons from interactions."""
        task_type = interaction.get("task_type", "unknown")
        success = interaction.get("success", False)
        action = interaction.get("action_taken", "")
        outcome = interaction.get("outcome", "")

        # Determine category
        category_map = {
            "code": "coding",
            "review": "coding",
            "implement": "coding",
            "analyze": "problem_solving",
            "research": "problem_solving",
            "decide": "decision_making",
            "approve": "decision_making",
            "communicate": "communication",
            "meeting": "communication",
        }

        category = "general"
        for keyword, cat in category_map.items():
            if keyword in task_type.lower():
                category = cat
                break

        # Create lesson
        if success:
            lesson_content = f"Successful approach: {action}"
            tags = ["success", task_type]
        else:
            lesson_content = f"Approach that failed: {action}. Outcome: {outcome}"
            tags = ["failure", task_type, "avoid"]

        self.add_lesson(
            category=category,
            content=lesson_content,
            source_context=interaction.get("input_context", ""),
            outcome="success" if success else "failure",
            importance=0.7 if success else 0.5,
            tags=tags
        )

    def add_lesson(
        self,
        category: str,
        content: str,
        source_context: str,
        outcome: str,
        importance: float = 0.5,
        tags: List[str] = None  # type: ignore
    ) -> Lesson:
        """
        Manually add a lesson learned.

        Args:
            category: Category of the lesson
            content: The lesson itself
            source_context: Where this lesson came from
            outcome: Was it from success or failure
            importance: How important is this lesson (0-1)
            tags: Tags for categorization
        """
        lesson_id = self._generate_id(content)
        lesson = Lesson(
            id=lesson_id,
            category=category,
            content=content,
            source_context=source_context,
            outcome=outcome,
            importance=importance,
            tags=tags or []
        )
        self.lessons[lesson_id] = lesson
        self._save_to_disk()
        return lesson

    def add_pattern(
        self,
        pattern_type: str,
        description: str,
        examples: List[str] = None  # type: ignore
    ) -> Pattern:
        """Add a recognized pattern."""
        pattern_id = self._generate_id(description)
        pattern = Pattern(
            id=pattern_id,
            pattern_type=pattern_type,
            description=description,
            examples=examples or []
        )
        self.patterns[pattern_id] = pattern
        self.stats["patterns_recognized"] += 1
        self._save_to_disk()
        return pattern

    # ============================================================
    #  APPLYING LEARNING
    # ============================================================

    def get_relevant_lessons(
        self,
        context: str,
        category: str = None,  # type: ignore
        limit: int = 5
    ) -> List[Lesson]:
        """
        Get lessons relevant to a given context.

        Args:
            context: Current situation/context
            category: Filter by category
            limit: Maximum lessons to return
        """
        context_words = set(context.lower().split())
        scored_lessons = []

        for lesson in self.lessons.values():
            # Category filter
            if category and lesson.category != category:
                continue

            # Calculate relevance score
            lesson_words = set(lesson.content.lower().split())
            lesson_words.update(lesson.source_context.lower().split())

            overlap = len(context_words & lesson_words)
            tag_bonus = sum(1 for tag in lesson.tags if tag.lower() in context.lower())

            # Score based on relevance, importance, and success rate
            success_rate = lesson.times_successful / max(lesson.times_applied, 1)
            score = (overlap * 2 + tag_bonus * 3) * lesson.importance * (0.5 + success_rate)

            if score > 0:
                scored_lessons.append((score, lesson))

        # Sort by score and return top lessons
        scored_lessons.sort(reverse=True, key=lambda x: x[0])
        return [lesson for _, lesson in scored_lessons[:limit]]

    def apply_lesson(self, lesson_id: str, successful: bool) -> None:
        """
        Record that a lesson was applied.

        Args:
            lesson_id: ID of the lesson
            successful: Was applying it successful
        """
        if lesson_id in self.lessons:
            self.lessons[lesson_id].times_applied += 1
            if successful:
                self.lessons[lesson_id].times_successful += 1
                # Increase importance for successful lessons
                self.lessons[lesson_id].importance = min(
                    1.0,
                    self.lessons[lesson_id].importance + 0.05
                )
            else:
                # Decrease importance for unsuccessful lessons
                self.lessons[lesson_id].importance = max(
                    0.1,
                    self.lessons[lesson_id].importance - 0.05
                )

            self.stats["lessons_applied"] += 1
            self._save_to_disk()

    def get_advice_for_task(self, task_type: str, context: str) -> str:
        """
        Get advice based on past learning for a task.

        Args:
            task_type: Type of task
            context: Current context

        Returns:
            Advice string based on lessons and patterns
        """
        advice_parts = []

        # Get relevant lessons
        lessons = self.get_relevant_lessons(
            f"{task_type} {context}",
            limit=3
        )

        if lessons:
            advice_parts.append("Based on past experience:")
            for lesson in lessons:
                if lesson.outcome == "success":
                    advice_parts.append(f"  ✓ {lesson.content}")
                else:
                    advice_parts.append(f"  ✗ Avoid: {lesson.content}")

        # Get relevant patterns
        relevant_patterns = [
            p for p in self.patterns.values()
            if any(word in p.description.lower() for word in task_type.lower().split())
        ]

        if relevant_patterns:
            advice_parts.append("\nRecognized patterns:")
            for pattern in relevant_patterns[:2]:
                advice_parts.append(f"  • {pattern.description}")

        return "\n".join(advice_parts) if advice_parts else ""

    # ============================================================
    #  ANALYSIS
    # ============================================================

    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze learning performance and provide insights."""
        total = self.stats["total_interactions"]
        if total == 0:
            return {"status": "no_data", "message": "No interactions recorded yet"}

        success_rate = self.stats["successful_outcomes"] / total

        # Analyze lessons
        lesson_stats = defaultdict(int)
        for lesson in self.lessons.values():
            lesson_stats[lesson.category] += 1

        # Find most effective lessons
        effective_lessons = sorted(
            self.lessons.values(),
            key=lambda l: l.times_successful / max(l.times_applied, 1) * l.importance,
            reverse=True
        )[:5]

        return {
            "total_interactions": total,
            "success_rate": success_rate,
            "total_lessons": len(self.lessons),
            "total_patterns": len(self.patterns),
            "lessons_by_category": dict(lesson_stats),
            "most_effective_lessons": [
                {"content": l.content, "success_rate": l.times_successful / max(l.times_applied, 1)}
                for l in effective_lessons
            ],
            "lessons_applied": self.stats["lessons_applied"]
        }

    def get_improvement_suggestions(self) -> List[str]:
        """Get suggestions for improvement based on learning data."""
        suggestions = []
        analysis = self.analyze_performance()

        if analysis.get("status") == "no_data":
            return ["Start recording interactions to enable learning"]

        success_rate = analysis.get("success_rate", 0)

        if success_rate < 0.5:
            suggestions.append("Success rate is low. Review failed interactions for patterns.")

        # Check for underused successful lessons
        underused = [
            l for l in self.lessons.values()
            if l.outcome == "success" and l.times_applied < 3 and l.importance > 0.6
        ]
        if underused:
            suggestions.append(f"You have {len(underused)} successful lessons that could be applied more often.")

        # Check category balance
        categories = analysis.get("lessons_by_category", {})
        if categories:
            min_cat = min(categories.items(), key=lambda x: x[1])
            suggestions.append(f"Consider focusing on learning more about '{min_cat[0]}' (only {min_cat[1]} lessons)")

        return suggestions if suggestions else ["Keep up the good work! Continue recording interactions."]

    def export_learning_summary(self) -> str:
        """Export a human-readable learning summary."""
        analysis = self.analyze_performance()

        lines = [
            f"=== Learning Summary for {self.agent_name} ===",
            "",
            f"Total Interactions: {analysis.get('total_interactions', 0)}",
            f"Success Rate: {analysis.get('success_rate', 0):.1%}",
            f"Lessons Learned: {analysis.get('total_lessons', 0)}",
            f"Patterns Recognized: {analysis.get('total_patterns', 0)}",
            "",
            "Lessons by Category:"
        ]

        for cat, count in analysis.get("lessons_by_category", {}).items():
            lines.append(f"  - {cat}: {count}")

        lines.append("\nMost Effective Lessons:")
        for lesson in analysis.get("most_effective_lessons", []):
            lines.append(f"  ✓ {lesson['content']}... ({lesson['success_rate']:.0%} success)")

        lines.append("\nSuggestions:")
        for suggestion in self.get_improvement_suggestions():
            lines.append(f"  → {suggestion}")

        return "\n".join(lines)
