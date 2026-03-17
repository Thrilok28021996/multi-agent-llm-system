"""
Async Task Queue with Worker Pool for Company AGI.

Provides advanced task scheduling:
- Async task queue with priority support
- Dynamic worker pool sizing
- Task dependency resolution
- Queue monitoring and metrics
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Coroutine, Dict, List, Optional, Set

from ui.console import console


class QueuePriority(Enum):
    """Queue priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class TaskState(Enum):
    """Task execution states."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"  # Waiting for dependencies


@dataclass
class TaskResult:
    """Result of task execution."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "result": str(self.result) if self.result else None,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
        }


@dataclass
class QueuedTask:
    """A task in the queue."""
    task_id: str
    name: str
    coro: Coroutine[Any, Any, Any]
    priority: QueuePriority = QueuePriority.NORMAL
    state: TaskState = TaskState.PENDING
    dependencies: Set[str] = field(default_factory=set)
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[TaskResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def __lt__(self, other: "QueuedTask") -> bool:
        """Compare by priority for heap ordering."""
        # Higher priority = should come first (negate for min heap)
        return self.priority.value > other.priority.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "priority": self.priority.value,
            "state": self.state.value,
            "dependencies": list(self.dependencies),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result.to_dict() if self.result else None,
        }


@dataclass
class QueueMetrics:
    """Metrics for the task queue."""
    total_tasks: int = 0
    pending_tasks: int = 0
    running_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    avg_wait_time_ms: float = 0.0
    avg_execution_time_ms: float = 0.0
    total_wait_time_ms: float = 0.0
    total_execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "pending_tasks": self.pending_tasks,
            "running_tasks": self.running_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "cancelled_tasks": self.cancelled_tasks,
            "avg_wait_time_ms": self.avg_wait_time_ms,
            "avg_execution_time_ms": self.avg_execution_time_ms,
        }


class AsyncTaskQueue:
    """
    Async task queue with worker pool.

    Features:
    - Priority-based task scheduling
    - Dependency resolution
    - Dynamic worker pool
    - Task monitoring and metrics
    """

    def __init__(
        self,
        max_workers: int = 4,
        max_queue_size: int = 1000,
    ):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size

        self._queue: asyncio.PriorityQueue[QueuedTask] = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._tasks: Dict[str, QueuedTask] = {}
        self._workers: List[asyncio.Task[None]] = []
        self._running = False
        self._lock = asyncio.Lock()

        # Metrics tracking
        self._metrics = QueueMetrics()
        self._wait_times: List[float] = []
        self._execution_times: List[float] = []

        # Event for task completion notifications
        self._completion_events: Dict[str, asyncio.Event] = {}

    async def start(self) -> None:
        """Start the worker pool."""
        if self._running:
            return

        self._running = True

        # Create workers
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self._workers.append(worker)

    async def stop(self, timeout: float = 30.0) -> None:
        """Stop the worker pool."""
        self._running = False

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish
        if self._workers:
            await asyncio.wait(self._workers, timeout=timeout)

        self._workers.clear()

    async def _worker_loop(self, worker_id: str) -> None:
        """Worker loop that processes tasks."""
        while self._running:
            try:
                # Get task from queue with timeout
                try:
                    task = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                # Check if dependencies are met
                if not await self._check_dependencies(task):
                    # Re-queue the task
                    task.state = TaskState.BLOCKED
                    await self._queue.put(task)
                    await asyncio.sleep(0.1)  # Small delay before retry
                    continue

                # Execute task
                await self._execute_task(task, worker_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Worker should not crash, but log the error
                console.warning(f"[TaskQueue] Worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)

    async def _check_dependencies(self, task: QueuedTask) -> bool:
        """Check if task dependencies are satisfied."""
        if not task.dependencies:
            return True

        for dep_id in task.dependencies:
            dep_task = self._tasks.get(dep_id)
            if not dep_task:
                # Dependency doesn't exist - consider it satisfied
                continue
            if dep_task.state not in [TaskState.COMPLETED]:
                return False

        return True

    async def _execute_task(self, task: QueuedTask, worker_id: str) -> None:
        """Execute a single task."""
        # worker_id can be used for logging/debugging
        _ = worker_id
        async with self._lock:
            task.state = TaskState.RUNNING
            task.started_at = datetime.now().isoformat()
            self._metrics.running_tasks += 1
            self._metrics.pending_tasks -= 1

        # Calculate wait time
        created = datetime.fromisoformat(task.created_at)
        started = datetime.fromisoformat(task.started_at) if task.started_at else datetime.now()
        wait_time_ms = (started - created).total_seconds() * 1000
        self._wait_times.append(wait_time_ms)

        start_time = time.perf_counter()
        success = False
        result_value: Any = None
        error_msg: Optional[str] = None

        try:
            # Execute the coroutine
            result_value = await task.coro
            success = True
        except asyncio.CancelledError:
            error_msg = "Task cancelled"
            task.state = TaskState.CANCELLED
        except Exception as e:
            error_msg = str(e)
            task.state = TaskState.FAILED

        duration_ms = (time.perf_counter() - start_time) * 1000
        self._execution_times.append(duration_ms)

        # Update task
        async with self._lock:
            task.completed_at = datetime.now().isoformat()
            task.result = TaskResult(
                task_id=task.task_id,
                success=success,
                result=result_value,
                error=error_msg,
                started_at=task.started_at,
                completed_at=task.completed_at,
                duration_ms=duration_ms,
            )

            if success:
                task.state = TaskState.COMPLETED
                self._metrics.completed_tasks += 1
            elif task.state == TaskState.CANCELLED:
                self._metrics.cancelled_tasks += 1
            else:
                self._metrics.failed_tasks += 1

            self._metrics.running_tasks -= 1

            # Update averages
            if self._wait_times:
                self._metrics.avg_wait_time_ms = sum(self._wait_times) / len(self._wait_times)
                self._metrics.total_wait_time_ms = sum(self._wait_times)
            if self._execution_times:
                self._metrics.avg_execution_time_ms = sum(self._execution_times) / len(self._execution_times)
                self._metrics.total_execution_time_ms = sum(self._execution_times)

        # Signal completion
        if task.task_id in self._completion_events:
            self._completion_events[task.task_id].set()

        self._queue.task_done()

    async def submit(
        self,
        coro: Coroutine[Any, Any, Any],
        name: Optional[str] = None,
        priority: QueuePriority = QueuePriority.NORMAL,
        dependencies: Optional[Set[str]] = None,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Submit a task to the queue.

        Args:
            coro: Coroutine to execute
            name: Task name
            priority: Task priority
            dependencies: Set of task IDs this task depends on
            task_id: Optional custom task ID
            metadata: Optional metadata

        Returns:
            Task ID
        """
        task_id = task_id or str(uuid.uuid4())[:8]
        name = name or f"task-{task_id}"

        task = QueuedTask(
            task_id=task_id,
            name=name,
            coro=coro,
            priority=priority,
            state=TaskState.QUEUED,
            dependencies=dependencies or set(),
            metadata=metadata or {},
        )

        async with self._lock:
            self._tasks[task_id] = task
            self._metrics.total_tasks += 1
            self._metrics.pending_tasks += 1
            self._completion_events[task_id] = asyncio.Event()

        await self._queue.put(task)

        return task_id

    async def submit_batch(
        self,
        tasks: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Submit multiple tasks.

        Args:
            tasks: List of task dicts with keys: coro, name, priority, dependencies

        Returns:
            List of task IDs
        """
        task_ids = []
        for task_dict in tasks:
            task_id = await self.submit(
                coro=task_dict["coro"],
                name=task_dict.get("name"),
                priority=task_dict.get("priority", QueuePriority.NORMAL),
                dependencies=task_dict.get("dependencies"),
                metadata=task_dict.get("metadata"),
            )
            task_ids.append(task_id)
        return task_ids

    async def wait(
        self,
        task_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[TaskResult]:
        """Wait for a task to complete."""
        if task_id not in self._completion_events:
            task = self._tasks.get(task_id)
            if task and task.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED]:
                return task.result
            return None

        event = self._completion_events[task_id]

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

        task = self._tasks.get(task_id)
        return task.result if task else None

    async def wait_all(
        self,
        task_ids: List[str],
        timeout: Optional[float] = None,
    ) -> Dict[str, Optional[TaskResult]]:
        """Wait for multiple tasks to complete."""
        results = {}

        async def wait_one(tid: str) -> None:
            results[tid] = await self.wait(tid, timeout)

        await asyncio.gather(*[wait_one(tid) for tid in task_ids])
        return results

    async def cancel(self, task_id: str) -> bool:
        """Cancel a task."""
        task = self._tasks.get(task_id)
        if not task:
            return False

        if task.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED]:
            return False

        async with self._lock:
            task.state = TaskState.CANCELLED
            self._metrics.cancelled_tasks += 1
            if task.state == TaskState.PENDING:
                self._metrics.pending_tasks -= 1

        return True

    def get_task(self, task_id: str) -> Optional[QueuedTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_task_state(self, task_id: str) -> Optional[TaskState]:
        """Get task state."""
        task = self._tasks.get(task_id)
        return task.state if task else None

    def get_metrics(self) -> QueueMetrics:
        """Get queue metrics."""
        return self._metrics

    def get_pending_tasks(self) -> List[QueuedTask]:
        """Get all pending tasks."""
        return [
            t for t in self._tasks.values()
            if t.state in [TaskState.PENDING, TaskState.QUEUED, TaskState.BLOCKED]
        ]

    def get_running_tasks(self) -> List[QueuedTask]:
        """Get all running tasks."""
        return [t for t in self._tasks.values() if t.state == TaskState.RUNNING]

    def clear_completed(self) -> int:
        """Clear completed tasks from memory."""
        to_remove = [
            tid for tid, task in self._tasks.items()
            if task.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED]
        ]

        for tid in to_remove:
            del self._tasks[tid]
            if tid in self._completion_events:
                del self._completion_events[tid]

        return len(to_remove)

    def get_dashboard(self, color: bool = True) -> str:
        """Get queue dashboard as formatted string."""
        colors = {
            "header": "\033[1;36m",
            "running": "\033[32m",
            "pending": "\033[33m",
            "failed": "\033[31m",
            "reset": "\033[0m",
            "bold": "\033[1m",
        }

        if not color:
            colors = {k: "" for k in colors}

        c = colors
        m = self._metrics

        lines = [
            f"{c['bold']}Task Queue Dashboard{c['reset']}",
            "=" * 50,
            f"\n{c['header']}Queue Status:{c['reset']}",
            f"  Workers: {len(self._workers)}/{self.max_workers}",
            f"  Queue Size: {self._queue.qsize()}/{self.max_queue_size}",
            f"\n{c['header']}Task Statistics:{c['reset']}",
            f"  {c['pending']}Pending:{c['reset']}   {m.pending_tasks}",
            f"  {c['running']}Running:{c['reset']}   {m.running_tasks}",
            f"  Completed: {m.completed_tasks}",
            f"  {c['failed']}Failed:{c['reset']}    {m.failed_tasks}",
            f"  Cancelled: {m.cancelled_tasks}",
            f"  Total:     {m.total_tasks}",
            f"\n{c['header']}Performance:{c['reset']}",
            f"  Avg Wait Time:      {m.avg_wait_time_ms:.1f}ms",
            f"  Avg Execution Time: {m.avg_execution_time_ms:.1f}ms",
        ]

        # Show running tasks
        running = self.get_running_tasks()
        if running:
            lines.append(f"\n{c['header']}Running Tasks:{c['reset']}")
            for task in running[:5]:
                lines.append(f"  [{task.task_id}] {task.name}")

        return "\n".join(lines)


# Global task queue instance
_task_queue: Optional[AsyncTaskQueue] = None


async def get_task_queue(
    max_workers: int = 4,
    auto_start: bool = True,
) -> AsyncTaskQueue:
    """Get or create the global task queue."""
    global _task_queue
    if _task_queue is None:
        _task_queue = AsyncTaskQueue(max_workers=max_workers)
        if auto_start:
            await _task_queue.start()
    return _task_queue


async def reset_task_queue() -> None:
    """Reset the global task queue."""
    global _task_queue
    if _task_queue:
        await _task_queue.stop()
    _task_queue = None
