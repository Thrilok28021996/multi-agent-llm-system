"""Message bus for inter-agent communication."""

import json
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import PriorityQueue
from typing import Any, Callable, Dict, List

from ui.console import console


@dataclass(order=True)
class PrioritizedMessage:
    """Message with priority for queue ordering."""

    priority: int
    timestamp: float = field(compare=False)
    message: Any = field(compare=False)


class MessageBus:
    """
    Central message bus for agent communication.
    Supports pub/sub, direct messaging, and broadcast.
    """

    def __init__(self, log_dir: str = None):
        # Registered agents
        self.agents: Dict[str, Any] = {}

        # Topic subscriptions
        self.subscriptions: Dict[str, List[str]] = defaultdict(list)

        # Message queues per agent
        self.queues: Dict[str, PriorityQueue] = {}

        # Message history for audit
        self.message_history: List[Dict[str, Any]] = []

        # Callbacks for message handlers
        self.handlers: Dict[str, List[Callable]] = defaultdict(list)

        # Threading
        self._lock = threading.Lock()

        # Logging
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    #  AGENT REGISTRATION
    # ============================================================

    def register_agent(self, agent_name: str, agent: Any) -> None:
        """Register an agent with the message bus."""
        with self._lock:
            self.agents[agent_name] = agent
            self.queues[agent_name] = PriorityQueue()

            # Set message callback on agent
            if hasattr(agent, "set_message_callback"):
                agent.set_message_callback(
                    lambda msg: self.send_message(msg)
                )

    def unregister_agent(self, agent_name: str) -> None:
        """Unregister an agent from the message bus."""
        with self._lock:
            if agent_name in self.agents:
                del self.agents[agent_name]
            if agent_name in self.queues:
                del self.queues[agent_name]
            # Remove from all subscriptions
            for subscribers in self.subscriptions.values():
                if agent_name in subscribers:
                    subscribers.remove(agent_name)

    def get_registered_agents(self) -> List[str]:
        """Get list of registered agent names."""
        return list(self.agents.keys())

    # ============================================================
    #  TOPIC SUBSCRIPTION
    # ============================================================

    def subscribe(self, agent_name: str, topic: str) -> None:
        """Subscribe an agent to a topic."""
        with self._lock:
            if agent_name not in self.subscriptions[topic]:
                self.subscriptions[topic].append(agent_name)

    def unsubscribe(self, agent_name: str, topic: str) -> None:
        """Unsubscribe an agent from a topic."""
        with self._lock:
            if agent_name in self.subscriptions[topic]:
                self.subscriptions[topic].remove(agent_name)

    def get_subscribers(self, topic: str) -> List[str]:
        """Get list of agents subscribed to a topic."""
        return self.subscriptions.get(topic, [])

    # ============================================================
    #  MESSAGE SENDING
    # ============================================================

    def send_message(self, message: Any) -> bool:
        """
        Send a message through the bus.

        Args:
            message: Message object with sender, recipient, content, etc.
        """
        try:
            msg_dict = message.to_dict() if hasattr(message, "to_dict") else {
                "sender": getattr(message, "sender", "unknown"),
                "recipient": getattr(message, "recipient", "all"),
                "content": getattr(message, "content", str(message)),
                "message_type": getattr(message, "message_type", "general"),
                "priority": getattr(message, "priority", 1),
                "timestamp": datetime.now().isoformat()
            }

            recipient = msg_dict["recipient"]
            priority = msg_dict.get("priority", 1)

            # Log message
            self._log_message(msg_dict)

            if recipient == "all":
                # Broadcast to all agents except sender
                return self._broadcast(message, msg_dict["sender"])
            elif recipient in self.agents:
                # Direct message
                return self._deliver(recipient, message, priority)
            else:
                # Check if it's a topic
                if recipient in self.subscriptions:
                    return self._publish_to_topic(recipient, message)

            return False

        except Exception as e:
            console.warning(f"Error sending message: {e}")
            return False

    def _deliver(self, recipient: str, message: Any, priority: int) -> bool:
        """Deliver message to a specific agent."""
        if recipient not in self.queues:
            return False

        prioritized = PrioritizedMessage(
            priority=-priority,  # Negative for max-priority behavior
            timestamp=datetime.now().timestamp(),
            message=message
        )

        self.queues[recipient].put(prioritized)

        # Also directly deliver to agent's inbox if available
        if recipient in self.agents:
            agent = self.agents[recipient]
            if hasattr(agent, "receive_message"):
                agent.receive_message(message)

        return True

    def _broadcast(self, message: Any, exclude_sender: str = None) -> bool:
        """Broadcast message to all agents."""
        success = True
        for agent_name in self.agents:
            if agent_name != exclude_sender:
                if not self._deliver(agent_name, message, 1):
                    success = False
        return success

    def _publish_to_topic(self, topic: str, message: Any) -> bool:
        """Publish message to all subscribers of a topic."""
        subscribers = self.subscriptions.get(topic, [])
        success = True
        for subscriber in subscribers:
            if not self._deliver(subscriber, message, 1):
                success = False
        return success

    def publish(
        self,
        topic: str,
        sender: str,
        content: str,
        priority: int = 1,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Publish a message to a topic.

        Args:
            topic: Topic to publish to
            sender: Sender agent name
            content: Message content
            priority: Message priority (1-5)
            metadata: Additional metadata
        """
        from agents.base_agent import Message

        message = Message(
            sender=sender,
            recipient=topic,
            content=content,
            message_type="topic",
            priority=priority,
            metadata=metadata or {}
        )

        return self._publish_to_topic(topic, message)

    # ============================================================
    #  MESSAGE RECEIVING
    # ============================================================

    def get_messages(
        self,
        agent_name: str,
        limit: int = None
    ) -> List[Any]:
        """
        Get pending messages for an agent.

        Args:
            agent_name: Name of the agent
            limit: Maximum number of messages to retrieve
        """
        if agent_name not in self.queues:
            return []

        messages = []
        queue = self.queues[agent_name]

        count = 0
        while not queue.empty():
            if limit and count >= limit:
                break
            prioritized = queue.get()
            messages.append(prioritized.message)
            count += 1

        return messages

    def peek_messages(self, agent_name: str) -> int:
        """Get count of pending messages without removing them."""
        if agent_name not in self.queues:
            return 0
        return self.queues[agent_name].qsize()

    # ============================================================
    #  HANDLER REGISTRATION
    # ============================================================

    def register_handler(
        self,
        message_type: str,
        handler: Callable
    ) -> None:
        """
        Register a handler for a message type.

        Args:
            message_type: Type of message to handle
            handler: Callback function(message) -> None
        """
        self.handlers[message_type].append(handler)

    def process_handlers(self, message: Any) -> None:
        """Process all registered handlers for a message."""
        msg_type = getattr(message, "message_type", "general")
        for handler in self.handlers.get(msg_type, []):
            try:
                handler(message)
            except Exception as e:
                console.warning(f"Handler error: {e}")

    # ============================================================
    #  COMPANY-WIDE COMMUNICATION
    # ============================================================

    def call_meeting(
        self,
        organizer: str,
        attendees: List[str],
        agenda: str
    ) -> str:
        """
        Simulate a company meeting.

        Args:
            organizer: Agent organizing the meeting
            attendees: List of agent names to attend
            agenda: Meeting agenda

        Returns:
            Meeting ID
        """
        from agents.base_agent import Message

        meeting_id = f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Notify all attendees
        for attendee in attendees:
            message = Message(
                sender=organizer,
                recipient=attendee,
                content=f"[MEETING INVITE]\nMeeting ID: {meeting_id}\nOrganizer: {organizer}\nAgenda: {agenda}",
                message_type="meeting",
                priority=4,
                metadata={"meeting_id": meeting_id, "attendees": attendees}
            )
            self.send_message(message)

        return meeting_id

    def announce(self, sender: str, announcement: str) -> None:
        """Make a company-wide announcement."""
        from agents.base_agent import Message

        message = Message(
            sender=sender,
            recipient="all",
            content=f"[ANNOUNCEMENT] {announcement}",
            message_type="announcement",
            priority=5
        )
        self.send_message(message)

    def request_decision(
        self,
        from_agent: str,
        to_agent: str,
        question: str,
        options: List[str] = None
    ) -> None:
        """Request a decision from another agent."""
        from agents.base_agent import Message

        content = f"[DECISION REQUIRED]\n{question}"
        if options:
            content += "\nOptions:\n" + "\n".join(f"  - {o}" for o in options)

        message = Message(
            sender=from_agent,
            recipient=to_agent,
            content=content,
            message_type="decision",
            priority=4,
            metadata={"options": options}
        )
        self.send_message(message)

    def submit_report(
        self,
        from_agent: str,
        to_agent: str,
        report_type: str,
        report_content: str
    ) -> None:
        """Submit a report to another agent."""
        from agents.base_agent import Message

        message = Message(
            sender=from_agent,
            recipient=to_agent,
            content=f"[REPORT: {report_type}]\n{report_content}",
            message_type="report",
            priority=2,
            metadata={"report_type": report_type}
        )
        self.send_message(message)

    # ============================================================
    #  LOGGING & HISTORY
    # ============================================================

    def _log_message(self, msg_dict: Dict[str, Any]) -> None:
        """Log a message to history."""
        self.message_history.append(msg_dict)

        # Keep history bounded
        if len(self.message_history) > 1000:
            self.message_history = self.message_history[-500:]

        # Write to file if log_dir is set
        if self.log_dir:
            log_file = self.log_dir / f"messages_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(log_file, "a") as f:
                f.write(json.dumps(msg_dict) + "\n")

    def get_message_history(
        self,
        limit: int = 100,
        sender: str = None,
        recipient: str = None,
        message_type: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get message history with optional filters.

        Args:
            limit: Maximum messages to return
            sender: Filter by sender
            recipient: Filter by recipient
            message_type: Filter by message type
        """
        history = self.message_history

        if sender:
            history = [m for m in history if m.get("sender") == sender]
        if recipient:
            history = [m for m in history if m.get("recipient") == recipient]
        if message_type:
            history = [m for m in history if m.get("message_type") == message_type]

        return history[-limit:]

    def get_communication_stats(self) -> Dict[str, Any]:
        """Get statistics about message bus activity."""
        stats = {
            "total_messages": len(self.message_history),
            "registered_agents": len(self.agents),
            "active_topics": len(self.subscriptions),
            "pending_messages": sum(
                q.qsize() for q in self.queues.values()
            ),
            "messages_by_type": defaultdict(int),
            "messages_by_sender": defaultdict(int)
        }

        for msg in self.message_history:
            stats["messages_by_type"][msg.get("message_type", "unknown")] += 1
            stats["messages_by_sender"][msg.get("sender", "unknown")] += 1

        return dict(stats)

    def clear_history(self) -> None:
        """Clear message history."""
        self.message_history.clear()

    def clear_all_queues(self) -> None:
        """Clear all message queues."""
        for queue in self.queues.values():
            while not queue.empty():
                queue.get()

    # ============================================================
    #  AGENT-TO-AGENT COLLABORATION
    # ============================================================

    def request_help(self, from_agent: str, to_agent: str, question: str, context: str = "") -> None:
        """Agent requests help from another agent."""
        from agents.base_agent import Message
        content = f"[HELP REQUEST]\nFrom: {from_agent}\nQuestion: {question}"
        if context:
            content += f"\nContext: {context}"
        message = Message(
            sender=from_agent,
            recipient=to_agent,
            content=content,
            message_type="help_request",
            priority=4,
            metadata={"type": "help_request", "question": question}
        )
        self.send_message(message)

    def notify_issue(self, from_agent: str, to_agent: str, issue: str, severity: str = "medium") -> None:
        """Agent notifies another agent about an issue found."""
        from agents.base_agent import Message
        message = Message(
            sender=from_agent,
            recipient=to_agent,
            content=f"[ISSUE NOTIFICATION] Severity: {severity}\n{issue}",
            message_type="issue_notification",
            priority=3 if severity == "medium" else 5,
            metadata={"type": "issue", "severity": severity}
        )
        self.send_message(message)

    def share_finding(self, from_agent: str, to_agent: str, finding: str) -> None:
        """Agent shares a research finding with another agent."""
        from agents.base_agent import Message
        message = Message(
            sender=from_agent,
            recipient=to_agent,
            content=f"[SHARED FINDING]\n{finding}",
            message_type="finding",
            priority=2,
            metadata={"type": "shared_finding"}
        )
        self.send_message(message)
