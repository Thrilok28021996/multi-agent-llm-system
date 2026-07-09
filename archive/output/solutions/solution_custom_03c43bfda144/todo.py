#!/usr/bin/env python3
"""
Simple CLI todo list app.
- Single file, Python stdlib only (json, argparse, pathlib, sys, os)
- Local JSON persistence, no external dependencies
- Cross-platform (macOS/Linux/Windows)
"""

import json
import argparse
import sys
from pathlib import Path

# Constants
TODO_FILE = Path.home() / ".todo.json"

def load_tasks():
    """Load tasks from JSON file. Returns list of tasks (dicts with 'id', 'text', 'done')."""
    if not TODO_FILE.exists():
        return []
    try:
        with open(TODO_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

def save_tasks(tasks):
    """Save tasks to JSON file."""
    try:
        with open(TODO_FILE, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2)
    except IOError as e:
        print(f"Error saving tasks: {e}", file=sys.stderr)
        sys.exit(1)

def add_task(text):
    """Add a new task with given text."""
    tasks = load_tasks()
    task_id = len(tasks) + 1
    task = {"id": task_id, "text": text, "done": False}
    tasks.append(task)
    save_tasks(tasks)
    print(f"Added task {task_id}: '{text}'")

def list_tasks():
    """List all tasks."""
    tasks = load_tasks()
    if not tasks:
        print("No tasks found.")
        return
    for task in tasks:
        status = "[x]" if task["done"] else "[ ]"
        print(f"{task['id']}. {status} {task['text']}")

def complete_task(task_id):
    """Mark a task as done by ID."""
    tasks = load_tasks()
    found = False
    for task in tasks:
        if task["id"] == task_id:
            task["done"] = True
            found = True
            break
    if not found:
        print(f"Error: Task ID {task_id} not found.")
        sys.exit(1)
    save_tasks(tasks)
    print(f"Marked task {task_id} as done.")

def delete_task(task_id):
    """Delete a task by ID."""
    tasks = load_tasks()
    new_tasks = [t for t in tasks if t["id"] != task_id]
    if len(new_tasks) == len(tasks):
        print(f"Error: Task ID {task_id} not found.")
        sys.exit(1)
    save_tasks(new_tasks)
    print(f"Deleted task {task_id}.")

def clear_all():
    """Delete all tasks."""
    tasks = load_tasks()
    if not tasks:
        print("No tasks to clear.")
        return
    confirm = input("Are you sure you want to delete all tasks? [y/N] ").strip().lower()
    if confirm == "y":
        save_tasks([])
        print("All tasks deleted.")
    else:
        print("Cancelled.")

def main():
    parser = argparse.ArgumentParser(description="Simple CLI todo list")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # add command
    add_parser = subparsers.add_parser("add", help="Add a new task")
    add_parser.add_argument("text", type=str, help="Task description")

    # list command
    subparsers.add_parser("list", help="List all tasks")

    # complete command
    complete_parser = subparsers.add_parser("complete", help="Mark a task as done")
    complete_parser.add_argument("id", type=int, help="Task ID")

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a task")
    delete_parser.add_argument("id", type=int, help="Task ID")

    # clear command
    subparsers.add_parser("clear", help="Delete all tasks")

    args = parser.parse_args()

    if args.command == "add":
        add_task(args.text)
    elif args.command == "list":
        list_tasks()
    elif args.command == "complete":
        complete_task(args.id)
    elif args.command == "delete":
        delete_task(args.id)
    elif args.command == "clear":
        clear_all()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()