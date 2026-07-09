# AUTO-GENERATED SCAFFOLD — Review and complete TODOs before production use
#!/usr/bin/env python3
"""
Tests for the todo list application.
Uses unittest and mocks to test functionality without file I/O.
"""

import unittest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from io import StringIO
import sys

# Import the module under test
import todo

class TestTodoApp(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for tests
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test_tasks.json")
        
        # Mock the DATA_FILE to use our test file
        self.patcher = patch.object(todo, 'DATA_FILE', 
                                    todo.Path(self.test_file))
        self.patcher.start()
        
        # Ensure clean state
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()
        if os.path.exists(self.test_dir):
            import shutil
            shutil.rmtree(self.test_dir)
    
    def test_add_task(self):
        """Test adding a task."""
        todo.add_task("Test task")
        
        # Verify file was created and contains the task
        self.assertTrue(os.path.exists(self.test_file))
        
        with open(self.test_file, 'r') as f:
            tasks = json.load(f)
        
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["description"], "Test task")
        self.assertFalse(tasks[0]["completed"])
    
    def test_list_tasks_empty(self):
        """Test listing tasks when none exist."""
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        todo.list_tasks()
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        self.assertIn("No tasks found.", output)
    
    def test_complete_task(self):
        """Test marking a task as completed."""
        # First add a task
        todo.add_task("Test task")
        
        # Then complete it
        todo.complete_task(1)
        
        # Verify it's marked as completed
        tasks = todo.load_tasks()
        self.assertTrue(tasks[0]["completed"])
    
    def test_delete_task(self):
        """Test deleting a task."""
        # Add two tasks
        todo.add_task("First task")
        todo.add_task("Second task")
        
        # Delete the first one
        todo.delete_task(1)
        
        # Verify only one task remains
        tasks = todo.load_tasks()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["description"], "Second task")
    
    def test_invalid_command(self):
        """Test handling of invalid commands."""
        # Mock sys.argv for testing
        with patch('sys.argv', ['todo.py', 'invalid']):
            captured_output = StringIO()
            sys.stdout = captured_output
            
            todo.main()
            
            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()
            
            self.assertIn("Unknown command", output)

if __name__ == '__main__':
    unittest.main()