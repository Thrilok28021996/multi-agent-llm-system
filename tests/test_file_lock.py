"""Tests for file locking utilities."""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.file_lock import (
    FileLock,
    file_lock,
    atomic_write_json,
    safe_read_json,
)


class TestFileLock:
    """Tests for FileLock class."""

    def test_acquire_release(self, tmp_path):
        """Test basic lock acquire and release."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        lock = FileLock(test_file, exclusive=True)
        assert not lock.locked

        lock.acquire()
        assert lock.locked

        lock.release()
        assert not lock.locked

    def test_context_manager(self, tmp_path):
        """Test lock as context manager."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        with FileLock(test_file, exclusive=True) as lock:
            assert lock.locked

        assert not lock.locked

    def test_file_lock_context_manager(self, tmp_path):
        """Test file_lock context manager function."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        with file_lock(test_file, exclusive=True) as lock:
            assert lock.locked

        assert not lock.locked


class TestAtomicWriteJson:
    """Tests for atomic_write_json function."""

    def test_write_new_file(self, tmp_path):
        """Test writing to a new file."""
        test_file = tmp_path / "new.json"
        data = {"key": "value", "number": 42}

        atomic_write_json(test_file, data)

        assert test_file.exists()
        loaded = json.loads(test_file.read_text())
        assert loaded == data

    def test_overwrite_existing_file(self, tmp_path):
        """Test overwriting an existing file."""
        test_file = tmp_path / "existing.json"
        test_file.write_text('{"old": "data"}')

        new_data = {"new": "data"}
        atomic_write_json(test_file, new_data)

        loaded = json.loads(test_file.read_text())
        assert loaded == new_data

    def test_creates_parent_directories(self, tmp_path):
        """Test that parent directories are created."""
        test_file = tmp_path / "nested" / "dir" / "file.json"
        data = {"test": True}

        atomic_write_json(test_file, data)

        assert test_file.exists()
        loaded = json.loads(test_file.read_text())
        assert loaded == data

    def test_custom_indent(self, tmp_path):
        """Test custom indentation."""
        test_file = tmp_path / "indented.json"
        data = {"key": "value"}

        atomic_write_json(test_file, data, indent=4)

        content = test_file.read_text()
        assert "    " in content  # 4-space indent


class TestSafeReadJson:
    """Tests for safe_read_json function."""

    def test_read_existing_file(self, tmp_path):
        """Test reading an existing JSON file."""
        test_file = tmp_path / "existing.json"
        data = {"key": "value"}
        test_file.write_text(json.dumps(data))

        result = safe_read_json(test_file)
        assert result == data

    def test_read_nonexistent_file(self, tmp_path):
        """Test reading a nonexistent file returns default."""
        test_file = tmp_path / "nonexistent.json"
        default = {"default": True}

        result = safe_read_json(test_file, default=default)
        assert result == default

    def test_read_invalid_json(self, tmp_path):
        """Test reading invalid JSON returns default."""
        test_file = tmp_path / "invalid.json"
        test_file.write_text("not valid json {{{")
        default = {"fallback": True}

        result = safe_read_json(test_file, default=default)
        assert result == default

    def test_read_empty_file(self, tmp_path):
        """Test reading empty file returns default."""
        test_file = tmp_path / "empty.json"
        test_file.write_text("")
        default = {"empty_default": True}

        result = safe_read_json(test_file, default=default)
        assert result == default


class TestFileLockIntegration:
    """Integration tests for file locking."""

    def test_atomic_write_and_read(self, tmp_path):
        """Test atomic write followed by safe read."""
        test_file = tmp_path / "integration.json"
        data = {"items": [1, 2, 3], "nested": {"a": "b"}}

        atomic_write_json(test_file, data)
        result = safe_read_json(test_file)

        assert result == data

    def test_multiple_writes(self, tmp_path):
        """Test multiple sequential writes."""
        test_file = tmp_path / "multiple.json"

        for i in range(5):
            data = {"iteration": i}
            atomic_write_json(test_file, data)

        result = safe_read_json(test_file)
        assert result == {"iteration": 4}

    def test_lock_prevents_corruption(self, tmp_path):
        """Test that lock prevents data corruption on concurrent access."""
        test_file = tmp_path / "locked.json"
        initial_data = {"counter": 0}
        atomic_write_json(test_file, initial_data)

        # Simulate locked write
        with file_lock(test_file, exclusive=True):
            data = safe_read_json(test_file)
            data["counter"] += 1
            atomic_write_json(test_file, data)

        result = safe_read_json(test_file)
        assert result["counter"] == 1
