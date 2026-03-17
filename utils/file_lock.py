"""Cross-platform file locking utilities to prevent race conditions."""

import json
import logging
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

logger = logging.getLogger(__name__)

# Module-level variables for platform-specific locking
fcntl: Any = None
msvcrt: Any = None
LOCK_AVAILABLE = False

if sys.platform == 'win32':
    try:
        import msvcrt as msvcrt_module
        msvcrt = msvcrt_module
        LOCK_AVAILABLE = True
    except ImportError:
        pass
else:
    try:
        import fcntl as fcntl_module
        fcntl = fcntl_module
        LOCK_AVAILABLE = True
    except ImportError:
        pass


@dataclass
class LockConfig:
    """Configuration for file locking."""
    timeout: float = 10.0  # Maximum time to wait for lock (seconds)
    retry_interval: float = 0.1  # Time between lock attempts
    exclusive: bool = True  # Exclusive (write) or shared (read) lock


class FileLockError(Exception):
    """Exception raised when file locking fails."""
    pass


class FileLockTimeoutError(FileLockError):
    """Exception raised when lock acquisition times out."""
    pass


class FileLock:
    """
    Cross-platform file lock implementation.

    Uses fcntl on Unix/macOS and msvcrt on Windows.
    Provides both exclusive (write) and shared (read) locking.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        exclusive: bool = True,
        timeout: float = 10.0
    ):
        """
        Initialize a file lock.

        Args:
            file_path: Path to the file to lock
            exclusive: True for exclusive (write) lock, False for shared (read)
            timeout: Maximum time to wait for lock acquisition
        """
        self.file_path = Path(file_path)
        self.lock_path = self.file_path.with_suffix(self.file_path.suffix + '.lock')
        self.exclusive = exclusive
        self.timeout = timeout
        self._lock_file = None
        self._locked = False

    def acquire(self) -> bool:
        """
        Acquire the file lock.

        Returns:
            True if lock was acquired successfully

        Raises:
            FileLockTimeoutError: If lock cannot be acquired within timeout
        """
        if not LOCK_AVAILABLE:
            # Fallback: no locking available, proceed without lock
            return True

        start_time = time.time()

        # Create lock file directory if needed
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        while True:
            try:
                # Open lock file
                self._lock_file = open(self.lock_path, 'w')

                if sys.platform == 'win32':
                    # Windows locking
                    if self.exclusive:
                        msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        # Windows doesn't have native shared locks, use exclusive
                        msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    # Unix/macOS locking
                    if self.exclusive:
                        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    else:
                        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)

                self._locked = True
                return True

            except (IOError, OSError, BlockingIOError):
                # Lock is held by another process
                if self._lock_file:
                    self._lock_file.close()
                    self._lock_file = None

                # Check timeout
                elapsed = time.time() - start_time
                if elapsed >= self.timeout:
                    raise FileLockTimeoutError(
                        f"Could not acquire lock for {self.file_path} within {self.timeout}s"
                    )

                # Wait before retry
                time.sleep(min(0.1, self.timeout - elapsed))

    def release(self) -> None:
        """Release the file lock."""
        if not self._lock_file:
            return

        try:
            if LOCK_AVAILABLE:
                if sys.platform == 'win32':
                    try:
                        msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                    except (IOError, OSError):
                        pass
                else:
                    try:
                        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                    except (IOError, OSError):
                        pass
        finally:
            if self._lock_file:
                self._lock_file.close()
                self._lock_file = None
            self._locked = False

            # Clean up lock file
            try:
                if self.lock_path.exists():
                    self.lock_path.unlink()
            except (IOError, OSError):
                pass

    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False

    @property
    def locked(self) -> bool:
        """Check if lock is currently held."""
        return self._locked


@contextmanager
def file_lock(
    file_path: Union[str, Path],
    exclusive: bool = True,
    timeout: float = 10.0
):
    """
    Context manager for file locking.

    Usage:
        with file_lock('/path/to/file.json', exclusive=True):
            # File is locked here
            data = read_file()
            write_file(modified_data)
        # Lock is released here

    Args:
        file_path: Path to the file to lock
        exclusive: True for write lock, False for read lock
        timeout: Maximum time to wait for lock
    """
    lock = FileLock(file_path, exclusive=exclusive, timeout=timeout)
    try:
        lock.acquire()
        yield lock
    finally:
        lock.release()


def atomic_write_json(
    file_path: Union[str, Path],
    data: Any,
    indent: int = 2,
    timeout: float = 10.0
) -> None:
    """
    Atomically write JSON data to a file with locking.

    Uses a temporary file and atomic rename to prevent partial writes.

    Args:
        file_path: Path to the JSON file
        data: Data to serialize and write
        indent: JSON indentation level
        timeout: Lock timeout in seconds
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_lock(file_path, exclusive=True, timeout=timeout):
        # Write to temporary file first
        fd, temp_path = tempfile.mkstemp(
            dir=file_path.parent,
            prefix=f'.{file_path.stem}_',
            suffix='.tmp'
        )

        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=indent)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename (on POSIX systems)
            # On Windows, we need to remove the target first
            if sys.platform == 'win32' and file_path.exists():
                file_path.unlink()

            os.rename(temp_path, file_path)

        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise


def safe_read_json(
    file_path: Union[str, Path],
    default: Any = None,
    timeout: float = 10.0
) -> Any:
    """
    Safely read JSON data from a file with locking.

    Args:
        file_path: Path to the JSON file
        default: Default value if file doesn't exist or is invalid
        timeout: Lock timeout in seconds

    Returns:
        Parsed JSON data or default value
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return default

    try:
        with file_lock(file_path, exclusive=False, timeout=timeout):
            return json.loads(file_path.read_text())
    except (json.JSONDecodeError, FileLockError) as e:
        logger.warning("Could not read %s: %s", file_path, e)
        return default
