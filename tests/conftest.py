"""Pytest configuration and fixtures for Company AGI tests."""

import sys
from pathlib import Path

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def sample_problem():
    """Return a sample problem description."""
    return {
        "title": "Sample Problem",
        "description": "This is a sample problem description for testing purposes.",
        "severity": "medium",
        "source": "test",
        "domain": "testing"
    }


@pytest.fixture
def sample_code():
    """Return sample code for testing."""
    return '''
def hello_world():
    """Print hello world."""
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()
'''
