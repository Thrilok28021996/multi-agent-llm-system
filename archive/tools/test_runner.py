"""Test runner for generated code - supports multiple languages and frameworks."""

import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class TestFramework(Enum):
    """Supported test frameworks."""
    PYTEST = "pytest"
    UNITTEST = "unittest"
    JEST = "jest"
    MOCHA = "mocha"
    GO_TEST = "go_test"
    CARGO_TEST = "cargo_test"
    JUNIT = "junit"


@dataclass
class TestResult:
    """Result of a test run."""
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    total: int = 0
    duration: float = 0.0
    output: str = ""
    error_output: str = ""
    success: bool = True
    framework: str = ""
    test_files: List[str] = field(default_factory=list)
    failures: List[Dict[str, str]] = field(default_factory=list)


class TestRunner:
    """
    Multi-language test runner for generated code.
    Supports Python, JavaScript/TypeScript, Go, Rust, and Java.
    """

    def __init__(self, workspace_root: str = ".", timeout: int = 300):
        self.workspace_root = Path(workspace_root).resolve()
        self.timeout = timeout

    def detect_framework(self, project_path: str) -> Optional[TestFramework]:
        """Detect the test framework based on project files."""
        path = Path(project_path)

        # Check for Python
        if (path / "pytest.ini").exists() or (path / "pyproject.toml").exists():
            return TestFramework.PYTEST
        if (path / "setup.py").exists() and (path / "tests").exists():
            return TestFramework.PYTEST

        # Check for JavaScript/TypeScript
        if (path / "package.json").exists():
            try:
                with open(path / "package.json") as f:
                    pkg = json.load(f)
                    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                    if "jest" in deps:
                        return TestFramework.JEST
                    if "mocha" in deps:
                        return TestFramework.MOCHA
            except Exception:
                pass

        # Check for Go
        if (path / "go.mod").exists():
            return TestFramework.GO_TEST

        # Check for Rust
        if (path / "Cargo.toml").exists():
            return TestFramework.CARGO_TEST

        # Check for Java
        if (path / "pom.xml").exists() or (path / "build.gradle").exists():
            return TestFramework.JUNIT

        # Default to pytest if there are Python test files
        if list(path.glob("**/test_*.py")) or list(path.glob("**/*_test.py")):
            return TestFramework.PYTEST

        return None

    def run_tests(
        self,
        project_path: str,
        framework: Optional[TestFramework] = None,
        verbose: bool = True,
        coverage: bool = False
    ) -> TestResult:
        """
        Run tests for a project.

        Args:
            project_path: Path to the project
            framework: Test framework to use (auto-detect if not specified)
            verbose: Enable verbose output
            coverage: Enable code coverage

        Returns:
            TestResult with details about the test run
        """
        path = Path(project_path).resolve()

        if not path.exists():
            return TestResult(
                success=False,
                error_output=f"Project path does not exist: {path}"
            )

        # Auto-detect framework if not specified
        if framework is None:
            framework = self.detect_framework(str(path))

        if framework is None:
            return TestResult(
                success=False,
                error_output="Could not detect test framework. No test files found."
            )

        # Run the appropriate test command
        if framework == TestFramework.PYTEST:
            return self._run_pytest(path, verbose, coverage)
        elif framework == TestFramework.UNITTEST:
            return self._run_unittest(path, verbose)
        elif framework == TestFramework.JEST:
            return self._run_jest(path, verbose, coverage)
        elif framework == TestFramework.MOCHA:
            return self._run_mocha(path, verbose)
        elif framework == TestFramework.GO_TEST:
            return self._run_go_test(path, verbose, coverage)
        elif framework == TestFramework.CARGO_TEST:
            return self._run_cargo_test(path, verbose)
        elif framework == TestFramework.JUNIT:
            return self._run_junit(path, verbose)
        else:
            return TestResult(
                success=False,
                error_output=f"Unsupported framework: {framework}"
            )

    def _run_pytest(self, path: Path, verbose: bool, coverage: bool) -> TestResult:
        """Run Python tests with pytest."""
        cmd = ["python", "-m", "pytest"]
        if verbose:
            cmd.append("-v")
        if coverage:
            cmd.extend(["--cov", "--cov-report=term-missing"])
        cmd.append(str(path))

        return self._execute_test_command(cmd, path, "pytest")

    def _run_unittest(self, path: Path, verbose: bool) -> TestResult:
        """Run Python tests with unittest."""
        cmd = ["python", "-m", "unittest", "discover"]
        if verbose:
            cmd.append("-v")
        cmd.extend(["-s", str(path)])

        return self._execute_test_command(cmd, path, "unittest")

    def _run_jest(self, path: Path, verbose: bool, coverage: bool) -> TestResult:
        """Run JavaScript/TypeScript tests with Jest."""
        cmd = ["npx", "jest"]
        if verbose:
            cmd.append("--verbose")
        if coverage:
            cmd.append("--coverage")
        cmd.append("--json")

        return self._execute_test_command(cmd, path, "jest")

    def _run_mocha(self, path: Path, verbose: bool) -> TestResult:
        """Run JavaScript tests with Mocha."""
        cmd = ["npx", "mocha"]
        if verbose:
            cmd.append("--reporter=spec")

        return self._execute_test_command(cmd, path, "mocha")

    def _run_go_test(self, path: Path, verbose: bool, coverage: bool) -> TestResult:
        """Run Go tests."""
        cmd = ["go", "test"]
        if verbose:
            cmd.append("-v")
        if coverage:
            cmd.append("-cover")
        cmd.append("./...")

        return self._execute_test_command(cmd, path, "go_test")

    def _run_cargo_test(self, path: Path, verbose: bool) -> TestResult:
        """Run Rust tests with cargo."""
        cmd = ["cargo", "test"]
        if verbose:
            cmd.append("--verbose")

        return self._execute_test_command(cmd, path, "cargo_test")

    def _run_junit(self, path: Path, verbose: bool) -> TestResult:
        """Run Java tests with Maven or Gradle."""
        # Check if using Maven or Gradle
        if (path / "pom.xml").exists():
            cmd = ["mvn", "test"]
            if not verbose:
                cmd.append("-q")
        elif (path / "build.gradle").exists():
            cmd = ["./gradlew", "test"]
            if not verbose:
                cmd.append("--quiet")
        else:
            return TestResult(
                success=False,
                error_output="No Maven or Gradle build file found"
            )

        return self._execute_test_command(cmd, path, "junit")

    def _execute_test_command(
        self,
        cmd: List[str],
        working_dir: Path,
        framework: str
    ) -> TestResult:
        """Execute a test command and parse the results."""
        try:
            result = subprocess.run(
                cmd,
                cwd=str(working_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            output = result.stdout
            error_output = result.stderr
            success = result.returncode == 0

            # Parse test results from output
            test_result = self._parse_test_output(
                output, error_output, framework, success
            )

            return test_result

        except subprocess.TimeoutExpired:
            return TestResult(
                success=False,
                error_output=f"Test execution timed out after {self.timeout} seconds",
                framework=framework
            )
        except FileNotFoundError as e:
            return TestResult(
                success=False,
                error_output=f"Test command not found: {e}",
                framework=framework
            )
        except Exception as e:
            return TestResult(
                success=False,
                error_output=f"Test execution failed: {str(e)}",
                framework=framework
            )

    def _parse_test_output(
        self,
        stdout: str,
        stderr: str,
        framework: str,
        success: bool
    ) -> TestResult:
        """Parse test output to extract results."""
        result = TestResult(
            output=stdout,
            error_output=stderr,
            success=success,
            framework=framework
        )

        # Simple parsing based on framework
        if framework == "pytest":
            # Look for pytest summary line like "5 passed, 2 failed"
            for line in stdout.split("\n"):
                if "passed" in line or "failed" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "passed" in part.lower() and i > 0:
                            try:
                                result.passed = int(parts[i-1])
                            except ValueError:
                                pass
                        if "failed" in part.lower() and i > 0:
                            try:
                                result.failed = int(parts[i-1])
                            except ValueError:
                                pass
                        if "skipped" in part.lower() and i > 0:
                            try:
                                result.skipped = int(parts[i-1])
                            except ValueError:
                                pass
                        if "error" in part.lower() and i > 0:
                            try:
                                result.errors = int(parts[i-1])
                            except ValueError:
                                pass

        elif framework == "jest":
            # Try to parse Jest JSON output
            try:
                # Find JSON in output
                if "{" in stdout:
                    json_start = stdout.index("{")
                    json_str = stdout[json_start:]
                    data = json.loads(json_str)
                    result.passed = data.get("numPassedTests", 0)
                    result.failed = data.get("numFailedTests", 0)
                    result.skipped = data.get("numPendingTests", 0)
            except (json.JSONDecodeError, ValueError):
                pass

        result.total = result.passed + result.failed + result.skipped + result.errors
        result.success = result.failed == 0 and result.errors == 0

        return result


def run_project_tests(project_path: str, verbose: bool = True) -> TestResult:
    """Convenience function to run tests for a project."""
    runner = TestRunner()
    return runner.run_tests(project_path, verbose=verbose)
