"""
Git Integration Tools for Company AGI.

Provides Claude Code-style git operations with:
- Smart commit message generation
- PR creation with summaries
- Branch management
- Diff analysis
"""

import asyncio
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class GitChangeType(Enum):
    """Types of git changes."""
    ADDED = "A"
    MODIFIED = "M"
    DELETED = "D"
    RENAMED = "R"
    COPIED = "C"
    UNTRACKED = "?"
    IGNORED = "!"


@dataclass
class GitChange:
    """A single file change in git."""
    path: str
    change_type: GitChangeType
    old_path: Optional[str] = None  # For renames
    additions: int = 0
    deletions: int = 0
    is_staged: bool = False
    diff_preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "change_type": self.change_type.value,
            "old_path": self.old_path,
            "additions": self.additions,
            "deletions": self.deletions,
            "is_staged": self.is_staged,
        }


@dataclass
class GitCommit:
    """A git commit."""
    hash: str
    short_hash: str
    message: str
    author: str
    author_email: str
    date: datetime
    files_changed: int = 0
    additions: int = 0
    deletions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hash": self.hash,
            "short_hash": self.short_hash,
            "message": self.message,
            "author": self.author,
            "author_email": self.author_email,
            "date": self.date.isoformat(),
            "files_changed": self.files_changed,
            "additions": self.additions,
            "deletions": self.deletions,
        }


@dataclass
class GitBranch:
    """A git branch."""
    name: str
    is_current: bool = False
    is_remote: bool = False
    tracking: Optional[str] = None
    ahead: int = 0
    behind: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "is_current": self.is_current,
            "is_remote": self.is_remote,
            "tracking": self.tracking,
            "ahead": self.ahead,
            "behind": self.behind,
        }


@dataclass
class PRInfo:
    """Pull request information."""
    title: str
    body: str
    base: str
    head: str
    url: Optional[str] = None
    number: Optional[int] = None
    state: str = "open"


@dataclass
class GitResult:
    """Result of a git operation."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    command: Optional[str] = None


class GitTools:
    """
    Git integration tools.

    Features:
    - Status and diff analysis
    - Smart commit message generation
    - PR creation with auto-summary
    - Branch management
    """

    CO_AUTHOR = "Co-Authored-By: Claude <noreply@anthropic.com>"

    def __init__(
        self,
        repo_path: str = ".",
        default_branch: str = "main",
        summarizer: Optional[Any] = None,  # LLM for summaries
    ):
        self.repo_path = Path(repo_path).resolve()
        self.default_branch = default_branch
        self.summarizer = summarizer

    def _run_git(
        self,
        args: List[str],
        capture_output: bool = True,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Run a git command."""
        cmd = ["git", "-C", str(self.repo_path)] + args
        return subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            check=check,
        )

    async def _run_git_async(
        self,
        args: List[str],
    ) -> subprocess.CompletedProcess[str]:
        """Run a git command asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run_git, args)

    def is_repo(self) -> bool:
        """Check if current directory is a git repo."""
        result = self._run_git(["rev-parse", "--git-dir"])
        return result.returncode == 0

    def get_status(self) -> GitResult:
        """Get git status with detailed change information."""
        if not self.is_repo():
            return GitResult(success=False, error="Not a git repository")

        # Get status with porcelain format
        result = self._run_git(["status", "--porcelain", "-u"])
        if result.returncode != 0:
            return GitResult(success=False, error=result.stderr)

        changes: List[GitChange] = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            # Parse porcelain format: XY PATH or XY ORIG -> PATH
            index_status = line[0]
            worktree_status = line[1]
            path_part = line[3:]

            # Handle renames
            old_path = None
            if " -> " in path_part:
                old_path, path_part = path_part.split(" -> ")

            # Determine change type and staging
            if index_status == "?":
                change_type = GitChangeType.UNTRACKED
                is_staged = False
            elif index_status == "!":
                change_type = GitChangeType.IGNORED
                is_staged = False
            elif index_status != " ":
                # Staged change
                change_type = self._parse_change_type(index_status)
                is_staged = True
            else:
                # Unstaged change
                change_type = self._parse_change_type(worktree_status)
                is_staged = False

            changes.append(GitChange(
                path=path_part,
                change_type=change_type,
                old_path=old_path,
                is_staged=is_staged,
            ))

        return GitResult(success=True, data=changes)

    def _parse_change_type(self, code: str) -> GitChangeType:
        """Parse change type from status code."""
        mapping = {
            "A": GitChangeType.ADDED,
            "M": GitChangeType.MODIFIED,
            "D": GitChangeType.DELETED,
            "R": GitChangeType.RENAMED,
            "C": GitChangeType.COPIED,
            "?": GitChangeType.UNTRACKED,
            "!": GitChangeType.IGNORED,
        }
        return mapping.get(code, GitChangeType.MODIFIED)

    def get_diff(
        self,
        staged: bool = False,
        file_path: Optional[str] = None,
    ) -> GitResult:
        """Get diff of changes."""
        args = ["diff"]
        if staged:
            args.append("--staged")
        if file_path:
            args.extend(["--", file_path])

        result = self._run_git(args)
        if result.returncode != 0:
            return GitResult(success=False, error=result.stderr)

        return GitResult(success=True, data=result.stdout)

    def get_diff_stats(self, staged: bool = False) -> GitResult:
        """Get diff statistics."""
        args = ["diff", "--stat"]
        if staged:
            args.append("--staged")

        result = self._run_git(args)
        if result.returncode != 0:
            return GitResult(success=False, error=result.stderr)

        return GitResult(success=True, data=result.stdout)

    def get_log(
        self,
        limit: int = 10,
        format_str: Optional[str] = None,
    ) -> GitResult:
        """Get recent commits."""
        # Custom format: hash|short|message|author|email|date
        fmt = format_str or "%H|%h|%s|%an|%ae|%ai"
        args = ["log", f"-{limit}", f"--format={fmt}"]

        result = self._run_git(args)
        if result.returncode != 0:
            return GitResult(success=False, error=result.stderr)

        commits: List[GitCommit] = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 6:
                commits.append(GitCommit(
                    hash=parts[0],
                    short_hash=parts[1],
                    message=parts[2],
                    author=parts[3],
                    author_email=parts[4],
                    date=datetime.fromisoformat(parts[5].replace(" ", "T")),
                ))

        return GitResult(success=True, data=commits)

    def get_branches(self) -> GitResult:
        """Get list of branches."""
        result = self._run_git(["branch", "-a", "-vv"])
        if result.returncode != 0:
            return GitResult(success=False, error=result.stderr)

        branches: List[GitBranch] = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            is_current = line.startswith("*")
            line = line[2:].strip()

            # Parse branch info
            parts = line.split()
            name = parts[0]

            is_remote = name.startswith("remotes/")
            if is_remote:
                name = name.replace("remotes/", "")

            # Parse tracking info if present
            tracking = None
            ahead = behind = 0
            tracking_match = re.search(r"\[(.+?)\]", line)
            if tracking_match:
                tracking_info = tracking_match.group(1)
                if ":" in tracking_info:
                    tracking = tracking_info.split(":")[0].strip()
                else:
                    tracking = tracking_info.strip()

                ahead_match = re.search(r"ahead (\d+)", tracking_info)
                if ahead_match:
                    ahead = int(ahead_match.group(1))

                behind_match = re.search(r"behind (\d+)", tracking_info)
                if behind_match:
                    behind = int(behind_match.group(1))

            branches.append(GitBranch(
                name=name,
                is_current=is_current,
                is_remote=is_remote,
                tracking=tracking,
                ahead=ahead,
                behind=behind,
            ))

        return GitResult(success=True, data=branches)

    def get_current_branch(self) -> GitResult:
        """Get current branch name."""
        result = self._run_git(["branch", "--show-current"])
        if result.returncode != 0:
            return GitResult(success=False, error=result.stderr)

        return GitResult(success=True, data=result.stdout.strip())

    def stage_files(self, files: List[str]) -> GitResult:
        """Stage files for commit."""
        if not files:
            return GitResult(success=False, error="No files specified")

        result = self._run_git(["add"] + files)
        if result.returncode != 0:
            return GitResult(success=False, error=result.stderr)

        return GitResult(success=True, data=f"Staged {len(files)} files")

    def stage_all(self) -> GitResult:
        """Stage all changes."""
        result = self._run_git(["add", "-A"])
        if result.returncode != 0:
            return GitResult(success=False, error=result.stderr)

        return GitResult(success=True, data="All changes staged")

    def unstage_files(self, files: List[str]) -> GitResult:
        """Unstage files."""
        result = self._run_git(["reset", "HEAD"] + files)
        if result.returncode != 0:
            return GitResult(success=False, error=result.stderr)

        return GitResult(success=True, data=f"Unstaged {len(files)} files")

    def generate_commit_message(
        self,
        changes: Optional[List[GitChange]] = None,
        diff: Optional[str] = None,
    ) -> str:
        """Generate a smart commit message based on changes."""
        if changes is None:
            status = self.get_status()
            changes = status.data if status.success else []

        if diff is None:
            diff_result = self.get_diff(staged=True)
            diff = diff_result.data if diff_result.success else ""

        # Ensure we have valid values
        changes_list = changes or []
        diff_str = diff or ""

        # If we have a summarizer (LLM), use it
        if self.summarizer and diff_str:
            return self._llm_commit_message(changes_list, diff_str)

        # Otherwise, generate based on file analysis
        return self._analyze_changes_for_message(changes_list, diff_str)

    def _llm_commit_message(
        self,
        changes: List[GitChange],
        diff: str,
    ) -> str:
        """Use LLM to generate commit message."""
        # Truncate diff if too long
        diff_preview = diff

        prompt = f"""Generate a concise git commit message for these changes.
Follow conventional commit format (type: description).
Focus on the "why" not just the "what".

Files changed:
{chr(10).join(f"- {c.change_type.value} {c.path}" for c in changes[:20])}

Diff preview:
{diff_preview}

Commit message:"""

        try:
            if self.summarizer is not None:
                if hasattr(self.summarizer, 'generate'):
                    message = self.summarizer.generate(prompt)  # type: ignore[union-attr]
                elif callable(self.summarizer):
                    message = self.summarizer(prompt)
                else:
                    return self._analyze_changes_for_message(changes, diff)
            else:
                return self._analyze_changes_for_message(changes, diff)

            # Clean up the message
            message = str(message).strip()
            if message:
                return message
        except Exception:
            pass

        return self._analyze_changes_for_message(changes, diff)

    def _analyze_changes_for_message(
        self,
        changes: List[GitChange],
        diff: str,
    ) -> str:
        """Analyze changes to generate commit message."""
        if not changes:
            return "chore: update files"

        # Count change types
        added = sum(1 for c in changes if c.change_type == GitChangeType.ADDED)
        modified = sum(1 for c in changes if c.change_type == GitChangeType.MODIFIED)
        deleted = sum(1 for c in changes if c.change_type == GitChangeType.DELETED)

        # Analyze file types
        file_types: Dict[str, int] = {}
        for c in changes:
            ext = Path(c.path).suffix or "other"
            file_types[ext] = file_types.get(ext, 0) + 1

        # Detect common patterns
        paths = [c.path for c in changes]
        all_tests = all("test" in p.lower() for p in paths)
        all_docs = all(p.endswith((".md", ".rst", ".txt")) for p in paths)
        all_config = all(p.endswith((".json", ".yaml", ".yml", ".toml")) for p in paths)

        # Generate message based on patterns
        if all_tests:
            return "test: update test files"
        elif all_docs:
            return "docs: update documentation"
        elif all_config:
            return "chore: update configuration"
        elif added > 0 and modified == 0 and deleted == 0:
            if added == 1:
                return f"feat: add {Path(changes[0].path).name}"
            return f"feat: add {added} new files"
        elif deleted > 0 and added == 0 and modified == 0:
            return f"chore: remove {deleted} files"
        elif modified == 1 and added == 0 and deleted == 0:
            filename = Path(changes[0].path).name
            # Try to detect fix vs update
            if diff and ("fix" in diff.lower() or "bug" in diff.lower()):
                return f"fix: update {filename}"
            return f"refactor: update {filename}"
        else:
            parts = []
            if added:
                parts.append(f"add {added}")
            if modified:
                parts.append(f"update {modified}")
            if deleted:
                parts.append(f"remove {deleted}")
            return f"chore: {', '.join(parts)} files"

    def commit(
        self,
        message: Optional[str] = None,
        add_co_author: bool = True,
    ) -> GitResult:
        """Create a commit."""
        if message is None:
            message = self.generate_commit_message()

        # Add co-author line
        if add_co_author and self.CO_AUTHOR not in message:
            message = f"{message}\n\n{self.CO_AUTHOR}"

        # Use heredoc-style commit
        result = self._run_git(["commit", "-m", message])
        if result.returncode != 0:
            return GitResult(success=False, error=result.stderr, command="git commit")

        return GitResult(success=True, data=message)

    def push(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
        set_upstream: bool = False,
    ) -> GitResult:
        """Push commits to remote."""
        args = ["push"]
        if set_upstream:
            args.append("-u")
        args.append(remote)
        if branch:
            args.append(branch)

        result = self._run_git(args)
        if result.returncode != 0:
            return GitResult(success=False, error=result.stderr)

        return GitResult(success=True, data="Push successful")

    def create_branch(
        self,
        name: str,
        checkout: bool = True,
    ) -> GitResult:
        """Create a new branch."""
        args = ["checkout", "-b", name] if checkout else ["branch", name]
        result = self._run_git(args)
        if result.returncode != 0:
            return GitResult(success=False, error=result.stderr)

        return GitResult(success=True, data=f"Created branch: {name}")

    def checkout(self, ref: str) -> GitResult:
        """Checkout a branch or commit."""
        result = self._run_git(["checkout", ref])
        if result.returncode != 0:
            return GitResult(success=False, error=result.stderr)

        return GitResult(success=True, data=f"Checked out: {ref}")

    def generate_pr_body(
        self,
        base_branch: Optional[str] = None,
    ) -> str:
        """Generate PR body with summary of changes."""
        base = base_branch or self.default_branch

        # Get commits since base
        result = self._run_git(["log", f"{base}..HEAD", "--oneline"])
        commits = result.stdout.strip().split("\n") if result.returncode == 0 else []

        # Get diff stats
        diff_stats = self._run_git(["diff", f"{base}..HEAD", "--stat"])
        stats = diff_stats.stdout if diff_stats.returncode == 0 else ""

        # Build PR body
        body_parts = [
            "## Summary",
            "",
        ]

        # Add commit summaries
        if commits:
            for commit in commits[:10]:
                body_parts.append(f"- {commit}")
            if len(commits) > 10:
                body_parts.append(f"- ... and {len(commits) - 10} more commits")
        else:
            body_parts.append("- No commits yet")

        body_parts.extend([
            "",
            "## Changes",
            "",
            "```",
            stats if stats else "No changes",
            "```",
            "",
            "## Test plan",
            "",
            "- [ ] Tests pass locally",
            "- [ ] Manual testing completed",
            "",
            "---",
            "Generated with Claude Code",
        ])

        return "\n".join(body_parts)

    def create_pr(
        self,
        title: Optional[str] = None,
        body: Optional[str] = None,
        base: Optional[str] = None,
    ) -> GitResult:
        """Create a pull request using gh CLI."""
        # Check if gh is available
        result = subprocess.run(["which", "gh"], capture_output=True)
        if result.returncode != 0:
            return GitResult(success=False, error="GitHub CLI (gh) not installed")

        base = base or self.default_branch

        # Generate title if not provided
        if not title:
            branch_result = self.get_current_branch()
            if branch_result.success:
                title = branch_result.data.replace("-", " ").replace("_", " ").title()
            else:
                title = "Update"

        # Generate body if not provided
        if not body:
            body = self.generate_pr_body(base)

        # Create PR
        args = [
            "gh", "pr", "create",
            "--title", title,
            "--body", body,
            "--base", base,
        ]

        result = subprocess.run(args, capture_output=True, text=True, cwd=str(self.repo_path))
        if result.returncode != 0:
            return GitResult(success=False, error=result.stderr)

        # Extract PR URL from output
        url = result.stdout.strip()

        return GitResult(
            success=True,
            data=PRInfo(
                title=title or "Update",
                body=body or "",
                base=base,
                head="",  # Would need to parse
                url=url,
            )
        )


# Singleton instance
_git_tools: Optional[GitTools] = None


def get_git_tools(repo_path: str = ".") -> GitTools:
    """Get or create the global git tools."""
    global _git_tools
    if _git_tools is None:
        _git_tools = GitTools(repo_path=repo_path)
    return _git_tools


def reset_git_tools() -> None:
    """Reset the global git tools."""
    global _git_tools
    _git_tools = None
