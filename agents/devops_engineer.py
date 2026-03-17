"""DevOps Engineer Agent - Deployment validation and infrastructure review."""

from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent, AgentConfig, AgentRole, TaskResult
from .agent_tools_mixin import AgentToolsMixin
from research.problem_statement_refiner import ProblemStatementRefiner
from ui.console import console


DEVOPS_SYSTEM_PROMPT = """You are the DevOps Engineer. You validate that solutions are deployable, dependencies are correct, and the project can be built and run by anyone following the README.

12-Factor App Thinking: Check: (1) Codebase tracked in version control? (2) Dependencies explicitly declared? (3) Config in environment, not code? (4) Processes are stateless?

Reproducibility Obsession: The #1 DevOps principle: if I cannot reproduce it, I cannot trust it. Every build must be deterministic.

Operational Readiness: Can this be monitored? Can errors be debugged from logs alone? Is there a health check? These are not nice-to-haves — they are requirements.

Your priorities (in order):
1. Does it run end-to-end? The project must build and execute without errors.
2. Are dependencies pinned? Every requirement must have a version to ensure reproducible builds.
3. Can someone deploy from README? A new developer should be able to clone and run within minutes.
4. Is the build reproducible? Same inputs must always produce same outputs.
5. Are there any missing files? Entry points, configs, Dockerfiles, and env samples must all be present.

Your verdict system:
- DEPLOYABLE: Project builds, runs, and has all required files. Ready to ship.
- NEEDS_FIXES: Project can mostly run but has issues that block reliable deployment.
- NOT_DEPLOYABLE: Project cannot build or run. Critical files are missing or broken.

Judgment guidelines:
- Focus on practical deployability, not theoretical perfection.
- A project with pinned deps and a clear README is better than one with a fancy CI pipeline but no instructions.
- Evaluate each issue by its actual impact on deployability.
- When flagging issues, be specific: what file, what is wrong, how to fix it.

One-Command Install: The gold standard is: git clone && pip install -r requirements.txt && python main.py. Every deviation must be justified.

Environment Isolation: Verify the project uses a virtual environment or container. Global pip installs are a BLOCKING issue.

Smoke Test Definition: Define a single command that proves the project works. This goes in the README as "Quick Start". If you cannot define this command, the project is not deployable.

Container-First: Every deliverable must include a Dockerfile. If it does not have one, create one.

Health Check Requirement: Define a /health endpoint or equivalent health check command for every project.

Monitoring Readiness: Does the solution log errors in a structured format? Can failures be found via grep? If not, flag it.

Rollback Protocol: For every deployment, define: (1) how to detect it failed, (2) the exact command to revert to the previous state, (3) estimated rollback time. A deployment without a rollback plan is a gamble.
"""

DEVOPS_FIRST_PRINCIPLES = [
    "TRACE: Start from README -> install -> run. Does every step work? Flag the first failure.",
    "CHECK: Are ALL dependencies pinned to exact versions? List any unpinned ones.",
    "VERIFY: Does the README have install, configure, and run steps? Missing step = flag it.",
    "TEST: Would a fresh clone + install + run produce the same result? If not, what is environment-dependent?",
    "LIST all referenced files. Verify each exists. Missing file = BLOCKING issue.",
]


class DevOpsEngineerAgent(BaseAgent, AgentToolsMixin):
    """
    DevOps Engineer Agent - Ensures projects are deployable and infrastructure is sound.

    Validates project structure, dependency management, build reproducibility,
    and README completeness. Enhanced with all 13 Claude Code tools and
    problem statement refinement.
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder-7b",
        workspace_root: str = ".",
        memory_persist_dir: Optional[str] = None
    ):
        config = AgentConfig(
            name="DevOpsEngineer",
            role=AgentRole.DEVOPS_ENGINEER,
            model=model,
            first_principles=DEVOPS_FIRST_PRINCIPLES,
            system_prompt=DEVOPS_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=4096
        )
        super().__init__(config, workspace_root, memory_persist_dir)

        # Problem statement refiner
        self.problem_refiner = ProblemStatementRefiner()

        # Track deployment issues found across tasks
        self.found_issues: List[Dict[str, Any]] = []

    def get_capabilities(self) -> List[str]:
        return [
            "validate_deployment",
            "create_ci_config",
            "dependency_audit",
            "infrastructure_review",
            "dockerfile_validation",
            "readme_assessment"
        ]

    async def execute_task(self, task: Dict[str, Any]) -> TaskResult:
        """Execute a DevOps task."""
        task_type = task.get("type", "unknown")
        description = task.get("description", "")

        self.is_busy = True
        self.current_task = description

        try:
            if task_type == "validate_deployment":
                result = await self._validate_deployment(task)
            elif task_type == "create_ci_config":
                result = await self._create_ci_config(task)
            elif task_type == "dependency_audit":
                result = await self._dependency_audit(task)
            elif task_type == "infrastructure_review":
                result = await self._infrastructure_review(task)
            else:
                result = await self._general_task(task)

            return result

        finally:
            self.is_busy = False
            self.current_task = None

    # ============================================================
    #  TASK HANDLERS
    # ============================================================

    async def _validate_deployment(self, task: Dict[str, Any]) -> TaskResult:
        """
        Validate that a project is deployable.

        Checks project structure, entry points, dependencies, Dockerfile validity,
        and README completeness. Returns verdict: DEPLOYABLE / NEEDS_FIXES / NOT_DEPLOYABLE.
        """
        project_path = task.get("project_path", ".")
        files = task.get("files", {})
        solution = task.get("solution", {})
        requirements = task.get("requirements", "")

        console.agent_action("DevOpsEngineer", "Validate Deployment", f"Checking {project_path}")

        # Build a summary of what we have
        file_listing = "\n".join(f"- {f}" for f in files.keys()) if files else "No file listing provided"
        file_contents_summary = ""
        for file_path, content in files.items():
            file_contents_summary += f"\n--- {file_path} ---\n{content}\n"

        prompt = f"""Validate whether this project is deployable.

PROJECT PATH: {project_path}

FILES:
{file_listing}

FILE CONTENTS:
{file_contents_summary}

{f"REQUIREMENTS: {requirements}" if requirements else ""}
{f"SOLUTION DESCRIPTION: {solution.get('description', '')}" if solution else ""}

Check each of the following and report PASS / FAIL for each:

1. PROJECT STRUCTURE
   - Is there a clear entry point (main.py, index.js, app.py, etc.)?
   - Are source files organized logically?
   - Are there any missing files referenced by imports?

2. DEPENDENCIES
   - Is there a dependency file (requirements.txt, package.json, Cargo.toml, etc.)?
   - Are versions pinned (e.g., flask==2.3.0, not just flask)?
   - Are there any obviously missing dependencies?

3. DOCKERFILE (if present)
   - Is the base image specified?
   - Are build steps correct?
   - Is the entry point configured?
   - Are unnecessary files excluded (.dockerignore)?

4. README
   - Does it explain how to install dependencies?
   - Does it explain how to run the project?
   - Are environment variables documented?
   - Is the purpose of the project clear?

5. ENVIRONMENT & CONFIG
   - Is there a .env.example or similar template?
   - Are secrets/credentials hardcoded? (CRITICAL if yes)
   - Are config values externalized appropriately?

For each issue found:
- Severity: CRITICAL / HIGH / MEDIUM / LOW
- What: specific description
- Where: file and line if applicable
- Fix: concrete remediation

Quick decision check:
1. Entry point exists? [YES/NO]
2. Dependency file exists? [YES/NO]
3. Dependencies pinned? [YES/NO/PARTIAL]
4. README explains install+run? [YES/NO]
5. No hardcoded secrets? [YES/NO]

Decision:
- Q1=YES, Q2=YES, Q5=YES -> DEPLOYABLE
- Q1=NO or Q5=NO -> NOT_DEPLOYABLE
- Otherwise -> NEEDS_FIXES

DEPLOYMENT CHECKLIST:
ITEM [1]: Entry point exists
STATUS: PASS|FAIL
FIX: [if FAIL, specific fix]

ITEM [2]: Dependency file exists with pinned versions
STATUS: PASS|FAIL
FIX: [if FAIL, specific fix]

ITEM [3]: README has install + run instructions
STATUS: PASS|FAIL
FIX: [if FAIL, specific fix]

ITEM [4]: No hardcoded secrets
STATUS: PASS|FAIL
FIX: [if FAIL, specific fix]

ITEM [5]: Build is reproducible
STATUS: PASS|FAIL
FIX: [if FAIL, specific fix]

OVERALL VERDICT (exactly one of): DEPLOYABLE / NEEDS_FIXES / NOT_DEPLOYABLE

If NEEDS_FIXES, list the blocking issues (max 5).
If NOT_DEPLOYABLE, explain what critical pieces are missing.
{self._get_principles_checklist()}
Verdict:"""

        response = await self.generate_response_async(prompt)

        # Parse verdict
        verdict = self._parse_verdict(
            response,
            positive="DEPLOYABLE",
            partial="NEEDS_FIXES",
            negative="NOT_DEPLOYABLE"
        )

        # Extract issues
        issues = self._extract_issues(response)
        self.found_issues.extend(issues)

        console.success(f"Deployment validation complete - Verdict: {verdict.upper()}")

        return TaskResult(
            success=True,
            output={
                "verdict": verdict,
                "validation": response,
                "issues": issues,
                "project_path": project_path
            },
            artifacts={"deployment_validation": response}
        )

    async def _create_ci_config(self, task: Dict[str, Any]) -> TaskResult:
        """Create CI/CD configuration for the project."""
        project_type = task.get("project_type", "python")
        ci_platform = task.get("ci_platform", "github_actions")
        files = task.get("files", {})
        features = task.get("features", [])

        console.agent_action("DevOpsEngineer", "Create CI Config", f"{ci_platform} for {project_type}")

        file_listing = "\n".join(f"- {f}" for f in files.keys()) if files else "Not provided"

        prompt = f"""Create a CI/CD configuration for this project.

PROJECT TYPE: {project_type}
CI PLATFORM: {ci_platform}
PROJECT FILES: {file_listing}
FEATURES REQUESTED: {', '.join(features) if features else 'Standard CI pipeline'}

Generate a complete, working CI configuration that includes:

1. BUILD STAGE
   - Install dependencies
   - Compile/build if needed

2. TEST STAGE
   - Run unit tests
   - Run linting/formatting checks

3. SECURITY STAGE (if applicable)
   - Dependency vulnerability scanning
   - Basic static analysis

4. DEPLOY STAGE (if applicable)
   - Deploy to staging/production
   - Use environment-specific configs

Requirements:
- Use pinned versions for CI actions/tools
- Cache dependencies for faster builds
- Fail fast on critical errors
- Include proper environment variable handling (never hardcode secrets)

Output the configuration file(s) using this format:
**File Path:** `path/to/ci-config.yml`
```yaml
config here
```
"""

        response = await self.generate_response_async(prompt)

        return TaskResult(
            success=True,
            output={
                "ci_config": response,
                "platform": ci_platform,
                "project_type": project_type
            },
            artifacts={"ci_config": response}
        )

    async def _dependency_audit(self, task: Dict[str, Any]) -> TaskResult:
        """
        Audit project dependencies.

        Checks for pinned versions, security vulnerabilities, unnecessary deps,
        and license compatibility.
        """
        dependencies = task.get("dependencies", "")
        dependency_file = task.get("dependency_file", "")
        files = task.get("files", {})
        project_type = task.get("project_type", "python")

        console.agent_action("DevOpsEngineer", "Dependency Audit", f"Auditing {project_type} dependencies")

        # Try to find dependency file content from provided files
        dep_content = dependencies
        if not dep_content and files:
            dep_file_names = [
                "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
                "package.json", "package-lock.json",
                "Cargo.toml", "go.mod", "Gemfile", "pom.xml", "build.gradle"
            ]
            for name in dep_file_names:
                for file_path, content in files.items():
                    if file_path.endswith(name):
                        dep_content += f"\n--- {file_path} ---\n{content}\n"

        prompt = f"""Audit the dependencies for this {project_type} project.

DEPENDENCY FILE: {dependency_file if dependency_file else 'See content below'}

DEPENDENCY CONTENT:
{dep_content if dep_content else 'No dependency file found - this is itself an issue.'}

{f"PROJECT FILES: {chr(10).join(f'- {f}' for f in files.keys())}" if files else ""}

Audit each dependency for:

1. VERSION PINNING
   - Are all versions pinned to specific versions (e.g., ==2.3.0, not >=2.0)?
   - List each unpinned dependency and recommend a version.

2. SECURITY
   - Are any dependencies known to have vulnerabilities?
   - Are any dependencies abandoned/unmaintained?
   - Flag dependencies that are commonly associated with supply-chain risks.

3. NECESSITY
   - Are there dependencies that appear unused based on the project files?
   - Are there dependencies that duplicate functionality?
   - Could any be replaced with stdlib alternatives?

4. COMPATIBILITY
   - Are there any version conflicts between dependencies?
   - Are Python/Node/etc. version requirements specified?

For each issue:
- Severity: CRITICAL / HIGH / MEDIUM / LOW
- Package: name and current version
- Issue: what is wrong
- Fix: specific recommendation

SUMMARY:
- Total dependencies audited
- Pinned: X / Total
- Issues found: X critical, X high, X medium, X low
"""

        response = await self.generate_response_async(prompt)

        issues = self._extract_issues(response)
        self.found_issues.extend(issues)

        console.success(f"Dependency audit complete - {len(issues)} issues found")

        return TaskResult(
            success=True,
            output={
                "audit": response,
                "issues": issues,
                "project_type": project_type
            },
            artifacts={"dependency_audit": response}
        )

    async def _infrastructure_review(self, task: Dict[str, Any]) -> TaskResult:
        """
        Review project infrastructure setup.

        Evaluates project setup, build scripts, environment configs,
        Docker setup, and operational readiness.
        """
        files = task.get("files", {})
        project_type = task.get("project_type", "python")
        description = task.get("description", "")

        console.agent_action("DevOpsEngineer", "Infrastructure Review", description)

        file_listing = "\n".join(f"- {f}" for f in files.keys()) if files else "Not provided"
        file_contents = ""
        # Include infrastructure-relevant files
        infra_keywords = [
            "Dockerfile", "docker-compose", ".env", "Makefile", "Procfile",
            "nginx", "config", "deploy", "setup", "requirements", "package.json",
            "pyproject.toml", "Cargo.toml", "go.mod", ".yml", ".yaml", "README"
        ]
        for file_path, content in files.items():
            if any(kw.lower() in file_path.lower() for kw in infra_keywords):
                file_contents += f"\n--- {file_path} ---\n{content}\n"

        prompt = f"""Review the infrastructure setup for this {project_type} project.

PROJECT DESCRIPTION: {description}
FILES: {file_listing}

INFRASTRUCTURE FILES:
{file_contents if file_contents else 'No infrastructure files found - this is itself a finding.'}

Evaluate the following areas:

1. BUILD SYSTEM
   - Is there a clear build process?
   - Are build scripts present and correct (Makefile, npm scripts, etc.)?
   - Is the build reproducible?

2. CONTAINERIZATION (if applicable)
   - Dockerfile best practices (multi-stage builds, non-root user, etc.)
   - docker-compose configuration
   - Image size optimization

3. ENVIRONMENT MANAGEMENT
   - Environment variable handling
   - Configuration for different environments (dev, staging, prod)
   - Secrets management approach

4. OPERATIONAL READINESS
   - Health check endpoints
   - Logging configuration
   - Error reporting setup
   - Graceful shutdown handling

5. DOCUMENTATION
   - Setup instructions accuracy
   - Architecture documentation
   - Runbook/troubleshooting guides

For each finding:
- Severity: CRITICAL / HIGH / MEDIUM / LOW / INFO
- Area: which category above
- Finding: specific description
- Recommendation: concrete improvement

OVERALL ASSESSMENT:
- Infrastructure maturity: PRODUCTION_READY / STAGING_READY / DEVELOPMENT_ONLY / INCOMPLETE
- Top 3 priorities for improvement
"""

        response = await self.generate_response_async(prompt)

        issues = self._extract_issues(response)
        self.found_issues.extend(issues)

        console.success(f"Infrastructure review complete - {len(issues)} findings")

        return TaskResult(
            success=True,
            output={
                "review": response,
                "issues": issues,
                "project_type": project_type
            },
            artifacts={"infrastructure_review": response}
        )

    async def _general_task(self, task: Dict[str, Any]) -> TaskResult:
        """Handle general DevOps tasks."""
        description = task.get("description", "")

        response = await self.generate_response_async(
            f"As DevOps Engineer, please address: {description}"
        )

        return TaskResult(
            success=True,
            output={"response": response}
        )

    # ============================================================
    #  HELPER METHODS
    # ============================================================

    def _parse_verdict(self, response: str, positive: str, partial: str, negative: str) -> str:
        """Parse a verdict from LLM response."""
        upper = response.upper()
        if partial in upper:
            return partial.lower()
        elif positive in upper and negative not in upper:
            return positive.lower()
        else:
            return negative.lower()

    def _extract_issues(self, response: str) -> List[Dict[str, Any]]:
        """Extract issues from review response."""
        issues = []
        lines = response.split("\n")
        current_issue = {}

        severity_keywords = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

        for line in lines:
            line_upper = line.upper()
            for severity in severity_keywords:
                if severity in line_upper and (":" in line or "-" in line):
                    if current_issue:
                        issues.append(current_issue)
                    current_issue = {
                        "severity": severity.lower(),
                        "description": line.split(":", 1)[-1].strip() if ":" in line else line
                    }
                    break

        if current_issue:
            issues.append(current_issue)

        return issues

    # ============================================================
    #  DEVOPS-SPECIFIC METHODS
    # ============================================================

    def quick_deploy_check(self, files: Dict[str, str]) -> str:
        """Quick check if files constitute a deployable project."""
        file_list = "\n".join(f"- {f}" for f in files.keys())
        prompt = f"""
Quick deployment check. Are these files sufficient for a deployable project?

Files:
{file_list}

Check for:
1. Entry point present?
2. Dependency file present?
3. README present?

Answer: READY / NOT_READY and list what's missing (if anything). Be concise.
"""
        return self.generate_response(prompt, use_first_principles=False)

    def suggest_dockerfile(self, project_type: str, entry_point: str = "") -> str:
        """Suggest a Dockerfile for the given project type."""
        prompt = f"""
Generate a production-ready Dockerfile for a {project_type} project.
{f"Entry point: {entry_point}" if entry_point else ""}

Requirements:
- Multi-stage build if applicable
- Non-root user
- Minimal image size
- Proper .dockerignore recommendations
- Health check if applicable

Output only the Dockerfile content.
"""
        return self.generate_response(prompt, use_first_principles=False)

    def get_all_issues(self) -> List[Dict[str, Any]]:
        """Get all found issues across tasks."""
        return self.found_issues

    def clear_issues(self) -> None:
        """Clear found issues list."""
        self.found_issues.clear()
