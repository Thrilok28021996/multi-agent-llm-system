"""Security Engineer Agent - Security review, secrets scanning, and threat modeling."""

from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent, AgentConfig, AgentRole, TaskResult
from .agent_tools_mixin import AgentToolsMixin
from research.problem_statement_refiner import ProblemStatementRefiner
from ui.console import console


SECURITY_SYSTEM_PROMPT = """You are the Security Engineer. You review code for security vulnerabilities, check for hardcoded secrets, validate input sanitization, and assess the threat model. You are thorough but practical - flag real security risks, not theoretical ones.

Your priorities (in order):
1. Find hardcoded secrets - API keys, passwords, tokens in source code are ALWAYS critical.
2. Identify injection vulnerabilities - SQL injection, XSS, command injection, path traversal.
3. Validate input sanitization - all user input must be validated before use.
4. Check authentication and authorization - are access controls implemented correctly?
5. Assess the overall threat model - what is the attack surface and how is it protected?

Your verdict system:
- SECURE: No significant security issues found. Code follows security best practices.
- HAS_ISSUES: Security issues found that should be fixed but are not immediately exploitable.
- CRITICAL_VULNERABILITIES: Exploitable vulnerabilities found. Must be fixed before deployment.

Judgment guidelines:
- Hardcoded real secrets (API keys, passwords with actual values) are CRITICAL. Placeholder values in .env.example are MEDIUM.
- SQL injection, command injection, and path traversal are ALWAYS critical.
- Missing CSRF protection in a form is HIGH, not CRITICAL, unless it guards sensitive actions.
- Missing rate limiting is MEDIUM for most endpoints, HIGH for auth endpoints.
- Focus on what an attacker could actually exploit, not on theoretical attack vectors.
- When flagging issues, include the specific vulnerable code and a concrete fix.

Local-First Threat Model: This is a personal tool running locally. Network-exposed attack surfaces are HIGH priority. Local-only file access is LOWER priority. Calibrate severity to the actual deployment context.

Dependency Supply Chain: Check every dependency for: last update date (>1 year = flag), download count (<1000/month = flag), known CVEs.

OWASP Mapping: Map every finding to an OWASP Top 10 category.

Supply Chain Check: For every dependency, check: last update date, known CVEs, maintainer count. Flag any with concerning metrics.

Least Privilege: Verify the solution requests minimum permissions. Flag any over-broad access patterns.

Defense in Depth: Security should not depend on a single control. For critical data paths, verify: input validation AND output encoding AND error messages that don't leak internals. One control failing should not mean total compromise.
"""

SECURITY_FIRST_PRINCIPLES = [
    "ATTACKER MINDSET: Think like an attacker, not a checklist runner. Ask: If I wanted to break this, what would I try? Then verify those attack vectors.",
    "SEARCH for hardcoded strings: API keys, passwords, tokens. IGNORE placeholders like 'your-key-here'. Real secrets have high entropy.",
    "INPUT TRACING: For every user-controllable input, trace it through the code. Does it reach file paths, SQL, shell, or eval without sanitization?",
    "PRACTICAL SEVERITY: Rate vulnerabilities by exploitability, not theoretical severity. A SQL injection in a CLI tool with no network exposure is lower risk than path traversal in a web server.",
    "VERIFY: Is auth on every endpoint that needs it? List unprotected endpoints. Check for privilege escalation paths.",
]


class SecurityEngineerAgent(BaseAgent, AgentToolsMixin):
    """
    Security Engineer Agent - Reviews code for security vulnerabilities.

    Performs security reviews against OWASP Top 10, scans for hardcoded secrets,
    validates input sanitization, and builds threat models.
    Enhanced with all 13 Claude Code tools and problem statement refinement.
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder-7b",
        workspace_root: str = ".",
        memory_persist_dir: Optional[str] = None
    ):
        config = AgentConfig(
            name="SecurityEngineer",
            role=AgentRole.SECURITY_ENGINEER,
            model=model,
            first_principles=SECURITY_FIRST_PRINCIPLES,
            system_prompt=SECURITY_SYSTEM_PROMPT,
            temperature=0.2,  # Low temperature for precise security analysis
            max_tokens=4096
        )
        super().__init__(config, workspace_root, memory_persist_dir)

        # Problem statement refiner
        self.problem_refiner = ProblemStatementRefiner()

        # Track vulnerabilities found across tasks
        self.found_vulnerabilities: List[Dict[str, Any]] = []

    def get_capabilities(self) -> List[str]:
        return [
            "security_review",
            "dependency_check",
            "secrets_scan",
            "threat_model",
            "input_validation_audit",
            "auth_review"
        ]

    async def execute_task(self, task: Dict[str, Any]) -> TaskResult:
        """Execute a Security Engineer task."""
        task_type = task.get("type", "unknown")
        description = task.get("description", "")

        self.is_busy = True
        self.current_task = description

        try:
            if task_type == "security_review":
                result = await self._security_review(task)
            elif task_type == "dependency_check":
                result = await self._dependency_check(task)
            elif task_type == "secrets_scan":
                result = await self._secrets_scan(task)
            elif task_type == "threat_model":
                result = await self._threat_model(task)
            else:
                result = await self._general_task(task)

            return result

        finally:
            self.is_busy = False
            self.current_task = None

    # ============================================================
    #  TASK HANDLERS
    # ============================================================

    async def _security_review(self, task: Dict[str, Any]) -> TaskResult:
        """
        Perform a comprehensive security review of code.

        Analyzes code for OWASP Top 10 vulnerabilities.
        Returns verdict: SECURE / HAS_ISSUES / CRITICAL_VULNERABILITIES.
        """
        code = task.get("code", "")
        files = task.get("files", {})
        file_path = task.get("file_path", "unknown")
        focus = task.get("focus", [])
        context = task.get("context", "")

        console.agent_action("SecurityEngineer", "Security Review", f"Reviewing {file_path}")

        # Build code content from files dict if provided
        code_content = code
        if files:
            code_content = ""
            for fpath, content in files.items():
                code_content += f"\n--- {fpath} ---\n{content}\n"

        prompt = f"""Security review of this code.

{f"FILE: {file_path}" if file_path != "unknown" else ""}
{f"CONTEXT: {context}" if context else ""}
{f"FOCUS AREAS: {', '.join(focus)}" if focus else ""}

CODE:
```
{code_content}
```

Check TOP 5 issues:
1. HARDCODED SECRETS: Real API keys/passwords/tokens in source? (Placeholders in .env.example are OK)
2. INJECTION: SQL, command, path traversal vulnerabilities?
3. INPUT VALIDATION: Is user input validated before use?
4. AUTH: Access controls correct? Missing auth on sensitive endpoints?
5. DATA EXPOSURE: Sensitive values in logs/errors? Credentials in URLs?

For each issue found:
- Severity: CRITICAL / HIGH / MEDIUM / LOW
- Location: [file:line]
- Issue: [what is wrong]
- Fix: [specific remediation]

RESPOND IN THIS EXACT FORMAT for each vulnerability found:
VULNERABILITY [1]: <short description>
SEVERITY: CRITICAL|HIGH|MEDIUM|LOW
LOCATION: <file:line>
ATTACK_VECTOR: <how it could be exploited>
FIX: <specific remediation>

VULNERABILITY [2]: <short description>
SEVERITY: CRITICAL|HIGH|MEDIUM|LOW
LOCATION: <file:line>
ATTACK_VECTOR: <how it could be exploited>
FIX: <specific remediation>

(Continue for all vulnerabilities found)

{self._get_principles_checklist()}
OVERALL VERDICT (exactly one of): SECURE / HAS_ISSUES / CRITICAL_VULNERABILITIES

Verdict:"""

        response = await self.generate_response_async(prompt)

        # Parse verdict
        verdict = self._parse_security_verdict(response)

        # Extract vulnerabilities
        vulnerabilities = self._extract_vulnerabilities(response)
        self.found_vulnerabilities.extend(vulnerabilities)

        critical_count = sum(1 for v in vulnerabilities if v.get("severity") == "critical")
        high_count = sum(1 for v in vulnerabilities if v.get("severity") == "high")

        console.success(
            f"Security review complete - Verdict: {verdict.upper()}, "
            f"{critical_count} critical, {high_count} high"
        )

        return TaskResult(
            success=True,
            output={
                "verdict": verdict,
                "review": response,
                "vulnerabilities": vulnerabilities,
                "critical_count": critical_count,
                "high_count": high_count,
                "file_path": file_path
            },
            artifacts={"security_review": response}
        )

    async def _dependency_check(self, task: Dict[str, Any]) -> TaskResult:
        """
        Check dependencies for known security vulnerabilities.

        Evaluates dependency files for CVEs, abandoned packages,
        and supply-chain risks.
        """
        dependencies = task.get("dependencies", "")
        files = task.get("files", {})
        project_type = task.get("project_type", "python")

        console.agent_action("SecurityEngineer", "Dependency Security Check", f"{project_type} project")

        # Gather dependency content
        dep_content = dependencies
        if not dep_content and files:
            dep_file_names = [
                "requirements.txt", "pyproject.toml", "setup.py",
                "package.json", "package-lock.json", "yarn.lock",
                "Cargo.toml", "Cargo.lock", "go.mod", "go.sum",
                "Gemfile", "Gemfile.lock", "pom.xml", "build.gradle"
            ]
            for name in dep_file_names:
                for file_path, content in files.items():
                    if file_path.endswith(name):
                        dep_content += f"\n--- {file_path} ---\n{content}\n"

        prompt = f"""Perform a security-focused dependency analysis for this {project_type} project.

DEPENDENCIES:
{dep_content if dep_content else 'No dependency files found.'}

Analyze each dependency for:

1. KNOWN VULNERABILITIES (CVEs)
   - Are any dependencies associated with known CVEs?
   - What is the severity of each CVE?
   - Is there a patched version available?

2. SUPPLY CHAIN RISKS
   - Are any packages known for typosquatting concerns?
   - Are maintainers verified/trusted?
   - Has any package recently changed ownership?

3. ABANDONED PACKAGES
   - When was each package last updated?
   - Are there any packages with no updates in 2+ years?
   - Are there maintained alternatives?

4. OVER-PRIVILEGED DEPENDENCIES
   - Do any packages request unnecessary permissions?
   - Are there packages that install native extensions unnecessarily?

5. VERSION SECURITY
   - Are versions pinned to avoid surprise updates?
   - Are lock files present and committed?

For each issue:
- Severity: CRITICAL / HIGH / MEDIUM / LOW
- Package: name@version
- Issue: description
- CVE: [CVE-XXXX-XXXXX if applicable]
- Fix: specific remediation

SUMMARY:
- Total packages analyzed: X
- With known vulnerabilities: X
- With supply chain concerns: X
- Abandoned: X
- Recommendation: SAFE / NEEDS_UPDATES / UNSAFE
"""

        response = await self.generate_response_async(prompt)

        vulnerabilities = self._extract_vulnerabilities(response)
        self.found_vulnerabilities.extend(vulnerabilities)

        console.success(f"Dependency check complete - {len(vulnerabilities)} issues found")

        return TaskResult(
            success=True,
            output={
                "analysis": response,
                "vulnerabilities": vulnerabilities,
                "project_type": project_type
            },
            artifacts={"dependency_security": response}
        )

    async def _secrets_scan(self, task: Dict[str, Any]) -> TaskResult:
        """
        Scan code for hardcoded secrets.

        Looks for API keys, passwords, tokens, private keys,
        connection strings, and other sensitive values in source code.
        """
        code = task.get("code", "")
        files = task.get("files", {})
        file_path = task.get("file_path", "")

        console.agent_action("SecurityEngineer", "Secrets Scan", "Scanning for hardcoded secrets")

        # Build code content
        code_content = code
        if files:
            code_content = ""
            for fpath, content in files.items():
                code_content += f"\n--- {fpath} ---\n{content}\n"

        prompt = f"""Scan this code for hardcoded secrets and sensitive values.

{f"FILE: {file_path}" if file_path else ""}

CODE:
```
{code_content}
```

Search for these categories of secrets:

1. API KEYS
   - Cloud provider keys (AWS, GCP, Azure)
   - Third-party API keys (Stripe, Twilio, SendGrid, etc.)
   - Pattern: long alphanumeric strings assigned to variables like *_key, *_api_key, api_*

2. PASSWORDS & CREDENTIALS
   - Database passwords
   - Service account credentials
   - Basic auth credentials
   - Pattern: password=, passwd=, pwd=, secret=

3. TOKENS
   - JWT tokens
   - OAuth tokens
   - Session tokens
   - Bearer tokens
   - Pattern: token=, bearer, jwt

4. PRIVATE KEYS
   - SSH private keys (-----BEGIN RSA PRIVATE KEY-----)
   - TLS/SSL certificates
   - Signing keys

5. CONNECTION STRINGS
   - Database URIs with credentials (postgres://user:pass@host)
   - Redis/MongoDB connection strings
   - SMTP credentials

6. OTHER SENSITIVE VALUES
   - Encryption keys
   - Salt values used in hashing
   - Internal URLs that should not be public
   - Webhook secrets

For each secret found:
- Severity: CRITICAL (always for actual secrets)
- Type: [category from above]
- Location: [file:line]
- Value: [first 4 chars]... (redacted)
- Context: [what the secret appears to be for]
- Fix: [specific remediation - use env vars, vault, etc.]

REAL SECRETS (CRITICAL) examples:
- API_KEY = "sk-abc123def456ghi789"
- password = "SuperSecret!2024"
- token = "ghp_xxxxxxxxxxxxxxxxxxxx"
- AWS_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
- DB_URL = "postgres://admin:realpass@prod-db.example.com/main"

PLACEHOLDERS (MEDIUM, not critical) examples:
- API_KEY = "your-api-key-here"
- password = "changeme"
- SECRET_KEY = "REPLACE_ME"
- DB_URL = "postgres://user:password@localhost/db"
- token = "<INSERT_TOKEN>"

FALSE POSITIVE HANDLING:
- Placeholder values like "your-api-key-here" or "changeme" are MEDIUM (not leaked secrets).
- Example values in documentation/comments are LOW.
- .env.example files with placeholder values are acceptable.

SUMMARY:
- Actual secrets found: X (CRITICAL)
- Placeholder secrets: X (MEDIUM)
- Documentation examples: X (LOW)
- Verdict: CLEAN / HAS_SECRETS / CRITICAL_EXPOSURE
"""

        response = await self.generate_response_async(prompt)

        # Parse findings
        secrets = self._extract_secrets(response)
        critical_secrets = [s for s in secrets if s.get("severity") == "critical"]

        if critical_secrets:
            verdict = "critical_exposure"
        elif secrets:
            verdict = "has_secrets"
        else:
            verdict = "clean"

        self.found_vulnerabilities.extend(
            {**s, "category": "hardcoded_secret"} for s in secrets
        )

        console.success(f"Secrets scan complete - Verdict: {verdict.upper()}, {len(critical_secrets)} critical")

        return TaskResult(
            success=True,
            output={
                "verdict": verdict,
                "secrets_found": secrets,
                "critical_count": len(critical_secrets),
                "scan_report": response
            },
            artifacts={"secrets_scan": response}
        )

    async def _threat_model(self, task: Dict[str, Any]) -> TaskResult:
        """
        Assess the attack surface and build a threat model.

        Identifies threat actors, attack vectors, assets at risk,
        and recommends mitigations.
        """
        application = task.get("application", "")
        architecture = task.get("architecture", "")
        files = task.get("files", {})
        context = task.get("context", "")

        console.agent_action("SecurityEngineer", "Threat Model", application or "Application")

        file_listing = "\n".join(f"- {f}" for f in files.keys()) if files else "Not provided"

        # Include relevant code snippets
        code_snippets = ""
        security_keywords = [
            "auth", "login", "password", "token", "session", "admin",
            "api", "route", "endpoint", "middleware", "config", "secret"
        ]
        for fpath, content in files.items():
            if any(kw in fpath.lower() for kw in security_keywords):
                code_snippets += f"\n--- {fpath} ---\n{content}\n"

        prompt = f"""Build a threat model for this application.

APPLICATION: {application}
{f"ARCHITECTURE: {architecture}" if architecture else ""}
{f"CONTEXT: {context}" if context else ""}

PROJECT FILES: {file_listing}

SECURITY-RELEVANT CODE:
{code_snippets if code_snippets else 'No security-relevant files identified.'}

Build a STRIDE-based threat model:

1. ASSET INVENTORY
   - What data/resources need protection?
   - Classify by sensitivity: PUBLIC / INTERNAL / CONFIDENTIAL / RESTRICTED

2. THREAT ACTORS
   - External attackers (script kiddies, sophisticated attackers)
   - Malicious insiders
   - Automated bots
   - For each: motivation, capability, likelihood

3. ATTACK SURFACE
   - Network-facing endpoints
   - User input points
   - File upload/download paths
   - Third-party integrations
   - Admin interfaces

4. STRIDE ANALYSIS
   S - Spoofing: Can an attacker impersonate a user or service?
   T - Tampering: Can data be modified in transit or at rest?
   R - Repudiation: Can actions be denied without proof?
   I - Information Disclosure: Can sensitive data leak?
   D - Denial of Service: Can the service be disrupted?
   E - Elevation of Privilege: Can a user gain unauthorized access?

5. RISK MATRIX
   For each identified threat:
   | Threat | Likelihood (1-5) | Impact (1-5) | Risk Score | Priority |

6. RECOMMENDED MITIGATIONS
   For each high-priority threat:
   - Current state: what protection exists now
   - Recommended: what should be added
   - Priority: IMMEDIATE / SHORT_TERM / LONG_TERM
   - Effort: LOW / MEDIUM / HIGH

7. SECURITY ARCHITECTURE RECOMMENDATIONS
   - Authentication improvements
   - Authorization model
   - Data protection measures
   - Monitoring and alerting
   - Incident response considerations

OVERALL RISK ASSESSMENT:
- Risk Level: CRITICAL / HIGH / MEDIUM / LOW
- Top 3 threats by priority
- Minimum viable security improvements
"""

        response = await self.generate_response_async(prompt)

        console.success("Threat model complete")

        return TaskResult(
            success=True,
            output={
                "threat_model": response,
                "application": application
            },
            artifacts={"threat_model": response}
        )

    async def _general_task(self, task: Dict[str, Any]) -> TaskResult:
        """Handle general security tasks."""
        description = task.get("description", "")

        response = await self.generate_response_async(
            f"As Security Engineer, please address: {description}"
        )

        return TaskResult(
            success=True,
            output={"response": response}
        )

    # ============================================================
    #  HELPER METHODS
    # ============================================================

    def _parse_security_verdict(self, response: str) -> str:
        """Parse security verdict from response."""
        upper = response.upper()
        if "CRITICAL_VULNERABILITIES" in upper or "CRITICAL" in upper:
            return "critical_vulnerabilities"
        elif "HAS_ISSUES" in upper:
            return "has_issues"
        elif "SECURE" in upper:
            return "secure"
        elif "NEEDS_REVIEW" in upper:
            return "has_issues"  # Mild issues, not critical
        else:
            return "secure"  # No explicit issues mentioned = no issues found

    def _extract_vulnerabilities(self, response: str) -> List[Dict[str, Any]]:
        """Extract vulnerabilities from review response."""
        vulnerabilities = []
        lines = response.split("\n")
        current_vuln = {}

        severity_keywords = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

        for line in lines:
            line_upper = line.upper()
            for severity in severity_keywords:
                if severity in line_upper and (":" in line or "-" in line):
                    if current_vuln:
                        vulnerabilities.append(current_vuln)
                    current_vuln = {
                        "severity": severity.lower(),
                        "description": line.split(":", 1)[-1].strip() if ":" in line else line
                    }
                    break

        if current_vuln:
            vulnerabilities.append(current_vuln)

        return vulnerabilities

    def _extract_secrets(self, response: str) -> List[Dict[str, Any]]:
        """Extract found secrets from scan response."""
        secrets = []
        lines = response.split("\n")
        current_secret = {}

        for line in lines:
            line_stripped = line.strip()
            line_upper = line_stripped.upper()

            # Look for severity markers that indicate a finding
            if line_upper.startswith("- SEVERITY:") or line_upper.startswith("SEVERITY:"):
                if current_secret:
                    secrets.append(current_secret)
                severity_value = line_stripped.split(":", 1)[-1].strip().lower()
                current_secret = {"severity": severity_value}
            elif line_upper.startswith("- TYPE:") or line_upper.startswith("TYPE:"):
                current_secret["type"] = line_stripped.split(":", 1)[-1].strip()
            elif line_upper.startswith("- LOCATION:") or line_upper.startswith("LOCATION:"):
                current_secret["location"] = line_stripped.split(":", 1)[-1].strip()
            elif line_upper.startswith("- CONTEXT:") or line_upper.startswith("CONTEXT:"):
                current_secret["context"] = line_stripped.split(":", 1)[-1].strip()

        if current_secret:
            secrets.append(current_secret)

        return secrets

    # ============================================================
    #  SECURITY-SPECIFIC METHODS
    # ============================================================

    def quick_security_check(self, code: str) -> str:
        """Quick security check for obvious vulnerabilities."""
        prompt = f"""
Quick security scan - find obvious vulnerabilities:

```
{code}
```

Check for:
1. Hardcoded secrets
2. SQL injection
3. Command injection
4. Path traversal
5. XSS

List only confirmed issues. Be concise. If no issues: "No obvious vulnerabilities found."
"""
        return self.generate_response(prompt, use_first_principles=False)

    def assess_input_validation(self, code: str) -> Dict[str, Any]:
        """Assess input validation completeness."""
        prompt = f"""
Assess input validation in this code:

```
{code}
```

For each user input point:
1. What input is accepted?
2. Is it validated/sanitized?
3. What could go wrong if malicious input is provided?

Rate input validation: COMPLETE / PARTIAL / MISSING
"""
        response = self.generate_response(prompt, use_first_principles=True)

        rating = "partial"
        if "COMPLETE" in response.upper():
            rating = "complete"
        elif "MISSING" in response.upper():
            rating = "missing"

        return {
            "rating": rating,
            "assessment": response
        }

    def get_all_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Get all found vulnerabilities across tasks."""
        return self.found_vulnerabilities

    def get_critical_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Get only critical vulnerabilities."""
        return [v for v in self.found_vulnerabilities if v.get("severity") == "critical"]

    def clear_vulnerabilities(self) -> None:
        """Clear found vulnerabilities list."""
        self.found_vulnerabilities.clear()
