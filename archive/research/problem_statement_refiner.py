"""
Problem Statement Refinement System

This module takes any problem (discovered, stated, or vague) and refines it into
a clear, concise, actionable problem statement that guides solution development.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import re


class ProblemClarity(Enum):
    """Problem statement clarity levels."""
    CLEAR = "clear"           # Well-defined, actionable
    MODERATE = "moderate"     # Somewhat clear but needs refinement
    VAGUE = "vague"          # Too vague, needs significant refinement
    UNCLEAR = "unclear"       # Cannot understand the problem


class ProblemType(Enum):
    """Types of problems."""
    FEATURE_REQUEST = "feature_request"     # New functionality needed
    BUG = "bug"                            # Something broken
    PERFORMANCE = "performance"             # Speed/efficiency issue
    SECURITY = "security"                   # Security vulnerability
    USABILITY = "usability"                # User experience issue
    TECHNICAL_DEBT = "technical_debt"       # Code quality/maintenance
    INTEGRATION = "integration"             # System integration issue
    DOCUMENTATION = "documentation"         # Missing/poor documentation
    UNKNOWN = "unknown"                     # Type not yet determined


@dataclass
class ProblemContext:
    """Context information about a problem."""
    domain: Optional[str] = None              # e.g., "web development", "data processing"
    technology_stack: List[str] = field(default_factory=list)  # Technologies involved
    constraints: List[str] = field(default_factory=list)       # Limitations
    stakeholders: List[str] = field(default_factory=list)      # Who is affected
    priority: str = "medium"                  # low, medium, high, critical
    existing_solutions: List[str] = field(default_factory=list)  # What's been tried


@dataclass
class RefinedProblemStatement:
    """A refined, actionable problem statement."""
    original_statement: str
    refined_statement: str
    problem_type: ProblemType
    clarity_level: ProblemClarity
    context: ProblemContext

    # Structured components
    what: str           # What is the problem?
    who: str            # Who is affected?
    where: str          # Where does it occur?
    when: str           # When does it happen?
    why: str            # Why is it a problem?
    how_much: str       # What is the impact/scope?

    # Solution guidance
    acceptance_criteria: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)

    # Metadata
    is_actionable: bool = True
    confidence: float = 1.0  # 0-1, how confident we are about the refinement
    refinement_notes: List[str] = field(default_factory=list)


class ProblemStatementRefiner:
    """
    Refines vague problem statements into clear, actionable ones.

    Process:
    1. Analyze original statement
    2. Extract key information
    3. Identify missing information
    4. Rewrite clearly and concisely
    5. Add structured components
    6. Validate actionability
    """

    def __init__(self):
        self.problem_indicators = {
            'bug': ['bug', 'error', 'broken', 'crash', 'fail', 'issue', 'not working'],
            'feature': ['need', 'want', 'add', 'create', 'build', 'implement', 'feature'],
            'performance': ['slow', 'fast', 'performance', 'optimize', 'speed', 'latency'],
            'security': ['security', 'vulnerability', 'exploit', 'breach', 'unsafe'],
            'usability': ['confusing', 'hard to use', 'ux', 'user experience', 'difficult'],
        }

    def refine(
        self,
        original_statement: str,
        context: Optional[ProblemContext] = None
    ) -> RefinedProblemStatement:
        """
        Refine a problem statement into a clear, actionable form.

        Args:
            original_statement: The original problem description
            context: Additional context about the problem

        Returns:
            RefinedProblemStatement with clear, concise statement
        """
        # Step 1: Assess current clarity
        clarity = self._assess_clarity(original_statement)

        # Step 2: Identify problem type
        problem_type = self._identify_type(original_statement)

        # Step 3: Extract structured information (5W1H)
        components = self._extract_components(original_statement)

        # Step 4: Generate refined statement
        refined = self._generate_refined_statement(
            original_statement,
            components,
            problem_type
        )

        # Step 5: Extract acceptance criteria
        criteria = self._extract_acceptance_criteria(original_statement, components)

        # Step 6: Identify constraints
        constraints = self._identify_constraints(original_statement, context)

        # Step 7: Define success metrics
        metrics = self._define_success_metrics(problem_type, components)

        # Step 8: Validate actionability
        is_actionable, notes = self._validate_actionability(refined, components)

        # Step 9: Calculate confidence
        confidence = self._calculate_confidence(clarity, components)

        return RefinedProblemStatement(
            original_statement=original_statement,
            refined_statement=refined,
            problem_type=problem_type,
            clarity_level=clarity,
            context=context or ProblemContext(),
            what=components['what'],
            who=components['who'],
            where=components['where'],
            when=components['when'],
            why=components['why'],
            how_much=components['how_much'],
            acceptance_criteria=criteria,
            constraints=constraints,
            success_metrics=metrics,
            is_actionable=is_actionable,
            confidence=confidence,
            refinement_notes=notes
        )

    def _assess_clarity(self, statement: str) -> ProblemClarity:
        """Assess how clear the problem statement is."""
        if len(statement.strip()) < 10:
            return ProblemClarity.UNCLEAR

        # Check for key components
        has_what = any(word in statement.lower() for word in ['need', 'want', 'issue', 'problem', 'broken'])
        has_context = len(statement.split()) > 15
        has_specifics = any(char.isdigit() for char in statement) or '"' in statement or "'" in statement

        if has_what and has_context and has_specifics:
            return ProblemClarity.CLEAR
        elif has_what and (has_context or has_specifics):
            return ProblemClarity.MODERATE
        elif has_what:
            return ProblemClarity.VAGUE
        else:
            return ProblemClarity.UNCLEAR

    def _identify_type(self, statement: str) -> ProblemType:
        """Identify the type of problem."""
        statement_lower = statement.lower()

        # Check against indicators
        scores = {}
        for problem_type, indicators in self.problem_indicators.items():
            score = sum(1 for indicator in indicators if indicator in statement_lower)
            if score > 0:
                scores[problem_type] = score

        if scores:
            best_match = max(scores, key=scores.get)
            type_map = {
                'bug': ProblemType.BUG,
                'feature': ProblemType.FEATURE_REQUEST,
                'performance': ProblemType.PERFORMANCE,
                'security': ProblemType.SECURITY,
                'usability': ProblemType.USABILITY,
            }
            return type_map.get(best_match, ProblemType.UNKNOWN)

        return ProblemType.UNKNOWN

    def _extract_components(self, statement: str) -> Dict[str, str]:
        """Extract 5W1H components (What, Who, Where, When, Why, How much)."""
        components = {
            'what': '',
            'who': '',
            'where': '',
            'when': '',
            'why': '',
            'how_much': ''
        }

        # Extract WHAT (the core problem)
        # Look for main verb + object patterns
        what_patterns = [
            r'(need to|want to|should|must|have to)\s+([^.!?]+)',
            r'(is|are|was|were)\s+(not working|broken|slow|missing|incorrect)',
            r'(cannot|can\'t|unable to)\s+([^.!?]+)',
        ]

        for pattern in what_patterns:
            match = re.search(pattern, statement, re.IGNORECASE)
            if match:
                components['what'] = match.group(0).strip()
                break

        if not components['what']:
            # Fallback: first sentence
            sentences = statement.split('.')
            components['what'] = sentences[0].strip() if sentences else statement

        # Extract WHO (users, systems, stakeholders)
        who_patterns = [
            r'(users?|customers?|clients?|developers?|team|system|service|application)',
            r'(for|by|to)\s+([\w\s]+?)\s+(need|want|require)',
        ]

        for pattern in who_patterns:
            match = re.search(pattern, statement, re.IGNORECASE)
            if match:
                components['who'] = match.group(1).strip() if match.lastindex == 1 else match.group(2).strip()
                break

        if not components['who']:
            components['who'] = "users"  # Default assumption

        # Extract WHERE (location, system, module)
        where_patterns = [
            r'in (the )?([\w\s]+?)(system|module|component|service|file|function)',
            r'on (the )?([\w\s]+?)(page|screen|endpoint|server)',
        ]

        for pattern in where_patterns:
            match = re.search(pattern, statement, re.IGNORECASE)
            if match:
                components['where'] = match.group(0).strip()
                break

        # Extract WHEN (timing, frequency)
        when_patterns = [
            r'(always|sometimes|occasionally|frequently|rarely|never)',
            r'when ([\w\s]+)',
            r'(during|after|before) ([\w\s]+)',
        ]

        for pattern in when_patterns:
            match = re.search(pattern, statement, re.IGNORECASE)
            if match:
                components['when'] = match.group(0).strip()
                break

        # Extract WHY (business impact, reason)
        why_patterns = [
            r'(because|since|as) ([^.!?]+)',
            r'(to|in order to|so that) ([^.!?]+)',
            r'(causing|resulting in|leading to) ([^.!?]+)',
        ]

        for pattern in why_patterns:
            match = re.search(pattern, statement, re.IGNORECASE)
            if match:
                components['why'] = match.group(0).strip()
                break

        # Extract HOW MUCH (scope, scale, impact)
        howmuch_patterns = [
            r'(\d+)\s*(users?|requests?|times?|percent|%)',
            r'(all|many|most|some|few) (users?|customers?|requests?)',
            r'(critical|major|minor|significant|negligible) (impact|issue|problem)',
        ]

        for pattern in howmuch_patterns:
            match = re.search(pattern, statement, re.IGNORECASE)
            if match:
                components['how_much'] = match.group(0).strip()
                break

        return components

    def _generate_refined_statement(
        self,
        original: str,
        components: Dict[str, str],
        problem_type: ProblemType
    ) -> str:
        """Generate a clear, concise refined statement."""
        # Template based on problem type
        templates = {
            ProblemType.BUG: "{who} experience {what} {where} {when}, causing {why}.",
            ProblemType.FEATURE_REQUEST: "{who} need {what} {where} to {why}.",
            ProblemType.PERFORMANCE: "{what} {where} is too slow {when}, impacting {who} because {why}.",
            ProblemType.SECURITY: "{what} {where} creates a security vulnerability affecting {who}.",
            ProblemType.USABILITY: "{who} find {what} {where} difficult to use because {why}.",
        }

        template = templates.get(problem_type, "{who} face {what} {where}.")

        # Fill in template
        refined = template.format(
            what=components['what'] or "an issue",
            who=components['who'] or "users",
            where=components['where'] or "in the system",
            when=components['when'] or "consistently",
            why=components['why'] or "it prevents normal operation"
        )

        # Clean up
        refined = re.sub(r'\s+', ' ', refined)  # Remove extra spaces
        refined = re.sub(r'\s+([,.])', r'\1', refined)  # Fix punctuation spacing

        # If original is already very clear and concise, keep it
        if len(original) < 200 and original.count('.') <= 2:
            return original

        return refined

    def _extract_acceptance_criteria(
        self,
        statement: str,
        components: Dict[str, str]
    ) -> List[str]:
        """Extract or generate acceptance criteria."""
        criteria = []

        # Look for explicit criteria
        criteria_patterns = [
            r'should ([\w\s]+)',
            r'must ([\w\s]+)',
            r'needs? to ([\w\s]+)',
            r'will ([\w\s]+)',
        ]

        for pattern in criteria_patterns:
            matches = re.findall(pattern, statement, re.IGNORECASE)
            criteria.extend([m.strip() for m in matches])

        # Generate default criteria if none found
        if not criteria:
            if components['what']:
                criteria.append(f"Resolve: {components['what']}")
            if components['who']:
                criteria.append(f"Works for: {components['who']}")
            criteria.append("No new bugs introduced")
            criteria.append("Passes all tests")

        return criteria[:5]  # Limit to 5 most important

    def _identify_constraints(
        self,
        statement: str,
        context: Optional[ProblemContext]
    ) -> List[str]:
        """Identify constraints."""
        constraints = []

        # Extract from statement
        constraint_patterns = [
            r'(without|must not|cannot) ([\w\s]+)',
            r'(within|under) (\d+) (days?|weeks?|hours?|minutes?)',
            r'(limited to|restricted to|only) ([\w\s]+)',
        ]

        for pattern in constraint_patterns:
            matches = re.findall(pattern, statement, re.IGNORECASE)
            constraints.extend([' '.join(m) for m in matches])

        # Add context constraints
        if context and context.constraints:
            constraints.extend(context.constraints)

        return constraints[:5]

    def _define_success_metrics(
        self,
        problem_type: ProblemType,
        components: Dict[str, str]
    ) -> List[str]:
        """Define success metrics."""
        metrics = []

        type_metrics = {
            ProblemType.BUG: [
                "Zero occurrences of the bug",
                "All affected test cases pass",
                "No regression in related functionality"
            ],
            ProblemType.FEATURE_REQUEST: [
                "Feature implemented and tested",
                "User acceptance criteria met",
                "Documentation updated"
            ],
            ProblemType.PERFORMANCE: [
                "Performance target met (response time, throughput)",
                "No degradation in other areas",
                "Load testing passed"
            ],
            ProblemType.SECURITY: [
                "Vulnerability patched",
                "Security audit passed",
                "No new vulnerabilities introduced"
            ],
        }

        metrics = type_metrics.get(problem_type, [
            "Problem resolved as stated",
            "Solution tested and verified",
            "Stakeholders satisfied"
        ])

        return metrics

    def _validate_actionability(
        self,
        refined: str,
        components: Dict[str, str]
    ) -> tuple[bool, List[str]]:
        """Validate that the statement is actionable."""
        notes = []
        is_actionable = True

        # Check for key components
        if not components['what']:
            is_actionable = False
            notes.append("Missing clear description of WHAT the problem is")

        if not components['who']:
            notes.append("WHO is affected is unclear (defaulting to 'users')")

        if not components['where']:
            notes.append("WHERE the problem occurs is unclear")

        # Check statement length
        if len(refined) < 20:
            is_actionable = False
            notes.append("Statement too short - lacks detail")

        if len(refined) > 500:
            notes.append("Statement quite long - consider simplifying")

        # Check for vague terms
        vague_terms = ['somehow', 'something', 'maybe', 'possibly', 'unclear']
        if any(term in refined.lower() for term in vague_terms):
            notes.append("Contains vague language - needs more specificity")

        return is_actionable, notes

    def _calculate_confidence(
        self,
        clarity: ProblemClarity,
        components: Dict[str, str]
    ) -> float:
        """Calculate confidence in the refinement."""
        base_confidence = {
            ProblemClarity.CLEAR: 0.9,
            ProblemClarity.MODERATE: 0.7,
            ProblemClarity.VAGUE: 0.5,
            ProblemClarity.UNCLEAR: 0.3,
        }

        confidence = base_confidence[clarity]

        # Adjust based on component completeness
        filled_components = sum(1 for v in components.values() if v)
        confidence += (filled_components / len(components)) * 0.1

        return min(confidence, 1.0)

    def format_refined_statement(self, refined: RefinedProblemStatement) -> str:
        """Format refined statement for display."""
        output = []
        output.append("="*80)
        output.append("REFINED PROBLEM STATEMENT")
        output.append("="*80)
        output.append("")

        output.append("ORIGINAL:")
        output.append(f"  {refined.original_statement}")
        output.append("")

        output.append("REFINED:")
        output.append(f"  {refined.refined_statement}")
        output.append("")

        output.append(f"TYPE: {refined.problem_type.value}")
        output.append(f"CLARITY: {refined.clarity_level.value}")
        output.append(f"ACTIONABLE: {'Yes' if refined.is_actionable else 'No'}")
        output.append(f"CONFIDENCE: {refined.confidence:.0%}")
        output.append("")

        output.append("STRUCTURED ANALYSIS:")
        output.append(f"  WHAT: {refined.what}")
        output.append(f"  WHO:  {refined.who}")
        output.append(f"  WHERE: {refined.where}")
        output.append(f"  WHEN: {refined.when}")
        output.append(f"  WHY:  {refined.why}")
        output.append(f"  SCOPE: {refined.how_much}")
        output.append("")

        if refined.acceptance_criteria:
            output.append("ACCEPTANCE CRITERIA:")
            for i, criterion in enumerate(refined.acceptance_criteria, 1):
                output.append(f"  {i}. {criterion}")
            output.append("")

        if refined.constraints:
            output.append("CONSTRAINTS:")
            for constraint in refined.constraints:
                output.append(f"  - {constraint}")
            output.append("")

        if refined.success_metrics:
            output.append("SUCCESS METRICS:")
            for metric in refined.success_metrics:
                output.append(f"  ✓ {metric}")
            output.append("")

        if refined.refinement_notes:
            output.append("NOTES:")
            for note in refined.refinement_notes:
                output.append(f"  ! {note}")
            output.append("")

        output.append("="*80)

        return "\n".join(output)


# Convenience function
def refine_problem(statement: str) -> str:
    """Quick problem refinement."""
    refiner = ProblemStatementRefiner()
    refined = refiner.refine(statement)
    return refiner.format_refined_statement(refined)


# Example usage
if __name__ == "__main__":
    # Test with various problem statements
    test_statements = [
        "Users can't login",
        "The app is slow and users are complaining about poor performance during peak hours",
        "We need authentication for our API to prevent unauthorized access",
        "There's a bug somewhere in the checkout process that causes some orders to fail",
    ]

    refiner = ProblemStatementRefiner()

    for statement in test_statements:
        print(refine_problem(statement))
        print("\n")
