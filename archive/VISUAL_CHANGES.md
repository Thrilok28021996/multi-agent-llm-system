# Visual Changes — Before & After

## Fix 1: Department Member

**File**: `company/organization.py:54`

```diff
  "engineering": DepartmentInfo(
      head="CTO",
-     members=["CTO", "Developer", "DevOps", "SecurityEngineer"],
+     members=["CTO", "Developer", "DevOpsEngineer", "SecurityEngineer"],
      responsibilities=["Architecture", "Implementation", "Deployment", "Security"],
      decision_authority="technical"
  ),
```

---

## Fix 2: OKR Dictionary Keys

**File**: `company/performance.py:21-60`

```diff
  DEFAULT_OKRS: Dict[str, AgentOKR] = {
-     "ceo": AgentOKR(
+     "CEO": AgentOKR(
          objective="Ensure every shipped solution solves the stated problem",
          key_results=["Approval based on evidence not assumption", "< 20% rejection rate"]
      ),
-     "cto": AgentOKR(
+     "CTO": AgentOKR(
          objective="Design architectures that developers can build on first try",
          key_results=["< 2 redesigns per project", "Developer asks 0 clarifying questions"]
      ),
-     "product_manager": AgentOKR(
+     "ProductManager": AgentOKR(
          objective="Define requirements so clear that there is zero ambiguity",
          key_results=["Acceptance criteria are testable commands", "< 1 rescope per project"]
      ),
-     "researcher": AgentOKR(
+     "Researcher": AgentOKR(
          objective="Find real problems backed by data from multiple sources",
          key_results=["> 3 sources per problem", "Cross-validation score > 0.7"]
      ),
-     "developer": AgentOKR(
+     "Developer": AgentOKR(
          objective="Write code that passes QA on the first attempt",
          key_results=["< 20% rework rate", "All files are runnable"]
      ),
-     "qa_engineer": AgentOKR(
+     "QAEngineer": AgentOKR(
          objective="Catch real bugs, not style issues",
          key_results=["< 10% false positive rate", "Every FAIL has a specific blocking issue"]
      ),
-     "devops_engineer": AgentOKR(
+     "DevOpsEngineer": AgentOKR(
          objective="Every solution is deployable with one command",
          key_results=["Entry point exists", "Dependencies are pinned"]
      ),
-     "data_analyst": AgentOKR(
+     "DataAnalyst": AgentOKR(
          objective="Provide unbiased cross-validation of research findings",
          key_results=["Detect > 80% of biased sources", "Confidence scores within 0.1 of actual"]
      ),
-     "security_engineer": AgentOKR(
+     "SecurityEngineer": AgentOKR(
          objective="Identify real security vulnerabilities, not false alarms",
          key_results=["Zero false positives on placeholder values", "Catch all hardcoded real secrets"]
      ),
  }
```

---

## Fix 3: Progress Phase IDs

**File**: `orchestrator/workflow.py:157-169` (and line 4732-4750)

```diff
  self.progress.add_phases([
      {"id": "research", "name": "Research"},
      {"id": "data_analysis", "name": "Data Analysis"},
      {"id": "analysis", "name": "Analysis"},
-     {"id": "opportunity", "name": "Opportunity Eval"},
-     {"id": "design", "name": "Technical Design"},
+     {"id": "opportunity_evaluation", "name": "Opportunity Eval"},
+     {"id": "technical_design", "name": "Technical Design"},
      {"id": "design_review", "name": "Design Review"},
      {"id": "implementation", "name": "Implementation"},
      {"id": "code_execution", "name": "Code Execution"},
      {"id": "qa_validation", "name": "QA Validation"},
      {"id": "security_review", "name": "Security Review"},
      {"id": "ceo_approval", "name": "CEO Approval"},
      {"id": "delivery", "name": "Delivery"},
      {"id": "retrospective", "name": "Retrospective"},
  ])
```

**File**: `orchestrator/workflow.py` (7 method calls)

```diff
- self.progress.skip_phase("opportunity")
+ self.progress.skip_phase("opportunity_evaluation")

- self.progress.start_phase("opportunity")
+ self.progress.start_phase("opportunity_evaluation")

- self.progress.complete_phase("opportunity")
+ self.progress.complete_phase("opportunity_evaluation")

- self.progress.fail_phase("opportunity", "Rejected by CEO after pivot")
+ self.progress.fail_phase("opportunity_evaluation", "Rejected by CEO after pivot")

- self.progress.skip_phase("design")
+ self.progress.skip_phase("technical_design")

- self.progress.start_phase("design")
+ self.progress.start_phase("technical_design")

- self.progress.complete_phase("design")
+ self.progress.complete_phase("technical_design")
```

---

## Fix 4: Model Defaults

**File**: `config/config_loader.py:60-68`

```diff
  @dataclass
  class AgentModels:
      """Model assignments for each agent.
  
      Defaults match config/models.py MODEL_CONFIGS (optimized for 16GB RAM).
      """
-     ceo: str = "qwen3-8b"
-     cto: str = "qwen3-8b"
-     product_manager: str = "ministral-8b"
-     researcher: str = "ministral-8b"
-     developer: str = "qwen2.5-coder-7b"
-     qa_engineer: str = "qwen2.5-coder-7b"
-     devops_engineer: str = "qwen2.5-coder-7b"
-     data_analyst: str = "qwen3-8b"
-     security_engineer: str = "qwen2.5-coder-7b"
+     ceo: str = "goekdenizguelmez/JOSIEFIED-Qwen3:8b"
+     cto: str = "goekdenizguelmez/JOSIEFIED-Qwen3:8b"
+     product_manager: str = "ministral-3:8b"
+     researcher: str = "ministral-3:8b"
+     developer: str = "thealxlabs/lumen:latest"
+     qa_engineer: str = "thealxlabs/lumen:latest"
+     devops_engineer: str = "thealxlabs/lumen:latest"
+     data_analyst: str = "goekdenizguelmez/JOSIEFIED-Qwen3:8b"
+     security_engineer: str = "thealxlabs/lumen:latest"
```

**File**: `config/config_loader.py:278-287`

```diff
  "models": {
-     "ceo": "qwen3-8b",
-     "cto": "qwen3-8b",
-     "product_manager": "ministral-8b",
-     "researcher": "ministral-8b",
-     "developer": "qwen2.5-coder-7b",
-     "qa_engineer": "qwen2.5-coder-7b",
-     "devops_engineer": "qwen2.5-coder-7b",
-     "data_analyst": "qwen3-8b",
-     "security_engineer": "qwen2.5-coder-7b"
+     "ceo": "goekdenizguelmez/JOSIEFIED-Qwen3:8b",
+     "cto": "goekdenizguelmez/JOSIEFIED-Qwen3:8b",
+     "product_manager": "ministral-3:8b",
+     "researcher": "ministral-3:8b",
+     "developer": "thealxlabs/lumen:latest",
+     "qa_engineer": "thealxlabs/lumen:latest",
+     "devops_engineer": "thealxlabs/lumen:latest",
+     "data_analyst": "goekdenizguelmez/JOSIEFIED-Qwen3:8b",
+     "security_engineer": "thealxlabs/lumen:latest"
  },
```

---

## Summary

| Component | Changed From | Changed To | Count |
|-----------|-------------|------------|-------|
| Department member | `DevOps` | `DevOpsEngineer` | 1 |
| OKR keys | snake_case | PascalCase | 9 |
| Progress phase IDs | Short names | Full enum values | 2 definitions + 7 calls |
| Model defaults | Generic names | Specific model IDs | 18 |
| **Total** | | | **41 lines** |

---

**All changes verified** ✅ — Run `python3 verify_naming_consistency.py` to confirm.
