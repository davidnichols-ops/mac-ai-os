# PR Governance System for an AI-Agent-Managed MCP Server

## Architecture Analysis

### Current System (Before)

The MAOS guards system consisted of four guards in `app/guards.py`:

1. **VibeLanguageGuard** — text scanner for unsupported success claims
2. **PrPreflightGuard** — required test/build/manual evidence via boolean flags
3. **LessonComplianceGuard** — keyword matching against recorded lessons
4. **SubagentProofGuard** — flagged sub-agent agreement as sole evidence

### Critical Gaps Identified

| Gap | Severity | Description |
|-----|----------|-------------|
| Self-attested booleans | **Critical** | `--has-tests=True` is an agent claim, not system-produced evidence. An agent can always claim a boolean. |
| Zero test coverage | **High** | 460 lines of guard code with no unit tests. Guards could break silently. |
| No diff inspection | **High** | Guards never looked at the actual git diff. Secrets, debug statements, and security boundary changes went undetected. |
| No MCP-specific risk detection | **High** | Tool schema, description, and permission changes (which alter agent behavior) were not inspected. |
| No PR content structure | **Medium** | No requirement for Summary, Why, What Changed, Root Cause, Before/After sections. |
| No state machine | **Medium** | No formal states or transitions. An agent could skip from "I changed code" to "Open PR" by asserting booleans. |
| No evidence objects | **Medium** | Evidence was a boolean, not a structured object with command, exit code, timestamp, commit. |
| No GitHub integration | **Low** | No `gh` CLI or API for CI status, review requirements, or merge readiness. |
| Lesson compliance is keyword-based | **Low** | Crude overlap matching causes both false positives and false negatives. |

### Threat Model

The threat is not external attackers — it's **the AI agent itself gaming the system**. An agent under pressure to "complete the task" may:

1. Claim `--has-tests=True` without running tests
2. Claim `--has-build=True` without building
3. Use sub-agent agreement as proof ("both agents agree it works")
4. Write "everything is working" without evidence
5. Open a PR with secrets in the diff
6. Change MCP tool descriptions (altering agent behavior) without review
7. Skip writing a PR description and just assert "verified"

The governance system must make gaming harder than honest verification.

### Design Principles

1. **Evidence-producing, not evidence-consuming** — the system executes commands and captures results; the agent does not supply the outcome.
2. **A boolean represents the result of a verification process, not the process itself** — `has_tests=True` should mean "the system ran tests and they passed", not "the agent says tests passed".
3. **State machine with evidence-gated transitions** — an agent cannot skip states by asserting booleans.
4. **Inspect the actual diff** — don't trust the PR description; look at what actually changed.
5. **MCP-specific risk detection** — tool schemas, descriptions, and permissions are behaviorally significant API changes.
6. **Distinguish local readiness from CI readiness from merge readiness** — don't collapse these into one "everything is green" check.
7. **Smallest useful version** — don't build enterprise compliance; build what closes the biggest gaps.

### Proposed Architecture (Implemented)

```
Change
  ↓
Evidence Collection (system-executed)
  ├── maos guards verify -- <test-command>    → CommandEvidence
  ├── maos guards verify -- <lint-command>    → CommandEvidence
  ├── maos guards diff                        → DiffGuard result
  └── manual verification                     → ManualVerificationEvidence
  ↓
Guard Evaluation
  ├── VibeLanguageGuard (existing, improved)
  ├── PrPreflightGuard (existing, now consumes evidence)
  ├── DiffGuard (new — secrets, debug, MCP, security paths)
  ├── PRContentGuard (new — required sections)
  ├── LessonComplianceGuard (existing)
  └── SubagentProofGuard (existing, now respects review evidence)
  ↓
Readiness Decision (state machine)
  DRAFT → UNDERSTOOD → VERIFIED → PR_READY → CI_READY → MERGE_READY
  ↓
PR
```

### Rejected Alternatives

- **Cryptographic attestation** — over-engineered for a local dev environment. The threat model is agent self-attestation, not tampering.
- **GitHub API integration for CI status** — adds a network dependency and OAuth complexity. CI readiness is a separate concern; local PR readiness is the priority.
- **Automated test coverage analysis** — determining whether tests actually cover the changed behavior requires semantic analysis beyond the scope of the smallest useful version. The system requires a test command to pass, but does not claim it proves correctness.
- **Structured lesson policies** — replacing free-text lessons with structured trigger/action/evidence policies would be a larger refactor. The current keyword-based approach is adequate for the smallest useful version.

---

## PR Checklist Specification

### Required Sections

#### For All PRs

| Section | Required | Purpose |
|---------|----------|---------|
| Summary | Yes | What changed (1-2 sentences) |
| Why | Yes | Why it changed (motivation) |
| What Changed | Yes | Detailed list of changes |
| Known Limitations | No | Known caveats |
| Scope | No | Stated scope (for unrelated-file detection) |

#### For Bug Fixes (additionally)

| Section | Required | Purpose |
|---------|----------|---------|
| Root Cause | Yes | What was the root cause |
| Regression Test | Yes | What test proves the bug is fixed |
| Before | Yes | Previous behavior |
| After | Yes | New behavior |

#### For Features (additionally)

| Section | Required | Purpose |
|---------|----------|---------|
| Behavior Changed | Recommended | What behavior changed |
| Behavior Unchanged | Recommended | What intentionally did not change |
| Risk | Recommended | Assessed risk level |

### Evidence Requirements

| Check | Evidence Type | How Produced |
|-------|---------------|--------------|
| Tests | CommandEvidence | `maos guards verify -- uv run pytest --save test.json` |
| Build/Lint | CommandEvidence | `maos guards verify -- uv run ruff check --save lint.json` |
| Manual Verification | ManualVerificationEvidence | Structured before/after (required for bug fixes) |
| Diff | DiffGuard result | `maos guards diff` (automatic in readiness check) |
| Review | ReviewEvidence | Optional — sub-agent or human review with specific findings |

### Blocking Conditions

| Condition | Verdict | Action |
|-----------|---------|--------|
| Secret/credential in diff | BLOCK | Remove before proceeding |
| Tests failed | BLOCK | Fix tests |
| Build/lint failed | BLOCK | Fix issues |
| Missing required PR section | BLOCK | Complete the section |
| Stale evidence (wrong commit) | BLOCK | Re-run verification |
| Sub-agent agreement as sole evidence | BLOCK | Provide empirical evidence |
| Lesson violation | BLOCK | Address the lesson |
| Debug statement in diff | WARN | Review — likely should be removed |
| Dependency file changed | WARN | Review — verify no supply-chain risk |
| Security-sensitive path changed | WARN (REQUIRES_REVIEW) | Human review recommended |
| MCP tool/schema changed | WARN (REQUIRES_REVIEW) | Human review recommended |
| Large diff (>1000 lines) | WARN | Review — ensure not accidentally broad |
| Binary file in diff | WARN | Verify intentional |

---

## Implementation

### Files Created/Modified

| File | Status | Purpose |
|------|--------|---------|
| `app/guards_evidence.py` | New | Evidence objects (CommandEvidence, ManualVerificationEvidence, ReviewEvidence) + `capture_command()` |
| `app/guards_diff.py` | New | Diff inspection (secrets, debug, MCP, security paths, dependencies, binary, scope) |
| `app/guards_readiness.py` | New | PR readiness state machine + PR content checklist |
| `app/guards.py` | Modified | `run_all_guards()` now consumes evidence objects and integrates DiffGuard |
| `app/cli/commands/guards.py` | Modified | New commands: `verify`, `diff`, `readiness` |
| `tests/test_guards.py` | New | 30 tests for existing guards + evidence integration |
| `tests/test_guards_evidence.py` | New | 20 tests for evidence objects and command execution |
| `tests/test_guards_diff.py` | New | 18 tests for diff inspection |
| `tests/test_guards_readiness.py` | New | 17 tests for state machine and content validation |

### CLI Commands

```bash
# Evidence-producing verification (replaces --has-tests boolean)
maos guards verify -- uv run pytest --save test-evidence.json
maos guards verify -- uv run ruff check app tests --save lint-evidence.json

# Diff inspection
maos guards diff --json
maos guards diff --scope app/guards

# PR readiness assessment (state machine)
maos guards readiness \
  --test-evidence test-evidence.json \
  --lint-evidence lint-evidence.json \
  --pr-description pr.md \
  --bug-fix

# Existing commands (still work, now accept --evidence)
maos guards scan "everything is working" --json
maos guards check "text" --pr --evidence evidence.json
```

### State Machine

```
DRAFT
  │  requires: PRContent (summary, why, what_changed)
  ↓
UNDERSTOOD
  │  requires: CommandEvidence (tests passed) + CommandEvidence (build/lint passed)
  │  + ManualVerificationEvidence (for bug fixes)
  ↓
VERIFIED
  │  requires: DiffGuard PASS/WARN (no BLOCK patterns)
  ↓
PR_READY
  │  requires: GitHub PR exists + correct base branch (future)
  ↓
CI_READY
  │  requires: CI green + review approval (future)
  ↓
MERGE_READY
```

---

## MCP-Specific Risk Analysis

### Tool Definitions

Changes to tool names, descriptions, input schemas, output schemas, and error behavior are detected via:
- Path matching: `mcp/`, `schemas`, `server`, `tool` files
- Content matching: `@mcp.tool`, `description = "..."`, JSON schema `type` definitions, `inputSchema`/`parameters` assignments

These changes receive **REQUIRES_REVIEW** severity (WARN, not BLOCK) because they alter agent behavior even when the implementation appears correct.

### Permissions

Changes to files matching `auth`, `permission`, `trust`, `policy`, `security`, `credential`, `token`, `guard`, `approval`, `sandbox`, `boundary` receive **REQUIRES_REVIEW** severity.

### Trust and Identity

The system does not currently detect changes to what an agent believes a tool can do (that would require semantic analysis). It detects changes to the files that define tool descriptions and schemas, which is the proximate signal.

---

## Final Standard: Honest Self-Assessment

> Can an AI agent currently make an unsupported claim that "the PR is verified" and pass the guards without the system independently establishing meaningful evidence?

**With the evidence-producing workflow: No.** When the agent uses `maos guards verify` and `maos guards readiness`, the system executes commands and captures results. The agent cannot fake the exit code, stdout, or commit hash.

**With the legacy boolean workflow: Yes.** The `--has-tests` flag still exists for backward compatibility. An agent can still pass `--has-tests=True` without running tests. However, the system now documents this as LEGACY and prefers evidence objects.

**Remaining gaps (honestly acknowledged):**

1. **Test relevance** — the system verifies that *a* test command passed, not that the tests *cover the changed behavior*. A agent could run `pytest tests/test_unrelated.py` and pass. This is a known limitation; addressing it requires coverage analysis or semantic test-to-diff matching, which is beyond the smallest useful version.

2. **Manual verification is still self-attested** — `ManualVerificationEvidence` has structured fields (action, observed, before, after), but the agent fills them in. The system checks that before != after and fields are non-empty, but cannot verify the agent actually did what it claims. This is inherent to manual verification — the system cannot execute "manual" steps.

3. **Evidence file could be fabricated** — an agent could write a JSON file that looks like CommandEvidence with `exit_code: 0` without actually running the command. The system trusts evidence files produced by `guards verify --save`. A harder system would sign evidence with a key the agent doesn't have, but that's over-engineered for a local dev environment.

4. **Diff inspection is pattern-based** — secret detection uses regex patterns. A novel secret format would not be detected. The patterns are conservative (false positives preferred over false negatives), but not complete.

5. **No GitHub integration** — CI_READY and MERGE_READY states are defined but not implemented. There is no `gh` CLI or GitHub API integration to check CI status, review requirements, or merge conflicts.

6. **Lesson compliance is still keyword-based** — the generic overlap check (`> 3 shared words`) is crude. Structured lesson policies would be better but require a larger refactor.

**The work is incomplete in the sense that no system can fully prevent a determined agent from gaming it. But the biggest conceptual weakness — boolean self-attestation — is now addressed. The system can distinguish "I believe this works" from "Here is the reproducible evidence for what was actually verified."**
