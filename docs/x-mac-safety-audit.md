# X-MaC Safety Audit — 2026-08-03

## Current State (from codebase audit)

| Component | Status | Evidence |
|-----------|--------|---------|
| Safety rule engine (`src/safety/`) | **Compiled but not wired** | `#![allow(dead_code, unused_imports)]` — rules load but don't gate deletions |
| Cleanup undo (`src/cleanup/undo.rs`) | **Partially implemented** | Records original + trashed paths for restore-from-Trash |
| Cleanup transaction (`src/cleanup/transaction.rs`) | **Partially implemented** | Transaction model exists but MCP execution is stubbed |
| Safety state machine | **Does not exist** | README claims PREVIEW→APPROVED→EXECUTING→VERIFIED→ROLLBACK but no code |
| Snapshot verification | **Not implemented** | No code to verify snapshot contains what rollback claims |
| Rollback under interruption | **Not tested** | No tests for kill-during-operation |
| Audit log | **Not implemented** | No permanent log of destructive operations |
| MCP cleanup execution | **Stubbed** | `mcp/mod.rs` explicitly says "Cleanup execution is a stub" |

## Critical Gaps

1. **The safety rule engine is dead code.** It compiles but is never called.
   Rules load from YAML but don't gate any deletions. This is the most dangerous
   gap — the README promises safety but the code doesn't enforce it.

2. **The safety state machine doesn't exist.** The README's contract
   ("PREVIEW → APPROVED → EXECUTING → VERIFIED → ROLLBACK AVAILABLE") is
   aspirational, not implemented. There is no enum driving cleanup phases.

3. **MCP cleanup execution is a stub.** The MCP server explicitly states
   cleanup execution and trash-emptying are stubs. No destructive operation
   actually runs through the safety system.

## Recommended Implementation Order

### Phase 1: Wire the safety rule engine (1-2 days)
1. Remove `#![allow(dead_code)]` from `src/safety/`
2. Call the rule engine before every cleanup action
3. Block actions rated "protected", require approval for "review"
4. Add tests: rule engine gates deletions correctly

### Phase 2: Implement the safety state machine (1-2 days)
1. Create `SafetyState` enum: PREVIEW → APPROVED → EXECUTING → VERIFIED → ROLLBACK_AVAILABLE
2. Drive cleanup through this state machine
3. No action can skip states (PREVIEW must happen before APPROVED)
4. Add tests: state transitions enforced, no skipping

### Phase 3: Implement snapshot verification (1 day)
1. Before execution, snapshot the affected files (paths, sizes, hashes)
2. After execution, verify the snapshot matches what was expected
3. If verification fails, mark as ROLLBACK_AVAILABLE and alert
4. Add tests: snapshot catches missing files, wrong sizes

### Phase 4: Implement the first real rollback workflow (2-3 days)
1. Pick one narrow workflow: "empty Xcode DerivedData"
2. Snapshot → Preview → Approve → Execute → Verify → Rollback available
3. Test under interruption (kill halfway)
4. Test under disk pressure (insufficient space for snapshot)
5. Test with permissions errors, locked files, disappearing files

### Phase 5: Add audit log (1 day)
1. Every destructive operation logs: what was proposed, approved, happened, verified
2. Log is append-only, tamper-evident
3. Log includes: timestamp, action, files affected, snapshot ID, outcome

## Safety State Machine Specification

```
PREVIEW
  ↓ (user reviews and approves)
APPROVED
  ↓ (system begins execution)
EXECUTING
  ↓ (execution completes)
VERIFIED
  ↓ (snapshot verified, rollback available)
ROLLBACK_AVAILABLE
  ↓ (user may undo within 7 days)
COMPLETE
```

**Transitions:**
- PREVIEW → APPROVED: only via explicit user approval
- APPROVED → EXECUTING: automatic, no user action needed
- EXECUTING → VERIFIED: only if snapshot verification passes
- EXECUTING → FAILED: if execution or verification fails
- VERIFIED → ROLLBACK_AVAILABLE: automatic
- ROLLBACK_AVAILABLE → COMPLETE: after 7 days or explicit user dismissal
- ROLLBACK_AVAILABLE → ROLLING_BACK: user requests undo
- ROLLING_BACK → COMPLETE: rollback finished

**Hard rules:**
- No state can be skipped (PREVIEW must happen before APPROVED)
- FAILED state requires manual intervention
- AI confidence NEVER bypasses safety controls (task #179)
- Every transition is logged to the audit log
