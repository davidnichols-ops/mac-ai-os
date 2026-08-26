# First Validation Campaign

## Target

**Trustcard**, local checkout:

```text
~/Projects/trustcard
```

Authorization basis: repository remote is `https://github.com/davidnichols-ops/trustcard.git`; campaign is offline and fixture-only.

## Objective

Validate whether Trustcard's MCP proxy preserves authorization and provenance when the observed tool surface changes or JSON-RPC request identity is manipulated.

This campaign does not contact external MCP servers, use credentials, or mutate production state.

## Tests

### 1. Native security regression suite

```bash
cd ~/Projects/trustcard
npm test
npm run lint
node --test test/adversarial.test.js
node --test test/audit-probes.test.js
node --test test/auth.test.js
node --test test/evidence-adapters.test.js
node --test test/proxy.test.js
node --test test/security-model.test.js
```

### 2. Manifest drift fixture

- Start a local fixture MCP server.
- Capture approved manifest and its hash.
- Change a tool name, description, input schema, and required capability one at a time.
- Send `tools/list` and `tools/call` through the local proxy.
- Confirm drift is denied or explicitly surfaced as a policy decision.

Expected evidence:

- approved manifest hash
- live manifest hash
- diff classification
- call decision
- receipt/audit record
- process exit status

### 3. Duplicate and replayed request IDs

- Send two local fixture requests with the same JSON-RPC ID.
- Replay a previously accepted request after the policy state changes.
- Confirm responses cannot be mis-associated with a different pending request.

Expected evidence:

- raw fixture messages
- request IDs
- pending-state transitions
- proxy responses
- policy/receipt state

### 4. Auto-update policy mutation

- Run only against a temporary copied manifest.
- Present a newly discovered fixture tool.
- Confirm the update is represented as a privileged state transition.
- Confirm the original approved manifest is unchanged.

Expected evidence:

- temporary manifest before/after hashes
- update decision
- approval or strict-mode result
- filesystem diff

### 5. Environment-boundary inspection

- Use synthetic marker variables only.
- Confirm the fixture child receives only the documented environment contract.
- Do not place credentials in the environment or logs.

Expected evidence:

- allowlisted environment-key set
- child observation
- proxy configuration
- absence of secret values

## Expected outcomes

| Outcome | Classification |
|---|---|
| Policy rejects drift/replay and evidence binds exact hashes | Supported security claim |
| Behavior is not exercised by existing tests | Missing validation |
| A reproducible unauthorized action occurs in a local fixture | Confirmed vulnerability candidate, subject to review |
| Static concern has no demonstrated impact | Suspected weakness |
| Test harness cannot establish behavior | Unverified hypothesis |

## Stop conditions

Stop immediately if a test would require:

- a production endpoint
- a third-party system
- real credentials or tokens
- destructive state changes
- uncontrolled network traffic
- unbounded resource consumption

## Reporting standard

No finding is promoted without a reproducible local test, captured output, exact commit, affected component, security impact, root cause, safe reproduction, and remediation recommendation.
