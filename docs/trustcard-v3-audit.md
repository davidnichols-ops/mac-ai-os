# Trustcard v3.0.0 Audit — 2026-08-03

## Current State

| Component | Status | Evidence |
|-----------|--------|---------|
| Tests | ✅ 514/514 pass | `node --test "test/*.test.js"` |
| npm published version | 2.2.1 | `npm view mcp-trustcard version` |
| package.json version | 3.0.0 | Unpublished |
| Git working tree | DIRTY | 8 modified + untracked files |
| Security | ✅ Clean | No hardcoded secrets in lib/ or bin/ |
| bruce-lee reference server | ✅ Functional | 87/100 scan score, 42/42 tests |
| mcp-server.js framework | ✅ Functional | 16/16 tests |
| README | STALE | Documents v2.2, not v3.0.0 |
| AGENTS.md | STALE | Claims 283 tests (actual: 514) |
| OPENHANDS-CONTEXT.md | STALE | Describes v2.0.0 |

## Blockers for Publishing v3.0.0

### High Priority
1. **Commit untracked v3 files** — bruce-lee server, mcp-server.js, new tests,
   package-lock.json. `npm publish` packs from committed tree; untracked files
   won't be included.
2. **Update README** — add v3.0.0 sections: evidence substrate, observatory,
   mcp-server framework, bruce-lee reference server.
3. **Update AGENTS.md** — fix test count from 283 to 514.

### Medium Priority
4. **Update OPENHANDS-CONTEXT.md** — or archive it, since it describes v2.0.0.
5. **Clean install test** — after publishing, install v3.0.0 from npm in a
   fresh directory and verify it works (task #518).

### Low Priority
6. **Architecture diagram** — create a public diagram showing manifest identity
   → capability identity → policy evaluation → agent identity → authorization
   → invocation → outcome (task #95).
7. **"Trustcard in five minutes" example** (task #93).

## Security Audit Findings

- ✅ No hardcoded secrets in lib/ or bin/
- ✅ The `AKIA` matches in `lib/checks.js` are the scanner's own detection regex
- ✅ `auth-issue`/`TRUSTCARD_AUTH_SECRET` references read from env, never hardcoded
- ✅ Data files are ecosystem crawl artifacts, not secrets
- ⚠️ npm token is revoked (task #516) — need new token from user to publish

## v3.0.0 Feature Inventory

1. **Evidence substrate** — universal discovery + Layer 1 existence verification
2. **Observatory** — ecosystem discovery and monitoring
3. **mcp-server.js framework** — minimal framework for well-behaved MCP servers
4. **bruce-lee reference server** — deterministic hash-chained agent decision
   audit log, 87/100 scan score, 3 tools, 42 tests
5. **Auth scope enforcement** (v2.3) — per-agent OAuth scopes, DevIssuer,
   IdpIntrospector, TokenValidator
6. **Three-engine danger fusion** (v2.2.1) — heuristic + semantic + injection
7. **Gate 2 invocation policy** (v2) — per-call argument re-validation
8. **Signed chained receipts** (v2) — Ed25519-signed, hash-chained
9. **Key rotation** (v2) — old-key-signs-new-key with expiring rotation certs
10. **Descriptor system** (v2) — protocol-neutral capability descriptor

## Recommended Publish Sequence

1. Commit all untracked v3 files (bruce-lee, mcp-server.js, tests, lock file)
2. Update README with v3.0.0 features
3. Update AGENTS.md test count
4. Get new npm token from user (BLOCKED — needs user action)
5. `npm publish` v3.0.0
6. Clean install test from npm
7. Run drift scan 24h after first scan
8. Tag release, update changelog
