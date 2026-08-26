# TC-E1 — Manifest expiry duration semantics

## Status

Suspected weakness / test-contract mismatch. Not a confirmed security vulnerability.

## Affected component

`lib/manifest.js`, `buildManifest`, `isoDaysFromNow`.

## Hypothesis

A manifest configured for 90 days should expire after approximately 90 elapsed 24-hour periods.

## Reproduction

```bash
cd ~/Projects/trustcard
node --input-type=module <<'JS'
import { buildManifest } from './lib/manifest.js';
const m = buildManifest([{name:'read', description:'read', inputSchema:{type:'object'}}]);
console.log(Date.parse(m.expiresAt) - Date.now());
JS
```

## Observation

The implementation uses `Date.setDate(Date.getDate() + days)`, which preserves calendar-day semantics. The existing strict test compares against a fixed 90 × 24-hour duration and failed by approximately one hour in the current timezone/date context.

## Impact

Low direct security impact. It may create an expiry-boundary mismatch between operators or validators that interpret “90 days” as fixed elapsed time versus calendar days. No bypass or unauthorized action was demonstrated.

## Remediation

Choose and document one contract:

- calendar-day validity: update the test and documentation; or
- fixed elapsed validity: add `days * 24 * 60 * 60 * 1000` to the current timestamp.

Add a timezone/DST regression test.
