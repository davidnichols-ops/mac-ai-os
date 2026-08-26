# TC-E3 — Missing declared lint command

## Status

Missing validation. Not a vulnerability.

## Observation

`npm run lint` exits with `Missing script: lint`.

## Impact

The package does not expose the expected static-analysis command. This weakens reproducibility of security and quality validation, but does not itself establish exploitable impact.

## Remediation

Add a deterministic lint script or document the supported lint/type/security command and include it in CI.
