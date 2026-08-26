# Trajectory OS integration

MAOS can consume a provider-neutral context pack from a local Trajectory OS
installation. Trajectory OS remains the owner of personal data; MAOS only
reads the rendered context pack when starting `maos do`.

## Enable it

Make sure the Trajectory OS CLI is available as `trajectory`, then run:

```bash
trajectory first-run
trajectory onboard
trajectory context daily
```

MAOS automatically tries `trajectory context daily --format markdown` for each
`maos do` invocation. If the command is unavailable or fails, MAOS continues
with its normal local context.

Optional environment variables:

```bash
export TRAJECTORY_COMMAND=trajectory
export TRAJECTORY_DB_PATH="$HOME/.local/share/trajectory/trajectory.db"
```

The bridge is read-only. Memory updates still require Trajectory OS’s normal
propose/approve workflow.
