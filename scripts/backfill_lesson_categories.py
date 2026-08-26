"""Backfill existing lessons with categories, keywords, and reminders.

Run once after the metadata column migration:
    uv run python scripts/backfill_lesson_categories.py
"""
from app.db import Database
from app.onboarding import record_lesson
import json

# (slug, category, keywords, reminder)
BACKFILL = [
    # --- tooling ---
    ("three_strikes_change_approach", "tooling",
     ["503", "error code", "ollama", "deepinfra", "402", "sunk cost", "tweak", "tweaking"],
     "Same error 3x? STOP tweaking. Change the tool, not the parameters."),
    ("venv_native_extensions_missing", "tooling",
     ["module not found", "pydantic_core", "mlx.core", "regex._regex", "native extension", "uv sync"],
     "Native extensions missing after venv upgrade? rm -rf .venv && uv sync --no-cache"),
    ("npm_broken_via_mise", "tooling",
     ["npm", "mise", "broken", "shim"],
     "npm via mise broken? Check node install has lib/node_modules/npm."),
    ("npm_mise_fix", "tooling",
     ["npm", "mise", "node@", "missing npm"],
     "mise npm broken = node install missing lib/node_modules/npm. Reinstall node."),
    ("npm_fallback_for_installability_check", "tooling",
     ["npm view", "npm broken", "installability"],
     "npm broken? Use a fallback npm or npx for installability checks."),
    ("npm_publish_cache_lag", "tooling",
     ["npm publish", "npm view", "cache lag", "cdn"],
     "After npm publish, npm view may show old version for minutes (CDN lag). Wait + retry."),
    ("mcp_server_no_bin_detection", "tooling",
     ["mcp server", "no bin", "npm package", "storybook/mcp"],
     "MCP package with no bin field = library, not a server. Skip it."),
    ("stale_proxy_path", "tooling",
     ["failed to list tools", "proxy binary", "mcp servers fail en masse"],
     "All MCP servers failing? Check proxy binary path in config.json."),
    ("find_exec_quoting_in_node", "tooling",
     ["execsync", "find -exec", "backslash semicolon", "quoting"],
     "Node execSync + find -exec has quoting issues. Use a shell script instead."),
    ("docker_desktop_first_run", "tooling",
     ["docker", "first run", "timeout", "docker desktop"],
     "Docker Desktop first run may timeout. Wait for it to fully start."),
    ("devin_env_var_config_syntax", "tooling",
     ["devin", "env var", "config", "config.json syntax"],
     "Devin config.json env vars need exact syntax. Check docs."),
    ("devin_oauth_token_location", "tooling",
     ["devin", "oauth", "token", "token location"],
     "Devin OAuth token stored at specific path. Check if auth issues."),
    ("devin_posttooluse_hooks_for_ambient_injections", "tooling",
     ["devin", "hooks", "posttooluse", "ambient injection"],
     "Use Devin postToolUse hooks for ambient context injection."),
    ("harness_parser_failure", "tooling",
     ["model fails to use tools", "harness", "parser", "protocol failure"],
     "Model not using tools? Isolate MODEL vs PROTOCOL vs HARNESS failure first."),

    # --- workflow ---
    ("verify_before_pr", "workflow",
     ["pr", "pull request", "verify", "verified", "unverified", "fix you haven't"],
     "Never open a PR with an unverified fix. Run the actual command, compare before/after."),
    ("no_devin_attribution", "workflow",
     ["co-author", "devin", "generated with", "ai branding", "commit message"],
     "No Devin/AI co-author trailers in commits. Ever."),
    ("subagent_corroboration_is_not_proof", "workflow",
     ["subagent", "sub-agent", "corroboration", "two agents", "agreeing"],
     "Two subagents agreeing != verified. Test against real data."),
    ("stale_ai_pr_review", "workflow",
     ["ai review", "gemini code assist", "stale review", "pr review"],
     "AI PR reviews can be stale (reviewing older commits). Check the diff hash."),
    ("pr_governance_evidence_producing_guards", "workflow",
     ["pr governance", "guards verify", "evidence", "self-attested", "readiness"],
     "Use guards verify + readiness, NOT self-attested --has-tests booleans."),
    ("verify-existing-api", "workflow",
     ["feature request", "existing api", "search codebase", "open source"],
     "Before opening a feature request, search the codebase for existing APIs."),
    ("reading_is_top_action_track_it", "workflow",
     ["read tool", "reading", "top action", "track reads"],
     "Reading is the #1 agent action (3:1 over edits). Track it via MAOS."),

    # --- environment ---
    ("dont_overload_local_machine_training", "environment",
     ["local training", "mlx lora", "overload", "m4", "semaphore", "training crash"],
     "Don't run local MLX training + heavy processes on M4. It's a laptop, not a rig."),
    ("disk_organization_sorting_logic", "environment",
     ["disk", "organize", "sorting", "projects", "documents"],
     "Disk org: ~/Projects=code, ~/Documents=docs, ~/Pictures=images, ~/bin=scripts."),
    ("hidden_system_data_locations", "environment",
     ["system data", "macos", "sleepimage", "swapfiles", "unified log"],
     "macOS System Data = sleepimage + swap + unified log. Safe to clean."),
    ("macos_case_sensitive_python_imports", "environment",
     ["macos", "case insensitive", "python import", "filters", "shadowed"],
     "macOS is case-insensitive. 'filters/' can shadow 'Filters/'. Check casing."),

    # --- api ---
    ("gmail_token_expired_sentinel_graph", "api",
     ["gmail", "token refresh", "oauth", "email failed", "400 bad request"],
     "Gmail OAuth token expired? Re-authenticate the Gmail integration in MAOS."),
    ("gmail_token_refresh_failure", "api",
     ["gmail", "token refresh", "send email failed", "oauth token"],
     "Gmail token expired = email tools fail. Re-auth to fix."),
    ("colab_exec_module_caching", "api",
     ["colab", "exec", "module cache", "kernel", "import cached"],
     "Colab caches imported modules. Use importlib.reload() after patching .py files."),
    ("colab_gpu_412_limit", "api",
     ["colab", "412", "gpu limit", "resource exhausted"],
     "Colab 412 = GPU limit hit. Wait or switch to a different account/runtime."),
    ("colab_session_loss_after_long_job", "api",
     ["colab", "session loss", "long job", "timeout", "disconnected"],
     "Colab kills long sessions. Save checkpoints frequently."),
    ("kaggle_p100_pytorch_compat", "api",
     ["kaggle", "p100", "sm_60", "pytorch", "cuda compatibility"],
     "Kaggle free GPU = P100 (sm_60), not T4. Install PyTorch with sm_60 support."),
    ("github_push_protection_fake_secrets", "api",
     ["github", "push protection", "fake secrets", "blocked commit", "secret pattern"],
     "GitHub push protection blocks fake/demo secrets too. Remove or use .gitignore."),
    ("marketplace_description_125_char_limit", "api",
     ["github marketplace", "action.yml", "description", "125 char"],
     "GitHub Marketplace action.yml description must be <125 chars."),
    ("env_file_injection_for_mcp_scanning", "api",
     ["mcp server", "api key", "env file", "credentials", "handshake timeout"],
     "MCP servers need API keys to start. Inject .env file before scanning."),
    ("node_env_file_no_shell_expansion", "api",
     ["node", "env-file", "shell expansion", "$home", "dotenv"],
     "Node --env-file does NOT expand $HOME. Use absolute paths or dotenv package."),
    ("trustcard_npm_version_mismatch", "api",
     ["trustcard", "npm install", "version mismatch", "colab", "fresh environment"],
     "trustcard on Colab installs old npm version. Pin the version explicitly."),
    ("trustcard_pressure_test_results", "api",
     ["trustcard", "pressure test", "mcp servers", "manifests"],
     "trustcard pressure test: 24/159 MCP servers produced usable manifests."),
    ("trustcard_v04_production_hardening", "api",
     ["trustcard", "v0.4", "production", "security features"],
     "trustcard v0.4 has 6 production security features over v0.3."),

    # --- architecture ---
    ("rules_must_be_unavoidable", "architecture",
     ["rules", "guard system", "unavoidable", "agent route around", "five layers"],
     "Rules not surfaced at the point of decision are suggestions. Wire into every layer."),
    ("ai_time_awareness_via_mcp_request_log", "architecture",
     ["time awareness", "mcp request log", "session", "tool call timestamps"],
     "Agents have no time sense. Track MCP call timestamps for session awareness."),
    ("ambient_time_awareness_injection", "architecture",
     ["ambient", "time awareness", "injection", "response wrapper", "opt-in"],
     "Ambient > opt-in. Inject critical info into response wrappers, not separate tools."),
    ("danger_detector_false_positive_tuning", "architecture",
     ["danger detector", "false positive", "destructive verb", "tuning"],
     "Danger detector false positives: tune destructive verb matching carefully."),
    ("openhands-vite-hmr-ingress-proxy", "architecture",
     ["openhands", "vite", "hmr", "websocket", "ingress proxy"],
     "Vite HMR WebSocket vs ingress proxy: configure proxy for /ws path."),
    ("openhands-vite-watch-reload-loop", "architecture",
     ["openhands", "vite", "watch", "reload loop", "file watcher", "hmr"],
     "Vite reload loop = state writes inside project tree. Exclude state dirs from watcher."),
]


def main():
    db = Database()
    updated = 0
    for slug, category, keywords, reminder in BACKFILL:
        # Re-record with metadata. record_lesson does upsert.
        # We need to preserve the existing text and confidence.
        from app.onboarding import list_lessons
        lessons = {l["slug"]: l for l in list_lessons(db)}
        existing = lessons.get(slug)
        if not existing:
            print(f"  SKIP {slug} (not found)")
            continue
        record_lesson(
            db, slug, existing["text"],
            confidence=existing["confidence"],
            source=existing["source"],
            category=category,
            keywords=keywords,
            reminder=reminder,
        )
        updated += 1
        print(f"  OK   {slug} -> {category} ({len(keywords)} keywords)")
    print(f"\nBackfilled {updated} lessons with metadata.")


if __name__ == "__main__":
    main()
