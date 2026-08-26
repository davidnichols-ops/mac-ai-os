"""Prompt generator for post-training data generation.

Generates diverse task prompts across categories that cover the full
range of capabilities the v4 model needs:
- Tool use (file ops, shell, git, grep, search)
- Coding (write, debug, refactor, test)
- Planning (multi-step, project setup)
- Recovery (error handling, debugging)
- Control token routing (FAST vs THINK vs VERIFY vs RECOVER vs ESCALATE)

Design goals:
- 5K+ truly unique prompts with zero duplicates (hash-based set enforcement)
- No single template produces more than 5% of total output (diversity cap)
- Each prompt carries a difficulty (1-3) for curriculum learning
- Rich variable pools (50+ file names, 30+ dirs, 15+ languages, etc.) so the
  combinatorial space vastly exceeds the requested count
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from collections.abc import Generator
from pathlib import Path

# ---------------------------------------------------------------------------
# Variable pools — deliberately large to maximize combinatorial diversity.
# ---------------------------------------------------------------------------

FILE_NAMES: list[str] = [
    "main.py",
    "app.py",
    "config.py",
    "settings.py",
    "models.py",
    "views.py",
    "routes.py",
    "handlers.py",
    "utils.py",
    "helpers.py",
    "cli.py",
    "server.py",
    "client.py",
    "database.py",
    "migrations.py",
    "schema.py",
    "types.py",
    "constants.py",
    "exceptions.py",
    "logger.py",
    "auth.py",
    "middleware.py",
    "tasks.py",
    "workers.py",
    "scheduler.py",
    "cache.py",
    "store.py",
    "repository.py",
    "service.py",
    "controller.py",
    "serializer.py",
    "validator.py",
    "parser.py",
    "renderer.py",
    "formatter.py",
    "exporter.py",
    "importer.py",
    "loader.py",
    "fetcher.py",
    "builder.py",
    "factory.py",
    "adapter.py",
    "wrapper.py",
    "proxy.py",
    "observer.py",
    "listener.py",
    "config.yaml",
    "config.yml",
    "settings.toml",
    "pyproject.toml",
    "package.json",
    "Dockerfile",
    "docker-compose.yml",
    "Makefile",
    ".env.example",
    "requirements.txt",
    "setup.cfg",
    "tox.ini",
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "tsconfig.json",
    "webpack.config.js",
    "vite.config.ts",
    ".eslintrc.js",
    ".prettierrc",
    "Cargo.toml",
    "go.mod",
    "go.sum",
    "build.gradle",
    "pom.xml",
    "CMakeLists.txt",
    "Gemfile",
    "mix.exs",
]

DIRECTORIES: list[str] = [
    "src/",
    "src/main/",
    "src/lib/",
    "src/core/",
    "src/api/",
    "src/cli/",
    "src/utils/",
    "src/models/",
    "src/services/",
    "src/handlers/",
    "src/components/",
    "src/hooks/",
    "src/pages/",
    "src/layouts/",
    "tests/",
    "tests/unit/",
    "tests/integration/",
    "tests/e2e/",
    "tests/fixtures/",
    "docs/",
    "docs/api/",
    "docs/guides/",
    "docs/architecture/",
    "scripts/",
    "scripts/deploy/",
    "scripts/migration/",
    "config/",
    "config/production/",
    "config/staging/",
    ".github/",
    ".github/workflows/",
    ".github/ISSUE_TEMPLATE/",
    "app/",
    "app/api/",
    "app/models/",
    "app/views/",
    "app/services/",
    "lib/",
    "lib/internal/",
    "lib/public/",
    "internal/",
    "pkg/",
    "cmd/",
    "bin/",
    "vendor/",
    "migrations/",
    "static/",
    "templates/",
    "locales/",
    "assets/",
    "public/",
]

LANGUAGES: list[str] = [
    "python",
    "rust",
    "go",
    "typescript",
    "javascript",
    "java",
    "kotlin",
    "swift",
    "c",
    "cpp",
    "csharp",
    "ruby",
    "php",
    "scala",
    "elixir",
    "haskell",
    "clojure",
    "perl",
    "lua",
    "dart",
]

ERROR_TYPES: list[str] = [
    "SyntaxError",
    "IndentationError",
    "ImportError",
    "ModuleNotFoundError",
    "AttributeError",
    "TypeError",
    "ValueError",
    "KeyError",
    "IndexError",
    "NameError",
    "UnboundLocalError",
    "RuntimeError",
    "StopIteration",
    "ZeroDivisionError",
    "OverflowError",
    "FloatingPointError",
    "RecursionError",
    "MemoryError",
    "TimeoutError",
    "ConnectionError",
    "ConnectionRefusedError",
    "ConnectionResetError",
    "FileNotFoundError",
    "PermissionError",
    "IsADirectoryError",
    "NotADirectoryError",
    "OSError",
    "IOError",
    "JSONDecodeError",
    "UnicodeDecodeError",
    "AssertionError",
    "NotImplementedError",
    "RuntimeWarning",
]

GIT_OPERATIONS: list[str] = [
    "commit",
    "push",
    "pull",
    "merge",
    "rebase",
    "cherry-pick",
    "revert",
    "reset",
    "checkout",
    "stash",
    "pop",
    "branch",
    "tag",
    "fetch",
    "bisect",
    "blame",
    "reflog",
    "squash",
    "amend",
    "archive",
]

CODE_PATTERNS: list[str] = [
    "function",
    "class",
    "decorator",
    "context manager",
    "generator",
    "iterator",
    "async function",
    "callback",
    "promise",
    "future",
    "closure",
    "lambda",
    "method",
    "property",
    "staticmethod",
    "classmethod",
    "abstract method",
    "magic method",
    "mixin",
    "protocol",
    "dataclass",
    "enum",
    "type alias",
    "generic",
]

SEARCH_PATTERNS: list[str] = [
    "function definitions",
    "class definitions",
    "import statements",
    "TODO comments",
    "FIXME markers",
    "deprecated annotations",
    "error handling blocks",
    "async/await usage",
    "type annotations",
    "decorator usage",
    "SQL queries",
    "API endpoint routes",
    "test cases",
    "configuration constants",
    "logging statements",
    "security-sensitive operations",
    "database transactions",
]

TASK_COMPLEXITIES: list[str] = ["simple", "moderate", "complex"]

# Extended fill values keyed by placeholder name used across templates.
FILL_VALUES: dict[str, list[str]] = {
    "file_name": FILE_NAMES,
    "path": [f"{d}{f}" for d in DIRECTORIES[:20] for f in FILE_NAMES[:10]][:80],
    "src": [f"{d}{f}" for d in DIRECTORIES[:10] for f in FILE_NAMES[:8]][:60],
    "dst": [f"{d}{f}" for d in DIRECTORIES[10:20] for f in FILE_NAMES[8:16]][:60],
    "dir": DIRECTORIES,
    "pattern": [
        "*.py",
        "*.ts",
        "*.tsx",
        "*.js",
        "*.jsx",
        "*.go",
        "*.rs",
        "*.java",
        "*.rb",
        "*.c",
        "*.cpp",
        "test_*",
        "*_test.go",
        "spec_*",
        "*.yaml",
        "*.yml",
        "*.json",
        "*.toml",
        "*.md",
        "*.cfg",
        "*.ini",
        "**/*.tsx",
        "**/*.py",
        "**/*_test.py",
        "Dockerfile*",
        "Makefile",
        "*.sh",
        "*.sql",
    ],
    "keyword": [
        "import",
        "TODO",
        "FIXME",
        "deprecated",
        "async",
        "await",
        "error",
        "warning",
        "hack",
        "bug",
        "review",
        "optimize",
        "refactor",
        "security",
        "password",
        "token",
        "secret",
        "config",
        "debug",
        "trace",
        "log",
        "cache",
        "lock",
        "mutex",
        "thread",
        "process",
        "signal",
        "timeout",
    ],
    "project_type": [
        "FastAPI",
        "Flask",
        "Django",
        "CLI tool",
        "React SPA",
        "Next.js",
        "Vue app",
        "data pipeline",
        "ETL service",
        "microservice",
        "REST API",
        "GraphQL API",
        "gRPC service",
        "WebSocket server",
        "web scraper",
        "crawler",
        "CLI game",
        "terminal UI",
        "desktop app",
        "mobile app",
        "browser extension",
        "VS Code extension",
        "Jupyter notebook",
        "Lambda function",
        "Cloudflare Worker",
        "Discord bot",
        "Telegram bot",
        "Slack app",
        "cron job",
        "daemon service",
    ],
    "content_desc": [
        "a Python class with type hints and docstrings",
        "a YAML configuration file with environment overrides",
        "a JSON schema definition for API validation",
        "a markdown documentation page with code examples",
        "a SQL migration script to add a new table",
        "a Dockerfile for a multi-stage Python build",
        "a TOML configuration with tool settings",
        "a shell script for automated deployment",
        "a GitHub Actions workflow for CI/CD",
        "a Makefile with test, lint, and deploy targets",
        "a TypeScript interface for API responses",
        "a Rust struct with serde serialization",
        "a Go package with initialization logic",
        "a React component with hooks and props",
        "a pytest fixture for database setup",
        "a Kubernetes deployment manifest",
        "a Terraform module for S3 buckets",
        "an OpenAPI specification document",
        "a changelog entry following semver",
        "a contributing guide with setup instructions",
    ],
    "old": [
        "old_function",
        "TODO",
        "placeholder",
        "debug print",
        "old API endpoint",
        "deprecated method",
        "legacy code",
        "hardcoded value",
        "magic number",
        "global variable",
        "print statement",
        "manual check",
        "inline regex",
        "nested if-else",
        "commented-out code",
    ],
    "new": [
        "new_function",
        "DONE",
        "implementation",
        "logger.info",
        "new API endpoint",
        "updated method",
        "refactored code",
        "config value",
        "named constant",
        "dependency injection",
        "structured logging",
        "validation function",
        "compiled regex",
        "guard clause",
        "removed code",
    ],
    "command": [
        "npm test",
        "npm run build",
        "python -m pytest",
        "python -m pytest -x --tb=short",
        "docker build -t app .",
        "docker compose up -d",
        "make deploy",
        "make lint",
        "curl localhost:8000/health",
        "curl -X POST localhost:8000/api",
        "go test ./...",
        "cargo test",
        "cargo build --release",
        "go build -o bin/app",
        "ruff check .",
        "ruff format .",
        "mypy src/",
        "eslint src/ --fix",
        "tsc --noEmit",
        "prettier --write .",
        "black .",
        "isort .",
        "flake8 .",
    ],
    "package": [
        "requests",
        "fastapi",
        "pytest",
        "pandas",
        "rich",
        "httpx",
        "pydantic",
        "sqlalchemy",
        "alembic",
        "celery",
        "redis",
        "aiohttp",
        "uvicorn",
        "gunicorn",
        "click",
        "typer",
        "structlog",
        "loguru",
        "pyyaml",
        "tomli",
        "orjson",
        "polars",
        "numpy",
        "scipy",
        "scikit-learn",
    ],
    "port": [
        "8000",
        "3000",
        "5432",
        "8080",
        "5000",
        "443",
        "9090",
        "3001",
        "4000",
        "5555",
        "7070",
        "8443",
        "9200",
        "6379",
    ],
    "file": [
        "requirements.txt",
        "pyproject.toml",
        "package.json",
        "Dockerfile",
        "docker-compose.yml",
        "Cargo.toml",
        "go.mod",
        "Gemfile",
        "setup.cfg",
        "Pipfile",
        "poetry.lock",
    ],
    "n": ["3", "5", "10", "15", "20", "25", "50", "100"],
    "branch": [
        "feature/auth",
        "bugfix/login",
        "release/v2",
        "hotfix/security",
        "feature/payment",
        "develop",
        "feature/search",
        "fix/memory-leak",
        "feature/i18n",
        "refactor/api-layer",
        "feature/webhooks",
        "fix/timeout",
        "feature/caching",
        "release/v3.0",
    ],
    "branch1": [
        "main",
        "develop",
        "feature/auth",
        "feature/payment",
        "release/v2",
        "feature/search",
        "refactor/api-layer",
    ],
    "branch2": [
        "develop",
        "main",
        "feature/payment",
        "feature/webhooks",
        "release/v3.0",
        "fix/timeout",
        "feature/caching",
    ],
    "message": [
        "fix: resolve login bug on expired sessions",
        "feat: add user profile with avatar upload",
        "refactor: simplify API response serialization",
        "docs: update deployment guide for v2",
        "test: add integration tests for auth flow",
        "chore: upgrade dependencies to latest patch",
        "perf: cache database query results",
        "fix: handle unicode in file paths",
        "feat: add pagination to search results",
        "refactor: extract validation logic to shared module",
        "fix: prevent race condition in cache invalidation",
        "feat: add WebSocket support for real-time updates",
        "docs: add architecture decision records",
        "test: add property-based tests for parser",
        "chore: remove deprecated API endpoints",
    ],
    "author": [
        "alice",
        "bob",
        "charlie",
        "devbot",
        "david",
        "eve",
        "frank",
        "grace",
        "henry",
        "iris",
        "jack",
        "karen",
    ],
    "days": ["1", "3", "7", "14", "30", "60", "90", "180"],
    "version": [
        "v1.0.0",
        "v1.2.0",
        "v2.0.0",
        "v2.1.0",
        "v0.5.0-beta",
        "v0.9.0-rc1",
        "v3.0.0",
        "v1.5.2",
        "v2.3.1",
        "v0.1.0-alpha",
    ],
    "module": [
        "auth",
        "database",
        "api",
        "utils",
        "models",
        "views",
        "services",
        "handlers",
        "middleware",
        "cache",
        "queue",
        "scheduler",
        "notifications",
        "payments",
        "search",
    ],
    "topic": [
        "FastAPI authentication patterns",
        "Python async best practices",
        "pytest fixtures and parametrize",
        "Docker multi-stage builds",
        "git rebase vs merge workflows",
        "SQL query optimization",
        "Redis caching strategies",
        "WebSocket implementation",
        "REST API versioning",
        "GraphQL schema design",
        "TypeScript generic constraints",
        "Rust ownership and borrowing",
        "Go concurrency with goroutines",
        "Kubernetes ingress controllers",
        "Terraform state management",
        "CI/CD pipeline design",
        "observability with OpenTelemetry",
        "rate limiting algorithms",
        "event-driven architecture",
        "CQRS and event sourcing",
    ],
    "framework": [
        "FastAPI",
        "Flask",
        "Django REST",
        "Streamlit",
        "Click CLI",
        "Typer CLI",
        "Pyramid",
        "Tornado",
        "aiohttp",
        "Sanic",
        "Litestar",
        "Starlette",
    ],
    "language": LANGUAGES,
    "task_desc": [
        "calculates the factorial of a number",
        "fetches data from an API and caches the result",
        "parses a CSV file and returns summary statistics",
        "validates an email address with regex",
        "sorts a list of dictionaries by multiple keys",
        "implements a rate limiter using token bucket",
        "converts markdown to HTML with syntax highlighting",
        "generates a random password with constraints",
        "reads a JSON config and validates against a schema",
        "monitors a directory for new files and processes them",
        "scrapes a webpage and extracts structured data",
        "compresses files using multiple algorithms",
        "schedules periodic tasks with retry logic",
        "transforms XML to JSON preserving attributes",
        "calculates moving averages over a time series",
        "validates JWT tokens and extracts claims",
        "generates a sitemap from a list of URLs",
        "merges multiple PDF files into one",
        "sends notifications via multiple channels",
        "tracks usage metrics and exports to CSV",
        "performs fuzzy string matching on names",
        "downloads files concurrently with progress bars",
        "encrypts and decrypts files with AES-256",
        "parses log files and aggregates error counts",
        "builds a decision tree from training data",
    ],
    "data_structure": [
        "LRU cache",
        "binary search tree",
        "hash map",
        "priority queue",
        "trie",
        "graph with BFS/DFS",
        "red-black tree",
        "skip list",
        "bloom filter",
        "ring buffer",
        "union-find",
        "segment tree",
        "AVL tree",
        "min-heap",
        "max-heap",
        "deque",
        "linked list",
        "circular buffer",
        "suffix array",
    ],
    "features": [
        "insert, delete, search operations",
        "thread-safe operations with locks",
        "serialization support for JSON",
        "iteration protocol support",
        "capacity limits with eviction",
        "persistence to disk",
        "bulk batch operations",
        "event-driven notifications",
        "time-based expiration",
        "memory-mapped storage",
    ],
    "code": [
        "def foo(x): return x * 2",
        "async def fetch(): ...",
        "class User: def __init__(self): ...",
        "lambda x: x + 1",
        "result = [x for x in data if x > 0]",
        "try: parse(input) except: pass",
        "for item in items: process(item)",
        "return {k: v for k, v in data.items() if v}",
        "with open(f) as fh: data = fh.read()",
        "await asyncio.gather(*tasks)",
        "map(lambda x: x*2, range(10))",
        "if x is not None and x > 0: ...",
        "class Base: def method(self): raise NotImplementedError",
        "data = json.loads(response.text)['results']",
        "while True: value = queue.get()",
    ],
    "error": ERROR_TYPES,
    "error_msg": [f"{e}: unexpected argument" for e in ERROR_TYPES[:10]]
    + [f"{e}: invalid value" for e in ERROR_TYPES[10:20]]
    + [
        "ConnectionRefusedError: [Errno 61] Connection refused",
        "TimeoutError: operation timed out after 30s",
        "JSONDecodeError: Expecting property name enclosed in double quotes",
        "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff",
        "FileNotFoundError: [Errno 2] No such file or directory",
        "PermissionError: [Errno 13] Permission denied",
        "RecursionError: maximum recursion depth exceeded",
        "MemoryError: Unable to allocate array",
        "KeyError: 'user_id'",
        "ValueError: invalid literal for int()",
    ],
    "function": [
        "calculate_total",
        "parse_date",
        "validate_input",
        "transform_data",
        "fetch_user",
        "process_payment",
        "generate_report",
        "sync_database",
        "render_template",
        "compress_image",
        "authenticate_request",
        "log_event",
        "cache_result",
        "enqueue_task",
        "parse_config",
        "validate_schema",
        "merge_records",
        "export_csv",
    ],
    "input": [
        "empty string",
        "null value",
        "negative number",
        "unicode characters",
        "very large integer",
        "nested list",
        "mixed-type dictionary",
        "malformed JSON",
        "binary data",
        "string with newlines",
        "array of empty objects",
        "deeply nested structure",
        "circular reference",
    ],
    "condition": [
        "the input is empty",
        "the database is down",
        "the API times out",
        "concurrent requests arrive",
        "the file doesn't exist",
        "memory is exhausted",
        "the network is unstable",
        "the disk is full",
        "the config is missing",
        "the token is expired",
        "the queue is full",
        "the connection is reset",
    ],
    "endpoint": [
        "/api/users",
        "/api/tasks",
        "/api/search",
        "/api/auth/login",
        "/api/orders",
        "/api/products",
        "/api/webhooks",
        "/api/payments",
        "/api/notifications",
        "/api/reports",
        "/api/v2/users",
        "/api/v2/orders",
        "/api/health",
        "/api/metrics",
        "/api/batch/export",
    ],
    "component": [
        "UserList",
        "SearchBar",
        "DataGrid",
        "LoginForm",
        "NavBar",
        "SideBar",
        "Modal",
        "Toast",
        "DataTable",
        "ChartView",
        "FileUpload",
        "DatePicker",
        "Tabs",
        "Accordion",
        "Pagination",
        "Breadcrumb",
    ],
    "class_name": [
        "DataProcessor",
        "ApiClient",
        "CacheStore",
        "EventHandler",
        "UserRepository",
        "PaymentGateway",
        "NotificationManager",
        "SearchEngine",
        "TaskScheduler",
        "ConfigLoader",
        "StateManager",
        "ConnectionPool",
        "RateLimiter",
        "MetricsCollector",
        "AuditLogger",
        "SessionManager",
    ],
    "service": [
        "AuthService",
        "PaymentGateway",
        "NotificationService",
        "SearchEngine",
        "UserService",
        "OrderService",
        "InventoryService",
        "BillingService",
        "EmailService",
        "StorageService",
        "CacheService",
        "QueueService",
        "WebhookService",
        "AnalyticsService",
        "RateLimitService",
    ],
    "domain": [
        "e-commerce",
        "task management",
        "social media",
        "healthcare",
        "fintech",
        "education",
        "real estate",
        "travel booking",
        "food delivery",
        "ride sharing",
        "streaming media",
        "project management",
        "CRM",
        "inventory management",
    ],
    "platform": [
        "AWS",
        "GCP",
        "Azure",
        "Docker",
        "Kubernetes",
        "Vercel",
        "Netlify",
        "Heroku",
        "DigitalOcean",
        "Cloudflare",
        "Fly.io",
        "Railway",
        "Render",
        "Lambda",
        "Cloud Run",
    ],
    "task": [
        "implement user authentication",
        "optimize database queries",
        "add real-time notifications",
        "migrate to microservices",
        "build a search feature",
        "set up CI/CD pipeline",
        "add role-based access control",
        "implement audit logging",
        "create a data export feature",
        "add multi-tenancy support",
        "implement webhook delivery",
        "build an admin dashboard",
        "add internationalization support",
        "set up monitoring alerts",
        "implement feature flags",
        "build a plugin system",
    ],
    "old_system": [
        "monolith",
        "REST API",
        "SQLite",
        "manual testing",
        "single-tenant",
        "synchronous processing",
        "CSV export",
        "basic auth",
        "polling-based updates",
        "file-based config",
    ],
    "new_system": [
        "microservices",
        "GraphQL",
        "PostgreSQL",
        "automated CI/CD",
        "multi-tenant",
        "event-driven async",
        "real-time streaming",
        "OAuth 2.0",
        "WebSocket push",
        "centralized config",
    ],
    "git_op": GIT_OPERATIONS,
    "code_pattern": CODE_PATTERNS,
    "search_pattern": SEARCH_PATTERNS,
    "complexity": TASK_COMPLEXITIES,
    "test_framework": [
        "pytest",
        "unittest",
        "nose2",
        "tox",
        "nox",
        "jest",
        "vitest",
        "mocha",
        "cypress",
        "playwright",
        "go test",
        "cargo test",
        "JUnit",
        "RSpec",
        "minitest",
    ],
    "metric": [
        "latency",
        "throughput",
        "memory usage",
        "CPU usage",
        "error rate",
        "request count",
        "response time",
        "queue depth",
        "cache hit rate",
        "connection count",
    ],
    "format": [
        "JSON",
        "YAML",
        "XML",
        "CSV",
        "TOML",
        "protobuf",
        "MessagePack",
        "Avro",
        "parquet",
        "plain text",
    ],
    "env": ["production", "staging", "development", "testing", "preview"],
    "severity": ["critical", "high", "medium", "low", "info"],
}


# ---------------------------------------------------------------------------
# Task categories with templates.
# Each template is a 3-tuple: (template_string, expected_mode, difficulty 1-3)
# ---------------------------------------------------------------------------

CATEGORIES: dict[str, dict] = {
    "tool_use_file_ops": {
        "weight": 0.15,
        "templates": [
            ("Create a file at {path} with the following content: {content_desc}", "FAST", 1),
            ("Read the file {path} and summarize what it does", "FAST", 1),
            ("Edit {path} to replace '{old}' with '{new}'", "FAST", 1),
            ("List all files in {dir} that match pattern {pattern}", "FAST", 1),
            ("Find all {language} files in {dir} that contain the word '{keyword}'", "FAST", 1),
            ("Create a directory structure for a {project_type} project at {dir}", "FAST", 1),
            ("Copy the contents of {src} to {dst} and verify the copy succeeded", "VERIFY", 2),
            ("Delete all temporary files matching {pattern} in {dir}", "VERIFY", 2),
            ("Move {src} to {dst} and update any references to the old path", "THINK", 2),
            ("Create a backup of {path} with a timestamp suffix", "FAST", 1),
            ("Append the text '{new}' to the end of {path}", "FAST", 1),
            ("Rename {src} to {dst} across the entire project", "FAST", 1),
            ("Create a symlink from {src} to {dst}", "FAST", 1),
            ("Find all empty directories in {dir} and list them", "FAST", 1),
            ("Count the total lines of {language} code in {dir}", "FAST", 1),
            ("Extract all lines containing '{keyword}' from {path} into a report", "FAST", 1),
            (
                "Create a new {file_name} file with a standard template for {project_type}",
                "FAST",
                1,
            ),
            ("Split {path} into smaller files of at most {n} lines each", "THINK", 2),
            ("Merge all files matching {pattern} in {dir} into a single {file_name}", "THINK", 2),
            ("Check if {path} exists and create it with {content_desc} if not", "FAST", 1),
            ("Find the largest file in {dir} matching {pattern} and report its size", "FAST", 1),
            ("Replace all tabs with spaces in every {pattern} file under {dir}", "FAST", 1),
            ("Create a compressed archive of {dir} named {file_name}", "FAST", 1),
            ("Extract the contents of {file_name} into {dir}", "FAST", 1),
            ("Find duplicate files in {dir} by comparing content hashes", "THINK", 2),
            ("Set file permissions on {path} to be readable only by the owner", "VERIFY", 2),
            ("Create a patch file showing the differences between {src} and {dst}", "FAST", 1),
            ("Apply the patch from {file_name} to the files in {dir}", "THINK", 2),
            ("Watch {dir} for new {pattern} files and log each creation", "FAST", 1),
            ("Find all broken symlinks in {dir} and list their targets", "FAST", 1),
            ("Create a file at {path} containing {content_desc} with UTF-8 encoding", "FAST", 1),
            ("Read the first {n} lines of {path} and display them", "FAST", 1),
            ("Read the last {n} lines of {path} and display them", "FAST", 1),
            ("Sort the lines in {path} alphabetically and write back", "FAST", 1),
            ("Remove all blank lines from {path} and save the result", "FAST", 1),
            ("Find all files in {dir} modified in the last {days} days", "FAST", 1),
            ("Compare {src} and {dst} line by line and show differences", "FAST", 1),
            ("Create a directory at {dir} with parent directories if needed", "FAST", 1),
            ("Find all files larger than {n} MB in {dir} and list them", "FAST", 1),
            ("Convert {path} from tabs to spaces with a width of 4", "FAST", 1),
            ("Create a new file {path} with a {language} boilerplate template", "FAST", 1),
            (
                "Recursively list all files in {dir} with their sizes in human-readable format",
                "FAST",
                1,
            ),
            ("Find all files in {dir} with the {pattern} extension and count them", "FAST", 1),
            ("Create a {file_name} file in {dir} with {content_desc}", "FAST", 1),
            ("Truncate {path} to the first {n} lines and save", "VERIFY", 2),
            ("Find all hard links to {path} in the filesystem under {dir}", "THINK", 2),
            ("Create a sparse file at {path} of size {n} MB", "FAST", 1),
            (
                "Synchronize the contents of {src} to {dst} deleting files not in source",
                "VERIFY",
                3,
            ),
            ("Find all files in {dir} owned by a specific user and list them", "FAST", 1),
        ],
    },
    "tool_use_shell": {
        "weight": 0.12,
        "templates": [
            ("Run the test suite and report any failures", "FAST", 1),
            ("Check the git status of the current repository", "FAST", 1),
            ("Install the package {package} and verify it works", "FAST", 1),
            ("Run '{command}' and analyze the output for errors", "FAST", 1),
            ("Start a local development server on port {port}", "FAST", 1),
            ("Run the linter on {dir} and fix any issues found", "THINK", 2),
            ("Set up a virtual environment and install all dependencies from {file}", "FAST", 1),
            ("Kill the process running on port {port} and restart it", "VERIFY", 2),
            ("Run '{command}' in the background and capture its PID", "FAST", 1),
            ("Check the system disk usage and report if any partition is over 80%", "FAST", 1),
            ("List all running processes that match '{keyword}' and their memory usage", "FAST", 1),
            ("Check the current environment variables and filter for '{keyword}'", "FAST", 1),
            ("Run '{command}' with a timeout of {n} seconds and capture output", "FAST", 1),
            ("Start a background job to monitor {dir} for changes", "FAST", 1),
            ("Execute '{command}' and pipe the output through grep for '{keyword}'", "FAST", 1),
            (
                "Run the {test_framework} suite with verbose output and save to {file_name}",
                "FAST",
                1,
            ),
            ("Check if port {port} is in use and identify the process", "FAST", 1),
            ("Create a cron job that runs '{command}' every {n} minutes", "THINK", 2),
            (
                "Run '{command}' and save stdout to {file_name} and stderr to a separate file",
                "FAST",
                1,
            ),
            ("Check the system load average and alert if above {n}", "FAST", 1),
            ("Run '{command}' with elevated privileges and log the result", "VERIFY", 2),
            ("Find all listening TCP ports and the processes using them", "FAST", 1),
            ("Execute '{command}' in a subshell and capture the exit code", "FAST", 1),
            ("Run '{command}' and format the output as {format}", "THINK", 2),
            ("Start a tmux session named '{keyword}' running '{command}'", "FAST", 1),
            ("Run '{command}' repeatedly {n} times and report the average duration", "THINK", 2),
            ("Check the network connectivity to localhost:{port} and report latency", "FAST", 1),
            ("Run '{command}' with environment variable {keyword}=test set", "FAST", 1),
            ("List all Docker containers and their status", "FAST", 1),
            ("Run '{command}' and count the number of lines in the output", "FAST", 1),
            ("Execute '{command}' and redirect both stdout and stderr to {file_name}", "FAST", 1),
            ("Check the uptime of the system and report the load average", "FAST", 1),
            ("Run '{command}' in {dir} and capture the output to a variable", "FAST", 1),
            ("Start a long-running process with nohup and log to {file_name}", "FAST", 1),
            ("Run '{command}' and exit immediately if it fails with a non-zero code", "FAST", 1),
            ("Check the available memory and warn if below {n} MB", "FAST", 1),
            ("Run '{command}' with xargs over all files matching {pattern} in {dir}", "THINK", 2),
            ("Execute '{command}' in a Docker container and mount {dir} as a volume", "THINK", 2),
            (
                "Run '{command}' and filter the output to show only lines with '{keyword}'",
                "FAST",
                1,
            ),
            ("Check the version of {package} and compare with the latest release", "FAST", 1),
            ("Run '{command}' with a retry logic of {n} attempts on failure", "THINK", 2),
            ("List all environment variables containing '{keyword}' in their name", "FAST", 1),
            ("Run '{command}' and sort the output alphabetically", "FAST", 1),
            (
                "Start a process that watches {file} for modifications and runs '{command}'",
                "THINK",
                2,
            ),
            ("Run '{command}' and extract the {n}th column from the output", "FAST", 1),
            ("Check the open file descriptors for the process matching '{keyword}'", "FAST", 1),
            ("Run '{command}' and tee the output to both the terminal and {file_name}", "FAST", 1),
            ("Execute '{command}' in parallel across {n} worker processes", "THINK", 2),
            ("Run '{command}' and report the {metric} of the execution", "THINK", 2),
        ],
    },
    "tool_use_git": {
        "weight": 0.10,
        "templates": [
            ("Show the last {n} commits in the repository", "FAST", 1),
            ("Create a new branch called {branch} and switch to it", "FAST", 1),
            ("Stage all changes and commit with message: {message}", "FAST", 1),
            ("Show the diff between {branch1} and {branch2}", "FAST", 1),
            ("Merge {branch} into the current branch and resolve any conflicts", "THINK", 2),
            ("Revert the last commit and explain why", "VERIFY", 2),
            ("Find all commits by {author} in the last {days} days", "FAST", 1),
            ("Create a tag {version} for the current commit", "FAST", 1),
            ("Show the git log for {file_name} with patch diffs", "FAST", 1),
            ("Stash the current changes with the message '{message}'", "FAST", 1),
            ("Pop the most recent stash and resolve any conflicts", "THINK", 2),
            ("Cherry-pick commit {version} from {branch} into the current branch", "THINK", 2),
            ("Rebase the current branch onto {branch1} and handle conflicts", "THINK", 3),
            ("Show all branches that have been merged into {branch1}", "FAST", 1),
            ("Delete the local branch {branch} after verifying it's merged", "VERIFY", 2),
            ("Fetch all remote branches and prune deleted ones", "FAST", 1),
            ("Show the blame for {file_name} to identify who wrote each line", "FAST", 1),
            ("Create an annotated tag {version} with message '{message}'", "FAST", 1),
            ("Show all commits that modified {file_name} in {dir}", "FAST", 1),
            ("Reset the working directory to the state of {branch1} (soft reset)", "VERIFY", 2),
            ("Find all branches that contain the commit with message '{keyword}'", "FAST", 1),
            ("Squash the last {n} commits into one with message: {message}", "THINK", 3),
            ("Show the reflog to find recently lost commits", "FAST", 1),
            ("Amend the last commit to include the message: {message}", "FAST", 1),
            ("Bisect to find which commit introduced the bug '{keyword}'", "THINK", 3),
            ("Show the diff of {file_name} between {branch1} and {branch2}", "FAST", 1),
            ("List all tags sorted by version number", "FAST", 1),
            ("Create a new remote called '{keyword}' pointing to {dir}", "FAST", 1),
            ("Push {branch} to the remote and set up tracking", "FAST", 1),
            ("Pull from {branch1} with a rebase instead of a merge", "THINK", 2),
            ("Show all files changed in the last {n} commits", "FAST", 1),
            ("Find commits that added or removed the word '{keyword}'", "FAST", 1),
            ("Archive the current branch into {file_name}", "FAST", 1),
            ("Show the short log with author statistics for the last {days} days", "FAST", 1),
            ("Resolve a merge conflict in {file_name} by keeping both changes", "THINK", 2),
            ("Check if the current branch is ahead or behind {branch1}", "FAST", 1),
            ("Show the commit history as a graph with the last {n} commits", "FAST", 1),
            ("Create a branch {branch} from a specific commit {version}", "FAST", 1),
            ("Find all files that have been renamed in the last {n} commits", "THINK", 2),
            ("Show the diff stat for all changes between {branch1} and {branch2}", "FAST", 1),
            ("Stage only the files matching {pattern} in {dir}", "FAST", 1),
            ("Unstage all changes and return to the HEAD state", "FAST", 1),
            ("Show which branches are not yet pushed to the remote", "FAST", 1),
            ("Find the merge-base of {branch1} and {branch2}", "FAST", 1),
            ("Create a worktree at {dir} for the branch {branch}", "THINK", 2),
            ("Show all submodules and their current commit hashes", "FAST", 1),
            ("Clean all untracked files in {dir} matching {pattern}", "VERIFY", 2),
            ("Show the commit that last modified line {n} in {file_name}", "FAST", 1),
            ("Restore {file_name} to its state in commit {version}", "VERIFY", 2),
        ],
    },
    "tool_use_search": {
        "weight": 0.08,
        "templates": [
            ("Search the codebase for all occurrences of '{keyword}'", "FAST", 1),
            ("Find all files that import {module}", "FAST", 1),
            ("Search for TODO comments in all {language} files", "FAST", 1),
            ("Find all {search_pattern} in {dir}", "FAST", 1),
            ("Search the web for documentation on {topic}", "FAST", 1),
            ("Find all test files that test the {module} module", "FAST", 1),
            ("Grep for error handling patterns in {dir}", "THINK", 2),
            ("Search for potential security vulnerabilities in {dir}", "THINK", 2),
            ("Find all {code_pattern} definitions in {dir}", "FAST", 1),
            ("Search for all uses of the {function} function across the codebase", "FAST", 1),
            ("Find all files in {dir} that contain '{keyword}' in their content", "FAST", 1),
            ("Search for deprecated API usage and list each occurrence", "THINK", 2),
            ("Find all {language} files that do not have corresponding test files", "THINK", 2),
            ("Search for all hardcoded strings that should be constants", "THINK", 2),
            ("Find all TODO and FIXME comments and group them by file", "FAST", 1),
            ("Search for all occurrences of '{keyword}' with {n} lines of context", "FAST", 1),
            ("Find all files that reference the {class_name} class", "FAST", 1),
            ("Search for all SQL queries in {dir} and list them", "THINK", 2),
            ("Find all async functions that don't have error handling", "THINK", 2),
            ("Search for all environment variable references in the codebase", "FAST", 1),
            ("Find all duplicate code blocks of at least {n} lines", "THINK", 3),
            ("Search for all uses of {package} and check if the version is current", "THINK", 2),
            ("Find all files in {dir} that are larger than {n} lines", "FAST", 1),
            ("Search for all regular expressions in the codebase and validate them", "THINK", 2),
            ("Find all functions with more than {n} parameters", "FAST", 1),
            ("Search for all comments containing '{keyword}' in {language} files", "FAST", 1),
            ("Find all files that import from {module} and list the imported names", "FAST", 1),
            ("Search for all print statements that should be logging calls", "THINK", 2),
            ("Find all unused imports in {dir} across {language} files", "FAST", 1),
            ("Search for all magic numbers in {dir} and suggest named constants", "THINK", 2),
            ("Find all files that have been modified in the last {days} days", "FAST", 1),
            ("Search for all uses of eval() or exec() in the codebase", "THINK", 2),
            ("Find all functions that return more than one type", "THINK", 2),
            ("Search for all password or token references in {dir}", "THINK", 2),
            ("Find all circular import dependencies in the project", "THINK", 3),
            ("Search for all files that match {pattern} and contain '{keyword}'", "FAST", 1),
            ("Find all places where exceptions are caught and silently ignored", "THINK", 2),
            ("Search for all uses of global variables in {dir}", "FAST", 1),
            ("Find all test functions that don't have assertions", "THINK", 2),
            ("Search for all TODO comments assigned to {author}", "FAST", 1),
            ("Find all files with no documentation or docstrings", "FAST", 1),
            (
                "Search for all uses of mutable default arguments in {language} functions",
                "THINK",
                2,
            ),
            ("Find all functions with cyclomatic complexity above {n}", "THINK", 2),
            ("Search for all bare except clauses in {dir}", "FAST", 1),
            ("Find all files that don't have a license header", "FAST", 1),
            ("Search for all uses of deprecated {language} features", "THINK", 2),
            ("Find all network calls that don't have a timeout set", "THINK", 2),
            ("Search for all hardcoded file paths in {dir}", "FAST", 1),
            ("Find all instances where {error} might be raised but isn't caught", "THINK", 2),
        ],
    },
    "coding_write": {
        "weight": 0.15,
        "templates": [
            ("Write a {language} function that {task_desc}", "FAST", 1),
            ("Create a {framework} application that {task_desc}", "THINK", 2),
            ("Implement a {data_structure} in {language} with {features}", "THINK", 2),
            ("Write a script that {task_desc} and saves the result to {file_name}", "FAST", 1),
            ("Create a REST API endpoint that {task_desc}", "THINK", 2),
            ("Write a CLI tool that {task_desc}", "THINK", 2),
            (
                "Implement a {class_name} class that {task_desc} with proper error handling",
                "THINK",
                2,
            ),
            ("Write a {code_pattern} that {task_desc}", "FAST", 1),
            ("Create a {language} module that {task_desc} with comprehensive logging", "THINK", 2),
            ("Write a {language} script that {task_desc} and handles edge cases", "THINK", 2),
            ("Implement a {language} {code_pattern} for {task_desc}", "FAST", 1),
            ("Create a web scraper that {task_desc} with rate limiting", "THINK", 2),
            ("Write a {language} function that {task_desc} using {package}", "FAST", 1),
            ("Build a {project_type} that {task_desc}", "THINK", 3),
            ("Implement a {language} service that {task_desc} with retry logic", "THINK", 2),
            ("Write a {language} utility that {task_desc} and exports to {format}", "FAST", 1),
            ("Create a {language} library that {task_desc} with a clean public API", "THINK", 3),
            (
                "Write a {language} program that {task_desc} with configuration from {file}",
                "FAST",
                1,
            ),
            ("Implement a {language} handler that {task_desc} with proper validation", "THINK", 2),
            ("Create a {language} plugin for {project_type} that {task_desc}", "THINK", 3),
            ("Write a {language} decorator that {task_desc}", "FAST", 1),
            ("Implement a {language} context manager that {task_desc}", "FAST", 1),
            ("Create a {language} generator that {task_desc} lazily", "THINK", 2),
            ("Write a {language} async function that {task_desc} concurrently", "THINK", 2),
            ("Build a {language} middleware that {task_desc}", "THINK", 2),
            ("Implement a {language} data class that {task_desc} with serialization", "FAST", 1),
            ("Write a {language} enum that {task_desc}", "FAST", 1),
            ("Create a {language} protocol that defines the interface for {task_desc}", "THINK", 2),
            ("Write a {language} mixin that {task_desc}", "FAST", 1),
            ("Implement a {language} factory that {task_desc}", "FAST", 1),
            ("Create a {language} adapter that {task_desc} for {package}", "THINK", 2),
            ("Write a {language} proxy that {task_desc} with lazy loading", "THINK", 2),
            ("Build a {language} observer that {task_desc} with event notifications", "THINK", 2),
            ("Implement a {language} iterator that {task_desc}", "FAST", 1),
            ("Write a {language} property descriptor that {task_desc}", "FAST", 1),
            ("Create a {language} abstract base class for {task_desc}", "THINK", 2),
            ("Write a {language} generic class that {task_desc} with type parameters", "THINK", 2),
            ("Implement a {language} singleton that {task_desc} thread-safely", "THINK", 2),
            ("Create a {language} command-line parser that {task_desc}", "FAST", 1),
            ("Write a {language} configuration loader that {task_desc} from {format}", "FAST", 1),
            ("Build a {language} scheduler that {task_desc} at regular intervals", "THINK", 2),
            ("Implement a {language} cache that {task_desc} with TTL support", "THINK", 2),
            (
                "Write a {language} validator that {task_desc} with detailed error messages",
                "FAST",
                1,
            ),
            ("Create a {language} parser that {task_desc} from {format} input", "THINK", 2),
            ("Write a {language} serializer that converts objects to {format}", "FAST", 1),
            ("Implement a {language} rate limiter that {task_desc}", "THINK", 2),
            ("Build a {language} circuit breaker that {task_desc}", "THINK", 3),
            (
                "Write a {language} retry mechanism that {task_desc} with exponential backoff",
                "THINK",
                2,
            ),
            ("Create a {language} health check endpoint that {task_desc}", "FAST", 1),
        ],
    },
    "coding_debug": {
        "weight": 0.12,
        "templates": [
            ("The following code has a bug: {code}. Find and fix it.", "THINK", 2),
            ("Tests are failing with error: {error_msg}. Debug and fix the issue.", "RECOVER", 2),
            ("The function {function} returns wrong results for {input}. Fix it.", "THINK", 2),
            ("Fix the type error in: {code}", "FAST", 1),
            (
                "The application crashes when {condition}. Find and fix the root cause.",
                "RECOVER",
                2,
            ),
            ("Memory leak detected in {module}. Identify and fix it.", "THINK", 2),
            ("Race condition in {code}. Fix the concurrency issue.", "THINK", 3),
            ("The API returns 500 errors intermittently. Debug the issue.", "RECOVER", 2),
            (
                "The {function} function throws {error} when called with {input}. Fix it.",
                "THINK",
                2,
            ),
            ("Debug the {error} that occurs in {module} during {condition}", "RECOVER", 2),
            ("The {language} code in {file_name} has a logic error. Find and fix it.", "THINK", 2),
            ("Fix the {error} in the following code: {code}", "FAST", 1),
            (
                "The test for {function} is flaky. Identify and fix the race condition.",
                "RECOVER",
                3,
            ),
            ("Debug why {function} returns None instead of the expected value", "THINK", 2),
            (
                "The {endpoint} endpoint returns incorrect status codes. Fix the routing.",
                "THINK",
                2,
            ),
            ("Fix the off-by-one error in: {code}", "FAST", 1),
            ("The {class_name} class has a memory leak. Identify and fix it.", "THINK", 2),
            ("Debug the {error_msg} that occurs when processing {input}", "RECOVER", 2),
            (
                "The {language} script hangs when {condition}. Find and fix the deadlock.",
                "THINK",
                3,
            ),
            ("Fix the {error} caused by {condition} in {module}", "FAST", 1),
            ("The {service} service times out when {condition}. Optimize and fix.", "RECOVER", 2),
            ("Debug why the {function} function is {n}x slower than expected", "THINK", 2),
            ("Fix the encoding issue when {function} processes {input}", "FAST", 1),
            (
                "The {module} module fails to import on {env}. Debug the dependency issue.",
                "RECOVER",
                2,
            ),
            ("Fix the {error} in the async code: {code}", "THINK", 2),
            ("Debug the infinite loop in {code} when {condition}", "THINK", 2),
            ("The {function} function has incorrect error handling. Fix it.", "FAST", 1),
            ("Fix the {error} that occurs during {git_op} in {dir}", "RECOVER", 2),
            ("Debug why {function} produces different results on {env} vs {env}", "THINK", 3),
            (
                "The {class_name} class doesn't handle {input} correctly. Fix the validation.",
                "THINK",
                2,
            ),
            ("Fix the resource leak in {code} where {condition}", "THINK", 2),
            ("Debug the {error_msg} in the {language} code at {file_name}", "RECOVER", 2),
            ("The {endpoint} endpoint fails when {condition}. Fix the error handling.", "THINK", 2),
            ("Fix the {error} in the regex pattern used by {function}", "FAST", 1),
            (
                "Debug why the {test_framework} test for {function} fails on CI but passes locally",
                "RECOVER",
                3,
            ),
            ("The {module} module has a circular import. Refactor to fix it.", "THINK", 2),
            ("Fix the {error} when {function} is called with {input}", "FAST", 1),
            (
                "Debug the {error_msg} that occurs during concurrent execution of {function}",
                "RECOVER",
                3,
            ),
            (
                "The {class_name} class violates the Liskov substitution principle. Fix it.",
                "THINK",
                3,
            ),
            ("Fix the {error} in the serialization logic of {class_name}", "THINK", 2),
            (
                "Debug why {function} returns inconsistent results across {language} versions",
                "THINK",
                3,
            ),
            (
                "The {service} service leaks database connections. Find and fix the leak.",
                "RECOVER",
                2,
            ),
            ("Fix the {error} in the {code_pattern} implementation: {code}", "THINK", 2),
            ("Debug the {error_msg} when the {module} module processes {input}", "RECOVER", 2),
            ("The {function} function has an incorrect default argument. Fix it.", "FAST", 1),
            ("Fix the {error} caused by improper exception handling in {code}", "THINK", 2),
            ("Debug why the {component} component re-renders infinitely", "THINK", 2),
            (
                "The {class_name} class has a thread-safety issue. Fix it with proper locking.",
                "THINK",
                3,
            ),
            ("Fix the {error} when {function} encounters {input} in {language}", "FAST", 1),
        ],
    },
    "coding_refactor": {
        "weight": 0.08,
        "templates": [
            ("Refactor {code} to be more readable and maintainable", "THINK", 2),
            ("Extract the duplicated logic in {code} into a shared function", "FAST", 1),
            ("Convert {code} from callbacks to async/await", "THINK", 2),
            ("Add type hints to {code} and fix any type errors", "FAST", 1),
            ("Simplify the conditional logic in {code}", "FAST", 1),
            ("Refactor the class hierarchy in {module} to use composition", "THINK", 3),
            ("Optimize {code} for better performance", "THINK", 2),
            ("Convert {code} from class-based to functional style", "THINK", 2),
            ("Refactor {file_name} to reduce cyclomatic complexity", "THINK", 2),
            ("Extract the {code_pattern} from {module} into its own module", "FAST", 1),
            ("Replace the nested conditionals in {code} with guard clauses", "FAST", 1),
            ("Refactor {function} to use early returns instead of deep nesting", "FAST", 1),
            ("Convert the {class_name} class from inheritance to composition", "THINK", 3),
            ("Refactor {code} to use {language} idiomatic patterns", "THINK", 2),
            ("Extract the magic numbers in {code} into named constants", "FAST", 1),
            ("Refactor {module} to remove circular dependencies", "THINK", 3),
            ("Replace the global variables in {code} with dependency injection", "THINK", 2),
            ("Refactor {code} to use the strategy pattern instead of conditionals", "THINK", 3),
            ("Convert the synchronous {function} to async with proper error handling", "THINK", 2),
            ("Refactor {code} to eliminate code duplication using generics", "THINK", 2),
            ("Replace the manual loops in {code} with comprehensions or map/filter", "FAST", 1),
            (
                "Refactor the {class_name} class to follow the single responsibility principle",
                "THINK",
                2,
            ),
            (
                "Extract the configuration logic from {module} into a separate config module",
                "FAST",
                1,
            ),
            (
                "Refactor {code} to use the builder pattern for complex object construction",
                "THINK",
                3,
            ),
            ("Convert the error handling in {code} from error codes to exceptions", "THINK", 2),
            ("Refactor {function} to accept a protocol instead of a concrete class", "THINK", 2),
            ("Replace the string concatenation in {code} with f-strings or format", "FAST", 1),
            ("Refactor {module} to use a repository pattern for data access", "THINK", 3),
            ("Convert the callback-based {function} to return a promise/future", "THINK", 2),
            ("Refactor {code} to use the factory method pattern", "THINK", 2),
            ("Extract the validation logic from {function} into a separate validator", "FAST", 1),
            ("Refactor {code} to use the observer pattern for event handling", "THINK", 3),
            ("Replace the if-elif chain in {code} with a dispatch dictionary", "FAST", 1),
            ("Refactor the {class_name} class to use the state pattern", "THINK", 3),
            ("Convert the mutable default arguments in {code} to use None sentinels", "FAST", 1),
            ("Refactor {module} to separate business logic from infrastructure code", "THINK", 3),
            ("Replace the manual parsing in {code} with a proper parser library", "THINK", 2),
            ("Refactor {code} to use the command pattern for undoable operations", "THINK", 3),
            ("Extract the logging setup from {module} into a dedicated logging config", "FAST", 1),
            ("Refactor {function} to use a pipeline pattern instead of nested calls", "THINK", 2),
            (
                "Convert the singleton {class_name} to use a dependency injection container",
                "THINK",
                3,
            ),
            ("Refactor {code} to use the template method pattern", "THINK", 2),
            ("Replace the hardcoded values in {code} with a configuration provider", "FAST", 1),
            ("Refactor {module} to use the unit of work pattern for transactions", "THINK", 3),
            ("Convert the synchronous I/O in {code} to non-blocking async I/O", "THINK", 2),
            ("Refactor {code} to use the chain of responsibility pattern", "THINK", 3),
            (
                "Extract the test data generation from {function} into a fixture or factory",
                "FAST",
                1,
            ),
            ("Refactor {code} to use the decorator pattern for cross-cutting concerns", "THINK", 2),
            ("Replace the boolean parameters in {function} with an enum or flag object", "FAST", 1),
        ],
    },
    "coding_test": {
        "weight": 0.08,
        "templates": [
            ("Write unit tests for the {function} function", "FAST", 1),
            ("Write integration tests for the {endpoint} API endpoint", "THINK", 2),
            ("Create test fixtures for testing {component}", "FAST", 1),
            ("Write property-based tests for {function}", "THINK", 2),
            ("Add test coverage for the error handling in {module}", "FAST", 1),
            ("Write tests for edge cases: empty input, null, overflow, unicode", "THINK", 2),
            ("Create a test suite for the {class_name} class with >90% coverage", "THINK", 2),
            ("Write mock tests for the {service} service", "FAST", 1),
            ("Write {test_framework} tests for the {function} function", "FAST", 1),
            ("Create end-to-end tests for the {endpoint} endpoint flow", "THINK", 3),
            ("Write parameterized tests for {function} with multiple inputs", "FAST", 1),
            ("Add snapshot tests for the {component} component", "FAST", 1),
            ("Write tests verifying that {function} raises {error} on {input}", "FAST", 1),
            ("Create a test harness for the {module} module with setup and teardown", "THINK", 2),
            ("Write performance tests for {function} measuring {metric}", "THINK", 2),
            ("Add regression tests for the bug fix in {function}", "FAST", 1),
            ("Write tests for the {class_name} class including all public methods", "THINK", 2),
            (
                "Create mock objects for the {service} service and write interaction tests",
                "THINK",
                2,
            ),
            ("Write tests for the {function} function covering all branches", "THINK", 2),
            ("Add fuzz tests for the {function} function with random {input}", "THINK", 3),
            ("Write contract tests for the {endpoint} API ensuring schema compliance", "THINK", 2),
            ("Create test doubles for {module} and verify {function} behavior", "THINK", 2),
            ("Write tests for the {code_pattern} in {module} with various inputs", "FAST", 1),
            ("Add tests for the concurrency behavior of {class_name}", "THINK", 3),
            ("Write tests for the {function} function with mocked time and I/O", "THINK", 2),
            ("Create a test suite for the {language} API in {file_name}", "THINK", 2),
            ("Write tests verifying the {function} function handles {condition}", "FAST", 1),
            ("Add tests for the {service} service retry logic on {error}", "THINK", 2),
            ("Write tests for the {module} module's configuration validation", "FAST", 1),
            ("Create integration tests for the {module} to {module} interaction", "THINK", 2),
            ("Write tests for the {function} function's serialization to {format}", "FAST", 1),
            ("Add tests for the {class_name} class's equality and hashing", "FAST", 1),
            (
                "Write tests for the {endpoint} endpoint's authentication and authorization",
                "THINK",
                2,
            ),
            ("Create tests for the {function} function's caching behavior", "THINK", 2),
            (
                "Write tests for the {module} module's graceful degradation on {condition}",
                "THINK",
                2,
            ),
            (
                "Add tests for the {function} function's input validation and sanitization",
                "FAST",
                1,
            ),
            ("Write tests for the {class_name} class's lifecycle and cleanup", "THINK", 2),
            ("Create tests for the {service} service's circuit breaker behavior", "THINK", 3),
            ("Write tests for the {function} function's rate limiting logic", "THINK", 2),
            ("Add tests for the {module} module's event handling and dispatch", "THINK", 2),
            ("Write tests for the {function} function's timeout and cancellation", "THINK", 2),
            ("Create tests for the {class_name} class's thread safety under load", "THINK", 3),
            ("Write tests for the {endpoint} endpoint's pagination and sorting", "FAST", 1),
            ("Add tests for the {function} function's error recovery on {error}", "THINK", 2),
            ("Write tests for the {module} module's logging and observability", "FAST", 1),
            ("Create tests for the {service} service's health check and readiness", "FAST", 1),
            ("Write tests for the {function} function's behavior with {input}", "FAST", 1),
            (
                "Add tests for the {class_name} class's configuration loading from {format}",
                "THINK",
                2,
            ),
            ("Write tests for the {module} module's graceful shutdown on signal", "THINK", 2),
        ],
    },
    "planning": {
        "weight": 0.07,
        "templates": [
            ("Plan the architecture for a {project_type} project", "THINK", 2),
            ("Design the database schema for a {domain} application", "THINK", 2),
            ("Create a deployment plan for {project_type} to {platform}", "THINK", 2),
            ("Break down the task '{task}' into subtasks and estimate effort", "THINK", 2),
            ("Design the API contract for a {service} service", "THINK", 2),
            ("Plan the migration from {old_system} to {new_system}", "THINK", 3),
            ("Design a CI/CD pipeline for a {project_type} project", "THINK", 2),
            ("Create a testing strategy for a {domain} application", "THINK", 2),
            ("Plan the implementation of {task} with a phased approach", "THINK", 3),
            ("Design the data model for a {domain} platform with multi-tenancy", "THINK", 3),
            ("Create a technical specification for {task}", "THINK", 2),
            ("Plan the API versioning strategy for a {service} service", "THINK", 2),
            ("Design the caching layer for a {domain} application using {package}", "THINK", 2),
            ("Create a rollout plan for {task} with feature flags", "THINK", 3),
            ("Plan the observability strategy for {project_type} on {platform}", "THINK", 2),
            ("Design the authentication and authorization flow for a {domain} app", "THINK", 3),
            (
                "Create a data migration plan from {old_system} to {new_system} with zero downtime",
                "THINK",
                3,
            ),
            ("Plan the microservices decomposition of a {project_type} monolith", "THINK", 3),
            ("Design the event-driven architecture for a {domain} notification system", "THINK", 3),
            ("Create a capacity planning document for {service} on {platform}", "THINK", 2),
            ("Plan the implementation of {task} with risk assessment", "THINK", 2),
            ("Design the API gateway pattern for a {project_type} backend", "THINK", 3),
            ("Create a disaster recovery plan for {service} on {platform}", "THINK", 3),
            (
                "Plan the transition from {old_system} to {new_system} with backward compatibility",
                "THINK",
                3,
            ),
            ("Design the search infrastructure for a {domain} application", "THINK", 2),
            ("Create a technical debt remediation plan for {module}", "THINK", 2),
            ("Plan the scaling strategy for {service} to handle {n}x traffic growth", "THINK", 3),
            ("Design the message queue architecture for a {domain} system", "THINK", 3),
            ("Create a security audit plan for a {project_type} application", "THINK", 2),
            ("Plan the implementation of {task} with team assignments and timeline", "THINK", 2),
            ("Design the data pipeline for a {domain} analytics platform", "THINK", 3),
            ("Create a rollback strategy for deploying {project_type} to {platform}", "THINK", 2),
            ("Plan the API documentation strategy for a {service} service", "THINK", 2),
            ("Design the WebSocket infrastructure for real-time {domain} features", "THINK", 3),
            ("Create a load testing plan for {service} on {platform}", "THINK", 2),
            ("Plan the multi-region deployment of {project_type} on {platform}", "THINK", 3),
            ("Design the configuration management strategy for {project_type}", "THINK", 2),
            ("Create a monitoring and alerting plan for {service}", "THINK", 2),
            ("Plan the database sharding strategy for a {domain} application", "THINK", 3),
            ("Design the file upload and storage system for a {domain} platform", "THINK", 2),
            ("Create a plan for implementing {task} with A/B testing", "THINK", 2),
            ("Plan the transition to event sourcing for {service}", "THINK", 3),
            ("Design the rate limiting architecture for a {domain} API", "THINK", 2),
            ("Create a plan for adding {language} support to {project_type}", "THINK", 2),
            ("Plan the zero-downtime deployment strategy for {service}", "THINK", 3),
            ("Design the audit logging system for a {domain} compliance requirement", "THINK", 2),
            ("Create a plan for implementing {task} with canary releases", "THINK", 3),
            ("Plan the data backup and recovery strategy for {domain} on {platform}", "THINK", 2),
            ("Design the webhook delivery system for a {service} service", "THINK", 3),
        ],
    },
    "recovery": {
        "weight": 0.05,
        "templates": [
            ("The previous approach failed. Try a different strategy to {task}", "RECOVER", 2),
            (
                "The build is broken after the last commit. Identify and fix the issue.",
                "RECOVER",
                2,
            ),
            ("The deployment failed with error: {error_msg}. Recover and redeploy.", "RECOVER", 2),
            ("Tests were passing but now fail. Find what changed and fix it.", "RECOVER", 2),
            ("The database migration failed halfway. Recover and complete it.", "RECOVER", 2),
            (
                "After the refactor, performance degraded. Identify and fix the regression.",
                "RECOVER",
                2,
            ),
            (
                "The API is returning incorrect data after the last update. Roll back and fix.",
                "ESCALATE",
                3,
            ),
            (
                "Security vulnerability detected. Assess severity and recommend action.",
                "ESCALATE",
                3,
            ),
            ("The {service} service is down. Diagnose and restore it.", "RECOVER", 2),
            (
                "The {env} environment is unreachable. Investigate and restore connectivity.",
                "RECOVER",
                2,
            ),
            (
                "A {severity} alert fired for {metric} on {service}. Investigate and resolve.",
                "RECOVER",
                2,
            ),
            (
                "The {git_op} operation left the repository in a bad state. Recover it.",
                "RECOVER",
                2,
            ),
            ("Data corruption detected in {module}. Assess damage and recover.", "ESCALATE", 3),
            ("The {function} function is causing {error} in {env}. Fix and verify.", "RECOVER", 2),
            ("The cache is serving stale data after the update. Invalidate and fix.", "RECOVER", 2),
            ("The {endpoint} endpoint is returning 502 errors. Diagnose and fix.", "RECOVER", 2),
            (
                "A failed deployment left {env} in an inconsistent state. Roll back safely.",
                "ESCALATE",
                3,
            ),
            (
                "The {language} build fails on {platform} but works locally. Debug and fix.",
                "RECOVER",
                2,
            ),
            ("The {module} module is leaking memory in {env}. Identify and patch.", "RECOVER", 2),
            ("After the {git_op}, {file_name} has conflicts. Resolve and restore.", "RECOVER", 2),
            (
                "The {service} service is in a crash loop. Diagnose and break the cycle.",
                "RECOVER",
                2,
            ),
            ("A bad config push broke {env}. Revert and implement safeguards.", "ESCALATE", 3),
            ("The {function} function corrupted data in {env}. Assess and recover.", "ESCALATE", 3),
            (
                "The {test_framework} suite is timing out on CI. Identify the hanging test.",
                "RECOVER",
                2,
            ),
            (
                "The {platform} deployment rolled back automatically. Investigate the cause.",
                "RECOVER",
                2,
            ),
            (
                "A {severity} incident: {service} is losing data. Contain and recover.",
                "ESCALATE",
                3,
            ),
            (
                "The {module} module's import is broken after a dependency upgrade. Fix it.",
                "RECOVER",
                2,
            ),
            ("The {env} database is locked. Diagnose the lock and restore access.", "RECOVER", 2),
            (
                "After the hotfix, {endpoint} returns {error}. Revert and apply correctly.",
                "RECOVER",
                2,
            ),
            (
                "The {service} service can't connect to {platform}. Restore the connection.",
                "RECOVER",
                2,
            ),
            (
                "A bad migration left {module} in a partial state. Roll back and re-apply.",
                "RECOVER",
                2,
            ),
            (
                "The {function} function is stuck in an infinite retry loop. Break it safely.",
                "RECOVER",
                2,
            ),
            (
                "The {env} environment has drift from the expected config. Detect and fix.",
                "THINK",
                2,
            ),
            (
                "A {severity} bug in {class_name} is causing data loss. Patch immediately.",
                "ESCALATE",
                3,
            ),
            (
                "The {service} service's health check is failing. Diagnose and restore.",
                "RECOVER",
                2,
            ),
            (
                "After the {language} upgrade, {module} no longer compiles. Fix the build.",
                "RECOVER",
                2,
            ),
            ("The {endpoint} endpoint's SSL certificate expired. Renew and restore.", "RECOVER", 2),
            ("A failed {git_op} lost uncommitted work. Recover from the reflog.", "RECOVER", 2),
            ("The {env} environment's disk is full. Clean up and restore services.", "RECOVER", 2),
            (
                "The {service} service is overwhelmed by a traffic spike. Scale and stabilize.",
                "RECOVER",
                2,
            ),
            ("A {severity} regression in {function} broke {env}. Identify and fix.", "ESCALATE", 3),
            (
                "The {module} module's background job is stuck. Diagnose and restart safely.",
                "RECOVER",
                2,
            ),
            (
                "After the config change, {service} can't authenticate. Restore access.",
                "RECOVER",
                2,
            ),
            ("The {platform} node is unhealthy. Cordone, drain, and replace it.", "RECOVER", 2),
            ("A bad deploy introduced a memory leak in {env}. Roll back and patch.", "ESCALATE", 3),
            (
                "The {function} function's timeout is too short under load. Adjust and verify.",
                "RECOVER",
                2,
            ),
            (
                "The {service} service's queue is backed up. Drain and restore throughput.",
                "RECOVER",
                2,
            ),
            ("A {severity} security patch needs emergency deployment to {env}.", "ESCALATE", 3),
            ("The {module} module's connection pool is exhausted. Diagnose and fix.", "RECOVER", 2),
            (
                "After the rollback, {endpoint} still shows old behavior. Clear caches and verify.",
                "RECOVER",
                2,
            ),
        ],
    },
}


# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------

# Maximum fraction of total output that any single template may produce.
MAX_TEMPLATE_FRACTION = 0.05
# Maximum regeneration attempts before giving up (safety valve).
MAX_ATTEMPTS_MULTIPLIER = 20


def _build_template_index() -> list[tuple[str, str, str, int, float]]:
    """Flatten categories into a weighted list of (category, template, mode, difficulty, weight)."""
    index: list[tuple[str, str, str, int, float]] = []
    for cat, config in CATEGORIES.items():
        n_templates = len(config["templates"])
        per_template_weight = config["weight"] / n_templates
        for template, mode, difficulty in config["templates"]:
            index.append((cat, template, mode, difficulty, per_template_weight))
    return index


def _fill_template(template: str, expected_mode: str, difficulty: int) -> dict:
    """Fill a template with random values and return a prompt dict.

    Args:
        template: Template string with {placeholder} tokens.
        expected_mode: The control token mode (FAST, THINK, VERIFY, RECOVER, ESCALATE).
        difficulty: Difficulty level 1-3 for curriculum learning.

    Returns:
        Dict with prompt, expected_mode, id, category (added by caller), and difficulty.
    """
    filled = template
    placeholders = re.findall(r"\{(\w+)\}", template)
    for ph in placeholders:
        values = FILL_VALUES.get(ph, ["value"])
        filled = filled.replace(f"{{{ph}}}", random.choice(values), 1)
    return {
        "prompt": filled,
        "expected_mode": expected_mode,
        "id": hashlib.md5(filled.encode()).hexdigest()[:12],
        "difficulty": difficulty,
    }


def generate_prompts(n: int = 5000, seed: int | None = None) -> list[dict]:
    """Generate n diverse, unique task prompts.

    Guarantees:
    - Zero duplicate prompt texts (hash-based set enforcement).
    - No single template produces more than MAX_TEMPLATE_FRACTION of output.
    - Each prompt has a difficulty field (1-3).

    Args:
        n: Number of unique prompts to generate.
        seed: Optional random seed for reproducibility.

    Returns:
        List of prompt dicts with keys: prompt, expected_mode, id, category, difficulty.
    """
    if seed is not None:
        random.seed(seed)

    template_index = _build_template_index()
    weights = [t[4] for t in template_index]
    n_templates = len(template_index)

    # Per-template usage cap to enforce diversity.
    max_per_template = max(1, int(n * MAX_TEMPLATE_FRACTION))
    template_usage: Counter = Counter()

    seen_hashes: set[str] = set()
    seen_texts: set[str] = set()
    prompts: list[dict] = []

    max_attempts = n * MAX_ATTEMPTS_MULTIPLIER
    attempts = 0

    while len(prompts) < n and attempts < max_attempts:
        attempts += 1

        # Weighted selection, but skip templates that hit the diversity cap.
        available_indices = [i for i in range(n_templates) if template_usage[i] < max_per_template]
        if not available_indices:
            # All templates hit the cap — reset caps to continue filling.
            template_usage.clear()
            available_indices = list(range(n_templates))

        avail_weights = [weights[i] for i in available_indices]
        chosen_pos = random.choices(range(len(available_indices)), weights=avail_weights)[0]
        idx = available_indices[chosen_pos]

        cat, template, mode, difficulty, _ = template_index[idx]
        prompt_data = _fill_template(template, mode, difficulty)
        prompt_text = prompt_data["prompt"]

        # Uniqueness check: skip duplicates.
        if prompt_text in seen_texts:
            continue

        seen_texts.add(prompt_text)
        seen_hashes.add(prompt_data["id"])
        prompt_data["category"] = cat
        prompts.append(prompt_data)
        template_usage[idx] += 1

    if len(prompts) < n:
        print(
            f"WARNING: Only generated {len(prompts)} unique prompts out of {n} requested "
            f"after {attempts} attempts. Consider expanding templates or variable pools.",
            file=sys.stderr,
        )

    return prompts


def generate_prompts_streaming(
    n: int = 5000, seed: int | None = None
) -> Generator[dict, None, None]:
    """Generate prompts one at a time (for memory efficiency).

    Same guarantees as generate_prompts: uniqueness, diversity caps, difficulty.

    Args:
        n: Number of unique prompts to generate.
        seed: Optional random seed for reproducibility.

    Yields:
        Prompt dicts with keys: prompt, expected_mode, id, category, difficulty.
    """
    if seed is not None:
        random.seed(seed)

    template_index = _build_template_index()
    weights = [t[4] for t in template_index]
    n_templates = len(template_index)

    max_per_template = max(1, int(n * MAX_TEMPLATE_FRACTION))
    template_usage: Counter = Counter()

    seen_texts: set[str] = set()
    count = 0
    max_attempts = n * MAX_ATTEMPTS_MULTIPLIER
    attempts = 0

    while count < n and attempts < max_attempts:
        attempts += 1

        available_indices = [i for i in range(n_templates) if template_usage[i] < max_per_template]
        if not available_indices:
            template_usage.clear()
            available_indices = list(range(n_templates))

        avail_weights = [weights[i] for i in available_indices]
        chosen_pos = random.choices(range(len(available_indices)), weights=avail_weights)[0]
        idx = available_indices[chosen_pos]

        cat, template, mode, difficulty, _ = template_index[idx]
        prompt_data = _fill_template(template, mode, difficulty)
        prompt_text = prompt_data["prompt"]

        if prompt_text in seen_texts:
            continue

        seen_texts.add(prompt_text)
        prompt_data["category"] = cat
        template_usage[idx] += 1
        count += 1
        yield prompt_data


def _print_stats(prompts: list[dict]) -> None:
    """Print generation statistics: uniqueness, category distribution, template diversity."""
    texts = [p["prompt"] for p in prompts]
    unique_count = len(set(texts))
    dup_count = len(prompts) - unique_count

    print(f"Generated {len(prompts)} prompts ({unique_count} unique, {dup_count} duplicates)")

    cats = Counter(p["category"] for p in prompts)
    print("Category distribution:")
    for cat, count in cats.most_common():
        pct = count / len(prompts) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")

    modes = Counter(p["expected_mode"] for p in prompts)
    print("Mode distribution:")
    for mode, count in modes.most_common():
        pct = count / len(prompts) * 100
        print(f"  {mode}: {count} ({pct:.1f}%)")

    diffs = Counter(p["difficulty"] for p in prompts)
    print("Difficulty distribution:")
    for diff in sorted(diffs):
        count = diffs[diff]
        pct = count / len(prompts) * 100
        print(f"  Level {diff}: {count} ({pct:.1f}%)")


def main() -> None:
    """CLI entry point for the prompt generator."""
    parser = argparse.ArgumentParser(
        description="Generate diverse task prompts for v4 post-training pipeline."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5000,
        help="Number of unique prompts to generate (default: 5000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: prompts/prompts_{count}.jsonl)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming generator for memory efficiency",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        default=True,
        help="Print generation statistics (default: on)",
    )
    args = parser.parse_args()

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent / "prompts" / f"prompts_{args.count}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.streaming:
        with open(output_path, "w") as f:
            count = 0
            for p in generate_prompts_streaming(args.count, seed=args.seed):
                f.write(json.dumps(p) + "\n")
                count += 1
        prompts = []  # stats not available in streaming mode without storing
        print(f"Generated {count} prompts (streaming) -> {output_path}")
    else:
        prompts = generate_prompts(args.count, seed=args.seed)
        with open(output_path, "w") as f:
            for p in prompts:
                f.write(json.dumps(p) + "\n")
        print(f"Generated {len(prompts)} prompts -> {output_path}")
        if args.stats:
            _print_stats(prompts)


if __name__ == "__main__":
    main()
