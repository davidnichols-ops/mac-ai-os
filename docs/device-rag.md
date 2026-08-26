# Device RAG

MAOS now provides one local, privacy-aware retrieval service over the user's
knowledge surface. It stores local MiniLM embeddings and SQLite FTS5 exact-text
chunks in `data/device_rag.db`, then combines both signals with reciprocal-rank
fusion. The server does not depend on the legacy Chroma collection; that native
collection segfaulted during live verification and was left untouched for a
separate repair rather than being trusted as the new service's foundation.

## Safety boundary

The default root is the current user's home folder, but the scanner prunes
macOS `Library`, application bundles, hidden folders, credentials, databases,
caches, dependencies, builds, model/vector data, binaries, and oversized
files. This is intentional: device-wide retrieval means broad useful coverage,
not copying every byte or secret into an embedding database.

Inspect the boundary before the first run:

```bash
uv run app device-rag policy
uv run app device-rag sync --dry-run --json
```

Start with a bounded empirical slice, then run the full incremental sync:

```bash
uv run app device-rag sync --root ~/mac-ai-os --max-files 25
uv run app device-rag search "DeviceRAGService"
uv run app device-rag sync
uv run app device-rag status
```

Subsequent syncs hash files and only re-embed changes. A complete scan also
removes deleted files from both indexes. A scan stopped with `--max-files`
never performs deletion reconciliation because it cannot prove unseen files
were deleted.

## Server

Start the existing MAOS Context API:

```bash
uv run app serve --port 7780
```

It binds only to loopback and requires the capability token in `~/.maos/token`
for every sensitive endpoint:

```bash
curl -H "Authorization: Bearer $(<~/.maos/token)" \
  "http://127.0.0.1:7777/rag/search?q=trajectory&top_k=8"

curl -H "Authorization: Bearer $(<~/.maos/token)" \
  -H "Content-Type: application/json" \
  -d '{"question":"What am I currently building?","top_k":8}' \
  http://127.0.0.1:7777/rag/query
```

Endpoints:

- `GET /rag/status` — coverage, freshness, roots, and policy.
- `GET /rag/search?q=...&top_k=8` — hybrid evidence retrieval.
- `POST /rag/query` — grounded synthesis with numbered source citations.
- `GET /memory/search?q=...` — compatibility alias for hybrid retrieval.

`/health` and the API index remain unauthenticated for local health checks.
Cross-origin access is restricted to localhost and explicitly allowed hosted origins.

Nevernote's production origin is allowed by default. Add other exact hosted
origins as a comma-separated local environment setting before starting MAOS:

```bash
MAOS_ALLOWED_ORIGINS="https://preview.example.vercel.app" uv run app serve
```

The server also supports the browser private-network preflight required when a
hosted HTTPS interface calls a loopback service. It still never binds outside
loopback and every RAG request still requires the capability token.
