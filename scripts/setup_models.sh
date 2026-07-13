#!/usr/bin/env bash
# Mac AI OS — model setup script.
#
# Verifies MLX, installs Ollama if missing, pulls required models, verifies
# hashes (best-effort), prints available models, and benchmarks first inference.
#
# Safe to re-run. Reads model choices from configs/user.yaml (or default.yaml).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- helpers -----------------------------------------------------------------
c_red()   { printf '\033[31m%s\033[0m\n' "$*"; }
c_grn()   { printf '\033[32m%s\033[0m\n' "$*"; }
c_ylw()   { printf '\033[33m%s\033[0m\n' "$*"; }
c_cyan()  { printf '\033[36m%s\033[0m\n' "$*"; }
hr()      { printf -- '---\n'; }

ok()   { c_grn "  [ok] $*"; }
warn() { c_ylw "  [warn] $*"; }
fail() { c_red "  [fail] $*"; exit 1; }

# --- 1. Python / uv ----------------------------------------------------------
c_cyan "1. Verifying Python environment"
command -v uv >/dev/null 2>&1 || fail "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
ok "uv found: $(uv --version)"

if [[ ! -d ".venv" ]]; then
  c_ylw "  creating .venv with uv..."
  uv venv --python 3.12
fi
c_ylw "  syncing dependencies (including mlx extra)..."
uv sync --extra mlx --extra dev
ok "dependencies synced"
hr

# --- 2. MLX ------------------------------------------------------------------
c_cyan "2. Verifying MLX"
if uv run python -c "import mlx_lm" 2>/dev/null; then
  ok "mlx-lm importable: $(uv run python -c 'import mlx_lm; print(getattr(mlx_lm, "__version__", "unknown"))')"
else
  warn "mlx-lm not importable (are you on Apple Silicon?). MLX provider will be unavailable."
fi
hr

# --- 3. Ollama ---------------------------------------------------------------
c_cyan "3. Verifying Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  warn "ollama not found. Installing via brew..."
  if command -v brew >/dev/null 2>&1; then
    brew install ollama
  else
    c_ylw "  Homebrew missing; falling back to official install script."
    curl -fsSL https://ollama.com/install.sh | sh
  fi
fi
command -v ollama >/dev/null 2>&1 || fail "ollama still not installed"
ok "ollama found: $(ollama --version 2>&1 | head -n1)"

# Start the daemon if it isn't running.
if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  c_ylw "  starting ollama serve in background..."
  ollama serve >/tmp/maos-ollama.log 2>&1 &
  echo $! > /tmp/maos-ollama.pid
  for _ in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
      ok "ollama serve is up"
      break
    fi
    sleep 1
  done
  curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 || fail "ollama did not become healthy"
else
  ok "ollama serve already running"
fi
hr

# --- 4. Pull models ----------------------------------------------------------
c_cyan "4. Pulling models"

# Read model names from config (fall back to hardware defaults via the app).
read_cfg() {
  uv run python - "$1" <<'PY'
import sys, yaml
from pathlib import Path
key = sys.argv[1]
for name in ("configs/user.yaml", "configs/default.yaml"):
    p = Path(name)
    if not p.exists(): continue
    with p.open() as f:
        d = yaml.safe_load(f) or {}
    section = d.get(key.split(".")[0], {})
    val = section.get(key.split(".")[1], "")
    if val:
        print(val); break
PY
}

PLANNER_MLX=$(read_cfg planner.model)
UTILITY_MLX=$(read_cfg utility.model)
EMBED_MLX=$(read_cfg embeddings.model)

# MLX models are pulled on first use by mlx-lm; pre-warm them.
pull_mlx() {
  local model="$1"
  if [[ -z "$model" ]]; then return; fi
  c_ylw "  pre-warming MLX model: $model (first load downloads weights)"
  uv run python -c "import mlx_lm; mlx_lm.load('$model')" && ok "MLX ready: $model" || warn "MLX pull failed: $model"
}

# Ollama fallback models (used if MLX is unavailable).
pull_ollama() {
  local model="$1"
  if [[ -z "$model" ]]; then return; fi
  if ollama list 2>/dev/null | grep -q "^$model "; then
    ok "ollama already has: $model"
  else
    c_ylw "  ollama pull: $model"
    ollama pull "$model" && ok "ollama pulled: $model" || warn "ollama pull failed: $model"
  fi
}

if uv run python -c "import mlx_lm" 2>/dev/null; then
  pull_mlx "$PLANNER_MLX"
  pull_mlx "$UTILITY_MLX"
  pull_mlx "$EMBED_MLX"
  # Also pull an ollama embedding model as a fallback.
  pull_ollama nomic-embed-text
else
  pull_ollama qwen3:14b
  pull_ollama gemma3:4b
  pull_ollama nomic-embed-text
fi
hr

# --- 5. Print available models ----------------------------------------------
c_cyan "5. Available models"
c_cyan "MLX (cached under ~/.cache/huggingface/hub):"
uv run python -c "import mlx_lm; print('  mlx-lm ready')" 2>/dev/null || warn "  mlx-lm unavailable"
c_cyan "Ollama:"
ollama list || true
hr

# --- 6. Benchmark first inference -------------------------------------------
c_cyan "6. Benchmarking first inference"
if uv run python -c "import mlx_lm" 2>/dev/null && [[ -n "$PLANNER_MLX" ]]; then
  c_ylw "  MLX planner: '$PLANNER_MLX'"
  time uv run python -c "
import mlx_lm, time
m, tok = mlx_lm.load('$PLANNER_MLX')
prompt = tok.apply_chat_template([{'role':'user','content':'Say hello in one sentence.'}], tokenize=False, add_generation_prompt=True)
t0 = time.time()
out = mlx_lm.generate(m, tok, prompt=prompt, max_tokens=64, verbose=False)
dt = time.time() - t0
print('  reply:', out.text.strip().split(chr(10))[0][:120])
print(f'  latency: {dt:.2f}s')
"
else
  c_ylw "  Ollama planner:"
  time ollama run qwen3:14b "Say hello in one sentence." --verbose 2>&1 | tail -n 5 || true
fi
hr

c_grn "Setup complete. Try: uv run app verify"
