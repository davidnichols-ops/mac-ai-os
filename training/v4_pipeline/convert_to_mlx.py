"""Convert a Together AI fine-tuned Qwen3.5-9B model to MLX 4-bit for local deployment.

Pipeline (run on Apple Silicon):
  a) Download the LoRA adapter from Together AI (via the `together` SDK).
  b) Merge the LoRA adapter into the base Qwen3.5-9B model (via `peft` + `transformers`).
  c) Convert the merged HF model to MLX format (via `mlx_lm.convert`).
  d) Quantize to 4-bit (via `mlx_lm.convert --quantize`).
  e) Save to the v4 MLX model directory.
  f) Verify the converted model loads and generates.

NOTE on dependencies:
  `mlx_lm`, `mlx_vlm`, `peft`, and `transformers` are NOT installed in the base
  uv environment. `mlx_lm` / `mlx_vlm` are Apple-Silicon only and live in the
  `mlx` extra (`uv sync --extra mlx`). `peft` and `transformers` are needed for
  the LoRA merge step and must be installed separately
  (`uv pip install peft transformers accelerate`). `together` is needed to
  download the adapter from Together AI (`uv pip install together`).

  All of these are imported lazily inside functions so this module imports
  cleanly without them (e.g. for `uv run python -c "import ..."` smoke tests
  and for static analysis). The actual conversion must be run on Apple Silicon
  with the deps installed.

NOTE on model architecture:
  The v2 model used a VLM architecture (`Qwen3_5ForConditionalGeneration`) and
  was loaded with `mlx_vlm` (see `app/models/mlxvlm_provider.py`). If the v4
  fine-tune keeps the VLM architecture, use `mlx_vlm` conversion instead of
  `mlx_lm`. This script defaults to `mlx_lm` per the v4 spec (text-only
  inference), but exposes `--use-vlm` to switch to the `mlx_vlm` path when the
  merged model is a conditional-generation / VLM checkpoint.

Usage:
  uv run python -m training.v4_pipeline.convert_to_mlx --job-id <together-job-id>
  uv run python -m training.v4_pipeline.convert_to_mlx --job-id <id> --quantize 8bit
  uv run python -m training.v4_pipeline.convert_to_mlx --job-id <id> --quantize none
  uv run python -m training.v4_pipeline.convert_to_mlx --job-id <id> --use-vlm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-9B"
# The v2 MLX model lived at this path; v4 gets its own directory.
DEFAULT_OUTPUT_PATH = os.path.expanduser(
    "~/Projects/local-operator/models/v4/mlx/"
    "Qwen3.5-9B-Adaptive-Operator-v4-MLX-4bit"
)
DEFAULT_WORK_DIR = str(Path(__file__).resolve().parent / "logs" / "convert")
ENV_FILE = os.path.expanduser("~/Projects/local-operator/.env")
QUANTIZE_CHOICES = ("4bit", "8bit", "none")

# Quantization presets understood by mlx_lm.convert / mlx_vlm.convert.
# mlx_lm uses --quantize with "4bit" / "8bit" / "none" (or a q-bits int).
QUANT_PRESETS: dict[str, dict[str, Any]] = {
    "4bit": {"quantize": True, "q_bits": 4, "group_size": 64},
    "8bit": {"quantize": True, "q_bits": 8, "group_size": 64},
    "none": {"quantize": False, "q_bits": None, "group_size": None},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(step: str, msg: str, **kwargs: Any) -> None:
    """Print a structured progress line for a pipeline step."""
    extra = ""
    if kwargs:
        extra = " " + " ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"[convert_to_mlx] [{step}] {msg}{extra}", flush=True)


def _load_together_api_key() -> str:
    """Load the Together AI API key from the local-operator .env file."""
    # Prefer an explicit env var if already exported.
    key = os.environ.get("TOGETHER_API_KEY")
    if key:
        return key
    env_path = Path(ENV_FILE)
    if not env_path.exists():
        raise FileNotFoundError(
            f"TOGETHER_API_KEY not set and env file not found: {env_path}"
        )
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == "TOGETHER_API_KEY":
            v = v.strip().strip('"').strip("'")
            if v:
                return v
    raise RuntimeError(
        "TOGETHER_API_KEY not found in env file. "
        f"Checked {env_path}."
    )


def _require(modname: str, hint: str) -> Any:
    """Import a module lazily, raising a helpful error if missing."""
    try:
        return __import__(modname)
    except ImportError as e:  # pragma: no cover - env dependent
        raise RuntimeError(
            f"{modname} is not installed. {hint}"
        ) from e


# ---------------------------------------------------------------------------
# Step a) Download the LoRA adapter from Together AI
# ---------------------------------------------------------------------------

def download_lora_adapter(
    job_id: str,
    dest_dir: Path,
    api_key: str | None = None,
) -> Path:
    """Download the LoRA adapter artifacts for a Together AI fine-tuning job.

    Returns the path to the directory containing adapter_model.safetensors and
    adapter_config.json.

    Together AI exposes fine-tuning job metadata via `client.fine_tuning.retrieve`
    and adapter files via `client.files.download_files`. The exact file ids are
    listed on the job's `model` / `output` artifact set. We download every
    adapter-related file (adapter_config.json, adapter_model.safetensors, and
    any tokenizer files) into `dest_dir`.
    """
    _log("download", "Downloading LoRA adapter from Together AI", job_id=job_id)
    together = _require(
        "together",
        "Install with: uv pip install together",
    )

    api_key = api_key or _load_together_api_key()
    client = together.Together(api_key=api_key)

    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1) Retrieve job metadata to find the model/adapter artifact.
    t0 = time.monotonic()
    try:
        job = client.fine_tuning.retrieve(id=job_id)
    except Exception as e:
        raise RuntimeError(
            f"Failed to retrieve fine-tuning job {job_id}: {e}"
        ) from e

    job_dict = _to_dict(job)
    _log(
        "download",
        "Retrieved job metadata",
        status=job_dict.get("status"),
        model_name=job_dict.get("model_name") or job_dict.get("model"),
        elapsed_s=round(time.monotonic() - t0, 1),
    )

    # 2) Resolve the adapter artifact id(s).
    # Together returns the trained model name and/or a set of downloadable
    # files. We look for the LoRA adapter files explicitly.
    file_ids: list[str] = []
    # Some job objects expose `model_output_name` or `output_path`.
    model_output = (
        job_dict.get("model_output_name")
        or job_dict.get("output_path")
        or job_dict.get("model_name")
        or job_dict.get("model")
    )

    # Try the files listing API to find adapter files.
    try:
        files = client.files.list()
        for f in files:
            fname = getattr(f, "filename", None) or getattr(f, "name", None) or ""
            fid = getattr(f, "id", None)
            if not fid:
                continue
            # Match adapter artifacts by name pattern.
            if any(
                pat in fname.lower()
                for pat in ("adapter", "lora", model_output or "____nomatch____")
            ):
                file_ids.append(fid)
    except Exception as e:
        _log("download", "files.list() unavailable, falling back to job output", warn=str(e))

    # If we couldn't enumerate files, try downloading via the model output name
    # directly (Together packages LoRA adapters under the output model name).
    if not file_ids and model_output:
        _log(
            "download",
            "No explicit file ids found; attempting download by model output name",
            model_output=model_output,
        )
        adapter_path = dest_dir / "adapter"
        adapter_path.mkdir(exist_ok=True)
        try:
            client.files.download_files(
                id=model_output,
                output=str(adapter_path),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download adapter by model output name "
                f"{model_output!r}: {e}. Verify the job id and that the adapter "
                f"is available for download."
            ) from e
        _log("download", "Adapter downloaded by model name", path=str(adapter_path))
        return adapter_path

    if not file_ids:
        raise RuntimeError(
            f"No downloadable adapter files found for job {job_id}. "
            f"Job metadata: {json.dumps(job_dict, default=str)[:1000]}"
        )

    # 3) Download each adapter file.
    adapter_path = dest_dir / "adapter"
    adapter_path.mkdir(exist_ok=True)
    for fid in file_ids:
        _log("download", "Downloading file", file_id=fid)
        try:
            client.files.download_files(id=fid, output=str(adapter_path))
        except Exception as e:
            _log("download", "File download failed (continuing)", file_id=fid, warn=str(e))

    # Sanity check: we need at least an adapter config.
    has_config = any(p.name == "adapter_config.json" for p in adapter_path.iterdir())
    has_weights = any(
        p.name.startswith("adapter_model") for p in adapter_path.iterdir()
    )
    if not (has_config or has_weights):
        raise RuntimeError(
            f"Downloaded adapter directory has no adapter artifacts: "
            f"{list(p.name for p in adapter_path.iterdir())}"
        )

    _log(
        "download",
        "Adapter downloaded",
        path=str(adapter_path),
        files=[p.name for p in adapter_path.iterdir()],
        elapsed_s=round(time.monotonic() - t0, 1),
    )
    return adapter_path


# ---------------------------------------------------------------------------
# Step b) Merge the LoRA adapter with the base model
# ---------------------------------------------------------------------------

def merge_lora(
    base_model: str,
    adapter_path: Path,
    merged_dir: Path,
) -> Path:
    """Merge a LoRA adapter into the base model and save a full HF checkpoint.

    Uses `peft.PeftModel` + `transformers.AutoModelForCausalLM`. The merged
    model is saved to `merged_dir` along with the tokenizer.
    """
    _log("merge", "Merging LoRA adapter into base model", base=base_model, adapter=str(adapter_path))
    transformers = _require(
        "transformers",
        "Install with: uv pip install transformers accelerate",
    )
    peft = _require("peft", "Install with: uv pip install peft")

    merged_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    # Load the base model in fp16 on CPU (we only need it for merging).
    _log("merge", "Loading base model (fp16, CPU)...")
    AutoModel = transformers.AutoModelForCausalLM
    AutoTokenizer = transformers.AutoTokenizer

    base_model_obj = AutoModel.from_pretrained(
        base_model,
        torch_dtype="float16",
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Load the LoRA adapter on top of the base model.
    _log("merge", "Loading LoRA adapter...")
    model = peft.PeftModel.from_pretrained(base_model_obj, str(adapter_path))

    # Merge adapter weights into the base weights and unload the adapter wrapper.
    _log("merge", "Merging adapter weights...")
    merged = model.merge_and_unload()

    # Save the merged model + tokenizer.
    _log("merge", "Saving merged model...", path=str(merged_dir))
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    _log(
        "merge",
        "Merge complete",
        path=str(merged_dir),
        elapsed_s=round(time.monotonic() - t0, 1),
    )
    return merged_dir


# ---------------------------------------------------------------------------
# Step c+d) Convert to MLX + quantize
# ---------------------------------------------------------------------------

def convert_to_mlx(
    hf_model_dir: Path,
    mlx_dir: Path,
    quantize: str,
    use_vlm: bool = False,
) -> Path:
    """Convert a HF model directory to MLX format with optional quantization.

    Uses `mlx_lm.convert` (or `mlx_vlm.convert` when `use_vlm=True`). The
    converter writes the MLX weights + config + tokenizer to `mlx_dir`.
    """
    preset = QUANT_PRESETS[quantize]
    _log(
        "convert",
        "Converting to MLX",
        src=str(hf_model_dir),
        dst=str(mlx_dir),
        quantize=quantize,
        use_vlm=use_vlm,
    )

    if use_vlm:
        mlx_vlm = _require(
            "mlx_vlm",
            "Install with: uv pip install mlx-vlm (Apple Silicon only)",
        )
        convert_mod = mlx_vlm.convert
    else:
        mlx_lm = _require(
            "mlx_lm",
            "Install with: uv sync --extra mlx (Apple Silicon only)",
        )
        convert_mod = mlx_lm.convert

    mlx_dir.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    # mlx_lm.convert exposes a `convert` function with signature:
    #   convert(hf_path, mlx_path, quantize=False, q_bits=4, q_group_size=64)
    # Some versions expose `from_pretrained`/`save` separately; we try the
    # high-level `convert` first and fall back to the lower-level API.
    quant = preset["quantize"]
    q_bits = preset["q_bits"]
    group_size = preset["group_size"]

    try:
        # Newer mlx_lm.convert signature.
        convert_mod(
            str(hf_model_dir),
            str(mlx_dir),
            quantize=quant,
            q_bits=q_bits if q_bits else 4,
            q_group_size=group_size if group_size else 64,
        )
    except TypeError:
        # Older / alternative signature: convert(hf_path, mlx_path, quantize=bool)
        convert_mod(str(hf_model_dir), str(mlx_dir), quantize=quant)
    except Exception as e:
        # Fall back to the CLI-style invocation via subprocess if the Python
        # API differs across versions.
        _log("convert", "Python API failed, falling back to CLI", warn=str(e))
        _convert_via_cli(hf_model_dir, mlx_dir, quantize, use_vlm)

    _log(
        "convert",
        "Conversion complete",
        path=str(mlx_dir),
        elapsed_s=round(time.monotonic() - t0, 1),
    )
    return mlx_dir


def _convert_via_cli(
    hf_model_dir: Path, mlx_dir: Path, quantize: str, use_vlm: bool
) -> None:
    """Fallback: invoke the mlx-lm / mlx-vlm convert CLI via subprocess."""
    import subprocess

    tool = "mlx_vlm" if use_vlm else "mlx_lm"
    cmd = [sys.executable, "-m", tool, "convert"]
    if use_vlm:
        cmd += ["--hf-path", str(hf_model_dir), "--mlx-path", str(mlx_dir)]
    else:
        cmd += ["--hf-path", str(hf_model_dir), "--mlx-path", str(mlx_dir)]
    if quantize == "4bit":
        cmd += ["--quantize", "--q-bits", "4"]
    elif quantize == "8bit":
        cmd += ["--quantize", "--q-bits", "8"]
    _log("convert", "Running CLI", cmd=" ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Step f) Verify the converted model loads
# ---------------------------------------------------------------------------

def verify_converted_model(mlx_dir: Path, use_vlm: bool = False) -> bool:
    """Verify the converted MLX model loads and can generate a token."""
    _log("verify", "Loading converted model for verification", path=str(mlx_dir))
    if use_vlm:
        mlx_vlm = _require("mlx_vlm", "Install mlx-vlm (Apple Silicon only)")
        model, processor = mlx_vlm.load(str(mlx_dir))
        out = mlx_vlm.generate(
            model, processor, prompt="Hello, are you an adaptive operator?", max_tokens=16
        )
        text = out.text if hasattr(out, "text") else str(out)
    else:
        mlx_lm = _require("mlx_lm", "Install mlx-lm (Apple Silicon only)")
        model, tokenizer = mlx_lm.load(str(mlx_dir))
        out = mlx_lm.generate(
            model, tokenizer, prompt="Hello, are you an adaptive operator?", max_tokens=16
        )
        text = out.text if hasattr(out, "text") else str(out)

    _log("verify", "Model loaded and generated", sample=text[:80].replace("\n", " "))
    return True


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_dict(obj: Any) -> dict[str, Any]:
    """Best-effort conversion of a Together SDK response object to a dict."""
    if isinstance(obj, dict):
        return obj
    # Pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {"raw": str(obj)}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    job_id: str,
    output_path: str = DEFAULT_OUTPUT_PATH,
    quantize: str = "4bit",
    base_model: str = DEFAULT_BASE_MODEL,
    work_dir: str = DEFAULT_WORK_DIR,
    use_vlm: bool = False,
    skip_download: bool = False,
    local_adapter: str | None = None,
) -> Path:
    """Run the full download → merge → convert → quantize → verify pipeline.

    Args:
        job_id: Together AI fine-tuning job id (used to download the adapter).
        output_path: Final MLX model output directory.
        quantize: One of "4bit", "8bit", "none".
        base_model: HuggingFace id of the base model to merge the adapter into.
        work_dir: Working directory for intermediate artifacts (adapter, merged).
        use_vlm: If True, use mlx_vlm instead of mlx_lm (for VLM architectures).
        skip_download: If True, skip the Together download step (use --local-adapter).
        local_adapter: Path to a local adapter directory (skips download).

    Returns:
        The final MLX model directory path.
    """
    if quantize not in QUANTIZE_CHOICES:
        raise ValueError(f"quantize must be one of {QUANTIZE_CHOICES}, got {quantize!r}")

    out_dir = Path(output_path).expanduser()
    work = Path(work_dir).expanduser()
    adapter_dir = work / "adapter"
    merged_dir = work / "merged"

    print("=" * 70, flush=True)
    print("  MLX Conversion Pipeline — Qwen3.5-9B Adaptive Operator v4", flush=True)
    print(f"  job_id      : {job_id}", flush=True)
    print(f"  base_model  : {base_model}", flush=True)
    print(f"  quantize    : {quantize}", flush=True)
    print(f"  use_vlm     : {use_vlm}", flush=True)
    print(f"  output      : {out_dir}", flush=True)
    print(f"  work_dir    : {work}", flush=True)
    print("=" * 70, flush=True)

    # Step a) Download the LoRA adapter.
    if skip_download or local_adapter:
        adapter_path = Path(local_adapter or adapter_dir).expanduser()
        if not adapter_path.exists():
            raise FileNotFoundError(f"Local adapter not found: {adapter_path}")
        _log("download", "Using local adapter (skipping Together download)", path=str(adapter_path))
    else:
        adapter_path = download_lora_adapter(job_id, work)

    # Step b) Merge LoRA into the base model.
    merge_lora(base_model, adapter_path, merged_dir)

    # Step c+d) Convert to MLX + quantize.
    convert_to_mlx(merged_dir, out_dir, quantize, use_vlm=use_vlm)

    # Step e) is implicit — convert_to_mlx writes directly to out_dir.

    # Step f) Verify the converted model loads.
    verify_converted_model(out_dir, use_vlm=use_vlm)

    # Write a small conversion manifest for reproducibility.
    manifest = {
        "job_id": job_id,
        "base_model": base_model,
        "quantize": quantize,
        "use_vlm": use_vlm,
        "output_path": str(out_dir),
        "created_by": "training.v4_pipeline.convert_to_mlx",
    }
    (out_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print("=" * 70, flush=True)
    print(f"  DONE — MLX model saved to: {out_dir}", flush=True)
    print("=" * 70, flush=True)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a Together AI fine-tuned Qwen3.5-9B model to MLX 4-bit.",
    )
    parser.add_argument(
        "--job-id",
        required=False,
        default=None,
        help="Together AI fine-tuning job id (used to download the LoRA adapter).",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output MLX model directory (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--quantize",
        choices=QUANTIZE_CHOICES,
        default="4bit",
        help="Quantization level (default: 4bit).",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="HuggingFace base model id to merge the adapter into.",
    )
    parser.add_argument(
        "--work-dir",
        default=DEFAULT_WORK_DIR,
        help="Working directory for intermediate artifacts.",
    )
    parser.add_argument(
        "--use-vlm",
        action="store_true",
        help="Use mlx_vlm instead of mlx_lm (for VLM architectures like v2).",
    )
    parser.add_argument(
        "--local-adapter",
        default=None,
        help="Path to a local LoRA adapter directory (skips Together download).",
    )
    args = parser.parse_args(argv)

    if not args.job_id and not args.local_adapter:
        parser.error("either --job-id or --local-adapter is required")

    try:
        run_pipeline(
            job_id=args.job_id or "local",
            output_path=args.output_path,
            quantize=args.quantize,
            base_model=args.base_model,
            work_dir=args.work_dir,
            use_vlm=args.use_vlm,
            skip_download=bool(args.local_adapter),
            local_adapter=args.local_adapter,
        )
    except KeyboardInterrupt:
        _log("main", "Interrupted by user")
        return 130
    except Exception as e:
        _log("main", "Pipeline failed", error=str(e))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
