#!/usr/bin/env python3
"""Submit a DPO (Direct Preference Optimization) job to Together AI.

Pipeline:
  1. Upload the DPO dataset JSONL to Together AI.
  2. Estimate the training price.
  3. Create a DPO fine-tuning job on Qwen/Qwen3.5-9B (optionally continuing
     from an SFT checkpoint via --checkpoint).
  4. Poll the job until it completes (or fails).
  5. Download the checkpoint / adapter.
  6. Persist job metadata to logs/dpo_job_metadata.json.

DPO data format (one JSON object per line):
    {"prompt": "...", "chosen": "...", "rejected": "..."}

The Together AI client is imported lazily so this module can be imported
without the ``together`` package installed (it is only needed at run time).

Prerequisites:
  - TOGETHER_API_KEY in the environment, or in a `.env` file
  - ``together`` Python package installed (pip install together)
  - DPO dataset at dpo/dpo_train.jsonl

Usage:
    cd /path/to/mac-ai-os
    uv run python -m training.v4_pipeline.train_dpo [--dry-run] \\
        [--checkpoint <sft-checkpoint-model-name>]

    # or directly:
    uv run python training/v4_pipeline/train_dpo.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# --- Paths ------------------------------------------------------------------

PIPELINE_DIR = Path(__file__).resolve().parent
DATASET_PATH = PIPELINE_DIR / "dpo" / "dpo_train.jsonl"
LOGS_DIR = PIPELINE_DIR / "logs"
METADATA_PATH = LOGS_DIR / "dpo_job_metadata.json"
CHECKPOINT_DIR = PIPELINE_DIR / "dpo" / "checkpoint"
SFT_METADATA_PATH = LOGS_DIR / "sft_job_metadata.json"

# Local .env (repo root or home) holds optional TOGETHER_API_KEY.
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

# --- Training parameters ----------------------------------------------------

BASE_MODEL = "Qwen/Qwen3.5-9B"
TRAINING_METHOD = "dpo"
DPO_BETA = 0.1
N_EPOCHS = 1
LEARNING_RATE = 5e-6
LORA_R = 8
SUFFIX = "adaptive_operator_v4_dpo"
POLL_INTERVAL = 60  # seconds
POLL_TIMEOUT = 7200  # 2 hours

# Professional-standard training hyperparameters (mirrors train_sft.py).
WARMUP_RATIO = 0.1  # 10% of steps for warmup
LR_SCHEDULER_TYPE = "cosine"  # cosine decay schedule
MAX_GRAD_NORM = 1.0  # gradient clipping
WEIGHT_DECAY = 0.01  # L2 regularization


# --- Environment / API key --------------------------------------------------


def load_api_key() -> str | None:
    """Resolve the Together AI API key from env or local-operator .env."""
    key = os.environ.get("TOGETHER_API_KEY")
    if key:
        return key
    if _ENV_FILE.exists():
        for line in _ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("TOGETHER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _get_client(api_key: str):
    """Create a Together AI client (lazy import)."""
    from together import Together

    return Together(api_key=api_key)


# --- Dataset helpers --------------------------------------------------------


def count_samples(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    n = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def validate_dpo_record(record: dict[str, Any]) -> bool:
    """Check that a DPO record has the required prompt/chosen/rejected fields."""
    return all(
        isinstance(record.get(f), str) and record[f].strip()
        for f in ("prompt", "chosen", "rejected")
    )


def _obj_to_dict(obj: Any) -> Any:
    """Best-effort conversion of a Together SDK response object to a dict."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read an attribute from either a pydantic-like object or a dict."""
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def resolve_checkpoint(explicit: str | None) -> str | None:
    """Resolve the SFT checkpoint to continue DPO from.

    Priority: explicit --checkpoint arg, then the SFT job metadata
    (logs/sft_job_metadata.json) model_output_name, then None.
    """
    if explicit:
        return explicit
    if SFT_METADATA_PATH.exists():
        try:
            sft_meta = json.loads(SFT_METADATA_PATH.read_text())
            job = sft_meta.get("job", {}) if isinstance(sft_meta.get("job"), dict) else {}
            name = job.get("model_output_name") or sft_meta.get("model_output_name")
            if name:
                return name
        except (json.JSONDecodeError, OSError):
            pass
    return None


# --- Pipeline steps ---------------------------------------------------------


def upload_dataset(client: Any, path: Path) -> str:
    """Upload a JSONL dataset to Together AI. Returns the file ID."""
    print("Uploading DPO dataset to Together AI...")
    result = client.files.upload(file=str(path))
    file_id = _get_attr(result, "id")
    if not file_id:
        raise RuntimeError(f"Upload succeeded but no file ID returned: {result}")
    print(f"  File ID: {file_id}")
    return file_id


def estimate_price(
    client: Any,
    file_id: str,
    *,
    model: str,
    n_epochs: int,
    lora_r: int,
    checkpoint: str | None,
) -> dict[str, Any]:
    """Estimate the price of a DPO fine-tuning job."""
    print("Estimating price...")
    kwargs: dict[str, Any] = dict(
        training_file=file_id,
        model=model,
        n_epochs=n_epochs,
        training_method={"method": "dpo"},
        training_type={"type": "Lora", "lora_r": lora_r},
    )
    if checkpoint:
        kwargs["from_checkpoint"] = checkpoint
    estimate = client.fine_tuning.estimate_price(**kwargs)
    data = _obj_to_dict(estimate)
    print(f"  Estimate: {json.dumps(data, indent=2, default=str)}")
    return data if isinstance(data, dict) else {"raw": data}


def create_job(
    client: Any,
    file_id: str,
    *,
    model: str,
    n_epochs: int,
    learning_rate: float,
    lora_r: int,
    dpo_beta: float,
    suffix: str,
    checkpoint: str | None,
    reference_model: str | None = None,
    warmup_ratio: float = WARMUP_RATIO,
    lr_scheduler_type: str = LR_SCHEDULER_TYPE,
    max_grad_norm: float = MAX_GRAD_NORM,
    weight_decay: float = WEIGHT_DECAY,
) -> str:
    """Create a DPO fine-tuning job. Returns the job ID.

    Together AI's DPO implementation uses the base model (or the SFT
    checkpoint when ``from_checkpoint`` is set) as the reference by default,
    which is the standard DPO behavior. An explicit ``reference_model`` can
    be supplied to override this if a different reference is needed.
    """
    print("Submitting DPO job...")
    kwargs: dict[str, Any] = dict(
        training_file=file_id,
        model=model,
        n_epochs=n_epochs,
        n_checkpoints=1,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        max_grad_norm=max_grad_norm,
        weight_decay=weight_decay,
        lora=True,
        lora_r=lora_r,
        suffix=suffix,
        training_method=TRAINING_METHOD,
        dpo_beta=dpo_beta,
    )
    if checkpoint:
        kwargs["from_checkpoint"] = checkpoint
        print(f"  Continuing from checkpoint: {checkpoint}")
    if reference_model:
        kwargs["reference_model"] = reference_model
        print(f"  Reference model: {reference_model}")
    else:
        print("  Reference model: (default — SFT checkpoint / base model)")
    job = client.fine_tuning.create(**kwargs)
    job_id = _get_attr(job, "id")
    if not job_id:
        raise RuntimeError(f"Job creation returned no ID: {job}")
    print(f"  Job ID: {job_id}")
    return job_id


def wait_for_completion(
    client: Any,
    job_id: str,
    *,
    poll_interval: int = POLL_INTERVAL,
    timeout: int = POLL_TIMEOUT,
) -> dict[str, Any]:
    """Poll a fine-tuning job until it reaches a terminal state."""
    print(f"Waiting for completion (polling every {poll_interval}s, "
          f"timeout {timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        job = client.fine_tuning.retrieve(id=job_id)
        status = _get_attr(job, "status", "unknown")
        elapsed = time.time() - start
        print(f"  Job {job_id}: {status} ({elapsed:.0f}s elapsed)")
        if status in ("completed", "failed", "cancelled"):
            return _obj_to_dict(job) or {"status": status}
        time.sleep(poll_interval)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def download_checkpoint(client: Any, job_id: str, output_dir: Path) -> Path:
    """Download the trained checkpoint/adapter for a completed job."""
    job = client.fine_tuning.retrieve(id=job_id)
    status = _get_attr(job, "status")
    if status != "completed":
        raise ValueError(f"Job {job_id} is not completed (status: {status})")

    model_name = (
        _get_attr(job, "model_output_name")
        or _get_attr(job, "model_output_name", "")
        or ""
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint model name: {model_name or '(none reported)'}")
    print(f"Checkpoint directory: {output_dir}")

    (output_dir / "model_name.txt").write_text(model_name)

    try:
        checkpoints = client.fine_tuning.list_checkpoints(id=job_id)
        ckpts = _obj_to_dict(checkpoints)
        (output_dir / "checkpoints.json").write_text(
            json.dumps(ckpts, indent=2, default=str)
        )
        print(f"  Checkpoint metadata saved to {output_dir / 'checkpoints.json'}")
    except Exception as e:
        print(f"  list_checkpoints not available (non-fatal): {e}")

    return output_dir


def save_metadata(metadata: dict[str, Any], path: Path) -> None:
    """Persist job metadata as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, default=str))
    print(f"Metadata saved to {path}")


# --- Main -------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Submit a DPO fine-tuning job to Together AI.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate price only; do not submit the job.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH,
                        help=f"Dataset JSONL (default: {DATASET_PATH})")
    parser.add_argument("--model", default=BASE_MODEL, help="Base model name.")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS,
                        help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Learning rate.")
    parser.add_argument("--lora-r", type=int, default=LORA_R,
                        help="LoRA rank.")
    parser.add_argument("--dpo-beta", type=float, default=DPO_BETA,
                        help="DPO beta (KL penalty strength).")
    parser.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO,
                        help="Fraction of training steps used for warmup.")
    parser.add_argument("--lr-scheduler-type", default=LR_SCHEDULER_TYPE,
                        choices=["cosine", "linear", "constant"],
                        help="Learning rate scheduler type.")
    parser.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM,
                        help="Maximum gradient norm for gradient clipping.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="L2 weight decay (regularization).")
    parser.add_argument("--suffix", default=SUFFIX,
                        help="Model name suffix.")
    parser.add_argument("--checkpoint", default=None,
                        help="SFT checkpoint model name to continue from "
                             "(auto-detected from sft_job_metadata.json if omitted).")
    parser.add_argument("--reference-model", default=None,
                        help="Explicit DPO reference model. If omitted, the SFT "
                             "checkpoint (or base model) is used as the reference, "
                             "which is standard DPO behavior.")
    parser.add_argument("--no-wait", action="store_true",
                        help="Exit immediately after job creation.")
    parser.add_argument("--poll-interval", type=int, default=POLL_INTERVAL,
                        help="Polling interval in seconds.")
    parser.add_argument("--timeout", type=int, default=POLL_TIMEOUT,
                        help="Polling timeout in seconds.")
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR,
                        help="Directory to save checkpoint metadata.")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    sample_count = count_samples(args.dataset)
    checkpoint = resolve_checkpoint(args.checkpoint)

    print(f"Dataset:       {args.dataset}")
    print(f"Samples:       {sample_count}")
    print(f"Model:         {args.model}")
    print(f"Method:        {TRAINING_METHOD}")
    print(f"DPO beta:      {args.dpo_beta}")
    print(f"Epochs:        {args.epochs}")
    print(f"LR:            {args.learning_rate}")
    print(f"LoRA r:        {args.lora_r}")
    print(f"Suffix:        {args.suffix}")
    print(f"Warmup:        {args.warmup_ratio}")
    print(f"LR scheduler:  {args.lr_scheduler_type}")
    print(f"Grad clip:     {args.max_grad_norm}")
    print(f"Weight decay:  {args.weight_decay}")
    print(f"Checkpoint:    {checkpoint or '(none — training from base model)'}")
    print(f"Reference:     {args.reference_model or '(default — SFT checkpoint / base model)'}")
    print()

    api_key = load_api_key()
    if not api_key:
        print("ERROR: TOGETHER_API_KEY not set. Expected in env or "
              f"{_ENV_FILE}", file=sys.stderr)
        return 1

    client = _get_client(api_key)

    # Step 1: upload dataset
    file_id = upload_dataset(client, args.dataset)
    print()

    # Step 2: estimate price
    try:
        estimate = estimate_price(
            client, file_id,
            model=args.model, n_epochs=args.epochs, lora_r=args.lora_r,
            checkpoint=checkpoint,
        )
    except Exception as e:
        print(f"  Price estimation failed (non-fatal): {e}")
        estimate = {}
    print()

    metadata: dict[str, Any] = {
        "stage": "dpo",
        "dataset": str(args.dataset),
        "sample_count": sample_count,
        "file_id": file_id,
        "estimate": estimate,
        "model": args.model,
        "training_method": TRAINING_METHOD,
        "dpo_beta": args.dpo_beta,
        "n_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "suffix": args.suffix,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "max_grad_norm": args.max_grad_norm,
        "weight_decay": args.weight_decay,
        "from_checkpoint": checkpoint,
        "reference_model": args.reference_model,
    }

    if args.dry_run:
        print("DRY RUN -- job not submitted. Remove --dry-run to submit.")
        save_metadata(metadata, METADATA_PATH)
        return 0

    # Confirm before spending real money.
    print("About to submit a DPO fine-tuning job.")
    print("This will cost real money (~$4 minimum, ~$0.48/M tokens).")
    response = input("Proceed? [y/N] ")
    if response.lower() != "y":
        print("Aborted.")
        save_metadata(metadata, METADATA_PATH)
        return 0

    # Step 3: create job
    job_id = create_job(
        client, file_id,
        model=args.model, n_epochs=args.epochs,
        learning_rate=args.learning_rate, lora_r=args.lora_r,
        dpo_beta=args.dpo_beta, suffix=args.suffix,
        checkpoint=checkpoint,
        reference_model=args.reference_model,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
    )
    metadata["job_id"] = job_id
    metadata["status"] = "created"
    save_metadata(metadata, METADATA_PATH)
    print()

    if args.no_wait:
        print("Job submitted. Skipping wait (--no-wait).")
        print(f"Monitor at https://api.together.ai  (job id: {job_id})")
        return 0

    # Step 4: wait for completion
    print("Waiting for completion (this may take 1-2 hours)...")
    try:
        result = wait_for_completion(
            client, job_id,
            poll_interval=args.poll_interval, timeout=args.timeout,
        )
    except TimeoutError as e:
        print(f"Job timed out: {e}")
        print(f"Check status at https://api.together.ai  (job id: {job_id})")
        metadata["status"] = "timeout"
        save_metadata(metadata, METADATA_PATH)
        return 1

    status = _get_attr(result, "status", "unknown")
    metadata["status"] = status
    metadata["job"] = result
    print(f"Job status: {status}")

    # Step 5: download checkpoint
    if status == "completed":
        print("Job completed successfully!")
        try:
            ckpt_dir = download_checkpoint(client, job_id, args.checkpoint_dir)
            metadata["checkpoint_dir"] = str(ckpt_dir)
            print("Done. Next step: convert to MLX or deploy an endpoint.")
        except Exception as e:
            print(f"Checkpoint download failed (non-fatal): {e}")
            metadata["checkpoint_error"] = str(e)
    else:
        print(f"Job did not complete successfully: {status}")

    # Step 6: save metadata
    save_metadata(metadata, METADATA_PATH)
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
