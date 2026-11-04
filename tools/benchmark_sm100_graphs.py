"""Compare compiled PyTorch and MegaBake CUDA graphs on local model checkpoints.

The default settings reproduce the capacity-192, batch-one BF16 audit. Each
model runs in a fresh process, including five greedy-prompt correctness checks.
The CUDA graph paths use different attention and cache implementations.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    "smollm-135m",
    "qwen2-0.5b",
    "qwen2.5-0.5b-instruct",
    "deepseek-draft",
    "gemma-3-270m",
    "smollm-1.7b",
    "gemma-2b",
    "qwen3.5-0.8b",
)


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models-root", type=Path, required=True, help="directory with checkpoints"
    )
    parser.add_argument("--out", type=Path, required=True, help="new or empty results directory")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--threads", type=int, choices=(128, 256), default=128)
    parser.add_argument("--stages", type=int, choices=(1, 2), default=1)
    parser.add_argument("--capacity", type=int, default=192)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=24)
    args = parser.parse_args()
    if args.threads == 128 and args.stages != 1:
        parser.error("the 128-thread variant requires one accumulator stage")
    if args.warmups < 0 or args.samples < 1 or args.capacity < args.warmups + args.samples:
        parser.error("warmups must be non-negative and capacity must hold all timed samples")
    models = list(dict.fromkeys(args.models))
    models_root = args.models_root.expanduser().resolve()
    missing = [name for name in models if not (models_root / name).is_dir()]
    if missing:
        parser.error(f"missing checkpoint directories in {models_root}: {', '.join(missing)}")
    out = args.out.expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f"results directory must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    # Ignore inherited MegaBake tuning/architecture flags so both thread
    # options have the documented shape heuristic and no tuning file.
    env = {key: value for key, value in os.environ.items() if not key.startswith("MEGABAKE_")}
    settings = {
        "MEGABAKE_SM100_TCGEN": "1",
        "MEGABAKE_SM100_TCGEN_DECODE": "1",
        "MEGABAKE_SM100_TCGEN_TILE": "auto",
        "MEGABAKE_SM100_TCGEN_MAX_ROWS": "32",
        "MEGABAKE_SM100_TOKEN_TILE": "8",
        "MEGABAKE_SM100_SPLIT_K": "8",
        "MEGABAKE_SM100_EPILOGUE": "1",
        "MEGABAKE_SM100_PERSISTENT": "1",
        "MEGABAKE_SM100_PERSISTENT_WIDTH": str(args.threads),
        "MEGABAKE_SM100_PERSISTENT_STAGES": str(args.stages),
        "MEGABAKE_PREFILL_VALUE_BLOCK": "0",
    }
    env.update(settings)
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TORCHINDUCTOR_COMPILE_THREADS="1",
    )
    pythonpath = [str(ROOT / "src"), str(ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    metadata = {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "models_root": str(models_root),
        "models": models,
        "capacity": args.capacity,
        "warmups": args.warmups,
        "samples": args.samples,
        "settings": settings,
        "versions": {
            name: _version(name) for name in ("torch", "transformers", "nvidia-cutlass-dsl")
        },
        "results": {},
    }
    (out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    failures = 0
    for name in models:
        result = out / f"{name}.json"
        log = out / f"{name}.log"
        command = [
            sys.executable,
            str(ROOT / "tools" / "audit_sm100_models.py"),
            str(models_root / name),
            str(result),
            "--capacity",
            str(args.capacity),
            "--warmups",
            str(args.warmups),
            "--samples",
            str(args.samples),
        ]
        print(f"Running {name}; log: {log}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
        if completed.returncode != 0 or not result.is_file():
            failures += 1
            metadata["results"][name] = {"status": "failed", "exit_code": completed.returncode}
            print(f"{name} failed; see {log}", file=sys.stderr, flush=True)
        else:
            report = json.loads(result.read_text())
            torch_ms = report["timings"]["torch_graph"]["wall_median_ms"]
            megabake_ms = report["timings"]["megabake_graph"]["wall_median_ms"]
            prompts = report["prompts"]
            matches = {
                variant: sum(
                    prompt["variants"][variant]["greedy_first_difference_vs_native"] is None
                    for prompt in prompts
                )
                for variant in ("torch_graph", "megabake_graph")
            }
            row = {
                "model": name,
                "torch_graph_ms": torch_ms,
                "megabake_graph_ms": megabake_ms,
                "megabake_latency_change_percent": (megabake_ms / torch_ms - 1) * 100,
                "torch_prompts_matching_native": matches["torch_graph"],
                "megabake_prompts_matching_native": matches["megabake_graph"],
                "prompt_count": len(prompts),
            }
            rows.append(row)
            metadata["results"][name] = {"status": "ok", **row}
            print(f"{name}: torch {torch_ms:.3f} ms, MegaBake {megabake_ms:.3f} ms", flush=True)
        (out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if rows:
        with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Results: {out}; failures: {failures}", flush=True)
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
