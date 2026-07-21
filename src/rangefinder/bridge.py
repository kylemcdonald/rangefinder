"""Run a foreign-toolchain comparator (APAV, ms_deisotope, APyT) in a separate
Python environment and load its outputs back.

The comparator runs as ``python -m rangefinder.bridge_runner`` under the given
environment's interpreter, so that environment must have rangefinder (and the
comparator package) installed. Environments are keyed by method; the caller
supplies the project ``paths`` used to locate the environments, config, and log
directory.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from rangefinder.analysis.pipeline_utils import load_method_outputs
from rangefinder.utils.paths import ProjectPaths

DEFAULT_BRIDGE_ENVS = {
    "apav": ".venv312",
    "ms_deisotope": ".venv312",
    "apyt": ".venv",
}


def run_bridge_method(
    *,
    method_name: str,
    method_label: str,
    sample_path: Path,
    sample_slug: str,
    output_dir: Path,
    paths: ProjectPaths,
    bridge_envs: dict[str, str] | None = None,
) -> object:
    bridge_envs = bridge_envs or DEFAULT_BRIDGE_ENVS
    env_dir = bridge_envs.get(method_name)
    if env_dir is None:
        raise ValueError(f"Unknown bridge method: {method_name}")
    python_path = paths.root / env_dir / "bin" / "python"
    if not python_path.exists():
        raise FileNotFoundError(f"Bridge environment not found for {method_name}: {python_path}")
    log_path = paths.logs_dir / f"{sample_slug}_{method_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    libomp_dir = Path("/opt/homebrew/opt/libomp/lib")
    if libomp_dir.exists():
        existing = env.get("DYLD_LIBRARY_PATH", "")
        env["DYLD_LIBRARY_PATH"] = f"{libomp_dir}:{existing}" if existing else str(libomp_dir)
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            [
                str(python_path),
                "-m",
                "rangefinder.bridge_runner",
                "--method",
                method_name,
                "--sample",
                str(sample_path),
                "--output-dir",
                str(output_dir),
                "--config",
                str(paths.config_path),
            ],
            cwd=paths.root,
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
    return load_method_outputs(
        method_name=method_name,
        method_label=method_label,
        output_dir=output_dir,
    )
