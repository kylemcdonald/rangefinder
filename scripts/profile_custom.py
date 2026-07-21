#!/usr/bin/env python3
"""Profile run_custom_pipeline on a chosen control to find hot spots."""
from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from rangefinder import default_config_path  # noqa: E402

from rangefinder.analysis.custom_pipeline import run_custom_pipeline  # noqa: E402
from rangefinder.benchmark.truth import load_pos_arrays  # noqa: E402
from rangefinder.io.pos_loader import PosSampleData, PosSampleMetadata  # noqa: E402
from rangefinder.utils.config import load_config  # noqa: E402


def make_sample(path: Path):
    x, y, z, mz = load_pos_arrays(path)
    meta = PosSampleMetadata(
        path=path,
        sample_name=path.stem,
        sample_slug=path.stem,
        event_count=int(mz.size),
        file_size_bytes=int(mz.size) * 16,
        verified_big_endian_float32=True,
        verified_columns=("x_nm", "y_nm", "z_nm", "m_over_z_da"),
    )
    return PosSampleData(metadata=meta, x_nm=x, y_nm=y, z_nm=z, m_over_z_da=mz)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "controls" / "control_W_18K_breen_kuehbach_R18_53222.epos"
    out = ROOT / "tmp" / "profile_custom_out"
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(default_config_path())
    print(f"loading {path.name} ...", flush=True)
    t0 = time.perf_counter()
    sample = make_sample(path)
    print(f"loaded {sample.m_over_z_da.size:,} events in {time.perf_counter()-t0:.2f}s", flush=True)

    # warm run (also compiles anything cached) then profiled run
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    run_custom_pipeline(sample, out, config)
    prof.disable()
    dt = time.perf_counter() - t0
    print(f"\n=== run_custom_pipeline total: {dt:.2f}s ===\n", flush=True)

    s = io.StringIO()
    ps = pstats.Stats(prof, stream=s).sort_stats("cumulative")
    ps.print_stats(35)
    print(s.getvalue())

    s2 = io.StringIO()
    ps2 = pstats.Stats(prof, stream=s2).sort_stats("tottime")
    ps2.print_stats(30)
    print("=== BY TOTTIME ===")
    print(s2.getvalue())


if __name__ == "__main__":
    main()
