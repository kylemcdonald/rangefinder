#!/usr/bin/env python3
"""Run every detector on real ToF-SIMS spectra (detector-only transfer check).

The Zenodo 15446699 spectra (CC-BY) are genuine time-of-flight instrument
output in a channel/mz/intensity text format — no .pos, no APT physics. The
negative-polarity TSB-medium spectrum contains well-known inorganic ions at
exact masses (H, C, CH, O, OH, CN, Cl, CNO, S, PO2, PO3, SO3, ...). We score
each pipeline's detector by recall on that known-ion list and report peak
counts and runtime. Ground truth here is a curated reference list, not a
complete labeling, so in-list fraction is reported as coverage context only.
This does not validate Rangefinder's positive-ion APT assignment or composition
model on SIMS data.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from rangefinder import default_config_path  # noqa: E402

from rangefinder.io.pos_loader import PosSampleData, PosSampleMetadata  # noqa: E402
from rangefinder.utils.config import load_config  # noqa: E402
from rangefinder.utils.paths import slugify  # noqa: E402

# Known inorganic/small ions expected in negative-polarity ToF-SIMS of a
# phosphate-buffered biological growth medium (TSB) on a metal substrate.
KNOWN_NEGATIVE_IONS = {
    "H-": 1.00783,
    "C-": 12.0000,
    "CH-": 13.00783,
    "O-": 15.99491,
    "OH-": 17.00274,
    "C2-": 24.0000,
    "C2H-": 25.00783,
    "CN-": 26.00307,
    "Cl-35": 34.96885,
    "Cl-37": 36.96590,
    "CNO-": 41.99799,
    "PO2-": 62.96356,
    "PO3-": 78.95847,
    "SO3-": 79.95736,
    "H2PO4-": 96.96962,
}

SEED = 20260721


def load_tofsims_events(path: Path, *, max_mz: float, max_events: int, rng) -> np.ndarray:
    frame = pd.read_csv(path, sep="\t", comment="#")
    frame.columns = [c.strip().lower() for c in frame.columns]
    frame = frame[(frame["m/z"] > 0.4) & (frame["m/z"] <= max_mz) & (frame["intensity"] > 0)]
    mz = frame["m/z"].to_numpy(dtype=np.float64)
    counts = frame["intensity"].to_numpy(dtype=np.float64)
    total = counts.sum()
    scale = min(1.0, max_events / max(total, 1.0))
    sampled = rng.poisson(counts * scale) if scale < 1.0 else counts.astype(np.int64)
    # Channel widths for jitter: distance to next channel center.
    widths = np.diff(mz, append=mz[-1] + (mz[-1] - mz[-2] if mz.size > 1 else 0.001))
    events = np.repeat(mz, sampled)
    jitter = rng.uniform(-0.5, 0.5, size=events.size) * np.repeat(widths, sampled)
    return (events + jitter).astype(np.float32)


def main() -> None:
    from rangefinder.analysis.custom_pipeline import run_custom_pipeline
    from rangefinder.analysis.naive_pipeline import run_naive_pipeline
    from rangefinder.analysis.pyccapt_pipeline import run_pyccapt_pipeline
    from rangefinder.analysis.pyopenms_pipeline import run_pyopenms_pipeline

    runners = {
        "custom": lambda s, d, c: run_custom_pipeline(s, d, c, enable_segmentation=False),
        "naive": run_naive_pipeline,
        "pyccapt": run_pyccapt_pipeline,
        "pyopenms": run_pyopenms_pipeline,
    }
    config = copy.deepcopy(load_config(default_config_path()))
    # This is a detector transfer check. Rangefinder's assignment model is for
    # positive-ion APT data and is not meaningful for these negative SIMS ions.
    config["analysis"]["segmentation_enabled"] = False
    config["analysis"]["custom_family_hint_enabled"] = False
    config["analysis"]["element_pruning_enabled"] = False
    config["analysis"]["max_zoom_regions_per_method"] = 0
    rng = np.random.default_rng(SEED)
    output_root = ROOT / "outputs" / "benchmark_tofsims"
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for path in sorted((ROOT / "controls" / "extra").glob("tofsims_*.txt")):
        mz_events = load_tofsims_events(path, max_mz=200.0, max_events=3_000_000, rng=rng)
        print(f"{path.name}: {mz_events.size} pseudo-events")
        n = mz_events.size
        # PosSampleData currently requires APT coordinates. Random placeholders
        # satisfy that container; spatial analysis is disabled and never scored.
        radius = 30.0 * np.sqrt(rng.random(n))
        angle = rng.uniform(0, 2 * np.pi, n)
        slug = slugify(path.stem)
        metadata = PosSampleMetadata(
            path=path, sample_name=path.stem, sample_slug=slug,
            event_count=n, file_size_bytes=n * 16,
            verified_big_endian_float32=True,
            verified_columns=("x_nm", "y_nm", "z_nm", "m_over_z_da"),
        )
        sample = PosSampleData(
            metadata=metadata,
            x_nm=(radius * np.cos(angle)).astype(np.float32),
            y_nm=(radius * np.sin(angle)).astype(np.float32),
            z_nm=rng.uniform(0, 100, n).astype(np.float32),
            m_over_z_da=mz_events,
        )
        is_negative = "neg" in path.name.lower()
        for method, runner in runners.items():
            out_dir = output_root / slug / method
            started = time.perf_counter()
            try:
                artifacts = runner(sample, out_dir, config)
                detected = (
                    np.sort(artifacts.peaks["peak_mz_da"].to_numpy(dtype=float))
                    if not artifacts.peaks.empty
                    else np.asarray([])
                )
                entry = {
                    "spectrum": path.name,
                    "method": method,
                    "runtime_s": round(time.perf_counter() - started, 2),
                    "n_peaks": int(detected.size),
                }
                if is_negative:
                    hits = {}
                    for label, mass in KNOWN_NEGATIVE_IONS.items():
                        tolerance = max(0.05, mass * 500e-6)
                        hits[label] = bool(
                            detected.size and np.min(np.abs(detected - mass)) <= tolerance
                        )
                    entry["known_ion_recall"] = round(
                        sum(hits.values()) / len(hits), 4
                    )
                    entry["known_ions_missed"] = sorted(
                        label for label, hit in hits.items() if not hit
                    )
            except Exception as exc:  # noqa: BLE001
                entry = {"spectrum": path.name, "method": method, "error": repr(exc)}
            results.append(entry)
            print("  ", {k: v for k, v in entry.items() if k not in {"spectrum"}})
    frame = pd.DataFrame(results)
    frame.to_csv(output_root / "tofsims_results.csv", index=False)
    (output_root / "tofsims_results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    print(f"Wrote {output_root / 'tofsims_results.csv'}")


if __name__ == "__main__":
    main()
