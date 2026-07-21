#!/usr/bin/env python3
"""Detection-stage ablation: swap only the peak detector, hold integration and
the shared assignment stage constant, and score composition + detection.

This isolates the impact of the detection algorithm. Rangefinder's own
multi-resolution detector is compared against a naive prominence detector and
the standard 1D mass-spectrum detectors that APT ranging never adopted (CWT
ridge lines, persistent homology, Savitzky--Golay derivative zero-crossings).

Output: outputs/detection_ablation/detection_ablation_results.json (+ .csv).

Usage:
  .venv/bin/python scripts/run_detection_ablation.py [--datasets a,b] [--detectors rangefinder,naive,...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from rangefinder import default_config_path  # noqa: E402

warnings.filterwarnings("ignore", message="some peaks have a")

from rangefinder.benchmark.detectors import DETECTOR_LABELS, run_detection_ablation  # noqa: E402
from rangefinder.benchmark.metrics import (  # noqa: E402
    composition_metrics,
    peak_metrics_vs_lines,
    peak_metrics_vs_ranges,
)
from rangefinder.benchmark.truth import (  # noqa: E402
    BenchmarkDataset,
    allowed_extra_elements,
    attach_range_reference_truth,
    load_pos_arrays,
    real_control_datasets,
    synthetic_datasets,
)
from rangefinder.io.pos_loader import PosSampleData, PosSampleMetadata  # noqa: E402
from rangefinder.utils.config import load_config  # noqa: E402
from rangefinder.utils.paths import slugify  # noqa: E402

ALL_DETECTORS = ("rangefinder", "naive", "smoothed", "cwt", "persistent_homology", "derivative")


def _sample_data(dataset: BenchmarkDataset) -> PosSampleData:
    x, y, z, mz = dataset.arrays
    metadata = PosSampleMetadata(
        path=dataset.pos_path,
        sample_name=dataset.name,
        sample_slug=slugify(dataset.name),
        event_count=int(mz.size),
        file_size_bytes=int(mz.size) * 16,
        verified_big_endian_float32=True,
        verified_columns=("x_nm", "y_nm", "z_nm", "m_over_z_da"),
    )
    return PosSampleData(metadata=metadata, x_nm=x, y_nm=y, z_nm=z, m_over_z_da=mz)


def _recovered_composition(artifacts) -> dict[str, float]:
    frame = artifacts.elemental_composition
    if frame is None or frame.empty:
        return {}
    return {
        str(row["element"]): float(row["atomic_percent_weighted"])
        for _, row in frame.iterrows()
        if float(row["atomic_percent_weighted"]) >= 1.0e-4
    }


def run_one(dataset, detector, output_dir, config) -> dict[str, object]:
    result: dict[str, object] = {"dataset": dataset.name, "detector": detector, "label": DETECTOR_LABELS[detector]}
    started = time.perf_counter()
    artifacts = run_detection_ablation(_sample_data(dataset), output_dir, config, detector)
    result["runtime_s"] = round(time.perf_counter() - started, 2)
    recovered = _recovered_composition(artifacts)
    if dataset.truth_at_pct:
        result |= composition_metrics(
            recovered, dataset.truth_at_pct, allowed_extra_elements=allowed_extra_elements(dataset)
        )
        result["truth_source"] = dataset.truth_source
    detected_mz = (
        artifacts.peaks["peak_mz_da"].to_numpy(dtype=float)
        if artifacts.peaks is not None and not artifacts.peaks.empty
        else np.asarray([], dtype=float)
    )
    result["n_peaks"] = int(detected_mz.size)
    if dataset.truth_lines is not None:
        result |= peak_metrics_vs_lines(detected_mz, dataset.truth_lines, mass_resolving_power=dataset.mass_resolving_power)
    if dataset.ranges is not None and not dataset.ranges.empty:
        result |= peak_metrics_vs_ranges(detected_mz, dataset.ranges, dataset.arrays[3])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--detectors", type=str, default=",".join(ALL_DETECTORS))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "detection_ablation")
    args = parser.parse_args()

    config = load_config(default_config_path())
    controls_dir = ROOT / "controls"
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    nominal: dict[str, dict[str, float]] = {}
    nominal_path = controls_dir / "nominal.yaml"
    if nominal_path.exists():
        import yaml

        nominal = yaml.safe_load(nominal_path.read_text(encoding="utf-8")) or {}

    datasets = synthetic_datasets(controls_dir) + real_control_datasets(controls_dir, nominal)
    if args.datasets:
        wanted = {name.strip() for name in args.datasets.split(",")}
        datasets = [d for d in datasets if d.name in wanted]
    detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]

    all_results: list[dict[str, object]] = []
    for dataset in datasets:
        print(f"=== dataset: {dataset.name} ({dataset.kind})", flush=True)
        if dataset.arrays is None:
            dataset.arrays = load_pos_arrays(dataset.pos_path)
        attach_range_reference_truth(dataset, dataset.arrays[3])
        for detector in detectors:
            method_dir = output_root / slugify(dataset.name) / detector
            try:
                entry = run_one(dataset, detector, method_dir, config)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                entry = {"dataset": dataset.name, "detector": detector, "error": repr(exc)}
            all_results.append(entry)
            bits = [
                f"L1={entry['elemental_l1_at_pct']:.2f}" if "elemental_l1_at_pct" in entry else "",
                f"fab={entry['fabricated_at_pct']:.2f}" if "fabricated_at_pct" in entry else "",
                f"prec={entry['line_precision']:.3f}" if "line_precision" in entry else "",
                f"ppm={entry['median_ppm_error']:.1f}" if "median_ppm_error" in entry else "",
                f"t={entry.get('runtime_s','?')}s",
                "ERROR" if "error" in entry else "",
            ]
            print(f"  {detector}: " + " ".join(b for b in bits if b), flush=True)
        (output_root / "detection_ablation_results.json").write_text(
            json.dumps(all_results, indent=2, default=str), encoding="utf-8"
        )
    pd.json_normalize(all_results).to_csv(output_root / "detection_ablation_table.csv", index=False)
    print(f"Wrote {output_root / 'detection_ablation_results.json'}")


if __name__ == "__main__":
    main()
