#!/usr/bin/env python3
"""Run every analysis method against every benchmark dataset and score them.

Datasets: synthetic ground-truth materials (exact composition + line list)
and real public reference datasets in controls/ (nominal composition and/or
expert range files). Methods: all seven pipelines (custom, naive, pyccapt,
pyopenms in-process; apav, ms_deisotope via the 3.12 bridge; apyt via the
3.13 bridge).

Outputs land in outputs/benchmark/: per-dataset per-method artifacts,
benchmark_results.json, and benchmark_table.csv.

Usage:
  .venv/bin/python scripts/run_benchmark.py [--datasets name1,name2]
      [--methods custom,apav,...] [--skip-existing]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from rangefinder import default_config_path  # noqa: E402

from rangefinder.analysis.custom_pipeline import run_custom_pipeline  # noqa: E402
from rangefinder.analysis.naive_pipeline import run_naive_pipeline  # noqa: E402
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
from rangefinder.bridge import run_bridge_method  # noqa: E402
from rangefinder.io.pos_loader import PosSampleData, PosSampleMetadata  # noqa: E402
from rangefinder.utils.config import load_config  # noqa: E402
from rangefinder.utils.paths import ProjectPaths, slugify  # noqa: E402
from rangefinder.validation.synthetic import write_pos_file  # noqa: E402

BRIDGE_METHODS = ("apav", "ms_deisotope", "apyt")
IN_PROCESS_METHODS = {
    "custom": run_custom_pipeline,
    "naive": run_naive_pipeline,
}
METHOD_LABELS = {
    "custom": "Custom implementation",
    "naive": "Naive sideband",
    "pyccapt": "PyCCAPT",
    "pyopenms": "pyOpenMS",
    "apav": "APAV",
    "ms_deisotope": "ms_deisotope",
    "apyt": "APyT",
}
ALL_METHODS = ("custom", "naive", "pyccapt", "pyopenms", "apav", "ms_deisotope", "apyt")


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


def _isotope_scores(artifacts) -> dict[str, object]:
    scores: dict[str, object] = {}
    audit = artifacts.isotope_audit
    if audit is not None and not audit.empty and "audit_family_total_counts" in audit.columns:
        audit = audit[audit["audit_family_total_counts"] >= 1000.0]
        if not audit.empty:
            deltas = (audit["audit_fraction"] - audit["natural_fraction"]).abs()
            scores["audit_max_abs_isotope_error"] = float(deltas.max())
    anomalies = artifacts.isotope_anomalies
    flagged = strong = 0
    if anomalies is not None and not anomalies.empty and "status" in anomalies.columns:
        flagged = int(anomalies["status"].isin(["slight", "strong"]).sum())
        strong = int((anomalies["status"] == "strong").sum())
    scores["flagged_isotope_count"] = flagged
    scores["strong_isotope_count"] = strong
    return scores


def run_one(
    dataset: BenchmarkDataset,
    method: str,
    output_dir: Path,
    config: dict,
    paths: ProjectPaths,
    *,
    bridge_pos_path: Path,
    skip_existing: bool,
) -> dict[str, object]:
    from rangefinder.analysis.pipeline_utils import load_method_outputs

    result: dict[str, object] = {"dataset": dataset.name, "method": method}
    marker = output_dir / "benchmark_run.json"
    if skip_existing and marker.exists() and (output_dir / "elemental_composition.csv").exists():
        artifacts = load_method_outputs(
            method_name=method, method_label=METHOD_LABELS[method], output_dir=output_dir
        )
        result["runtime_s"] = json.loads(marker.read_text())["runtime_s"]
        result["reused"] = True
    else:
        started = time.perf_counter()
        if method in IN_PROCESS_METHODS:
            artifacts = IN_PROCESS_METHODS[method](_sample_data(dataset), output_dir, config)
        elif method in ("pyccapt", "pyopenms"):
            if method == "pyccapt":
                from rangefinder.analysis.pyccapt_pipeline import run_pyccapt_pipeline as runner
            else:
                from rangefinder.analysis.pyopenms_pipeline import run_pyopenms_pipeline as runner
            artifacts = runner(_sample_data(dataset), output_dir, config)
        elif method in BRIDGE_METHODS:
            artifacts = run_bridge_method(
                method_name=method,
                method_label=METHOD_LABELS[method],
                sample_path=bridge_pos_path,
                sample_slug=slugify(f"bench-{dataset.name}"),
                output_dir=output_dir,
                paths=paths,
            )
        else:
            raise ValueError(f"Unknown method {method}")
        result["runtime_s"] = round(time.perf_counter() - started, 2)
        marker.write_text(json.dumps({"runtime_s": result["runtime_s"]}))
    recovered = _recovered_composition(artifacts)
    result["recovered_composition_at_pct"] = {
        element: round(pct, 4)
        for element, pct in sorted(recovered.items(), key=lambda item: -item[1])
        if pct >= 0.01
    }
    if dataset.truth_at_pct:
        result |= composition_metrics(
            recovered,
            dataset.truth_at_pct,
            allowed_extra_elements=allowed_extra_elements(dataset),
        )
        result["truth_source"] = dataset.truth_source
    detected_mz = (
        artifacts.peaks["peak_mz_da"].to_numpy(dtype=float)
        if artifacts.peaks is not None and not artifacts.peaks.empty
        else np.asarray([], dtype=float)
    )
    result["n_peaks"] = int(detected_mz.size)
    if dataset.truth_lines is not None:
        result |= peak_metrics_vs_lines(
            detected_mz,
            dataset.truth_lines,
            mass_resolving_power=dataset.mass_resolving_power,
        )
    if dataset.ranges is not None and not dataset.ranges.empty:
        result |= peak_metrics_vs_ranges(detected_mz, dataset.ranges, dataset.arrays[3])
    result |= _isotope_scores(artifacts)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default=None, help="comma-separated dataset names")
    parser.add_argument("--methods", type=str, default=",".join(ALL_METHODS))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "benchmark")
    args = parser.parse_args()

    config = load_config(default_config_path())
    paths = ProjectPaths.from_root(ROOT)
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
        datasets = [dataset for dataset in datasets if dataset.name in wanted]
    methods = [name.strip() for name in args.methods.split(",") if name.strip()]

    scratch_dir = output_root / "_bridge_pos"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, object]] = []
    for dataset in datasets:
        print(f"=== dataset: {dataset.name} ({dataset.kind})", flush=True)
        if dataset.arrays is None:
            dataset.arrays = load_pos_arrays(dataset.pos_path)
        attach_range_reference_truth(dataset, dataset.arrays[3])
        if dataset.pos_path.suffix.lower() == ".epos":
            bridge_pos_path = scratch_dir / f"{dataset.name}.POS"
            if not bridge_pos_path.exists():
                write_pos_file(bridge_pos_path, *dataset.arrays)
        else:
            bridge_pos_path = dataset.pos_path
        for method in methods:
            method_dir = output_root / slugify(dataset.name) / method
            try:
                entry = run_one(
                    dataset,
                    method,
                    method_dir,
                    config,
                    paths,
                    bridge_pos_path=bridge_pos_path,
                    skip_existing=args.skip_existing,
                )
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                entry = {"dataset": dataset.name, "method": method, "error": repr(exc)}
            all_results.append(entry)
            summary_bits = [
                f"L1={entry['elemental_l1_at_pct']:.2f}" if "elemental_l1_at_pct" in entry else "",
                f"fab={entry['fabricated_at_pct']:.2f}" if "fabricated_at_pct" in entry else "",
                f"recall={entry.get('line_recall_weighted', entry.get('range_recall_ion_weighted', float('nan'))):.3f}"
                if ("line_recall_weighted" in entry or "range_recall_ion_weighted" in entry)
                else "",
                f"t={entry.get('runtime_s', '?')}s",
                "ERROR" if "error" in entry else "",
            ]
            print(f"  {method}: " + " ".join(bit for bit in summary_bits if bit), flush=True)
        # Persist incrementally so long runs are inspectable while running.
        (output_root / "benchmark_results.json").write_text(
            json.dumps(all_results, indent=2, default=str), encoding="utf-8"
        )
    flat = pd.json_normalize(all_results)
    flat.to_csv(output_root / "benchmark_table.csv", index=False)
    print(f"Wrote {output_root / 'benchmark_results.json'} and benchmark_table.csv")


if __name__ == "__main__":
    main()
