#!/usr/bin/env python3
"""Ablate adaptive ranging and isotope-envelope overlap quantification.

Peak detection and species assignment code are held fixed; each row changes
only the range-area policy and/or the post-assignment composition allocation.
Outputs are written incrementally to ``outputs/quantification_ablation``.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rangefinder import default_config_path  # noqa: E402
from rangefinder.analysis.custom_pipeline import run_custom_pipeline  # noqa: E402
from rangefinder.benchmark.metrics import composition_metrics, peak_metrics_vs_lines  # noqa: E402
from rangefinder.benchmark.truth import (  # noqa: E402
    allowed_extra_elements,
    attach_range_reference_truth,
    load_pos_arrays,
    real_control_datasets,
    synthetic_datasets,
)
from rangefinder.io.pos_loader import PosSampleData, PosSampleMetadata  # noqa: E402
from rangefinder.utils.config import load_config  # noqa: E402
from rangefinder.utils.paths import slugify  # noqa: E402


VARIANTS: dict[str, dict[str, object]] = {
    "legacy": {
        "adaptive_ranging_mode": "off",
        "overlap_deconvolution_enabled": False,
    },
    "eer": {
        "adaptive_ranging_mode": "eer",
        "overlap_deconvolution_enabled": False,
    },
    "hybrid_preserve_fit": {
        "adaptive_ranging_mode": "hybrid",
        "eer_hybrid_preserve_local_mixture": True,
        "overlap_deconvolution_enabled": False,
    },
    "hybrid_all": {
        "adaptive_ranging_mode": "hybrid",
        "eer_hybrid_preserve_local_mixture": False,
        "overlap_deconvolution_enabled": False,
    },
    "deconvolution": {
        "adaptive_ranging_mode": "off",
        "overlap_deconvolution_enabled": True,
    },
    "hybrid_all_deconvolution": {
        "adaptive_ranging_mode": "hybrid",
        "eer_hybrid_preserve_local_mixture": False,
        "overlap_deconvolution_enabled": True,
    },
    "eer_deconvolution": {
        "adaptive_ranging_mode": "eer",
        "overlap_deconvolution_enabled": True,
    },
}


def _sample(dataset) -> PosSampleData:
    x, y, z, mz = dataset.arrays
    return PosSampleData(
        metadata=PosSampleMetadata(
            path=dataset.pos_path,
            sample_name=dataset.name,
            sample_slug=slugify(dataset.name),
            event_count=int(mz.size),
            file_size_bytes=int(mz.size) * 16,
            verified_big_endian_float32=True,
            verified_columns=("x_nm", "y_nm", "z_nm", "m_over_z_da"),
        ),
        x_nm=x,
        y_nm=y,
        z_nm=z,
        m_over_z_da=mz,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--variants", type=str, default=",".join(VARIANTS))
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "quantification_ablation"
    )
    args = parser.parse_args()

    base_config = load_config(default_config_path())
    controls_dir = ROOT / "controls"
    nominal_path = controls_dir / "nominal.yaml"
    if nominal_path.exists():
        import yaml

        nominal = yaml.safe_load(nominal_path.read_text(encoding="utf-8")) or {}
    else:
        nominal = {}
    datasets = synthetic_datasets(controls_dir) + real_control_datasets(controls_dir, nominal)
    if args.datasets:
        wanted = {name.strip() for name in args.datasets.split(",")}
        datasets = [dataset for dataset in datasets if dataset.name in wanted]
    variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {', '.join(unknown)}")

    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for dataset in datasets:
        dataset.arrays = load_pos_arrays(dataset.pos_path)
        attach_range_reference_truth(dataset, dataset.arrays[3])
        sample = _sample(dataset)
        for variant in variants:
            config = copy.deepcopy(base_config)
            config["analysis"]["segmentation_enabled"] = False
            config["analysis"].update(VARIANTS[variant])
            started = time.perf_counter()
            try:
                artifacts = run_custom_pipeline(
                    sample,
                    output_root / slugify(dataset.name) / variant,
                    config,
                    enable_segmentation=False,
                )
                recovered = {
                    str(row["element"]): float(row["atomic_percent_weighted"])
                    for _, row in artifacts.elemental_composition.iterrows()
                    if float(row["atomic_percent_weighted"]) >= 1.0e-4
                }
                ranging = artifacts.diagnostics.get("adaptive_ranging", {})
                deconvolution = artifacts.diagnostics.get("overlap_deconvolution", {})
                entry: dict[str, object] = {
                    "dataset": dataset.name,
                    "dataset_kind": dataset.kind,
                    "truth_source": dataset.truth_source,
                    "variant": variant,
                    "runtime_s": round(time.perf_counter() - started, 3),
                    "n_peaks": int(artifacts.peaks.shape[0]),
                    "adaptive_selected_peak_count": int(ranging.get("selected_peak_count", 0)),
                    "adaptive_selected_area_fraction": float(
                        ranging.get("selected_area_fraction", 0.0)
                    ),
                    "deconvolution_accepted_components": int(
                        deconvolution.get("accepted_component_count", 0)
                    ),
                    "deconvolution_rejected_components": int(
                        deconvolution.get("rejected_component_count", 0)
                    ),
                    "deconvolution_changed_area_fraction": float(
                        deconvolution.get("changed_area_fraction", 0.0)
                    ),
                }
                if dataset.truth_at_pct:
                    entry |= composition_metrics(
                        recovered,
                        dataset.truth_at_pct,
                        allowed_extra_elements=allowed_extra_elements(dataset),
                    )
                if dataset.truth_lines is not None:
                    entry |= peak_metrics_vs_lines(
                        artifacts.peaks["peak_mz_da"].to_numpy(dtype=float),
                        dataset.truth_lines,
                        mass_resolving_power=dataset.mass_resolving_power,
                    )
            except Exception as exc:  # noqa: BLE001
                entry = {
                    "dataset": dataset.name,
                    "dataset_kind": dataset.kind,
                    "variant": variant,
                    "error": repr(exc),
                }
            results.append(entry)
            print(
                f"{dataset.name} / {variant}: "
                f"L1={entry.get('elemental_l1_at_pct', float('nan')):.3f} "
                f"adaptive={entry.get('adaptive_selected_peak_count', 0)} "
                f"deconv={entry.get('deconvolution_accepted_components', 0)}",
                flush=True,
            )
            pd.DataFrame(results).to_csv(output_root / "quantification_ablation.csv", index=False)
            (output_root / "quantification_ablation.json").write_text(
                json.dumps(results, indent=2, default=str), encoding="utf-8"
            )
    print(f"Wrote {output_root / 'quantification_ablation.csv'}")


if __name__ == "__main__":
    main()
