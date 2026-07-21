#!/usr/bin/env python3
"""Ablation study: rerun the custom pipeline with each defense disabled.

Quantifies what every stage contributes on ground-truth controls: composition
L1, fabricated-element mass, and weighted line recall. Feeds the ablation
table of the Rangefinder paper.
"""
from __future__ import annotations

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

# "no_X" disables a stage that is ON in the default pipeline (shows its cost).
# "with_X" RE-ENABLES a stage that is OFF by default (shows it does not help,
# justifying its removal in favor of the simpler pipeline).
ABLATIONS: dict[str, dict[str, object]] = {
    "full": {},
    "no_envelope_veto": {"molecular_envelope_penalty": 0.0},
    "no_atomic_support_penalty": {"atomic_unsupported_element_penalty": 0.0},
    "no_element_pruning": {"element_pruning_enabled": False},
    "no_probability_floor": {"composition_probability_floor": 0.0},
    "no_family_hints": {"custom_family_hint_enabled": False},
    "no_crowded_window_fit": {"custom_local_fit_enabled": False},
    "no_molecular_families": {"custom_molecular_family_recovery_enabled": False},
    "no_member_verification": {"custom_family_member_verify_enabled": False},
    "with_log_peakiness": {"custom_log_peakiness_enabled": True},
    "with_family_completion": {"custom_family_completion_tolerance_scale": 2.5},
}

ABLATION_DATASETS = {
    "synthetic_al_mg_si",
    "synthetic_si_o",
    "synthetic_zn_cu_al",
    "synthetic_zn_layer_on_al",
    "control_W_18K_breen_kuehbach_R18_53222",
    "control_Ck10_steel_felfer_R56_01769",
    "control_Si_apav_usa_denton_smith",
    "feoxide_10677562_R5076_69145-v01",
    "li_14848236_R5076_68722",
}


def main() -> None:
    base_config = load_config(default_config_path())
    controls_dir = ROOT / "controls"
    nominal = {}
    nominal_path = controls_dir / "nominal.yaml"
    if nominal_path.exists():
        import yaml

        nominal = yaml.safe_load(nominal_path.read_text(encoding="utf-8")) or {}
    datasets = [
        dataset
        for dataset in synthetic_datasets(controls_dir) + real_control_datasets(controls_dir, nominal)
        if dataset.name in ABLATION_DATASETS
    ]
    output_root = ROOT / "outputs" / "benchmark_ablation"
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for dataset in datasets:
        if dataset.arrays is None:
            dataset.arrays = load_pos_arrays(dataset.pos_path)
        attach_range_reference_truth(dataset, dataset.arrays[3])
        x, y, z, mz = dataset.arrays
        metadata = PosSampleMetadata(
            path=dataset.pos_path, sample_name=dataset.name, sample_slug=slugify(dataset.name),
            event_count=int(mz.size), file_size_bytes=int(mz.size) * 16,
            verified_big_endian_float32=True,
            verified_columns=("x_nm", "y_nm", "z_nm", "m_over_z_da"),
        )
        sample = PosSampleData(metadata=metadata, x_nm=x, y_nm=y, z_nm=z, m_over_z_da=mz)
        for ablation, overrides in ABLATIONS.items():
            config = copy.deepcopy(base_config)
            config["analysis"]["segmentation_enabled"] = False
            for key, value in overrides.items():
                config["analysis"][key] = value
            out_dir = output_root / slugify(dataset.name) / ablation
            started = time.perf_counter()
            try:
                artifacts = run_custom_pipeline(sample, out_dir, config, enable_segmentation=False)
                recovered = {
                    str(row["element"]): float(row["atomic_percent_weighted"])
                    for _, row in artifacts.elemental_composition.iterrows()
                    if float(row["atomic_percent_weighted"]) >= 1.0e-4
                }
                entry = {
                    "dataset": dataset.name,
                    "ablation": ablation,
                    "runtime_s": round(time.perf_counter() - started, 2),
                    "n_peaks": int(artifacts.peaks.shape[0]),
                }
                if dataset.truth_at_pct:
                    entry |= composition_metrics(
                        recovered,
                        dataset.truth_at_pct,
                        allowed_extra_elements=allowed_extra_elements(dataset),
                    )
                if dataset.truth_lines is not None:
                    detected = artifacts.peaks["peak_mz_da"].to_numpy(dtype=float)
                    entry |= peak_metrics_vs_lines(
                        detected, dataset.truth_lines,
                        mass_resolving_power=dataset.mass_resolving_power,
                    )
            except Exception as exc:  # noqa: BLE001
                entry = {"dataset": dataset.name, "ablation": ablation, "error": repr(exc)}
            results.append(entry)
            print(
                f"{dataset.name} / {ablation}: "
                f"L1={entry.get('elemental_l1_at_pct', float('nan')):.2f} "
                f"fab={entry.get('fabricated_at_pct', float('nan')):.2f}",
                flush=True,
            )
        pd.DataFrame(results).to_csv(output_root / "ablation_results.csv", index=False)
    (output_root / "ablation_results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    print(f"Wrote {output_root / 'ablation_results.csv'}")


if __name__ == "__main__":
    main()
