#!/usr/bin/env python3
"""Run the custom pipeline against ground-truth controls.

Two kinds of controls are processed:
1. Synthetic POS files generated from known compositions with realistic
   normalized-TOF peak widths and thermal tails. Recovered compositions,
   isotope fractions, and anomaly flags are scored against the ground truth.
2. Any real reference datasets placed in controls/ (e.g. downloaded public APT
   datasets of known materials). These are analyzed and summarized; if
   controls/nominal.yaml provides nominal compositions, an L1 score is added.

Outputs land in outputs/validation/: per-control method outputs, a
validation_elements.csv table, and validation_summary.json consumed by the
report builder.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from rangefinder import default_config_path  # noqa: E402

from rangefinder.analysis.custom_pipeline import run_custom_pipeline  # noqa: E402
from rangefinder.analysis.naive_pipeline import run_naive_pipeline  # noqa: E402
from rangefinder.io.pos_loader import PosSampleData, PosSampleMetadata  # noqa: E402
from rangefinder.utils.config import load_config  # noqa: E402
from rangefinder.utils.paths import slugify  # noqa: E402
from rangefinder.validation.synthetic import (  # noqa: E402
    default_validation_materials,
    synthesize_material_events,
    write_pos_file,
)

SEED = 20260720


def _sample_data_from_arrays(name: str, path: Path, x, y, z, mz) -> PosSampleData:
    metadata = PosSampleMetadata(
        path=path,
        sample_name=name,
        sample_slug=slugify(name),
        event_count=int(mz.size),
        file_size_bytes=int(mz.size) * 16,
        verified_big_endian_float32=True,
        verified_columns=("x_nm", "y_nm", "z_nm", "m_over_z_da"),
    )
    return PosSampleData(metadata=metadata, x_nm=x, y_nm=y, z_nm=z, m_over_z_da=mz)


def _load_pos_arrays(path: Path):
    if path.suffix.lower() == ".epos":
        # epos records are 11 4-byte fields; the first four are x, y, z, m/z.
        raw = np.fromfile(path, dtype=">f4").reshape(-1, 11)
    else:
        raw = np.fromfile(path, dtype=">f4").reshape(-1, 4)
    return (
        raw[:, 0].astype(np.float32),
        raw[:, 1].astype(np.float32),
        raw[:, 2].astype(np.float32),
        raw[:, 3].astype(np.float32),
    )


def _method_element_percents(artifacts) -> dict[str, float]:
    frame = artifacts.elemental_composition
    if frame.empty:
        return {}
    return {
        str(row["element"]): float(row["atomic_percent_weighted"])
        for _, row in frame.iterrows()
    }


def _score_against_truth(recovered: dict[str, float], truth: dict[str, float]) -> dict[str, object]:
    truth_pct = {element: 100.0 * value for element, value in truth.items()}
    elements = sorted(set(recovered) | set(truth_pct))
    l1 = float(
        sum(abs(recovered.get(element, 0.0) - truth_pct.get(element, 0.0)) for element in elements)
    )
    major_truth = {element: pct for element, pct in truth_pct.items() if pct >= 1.0}
    max_major_error = max(
        (abs(recovered.get(element, 0.0) - pct) for element, pct in major_truth.items()),
        default=np.nan,
    )
    top_truth = max(truth_pct, key=truth_pct.get) if truth_pct else None
    top_recovered = max(recovered, key=recovered.get) if recovered else None
    return {
        "elemental_l1_at_pct": l1,
        "max_major_element_error_at_pct": float(max_major_error),
        "top_element_true": top_truth,
        "top_element_recovered": top_recovered,
        "top_element_correct": bool(top_truth == top_recovered),
    }


def _isotope_scores(artifacts) -> dict[str, object]:
    scores: dict[str, object] = {}
    audit = artifacts.isotope_audit
    if not audit.empty and "audit_family_total_counts" in audit.columns:
        # Score only families the audit could actually measure; a family fit on
        # a near-empty window reports meaningless fractions.
        audit = audit[audit["audit_family_total_counts"] >= 1000.0]
    if not audit.empty:
        deltas = (audit["audit_fraction"] - audit["natural_fraction"]).abs()
        scores["audit_max_abs_isotope_error"] = float(deltas.max())
        scores["audit_families"] = int(audit["element"].nunique())
    isotopes = artifacts.isotope_composition
    if not isotopes.empty and "isotope_resolved" in isotopes.columns:
        resolved = isotopes[isotopes["isotope_resolved"] == True]  # noqa: E712
        scores["resolved_isotope_families"] = int(resolved["element"].nunique()) if not resolved.empty else 0
    anomalies = artifacts.isotope_anomalies
    flagged = 0
    strong = 0
    if not anomalies.empty and "status" in anomalies.columns:
        flagged = int(anomalies["status"].isin(["slight", "strong"]).sum())
        strong = int((anomalies["status"] == "strong").sum())
    scores["flagged_isotope_count"] = flagged
    scores["strong_isotope_count"] = strong
    return scores


def main() -> None:
    config = load_config(default_config_path())
    controls_dir = ROOT / "controls"
    synthetic_dir = controls_dir / "synthetic"
    output_root = ROOT / "outputs" / "validation"
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    summary: dict[str, object] = {"synthetic": [], "real_controls": []}
    element_rows: list[dict[str, object]] = []

    methods = [
        ("custom", run_custom_pipeline),
        ("naive", run_naive_pipeline),
    ]
    try:
        from rangefinder.analysis.pyccapt_pipeline import run_pyccapt_pipeline

        methods.append(("pyccapt", run_pyccapt_pipeline))
    except Exception as exc:  # noqa: BLE001
        print(f"pyccapt unavailable for validation: {exc!r}")

    for material in default_validation_materials():
        print(f"=== synthetic control: {material.name}")
        x, y, z, mz = synthesize_material_events(material, rng=rng)
        pos_path = synthetic_dir / f"{material.name}.POS"
        write_pos_file(pos_path, x, y, z, mz)
        sample_data = _sample_data_from_arrays(material.name, pos_path, x, y, z, mz)
        truth = material.true_atomic_fractions()
        material_entry: dict[str, object] = {
            "name": material.name,
            "event_count": int(mz.size),
            "background_fraction": material.background_fraction,
            "mass_resolving_power": material.mass_resolving_power,
            "true_composition_at_pct": {k: 100.0 * v for k, v in sorted(truth.items())},
            "methods": {},
        }
        for method_name, runner in methods:
            method_dir = output_root / material.name / method_name
            try:
                artifacts = runner(sample_data, method_dir, config)
            except Exception as exc:  # noqa: BLE001
                material_entry["methods"][method_name] = {"error": repr(exc)}
                print(f"  {method_name}: FAILED {exc!r}")
                continue
            recovered = _method_element_percents(artifacts)
            entry = _score_against_truth(recovered, truth)
            entry |= _isotope_scores(artifacts)
            entry["recovered_composition_at_pct"] = {
                element: round(pct, 3)
                for element, pct in sorted(recovered.items(), key=lambda item: -item[1])
                if pct >= 0.05
            }
            segmentation = (
                artifacts.diagnostics.get("segmentation")
                if isinstance(artifacts.diagnostics, dict)
                else None
            )
            if isinstance(segmentation, dict) and method_name == "custom":
                expected_regions = int(material.extra.get("expected_regions", 1))
                detected_regions = int(segmentation.get("n_regions", 1))
                entry["expected_regions"] = expected_regions
                entry["n_regions_detected"] = detected_regions
                entry["regions_correct"] = bool(detected_regions == expected_regions)
                detected_tops = [
                    max(region["top_elements_at_pct"], key=region["top_elements_at_pct"].get)
                    for region in segmentation.get("region_summary", [])
                    if isinstance(region.get("top_elements_at_pct"), dict) and region["top_elements_at_pct"]
                ]
                if detected_tops:
                    entry["region_top_elements"] = detected_tops
                expected_tops = material.extra.get("expected_region_top_elements")
                if isinstance(expected_tops, list) and expected_tops:
                    entry["region_top_elements_correct"] = bool(
                        sorted(detected_tops) == sorted(str(e) for e in expected_tops)
                    )
                print(
                    f"    segmentation: {detected_regions} region(s) "
                    f"(expected {expected_regions}) tops={detected_tops or '-'}"
                )
            material_entry["methods"][method_name] = entry
            print(
                f"  {method_name}: L1={entry['elemental_l1_at_pct']:.2f} "
                f"top_ok={entry['top_element_correct']} flags={entry.get('flagged_isotope_count')}"
            )
            for element in sorted(set(recovered) | set(truth)):
                element_rows.append(
                    {
                        "control": material.name,
                        "method": method_name,
                        "element": element,
                        "true_at_pct": 100.0 * truth.get(element, 0.0),
                        "recovered_at_pct": recovered.get(element, 0.0),
                    }
                )
        summary["synthetic"].append(material_entry)

    nominal_path = controls_dir / "nominal.yaml"
    nominal: dict[str, dict[str, float]] = {}
    if nominal_path.exists():
        import yaml

        nominal = yaml.safe_load(nominal_path.read_text(encoding="utf-8")) or {}
    suffix_priority = {".pos": 0, ".epos": 1}
    real_controls = sorted(
        {
            path
            for pattern in ("*.pos", "*.POS", "*.epos", "*.EPOS")
            for path in controls_dir.glob(pattern)
        },
        key=lambda path: (path.stem, suffix_priority.get(path.suffix.lower(), 2)),
    )
    seen_stems: set[str] = set()
    for path in real_controls:
        if path.stem in seen_stems:
            continue
        seen_stems.add(path.stem)
        print(f"=== real control: {path.name}")
        try:
            x, y, z, mz = _load_pos_arrays(path)
            sample_data = _sample_data_from_arrays(path.stem, path, x, y, z, mz)
            method_dir = output_root / slugify(path.stem) / "custom"
            artifacts = run_custom_pipeline(sample_data, method_dir, config)
        except Exception as exc:  # noqa: BLE001
            summary["real_controls"].append({"name": path.name, "error": repr(exc)})
            print(f"  FAILED {exc!r}")
            continue
        recovered = _method_element_percents(artifacts)
        entry: dict[str, object] = {
            "name": path.name,
            "event_count": int(mz.size),
            "recovered_composition_at_pct": {
                element: round(pct, 3)
                for element, pct in sorted(recovered.items(), key=lambda item: -item[1])
                if pct >= 0.05
            },
        }
        entry |= _isotope_scores(artifacts)
        nominal_entry = nominal.get(path.name) or nominal.get(path.stem)
        if isinstance(nominal_entry, dict) and nominal_entry:
            truth = {
                str(element): float(pct) / 100.0 for element, pct in nominal_entry.items()
            }
            entry |= _score_against_truth(recovered, truth)
            entry["nominal_composition_at_pct"] = nominal_entry
        summary["real_controls"].append(entry)
        print(f"  top: {list(entry['recovered_composition_at_pct'].items())[:4]}")

    pd.DataFrame(element_rows).to_csv(output_root / "validation_elements.csv", index=False)
    (output_root / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Wrote {output_root / 'validation_summary.json'}")


if __name__ == "__main__":
    main()
