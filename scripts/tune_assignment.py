#!/usr/bin/env python3
"""Tune peak-assignment scoring against trusted expert range files.

The expensive peak-detection stage is intentionally not part of this loop.
Run the custom benchmark once, then this script reuses each cached peaks.csv
and the raw m/z values while rebuilding assignments for every trial. Species
libraries are cached across trials, making supervised scoring experiments
fast enough to be reproducible rather than anecdotal.

By default the training set is the three primary controls whose range files
are suitable as species-identification truth. The broad/incomplete Si ranges
and historical/deuterium extra files remain useful held-out diagnostics, but
are not silently treated as clean labels.

Examples:
  .venv/bin/python scripts/tune_assignment.py \
      --benchmark-dir outputs/benchmark_assignment_v4
  .venv/bin/python scripts/tune_assignment.py \
      --benchmark-dir outputs/benchmark_assignment_v4 \
      --optimize molecular --maxiter 8 --popsize 8
"""
from __future__ import annotations

import argparse
import ast
import copy
import itertools
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rangefinder import default_config_path  # noqa: E402
from rangefinder.analysis.assignment import assign_peaks  # noqa: E402
from rangefinder.benchmark.metrics import assignment_metrics_vs_ranges  # noqa: E402
from rangefinder.benchmark.truth import (  # noqa: E402
    BenchmarkDataset,
    load_pos_arrays,
    real_control_datasets,
)
from rangefinder.utils.config import load_config  # noqa: E402
from rangefinder.utils.paths import slugify  # noqa: E402


TRUSTED_TRAINING_DATASETS = (
    "control_Ck10_steel_felfer_R56_01769",
    "control_MoHf_leitner_R21_08680",
    "control_ODSsteel_wang_R31_06365",
)

OBJECT_COLUMNS = {
    "detection_sources",
    "family_hint_candidates",
    "molecular_family_candidates",
    "molecular_family_formula",
}

PARAMETER_PRESETS: dict[str, tuple[tuple[str, float, float], ...]] = {
    "molecular": (
        ("molecular_family_hint_weight", 0.0, 8.0),
        ("molecular_family_contradiction_penalty", 0.0, 8.0),
    ),
    "scoring": (
        ("mass_score_weight", 1.5, 6.0),
        ("natural_score_weight", 0.0, 2.0),
        ("family_score_weight", 0.4, 4.0),
        ("element_prior_weight", 0.0, 2.0),
        ("charge_prior_weight", 0.0, 1.5),
        ("family_prior_weight", 0.0, 2.0),
        ("family_hint_weight", 0.0, 2.0),
        ("family_hint_multi_weight", 0.0, 1.2),
        ("family_hint_score_saturation", 0.4, 8.0),
        ("molecular_envelope_penalty", 0.0, 5.0),
        ("molecular_complexity_penalty_per_extra_atom", 0.0, 2.0),
        ("molecular_unsupported_element_penalty", 0.0, 2.0),
        ("atomic_unsupported_element_penalty", 0.0, 2.0),
    ),
}


def _parse_object(value: object) -> object:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return value


def load_cached_peaks(path: Path) -> pd.DataFrame:
    peaks = pd.read_csv(path)
    for column in OBJECT_COLUMNS & set(peaks.columns):
        peaks[column] = peaks[column].map(_parse_object)
    # Older cached tables were produced when family recovery overwrote the
    # observed centroid. Reconstruct the candidate-specific reference fields
    # and restore the measured/raw-refined position so tuning exercises the
    # current algorithm without rerunning expensive detection first.
    if "peak_mz_raw_da" in peaks.columns:
        observed = peaks["peak_mz_raw_da"].where(
            peaks["peak_mz_raw_da"].notna(),
            peaks["peak_mz_da"],
        )
        if "family_hint_element" in peaks.columns:
            atomic_mask = peaks["family_hint_element"].notna()
            if "family_hint_reference_mz_da" not in peaks.columns:
                peaks["family_hint_reference_mz_da"] = np.nan
            missing_atomic_reference = atomic_mask & peaks[
                "family_hint_reference_mz_da"
            ].isna()
            peaks.loc[
                missing_atomic_reference,
                "family_hint_reference_mz_da",
            ] = peaks.loc[
                missing_atomic_reference,
                "peak_mz_da",
            ]
        if "molecular_family_formula" in peaks.columns:
            molecular_mask = peaks["molecular_family_formula"].map(
                lambda value: isinstance(value, dict) and bool(value)
            )
            if "molecular_family_reference_mz_da" not in peaks.columns:
                peaks["molecular_family_reference_mz_da"] = np.nan
            missing_molecular_reference = molecular_mask & peaks[
                "molecular_family_reference_mz_da"
            ].isna()
            peaks.loc[
                missing_molecular_reference,
                "molecular_family_reference_mz_da",
            ] = peaks.loc[
                missing_molecular_reference,
                "peak_mz_da",
            ]
        peaks["peak_mz_da"] = observed
    return peaks


@dataclass
class TrainingControl:
    dataset: BenchmarkDataset
    peaks: pd.DataFrame
    mz: np.ndarray
    library_cache: dict = field(default_factory=dict)


def _range_path_for_dataset(dataset: BenchmarkDataset) -> Path | None:
    if dataset.ranges is None:
        return None
    for suffix in (".rrng", ".RRNG", ".rng.fig.txt"):
        path = dataset.pos_path.with_suffix(suffix)
        if path.exists():
            return path
    return None


def load_controls(
    benchmark_dir: Path,
    requested_names: set[str],
) -> list[TrainingControl]:
    nominal_path = ROOT / "controls" / "nominal.yaml"
    nominal: dict[str, dict[str, float]] = {}
    if nominal_path.exists():
        import yaml

        nominal = yaml.safe_load(nominal_path.read_text(encoding="utf-8")) or {}
    registry = {
        dataset.name: dataset
        for dataset in real_control_datasets(ROOT / "controls", nominal)
    }
    controls: list[TrainingControl] = []
    missing: list[str] = []
    for name in sorted(requested_names):
        dataset = registry.get(name)
        if dataset is None or dataset.ranges is None or dataset.ranges.empty:
            missing.append(f"{name} (no registered range truth)")
            continue
        peaks_path = benchmark_dir / slugify(name) / "custom" / "peaks.csv"
        if not peaks_path.exists():
            missing.append(f"{name} ({peaks_path} is missing)")
            continue
        mz = load_pos_arrays(dataset.pos_path)[3]
        controls.append(
            TrainingControl(
                dataset=dataset,
                peaks=load_cached_peaks(peaks_path),
                mz=np.asarray(mz, dtype=np.float64),
            )
        )
    if missing:
        raise FileNotFoundError("Cannot load training controls:\n  " + "\n  ".join(missing))
    return controls


def evaluate_parameters(
    controls: list[TrainingControl],
    base_config: dict,
    parameters: dict[str, float],
) -> tuple[float, list[dict[str, Any]]]:
    config = copy.deepcopy(base_config)
    config["analysis"].update(parameters)
    rows: list[dict[str, Any]] = []
    for control in controls:
        assignments, _ = assign_peaks(
            control.peaks,
            config=config,
            library_cache=control.library_cache,
        )
        metrics = assignment_metrics_vs_ranges(
            control.peaks,
            assignments,
            control.dataset.ranges,
            control.mz,
        )
        rows.append(
            {
                "dataset": control.dataset.name,
                **metrics,
            }
        )
    exact = np.asarray(
        [row["assignment_formula_accuracy"] for row in rows],
        dtype=np.float64,
    )
    exact_recall = np.asarray(
        [row["assignment_formula_recall"] for row in rows],
        dtype=np.float64,
    )
    element = np.asarray(
        [row["assignment_element_hit_rate"] for row in rows],
        dtype=np.float64,
    )
    ion_exact = np.asarray(
        [row["assignment_formula_accuracy_ion_weighted"] for row in rows],
        dtype=np.float64,
    )
    # End-to-end formula recall dominates so an optimizer cannot improve its
    # apparent accuracy by dropping difficult peaks. Covered-peak accuracy and
    # trace elements still matter, while the already-near-perfect ion-weighted
    # score receives only a small tie-breaking weight.
    score = float(
        0.52 * exact_recall.mean()
        + 0.23 * exact.mean()
        + 0.10 * element.mean()
        + 0.05 * ion_exact.mean()
        + 0.10 * exact_recall.min()
    )
    return score, rows


def _formula_dict(value: object) -> dict[str, int]:
    parsed = _parse_object(value)
    if not isinstance(parsed, dict):
        return {}
    return {
        str(element): int(count)
        for element, count in parsed.items()
        if int(count) > 0
    }


def range_assignment_comparison(
    control: TrainingControl,
    assignments: pd.DataFrame,
    *,
    min_range_ions: int = 200,
) -> pd.DataFrame:
    top = assignments.loc[assignments["rank"] == 1].copy()
    peak_frame = control.peaks.copy()
    peak_frame["integrated_area"] = peak_frame.get(
        "integrated_area",
        pd.Series(np.zeros(peak_frame.shape[0])),
    ).fillna(0.0)
    fields = [
        "peak_id",
        "species_label",
        "isotopologue_label",
        "charge",
        "probability",
        "element_counts",
    ]
    peak_frame = peak_frame.merge(top[fields], on="peak_id", how="left")
    mz_sorted = np.sort(control.mz)
    rows: list[dict[str, object]] = []
    for range_index, reference in control.dataset.ranges.iterrows():
        lo = float(reference["mz_lo_da"])
        hi = float(reference["mz_hi_da"])
        ion_count = int(
            np.searchsorted(mz_sorted, hi, side="right")
            - np.searchsorted(mz_sorted, lo, side="left")
        )
        if ion_count < min_range_ions:
            continue
        candidates = peak_frame[
            (peak_frame["peak_mz_da"] >= lo) & (peak_frame["peak_mz_da"] <= hi)
        ].sort_values(["integrated_area", "peak_mz_da"], ascending=[False, True])
        expected = _formula_dict(reference["formula"])
        selected = candidates.iloc[0] if not candidates.empty else None
        predicted = _formula_dict(selected["element_counts"]) if selected is not None else {}
        rows.append(
            {
                "dataset": control.dataset.name,
                "range_index": int(range_index),
                "range_lo_da": lo,
                "range_hi_da": hi,
                "range_center_da": 0.5 * (lo + hi),
                "range_ions": ion_count,
                "expected_formula": json.dumps(expected, sort_keys=True),
                "covered": selected is not None,
                "exact": selected is not None and predicted == expected,
                "peak_id": selected["peak_id"] if selected is not None else None,
                "peak_mz_da": float(selected["peak_mz_da"]) if selected is not None else np.nan,
                "predicted_formula": json.dumps(predicted, sort_keys=True),
                "species_label": selected["species_label"] if selected is not None else None,
                "isotopologue_label": (
                    selected["isotopologue_label"] if selected is not None else None
                ),
                "charge": float(selected["charge"]) if selected is not None else np.nan,
                "probability": (
                    float(selected["probability"]) if selected is not None else np.nan
                ),
                "molecular_family_label": (
                    selected.get("molecular_family_label") if selected is not None else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _print_evaluation(
    label: str,
    score: float,
    rows: list[dict[str, Any]],
    parameters: dict[str, float],
) -> None:
    print(f"\n{label}: objective={score:.6f}")
    for row in rows:
        print(
            "  "
            f"{row['dataset']}: "
            f"exact={row['assignment_formula_accuracy']:.4f}, "
            f"recall={row['assignment_formula_recall']:.4f}, "
            f"element={row['assignment_element_hit_rate']:.4f}, "
            f"ion-exact={row['assignment_formula_accuracy_ion_weighted']:.4f}, "
            f"covered={row['n_reference_assignments_covered']}/"
            f"{row['n_reference_assignments']}"
        )
    print("  parameters=" + json.dumps(parameters, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=ROOT / "outputs" / "benchmark_assignment_v4",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(TRUSTED_TRAINING_DATASETS),
        help="comma-separated control names, or 'all' for every cached ranged control",
    )
    parser.add_argument(
        "--optimize",
        choices=("none", *PARAMETER_PRESETS),
        default="none",
    )
    parser.add_argument("--maxiter", type=int, default=8)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="fixed analysis setting (repeatable)",
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="NAME=V1,V2,...",
        help="evaluate a Cartesian parameter grid (repeatable)",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--diagnostics-csv", type=Path, default=None)
    args = parser.parse_args()

    if args.datasets == "all":
        names = {
            path.parent.parent.name
            for path in args.benchmark_dir.glob("*/custom/peaks.csv")
        }
        # Slugs are not necessarily reversible. Resolve through the registry.
        nominal_path = ROOT / "controls" / "nominal.yaml"
        nominal: dict[str, dict[str, float]] = {}
        if nominal_path.exists():
            import yaml

            nominal = yaml.safe_load(nominal_path.read_text(encoding="utf-8")) or {}
        names = {
            dataset.name
            for dataset in real_control_datasets(ROOT / "controls", nominal)
            if dataset.ranges is not None
            and (args.benchmark_dir / slugify(dataset.name) / "custom" / "peaks.csv").exists()
        }
    else:
        names = {name.strip() for name in args.datasets.split(",") if name.strip()}

    fixed: dict[str, float] = {}
    for item in args.set:
        name, separator, raw_value = item.partition("=")
        if not separator:
            parser.error(f"--set expects NAME=VALUE, got {item!r}")
        fixed[name] = float(raw_value)
    sweep_dimensions: list[tuple[str, list[float]]] = []
    for item in args.sweep:
        name, separator, raw_values = item.partition("=")
        if not separator or not raw_values:
            parser.error(f"--sweep expects NAME=V1,V2,..., got {item!r}")
        values = [float(value) for value in raw_values.split(",") if value.strip()]
        if not values:
            parser.error(f"--sweep has no values: {item!r}")
        sweep_dimensions.append((name, values))

    controls = load_controls(args.benchmark_dir, names)
    config = load_config(default_config_path())
    baseline_parameters = {
        name: float(config["analysis"][name])
        for name in (
            "molecular_family_hint_weight",
            "molecular_family_contradiction_penalty",
        )
    }
    baseline_parameters.update(fixed)
    started = time.perf_counter()
    baseline_score, baseline_rows = evaluate_parameters(
        controls,
        config,
        baseline_parameters,
    )
    _print_evaluation("baseline", baseline_score, baseline_rows, baseline_parameters)

    best_parameters = dict(baseline_parameters)
    best_score = baseline_score
    best_rows = baseline_rows
    if sweep_dimensions:
        sweep_names = [name for name, _ in sweep_dimensions]
        for values in itertools.product(*(values for _, values in sweep_dimensions)):
            trial = dict(fixed)
            trial.update(
                {name: value for name, value in zip(sweep_names, values, strict=True)}
            )
            score, rows = evaluate_parameters(controls, config, trial)
            _print_evaluation("sweep", score, rows, trial)
            if score > best_score + 1.0e-12:
                best_score = score
                best_parameters = trial
                best_rows = rows
    if args.optimize != "none":
        dimensions = PARAMETER_PRESETS[args.optimize]
        parameter_names = [name for name, _, _ in dimensions]
        bounds = [(low, high) for _, low, high in dimensions]
        evaluations = 0

        def objective(values: np.ndarray) -> float:
            nonlocal evaluations, best_parameters, best_score, best_rows
            trial = dict(fixed)
            trial.update(
                {name: float(value) for name, value in zip(parameter_names, values, strict=True)}
            )
            score, rows = evaluate_parameters(controls, config, trial)
            evaluations += 1
            if score > best_score + 1.0e-12:
                best_score = score
                best_parameters = trial
                best_rows = rows
                _print_evaluation(f"best after {evaluations} trials", score, rows, trial)
            return -score

        result = differential_evolution(
            objective,
            bounds,
            seed=args.seed,
            maxiter=args.maxiter,
            popsize=args.popsize,
            polish=False,
            updating="immediate",
            workers=1,
            atol=1.0e-6,
            tol=1.0e-4,
        )
        result_parameters = dict(fixed)
        result_parameters.update(
            {
                name: float(value)
                for name, value in zip(parameter_names, result.x, strict=True)
            }
        )
        result_score, result_rows = evaluate_parameters(controls, config, result_parameters)
        if result_score > best_score:
            best_score = result_score
            best_parameters = result_parameters
            best_rows = result_rows
        print(
            f"\noptimizer: {result.message}; evaluations={result.nfev}; "
            f"elapsed={time.perf_counter() - started:.1f}s"
        )

    _print_evaluation("best", best_score, best_rows, best_parameters)
    payload = {
        "benchmark_dir": str(args.benchmark_dir),
        "datasets": [control.dataset.name for control in controls],
        "objective": best_score,
        "parameters": best_parameters,
        "metrics": best_rows,
        "elapsed_s": time.perf_counter() - started,
    }
    output_path = args.output_json or args.benchmark_dir / "assignment_tuning.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {output_path}")
    diagnostics: list[pd.DataFrame] = []
    best_config = copy.deepcopy(config)
    best_config["analysis"].update(best_parameters)
    for control in controls:
        assignments, _ = assign_peaks(
            control.peaks,
            config=best_config,
            library_cache=control.library_cache,
        )
        diagnostics.append(range_assignment_comparison(control, assignments))
    diagnostics_path = (
        args.diagnostics_csv
        or output_path.with_name(f"{output_path.stem}_diagnostics.csv")
    )
    pd.concat(diagnostics, ignore_index=True).to_csv(diagnostics_path, index=False)
    print(f"wrote {diagnostics_path}")


if __name__ == "__main__":
    main()
