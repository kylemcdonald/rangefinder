#!/usr/bin/env python3
"""Detector-only transfer check on MassBank-derived semi-synthetic spectra.

MassBank records carry curated, expert-annotated centroid peak lists for known
compounds (CC-BY). We render each peak list into a realistic profile spectrum
(Gaussian core + one-sided thermal tail + uniform background, matching the
APT synthesis model) at a chosen mass resolving power, then ask every
pipeline's detector to recover the known lines. This tests the detectors on
line patterns from an entirely different chemistry than APT — fragment ions
of organic molecules — with exact ground truth. It does not test Rangefinder's
species assignment or composition outside APT, and the rendering deliberately
retains the APT synthetic peak-shape model.

Usage: .venv/bin/python scripts/run_massbank_benchmark.py \
          [--records 150] [--mrp 800] [--methods custom,naive,pyccapt,pyopenms]
"""
from __future__ import annotations

import argparse
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

from rangefinder.benchmark.metrics import peak_metrics_vs_lines  # noqa: E402
from rangefinder.io.pos_loader import PosSampleData, PosSampleMetadata  # noqa: E402
from rangefinder.utils.config import load_config  # noqa: E402
from rangefinder.utils.paths import slugify  # noqa: E402

MASSBANK_DIR = ROOT / "tmp" / "massbank"
SEED = 20260721


def parse_massbank_record(path: Path) -> dict | None:
    accession = None
    peaks: list[tuple[float, float]] = []
    in_peaks = False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("ACCESSION:"):
            accession = line.split(":", 1)[1].strip()
        elif line.startswith("PK$PEAK:"):
            in_peaks = True
        elif in_peaks:
            tokens = line.strip().split()
            if len(tokens) < 3 or line.startswith("//"):
                in_peaks = False
                continue
            try:
                mz = float(tokens[0])
                rel = float(tokens[2])
            except ValueError:
                in_peaks = False
                continue
            peaks.append((mz, rel))
    if accession is None or not peaks:
        return None
    return {"accession": accession, "peaks": peaks}


def select_records(n_records: int, *, min_peaks: int, max_mz: float) -> list[dict]:
    records = []
    paths = sorted(MASSBANK_DIR.rglob("MSBNK-*.txt"))
    for path in paths:
        record = parse_massbank_record(path)
        if record is None:
            continue
        mzs = [mz for mz, _ in record["peaks"]]
        if len(record["peaks"]) < min_peaks or max(mzs) > max_mz or min(mzs) < 1.0:
            continue
        records.append(record)
    if not records:
        return []
    step = max(1, len(records) // n_records)
    return records[::step][:n_records]


def render_events(record: dict, *, mrp: float, total_ions: int, rng: np.random.Generator):
    """Render curated centroids into pseudo-events with the same peak-shape
    model used by the synthetic APT generator (normalized-TOF widths,
    one-sided exponential tail, uniform-in-t background)."""
    mzs = np.asarray([mz for mz, _ in record["peaks"]], dtype=np.float64)
    weights = np.asarray([max(rel, 1.0) for _, rel in record["peaks"]], dtype=np.float64)
    weights = weights / weights.sum()
    background_fraction = 0.02
    n_signal = int(total_ions * (1.0 - background_fraction))
    counts = rng.multinomial(n_signal, weights)
    chunks = []
    for mz0, n in zip(mzs, counts, strict=True):
        if n <= 0:
            continue
        t0 = np.sqrt(mz0)
        sigma_t = t0 / (2.0 * mrp) / 2.354820045
        t = t0 + rng.normal(0.0, sigma_t, size=int(n))
        tail = rng.random(int(n)) < 0.35
        t[tail] += rng.exponential(2.0 * sigma_t, size=int(tail.sum()))
        chunks.append(np.square(t))
    n_bg = total_ions - n_signal
    t_bg = rng.uniform(np.sqrt(0.8), np.sqrt(mzs.max() + 20.0), size=n_bg)
    chunks.append(np.square(t_bg))
    mz_events = np.concatenate(chunks).astype(np.float32)
    n = mz_events.size
    # PosSampleData currently requires APT coordinates. These placeholders are
    # ignored because spatial analysis is disabled; only detected m/z is scored.
    radius = 30.0 * np.sqrt(rng.random(n))
    angle = rng.uniform(0, 2 * np.pi, n)
    return (
        (radius * np.cos(angle)).astype(np.float32),
        (radius * np.sin(angle)).astype(np.float32),
        rng.uniform(0, 100, n).astype(np.float32),
        mz_events,
    )


def truth_lines(record: dict, *, mrp: float) -> pd.DataFrame:
    frame = pd.DataFrame(
        [{"mz_da": mz, "weight": max(rel, 1.0)} for mz, rel in record["peaks"]]
    ).sort_values("mz_da")
    frame["weight"] = frame["weight"] / frame["weight"].sum()
    merged: list[dict[str, float]] = []
    for _, row in frame.iterrows():
        fwhm = row["mz_da"] / mrp
        if merged and abs(row["mz_da"] - merged[-1]["mz_da"]) <= max(0.7 * fwhm, 0.02):
            total = merged[-1]["weight"] + row["weight"]
            merged[-1]["mz_da"] = (
                merged[-1]["mz_da"] * merged[-1]["weight"] + row["mz_da"] * row["weight"]
            ) / total
            merged[-1]["weight"] = total
        else:
            merged.append({"mz_da": float(row["mz_da"]), "weight": float(row["weight"])})
    return pd.DataFrame(merged)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=150)
    parser.add_argument("--min-peaks", type=int, default=12)
    parser.add_argument("--max-mz", type=float, default=470.0)
    parser.add_argument("--mrp", type=float, default=800.0)
    parser.add_argument("--total-ions", type=int, default=400_000)
    parser.add_argument("--methods", type=str, default="custom,naive,pyccapt,pyopenms")
    args = parser.parse_args()

    from rangefinder.analysis.custom_pipeline import run_custom_pipeline
    from rangefinder.analysis.naive_pipeline import run_naive_pipeline
    from rangefinder.analysis.pyccapt_pipeline import run_pyccapt_pipeline
    from rangefinder.analysis.pyopenms_pipeline import run_pyopenms_pipeline

    runners = {
        "custom": run_custom_pipeline,
        "naive": run_naive_pipeline,
        "pyccapt": run_pyccapt_pipeline,
        "pyopenms": run_pyopenms_pipeline,
    }
    methods = [m.strip() for m in args.methods.split(",") if m.strip() in runners]

    config = load_config(default_config_path())
    config = copy.deepcopy(config)
    config["analysis"]["max_mz"] = float(args.max_mz + 30.0)
    config["analysis"]["zoom_max_mz"] = min(120.0, float(args.max_mz))
    config["analysis"]["segmentation_enabled"] = False
    # Composition machinery is meaningless for organic fragments; keep the
    # run cheap and measure detection only.
    config["analysis"]["custom_family_hint_enabled"] = False
    config["analysis"]["element_pruning_enabled"] = False
    config["analysis"]["max_zoom_regions_per_method"] = 0

    records = select_records(args.records, min_peaks=args.min_peaks, max_mz=args.max_mz)
    print(f"{len(records)} MassBank records selected")
    rng = np.random.default_rng(SEED)
    output_root = ROOT / "outputs" / "benchmark_massbank"
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for idx, record in enumerate(records):
        x, y, z, mz = render_events(record, mrp=args.mrp, total_ions=args.total_ions, rng=rng)
        lines = truth_lines(record, mrp=args.mrp)
        slug = slugify(record["accession"])
        metadata = PosSampleMetadata(
            path=Path(f"/synthetic/{slug}"), sample_name=record["accession"], sample_slug=slug,
            event_count=int(mz.size), file_size_bytes=int(mz.size) * 16,
            verified_big_endian_float32=True,
            verified_columns=("x_nm", "y_nm", "z_nm", "m_over_z_da"),
        )
        sample = PosSampleData(metadata=metadata, x_nm=x, y_nm=y, z_nm=z, m_over_z_da=mz)
        for method in methods:
            out_dir = output_root / "runs" / slug / method
            started = time.perf_counter()
            try:
                if method == "custom":
                    artifacts = runners[method](sample, out_dir, config, enable_segmentation=False)
                else:
                    artifacts = runners[method](sample, out_dir, config)
                detected = (
                    artifacts.peaks["peak_mz_da"].to_numpy(dtype=float)
                    if not artifacts.peaks.empty
                    else np.asarray([])
                )
                entry = {
                    "accession": record["accession"],
                    "method": method,
                    "runtime_s": round(time.perf_counter() - started, 2),
                }
                entry |= peak_metrics_vs_lines(detected, lines, mass_resolving_power=args.mrp)
            except Exception as exc:  # noqa: BLE001
                entry = {"accession": record["accession"], "method": method, "error": repr(exc)}
            results.append(entry)
        if (idx + 1) % 10 == 0:
            print(f"[{idx + 1}/{len(records)}] done", flush=True)
            pd.DataFrame(results).to_csv(output_root / "massbank_results.csv", index=False)
    frame = pd.DataFrame(results)
    frame.to_csv(output_root / "massbank_results.csv", index=False)
    ok = frame[frame.get("error").isna()] if "error" in frame.columns else frame
    summary = (
        ok.groupby("method")[
            ["line_recall", "line_recall_weighted", "peak_precision", "n_false_peaks",
             "mass_accuracy_median_abs_ppm", "runtime_s"]
        ]
        .mean(numeric_only=True)
        .round(4)
    )
    summary.to_csv(output_root / "massbank_summary.csv")
    (output_root / "massbank_summary.json").write_text(
        json.dumps(
            {
                "n_records": len(records),
                "mrp": args.mrp,
                "total_ions": args.total_ions,
                "per_method": json.loads(summary.to_json(orient="index")),
            },
            indent=2,
        )
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
