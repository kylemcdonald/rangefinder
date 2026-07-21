from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_spectrum(
    centers: np.ndarray,
    counts: np.ndarray,
    peaks: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    max_mz: float,
) -> None:
    mask = centers <= max_mz
    plt.figure(figsize=(12, 5))
    plt.plot(centers[mask], counts[mask], color="#1f3b73", linewidth=0.9)
    if not peaks.empty:
        subset = peaks[peaks["peak_mz_da"] <= max_mz]
        plt.scatter(
            subset["peak_mz_da"],
            subset["peak_height"],
            s=10,
            color="#b22222",
            zorder=3,
        )
    plt.yscale("log")
    plt.xlabel("m/z (Da)")
    plt.ylabel("Counts")
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_zoom_regions(
    centers: np.ndarray,
    counts: np.ndarray,
    peaks: pd.DataFrame,
    assignments: pd.DataFrame,
    output_dir: Path,
    *,
    window_da: float,
    max_regions: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    top = assignments[assignments["rank"] == 1].copy()
    if top.empty:
        return []
    top = top.sort_values(["confidence", "probability"], ascending=[True, False])
    focus = top.head(max_regions)
    paths: list[Path] = []
    for _, row in focus.iterrows():
        peak_row = peaks.loc[peaks["peak_id"] == row["peak_id"]]
        if peak_row.empty:
            continue
        peak_mz = float(peak_row.iloc[0]["peak_mz_da"])
        mask = (centers >= peak_mz - window_da) & (centers <= peak_mz + window_da)
        if not mask.any():
            continue
        plt.figure(figsize=(8, 4))
        plt.plot(centers[mask], counts[mask], color="#314e52", linewidth=1.0)
        plt.axvline(peak_mz, color="#d1495b", linestyle="--", linewidth=0.9)
        plt.xlabel("m/z (Da)")
        plt.ylabel("Counts")
        plt.title(f"{row['peak_id']} | {row['species_label']} | {row['confidence']}")
        plt.tight_layout()
        output_path = output_dir / f"{row['peak_id']}.png"
        plt.savefig(output_path, dpi=180)
        plt.close()
        paths.append(output_path)
    return paths
