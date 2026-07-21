from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from ifes_apt_tc_data_modeling.pos.pos_reader import ReadPosFileFormat

from rangefinder.utils.paths import slugify


@dataclass(frozen=True)
class PosSampleMetadata:
    path: Path
    sample_name: str
    sample_slug: str
    event_count: int
    file_size_bytes: int
    verified_big_endian_float32: bool
    verified_columns: tuple[str, str, str, str]


@dataclass
class PosSampleData:
    metadata: PosSampleMetadata
    x_nm: np.ndarray
    y_nm: np.ndarray
    z_nm: np.ndarray
    m_over_z_da: np.ndarray

    def position_summary(self) -> pd.DataFrame:
        summary = []
        for axis_name, values in [
            ("x_nm", self.x_nm),
            ("y_nm", self.y_nm),
            ("z_nm", self.z_nm),
            ("m_over_z_da", self.m_over_z_da),
        ]:
            finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
            summary.append(
                {
                    "axis": axis_name,
                    "min": float(finite.min()) if finite.size else np.nan,
                    "max": float(finite.max()) if finite.size else np.nan,
                    "mean": float(finite.mean()) if finite.size else np.nan,
                    "std": float(finite.std()) if finite.size else np.nan,
                }
            )
        return pd.DataFrame(summary)


def list_pos_samples(data_dir: Path) -> list[PosSampleMetadata]:
    samples: list[PosSampleMetadata] = []
    for path in sorted(data_dir.glob("*.POS")):
        sample_name = path.stem
        file_size_bytes = path.stat().st_size
        event_count = file_size_bytes // 16
        samples.append(
            PosSampleMetadata(
                path=path,
                sample_name=sample_name,
                sample_slug=slugify(sample_name),
                event_count=event_count,
                file_size_bytes=file_size_bytes,
                verified_big_endian_float32=(file_size_bytes % 16 == 0),
                verified_columns=("x_nm", "y_nm", "z_nm", "m_over_z_da"),
            )
        )
    return samples


def load_pos_sample(metadata: PosSampleMetadata) -> PosSampleData:
    cache_dir = metadata.path.parent.parent / ".cache" / "pos_samples"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{metadata.sample_slug}.npz"
    source_mtime_ns = metadata.path.stat().st_mtime_ns
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                cached_mtime_ns = int(cached["source_mtime_ns"][0])
                cached_size_bytes = int(cached["source_size_bytes"][0])
                if cached_mtime_ns == source_mtime_ns and cached_size_bytes == metadata.file_size_bytes:
                    return PosSampleData(
                        metadata=metadata,
                        x_nm=np.asarray(cached["x_nm"], dtype=np.float32),
                        y_nm=np.asarray(cached["y_nm"], dtype=np.float32),
                        z_nm=np.asarray(cached["z_nm"], dtype=np.float32),
                        m_over_z_da=np.asarray(cached["m_over_z_da"], dtype=np.float32),
                    )
        except Exception:
            pass
    reader = ReadPosFileFormat(str(metadata.path))
    xyz = reader.get_reconstructed_positions()
    m_over_z = reader.get_mass_to_charge_state_ratio()
    if xyz is None or m_over_z is None:
        raise ValueError(f"Unable to parse POS file: {metadata.path}")
    xyz_mag = np.asarray(xyz.magnitude, dtype=np.float32)
    m_over_z_mag = np.asarray(m_over_z.magnitude, dtype=np.float32)
    sample_data = PosSampleData(
        metadata=metadata,
        x_nm=xyz_mag[:, 0],
        y_nm=xyz_mag[:, 1],
        z_nm=xyz_mag[:, 2],
        m_over_z_da=m_over_z_mag,
    )
    np.savez(
        cache_path,
        source_mtime_ns=np.asarray([source_mtime_ns], dtype=np.int64),
        source_size_bytes=np.asarray([metadata.file_size_bytes], dtype=np.int64),
        x_nm=sample_data.x_nm,
        y_nm=sample_data.y_nm,
        z_nm=sample_data.z_nm,
        m_over_z_da=sample_data.m_over_z_da,
    )
    return sample_data


def verify_pos_structure(path: Path) -> dict[str, object]:
    file_size_bytes = path.stat().st_size
    raw = path.read_bytes()[:64]
    big_endian_preview = np.frombuffer(raw, dtype=">f4").reshape(-1, 4)
    little_endian_preview = np.frombuffer(raw, dtype="<f4").reshape(-1, 4)
    return {
        "file": str(path),
        "file_size_bytes": file_size_bytes,
        "mod_16": file_size_bytes % 16,
        "preview_big_endian": big_endian_preview.tolist(),
        "preview_little_endian": little_endian_preview.tolist(),
        "verified": {
            "record_size_bytes": 16,
            "endianness": "big-endian float32",
            "column_order": ["x_nm", "y_nm", "z_nm", "m_over_z_da"],
        },
        "inferred": {
            "fourth_column_interpretation": "mass-to-charge ratio in daltons per charge",
        },
    }
