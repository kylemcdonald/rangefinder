from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "sample"


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    outputs_dir: Path
    logs_dir: Path
    docs_dir: Path
    cache_dir: Path
    config_path: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        source_config = root / "src" / "rangefinder" / "config" / "defaults.yaml"
        config_path = source_config if source_config.exists() else root / "config" / "defaults.yaml"
        return cls(
            root=root,
            data_dir=root / "data",
            outputs_dir=root / "outputs",
            logs_dir=root / "logs",
            docs_dir=root / "docs",
            cache_dir=root / ".cache",
            config_path=config_path,
        )

    def sample_dir(self, sample_slug: str) -> Path:
        return self.outputs_dir / sample_slug

    def sample_method_dir(self, sample_slug: str, method_name: str) -> Path:
        return self.sample_dir(sample_slug) / method_name


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
