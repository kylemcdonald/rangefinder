from __future__ import annotations

from pathlib import Path

from rangefinder.utils.paths import ProjectPaths


def test_project_paths_resolves_src_layout_config() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = ProjectPaths.from_root(root)
    assert paths.config_path == root / "src" / "rangefinder" / "config" / "defaults.yaml"
    assert paths.config_path.exists()
