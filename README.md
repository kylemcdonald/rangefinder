# Rangefinder

[![CI](https://github.com/kylemcdonald/rangefinder/actions/workflows/ci.yml/badge.svg)](https://github.com/kylemcdonald/rangefinder/actions/workflows/ci.yml)

Automated **ranging**, species identification, and composition for time-of-flight
(atom probe tomography) mass spectra. Rangefinder goes from a raw event list
(`.pos`/`.epos`, or any array of *m/z* values) to ranged peaks, identified ionic
species, and isotope-resolved composition — with **no user-supplied ranges, no
candidate element list, no interactive tuning, and no manual curation**.

It is deterministic (operator variance is exactly zero), pure Python
(NumPy/SciPy/pandas), and ships its own benchmark harness against the open-source
APT ecosystem.

## Install

```bash
python -m pip install rangefinder
```

For development:

```bash
git clone https://github.com/kylemcdonald/rangefinder.git
cd rangefinder
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Quick start

```python
from pathlib import Path

import rangefinder
from rangefinder.io.pos_loader import PosSampleData, PosSampleMetadata
from rangefinder.benchmark.truth import load_pos_arrays

sample_path = Path("your_sample.pos")
x, y, z, mz = load_pos_arrays(sample_path)
meta = PosSampleMetadata(path=sample_path, sample_name="s", sample_slug="s",
                         event_count=len(mz), file_size_bytes=len(mz) * 16,
                         verified_big_endian_float32=True,
                         verified_columns=("x_nm", "y_nm", "z_nm", "m_over_z_da"))
sample = PosSampleData(metadata=meta, x_nm=x, y_nm=y, z_nm=z, m_over_z_da=mz)

config = rangefinder.load_default_config()
artifacts = rangefinder.run_custom_pipeline(sample, Path("out"), config)
print(artifacts.elemental_composition)
```

## Layout

- `src/rangefinder/` — the package.
  - `analysis/` — the Rangefinder pipeline (`custom_pipeline`) plus the comparator
    front-ends (naive, PyCCAPT, pyOpenMS) and shared assignment/composition stages.
  - `benchmark/` — ground-truth registry, metrics, and detection-ablation harness.
  - `io/`, `references/`, `validation/`, `utils/` — POS loading, isotope tables,
    the synthetic generator, config/paths helpers.
  - `bridge.py`, `bridge_runner.py` — run foreign-toolchain comparators
    (APAV, ms_deisotope, APyT) in a separate environment.
  - `config/defaults.yaml` — the bundled default pipeline configuration
    (`rangefinder.default_config_path()`).
- `scripts/` — the benchmark and paper-table generators.
- `paper/` — `rangefinder.tex` and its auto-generated `tables/`.
- `controls/` — public CC-BY reference datasets (fetch large files with
  `scripts/download_benchmark_controls.sh`; reproducible raw data and generated
  synthetic controls are git-ignored, while provenance and range files are
  tracked).

## Reproduce the benchmark and paper

```bash
scripts/download_benchmark_controls.sh          # fetch public controls
scripts/bootstrap_env.sh                        # install comparator environments
python scripts/run_benchmark.py                 # all methods x all datasets
python scripts/run_detection_ablation.py        # detection-stage ablation
python scripts/run_ablation.py                  # pipeline-stage ablation
python scripts/build_paper_tables.py            # regenerate paper/tables/*.tex
tectonic paper/rangefinder.tex
```

Every number in the paper regenerates from these commands; table numbers are
never hand-edited.

## License

The software is released under the [MIT License](LICENSE). The reference data
and range files under `controls/` remain CC-BY-4.0 as documented in
[`controls/README.md`](controls/README.md).
