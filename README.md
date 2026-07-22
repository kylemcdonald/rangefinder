# Rangefinder

[![CI](https://github.com/kylemcdonald/rangefinder/actions/workflows/ci.yml/badge.svg)](https://github.com/kylemcdonald/rangefinder/actions/workflows/ci.yml)

Automated **ranging**, species identification, and composition for atom probe
tomography (APT) mass spectra. Rangefinder goes from an APT event list
(`.pos`/`.epos`, represented internally by reconstructed positions and *m/z*) to
ranged peaks, identified positive ionic species, and isotope-resolved composition
— with **no user-supplied ranges, no candidate element list, no interactive
tuning, and no manual curation**.

It is deterministic (operator variance is exactly zero), pure Python
(NumPy/SciPy/pandas), and ships its own benchmark harness against the open-source
APT ecosystem.

## Scope

Rangefinder is an **APT-specific end-to-end system**, not a general-purpose mass
spectrometry or spectroscopy toolkit. Its physical and chemical model assumes:

- event-level, one-dimensional time-of-flight *m/z* data;
- positive atomic and small molecular ions typical of field evaporation;
- APT-like peak widths and one-sided thermal tails;
- natural-isotope family patterns, APT charge-state priors, and elemental atomic
  percent as the final quantity; and
- reconstructed `x_nm`, `y_nm`, and `z_nm` coordinates for the public pipeline
  data model and optional spatial analyses.

The histogramming, local peak detection, centroid refinement, and area-integration
code is potentially reusable for other one-dimensional TOF mass spectra after
adaptation and domain-specific validation. The MassBank and ToF-SIMS studies in
this repository test **that detector only**. They do not validate Rangefinder's
species assignment or composition outside APT; the benchmark adapters convert
profile/centroid data to pseudo-events and disable APT family-assignment features.

Rangefinder does not currently model negative-ion assignment, adduct and fragment
chemistry, tandem MS, chromatography or ion mobility, arbitrary profile/centroid
inputs, instrument-pluggable resolution models, or non-mass-spectrometry axes.
For those use cases, use a domain-specific toolkit or extract and validate only
the low-level detector.

## Range and overlap auditing

Every detected peak now carries two independent range estimates:

- the production fixed/FWHM or crowded-window mixture-fit area; and
- a clean-room equal-error range estimate with asymmetric boundaries, estimated
  purity, recovery, background, missed signal, counting uncertainty, and the
  fixed-versus-adaptive discrepancy.

Equal-error ranging was inspired by the public
[Atom-Probe-Toolbox](https://github.com/peterfelfer/Atom-Probe-Toolbox), but no
GPL source was copied. It remains a **shadow diagnostic**, because the bundled
ablation found a large synthetic-data improvement but regressions on seven of
ten real composition controls.

Composition uses a guarded natural-isotope-envelope deconvolution only for
identifiable connected overlap groups with high-confidence species anchors,
full-rank and conditioned envelope matrices, adequate observed coverage,
optimizer convergence, and a residual pass. Rejected groups retain their
assignment probabilities unchanged. Output tables preserve the uncorrected
`atomic_percent_assignment` beside the promoted `atomic_percent_weighted`, and
`diagnostics.json` records every accepted or rejected component.

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
python scripts/run_quantification_ablation.py   # range/deconvolution promotion test
python scripts/build_paper_tables.py            # regenerate paper/tables/*.tex
tectonic paper/rangefinder.tex
```

Every number in the paper regenerates from these commands; table numbers are
never hand-edited.

## License

The software is released under the [MIT License](LICENSE). The reference data
and range files under `controls/` remain CC-BY-4.0 as documented in
[`controls/README.md`](controls/README.md).
