# APT control / reference datasets

Public atom probe tomography (APT) datasets of known materials, used as
controls to validate the mass-spectrum analysis pipeline. All files below were
downloaded on 2026-07-20 from a single Zenodo record curated by the
FAIRmat/NFDI atom-probe community:

- **Zenodo record:** https://zenodo.org/records/7979668
- **DOI:** 10.5281/zenodo.7979668
- **License:** CC-BY-4.0 (applies to every file in this directory)
- **Record title:** "German NFDI, FAIRmat-NFDI, NOMAD, NOMAD OASIS, pynxtools,
  example datasets for atom probe microscopy and electron microscopy"
- **Creators (record):** M. Kühbach et al., incl. Jesse Smith (Si), Andrew
  Breen (W), Peter Felfer / Martina Heller (Ck10 steel)

File formats (all big-endian):

- `.pos` — 16-byte records: 4 x float32 = x, y, z (nm), m/z (Da)
- `.epos` — 44-byte records: 9 x float32 (x, y, z, m/z, TOF, V_dc, V_pulse,
  detector x, detector y) + 2 x uint32 (pulses since last event, ions per pulse)

---

## 1. Silicon (pure Si) — APAV example dataset

| File | Size | Ions |
|---|---|---|
| `control_Si_apav_usa_denton_smith.pos` | 15,123,376 B (15.1 MB) | 945,211 |
| `control_Si_apav_usa_denton_smith.epos` | 41,589,284 B (41.6 MB) | 945,211 |
| `control_Si_apav_usa_denton_smith.rrng` | 1.4 kB | range file shipped with the dataset |

- **Source:** `usa_denton_smith_apav_si.zip` (55.7 MB) from Zenodo 7979668,
  https://zenodo.org/api/records/7979668/files/usa_denton_smith_apav_si.zip/content
  Original members `Si.pos`, `Si.epos`, `Si.RRNG`.
- **Provenance:** Example/test dataset of the APAV analysis package by
  Jesse Smith (University of North Texas, Denton); APAV: https://gitlab.com/jesseds/apav
- **Material / nominal composition:** Silicon microtip — nominally pure Si.
  The supplied range file additionally ranges minor C, O, Cr, Cu species.
- **Verification:** `.pos` size divisible by 16: yes; `.epos` size divisible by
  44: yes; both contain the same 945,211 ions. m/z min 0.00, max 378.3,
  median 14.04 Da, 99.83 % of values in [0, 300]. Reconstructed volume approx
  22 x 21 x 15 nm. Spectrum dominated by Si2+ at 14/14.5/15 Da (~73 % of ions
  in the 14 Da bin) and Si+ at 28/29/30 Da — consistent with pure Si.

## 2. Tungsten (pure W, measured at 18 K)

| File | Size | Ions |
|---|---|---|
| `control_W_18K_breen_kuehbach_R18_53222.epos` | 338,373,420 B (338.4 MB) | 7,690,305 |

- **Source:** `APM.LEAP.Datasets.2.zip` (215.2 MB) from Zenodo 7979668,
  https://zenodo.org/api/records/7979668/files/APM.LEAP.Datasets.2.zip/content
  Original member `R18_53222_W_18K-v01.epos`.
- **Provenance:** Per the Zenodo record description: "shared with Markus
  Kühbach by Andrew Breen during their time at the Max-Planck-Institut für
  Eisenforschung GmbH". Tungsten specimen measured at 18 K on a LEAP
  instrument. Also used as a reference dataset in the paraprobe /
  ifes_apt_tc_data_modeling ecosystem.
- **Material / nominal composition:** Pure tungsten (nominally ~100 at.% W).
- **Verification:** size divisible by 44: yes. m/z min 0.00, max 6334 (sparse
  noise tail, typical for raw epos), median 61.12 Da, 98.0 % of values in
  [0, 300]. Spectrum dominated by W3+ at 60.7/61/61.3/62 Da and W4+ at
  45.5/46/46.5 Da with the natural W isotope pattern (182/183/184/186) —
  consistent with pure W. Non-zero TOF / V_dc / detector columns present.
- **Caveat:** at 338 MB this file slightly exceeds the preferred 300 MB
  per-file size (the compressed download was 215 MB). Kept because a pure-W
  spectrum is the classic APT reference.

## 3. Ck10 low-carbon steel (Felfer, FAU Erlangen)

| File | Size | Ions |
|---|---|---|
| `control_Ck10_steel_felfer_R56_01769.pos` | 88,405,776 B (88.4 MB) | 5,525,361 |
| `control_Ck10_steel_felfer_R56_01769.rng.fig.txt` | 1.0 kB | range list shipped with the dataset |

- **Source:** `ger_erlangen_felfer_ck10.zip` (82.6 MB) from Zenodo 7979668,
  https://zenodo.org/api/records/7979668/files/ger_erlangen_felfer_ck10.zip/content
  Original members `R56_01769-v01.pos`, `R56_01769.rng.fig.txt`.
- **Provenance:** "Ck10 for fundamentals" test dataset of Peter Felfer's
  Atom-Probe-Toolbox:
  https://github.com/peterfelfer/Atom-Probe-Toolbox/tree/master/test%20data/Ck%2010%20steel%20for%20fundamentals
- **Material / nominal composition:** Ck10 (DIN; today C10E / 1.1121, EN
  10084) plain low-carbon steel. Nominal: 0.07-0.13 wt% C, 0.30-0.60 wt% Mn,
  Si <= 0.40 wt%, P/S <= 0.045 wt%, balance Fe. The shipped range list
  covers H, C, O, Al, Si, P, Cr, Mn, Fe, Cu, Ga (Ga is from FIB specimen
  preparation, not the alloy).
- **Verification:** size divisible by 16: yes. m/z min 0.52, max 421.5,
  median 27.97 Da, 99.66 % of values in [0, 300]. Spectrum dominated by Fe2+
  (54/56/57 Da bins, ~86 % of ions at 28 Da) with a clear C2+ peak at 6 Da —
  consistent with a low-carbon steel.

---

## Verification method

For each file: checked byte-size divisibility (16 for `.pos`, 44 for `.epos`),
then memory-mapped the file with numpy dtype `'>f4'` and inspected column 4
(m/z) over all records, plus x/y/z ranges and a 1-Da-binned histogram of the
mass spectrum to confirm the dominant charge-state peaks match the documented
material.

## Notes / caveats

- Total downloaded: ~354 MB (compressed zips); ~483 MB on disk.
- The Zenodo record warns that the `.rng`/`.rrng` files inside
  `APM.LEAP.Datasets.1.zip` are format examples not matched to their datasets;
  that caveat does NOT apply to the files kept here — the Si `.RRNG` and the
  Ck10 range list were distributed with their respective datasets (APAV docs /
  Atom-Probe-Toolbox). Still, treat range files as guidance, not ground truth.
- The two Si files (.pos and .epos) are the same measurement in two formats
  (identical ion count), i.e. this directory holds 3 independent control
  specimens, not 4.
- Exact impurity levels of the Si and W specimens are not certified; "pure"
  means nominally single-element specimens as documented by the data authors.
  Known nominal composition is qualitative (dominant element + expected minor
  species), which is the intended use for pipeline validation.

## Additional controls (added 2026-07-20)

**4. ODS ferritic steel (Portland State University, Wang)**
- `control_ODSsteel_wang_R31_06365.pos` (77.9 MB, 4,868,202 ions) + `.rrng`
- Source: Zenodo record 7979668 (`usa_portland_wang.zip`), DOI 10.5281/zenodo.7979668, CC-BY-4.0.
- Material: oxide-dispersion-strengthened ferritic steel; range file covers Fe, Cr, Y, Ti, O, Mn, Si, V (14YWT/MA957 class). No certified composition shipped.
- Verified: size divisible by 16; m/z median 27.97 Da (Fe2+ region); all sampled m/z in [0, 300].

## Additional controls in `extra/` (added 2026-07-21, all CC-BY-4.0)

Downloaded by `scripts/download_benchmark_controls.sh`; Zenodo API metadata
snapshots in `tmp/downloads/zenodo_<record>.json`.

- **H/D-charged steel** (Zenodo 5534859): three smallest `.POS` files
  (2–3.7 MB; fast smoke tests) + the shared expert `range file.RRNG`
  (copied per stem for automatic association). Fe-matrix specimens with
  C/Si/O/Mn/Cr minors; the range file is the expert answer key.
- **FeO wüstite** (Zenodo 10677562, `R5076_69145-v01.epos`, 217 MB,
  LEAP-5000 XS): exact 1:1 Fe:O stoichiometry target. APT physically
  undercounts O in oxides — a bias shared by every analysis method.
- **Pure Li** (Zenodo 14848236, `R5076_68722.epos`, 286 MB): single-element
  control with a strong 7Li/6Li isotope signature and hydride satellites.
- **(V,Al)N nitride** (Zenodo 7788883, `VAlN_film_plan-view_800C`, 449 MB
  `.epos` + expert `.rrng`; Sälker et al.): transition-metal nitride film,
  a chemistry class outside the metals/steels/oxide of the rest of the set.
  Composition truth is range-file-derived (elements N, Al, V, Ar, C, O, Ga).
  Added 2026-07-21. NOTE: Zenodo 5838237 "Pristine FeO" (R76_45587) was
  downloaded and REJECTED — its spectrum shows almost no oxygen and a large
  unranged peak at the 55-position, so a Fe:O=1:1 label would be false truth.
- **ToF-SIMS spectra** (Zenodo 15446699, two ~19 MB `.txt`,
  channel/m-z/intensity): real non-APT time-of-flight spectra used for
  foreign-format detector robustness (known inorganic ions in the
  negative-polarity TSB spectrum act as a reference line list).
- **MassBank 2024.11** (Zenodo 14221628, zip in `tmp/downloads/`): curated
  centroid peak lists for known compounds; rendered into semi-synthetic
  profile spectra by `scripts/run_massbank_benchmark.py`.

**5. Mo-Hf alloy (Montanuniversitaet Leoben, Leitner)**
- `control_MoHf_leitner_R21_08680.pos` (73.3 MB, 4,583,568 ions) + `.rrng`
- Source: Zenodo record 7979668 (`aut_leoben_leitner.zip`), DOI 10.5281/zenodo.7979668, CC-BY-4.0.
- Material: MHC-class molybdenum alloy; range file covers Mo, Hf, C, B, N, O, Zr, plus FIB Ga.
- Verified: size divisible by 16; m/z median 45.94 Da (Mo2+ region); all sampled m/z in [0, 300].
