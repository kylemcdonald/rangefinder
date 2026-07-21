#!/usr/bin/env python3
"""Generate the Rangefinder paper's result tables from benchmark outputs.

Reads outputs/benchmark/benchmark_results.json, outputs/benchmark_ablation/,
outputs/benchmark_massbank/, outputs/benchmark_tofsims/ and writes
paper/tables/*.tex. Never edits numbers by hand: rerun this script after any
benchmark change.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "paper" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = ["custom", "apav", "pyccapt", "pyopenms", "ms_deisotope", "apyt", "naive"]
METHOD_TEX = {
    "custom": r"\rf{}",
    "apav": "APAV",
    "pyccapt": "PyCCAPT",
    "pyopenms": "pyOpenMS",
    "ms_deisotope": r"ms\_deisotope",
    "apyt": "APyT",
    "naive": "na\\\"ive",
}
DATASET_TEX = {
    "synthetic_al_mg_si": "Syn.\\ Al--Mg--Si",
    "synthetic_si_o": "Syn.\\ Si--O oxide",
    "synthetic_zn_cu_al": "Syn.\\ Zn--Cu--Al",
    "synthetic_zn_layer_on_al": "Syn.\\ Zn/Al layered",
    "control_Si_apav_usa_denton_smith": "Si (+Ni silicide)",
    "control_W_18K_breen_kuehbach_R18_53222": "Pure W (18\\,K)",
    "control_Ck10_steel_felfer_R56_01769": "Ck10 steel",
    "control_ODSsteel_wang_R31_06365": "ODS steel",
    "control_MoHf_leitner_R21_08680": "Mo--Hf alloy",
    "vain_planview_800c_7788883": "(V,Al)N nitride",
    "feoxide_10677562_R5076_69145-v01": "FeO w\\\"ustite",
    "li_14848236_R5076_68722": "Pure Li",
    "steelHD_5534859_70a59eff-003c-4337-832a-604c260dc623": "H/D steel A",
    "steelHD_5534859_8fe936a9-c8bf-417a-8006-bff810375bc7": "H/D steel B",
    "steelHD_5534859_aa137df8-ce0e-481e-93e6-a22cb5a34882": "H/D steel C",
}
TRUTH_TEX = {"synthetic": "exact", "nominal": "nominal", "range-file": "range file"}


def fmt(value, digits=2, best=False):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    text = f"{value:.{digits}f}"
    return rf"\textbf{{{text}}}" if best else text


def load_benchmark() -> pd.DataFrame:
    path = ROOT / "outputs" / "benchmark" / "benchmark_results.json"
    frame = pd.DataFrame(json.loads(path.read_text()))
    return frame


def composition_table(frame: pd.DataFrame) -> None:
    has_truth = frame[frame["elemental_l1_at_pct"].notna()] if "elemental_l1_at_pct" in frame.columns else pd.DataFrame()
    datasets = [d for d in DATASET_TEX if d in set(has_truth["dataset"])]
    header_methods = [m for m in METHOD_ORDER if m in set(frame["method"])]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Composition recovery: L1 distance to truth (at.\%, lower is",
        r"better; bold = best per dataset) with fabricated signal --- at.\%",
        r"assigned to elements absent from the truth --- in parentheses.",
        r"Truth sources: exact (synthetic), nominal composition, or expert",
        r"range file applied to the raw spectrum. Detector-only comparators",
        r"share \rf{}'s assignment stage (Sec.~\ref{sec:benchmark}); APAV and",
        r"APyT use their own quantification.}",
        r"\label{tab:composition}",
        r"\small",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{ll" + "r" * len(header_methods) + "}",
        r"\toprule",
        "Dataset & Truth & " + " & ".join(METHOD_TEX[m] for m in header_methods) + r" \\",
        r"\midrule",
    ]
    for dataset in datasets:
        sub = has_truth[has_truth["dataset"] == dataset].set_index("method")
        l1 = {m: sub.loc[m, "elemental_l1_at_pct"] if m in sub.index else np.nan for m in header_methods}
        fab = {m: sub.loc[m, "fabricated_at_pct"] if m in sub.index else np.nan for m in header_methods}
        finite = {m: v for m, v in l1.items() if np.isfinite(v)}
        best = min(finite, key=finite.get) if finite else None
        truth_source = sub["truth_source"].dropna().iloc[0] if "truth_source" in sub.columns and not sub["truth_source"].dropna().empty else ""
        cells = []
        for m in header_methods:
            if not np.isfinite(l1[m]):
                cells.append("--")
                continue
            cell = fmt(l1[m], 2, best=(m == best))
            if np.isfinite(fab.get(m, np.nan)):
                cell += f" ({fab[m]:.2f})"
            cells.append(cell)
        lines.append(
            f"{DATASET_TEX[dataset]} & {TRUTH_TEX.get(truth_source, truth_source)} & "
            + " & ".join(cells) + r" \\"
        )
    # Mean row over datasets where every method has a value
    complete = has_truth.pivot_table(index="dataset", columns="method", values="elemental_l1_at_pct")
    complete = complete.dropna(axis=0, how="any")
    if not complete.empty:
        lines.append(r"\midrule")
        means = complete.mean()
        best = means.idxmin()
        lines.append(
            f"Mean (n={complete.shape[0]} complete) & & "
            + " & ".join(fmt(means.get(m, np.nan), 2, best=(m == best)) for m in header_methods)
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    (TABLES / "composition_table.tex").write_text("\n".join(lines) + "\n")


def detection_table(frame: pd.DataFrame) -> None:
    header_methods = [m for m in METHOD_ORDER if m in set(frame["method"])]
    synthetic = frame[frame["line_recall_weighted"].notna()] if "line_recall_weighted" in frame.columns else pd.DataFrame()
    ranged = frame[frame["range_recall_ion_weighted"].notna()] if "range_recall_ion_weighted" in frame.columns else pd.DataFrame()
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Peak detection. Synthetic specimens: intensity-weighted",
        r"recall of known emitted lines, precision of detected peaks, and",
        r"median absolute mass error of matched lines. Public datasets with",
        r"expert range files: ion-weighted recall of reference ranges.",
        r"Bold = best per column.}",
        r"\label{tab:detection}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Synthetic (exact lines)} & Range files \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}",
        r"Method & Recall$_w$ & Precision & $|\Delta m|$ (ppm) & Recall$_{ion}$ \\",
        r"\midrule",
    ]
    stats = {}
    for m in header_methods:
        stats[m] = {
            "recall": synthetic[synthetic["method"] == m]["line_recall_weighted"].mean() if not synthetic.empty else np.nan,
            "precision": synthetic[synthetic["method"] == m]["peak_precision"].mean() if not synthetic.empty else np.nan,
            "ppm": synthetic[synthetic["method"] == m]["mass_accuracy_median_abs_ppm"].mean() if not synthetic.empty else np.nan,
            "range_recall": ranged[ranged["method"] == m]["range_recall_ion_weighted"].mean() if not ranged.empty else np.nan,
        }
    best = {
        "recall": max((m for m in header_methods if np.isfinite(stats[m]["recall"])), key=lambda m: stats[m]["recall"], default=None),
        "precision": max((m for m in header_methods if np.isfinite(stats[m]["precision"])), key=lambda m: stats[m]["precision"], default=None),
        "ppm": min((m for m in header_methods if np.isfinite(stats[m]["ppm"])), key=lambda m: stats[m]["ppm"], default=None),
        "range_recall": max((m for m in header_methods if np.isfinite(stats[m]["range_recall"])), key=lambda m: stats[m]["range_recall"], default=None),
    }
    for m in header_methods:
        lines.append(
            f"{METHOD_TEX[m]} & "
            f"{fmt(stats[m]['recall'], 3, m == best['recall'])} & "
            f"{fmt(stats[m]['precision'], 3, m == best['precision'])} & "
            f"{fmt(stats[m]['ppm'], 1, m == best['ppm'])} & "
            f"{fmt(stats[m]['range_recall'], 3, m == best['range_recall'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "detection_table.tex").write_text("\n".join(lines) + "\n")


def ablation_table() -> None:
    path = ROOT / "outputs" / "benchmark_ablation" / "ablation_results.csv"
    if not path.exists():
        (TABLES / "ablation_table.tex").write_text("% ablation pending\n")
        return
    frame = pd.read_csv(path)
    frame = frame[frame.get("error").isna()] if "error" in frame.columns else frame
    order = [
        ("full", "Full pipeline (default)"),
        ("no_family_hints", "-- family fingerprints"),
        ("no_molecular_families", "-- molecular-family recovery"),
        ("no_element_pruning", "-- element pruning"),
        ("no_crowded_window_fit", "-- crowded-window deblending"),
        ("no_probability_floor", "-- probability floor"),
        ("no_member_verification", "-- member verification"),
        ("no_envelope_veto", "-- isotopologue-envelope veto"),
        ("no_atomic_support_penalty", "-- atomic support penalty"),
        (r"\addlinespace", None),
        ("with_log_peakiness", "++ log-peakiness channel (off by default)"),
        ("with_family_completion", "++ family completion (off by default)"),
    ]
    sio = frame[frame["dataset"] == "synthetic_si_o"].set_index("ablation")
    w = frame[frame["dataset"] == "control_W_18K_breen_kuehbach_R18_53222"].set_index("ablation")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation: mean composition L1 and mean fabricated signal",
        r"(at.\%) over the nine ablation controls, plus two sentinel cases:",
        r"the Si--O oxide (molecular-ion apportionment) and pure W",
        r"(phantom-element suppression). Rows above the rule disable one",
        r"default stage (higher is worse); the two \texttt{++} rows re-enable",
        r"stages that are \emph{off} by default, showing they do not help.}",
        r"\label{tab:ablation}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Variant & mean L1 & mean fab. & Si--O L1 & W fab. \\",
        r"\midrule",
    ]
    for key, label in order:
        if label is None:
            lines.append(key)  # raw LaTeX spacer (e.g. \addlinespace)
            continue
        sub = frame[frame["ablation"] == key]
        if sub.empty:
            continue
        lines.append(
            f"{label} & {fmt(sub['elemental_l1_at_pct'].mean(), 2)} & "
            f"{fmt(sub['fabricated_at_pct'].mean(), 2)} & "
            f"{fmt(sio.loc[key, 'elemental_l1_at_pct'] if key in sio.index else np.nan, 2)} & "
            f"{fmt(w.loc[key, 'fabricated_at_pct'] if key in w.index else np.nan, 2)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "ablation_table.tex").write_text("\n".join(lines) + "\n")


def foreign_table() -> None:
    mb_path = ROOT / "outputs" / "benchmark_massbank" / "massbank_summary.json"
    ts_path = ROOT / "outputs" / "benchmark_tofsims" / "tofsims_results.json"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Out-of-domain spectra. Left: 120 MassBank-derived",
        r"semi-synthetic profile spectra (organic fragment chemistry, exact",
        r"curated line lists). Right: real ToF-SIMS negative-polarity",
        r"spectra; recall on a reference list of known inorganic ions.",
        r"Bridge-based comparators are omitted here (in-process detectors",
        r"only).}",
        r"\label{tab:foreign}",
        r"\small",
        r"\begin{tabular}{lrrrr|rr}",
        r"\toprule",
        r" & \multicolumn{4}{c|}{MassBank semi-synthetic} & \multicolumn{2}{c}{ToF-SIMS} \\",
        r"Method & Recall$_w$ & Precision & False pk & $|\Delta m|$ ppm & Ion recall & Peaks \\",
        r"\midrule",
    ]
    mb = json.loads(mb_path.read_text())["per_method"] if mb_path.exists() else {}
    ts_rows = json.loads(ts_path.read_text()) if ts_path.exists() else []
    ts_by_method: dict[str, dict] = {}
    for row in ts_rows:
        if "error" in row or "known_ion_recall" not in row:
            continue
        ts_by_method.setdefault(row["method"], []).append(row)
    for m in ["custom", "pyccapt", "pyopenms", "naive"]:
        stats = mb.get(m, {})
        ts_list = ts_by_method.get(m, [])
        ion_recall = np.mean([r["known_ion_recall"] for r in ts_list]) if ts_list else np.nan
        n_peaks = np.mean([r["n_peaks"] for r in ts_list]) if ts_list else np.nan
        lines.append(
            f"{METHOD_TEX[m]} & {fmt(stats.get('line_recall_weighted'), 3)} & "
            f"{fmt(stats.get('peak_precision'), 3)} & {fmt(stats.get('n_false_peaks'), 1)} & "
            f"{fmt(stats.get('mass_accuracy_median_abs_ppm'), 1)} & "
            f"{fmt(ion_recall, 3)} & {fmt(n_peaks, 0)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "foreign_table.tex").write_text("\n".join(lines) + "\n")


DETECTOR_ORDER = ["rangefinder", "naive", "smoothed", "cwt", "persistent_homology", "derivative"]
DETECTOR_TEX = {
    "rangefinder": r"\rf{} (multi-resolution)",
    "naive": "Naive prominence",
    "smoothed": "Savitzky--Golay prominence",
    "cwt": "CWT ridge lines",
    "persistent_homology": "Persistent homology",
    "derivative": "Derivative zero-crossing",
}


def detection_ablation_table() -> None:
    path = ROOT / "outputs" / "detection_ablation" / "detection_ablation_results.json"
    if not path.exists():
        (TABLES / "detection_ablation_table.tex").write_text("% detection ablation pending\n")
        return
    frame = pd.DataFrame(json.loads(path.read_text()))
    frame = frame[~frame.get("error").notna()] if "error" in frame.columns else frame
    for column in ["elemental_l1_at_pct", "fabricated_at_pct", "peak_precision",
                   "mass_accuracy_median_abs_ppm", "line_recall_weighted"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    detectors = [d for d in DETECTOR_ORDER if d in set(frame["detector"])]
    comp = frame[frame["elemental_l1_at_pct"].notna()]
    pivot = comp.pivot_table(index="dataset", columns="detector", values="elemental_l1_at_pct").dropna(axis=0, how="any")
    syn = frame[frame["line_recall_weighted"].notna()]
    stats = {}
    for d in detectors:
        sub = comp[comp["detector"] == d]
        syn_d = syn[syn["detector"] == d]
        stats[d] = {
            "mean_l1": pivot[d].mean() if d in pivot.columns else np.nan,
            "worst_l1": pivot[d].max() if d in pivot.columns else np.nan,
            "n_bad": int((pivot[d] > 25).sum()) if d in pivot.columns else 0,
            "fab": sub["fabricated_at_pct"].mean(),
            "prec": syn_d["peak_precision"].mean(),
            "ppm": syn_d["mass_accuracy_median_abs_ppm"].mean(),
        }
    best = {
        "mean_l1": min(detectors, key=lambda d: stats[d]["mean_l1"]),
        "worst_l1": min(detectors, key=lambda d: stats[d]["worst_l1"]),
        "fab": min(detectors, key=lambda d: stats[d]["fab"]),
        "prec": max(detectors, key=lambda d: stats[d]["prec"]),
        "ppm": min(detectors, key=lambda d: stats[d]["ppm"]),
    }
    n = pivot.shape[0]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Detection ablation: only the peak detector is swapped; area",
        r"integration and \rf{}'s assignment stage are held identical, so scores",
        r"reflect detection alone. Columns: mean and worst-case composition L1",
        rf"(at.\%) over the {n} datasets every detector completed, count of",
        r"catastrophic failures (L1$>$25\,at.\%), mean fabricated signal, and on",
        r"the synthetic specimens peak precision and median mass error. \rf{}'s",
        r"multi-resolution detector is the only one that never fails",
        r"catastrophically; its mass accuracy reflects raw-event centroiding and",
        r"family recentering that bin-apex detectors cannot match. Bold = best.}",
        r"\label{tab:detection_ablation}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Detector & mean L1 & worst L1 & \#L1$>$25 & fab. & Prec. & $|\Delta m|$ ppm \\",
        r"\midrule",
    ]
    for d in detectors:
        s = stats[d]
        lines.append(
            f"{DETECTOR_TEX[d]} & "
            f"{fmt(s['mean_l1'], 2, d == best['mean_l1'])} & "
            f"{fmt(s['worst_l1'], 2, d == best['worst_l1'])} & "
            f"{int(s['n_bad'])} & "
            f"{fmt(s['fab'], 2, d == best['fab'])} & "
            f"{fmt(s['prec'], 3, d == best['prec'])} & "
            f"{fmt(s['ppm'], 1, d == best['ppm'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "detection_ablation_table.tex").write_text("\n".join(lines) + "\n")


def runtime_table(frame: pd.DataFrame) -> None:
    header_methods = [m for m in METHOD_ORDER if m in set(frame["method"])]
    frame = frame.copy()
    frame["runtime_s"] = pd.to_numeric(frame.get("runtime_s"), errors="coerce")
    synthetic = frame[frame["dataset"].str.startswith("synthetic")]
    big = frame[frame["dataset"] == "control_W_18K_breen_kuehbach_R18_53222"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Median wall-clock runtime (single process, Apple Silicon):",
        r"1.5M-event synthetic specimens and the 7.7M-event pure-W dataset.",
        r"Bridge methods (APAV, ms\_deisotope, APyT) include subprocess and",
        r"file I/O overhead.}",
        r"\label{tab:runtime}",
        r"\small",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method & 1.5M events (s) & 7.7M events (s) \\",
        r"\midrule",
    ]
    for m in header_methods:
        lines.append(
            f"{METHOD_TEX[m]} & "
            f"{fmt(synthetic[synthetic['method'] == m]['runtime_s'].median(), 1)} & "
            f"{fmt(big[big['method'] == m]['runtime_s'].median(), 1)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "runtime_table.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    frame = load_benchmark()
    frame = frame[~frame.get("error").notna()] if "error" in frame.columns else frame
    for column in ["elemental_l1_at_pct", "fabricated_at_pct", "line_recall_weighted",
                   "peak_precision", "mass_accuracy_median_abs_ppm", "range_recall_ion_weighted"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    composition_table(frame)
    detection_table(frame)
    detection_ablation_table()
    ablation_table()
    foreign_table()
    runtime_table(frame)
    print("Tables written to", TABLES)


if __name__ == "__main__":
    main()
