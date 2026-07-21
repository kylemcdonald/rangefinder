from __future__ import annotations

import numpy as np
import pandas as pd

from rangefinder.analysis.tof_fitting import fit_isotope_family_tof, fit_isotope_family_tof_tail
from rangefinder.references.isotopes import isotope_table


def _default_sigma_da_init(
    charge: int,
    observed_fwhm_da: float | None,
    *,
    config: dict,
) -> float:
    if observed_fwhm_da is not None and np.isfinite(observed_fwhm_da) and observed_fwhm_da > 0:
        return float(max(observed_fwhm_da / 2.354820045, 0.005))
    analysis_cfg = config["analysis"]
    fallback = analysis_cfg.get("isotope_audit_default_fwhm_da_by_charge", {})
    fwhm_da = float(fallback.get(str(int(charge)), fallback.get("default", 0.08)))
    return float(max(fwhm_da / 2.354820045, 0.005))


def _family_overlap_risks(
    *,
    element: str,
    target_mz_da: np.ndarray,
    peaks: pd.DataFrame,
    assignments: pd.DataFrame,
    config: dict,
) -> list[bool]:
    """Flag each family target position where a non-family peak sits close enough
    to contaminate the family fit. The audit fit only models the family plus a
    linear baseline, so an interfering species inside the window is silently
    absorbed into the nearest isotope component."""
    analysis_cfg = config["analysis"]
    window_scale = float(analysis_cfg.get("isotope_audit_overlap_fwhm_scale", 1.5))
    min_window_da = float(analysis_cfg.get("isotope_audit_overlap_min_window_da", 0.06))
    if peaks.empty:
        return [False] * int(target_mz_da.size)
    top = assignments[assignments["rank"] == 1].copy() if not assignments.empty else pd.DataFrame()
    top_by_peak: dict[str, pd.Series] = {}
    if not top.empty:
        for _, row in top.iterrows():
            top_by_peak[str(row["peak_id"])] = row
    risks: list[bool] = []
    peak_mz = peaks["peak_mz_da"].to_numpy(dtype=np.float64)
    peak_fwhm = (
        peaks["fwhm_da"].fillna(0.0).to_numpy(dtype=np.float64)
        if "fwhm_da" in peaks.columns
        else np.zeros(peak_mz.size, dtype=np.float64)
    )
    peak_ids = peaks["peak_id"].astype(str).to_numpy()
    for target in np.asarray(target_mz_da, dtype=np.float64):
        risk = False
        for idx in range(peak_mz.size):
            window = max(min_window_da, window_scale * float(peak_fwhm[idx]))
            if abs(float(peak_mz[idx]) - float(target)) > window:
                continue
            top_row = top_by_peak.get(str(peak_ids[idx]))
            if top_row is None:
                # A peak with no ranked assignment near the target is still a
                # contamination risk for the family-only fit.
                risk = True
                break
            element_counts = top_row.get("element_counts")
            species_is_family_atom = (
                str(top_row.get("category", "")) == "atomic"
                and isinstance(element_counts, dict)
                and len(element_counts) == 1
                and str(element) in element_counts
            )
            if not species_is_family_atom and str(top_row.get("species_label", "")) != "unassigned":
                risk = True
                break
            if str(top_row.get("species_label", "")) == "unassigned":
                risk = True
                break
        risks.append(bool(risk))
    return risks


def _interference_positions(
    *,
    element: str,
    target_mz_da: np.ndarray,
    window_margin_da: float,
    assignments: pd.DataFrame,
    peaks: pd.DataFrame,
    top_elements: list[str],
    isotope_ref: pd.DataFrame,
    max_positions: int = 8,
    dedupe_da: float = 0.004,
) -> np.ndarray:
    """Known interference positions inside a family's audit window.

    Two sources: (a) reference positions of peaks the assignment stage placed in
    the window as other species, and (b) single-hydride positions of the sample's
    major elements (including the family element itself) -- the classic +1-Da
    contamination that inflates the heavy isotope of a family.
    """
    window_left = float(np.min(target_mz_da) - window_margin_da)
    window_right = float(np.max(target_mz_da) + window_margin_da)
    candidates: list[float] = []
    if not assignments.empty:
        top = assignments[
            (assignments["rank"] == 1)
            & (assignments["species_label"] != "unassigned")
            & assignments["candidate_mz_da"].notna()
        ]
        for _, row in top.iterrows():
            position = float(row["candidate_mz_da"])
            if not (window_left <= position <= window_right):
                continue
            element_counts = row.get("element_counts")
            is_family_atom = (
                str(row.get("category", "")) == "atomic"
                and isinstance(element_counts, dict)
                and len(element_counts) == 1
                and str(element) in element_counts
            )
            if not is_family_atom:
                candidates.append(position)
    hydrogen_mass = 1.007825
    for other in top_elements:
        family = isotope_ref[isotope_ref["symbol"] == str(other)]
        if family.empty:
            continue
        dominant_mass = float(family.loc[family["natural_abundance"].idxmax(), "mass_da"])
        for charge in (1, 2, 3):
            position = (dominant_mass + hydrogen_mass) / float(charge)
            if window_left <= position <= window_right:
                candidates.append(float(position))
    kept: list[float] = []
    targets = np.asarray(target_mz_da, dtype=np.float64)
    for position in sorted(candidates):
        if targets.size and float(np.min(np.abs(targets - position))) < dedupe_da:
            continue
        if kept and min(abs(position - existing) for existing in kept) < dedupe_da:
            continue
        kept.append(float(position))
    return np.asarray(kept[:max_positions], dtype=np.float64)


def audit_isotope_families(
    *,
    mz_values: np.ndarray,
    peaks: pd.DataFrame,
    assignments: pd.DataFrame,
    isotope_composition: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    analysis_cfg = config["analysis"]
    if not bool(analysis_cfg.get("isotope_audit_enabled", True)):
        return pd.DataFrame()
    if isotope_composition.empty:
        return pd.DataFrame()
    resolved = isotope_composition.copy()
    if "isotope_resolved" in resolved.columns:
        resolved = resolved[resolved["isotope_resolved"] == True].copy()  # noqa: E712
    if resolved.empty:
        return pd.DataFrame()
    top = assignments[assignments["rank"] == 1].copy() if not assignments.empty else pd.DataFrame()
    top_atomic = top[top["category"] == "atomic"].copy() if not top.empty else pd.DataFrame()
    peaks_by_id = peaks.set_index("peak_id") if not peaks.empty else pd.DataFrame()
    tail_enabled = bool(analysis_cfg.get("isotope_audit_tail_enabled", True))
    isotope_ref = isotope_table()
    top_elements: list[str] = []
    if not top.empty and not peaks_by_id.empty and "integrated_area" in peaks_by_id.columns:
        element_area: dict[str, float] = {}
        area_by_peak = peaks_by_id["integrated_area"].to_dict()
        for _, row in top.iterrows():
            element_counts = row.get("element_counts")
            if not isinstance(element_counts, dict):
                continue
            area = float(area_by_peak.get(str(row["peak_id"]), 0.0) or 0.0)
            for symbol, count in element_counts.items():
                element_area[str(symbol)] = element_area.get(str(symbol), 0.0) + area * float(count)
        top_elements = [
            symbol
            for symbol, _ in sorted(element_area.items(), key=lambda item: -item[1])[:6]
        ]
    rows: list[dict[str, object]] = []
    for element, group in resolved.groupby("element", sort=True):
        charges = group["selected_charge"].dropna().astype(int)
        if charges.empty:
            continue
        selected_charge = int(charges.iloc[0])
        natural = isotope_ref[isotope_ref["symbol"] == str(element)].copy()
        natural = natural.sort_values("mass_number").reset_index(drop=True)
        if natural.shape[0] < 2:
            continue
        target_mz_da = natural["mass_da"].to_numpy(dtype=np.float64) / float(selected_charge)
        family_assignments = top_atomic[
            (top_atomic["charge"].fillna(0).astype(int) == selected_charge)
            & (
                top_atomic["element_counts"].map(
                    lambda value: isinstance(value, dict)
                    and len(value) == 1
                    and str(element) in value
                )
            )
        ].copy()
        observed_fwhm = None
        calibration_shift_da = 0.0
        if not family_assignments.empty and not peaks_by_id.empty:
            family_peak_ids = family_assignments["peak_id"].astype(str).tolist()
            family_peaks = peaks_by_id.loc[peaks_by_id.index.intersection(family_peak_ids)]
            if not family_peaks.empty and "fwhm_da" in family_peaks.columns:
                observed_fwhm = float(np.median(family_peaks["fwhm_da"].to_numpy(dtype=float)))
            if not family_peaks.empty and "peak_mz_raw_da" in family_peaks.columns:
                # Calibration-recovered peaks were moved onto the reference mass
                # scale, but the raw events were not; the audit fit must be
                # allowed to find the family at its raw (shifted) position.
                raw_delta = (
                    family_peaks["peak_mz_raw_da"].astype(float)
                    - family_peaks["peak_mz_da"].astype(float)
                ).dropna()
                if not raw_delta.empty:
                    calibration_shift_da = float(np.median(raw_delta.to_numpy(dtype=float)))
        sigma_da_init = _default_sigma_da_init(selected_charge, observed_fwhm, config=config)
        interference_mz = _interference_positions(
            element=str(element),
            target_mz_da=target_mz_da,
            window_margin_da=float(analysis_cfg["isotope_audit_window_margin_da"]),
            assignments=assignments,
            peaks=peaks,
            top_elements=top_elements,
            isotope_ref=isotope_ref,
        )
        # The family shift models ppm-scale calibration error; a flat mDa budget
        # would let the family slide onto neighboring hydride or overlap peaks.
        # Calibration-recovered families keep their measured (larger) shift.
        reference_mz = float(np.median(target_mz_da))
        shift_tolerance = min(
            float(analysis_cfg["isotope_audit_shift_tolerance_da"]),
            max(
                float(analysis_cfg.get("isotope_audit_shift_ppm", 300.0)) * 1.0e-6 * reference_mz,
                float(analysis_cfg.get("isotope_audit_shift_floor_da", 0.008)),
            ),
        )
        if abs(calibration_shift_da) > 0.005:
            shift_tolerance = max(shift_tolerance, abs(calibration_shift_da) + 0.02)
        fit = fit_isotope_family_tof(
            np.asarray(mz_values, dtype=np.float64),
            target_mz_da=target_mz_da,
            sigma_da_init=sigma_da_init,
            window_margin_da=float(analysis_cfg["isotope_audit_window_margin_da"]),
            shift_tolerance_da=shift_tolerance,
            sigma_floor_da=float(analysis_cfg["isotope_audit_sigma_floor_da"]),
            sigma_ceiling_da=float(analysis_cfg["isotope_audit_sigma_ceiling_da"]),
            bins_per_sigma=float(analysis_cfg["isotope_audit_bins_per_sigma"]),
            shift_penalty=float(analysis_cfg["isotope_audit_shift_penalty"]),
            sigma_penalty=float(analysis_cfg["isotope_audit_sigma_penalty"]),
            fixed_mz_da=interference_mz,
        )
        if fit is None:
            continue
        fitted_counts = np.asarray(fit["fitted_counts"], dtype=np.float64)
        total_counts = float(np.sum(fitted_counts))
        if total_counts < float(analysis_cfg["isotope_audit_min_total_counts"]):
            continue
        tail_fit = None
        if tail_enabled:
            tail_fit = fit_isotope_family_tof_tail(
                np.asarray(mz_values, dtype=np.float64),
                target_mz_da=target_mz_da,
                sigma_da_init=sigma_da_init,
                window_margin_da=float(analysis_cfg["isotope_audit_window_margin_da"]),
                shift_tolerance_da=shift_tolerance,
                sigma_floor_da=float(analysis_cfg["isotope_audit_sigma_floor_da"]),
                sigma_ceiling_da=float(analysis_cfg["isotope_audit_sigma_ceiling_da"]),
                bins_per_sigma=float(analysis_cfg["isotope_audit_bins_per_sigma"]),
                shift_penalty=float(analysis_cfg["isotope_audit_shift_penalty"]),
                sigma_penalty=float(analysis_cfg["isotope_audit_sigma_penalty"]),
                tail_tau_max_sigma_scale=float(analysis_cfg.get("isotope_audit_tail_tau_max_sigma_scale", 6.0)),
                tail_penalty=float(analysis_cfg.get("isotope_audit_tail_penalty", 0.05)),
                fixed_mz_da=interference_mz,
            )
        tail_fraction = (
            np.asarray(tail_fit["fitted_fraction"], dtype=np.float64)
            if tail_fit is not None
            else np.full(target_mz_da.size, np.nan, dtype=np.float64)
        )
        tail_tau_t = float(tail_fit["tau_t"]) if tail_fit is not None else np.nan
        overlap_risks = _family_overlap_risks(
            element=str(element),
            target_mz_da=target_mz_da,
            peaks=peaks,
            assignments=assignments,
            config=config,
        )
        fitted_mz = np.asarray(fit["fitted_mz_da"], dtype=np.float64)
        fitted_fraction = np.asarray(fit["fitted_fraction"], dtype=np.float64)
        for idx, isotope_row in natural.iterrows():
            shape_delta = (
                float(abs(fitted_fraction[idx] - tail_fraction[idx]))
                if np.isfinite(tail_fraction[idx])
                else np.nan
            )
            rows.append(
                {
                    "element": str(element),
                    "isotope_label": str(isotope_row["isotope_label"]),
                    "audit_charge": int(selected_charge),
                    "target_mz_da": float(target_mz_da[idx]),
                    "audit_mz_da": float(fitted_mz[idx]),
                    "audit_counts": float(fitted_counts[idx]),
                    "audit_fraction": float(fitted_fraction[idx]),
                    "audit_fraction_tail": float(tail_fraction[idx]) if np.isfinite(tail_fraction[idx]) else np.nan,
                    "audit_shape_delta": shape_delta,
                    "audit_tail_tau_t": tail_tau_t,
                    "overlap_risk": bool(overlap_risks[idx]),
                    "natural_fraction": float(isotope_row["natural_abundance"]),
                    "fraction_diff": float(fitted_fraction[idx] - float(isotope_row["natural_abundance"])),
                    "calibration_shift_da": calibration_shift_da,
                    "audit_interference_positions": int(interference_mz.size),
                    "audit_interference_counts": float(
                        np.sum(np.asarray(fit.get("interference_counts", []), dtype=np.float64))
                    ),
                    "audit_family_total_counts": total_counts,
                    "audit_sigma_t": float(fit["sigma_t"]),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(["element", "isotope_label"], ignore_index=True)
    return frame


def reconcile_isotope_anomalies(
    anomalies: pd.DataFrame,
    isotope_audit: pd.DataFrame,
    *,
    config: dict,
) -> pd.DataFrame:
    """Gate counting-statistics anomaly flags against systematic effects.

    The raw anomaly table uses binomial variance only, so with 1e5+ counts a
    sub-percent ranging or peak-shape bias produces arbitrarily large z-scores.
    This pass (a) adds a systematic floor to the variance, (b) requires the
    independent event-level audit fit to agree in direction and rough size, and
    (c) downgrades isotopes whose family windows carry known overlap risk.
    """
    if anomalies.empty:
        return anomalies
    analysis_cfg = config["analysis"]
    systematic_floor = float(analysis_cfg.get("isotope_anomaly_systematic_fraction_floor", 0.005))
    audit_support_ratio = float(analysis_cfg.get("isotope_anomaly_audit_support_ratio", 0.4))
    slight_abs_diff = float(analysis_cfg["slight_anomaly_min_abs_fraction_diff"])
    strong_abs_diff = float(analysis_cfg["strong_anomaly_min_abs_fraction_diff"])
    slight_z = float(analysis_cfg["slight_anomaly_min_z"])
    strong_z = float(analysis_cfg["strong_anomaly_min_z"])
    reconciled = anomalies.copy()
    reconciled["status_counting_only"] = reconciled["status"]
    audit_lookup: dict[tuple[str, str], pd.Series] = {}
    if not isotope_audit.empty:
        for _, row in isotope_audit.iterrows():
            audit_lookup[(str(row["element"]), str(row["isotope_label"]))] = row
    adjusted_status: list[str] = []
    adjusted_z: list[float] = []
    audit_fraction_col: list[float] = []
    audit_supported_col: list[object] = []
    overlap_col: list[object] = []
    systematic_col: list[float] = []
    notes_col: list[str] = []
    for _, row in reconciled.iterrows():
        obs = float(row["observed_fraction"])
        exp = float(row["expected_fraction"])
        diff = obs - exp
        z_counting = float(row["z_score"])
        counting_var = (diff / z_counting) ** 2 if abs(z_counting) > 1.0e-12 else 1.0e-12
        audit_row = audit_lookup.get((str(row["element"]), str(row["isotope_label"])))
        shape_delta = np.nan
        audit_fraction = np.nan
        overlap_risk = None
        audit_supported = None
        notes: list[str] = []
        if audit_row is not None:
            audit_fraction = float(audit_row["audit_fraction"])
            shape_delta = float(audit_row.get("audit_shape_delta", np.nan))
            overlap_risk = bool(audit_row.get("overlap_risk", False))
            audit_diff = audit_fraction - exp
            same_direction = np.sign(audit_diff) == np.sign(diff) and abs(diff) > 0
            large_enough = abs(audit_diff) >= audit_support_ratio * abs(diff)
            audit_supported = bool(same_direction and large_enough)
        systematic = systematic_floor
        if np.isfinite(shape_delta):
            systematic = max(systematic, float(shape_delta))
        z_adjusted = diff / np.sqrt(counting_var + systematic**2)
        abs_diff = abs(diff)
        if abs_diff >= strong_abs_diff and abs(z_adjusted) >= strong_z:
            status = "strong"
        elif abs_diff >= slight_abs_diff and abs(z_adjusted) >= slight_z:
            status = "slight"
        else:
            status = "none"
        if status != "none":
            if audit_supported is None:
                if status == "strong":
                    status = "slight"
                notes.append("no independent audit fit; capped at slight")
            elif not audit_supported:
                status = "slight" if status == "strong" else status
                notes.append("audit fit does not reproduce the deviation")
            if overlap_risk:
                if status == "strong":
                    status = "slight"
                notes.append("family window has overlap risk from a non-family peak")
        adjusted_status.append(status)
        adjusted_z.append(float(z_adjusted))
        audit_fraction_col.append(audit_fraction)
        audit_supported_col.append(audit_supported)
        overlap_col.append(overlap_risk)
        systematic_col.append(float(systematic))
        notes_col.append("; ".join(notes))
    reconciled["audit_fraction"] = audit_fraction_col
    reconciled["audit_supported"] = audit_supported_col
    reconciled["overlap_risk"] = overlap_col
    reconciled["systematic_fraction_uncertainty"] = systematic_col
    reconciled["z_score_adjusted"] = adjusted_z
    reconciled["status"] = adjusted_status
    reconciled["gating_notes"] = notes_col
    return reconciled
