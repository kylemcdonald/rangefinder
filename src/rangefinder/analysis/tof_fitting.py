from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erfc, erfcx


SIGMA_REFERENCE_MZ_DA = 28.0


def sigma_bound_t(sigma_da: float) -> float:
    """Convert a Da-valued width bound, defined at the reference mass, into the
    mass-independent normalized-TOF width it implies."""
    return sigma_da_to_sigma_t(SIGMA_REFERENCE_MZ_DA, float(sigma_da))


def mz_to_tof(mz: np.ndarray | float) -> np.ndarray | float:
    return np.sqrt(np.clip(np.asarray(mz, dtype=np.float64), 1.0e-12, None))


def sigma_da_to_sigma_t(mz_da: float, sigma_da: float) -> float:
    return float(sigma_da) / max(2.0 * np.sqrt(max(float(mz_da), 1.0e-12)), 1.0e-9)


def fwhm_da_from_sigma_t(mz_da: float, sigma_t: float) -> float:
    t_value = float(mz_to_tof(mz_da))
    fwhm_t = 2.354820045 * float(sigma_t)
    return float(max(2.0 * t_value * fwhm_t, 0.0))


def _histogram_tof(mz_values: np.ndarray, *, t_min: float, t_max: float, dt: float):
    tof_values = mz_to_tof(mz_values)
    edges = np.arange(t_min, t_max + dt, dt, dtype=np.float64)
    if edges.size < 8:
        edges = np.linspace(t_min, t_max, 9, dtype=np.float64)
    counts, edges = np.histogram(tof_values, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts.astype(np.float64), float(edges[1] - edges[0])


def _baseline_initialization(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if y.size == 0:
        return 0.0, 0.0
    side_count = max(2, min(8, y.size // 4))
    left_y = float(np.median(y[:side_count]))
    right_y = float(np.median(y[-side_count:]))
    span = max(float(x[-1] - x[0]), 1.0e-9)
    intercept = max(0.0, 0.5 * (left_y + right_y))
    slope = (right_y - left_y) / span
    return intercept, slope


def _clip_initial_guess(parameter0: np.ndarray, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> np.ndarray:
    eps = 1.0e-9
    safe_upper = np.maximum(upper_bounds - eps, lower_bounds + eps)
    safe_lower = np.minimum(lower_bounds + eps, safe_upper)
    return np.clip(parameter0, safe_lower, safe_upper)


def _fit_histogram_dt(
    *,
    reference_mz_da: float,
    sigma_floor_da: float,
    bins_per_sigma: float,
) -> float:
    # Keep the fitting histogram fine enough to resolve narrow isotope triplets even
    # when the seed widths are initially over-broadened by the detector front end.
    # Peak width is ~constant in t across the spectrum, so the floor is converted
    # at the fixed reference mass rather than the local window mass.
    return max(sigma_bound_t(float(sigma_floor_da)) / max(float(bins_per_sigma), 1.0), 1.0e-5)


def fit_local_tof_peak_mixture(
    mz_values: np.ndarray,
    *,
    seed_mz_da: np.ndarray,
    seed_fwhm_da: np.ndarray,
    window_margin_da: float,
    center_tolerance_da: float,
    sigma_floor_da: float,
    sigma_ceiling_da: float,
    bins_per_sigma: float = 5.0,
    offset_penalty: float = 0.2,
    sigma_penalty: float = 0.1,
) -> dict[str, object] | None:
    seed_mz_da = np.asarray(seed_mz_da, dtype=np.float64)
    seed_fwhm_da = np.asarray(seed_fwhm_da, dtype=np.float64)
    if seed_mz_da.size == 0:
        return None
    window_left = float(np.min(seed_mz_da) - window_margin_da)
    window_right = float(np.max(seed_mz_da) + window_margin_da)
    local_mz = np.asarray(mz_values, dtype=np.float64)
    local_mz = local_mz[(local_mz >= window_left) & (local_mz <= window_right)]
    if local_mz.size < max(20, 4 * seed_mz_da.size):
        return None
    seed_t = mz_to_tof(seed_mz_da)
    reference_mz_da = float(np.median(seed_mz_da))
    sigma_da_init = float(np.median(np.maximum(seed_fwhm_da / 2.354820045, sigma_floor_da)))
    sigma_t_init = sigma_da_to_sigma_t(reference_mz_da, sigma_da_init)
    dt = _fit_histogram_dt(
        reference_mz_da=reference_mz_da,
        sigma_floor_da=float(sigma_floor_da),
        bins_per_sigma=float(bins_per_sigma),
    )
    t_min = float(mz_to_tof(window_left))
    t_max = float(mz_to_tof(window_right))
    x_t, y_t, dt = _histogram_tof(local_mz, t_min=t_min, t_max=t_max, dt=dt)
    if y_t.size < 8 or float(y_t.sum()) <= 0.0:
        return None
    t_ref = float(np.mean(seed_t))
    baseline_intercept0, baseline_slope0 = _baseline_initialization(x_t, y_t)
    sigma_t0 = float(np.clip(sigma_t_init, sigma_bound_t(sigma_floor_da), sigma_bound_t(sigma_ceiling_da)))
    center_offsets0 = np.zeros(seed_t.size, dtype=np.float64)
    baseline_at_seed = baseline_intercept0 + baseline_slope0 * (seed_t - t_ref)
    nearest_idx = np.searchsorted(x_t, seed_t).clip(0, x_t.size - 1)
    amplitude0 = np.maximum(y_t[nearest_idx] - baseline_at_seed, 1.0)
    parameter0 = np.concatenate(
        [
            amplitude0,
            center_offsets0,
            np.asarray([sigma_t0, baseline_intercept0, baseline_slope0], dtype=np.float64),
        ]
    )
    max_y = float(max(np.max(y_t), 1.0))
    center_tolerance_t = np.asarray(
        [sigma_da_to_sigma_t(mz, center_tolerance_da) for mz in seed_mz_da],
        dtype=np.float64,
    )
    sigma_t_floor = sigma_bound_t(sigma_floor_da)
    sigma_t_ceiling = sigma_bound_t(sigma_ceiling_da)
    span = max(float(x_t[-1] - x_t[0]), 1.0e-9)
    lower_bounds = np.concatenate(
        [
            np.zeros(seed_t.size, dtype=np.float64),
            -center_tolerance_t,
            np.asarray([sigma_t_floor, 0.0, -max_y / span], dtype=np.float64),
        ]
    )
    upper_bounds = np.concatenate(
        [
            np.full(seed_t.size, max_y * 6.0, dtype=np.float64),
            center_tolerance_t,
            np.asarray([sigma_t_ceiling, max_y * 2.0, max_y / span], dtype=np.float64),
        ]
    )

    def residuals(parameter_vector: np.ndarray) -> np.ndarray:
        amplitudes = parameter_vector[: seed_t.size]
        offsets = parameter_vector[seed_t.size : 2 * seed_t.size]
        sigma_t = float(parameter_vector[-3])
        baseline_intercept = float(parameter_vector[-2])
        baseline_slope = float(parameter_vector[-1])
        center_t = seed_t + offsets
        baseline = baseline_intercept + baseline_slope * (x_t - t_ref)
        profiles = np.exp(-0.5 * ((x_t[:, None] - center_t[None, :]) / max(sigma_t, 1.0e-9)) ** 2)
        model = baseline + profiles @ amplitudes
        weights = np.sqrt(np.maximum(y_t, 1.0))
        penalty_offsets = np.sqrt(float(offset_penalty)) * offsets / np.maximum(center_tolerance_t, 1.0e-9)
        penalty_sigma = np.asarray(
            [np.sqrt(float(sigma_penalty)) * (sigma_t - sigma_t0) / max(sigma_t0, 1.0e-9)],
            dtype=np.float64,
        )
        return np.concatenate([(model - y_t) / weights, penalty_offsets, penalty_sigma])

    parameter0 = _clip_initial_guess(parameter0, lower_bounds, upper_bounds)
    result = least_squares(
        residuals,
        parameter0,
        bounds=(lower_bounds, upper_bounds),
        method="trf",
        loss="soft_l1",
        f_scale=max(np.sqrt(max_y), 5.0),
        max_nfev=500,
    )
    if not result.success:
        return None
    amplitudes = result.x[: seed_t.size]
    offsets = result.x[seed_t.size : 2 * seed_t.size]
    sigma_t = float(result.x[-3])
    baseline_intercept = float(result.x[-2])
    baseline_slope = float(result.x[-1])
    center_t = seed_t + offsets
    baseline = baseline_intercept + baseline_slope * (x_t - t_ref)
    profiles = np.exp(-0.5 * ((x_t[:, None] - center_t[None, :]) / max(sigma_t, 1.0e-9)) ** 2)
    component_curves = profiles * amplitudes[None, :]
    fitted_counts = component_curves.sum(axis=0)
    fitted_mz_da = np.square(center_t)
    fitted_fwhm_da = np.asarray([fwhm_da_from_sigma_t(mz, sigma_t) for mz in fitted_mz_da], dtype=np.float64)
    return {
        "fitted_mz_da": fitted_mz_da,
        "fitted_counts": fitted_counts,
        "fitted_fwhm_da": fitted_fwhm_da,
        "sigma_t": sigma_t,
        "baseline_intercept": baseline_intercept,
        "baseline_slope": baseline_slope,
        "x_t": x_t,
        "y_t": y_t,
        "baseline_curve": baseline,
        "component_curves": component_curves,
    }


def _emg_profile_matrix(
    x_t: np.ndarray,
    center_t: np.ndarray,
    sigma_t: float,
    tau_t: float,
) -> np.ndarray:
    # Exponentially modified Gaussian in normalized time-of-flight: a Gaussian core
    # convolved with a one-sided exponential delay toward larger t, the direction in
    # which thermally delayed evaporation events actually arrive. Written with erfcx
    # for numerical stability at small tau.
    sigma = max(float(sigma_t), 1.0e-9)
    tau = max(float(tau_t), 1.0e-9)
    delta = x_t[:, None] - center_t[None, :]
    z = sigma / (tau * np.sqrt(2.0)) - delta / (sigma * np.sqrt(2.0))
    # Two-branch evaluation for numerical stability: erfcx(z) overflows for very
    # negative z (far into the tail), where the exponential form is safe because
    # its exponent sigma^2/(2 tau^2) - delta/tau is strictly negative there.
    positive = z >= 0.0
    out = np.empty_like(z)
    z_pos = np.where(positive, z, 0.0)
    out_pos = erfcx(z_pos) * np.exp(-0.5 * (delta / sigma) ** 2)
    exponent = np.clip(sigma**2 / (2.0 * tau**2) - delta / tau, None, 0.0)
    out_neg = np.exp(exponent) * erfc(np.where(positive, 0.0, z))
    out = np.where(positive, out_pos, out_neg)
    return (1.0 / (2.0 * tau)) * out


def fit_isotope_family_tof_tail(
    mz_values: np.ndarray,
    *,
    target_mz_da: np.ndarray,
    sigma_da_init: float,
    window_margin_da: float,
    shift_tolerance_da: float,
    sigma_floor_da: float,
    sigma_ceiling_da: float,
    bins_per_sigma: float = 6.0,
    shift_penalty: float = 0.15,
    sigma_penalty: float = 0.1,
    tail_tau_max_sigma_scale: float = 6.0,
    tail_penalty: float = 0.05,
    fixed_mz_da: np.ndarray | None = None,
) -> dict[str, object] | None:
    """Fit a full isotope family with a shared EMG (Gaussian + exponential tail) shape.

    Same windowing and penalty conventions as fit_isotope_family_tof, but each
    component is area-parameterized so the shared tail cannot silently change the
    relative isotope fractions through normalization.
    """
    target_mz_da = np.asarray(target_mz_da, dtype=np.float64)
    if target_mz_da.size < 2:
        return None
    window_left = float(np.min(target_mz_da) - window_margin_da)
    window_right = float(np.max(target_mz_da) + window_margin_da)
    local_mz = np.asarray(mz_values, dtype=np.float64)
    local_mz = local_mz[(local_mz >= window_left) & (local_mz <= window_right)]
    if local_mz.size < 20:
        return None
    fixed_mz_da = (
        np.asarray(fixed_mz_da, dtype=np.float64)
        if fixed_mz_da is not None
        else np.asarray([], dtype=np.float64)
    )
    target_t = mz_to_tof(target_mz_da)
    fixed_t = mz_to_tof(fixed_mz_da) if fixed_mz_da.size else np.asarray([], dtype=np.float64)
    reference_mz_da = float(np.median(target_mz_da))
    sigma_t0 = sigma_da_to_sigma_t(reference_mz_da, sigma_da_init)
    dt = _fit_histogram_dt(
        reference_mz_da=reference_mz_da,
        sigma_floor_da=float(sigma_floor_da),
        bins_per_sigma=float(bins_per_sigma),
    )
    t_min = float(mz_to_tof(window_left))
    t_max = float(mz_to_tof(window_right))
    x_t, y_t, dt = _histogram_tof(local_mz, t_min=t_min, t_max=t_max, dt=dt)
    if y_t.size < 8 or float(y_t.sum()) <= 0.0:
        return None
    t_ref = float(np.mean(target_t))
    baseline_intercept0, baseline_slope0 = _baseline_initialization(x_t, y_t)
    shift_tolerance_t = sigma_da_to_sigma_t(float(np.median(target_mz_da)), shift_tolerance_da)
    baseline_at_targets = baseline_intercept0 + baseline_slope0 * (target_t - t_ref)
    nearest_idx = np.searchsorted(x_t, target_t).clip(0, x_t.size - 1)
    height0 = np.maximum(y_t[nearest_idx] - baseline_at_targets, 1.0)
    sigma_t_floor = sigma_bound_t(sigma_floor_da)
    sigma_t_ceiling = sigma_bound_t(sigma_ceiling_da)
    sigma_t0 = float(np.clip(sigma_t0, sigma_t_floor, sigma_t_ceiling))
    # Area parameterization: height ~= area / (sigma * sqrt(2 pi)) for a small tail.
    area0 = height0 * sigma_t0 * np.sqrt(2.0 * np.pi)
    fixed_area0 = (
        np.maximum(
            y_t[np.searchsorted(x_t, fixed_t).clip(0, x_t.size - 1)] * 0.25 * sigma_t0 * np.sqrt(2.0 * np.pi),
            1.0,
        )
        if fixed_t.size
        else np.asarray([], dtype=np.float64)
    )
    tau0 = float(max(sigma_t0 * 0.5, 1.0e-6))
    tau_max = float(max(sigma_t_ceiling * float(tail_tau_max_sigma_scale), tau0 * 2.0))
    parameter0 = np.concatenate(
        [
            area0,
            fixed_area0,
            np.asarray([0.0, sigma_t0, tau0, baseline_intercept0, baseline_slope0], dtype=np.float64),
        ]
    )
    max_y = float(max(np.max(y_t), 1.0))
    max_area = max_y * 6.0 * sigma_t_ceiling * np.sqrt(2.0 * np.pi)
    span = max(float(x_t[-1] - x_t[0]), 1.0e-9)
    lower_bounds = np.concatenate(
        [
            np.zeros(target_t.size + fixed_t.size, dtype=np.float64),
            np.asarray([-shift_tolerance_t, sigma_t_floor, 1.0e-6, 0.0, -max_y / span], dtype=np.float64),
        ]
    )
    upper_bounds = np.concatenate(
        [
            np.full(target_t.size + fixed_t.size, max_area, dtype=np.float64),
            np.asarray([shift_tolerance_t, sigma_t_ceiling, tau_max, max_y * 2.0, max_y / span], dtype=np.float64),
        ]
    )

    def residuals(parameter_vector: np.ndarray) -> np.ndarray:
        areas = parameter_vector[: target_t.size]
        fixed_areas = parameter_vector[target_t.size : target_t.size + fixed_t.size]
        delta_t = float(parameter_vector[-5])
        sigma_t = float(parameter_vector[-4])
        tau_t = float(parameter_vector[-3])
        baseline_intercept = float(parameter_vector[-2])
        baseline_slope = float(parameter_vector[-1])
        center_t = target_t + delta_t
        baseline = baseline_intercept + baseline_slope * (x_t - t_ref)
        profiles = _emg_profile_matrix(x_t, center_t, sigma_t, tau_t)
        model = baseline + profiles @ areas
        if fixed_t.size:
            model = model + _emg_profile_matrix(x_t, fixed_t, sigma_t, tau_t) @ fixed_areas
        weights = np.sqrt(np.maximum(y_t, 1.0))
        penalty_shift = np.asarray(
            [np.sqrt(float(shift_penalty)) * delta_t / max(shift_tolerance_t, 1.0e-9)],
            dtype=np.float64,
        )
        penalty_sigma = np.asarray(
            [np.sqrt(float(sigma_penalty)) * (sigma_t - sigma_t0) / max(sigma_t0, 1.0e-9)],
            dtype=np.float64,
        )
        penalty_tau = np.asarray(
            [np.sqrt(float(tail_penalty)) * tau_t / max(sigma_t0, 1.0e-9)],
            dtype=np.float64,
        )
        return np.concatenate([(model - y_t) / weights, penalty_shift, penalty_sigma, penalty_tau])

    parameter0 = _clip_initial_guess(parameter0, lower_bounds, upper_bounds)
    result = least_squares(
        residuals,
        parameter0,
        bounds=(lower_bounds, upper_bounds),
        method="trf",
        loss="soft_l1",
        f_scale=max(np.sqrt(max_y), 5.0),
        max_nfev=600,
    )
    if not result.success:
        return None
    areas = result.x[: target_t.size]
    fixed_areas = result.x[target_t.size : target_t.size + fixed_t.size]
    delta_t = float(result.x[-5])
    sigma_t = float(result.x[-4])
    tau_t = float(result.x[-3])
    baseline_intercept = float(result.x[-2])
    baseline_slope = float(result.x[-1])
    center_t = target_t + delta_t
    baseline = baseline_intercept + baseline_slope * (x_t - t_ref)
    profiles = _emg_profile_matrix(x_t, center_t, sigma_t, tau_t)
    component_curves = profiles * areas[None, :]
    fitted_counts = component_curves.sum(axis=0)
    if fixed_t.size:
        interference_counts = (_emg_profile_matrix(x_t, fixed_t, sigma_t, tau_t) * fixed_areas[None, :]).sum(axis=0)
    else:
        interference_counts = np.asarray([], dtype=np.float64)
    fitted_mz_da = np.square(center_t)
    return {
        "fitted_mz_da": fitted_mz_da,
        "fitted_counts": fitted_counts,
        "fitted_fraction": fitted_counts / max(float(np.sum(fitted_counts)), 1.0e-9),
        "interference_mz_da": fixed_mz_da,
        "interference_counts": interference_counts,
        "sigma_t": sigma_t,
        "tau_t": tau_t,
        "delta_t": delta_t,
        "baseline_intercept": baseline_intercept,
        "baseline_slope": baseline_slope,
        "x_t": x_t,
        "y_t": y_t,
        "baseline_curve": baseline,
        "component_curves": component_curves,
    }


def fit_isotope_family_tof(
    mz_values: np.ndarray,
    *,
    target_mz_da: np.ndarray,
    sigma_da_init: float,
    window_margin_da: float,
    shift_tolerance_da: float,
    sigma_floor_da: float,
    sigma_ceiling_da: float,
    bins_per_sigma: float = 6.0,
    shift_penalty: float = 0.15,
    sigma_penalty: float = 0.1,
    fixed_mz_da: np.ndarray | None = None,
) -> dict[str, object] | None:
    """fixed_mz_da: optional known interference positions (e.g. hydrides or other
    species' peaks inside the window). They share the family width but not the
    family shift, and their amplitudes are free, so overlapping intensity is
    apportioned by shape instead of being absorbed into the nearest isotope."""
    target_mz_da = np.asarray(target_mz_da, dtype=np.float64)
    if target_mz_da.size < 2:
        return None
    fixed_mz_da = (
        np.asarray(fixed_mz_da, dtype=np.float64)
        if fixed_mz_da is not None
        else np.asarray([], dtype=np.float64)
    )
    window_left = float(np.min(target_mz_da) - window_margin_da)
    window_right = float(np.max(target_mz_da) + window_margin_da)
    local_mz = np.asarray(mz_values, dtype=np.float64)
    local_mz = local_mz[(local_mz >= window_left) & (local_mz <= window_right)]
    if local_mz.size < 20:
        return None
    target_t = mz_to_tof(target_mz_da)
    reference_mz_da = float(np.median(target_mz_da))
    sigma_t0 = sigma_da_to_sigma_t(reference_mz_da, sigma_da_init)
    dt = _fit_histogram_dt(
        reference_mz_da=reference_mz_da,
        sigma_floor_da=float(sigma_floor_da),
        bins_per_sigma=float(bins_per_sigma),
    )
    t_min = float(mz_to_tof(window_left))
    t_max = float(mz_to_tof(window_right))
    x_t, y_t, dt = _histogram_tof(local_mz, t_min=t_min, t_max=t_max, dt=dt)
    if y_t.size < 8 or float(y_t.sum()) <= 0.0:
        return None
    t_ref = float(np.mean(target_t))
    fixed_t = mz_to_tof(fixed_mz_da) if fixed_mz_da.size else np.asarray([], dtype=np.float64)
    baseline_intercept0, baseline_slope0 = _baseline_initialization(x_t, y_t)
    shift_tolerance_t = sigma_da_to_sigma_t(float(np.median(target_mz_da)), shift_tolerance_da)
    baseline_at_targets = baseline_intercept0 + baseline_slope0 * (target_t - t_ref)
    nearest_idx = np.searchsorted(x_t, target_t).clip(0, x_t.size - 1)
    amplitude0 = np.maximum(y_t[nearest_idx] - baseline_at_targets, 1.0)
    fixed_amplitude0 = (
        np.maximum(y_t[np.searchsorted(x_t, fixed_t).clip(0, x_t.size - 1)] * 0.25, 1.0)
        if fixed_t.size
        else np.asarray([], dtype=np.float64)
    )
    parameter0 = np.concatenate(
        [
            amplitude0,
            fixed_amplitude0,
            np.asarray([0.0, sigma_t0, baseline_intercept0, baseline_slope0], dtype=np.float64),
        ]
    )
    max_y = float(max(np.max(y_t), 1.0))
    sigma_t_floor = sigma_bound_t(sigma_floor_da)
    sigma_t_ceiling = sigma_bound_t(sigma_ceiling_da)
    span = max(float(x_t[-1] - x_t[0]), 1.0e-9)
    lower_bounds = np.concatenate(
        [
            np.zeros(target_t.size + fixed_t.size, dtype=np.float64),
            np.asarray([-shift_tolerance_t, sigma_t_floor, 0.0, -max_y / span], dtype=np.float64),
        ]
    )
    upper_bounds = np.concatenate(
        [
            np.full(target_t.size + fixed_t.size, max_y * 6.0, dtype=np.float64),
            np.asarray([shift_tolerance_t, sigma_t_ceiling, max_y * 2.0, max_y / span], dtype=np.float64),
        ]
    )

    def residuals(parameter_vector: np.ndarray) -> np.ndarray:
        amplitudes = parameter_vector[: target_t.size]
        fixed_amplitudes = parameter_vector[target_t.size : target_t.size + fixed_t.size]
        delta_t = float(parameter_vector[-4])
        sigma_t = float(parameter_vector[-3])
        baseline_intercept = float(parameter_vector[-2])
        baseline_slope = float(parameter_vector[-1])
        center_t = target_t + delta_t
        baseline = baseline_intercept + baseline_slope * (x_t - t_ref)
        profiles = np.exp(-0.5 * ((x_t[:, None] - center_t[None, :]) / max(sigma_t, 1.0e-9)) ** 2)
        model = baseline + profiles @ amplitudes
        if fixed_t.size:
            fixed_profiles = np.exp(-0.5 * ((x_t[:, None] - fixed_t[None, :]) / max(sigma_t, 1.0e-9)) ** 2)
            model = model + fixed_profiles @ fixed_amplitudes
        weights = np.sqrt(np.maximum(y_t, 1.0))
        penalty_shift = np.asarray(
            [np.sqrt(float(shift_penalty)) * delta_t / max(shift_tolerance_t, 1.0e-9)],
            dtype=np.float64,
        )
        penalty_sigma = np.asarray(
            [np.sqrt(float(sigma_penalty)) * (sigma_t - sigma_t0) / max(sigma_t0, 1.0e-9)],
            dtype=np.float64,
        )
        return np.concatenate([(model - y_t) / weights, penalty_shift, penalty_sigma])

    parameter0 = _clip_initial_guess(parameter0, lower_bounds, upper_bounds)
    result = least_squares(
        residuals,
        parameter0,
        bounds=(lower_bounds, upper_bounds),
        method="trf",
        loss="soft_l1",
        f_scale=max(np.sqrt(max_y), 5.0),
        max_nfev=500,
    )
    if not result.success:
        return None
    amplitudes = result.x[: target_t.size]
    fixed_amplitudes = result.x[target_t.size : target_t.size + fixed_t.size]
    delta_t = float(result.x[-4])
    sigma_t = float(result.x[-3])
    baseline_intercept = float(result.x[-2])
    baseline_slope = float(result.x[-1])
    center_t = target_t + delta_t
    baseline = baseline_intercept + baseline_slope * (x_t - t_ref)
    profiles = np.exp(-0.5 * ((x_t[:, None] - center_t[None, :]) / max(sigma_t, 1.0e-9)) ** 2)
    component_curves = profiles * amplitudes[None, :]
    fitted_counts = component_curves.sum(axis=0)
    if fixed_t.size:
        fixed_profiles = np.exp(-0.5 * ((x_t[:, None] - fixed_t[None, :]) / max(sigma_t, 1.0e-9)) ** 2)
        interference_counts = (fixed_profiles * fixed_amplitudes[None, :]).sum(axis=0)
    else:
        interference_counts = np.asarray([], dtype=np.float64)
    fitted_mz_da = np.square(center_t)
    return {
        "fitted_mz_da": fitted_mz_da,
        "fitted_counts": fitted_counts,
        "fitted_fraction": fitted_counts / max(float(np.sum(fitted_counts)), 1.0e-9),
        "interference_mz_da": fixed_mz_da,
        "interference_counts": interference_counts,
        "sigma_t": sigma_t,
        "delta_t": delta_t,
        "baseline_intercept": baseline_intercept,
        "baseline_slope": baseline_slope,
        "x_t": x_t,
        "y_t": y_t,
        "baseline_curve": baseline,
        "component_curves": component_curves,
    }
