import numpy as np
from scipy.signal import find_peaks, savgol_filter
from sklearn.linear_model import LinearRegression


# -----------------------------
# HELPERS
# -----------------------------
def _safe_ptp(a: np.ndarray) -> float:
    """Range with protection against tiny/degenerate arrays."""
    r = float(np.ptp(a)) if len(a) else 0.0
    return r if r > 1e-12 else 1e-12


def smooth_signal(y: np.ndarray) -> np.ndarray:
    n = len(y)
    if n < 7:
        return y

    # Choose an odd window size ~ 1/5 of points, bounded.
    w = max(7, n // 5)
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w < 7:
        return y

    # polyorder must be < window length
    poly = 3 if w >= 9 else 2
    return savgol_filter(y, w, poly)


def decompose_signal(x: np.ndarray, y: np.ndarray):
    trend = smooth_signal(y)
    residual = y - trend
    return trend, residual


# -----------------------------
# TREND DETECTION 
# -----------------------------
def detect_trend(x: np.ndarray, y_trend: np.ndarray):
    x = np.asarray(x)
    y_trend = np.asarray(y_trend)

    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y_trend)
    slope = float(model.coef_[0])

    # Scale-aware slope threshold:
    # "meaningful slope" ≈ a few % of y-range across the x-span.
    x_span = _safe_ptp(x)
    y_span = _safe_ptp(y_trend)
    # slope has units y/x, so compare to (fraction*y_span)/x_span
    slope_eps = 0.03 * (y_span / x_span)  # 3% of y-range across domain

    if slope > slope_eps:
        return "increasing", slope, slope_eps
    elif slope < -slope_eps:
        return "decreasing", slope, slope_eps
    else:
        return "flat", slope, slope_eps


# -----------------------------
# PERIODICITY DETECTION 
# -----------------------------
def detect_periodicity(x: np.ndarray, y_resid: np.ndarray):
    x = np.asarray(x)
    y_resid = np.asarray(y_resid)

    if len(y_resid) < 12:
        return None, None

    # Dynamic prominence relative to residual scale
    resid_scale = np.std(y_resid)
    if resid_scale < 1e-12:
        return None, None

    # Distance heuristic: avoid detecting every tiny wiggle
    distance = max(2, len(y_resid) // 8)

    peaks, _ = find_peaks(y_resid, distance=distance, prominence=resid_scale * 0.3)
    troughs, _ = find_peaks(-y_resid, distance=distance, prominence=resid_scale * 0.3)

    # Need multiple cycles (peaks OR troughs)
    idx = peaks if len(peaks) >= 3 else troughs
    if len(idx) < 3:
        return None, None

    periods = np.diff(x[idx])
    periods = periods[periods > 0]
    if len(periods) < 2:
        return None, None

    avg_period = float(np.mean(periods))
    if avg_period <= 0:
        return None, None

    frequency = float(1.0 / avg_period)
    return avg_period, frequency


# -----------------------------
# SINE-LIKE DETECTION 
# -----------------------------
def is_sine_like(x: np.ndarray, y_resid: np.ndarray):
    x = np.asarray(x)
    y = np.asarray(y_resid)

    n = len(y)
    if n < 12:
        return False

    scale = np.std(y)
    if scale < 1e-12:
        return False

    # Find peaks/troughs on residual
    peaks, _ = find_peaks(y, prominence=scale * 0.3, distance=max(2, n // 8))
    troughs, _ = find_peaks(-y, prominence=scale * 0.3, distance=max(2, n // 8))

    if len(peaks) < 3 or len(troughs) < 3:
        return False

    # Spacing consistency (peaks)
    peak_dist = np.diff(x[peaks])
    if np.mean(peak_dist) <= 0:
        return False
    if np.std(peak_dist) > 0.35 * np.mean(peak_dist):
        return False

    # Amplitude consistency: peak-to-trough over matched cycles
    m = min(len(peaks), len(troughs))
    if m < 3:
        return False

    amps = y[peaks[:m]] - y[troughs[:m]]
    if np.mean(np.abs(amps)) < 1e-12:
        return False
    if np.std(amps) > 0.6 * np.mean(np.abs(amps)):
        return False

    # Zero crossings in residual around 0 (since detrended)
    zero_crossings = np.where(np.diff(np.sign(y)) != 0)[0]
    if len(zero_crossings) < 6:
        return False

    return True


# -----------------------------
# CURVATURE 
# -----------------------------
def curvature_score(x: np.ndarray, y: np.ndarray):
    """
    Scale-normalized curvature proxy based on discrete second derivative.
    Returns (curv_norm, convexity_sign) where convexity_sign is:
      +1 mostly convex up, -1 mostly concave down, 0 unclear.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(y) < 5:
        return 0.0, 0

    # Use finite differences w.r.t x (handles nonuniform x spacing)
    dx = np.diff(x)
    dy = np.diff(y)

    # protect against zero dx
    dx = np.where(np.abs(dx) < 1e-12, 1e-12, dx)
    dydx = dy / dx

    d2 = np.diff(dydx)  # approximate second derivative (scaled)
    if len(d2) == 0:
        return 0.0, 0

    # Normalize curvature by y-range / x-range to make it scale-aware
    x_span = _safe_ptp(x)
    y_span = _safe_ptp(y)
    curv_raw = float(np.median(np.abs(d2)))
    curv_norm = float(curv_raw / (y_span / x_span))  # unitless-ish

    # convexity sign (robust): majority sign of d2
    pos = np.sum(d2 > 0)
    neg = np.sum(d2 < 0)
    if pos > 2 * neg:
        sign = 1
    elif neg > 2 * pos:
        sign = -1
    else:
        sign = 0

    return curv_norm, sign


# -----------------------------
# MAIN ANALYSIS
# -----------------------------
def analyze_curve(curve):
    x = curve[:, 0].astype(float)
    y = curve[:, 1].astype(float)

    trend_y, resid_y = decompose_signal(x, y)

    # Trend on trend component (not raw)
    trend, slope, slope_eps = detect_trend(x, trend_y)

    # Periodicity + sine-like on residual
    period, freq = detect_periodicity(x, resid_y)
    sine_flag = is_sine_like(x, resid_y)

    if not sine_flag:
        period, freq = None, None

    curv_norm, convexity = curvature_score(x, trend_y)

    results = {
        "trend": trend,
        "slope": float(slope),
        "slope_threshold": float(slope_eps),
        "period": float(period) if period is not None else None,
        "frequency": float(freq) if freq is not None else None,
        "is_sine_like": bool(sine_flag),
        "curvature": float(curv_norm),
        "convexity": int(convexity),  # +1 convex up, -1 concave down, 0 unclear
    }

    # Drift threshold should also be scale-aware:
    # use a multiple of slope_eps rather than a fixed constant.
    drift_threshold = 2.0 * slope_eps

    # Interpretation
    if sine_flag:
        if abs(slope) > drift_threshold:
            drift_type = "positive drift" if slope > 0 else "negative drift"
            if freq is not None:
                results["interpretation"] = f"Oscillatory signal with {drift_type}, frequency ≈ {freq:.3f}"
            else:
                results["interpretation"] = f"Oscillatory signal with {drift_type}"
        else:
            if freq is not None:
                results["interpretation"] = f"Sine-like signal, frequency ≈ {freq:.3f}"
            else:
                results["interpretation"] = "Sine-like signal"

    else:
        # Use curvature to refine monotonic interpretations
        # Heuristic: curv_norm > ~0.6 indicates notable nonlinearity
        if trend == "increasing":
            if curv_norm > 0.6 and convexity >= 0:
                results["interpretation"] = "Increasing trend with accelerating growth"
            elif curv_norm > 0.6 and convexity < 0:
                results["interpretation"] = "Increasing trend with diminishing growth"
            else:
                results["interpretation"] = "Increasing trend"

        elif trend == "decreasing":
            if curv_norm > 0.6 and convexity <= 0:
                results["interpretation"] = "Decreasing trend with accelerating decline"
            elif curv_norm > 0.6 and convexity > 0:
                results["interpretation"] = "Decreasing trend with diminishing decline"
            else:
                results["interpretation"] = "Decreasing trend"

        else:
            # flat
            if curv_norm > 0.6:
                results["interpretation"] = "Mostly flat trend with nonlinear variation"
            else:
                results["interpretation"] = "No strong pattern detected"

    return results
