import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

from scipy.signal import savgol_filter

def smooth_signal(y):
    if len(y) < 15:
        return y
    return savgol_filter(y, 11, 3)


# -----------------------------
# TREND DETECTION
# -----------------------------
def detect_trend(x, y):
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    slope = model.coef_[0]

    if slope > 0.01:
        return "increasing", slope
    elif slope < -0.01:
        return "decreasing", slope
    else:
        return "flat", slope


# -----------------------------
# PERIODICITY DETECTION
# -----------------------------
def detect_periodicity(x, y):
    if len(y) < 10:
        return None, None

    distance = max(1, len(y) // 10)

    peaks, _ = find_peaks(y, distance=distance)

    if len(peaks) < 2:
        return None, None

    periods = np.diff(x[peaks])
    avg_period = np.mean(periods)

    if avg_period <= 0:
        return None, None

    frequency = 1.0 / avg_period
    return avg_period, frequency



# -----------------------------
# SINE-LIKE DETECTION
# -----------------------------
def is_sine_like(x, y):
    y = smooth_signal(y)

    peaks, _ = find_peaks(y, prominence=np.std(y) * 0.3)
    troughs, _ = find_peaks(-y, prominence=np.std(y) * 0.3)

    if len(peaks) < 3 or len(troughs) < 3:
        return False

    # Check consistent spacing
    peak_dist = np.diff(x[peaks])
    if np.std(peak_dist) > 0.3 * np.mean(peak_dist):
        return False

    # Check amplitude consistency
    min_len = min(len(peaks), len(troughs))

    if min_len < 2:
        return False  # not enough oscillations

    amplitudes = y[peaks[:min_len]] - y[troughs[:min_len]]


    # Check zero crossings
    zero_crossings = np.where(np.diff(np.sign(y - np.mean(y))))[0]
    if len(zero_crossings) < 4:
        return False

    return True




# -----------------------------
# MAIN ANALYSIS
# -----------------------------
def analyze_curve(curve):
    x = curve[:, 0]
    y = curve[:, 1]

    trend, slope = detect_trend(x, y)
    period, freq = detect_periodicity(x, y)
    sine_flag = is_sine_like(x, y)

    if not sine_flag:
        period, freq = None, None

    


    results = {
        "trend": trend,
        "slope": float(slope),
        "period": float(period) if period else None,
        "frequency": float(freq) if freq else None,
        "is_sine_like": sine_flag
    }

    drift_threshold = 0.02  # adjust if needed

    if sine_flag:
        if abs(slope) > drift_threshold:
            drift_type = "positive drift" if slope > 0 else "negative drift"
            results["interpretation"] = (
                f"Oscillatory signal with {drift_type}, "
                f"frequency ≈ {freq:.3f}" if freq else f"Oscillatory signal with {drift_type}"
            )
        else:
            results["interpretation"] = (
                f"Sine-like signal, frequency ≈ {freq:.3f}" if freq else "Sine-like signal"
            )

    elif trend == "increasing":
        results["interpretation"] = "Increasing trend"
    elif trend == "decreasing":
        results["interpretation"] = "Decreasing trend"
    else:
        results["interpretation"] = "No strong pattern detected"

    return results