"""
Scientific Plot Understanding — Hugging Face Space
Detects layout elements, reads text via OCR, segments curves,
reconstructs numeric data, and performs semantic analysis.
"""

import gradio as gr
import numpy as np
import cv2
import re
import json
import tempfile
import os
from pathlib import Path
from PIL import Image, ImageDraw
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import easyocr
from difflib import get_close_matches
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from sklearn.linear_model import LinearRegression

# ============================================================
# CONFIG
# ============================================================
# Change this to your HF model repo, e.g. "your-username/scientific-plot-understanding"
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "")

CLASS_NAMES = {
    0: "plot_area", 1: "legend", 2: "x_label", 3: "y_label",
    4: "x_ticks", 5: "y_ticks", 6: "title",
}

CLASS_COLORS = {
    "plot_area": "#00FF00", "legend": "#FF00FF", "x_label": "#00FFFF",
    "y_label": "#FFFF00", "x_ticks": "#FF8800", "y_ticks": "#0088FF",
    "title": "#FF0000",
}

# ============================================================
# MODEL LOADING
# ============================================================
def load_model():
    """Load YOLO model — from HF Hub if configured, else local best.pt."""
    if HF_MODEL_REPO:
        model_path = hf_hub_download(HF_MODEL_REPO, "best.pt")
    else:
        model_path = str(Path(__file__).parent / "best.pt")
    return YOLO(model_path)


model = load_model()
reader = easyocr.Reader(["en"], gpu=False)

# ============================================================
# UTILS
# ============================================================

def clamp_box(x1, y1, x2, y2, W, H, pad=10):
    return max(0, int(x1 - pad)), max(0, int(y1 - pad)), min(W, int(x2 + pad)), min(H, int(y2 + pad))

def dedupe_close(nums, tol=1.5):
    nums = sorted(nums)
    out = []
    for n in nums:
        if not out or abs(n - out[-1]) > tol:
            out.append(n)
    return out

def maybe_rotate_vertical(pil_img):
    w, h = pil_img.size
    return pil_img.rotate(-90, expand=True) if h > w else pil_img

def filter_by_spacing(ticks, tolerance=0.35):
    if len(ticks) < 3:
        return ticks
    ticks = sorted(ticks)
    diffs = np.diff(ticks)
    median_step = np.median(diffs)
    cleaned = [ticks[0]]
    for t in ticks[1:]:
        if abs((t - cleaned[-1]) - median_step) < median_step * tolerance:
            cleaned.append(t)
    return cleaned

def clean_label(text):
    lower = text.lower()
    if re.search(r'\bsin\b', lower): return "sin(x)"
    if re.search(r'\bcos\b', lower): return "cos(x)"
    if re.search(r'\btan\b', lower): return "tan(x)"
    if re.search(r'\blog\b', lower): return "log(x)"
    if re.search(r'\bexp\b', lower): return "exp(x)"

    corrections = {
        "Betwen": "Between", "Betweem": "Between", "Enupion": "Eruption",
        "Enuption": "Eruption", "Eruptlons": "Eruptions", "Etupilons": "Eruptions",
        "Falthful": "Faithful", "Falhful": "Faithful", "Hln": "Min", "lnj": "Min",
        "MlN": "Min", "BetwenEruptions": "Between Eruptions", "Wlaling": "Waiting",
        "Tine": "Time", "Duratilon": "Duration", "Velocitv": "Velocity",
        "Velocily": "Velocity", "Powerer": "Power", "Distarce": "Distance",
        "Acceleraion": "Acceleration",
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'(.)\1{1,}', r'\1', text)
    text = text.replace("))", ")").replace("((", "(")
    text = " ".join(text.split())

    KNOWN_LABELS = [
        "Time (s)", "Time (ms)", "Time (min)", "Samples", "Distance (m)",
        "Position (m)", "Displacement (m)", "Velocity", "Velocity (m/s)",
        "Speed (m/s)", "Acceleration (m/s^2)", "Amplitude", "Signal",
        "Signal Amplitude", "Frequency (Hz)", "Frequency (kHz)", "Phase",
        "Voltage (V)", "Current (A)", "Resistance (Ohm)", "Power (W)",
        "Energy (J)", "Force (N)", "Mass (kg)", "Pressure (kPa)",
        "Temperature (C)", "Wavelength (nm)", "Wavelength (um)", "Intensity",
        "Input", "Output", "Response", "Measurement", "Value", "Sample Data",
        "sin(x)", "cos(x)", "tan(x)", "exp(x)", "log(x)", "ln(x)",
        "x^2", "x^3", "Polynomial", "Linear", "Quadratic", "Cubic",
        "Model Output", "Prediction", "Ground Truth", "Error", "Eruptions",
    ]
    match = get_close_matches(text, KNOWN_LABELS, n=1, cutoff=0.6)
    if match:
        text = match[0]
    return text.strip()


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_ticks(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.equalizeHist(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 3)
    return gray

def preprocess_label(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray

def preprocess_legend(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

def regularize_ticks(ticks):
    if len(ticks) < 3:
        return ticks
    ticks = sorted(ticks)
    diffs = np.diff(ticks)
    step = np.median(diffs)
    if step == 0:
        return ticks
    start, end = ticks[0], ticks[-1]
    fixed, v = [], start
    while v <= end + step * 0.5:
        fixed.append(round(v, 2))
        v += step
    return fixed

def enforce_zero_tick(ticks):
    if not ticks:
        return ticks
    if any(t < 0 for t in ticks) and any(t > 0 for t in ticks) and 0.0 not in ticks:
        ticks.append(0.0)
    return sorted(ticks)

def remove_outlier_ticks(ticks):
    if len(ticks) < 4:
        return ticks
    ticks = sorted(ticks)
    diffs = np.diff(ticks)
    median_step = np.median(diffs)
    cleaned = [ticks[0]]
    for t in ticks[1:]:
        if abs(t - cleaned[-1] - median_step) < median_step * 0.4:
            cleaned.append(t)
    return cleaned


# ============================================================
# OCR
# ============================================================

def ocr_label(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.equalizeHist(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, 12, 7, 21)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)
    results = reader.readtext(
        gray, detail=0, paragraph=True,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789() /-",
    )
    return " ".join(results).strip()

def ocr_legend(pil_img):
    img = preprocess_legend(pil_img)
    results = reader.readtext(
        img, detail=1,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ",
        paragraph=False,
    )
    rows = []
    for (bbox, text, conf) in results:
        if conf < 0.4:
            continue
        pts = np.array(bbox)
        y_center = pts[:, 1].mean()
        rows.append((y_center, text.strip()))
    rows.sort(key=lambda x: x[0])
    labels = []
    for _, text in rows:
        m = re.search(r"Line\s*\d+", text)
        labels.append(m.group() if m else text)
    seen = set()
    clean = []
    for l in labels:
        if l not in seen:
            clean.append(l)
            seen.add(l)
    return clean

def ocr_ticks_sorted(pil_img, axis="y", mode="easyocr"):
    if isinstance(pil_img, Image.Image):
        img = preprocess_ticks(pil_img)
    else:
        img = pil_img

    if mode == "easyocr":
        results = reader.readtext(
            img, detail=1, allowlist="0123456789.-", paragraph=False,
            text_threshold=0.4, low_text=0.2, link_threshold=0.3,
        )
        values = []
        for (bbox, text, conf) in results:
            if conf < 0.35:
                continue
            nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
            if not nums:
                continue
            pts = np.array(bbox)
            pos = pts[:, 0].mean() if axis == "x" else pts[:, 1].mean()
            values.append((pos, float(nums[0])))
        values.sort(key=lambda x: x[0])
        return dedupe_close([v[1] for v in values], tol=0.2)

    # MODE B: contour splitting (for y_ticks)
    bin_img = cv2.bitwise_not(img)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    components = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > 12 and w > 3:
            components.append((x, y, w, h))
    if axis == "y":
        components.sort(key=lambda b: b[1])
    else:
        components.sort(key=lambda b: b[0])

    boxes = []
    current = None
    GAP_Y, GAP_X = 35, 25
    for (x, y, w, h) in components:
        if current is None:
            current = [x, y, x + w, y + h]
            continue
        if axis == "y":
            gap = y - current[3]
            if gap <= GAP_Y:
                current = [min(current[0], x), min(current[1], y),
                           max(current[2], x + w), max(current[3], y + h)]
            else:
                boxes.append(tuple(current))
                current = [x, y, x + w, y + h]
        else:
            gap = x - current[2]
            if gap <= GAP_X:
                current = [min(current[0], x), min(current[1], y),
                           max(current[2], x + w), max(current[3], y + h)]
            else:
                boxes.append(tuple(current))
                current = [x, y, x + w, y + h]
    if current is not None:
        boxes.append(tuple(current))

    values = []
    for (x1, y1, x2, y2) in boxes:
        crop = bin_img[y1:y2, x1:x2]
        crop = cv2.copyMakeBorder(crop, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=0)
        txt = reader.readtext(crop, detail=0, allowlist="0123456789.-")
        if not txt:
            continue
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", txt[0])
        if nums:
            values.append(float(nums[0]))

    values = sorted(values)
    if len(values) >= 3:
        step = np.median(np.diff(values))
        if values[0] == 0 and step > 0:
            values = [values[0] - step] + values
    return dedupe_close(values, tol=0.2)


# ============================================================
# YOLO LAYOUT DETECTION
# ============================================================

def detect_layout(yolo_model, img_path):
    res = yolo_model(img_path)[0]
    boxes = {}
    for b in res.boxes:
        cls = int(b.cls.item())
        name = CLASS_NAMES.get(cls, f"class_{cls}")
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        boxes.setdefault(name, []).append((x1, y1, x2, y2))
    return boxes


# ============================================================
# STAGE 2 — OCR
# ============================================================

def stage2(img_path, yolo_model=None):
    if yolo_model is None:
        yolo_model = model
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    boxes = detect_layout(yolo_model, img_path)

    plot_box_list = boxes.get("plot_area", None)
    if plot_box_list:
        xs1 = [b[0] for b in plot_box_list]
        ys1 = [b[1] for b in plot_box_list]
        xs2 = [b[2] for b in plot_box_list]
        ys2 = [b[3] for b in plot_box_list]
        plot_box = (min(xs1), min(ys1), max(xs2), max(ys2))
    else:
        plot_box = None

    final_boxes = {}
    crops = {}
    for name, box_list in boxes.items():
        if name in ["x_ticks", "y_ticks"] and isinstance(box_list, list):
            xs1 = [b[0] for b in box_list]
            ys1 = [b[1] for b in box_list]
            xs2 = [b[2] for b in box_list]
            ys2 = [b[3] for b in box_list]
            box = (min(xs1), min(ys1), max(xs2), max(ys2))
        else:
            box = box_list[0]
        x1, y1, x2, y2 = box

        if name == "y_ticks":
            y2 -= 10
            x1 += int((x2 - x1) * 0.35)
            x1 -= 6; y1 -= 6; x2 += 6; y2 += 6
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H, pad=2)
        elif name == "y_label":
            if plot_box:
                _, py1, _, py2 = plot_box
                y1, y2 = py1 - 10, py2 + 10
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H, pad=8)
        elif name == "title":
            y1 -= int((y2 - y1) * 0.8); y2 += 10; x1 -= 10; x2 += 10
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H, pad=4)
        elif name == "x_label":
            x1, y1, x2, y2 = clamp_box(x1, y1 + 8, x2, y2, W, H, pad=4)
        elif name == "x_ticks":
            x1 -= 15; x2 += 15; y1 -= 5; y2 += 5
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H, pad=4)
        elif name == "legend":
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H, pad=4)
        else:
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H, pad=6)

        final_boxes[name] = (x1, y1, x2, y2)
        crops[name] = img.crop((x1, y1, x2, y2))

    results = {}
    if "y_label" in crops:
        y_img = maybe_rotate_vertical(crops["y_label"])
        results["y_label"] = clean_label(ocr_label(y_img))
    else:
        results["y_label"] = ""
    results["x_label"] = clean_label(ocr_label(crops["x_label"])) if "x_label" in crops else ""
    results["title"] = clean_label(ocr_label(crops["title"])) if "title" in crops else ""

    if "x_ticks" in crops:
        results["x_ticks"] = ocr_ticks_sorted(crops["x_ticks"], axis="x", mode="easyocr")
    else:
        results["x_ticks"] = []

    if "y_ticks" in crops:
        results["y_ticks"] = ocr_ticks_sorted(crops["y_ticks"], axis="y", mode="contours")
    else:
        results["y_ticks"] = []

    if "legend" in crops:
        labels = ocr_legend(crops["legend"])
        line_nums = []
        for l in labels:
            m = re.search(r"Line\s*(\d+)", l)
            if m:
                line_nums.append(int(m.group(1)))
        if len(line_nums) >= 2:
            labels = [f"Line {i}" for i in range(min(line_nums), max(line_nums) + 1)]
        results["legend_labels"] = labels
    else:
        results["legend_labels"] = []

    return results


# ============================================================
# STAGE 3 — CURVE SEGMENTATION (color-based fallback)
# ============================================================

def segment_by_color(plot_img):
    img = np.array(plot_img)
    H, W, _ = img.shape
    diff_rg = np.abs(img[:, :, 0].astype(int) - img[:, :, 1].astype(int))
    diff_gb = np.abs(img[:, :, 1].astype(int) - img[:, :, 2].astype(int))
    color_mask = (diff_rg > 15) | (diff_gb > 15)
    ys, xs = np.where(color_mask)
    if len(ys) < 200:
        return []
    pixels = img[ys, xs]
    quantized = (pixels // 32) * 32
    uniq = np.unique(quantized, axis=0)
    components = []
    for c in uniq:
        sel = np.all(quantized == c, axis=1)
        if np.sum(sel) < 200:
            continue
        cmask = np.zeros((H, W), np.uint8)
        cmask[ys[sel], xs[sel]] = 255
        components.append(cmask)
    return components


def stage3(img_path, yolo_model=None):
    if yolo_model is None:
        yolo_model = model
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    boxes = detect_layout(yolo_model, img_path)
    if "plot_area" not in boxes:
        raise RuntimeError("plot_area not detected")
    plot_boxes = boxes["plot_area"]
    if isinstance(plot_boxes, list):
        xs1 = [b[0] for b in plot_boxes]
        ys1 = [b[1] for b in plot_boxes]
        xs2 = [b[2] for b in plot_boxes]
        ys2 = [b[3] for b in plot_boxes]
        plot_box = (min(xs1), min(ys1), max(xs2), max(ys2))
    else:
        plot_box = plot_boxes
    x1, y1, x2, y2 = clamp_box(*plot_box, W, H)
    plot_crop = img.crop((x1, y1, x2, y2))
    components = segment_by_color(plot_crop)
    return components


# ============================================================
# STAGE 4 — RECONSTRUCTION
# ============================================================

def mask_to_trace(mask):
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    ys, xs = np.where(mask > 0)
    col_map = {}
    for y, x in zip(ys, xs):
        col_map.setdefault(x, []).append(y)
    x_obs = np.array(sorted(col_map.keys()), dtype=np.float32)
    y_obs = np.array([np.median(col_map[int(x)]) for x in x_obs], dtype=np.float32)
    return np.column_stack([x_obs, y_obs]).astype(np.float32)


def stage4(img_path, components, yolo_model=None):
    if yolo_model is None:
        yolo_model = model
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    boxes = detect_layout(yolo_model, img_path)
    if "plot_area" not in boxes:
        raise RuntimeError("plot_area not detected")
    plot_boxes = boxes["plot_area"]
    if isinstance(plot_boxes, list):
        xs1 = [b[0] for b in plot_boxes]
        ys1 = [b[1] for b in plot_boxes]
        xs2 = [b[2] for b in plot_boxes]
        ys2 = [b[3] for b in plot_boxes]
        plot_box = (min(xs1), min(ys1), max(xs2), max(ys2))
    else:
        plot_box = plot_boxes
    x1, y1, x2, y2 = clamp_box(*plot_box, W, H)
    plot_crop = img.crop((x1, y1, x2, y2))
    crop_w, crop_h = plot_crop.size
    if not components:
        return None

    meta = stage2(img_path, yolo_model)
    xt = meta.get("x_ticks", [])
    yt = meta.get("y_ticks", [])
    if len(xt) < 2 or len(yt) < 2:
        # Can't reconstruct without axis ranges, return OCR-only
        return {
            "title": meta.get("title", ""), "x_label": meta.get("x_label", ""),
            "y_label": meta.get("y_label", ""), "x_ticks": xt, "y_ticks": yt,
            "legend": meta.get("legend_labels", []), "curves": [],
        }

    xmin, xmax = min(xt), max(xt)
    ymin, ymax = min(yt), max(yt)

    curves = []
    for comp in components:
        # For color-segmented masks the mask is already in plot_crop coords
        trace_px = mask_to_trace(comp)
        if trace_px.shape[0] < 5:
            continue
        data_x = xmin + (trace_px[:, 0] / crop_w) * (xmax - xmin)
        data_y = ymax - (trace_px[:, 1] / crop_h) * (ymax - ymin)
        curves.append(np.column_stack((data_x, data_y)).astype(np.float32))

    legend_labels = meta.get("legend_labels", [])
    if legend_labels:
        curves = sorted(curves, key=lambda c: len(c), reverse=True)
        curves = curves[:len(legend_labels)]

    return {
        "title": meta.get("title", ""), "x_label": meta.get("x_label", ""),
        "y_label": meta.get("y_label", ""), "x_ticks": xt, "y_ticks": yt,
        "legend": legend_labels, "curves": curves,
    }


# ============================================================
# SEMANTIC ANALYSIS
# ============================================================

def _safe_ptp(a):
    r = float(np.ptp(a)) if len(a) else 0.0
    return r if r > 1e-12 else 1e-12

def smooth_signal(y):
    n = len(y)
    if n < 7:
        return y
    w = max(7, n // 5)
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w < 7:
        return y
    poly = 3 if w >= 9 else 2
    return savgol_filter(y, w, poly)

def decompose_signal(x, y):
    trend = smooth_signal(y)
    residual = y - trend
    return trend, residual

def detect_trend(x, y_trend):
    x, y_trend = np.asarray(x), np.asarray(y_trend)
    mdl = LinearRegression().fit(x.reshape(-1, 1), y_trend)
    slope = float(mdl.coef_[0])
    x_span, y_span = _safe_ptp(x), _safe_ptp(y_trend)
    slope_eps = 0.03 * (y_span / x_span)
    if slope > slope_eps:
        return "increasing", slope, slope_eps
    elif slope < -slope_eps:
        return "decreasing", slope, slope_eps
    return "flat", slope, slope_eps

def detect_periodicity(x, y_resid):
    x, y_resid = np.asarray(x), np.asarray(y_resid)
    if len(y_resid) < 12:
        return None, None
    resid_scale = np.std(y_resid)
    if resid_scale < 1e-12:
        return None, None
    distance = max(2, len(y_resid) // 8)
    peaks, _ = find_peaks(y_resid, distance=distance, prominence=resid_scale * 0.3)
    troughs, _ = find_peaks(-y_resid, distance=distance, prominence=resid_scale * 0.3)
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
    return avg_period, float(1.0 / avg_period)

def is_sine_like(x, y_resid):
    """Detect if the curve is sine-like / oscillatory.

    IMPROVED: Relaxed thresholds vs original:
    - Requires only 2 peaks and 2 troughs (was 3)
    - Lower prominence threshold (0.2 * std, was 0.3)
    - Requires only 3 zero crossings (was 6)
    - Added autocorrelation-based fallback detection
    """
    x, y = np.asarray(x), np.asarray(y_resid)
    n = len(y)
    if n < 12:
        return False
    scale = np.std(y)
    if scale < 1e-6:
        return False

    # Primary: peak/trough detection with relaxed thresholds
    distance = max(2, n // 8)
    peaks, _ = find_peaks(y, prominence=scale * 0.2, distance=distance)
    troughs, _ = find_peaks(-y, prominence=scale * 0.2, distance=distance)

    if len(peaks) >= 2 and len(troughs) >= 2:
        # Check even spacing of peaks
        peak_dist = np.diff(x[peaks])
        if len(peak_dist) > 0 and np.mean(peak_dist) > 0:
            if np.std(peak_dist) <= 0.4 * np.mean(peak_dist):
                # Check zero crossings
                mean_y = np.mean(y)
                zero_crossings = np.where(np.diff(np.sign(y - mean_y)))[0]
                if len(zero_crossings) >= 3:
                    return True

    # Fallback: autocorrelation-based detection
    if n >= 20:
        y_centered = y - np.mean(y)
        norm = np.sum(y_centered ** 2)
        if norm > 1e-6:
            autocorr = np.correlate(y_centered, y_centered, mode='full')
            autocorr = autocorr[len(autocorr) // 2:] / norm
            if len(autocorr) > 10:
                ac_peaks, _ = find_peaks(autocorr[1:], height=0.3,
                                          distance=len(autocorr) // 10)
                if len(ac_peaks) >= 1:
                    return True

    return False

def curvature_score(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(y) < 5:
        return 0.0, 0
    dx = np.diff(x)
    dy = np.diff(y)
    dx = np.where(np.abs(dx) < 1e-12, 1e-12, dx)
    dydx = dy / dx
    d2 = np.diff(dydx)
    if len(d2) == 0:
        return 0.0, 0
    x_span, y_span = _safe_ptp(x), _safe_ptp(y)
    curv_raw = float(np.median(np.abs(d2)))
    curv_norm = float(curv_raw / (y_span / x_span))
    pos, neg = np.sum(d2 > 0), np.sum(d2 < 0)
    sign = 1 if pos > 2 * neg else (-1 if neg > 2 * pos else 0)
    return curv_norm, sign

def analyze_curve(curve):
    x = curve[:, 0].astype(float)
    y = curve[:, 1].astype(float)
    trend_y, resid_y = decompose_signal(x, y)
    trend, slope, slope_eps = detect_trend(x, trend_y)
    period, freq = detect_periodicity(x, resid_y)
    sine_flag = is_sine_like(x, resid_y)
    if not sine_flag:
        period, freq = None, None
    curv_norm, convexity = curvature_score(x, trend_y)

    results = {
        "trend": trend, "slope": round(float(slope), 4),
        "slope_threshold": round(float(slope_eps), 4),
        "period": round(float(period), 4) if period else None,
        "frequency": round(float(freq), 4) if freq else None,
        "is_sine_like": bool(sine_flag),
        "curvature": round(float(curv_norm), 4),
        "convexity": int(convexity),
    }

    drift_threshold = 2.0 * slope_eps
    if sine_flag:
        if abs(slope) > drift_threshold:
            drift = "positive drift" if slope > 0 else "negative drift"
            results["interpretation"] = f"Oscillatory signal with {drift}" + (f", frequency ≈ {freq:.3f}" if freq else "")
        else:
            results["interpretation"] = ("Sine-like signal" + (f", frequency ≈ {freq:.3f}" if freq else ""))
    else:
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
            results["interpretation"] = "Mostly flat trend with nonlinear variation" if curv_norm > 0.6 else "No strong pattern detected"
    return results


# ============================================================
# FULL PIPELINE — ties everything together
# ============================================================

def run_full_pipeline(image):
    """Main Gradio handler: image in → annotated layout, reconstruction plot, JSON."""
    # Save to temp file (YOLO needs a path)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
        if isinstance(image, np.ndarray):
            Image.fromarray(image).save(tmp_path)
        else:
            image.save(tmp_path)

    try:
        # --- Layout detection annotated image ---
        yolo_results = model(tmp_path)[0]
        annotated_bgr = yolo_results.plot()
        annotated_rgb = annotated_bgr[..., ::-1]

        # --- OCR (Stage 2) ---
        ocr_data = stage2(tmp_path, model)

        # --- Curve segmentation (Stage 3) ---
        try:
            components = stage3(tmp_path, model)
        except RuntimeError:
            components = []

        # --- Reconstruction (Stage 4) ---
        reconstruction = None
        if components and len(ocr_data.get("x_ticks", [])) >= 2 and len(ocr_data.get("y_ticks", [])) >= 2:
            reconstruction = stage4(tmp_path, components, model)

        # --- Build reconstruction plot ---
        recon_fig = None
        if reconstruction and reconstruction.get("curves"):
            fig, ax = plt.subplots(figsize=(7, 5))
            legend_labels = reconstruction.get("legend", [])
            for i, curve in enumerate(reconstruction["curves"]):
                label = legend_labels[i] if i < len(legend_labels) else f"Curve {i}"
                ax.plot(curve[:, 0], curve[:, 1], label=label)
            ax.set_title(reconstruction.get("title") or "Reconstructed Plot")
            ax.set_xlabel(reconstruction.get("x_label") or "X")
            ax.set_ylabel(reconstruction.get("y_label") or "Y")
            xt = reconstruction.get("x_ticks", [])
            yt = reconstruction.get("y_ticks", [])
            if len(xt) >= 2 and len(yt) >= 2:
                xmin, xmax = min(xt), max(xt)
                ymin, ymax = min(yt), max(yt)
                x_pad = 0.05 * (xmax - xmin)
                y_pad = 0.05 * (ymax - ymin)
                ax.set_xlim(xmin - x_pad, xmax + x_pad)
                ax.set_ylim(ymin - y_pad, ymax + y_pad)
                ax.autoscale(False)
                ax.set_xticks(xt)
                ax.set_yticks(yt)
            if legend_labels:
                ax.legend()
            ax.grid(True)
            fig.tight_layout()
            # Convert to image
            fig.canvas.draw()
            recon_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            recon_img = recon_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            recon_img = recon_img[:, :, :3]  # drop alpha
            plt.close(fig)
        else:
            recon_img = None

        # --- Semantic analysis ---
        analysis_results = []
        curves = reconstruction["curves"] if reconstruction else []
        for i, curve in enumerate(curves):
            a = analyze_curve(curve)
            a["curve_index"] = i
            analysis_results.append(a)

        # --- Build output JSON ---
        output = {
            "title": ocr_data.get("title", ""),
            "x_label": ocr_data.get("x_label", ""),
            "y_label": ocr_data.get("y_label", ""),
            "x_ticks": ocr_data.get("x_ticks", []),
            "y_ticks": ocr_data.get("y_ticks", []),
            "legend": ocr_data.get("legend_labels", []),
            "num_curves_detected": len(curves),
            "semantic_analysis": analysis_results,
        }
        output_json = json.dumps(output, indent=2, default=str)

        return (
            Image.fromarray(annotated_rgb),
            Image.fromarray(recon_img) if recon_img is not None else None,
            output_json,
        )
    finally:
        os.unlink(tmp_path)


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(title="Scientific Plot Understanding", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔬 Scientific Plot Understanding
        Upload an image of a scientific plot to:
        1. **Detect layout** — locate plot area, axes, labels, legend, title
        2. **Read text** — OCR axis labels, tick values, legend entries
        3. **Segment curves** — isolate individual data series
        4. **Reconstruct data** — convert pixels back to numeric (x, y) values
        5. **Analyze semantics** — trend, periodicity, curvature interpretation
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload a scientific plot")
            run_btn = gr.Button("Analyze Plot", variant="primary", size="lg")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Layout Detection"):
                    layout_output = gr.Image(label="Detected Layout Elements")
                with gr.TabItem("Reconstructed Plot"):
                    recon_output = gr.Image(label="Reconstructed Numeric Plot")
                with gr.TabItem("Full Analysis (JSON)"):
                    json_output = gr.Textbox(
                        label="Pipeline Output", lines=25,
                        show_copy_button=True,
                    )

    gr.Markdown(
        """
        ### How it works
        | Stage | What it does |
        |-------|-------------|
        | **YOLO Layout Detection** | Finds bounding boxes for plot_area, legend, x/y labels, x/y ticks, title |
        | **EasyOCR Text Extraction** | Reads text from each cropped region |
        | **Color-based Segmentation** | Isolates individual curves by color clustering |
        | **Pixel→Data Reconstruction** | Maps pixel coordinates to data coordinates using OCR'd tick values |
        | **Semantic Analysis** | Detects trend, periodicity, curvature, and provides interpretation |
        """
    )

    run_btn.click(
        fn=run_full_pipeline,
        inputs=[input_image],
        outputs=[layout_output, recon_output, json_output],
    )

if __name__ == "__main__":
    demo.launch()
