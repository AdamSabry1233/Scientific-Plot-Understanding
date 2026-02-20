from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import re
import easyocr
import cv2
import json
import sys
from difflib import get_close_matches


MODEL_PATH = "best.pt"
reader = easyocr.Reader(["en"], gpu=False)

CLASS_NAMES = {
    0: "plot_area",
    1: "legend",
    2: "x_label",
    3: "y_label",
    4: "x_ticks",
    5: "y_ticks",
    6: "title"
}

# ---------------- Utils ----------------

def clamp_box(x1, y1, x2, y2, W, H, pad=10):
    x1 = max(0, int(x1 - pad))
    y1 = max(0, int(y1 - pad))
    x2 = min(W, int(x2 + pad))
    y2 = min(H, int(y2 + pad))
    return x1, y1, x2, y2

def dedupe_close(nums, tol=1.5):
    nums = sorted(nums)
    out = []
    for n in nums:
        if not out or abs(n - out[-1]) > tol:
            out.append(n)
    return out

def maybe_rotate_vertical(pil_img):
    w, h = pil_img.size
    if h > w :   # likely vertical text
        return pil_img.rotate(-90, expand=True)
    return pil_img

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
    import re
    lower = text.lower()

    if re.search(r'\bsin\b', lower):
        return "sin(x)"

    if re.search(r'\bcos\b', lower):
        return "cos(x)"

    if re.search(r'\btan\b', lower):
        return "tan(x)"

    if re.search(r'\blog\b', lower):
        return "log(x)"

    if re.search(r'\bexp\b', lower):
        return "exp(x)"



    corrections = {
    # Old Faithful dataset fixes
    "Betwen": "Between",
    "Betweem": "Between",
    "Enupion": "Eruption",
    "Enuption": "Eruption",
    "Eruptlons": "Eruptions",
    "Etupilons": "Eruptions",
    "Falthful": "Faithful",
    "Falhful": "Faithful",

    # unit confusion
    "Hln": "Min",
    "lnj": "Min",
    "MlN": "Min",

    # spacing OCR mistakes
    "BetwenEruptions": "Between Eruptions",

    # previous ones you already had
    "Wlaling": "Waiting",
    "Tine": "Time",
    "Duratilon": "Duration",
    "Velocitv": "Velocity",
    "Velocily": "Velocity",
    "Powerer": "Power",
    "Distarce": "Distance",
    "Acceleraion": "Acceleration"
}


    for wrong, right in corrections.items():
        text = text.replace(wrong, right)

    # split merged words like BetwenEruptions → Betwen Eruptions
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)


    # collapse duplicate characters
    text = re.sub(r'(.)\1{1,}', r'\1', text)

    # bracket fixes
    text = text.replace("))", ")")
    text = text.replace("((", "(")

    # normalize spacing
    text = " ".join(text.split())


    # -----------------------------
    # FUZZY MATCHING (NEW PART)
    # -----------------------------
    KNOWN_LABELS = [
    # Time / sampling
    "Time (s)",
    "Time (ms)",
    "Time (min)",
    "Samples",

    # Distance / motion
    "Distance (m)",
    "Position (m)",
    "Displacement (m)",
    "Velocity",
    "Velocity (m/s)",
    "Speed (m/s)",
    "Acceleration (m/s^2)",

    # Signal processing
    "Amplitude",
    "Signal",
    "Signal Amplitude",
    "Frequency (Hz)",
    "Frequency (kHz)",
    "Phase",

    # Electrical
    "Voltage (V)",
    "Current (A)",
    "Resistance (Ohm)",
    "Power (W)",

    # Physics / engineering
    "Energy (J)",
    "Force (N)",
    "Mass (kg)",
    "Pressure (kPa)",
    "Temperature (C)",

    # Optics
    "Wavelength (nm)",
    "Wavelength (um)",
    "Intensity",

    # Generic plotting labels
    "Input",
    "Output",
    "Response",
    "Measurement",
    "Value",
    "Sample Data",

    # Math functions (VERY IMPORTANT)
    "sin(x)",
    "cos(x)",
    "tan(x)",
    "exp(x)",
    "log(x)",
    "ln(x)",
    "x^2",
    "x^3",
    "Polynomial",
    "Linear",
    "Quadratic",
    "Cubic",
    "Model Output",
    "Prediction",
    "Ground Truth",
    "Error",
    "Eruptions",
]


    match = get_close_matches(text, KNOWN_LABELS, n=1, cutoff=0.6)
    if match:
        text = match[0]

    return text.strip()



# ---------------- Preprocessing ----------------
def preprocess_ticks(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    gray = cv2.equalizeHist(gray)

    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    gray = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        3
    )

    return gray



def preprocess_label(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    return gray

def preprocess_legend(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    return gray

def regularize_ticks(ticks):
    if len(ticks) < 3:
        return ticks

    ticks = sorted(ticks)

    diffs = np.diff(ticks)
    step = np.median(diffs)

    if step == 0:
        return ticks

    start = ticks[0]
    end = ticks[-1]

    fixed = []
    v = start
    while v <= end + step * 0.5:
        fixed.append(round(v, 2))
        v += step

    return fixed



def enforce_zero_tick(ticks):
    if not ticks:
        return ticks

    has_neg = any(t < 0 for t in ticks)
    has_pos = any(t > 0 for t in ticks)

    if has_neg and has_pos and 0.0 not in ticks:
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



# ---------------- OCR ----------------
def ocr_label(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    # NEW — improves real plot text significantly
    gray = cv2.equalizeHist(gray)

    gray = cv2.fastNlMeansDenoising(gray, None, 12, 7, 21)

    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    gray = cv2.filter2D(gray, -1, kernel)


    results = reader.readtext(
        gray,
        detail=0,
        paragraph=True,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789() /-"
    )

    return " ".join(results).strip()



def ocr_legend(pil_img):
    img = preprocess_legend(pil_img)

    results = reader.readtext(
        img,
        detail=1,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ",
        paragraph=False
    )

    rows = []

    for (bbox, text, conf) in results:
        if conf < 0.4:
            continue

        pts = np.array(bbox)
        y_center = pts[:,1].mean()

        rows.append((y_center, text.strip()))

    # sort by vertical position
    rows.sort(key=lambda x: x[0])

    labels = []

    for _, text in rows:
        match = re.search(r"Line\s*\d+", text)
        if match:
            labels.append(match.group())
        else:
            labels.append(text)

    # remove duplicates but preserve order
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

    # Debug the exact image we OCR
    if axis == "y":
        cv2.imwrite("debug_y_ticks_rotated.png", img)
    else:
        cv2.imwrite("debug_x_ticks_bin.png", img)

    # -------------------------
    # MODE A: old behavior (works great for x_ticks)
    # -------------------------
    if mode == "easyocr":
        results = reader.readtext(
        img,
        detail=1,
        allowlist="0123456789.-",
        paragraph=False,
        text_threshold=0.4,
        low_text=0.2,
        link_threshold=0.3
)

        values = []
        for (bbox, text, conf) in results:
            if conf < 0.35:
                continue
            nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
            if not nums:
                continue

            pts = np.array(bbox)
            pos = pts[:,0].mean() if axis == "x" else pts[:,1].mean()
            values.append((pos, float(nums[0])))

        values.sort(key=lambda x: x[0])
        return dedupe_close([v[1] for v in values], tol=0.2)

    # -------------------------
    # -------------------------
    # MODE B: contour splitting (use for y_ticks)
    # -------------------------

    bin_img = cv2.bitwise_not(img)

    # Find contours
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Keep digit-like and minus-like contours
    components = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Keep anything reasonably tall (digits + minus fragments)
        if h > 12 and w > 3:
            components.append((x, y, w, h))


    # y-ticks are stacked vertically -> sort by y
    if axis == "y":
        components.sort(key=lambda b: b[1])
    else:
        components.sort(key=lambda b: b[0])

    # ---- group components into full tick labels ----
    boxes = []
    current = None

    # how far apart two separate tick labels are (in pixels)
    # (your crop is huge, so 20–35 is reasonable)
    GAP_Y = 35
    GAP_X = 25

    for (x, y, w, h) in components:
        if current is None:
            current = [x, y, x + w, y + h]
            continue

        if axis == "y":
            # new label when we're far below the previous label
            gap = y - current[3]
            if gap <= GAP_Y:
                # same label: expand bounds
                current[0] = min(current[0], x)
                current[1] = min(current[1], y)
                current[2] = max(current[2], x + w)
                current[3] = max(current[3], y + h)
            else:
                # finalize old label, start new
                boxes.append(tuple(current))
                current = [x, y, x + w, y + h]
        else:
            gap = x - current[2]
            if gap <= GAP_X:
                current[0] = min(current[0], x)
                current[1] = min(current[1], y)
                current[2] = max(current[2], x + w)
                current[3] = max(current[3], y + h)
            else:
                boxes.append(tuple(current))
                current = [x, y, x + w, y + h]

    if current is not None:
        boxes.append(tuple(current))

    # OCR each grouped label box
    values = []
    for (x1, y1, x2, y2) in boxes:
        crop = bin_img[y1:y2, x1:x2]

        crop = cv2.copyMakeBorder(
            crop, 8, 8, 8, 8,
            cv2.BORDER_CONSTANT,
            value=0
        )

        txt = reader.readtext(
            crop,
            detail=0,
            allowlist="0123456789.-"
        )

        if not txt:
            continue

        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", txt[0])
        if nums:
            values.append(float(nums[0]))

    # Y tick order should be bottom->top or top->bottom? (you can choose)
    # If you want numeric sort:
    values = sorted(values)
    # If ticks look symmetric but missing negative start, infer it
    if len(values) >= 3:
        step = np.median(np.diff(values))
        min_val = values[0]
        if min_val == 0 and step > 0:
            possible_neg = min_val - step
            values = [possible_neg] + values


    return dedupe_close(values, tol=0.2)






# ---------------- YOLO ----------------
def detect_layout(model, img_path):
    res = model(img_path)[0]
    boxes = {}

    for b in res.boxes:
        cls = int(b.cls.item())
        name = CLASS_NAMES.get(cls, f"class_{cls}")
        x1, y1, x2, y2 = b.xyxy[0].tolist()

        if name not in boxes:
            boxes[name] = []

        boxes[name].append((x1, y1, x2, y2))

    return boxes


# ---------------- MAIN ----------------
def stage2(img_path):
    model = YOLO(MODEL_PATH)
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    boxes = detect_layout(model, img_path)
    plot_box_list = boxes.get("plot_area", None)

    if plot_box_list:
        # usually only 1 plot_area, but merge just in case
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

        # Merge multiple boxes for tick regions
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
            # trim bottom to avoid x-axis numbers
            y2 = y2 - 10

            # shrink from LEFT so it doesn't grab y_label text
            x1 = x1 + int((x2 - x1) * 0.35)

            # NEW — expand tick region slightly
            x1 -= 6
            y1 -= 6
            x2 += 6
            y2 += 6

            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H, pad=2)


        elif name == "y_label":
            if plot_box is not None:
                _, py1, _, py2 = plot_box
                y1 = py1 - 10
                y2 = py2 + 10

            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H, pad=8)

        elif name == "title":
            # Expand upward significantly
            y1 -= int((y2 - y1) * 0.8)
            y2 += 10

            x1 -= 10
            x2 += 10

            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H, pad=4)



        elif name == "x_label":
            x1, y1, x2, y2 = clamp_box(x1, y1 + 8, x2, y2, W, H, pad=4)

        elif name == "x_ticks":
            x1 -= 15
            x2 += 15
            y1 -= 5
            y2 += 5

            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H, pad=4)




        elif name == "legend":
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H, pad=4)



        else:
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H, pad=6)
        final_boxes[name] = (x1, y1, x2, y2)
        crops[name] = img.crop((x1, y1, x2, y2))



    # ---------- DEBUG ----------
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)

    for name, (x1, y1, x2, y2) in final_boxes.items():
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=2)
        draw.text((x1+4, y1+4), name, fill="lime")

    overlay.save("debug_boxes_overlay.png")

    for k, crop in crops.items():
        crop.save(f"debug_{k}.png")

    results = {}

    # ---------------- LABEL OCR ----------------
    if "y_label" in crops:
        y_img = maybe_rotate_vertical(crops["y_label"])
        y_img.save("debug_y_label_rotated.png")
        results["y_label"] = clean_label(ocr_label(y_img))
    else:
        results["y_label"] = ""

    results["x_label"] = clean_label(ocr_label(crops["x_label"])) if "x_label" in crops else ""
    results["title"]   = clean_label(ocr_label(crops["title"])) if "title" in crops else ""

 
        # ---------------- X TICKS ----------------
    if "x_ticks" in crops:
        xt = ocr_ticks_sorted(crops["x_ticks"], axis="x", mode="easyocr")
        results["x_ticks"] = xt
    else:
        results["x_ticks"] = []

    # ---------------- Y TICKS ----------------
    if "y_ticks" in crops:
        yt = ocr_ticks_sorted(crops["y_ticks"], axis="y", mode="contours")
        results["y_ticks"] = yt
    else:
        results["y_ticks"] = []



    # ---------------- LEGEND ----------------
    if "legend" in crops:
        labels = ocr_legend(crops["legend"])

        # --- Auto-fill missing Line numbers ---
        line_nums = []

        for l in labels:
            m = re.search(r"Line\s*(\d+)", l)
            if m:
                line_nums.append(int(m.group(1)))

        if len(line_nums) >= 2:
            min_n = min(line_nums)
            max_n = max(line_nums)

            full_range = list(range(min_n, max_n + 1))
            labels = [f"Line {i}" for i in full_range]

        results["legend_labels"] = labels
    else:
        results["legend_labels"] = []

    return results

# ---------------- RUN ----------------
if __name__ == "__main__":
    img_path = sys.argv[1]
    result = stage2(img_path)
    print(json.dumps(result, indent=2))
