from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import re
import easyocr
import cv2
import json
import sys

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
    if h > w * 1.3:   # likely vertical text
        return pil_img.rotate(-90, expand=True)
    return pil_img


# ---------------- Preprocessing ----------------
def preprocess_ticks(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    gray = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    kernel = np.ones((2,2), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
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

    # Gentle resize only
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Light denoise, NO threshold
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

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
        allowlist="Series0123456789Line",
        paragraph=False
    )

    labels = []
    for (_, text, conf) in results:
        if conf > 0.35:
            labels.append(text)

    text = " ".join(labels)
    found = re.findall(r"(Series\s*\d+|Line\s*\d+)", text, re.IGNORECASE)
    return sorted(set(found))

def ocr_ticks_sorted(pil_img, axis="y"):
    img = preprocess_ticks(pil_img)
    results = reader.readtext(
        img,
        detail=1,
        allowlist="0123456789.-",
        paragraph=False
    )

    values = []
    for (bbox, text, conf) in results:
        if conf < 0.4:
            continue

        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
        if understand := nums:
            pts = np.array(bbox)
            pos = pts[:,1].mean() if axis=="y" else pts[:,0].mean()
            values.append((pos, float(nums[0])))

    values.sort()
    cleaned = dedupe_close([v[1] for v in values])

    return cleaned

# ---------------- YOLO ----------------
def detect_layout(model, img_path):
    res = model(img_path)[0]
    boxes = {}
    for b in res.boxes:
        cls = int(b.cls.item())
        name = CLASS_NAMES.get(cls, f"class_{cls}")
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        boxes[name] = (x1, y1, x2, y2)
    return boxes

# ---------------- MAIN ----------------
def stage2(img_path):
    model = YOLO(MODEL_PATH)
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    boxes = detect_layout(model, img_path)
    plot_box = boxes.get("plot_area", None)
    final_boxes = {}
    crops = {}
    for name, box in boxes.items():
        x1, y1, x2, y2 = box

        if name == "y_ticks":
            # trim bottom to avoid x-axis numbers
            y2 = y2 - 10

            # shrink from LEFT so it doesn't grab y_label text
            x1 = x1 + int((x2 - x1) * 0.35)

            # light padding only vertically
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H, pad=4)


        elif name == "y_label":
            if plot_box is not None:
                _, py1, _, py2 = plot_box
                y1 = py1 - 10
                y2 = py2 + 10

            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H, pad=8)

        elif name == "x_label":
            x1, y1, x2, y2 = clamp_box(x1, y1 + 8, x2, y2, W, H, pad=4)

        elif name == "x_ticks":
            x1, y1, x2, y2 = clamp_box(x1, y1 - 6, x2, y2, W, H, pad=6)

        elif name == "legend":
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H, pad=14)

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

    # Rotate Y label
    if "y_label" in crops:
        y_img = maybe_rotate_vertical(crops["y_label"])
        y_img.save("debug_y_label_rotated.png")
        results["y_label"] = ocr_label(y_img)
    else:
        results["y_label"] = ""

    results["x_label"] = ocr_label(crops["x_label"]) if "x_label" in crops else ""
    results["title"]   = ocr_label(crops["title"]) if "title" in crops else ""

    results["x_ticks"] = enforce_zero_tick(
    ocr_ticks_sorted(crops["x_ticks"], axis="x")
)

    raw_y = ocr_ticks_sorted(crops["y_ticks"], axis="y")
    raw_y = enforce_zero_tick(raw_y)
    results["y_ticks"] = remove_outlier_ticks(raw_y)



    results["legend_labels"] = ocr_legend(crops["legend"]) if "legend" in crops else []

    return results

# ---------------- RUN ----------------
if __name__ == "__main__":
    img_path = sys.argv[1]
    result = stage2(img_path)
    print(json.dumps(result, indent=2))
