import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from stage2_ocr import detect_layout, clamp_box, MODEL_PATH


CURVE_MASK_DIR = Path("output/curve_masks")


# -----------------------------
# LOAD CURVE MASKS (PRIMARY)
# -----------------------------
def load_curve_masks(img_path):
    stem = Path(img_path).stem
    masks = []

    for m in sorted(CURVE_MASK_DIR.glob(f"{stem}_curve_*.png")):
        mask = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            masks.append(mask)

    return masks


# -----------------------------
# FALLBACK SEGMENTATION (optional)
# -----------------------------
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


# -----------------------------
# STAGE 3
# -----------------------------
def stage3(img_path, model):

    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    boxes = detect_layout(model, img_path)

    if "plot_area" not in boxes:
        raise RuntimeError("plot_area not detected")

    plot_boxes = boxes["plot_area"]

    # merge multiple plot_area boxes if necessary
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

    # ---- PRIMARY: load generator masks ----
    components = load_curve_masks(img_path)

    # ---- FALLBACK: segmentation ----
    if len(components) == 0:
        components = segment_by_color(plot_crop)

    for i, comp in enumerate(components):
        cv2.imwrite(f"debug_curve_component_{i}.png", comp)

    print("Stage3 curves detected:", len(components))
    return components


if __name__ == "__main__":
    import sys
    stage3(sys.argv[1])
