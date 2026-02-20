import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

from stage2_ocr import detect_layout, clamp_box, MODEL_PATH, stage2
from stage3_segment import stage3
from semantic_analysis import analyze_curve

model = YOLO(MODEL_PATH)


# -----------------------------
# MASK → TRACE
# -----------------------------
def mask_to_trace(mask):
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    ys, xs = np.where(mask > 0)

    col_map = {}
    for y, x in zip(ys, xs):
        col_map.setdefault(x, []).append(y)

    x_obs = np.array(sorted(col_map.keys()), dtype=np.float32)
    y_obs = np.array([np.median(col_map[int(x)]) for x in x_obs], dtype=np.float32)

    # Fill missing columns using interpolation
   # x_full = np.arange(int(x_obs.min()), int(x_obs.max()) + 1, dtype=np.float32)
   # y_full = np.interp(x_full, x_obs, y_obs)

    # Light smoothing
   # y_full = cv2.GaussianBlur(y_full.reshape(-1,1), (1,5), 0).reshape(-1)

    # do NOT densify/blur yet (debug stable trace first)
    return np.column_stack([x_obs, y_obs]).astype(np.float32)

# -----------------------------
# NORMALIZATION
# -----------------------------



# -----------------------------
# AXIS RANGES
# -----------------------------
def get_axis_ranges(img_path):
    meta = stage2(img_path)

    xt = meta.get("x_ticks", [])
    yt = meta.get("y_ticks", [])

    if len(xt) >= 2 and len(yt) >= 2:
        xmin, xmax = min(xt), max(xt)
        ymin, ymax = min(yt), max(yt)
        return xmin, xmax, ymin, ymax

    raise RuntimeError("Axis ranges unavailable")


# -----------------------------
# STAGE 4
# -----------------------------
def stage4(img_path, components, model):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    boxes = detect_layout(model, img_path)

    if "plot_area" not in boxes:
        raise RuntimeError("plot_area not detected")

    plot_boxes = boxes["plot_area"]

    # Merge if multiple detections
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


    if components is None or len(components) == 0:
        return None


    xmin, xmax, ymin, ymax = get_axis_ranges(img_path)

    curves = []

    for comp in components:
        # comp is full canvas mask
        # crop it exactly like the image crop
        comp_crop = comp[y1:y2, x1:x2]

        trace_px = mask_to_trace(comp_crop)

        if trace_px.shape[0] < 5:
            continue

        # Direct linear pixel → data mapping (CORRECT)

        data_x = xmin + (trace_px[:, 0] / crop_w) * (xmax - xmin)

        # Flip Y because image origin is top-left
        data_y = ymax - (trace_px[:, 1] / crop_h) * (ymax - ymin)

        data_xy = np.column_stack((data_x, data_y)).astype(np.float32)

        curves.append(data_xy)

    # ---- FIX: define meta BEFORE using it ----
    meta = stage2(img_path)
    legend_labels = meta.get("legend_labels", [])

    if legend_labels:
        curves = sorted(curves, key=lambda c: len(c), reverse=True)
        curves = curves[:len(legend_labels)]


    reconstruction = {
        "title": meta.get("title", ""),
        "x_label": meta.get("x_label", ""),
        "y_label": meta.get("y_label", ""),
        "x_ticks": meta.get("x_ticks", []),
        "y_ticks": meta.get("y_ticks", []),
        "legend": meta.get("legend_labels", []),
        "curves": curves
    }

    print("Stage4 reconstructed curves:", len(curves))
    return reconstruction


# -----------------------------
# RUN PIPELINE
# -----------------------------
def run_pipeline(img_path):
    components = stage3(img_path, model)          # if stage3 expects model
    result = stage4(img_path, components, model)  # pass model here


    if result is None:
        print("No reconstruction available.")
        return

    curves = result["curves"]

    print("\n===== ANALYSIS =====")

    for i, curve in enumerate(curves):
        analysis = analyze_curve(curve)
        print(f"\nCurve {i}:")
        for k, v in analysis.items():
            print(f"{k}: {v}")

    plt.figure(figsize=(6, 4))

    legend_labels = result.get("legend", [])

    for i, curve in enumerate(curves):
        if i < len(legend_labels):
            plt.plot(curve[:, 0], curve[:, 1], label=legend_labels[i])
        else:
            plt.plot(curve[:, 0], curve[:, 1], label=f"Curve {i}")

    plt.title(result["title"] or "Reconstructed Plot")
    plt.xlabel(result["x_label"] or "X")
    plt.ylabel(result["y_label"] or "Y")

    ax = plt.gca()

    # Set limits FIRST (from OCR)
    xmin = min(result["x_ticks"])
    xmax = max(result["x_ticks"])
    ymin = min(result["y_ticks"])
    ymax = max(result["y_ticks"])

        # Set limits with small padding
    x_range = xmax - xmin
    y_range = ymax - ymin

    x_pad = 0.05 * x_range
    y_pad = 0.05 * y_range

    ax.set_xlim(xmin - x_pad, xmax + x_pad)
    ax.set_ylim(ymin - y_pad, ymax + y_pad)

    # Disable autoscale AFTER limits are set
    ax.autoscale(False)

    # Apply OCR ticks exactly
    ax.set_xticks(result["x_ticks"])
    ax.set_yticks(result["y_ticks"])


    if legend_labels:
        plt.legend()

    plt.grid(True)
    plt.show()



# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python stage4_reconstruct.py <image_path>")
        exit()

    run_pipeline(sys.argv[1])
