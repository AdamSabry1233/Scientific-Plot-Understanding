import json
import cv2
from pathlib import Path

IMG_DIR = Path("output/images")
GT_DIR  = Path("output/ground_truth")
OUT_DIR = Path("debug_vis")
OUT_DIR.mkdir(exist_ok=True)

CLASSES = {
    "plot_area": (0, 255, 0),
    "legend": (0, 0, 255),
    "x_label": (255, 0, 0),
    "y_label": (0, 255, 255),
    "x_ticks": (255, 255, 0),
    "y_ticks": (255, 0, 255),
    "title": (255, 128, 0)   # orange
}


def draw_yolo_box(img, bbox, color, label):
    h, w = img.shape[:2]
    xc, yc, bw, bh = bbox

    x1 = int((xc - bw/2) * w)
    y1 = int((yc - bh/2) * h)
    x2 = int((xc + bw/2) * w)
    y2 = int((yc + bh/2) * h)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img, label,
        (x1 + 5, y1 + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, color, 2
    )

for gt_file in sorted(GT_DIR.glob("*.json"))[:10]:
    with open(gt_file) as f:
        gt = json.load(f)

    if "boxes_yolo" not in gt:
        print(f"Skipping {gt_file.name} (no boxes_yolo)")
        continue

    img_path = IMG_DIR / gt["image"]
    img = cv2.imread(str(img_path))

    boxes = gt["boxes_yolo"]

    for name, color in CLASSES.items():
        if name in boxes and boxes[name] is not None:
            draw_yolo_box(img, boxes[name], color, name)


    out_path = OUT_DIR / f"{gt_file.stem}_viz.png"
    cv2.imwrite(str(out_path), img)
    print(f"Saved {out_path}")

print("Done. Check debug_vis folder.")
