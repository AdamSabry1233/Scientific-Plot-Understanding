import json
import random
from pathlib import Path
import shutil

SRC_IMG = Path("output/images")
SRC_GT  = Path("output/ground_truth")

DST = Path("yolo_layout")
IMG_TRAIN = DST / "images/train"
IMG_VAL   = DST / "images/val"
LAB_TRAIN = DST / "labels/train"
LAB_VAL   = DST / "labels/val"

for p in [IMG_TRAIN, IMG_VAL, LAB_TRAIN, LAB_VAL]:
    p.mkdir(parents=True, exist_ok=True)

# must match data.yaml
CLASSES = {
    "plot_area": 0,
    "legend": 1,
    "x_label": 2,
    "y_label": 3,
    "x_ticks": 4,
    "y_ticks": 5,
    "title": 6
}

def write_label_file(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(" ".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) + "\n")

def main(val_ratio=0.2, seed=42):
    random.seed(seed)

    gt_files = list(SRC_GT.glob("*.json"))
    random.shuffle(gt_files)

    n_val = int(len(gt_files) * val_ratio)
    val_set = set(gt_files[:n_val])

    for gt_path in gt_files:
        gt = json.loads(gt_path.read_text())

        # NEW SCHEMA
        if "boxes_yolo" not in gt:
            print(f"Skipping {gt_path.name} (no boxes_yolo)")
            continue

        boxes = gt["boxes_yolo"]

        img_name = gt["image"]
        img_src = SRC_IMG / img_name

        rows = []

        for name, class_id in CLASSES.items():
            if name in boxes and boxes[name] is not None:
                cx, cy, w, h = boxes[name]

                # safety clamp
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w  = max(0.0, min(1.0, w))
                h  = max(0.0, min(1.0, h))

                rows.append([class_id, cx, cy, w, h])

        if not rows:
            print(f"No labels in {gt_path.name}")
            continue

        is_val = gt_path in val_set
        img_dst = (IMG_VAL if is_val else IMG_TRAIN) / img_name
        lab_dst = (LAB_VAL if is_val else LAB_TRAIN) / (Path(img_name).stem + ".txt")

        shutil.copy2(img_src, img_dst)
        write_label_file(lab_dst, rows)

    print("YOLO dataset built")
    print("Train:", len(list(IMG_TRAIN.glob("*.png"))))
    print("Val:  ", len(list(IMG_VAL.glob("*.png"))))

if __name__ == "__main__":
    main()
