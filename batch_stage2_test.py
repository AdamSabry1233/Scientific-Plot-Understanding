import json
from pathlib import Path
from stage2_ocr import stage2   # import your function
import traceback

IMG_DIR = Path("output/images")
OUT_FILE = Path("batch_results.jsonl")

results = []
failures = []

images = sorted(IMG_DIR.glob("*.png"))

print(f"Running OCR on {len(images)} images...\n")

x_ticks_ok = 0
y_ticks_ok = 0
x_label_ok = 0
y_label_ok = 0

for i, img_path in enumerate(images, 1):
    try:
        result = stage2(str(img_path))

        record = {
            "image": img_path.name,
            "result": result
        }

        problems = []

        if len(result["x_ticks"]) < 2:
            problems.append("x_ticks_low")
        else:
            x_ticks_ok += 1

        if len(result["y_ticks"]) < 2:
            problems.append("y_ticks_low")
        else:
            y_ticks_ok += 1

        if result["x_label"] == "":
            problems.append("missing_x_label")
        else:
            x_label_ok += 1

        if result["y_label"] == "":
            problems.append("missing_y_label")
        else:
            y_label_ok += 1

        record["problems"] = problems

        if problems:
            failures.append(record)

        results.append(record)

        print(f"[{i}/{len(images)}] {img_path.name} problems={problems}")

    except Exception as e:
        print(f"[{i}/{len(images)}] ERROR on {img_path.name}")
        traceback.print_exc()
        failures.append({
            "image": img_path.name,
            "error": str(e)
        })


# write results
with open(OUT_FILE, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")


total = len(images)

print("\n========== SUMMARY ==========")
print(f"Total images: {total}")
print(f"Failures: {len(failures)}")
print(f"x_label success: {x_label_ok}/{total}")
print(f"y_label success: {y_label_ok}/{total}")
print(f"x_ticks success: {x_ticks_ok}/{total}")
print(f"y_ticks success: {y_ticks_ok}/{total}")
print(f"Saved results to {OUT_FILE}")
