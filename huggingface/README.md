---
title: Scientific Plot Understanding
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
license: mit
---

# Scientific Plot Understanding

Upload a scientific plot image and get:

- **Layout detection** — YOLO-based detection of plot area, axes, labels, legend, title
- **Text extraction** — OCR reads axis labels, tick values, and legend entries
- **Curve segmentation** — individual data series isolated by color clustering
- **Data reconstruction** — pixel coordinates mapped back to numeric (x, y) values
- **Semantic analysis** — trend, periodicity, curvature, and natural-language interpretation

## Model

The YOLO model (`best.pt`) was trained on 10,000 images (5,000 synthetic + 5,000 Kaggle-sourced plots) using YOLOv8n with 7 classes:

| Class ID | Name |
|----------|------|
| 0 | plot_area |
| 1 | legend |
| 2 | x_label |
| 3 | y_label |
| 4 | x_ticks |
| 5 | y_ticks |
| 6 | title |
