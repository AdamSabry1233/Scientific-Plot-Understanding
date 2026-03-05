# Scientific Plot Understanding Pipeline

An end-to-end system that takes a scientific plot image and produces structured numeric data, readable text, and semantic insights — fully automated, from pixels to understanding.

## What It Does

Given any scientific plot image, the pipeline:

1. **Detects layout** — locates the plot area, axes, legend, title
2. **Reads text** — extracts axis labels, tick values, legend entries, title via OCR
3. **Segments curves** — isolates individual curves as binary masks
4. **Reconstructs data** — maps curve pixels back to numeric (x, y) coordinates
5. **Analyzes semantics** — identifies trends, periodicity, and signal types

## Pipeline Stages

| Stage | Name | Script | Status |
|-------|------|--------|--------|
| 0 | Dataset Engine | `generate.py`, `generate_synthetic.py`, `generate_kagglehub.py` | Done |
| 0.5 | YOLO Dataset Builder | `build_yolo_dataset.py` | Done |
| 1 | Layout Detection (YOLO) | `best.pt` model | Done |
| 2 | OCR Text Extraction | `stage2_ocr.py` | Done |
| 3 | Curve Segmentation | `stage3_segment.py` | Done |
| 4 | Data Reconstruction | `stage4_reconstruct.py` | Done |
| 5 | Semantic Analysis | `semantic_analysis.py` | Done |

## System Flow

```
Image
  │
  ▼
YOLO Layout Detection (best.pt)
  │  plot_area, legend, x_label, y_label, x_ticks, y_ticks, title
  ▼
OCR Text Extraction (EasyOCR)
  │  axis labels, tick values, legend text, title
  ▼
Curve Segmentation
  │  binary masks per curve (pre-generated or color-based fallback)
  ▼
Data Reconstruction
  │  pixel coords → numeric (x, y) pairs using OCR axis scales
  ▼
Semantic Analysis
  │  trend detection, periodicity, sine-like pattern recognition
  ▼
Structured Output (JSON)
```

## Data Generation (Stage 0)

Three generators produce diverse training data, each outputting `images/`, `labels/`, `ground_truth/`, and `curve_masks/`:

| Generator | Output Dir | Data Source |
|-----------|-----------|-------------|
| `generate.py` | `output/` | Synthetic math functions (sin, cos, tanh, linear, x², exp, log) |
| `generate_synthetic.py` | `synthetic_output/` | Extended synthetic variant with more labels and titles |
| `generate_kagglehub.py` | `kaggle_output/` | Real-world Kaggle datasets (Titanic, Iris, Melbourne Housing, Retail) |

Generated data includes:
- Plot images with varied fonts, styles, grids, noise, markers, and legend positions
- YOLO-format bounding box labels (7 classes)
- Rich JSON ground truth with axis ranges, legend info, and precomputed YOLO coordinates
- Per-curve binary masks for segmentation ground truth

`build_yolo_dataset.py` converts the ground truth JSON into YOLO training format with an 80/20 train/val split.

## OCR Pipeline (Stage 2)

`stage2_ocr.py` crops YOLO-detected regions and applies targeted preprocessing before running EasyOCR:

- **Tick values** — 3x upscale, histogram equalization, non-local means denoising, adaptive threshold
- **Axis labels** — 2.5x upscale, Gaussian blur, binary threshold
- **Label cleaning** — fuzzy matching against 40+ known scientific labels
- **Tick regularization** — outlier removal, spacing consistency checks, zero-tick enforcement

Output example:
```json
{
  "x_label": "Time (s)",
  "y_label": "Velocity",
  "x_ticks": [0, 5, 10, 15],
  "y_ticks": [-2, 0, 2],
  "legend_labels": ["Line 0", "Line 1"],
  "title": "Velocity Profile"
}
```

## Curve Segmentation (Stage 3)

`stage3_segment.py` identifies curve pixels using two methods:
1. **Pre-generated masks** — loads `curve_masks/[stem]_curve_*.png` files (preferred)
2. **Color-based fallback** — quantizes colors and clusters into connected components

Returns a list of binary masks, one per detected curve.

## Data Reconstruction (Stage 4)

`stage4_reconstruct.py` converts pixel coordinates to real data values:
- Extracts pixel traces from curve masks (median y per x-column)
- Maps pixels to data coordinates using OCR-extracted axis scales
- Matches curves to legend entries by size ordering

## Semantic Analysis (Stage 5)

`semantic_analysis.py` interprets each curve:
- **Trend detection** — linear regression on smoothed signal (increasing / flat / decreasing)
- **Periodicity** — peak detection via `scipy.signal.find_peaks`, computes frequency
- **Sine-like recognition** — validates consistent peak spacing, amplitude, and zero crossings

## Project Structure

```
├── generate.py                 # Synthetic plot generation (math functions)
├── generate_synthetic.py       # Extended synthetic variant
├── generate_kagglehub.py       # Real-world Kaggle data plots
├── build_yolo_dataset.py       # Ground truth → YOLO format converter
│
├── stage2_ocr.py               # OCR text extraction pipeline
├── stage3_segment.py           # Curve segmentation
├── stage4_reconstruct.py       # Pixel → data reconstruction
├── semantic_analysis.py        # Signal analysis and interpretation
│
├── test.py                     # YOLO bounding box visualization
├── batch_stage2_test.py        # Batch OCR validation with problem flagging
├── dependency_test.py          # Package import verification
│
├── best.pt                     # Trained YOLO layout detection model
├── yolov8n.pt                  # YOLOv8 nano base weights
│
├── plots.yaml                  # Base YOLO dataset config
├── yolo_original.yaml          # Original synthetic dataset config
├── yolo_synthetic.yaml         # Extended synthetic dataset config
├── yolo_kaggle.yaml            # Kaggle dataset config
├── yolo_combined.yaml          # Combined (all sources) config
├── yolo_large_10k.yaml         # Large-scale 10k variant config
│
├── Scientific_Plot_Understanding_Complete.ipynb   # Full pipeline notebook
├── semantic_analysis_comparison.ipynb             # Semantic vs baseline comparison
│
├── scientific-plot-reader/     # Gradio web app (HuggingFace Spaces)
│   └── app.py
│
├── output/                     # Original synthetic dataset
├── synthetic_output/           # Extended synthetic dataset
├── kaggle_output/              # Kaggle-sourced dataset
└── runs/                       # YOLO training checkpoints
```

## YOLO Classes

All dataset configs use the same 7 layout classes:

| ID | Class |
|----|-------|
| 0 | plot_area |
| 1 | legend |
| 2 | x_label |
| 3 | y_label |
| 4 | x_ticks |
| 5 | y_ticks |
| 6 | title |

## Web App

The `scientific-plot-reader/` directory contains a Gradio app deployed to HuggingFace Spaces. Upload a plot image and get the full pipeline output — layout boxes, OCR text, reconstructed curves, and semantic analysis — in one interactive interface.

## Requirements

```
matplotlib>=3.5.0
numpy>=1.21.0
Pillow>=9.0.0
opencv-python>=4.5.0
ultralytics>=8.0.0
torch>=2.0.0
easyocr>=1.7.0
scikit-image>=0.19.0
PyYAML>=5.4.0
tqdm>=4.60.0
```

Install:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from stage4_reconstruct import run_pipeline

result = run_pipeline("path/to/plot.png")
# result contains: title, axis labels, ticks, legend, curves (numeric data)
```

Or use the Gradio web app:
```bash
cd scientific-plot-reader
python app.py
```
