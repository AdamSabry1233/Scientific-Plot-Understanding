Scientific Plot Understanding Pipeline

This repository builds an end-to-end system for understanding scientific plots — from raw images to reconstructed numeric data and semantic meaning.

The system is modular, scalable, and designed to transition smoothly from synthetic training data to real-world plots.

What This Project Does

Given an image of a scientific plot, the system aims to:

Detect plot layout (where things are)

Read text (labels, ticks, legends)

Segment curves

Reconstruct numeric data

Understand the plot’s meaning

Pipeline Overview
Stage 0 — Dataset Engine  DONE

Goal: Generate unlimited, perfectly labeled training data

Produces:

Diverse scientific plot images

Pixel-accurate bounding boxes

YOLO labels

Curve ground truth

Layout + text metadata

Output:

output/images/
output/labels/
output/ground_truth/   (rich JSON)


This stage feeds all future models.

Stage 1 — Layout Detection (YOLO) DONE

Goal: Find where plot elements are

YOLO detects:

Plot area

Legend

Axis-related regions

Input: Plot image
Output: Bounding boxes for layout elements

Stage 2 — OCR (Text Extraction) IN PROGRESS

Goal: Read plot text

OCR runs only on cropped regions:

X-axis label

Y-axis label

Tick values

Legend labels

Example output:

"x_label": "Time (s)"
"y_label": "Velocity"
"x_ticks": [-5, 0, 5, 10]
"legend_labels": ["Line 0", "Line 1"]

Stage 3 — Curve Segmentation

Goal: Identify curve pixels

A segmentation model outputs a binary mask of curve pixels inside the plot area.

Stage 4 — Pixel → Data Reconstruction

Goal: Convert pixels into numbers

Uses:

OCR axis scales

Curve mask

Coordinate mapping

Output:

[(x1, y1), (x2, y2), ...]

Stage 5 — Semantic Understanding

Goal: Understand what the plot represents

Adds:

Plot type detection

Trend analysis

Regression / classification

Example outputs:

“Sine wave signal”

“Frequency ≈ 0.2 Hz”

“Overall increasing trend”

Final System Flow
Image
  ↓
YOLO (layout)
  ↓
OCR (text)
  ↓
Curve Segmentation
  ↓
Data Reconstruction
  ↓
Semantic Analysis

Repository Files Explained
generate.py

Stage 0 – Dataset Engine

Generates diverse scientific plots (line, scatter, bar, hist, etc.)

Produces:

Images

YOLO label files

Rich JSON ground truth

Supports real-world variability:

Fonts, styles, noise, grids, scales, legends, titles

build_yolo_dataset.py

YOLO dataset builder

Converts generated ground truth into YOLO training format

Splits data into:

images/train
images/val
labels/train
labels/val

plots.yaml

YOLO dataset configuration

Defines dataset paths

Defines class IDs for layout detection

best.pt

Trained YOLO model

Detects plot layout elements (Stage 1)

stage2_ocr.py

Stage 2 – OCR extraction

Runs YOLO to detect layout

Crops relevant regions

Applies OCR only where needed

Outputs structured text data

batch_stage2_test.py

Batch OCR testing

Runs Stage 2 OCR on multiple images

Used for validation and debugging

test.py

Ground-truth visualization / debugging

Visualizes bounding boxes

Used to verify correctness of generated labels and YOLO outputs

Project Goal

Build a fully automated system that turns scientific plot images into:

📈 Structured numeric data

🧠 Interpretable insights

🤖 Machine-readable understanding
