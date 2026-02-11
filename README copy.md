#  Scientific Plot Understanding Pipeline

> An end-to-end system for understanding scientific plots  from raw images to reconstructed numeric data and semantic meaning.

The system is **modular**, **scalable**, and designed to transition smoothly from synthetic training data to real-world plots.

##  What This Project Does

Given an image of a scientific plot, the system aims to:

-  **Detect plot layout**  Find where plot elements are
-  **Read text**  Extract labels, ticks, legends via OCR
-  **Segment curves**  Identify curve pixels
-  **Reconstruct data**  Convert pixels to numeric coordinates
-  **Understand meaning**  Detect patterns and trends

---

##  Pipeline Overview

| Stage | Status | Task |
|-------|--------|------|
| **0** |  DONE | Dataset Engine  Generate training data |
| **1** |  DONE | Layout Detection (YOLO)  Detect plot elements |
| **2** |  IN PROGRESS | OCR  Extract text annotations |
| **3** |  TODO | Curve Segmentation  Identify curves |
| **4** |  TODO | Data Reconstruction  Pixel  Numbers |
| **5** |  TODO | Semantic Understanding  Interpret plots |

###  Stage 0  Dataset Engine 

**Goal:** Generate unlimited, perfectly labeled training data

Generates diverse scientific plots with:
-  Diverse plot types (line, scatter, bar, histogram)
-  Realistic styling (fonts, colors, noise)
-  Perfect labels (bounding boxes, ground truth)

**Outputs:**
- output/images/  Plot images
- output/labels/  YOLO format labels
- output/ground_truth/  Rich JSON metadata with curve coordinates

###  Stage 1  Layout Detection (YOLO) 

**Goal:** Find where plot elements are located

YOLO model detects:
- **Plot area**  The main data region
- **Legend**  Symbol explanations
- **Axis labels**  X and Y axis text
- **Tick regions**  Tick mark locations
- **Title**  Plot title

**Input:** Plot image  
**Output:** Bounding boxes for each element

###  Stage 2  OCR (Text Extraction) 

**Goal:** Read and extract text from detection regions

OCR runs only on cropped regions found by YOLO:

`json
{
  "x_label": "Time (s)",
  "y_label": "Velocity (m/s)",
  "x_ticks": [-5, 0, 5, 10],
  "y_ticks": [0, 10, 20, 30],
  "legend_labels": ["Experiment A", "Experiment B"]
}
`

**Tools used:**
- EasyOCR for text recognition
- Image preprocessing for better accuracy
- Rotation detection for vertical text

###  Stage 3  Curve Segmentation

**Goal:** Identify which pixels belong to data curves

- Produces binary mask of curve pixels
- Located within the plot area
- One mask per curve (or combined)

###  Stage 4  Data Reconstruction

**Goal:** Convert pixels to numeric coordinates

Uses:
- OCR axis scales
- Curve pixel masks
- Image-to-data coordinate mapping

**Output:** Numeric data points
`python
[(x1, y1), (x2, y2), (x3, y3), ...]
`

###  Stage 5  Semantic Understanding

**Goal:** Understand what the plot represents

Analysis includes:
- **Plot type detection**  "Sine wave", "Linear trend", etc.
- **Frequency analysis**  "Primary frequency  0.2 Hz"
- **Trend detection**  "Overall increasing", "Periodic"
- **Statistical summaries**  Mean, std dev, peaks

---

##  Repository Structure

### Core Scripts

#### generate.py 
**Stage 0  Dataset Generation**

Generates synthetic scientific plots with perfect labels

`ash
python generate.py
`

Creates:
- Diverse plot images
- YOLO-format annotations
- Rich JSON ground truth with curve data

#### uild_yolo_dataset.py 
**YOLO Dataset Builder**

Converts generated ground truth into YOLO training format

`ash
python build_yolo_dataset.py
`

Creates train/val splits:
- yolo_layout/images/train/ & images/val/
- yolo_layout/labels/train/ & labels/val/

#### stage2_ocr.py 
**Stage 2  OCR Extraction**

Runs YOLO layout detection, then extracts text

`ash
python stage2_ocr.py <image_path>
`

**Features:**
- YOLO-based region detection
- EasyOCR text extraction
- Tick parsing and deduplication
- JSON output with structured text

#### atch_stage2_test.py 
**Batch OCR Testing**

Runs Stage 2 OCR on multiple images for validation

`ash
python batch_stage2_test.py
`

### Configuration & Models

#### plots.yaml 
YOLO dataset configuration

- Dataset paths
- Class definitions (plot_area, legend, x_label, etc.)

#### est.pt 
Pre-trained YOLO model for plot layout detection

### Debugging & Validation

#### 	est.py 
Ground-truth visualization

- Renders bounding boxes from JSON
- Verifies YOLO output correctness
- Outputs debug images to debug_vis/

`ash
python test.py
`

---

##  Quick Start

### 1. Install Dependencies

`ash
pip install -r requirements.txt
`

### 2. Generate Training Data

`ash
python generate.py
`

### 3. Build YOLO Dataset

`ash
python build_yolo_dataset.py
`

### 4. Test OCR Pipeline

`ash
python batch_stage2_test.py
`

### 5. Visualize Results

`ash
python test.py
`

---

##  Dependencies

See equirements.txt for full list:

**Core Libraries:**
- **matplotlib**  Plot generation
- **numpy**  Numerical operations
- **PIL/Pillow**  Image processing
- **opencv-python (cv2)**  Computer vision
- **ultralytics (YOLO)**  Object detection
- **easyocr**  Text recognition

---

##  Project Goals

Build a fully automated system that transforms scientific plot images into:

1. **Structured numeric data**  Exact coordinates from pixels
2. **Interpretable insights**  Trends, patterns, relationships
3. **Machine-readable understanding**  JSON metadata and annotations

---

##  License

See LICENSE file

---

##  Contributing

Contributions welcome! Key areas for development:
- Stage 3: Curve segmentation models
- Stage 4: Improved coordinate mapping
- Stage 5: Semantic analysis enhancement
- General: Performance optimization, edge cases
