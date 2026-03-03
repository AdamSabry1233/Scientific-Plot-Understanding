import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator, FormatStrFormatter
import numpy as np
import json
import random
import pandas as pd
from pathlib import Path
import cv2
import kagglehub

OUT_IMG = Path("kaggle_output/images")
OUT_LBL = Path("kaggle_output/labels")
OUT_JSON = Path("kaggle_output/ground_truth")
OUT_MASK = Path("kaggle_output/curve_masks")

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)
OUT_JSON.mkdir(parents=True, exist_ok=True)
OUT_MASK.mkdir(parents=True, exist_ok=True)

CLASSES = {
    "plot_area": 0,
    "legend": 1,
    "x_label": 2,
    "y_label": 3,
    "x_ticks": 4,
    "y_ticks": 5,
    "title": 6
}

PLOT_TITLES = [
    "Data Analysis", "Trend Analysis", "Historical Data", "Performance Review",
    "Statistical Summary", "Data Visualization", "Comparative Analysis", 
    "Time Series Plot", "Distribution Analysis", ""
]

LEG_LOCS = ["upper left", "upper right", "lower left", "lower right", "best"]

# Popular Kaggle datasets to download
KAGGLE_DATASETS = [
    "wordsforthewise/titanic",
    "uciml/iris",
    "dansbecker/melbourne-housing-snapshot",
    "colearninglounge/online-retail-purchase-orders-dataset",
]

def download_kaggle_dataset(dataset_name):
    """Download a dataset from Kaggle using kagglehub"""
    try:
        path = kagglehub.dataset_download(dataset_name)
        return Path(path)
    except Exception as e:
        print(f"Error downloading {dataset_name}: {e}")
        return None

def load_csv_files(dataset_path):
    """Load all CSV files from a dataset path"""
    csv_files = list(Path(dataset_path).glob("*.csv"))
    dataframes = {}
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, nrows=1000)  # Limit rows for performance
            if len(df) > 10:  # Only use if has enough data
                dataframes[csv_file.stem] = df
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    return dataframes

def get_numeric_columns(df):
    """Get numeric columns from dataframe"""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def to_yolo_bbox(x1, y1, x2, y2, W, H):
    y1_flipped = H - y2
    y2_flipped = H - y1
    cx = ((x1 + x2) / 2) / W
    cy = ((y1_flipped + y2_flipped) / 2) / H
    w = (x2 - x1) / W
    h = (y2_flipped - y1_flipped) / H
    return cx, cy, w, h

def merge_boxes(boxes):
    if not boxes:
        return None
    xs1 = [b[0] for b in boxes]
    ys1 = [b[1] for b in boxes]
    xs2 = [b[2] for b in boxes]
    ys2 = [b[3] for b in boxes]
    return min(xs1), min(ys1), max(xs2), max(ys2)

def safe_float(text):
    t = text.strip().replace("−", "-")
    return float(t)

def to_list(box):
    return [float(x) for x in box]

def maybe(p):
    return random.random() < p

def expand_box(box, pad=6, W=None, H=None):
    if box is None:
        return None
    x1, y1, x2, y2 = box
    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad
    if W is not None and H is not None:
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)
    return (x1, y1, x2, y2)

def force_tick_text(ax):
    ax.tick_params(labelbottom=True, labelleft=True)

def maybe_sci_notation(ax):
    if random.random() < 0.3:
        axis = random.choice(["x", "y"])
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        if axis == "x":
            ax.xaxis.set_major_formatter(formatter)
        else:
            ax.yaxis.set_major_formatter(formatter)

def generate_plot_from_data(i, df, x_col, y_col, seed=None):
    """Generate a plot from dataframe columns"""
    if seed is not None:
        random.seed(seed + i)
        np.random.seed(seed + i)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 12
    ax.tick_params(labelsize=11)

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(False)

    # Extract data and clean
    try:
        x_data = pd.to_numeric(df[x_col].dropna(), errors='coerce').dropna().values
        y_data = pd.to_numeric(df[y_col].dropna(), errors='coerce').dropna().values
        
        if len(x_data) < 5 or len(y_data) < 5:
            return False
        
        # Handle mismatched lengths
        min_len = min(len(x_data), len(y_data))
        x_data = x_data[:min_len]
        y_data = y_data[:min_len]
        
        # Plot with random style
        plot_type = random.choice(["line", "scatter"])
        
        if plot_type == "line":
            ax.plot(x_data, y_data, linestyle="-", linewidth=2, color=random.choice(["blue", "orange", "green"]))
        else:
            ax.scatter(x_data, y_data, alpha=random.uniform(0.6, 0.9), s=random.uniform(20, 60))
        
        # Labels: sometimes blank
        x_label = ""
        if random.random() < 0.8:
            x_label = str(x_col)
        
        y_label = ""
        if random.random() < 0.8:
            y_label = str(y_col)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Title: sometimes blank
        title = ""
        if random.random() < 0.75:
            title = random.choice(PLOT_TITLES)
        
        if title:
            ax.set_title(title)

        force_tick_text(ax)

        # Legend sometimes
        if random.random() < 0.6:
            legend_label = f"{y_col} vs {x_col}"
            ax.legend([legend_label], loc=random.choice(LEG_LOCS), framealpha=random.uniform(0.6, 1.0))
            has_legend = True
        else:
            has_legend = False

    except Exception as e:
        print(f"Error plotting data: {e}")
        plt.close(fig)
        return False

    # Set tick locators and formatters
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Lock layout before reading bboxes
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    W, H = fig.canvas.get_width_height()

    # Get pixel bboxes
    plot_bbox = ax.get_window_extent(renderer=renderer).extents
    xlabel_bbox = expand_box(ax.xaxis.label.get_window_extent(renderer=renderer).extents, pad=8, W=W, H=H)
    ylabel_bbox = expand_box(ax.yaxis.label.get_window_extent(renderer=renderer).extents, pad=8, W=W, H=H)

    xticks = [t.get_window_extent(renderer=renderer).extents
              for t in ax.get_xticklabels() if t.get_text().strip() != ""]
    yticks = [t.get_window_extent(renderer=renderer).extents
              for t in ax.get_yticklabels() if t.get_text().strip() != ""]

    x_ticks_box = expand_box(merge_boxes(xticks), pad=3, W=W, H=H)
    y_ticks_box = expand_box(merge_boxes(yticks), pad=5, W=W, H=H)

    title_bbox = None
    if ax.title.get_text().strip() != "":
        title_bbox = expand_box(
            ax.title.get_window_extent(renderer=renderer).extents,
            pad=6, W=W, H=H
        )

    legend_box = None
    if has_legend:
        legend = ax.get_legend()
        if legend is not None:
            legend_box = legend.get_window_extent(renderer=renderer).extents

    # Save image
    img_path = OUT_IMG / f"plot_{i:06d}.png"
    fig.savefig(img_path)
    plt.close(fig)

    # YOLO labels
    labels = []

    def add_box(name, box):
        if box is None:
            return
        cx, cy, w, h = to_yolo_bbox(*box, W, H)
        labels.append((CLASSES[name], cx, cy, w, h))

    add_box("plot_area", plot_bbox)
    add_box("legend", legend_box)
    add_box("x_label", xlabel_bbox)
    add_box("y_label", ylabel_bbox)
    add_box("x_ticks", x_ticks_box)
    add_box("y_ticks", y_ticks_box)
    add_box("title", title_bbox)

    # Save YOLO labels
    label_path = OUT_LBL / f"plot_{i:06d}.txt"
    with open(label_path, "w") as f:
        for cls, cx, cy, w, h in labels:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # Save ground truth JSON
    boxes_px = {
        "plot_area": to_list(plot_bbox),
        "legend": to_list(legend_box) if legend_box is not None else None,
        "x_label": to_list(xlabel_bbox),
        "y_label": to_list(ylabel_bbox),
        "x_ticks": to_list(x_ticks_box) if x_ticks_box is not None else None,
        "y_ticks": to_list(y_ticks_box) if y_ticks_box is not None else None,
        "title": to_list(title_bbox) if title_bbox is not None else None
    }

    boxes_yolo = {}
    for k, v in boxes_px.items():
        if v is None:
            boxes_yolo[k] = None
        else:
            cx, cy, w, h = to_yolo_bbox(v[0], v[1], v[2], v[3], W, H)
            boxes_yolo[k] = [cx, cy, w, h]

    def extract_ticks(getter):
        out = []
        for t in getter():
            s = t.get_text().strip()
            if not s:
                continue
            try:
                out.append(safe_float(s))
            except:
                pass
        return out

    gt = {
        "image": img_path.name,
        "image_size": [W, H],
        "plot_type": "data_plot",
        "data_source": f"{x_col} vs {y_col}",
        "boxes_px": boxes_px,
        "boxes_yolo": boxes_yolo,
        "text": {
            "x_label": ax.get_xlabel(),
            "y_label": ax.get_ylabel(),
            "legend_labels": [f"{y_col}"] if has_legend else [],
            "x_ticks": extract_ticks(ax.get_xticklabels),
            "y_ticks": extract_ticks(ax.get_yticklabels),
        }
    }

    json_path = OUT_JSON / f"plot_{i:06d}.json"
    with open(json_path, "w") as f:
        json.dump(gt, f, indent=2)

    return True

def main(n=100, seed=1337):
    print(f"Starting to generate {n} plots from Kaggle datasets...")
    
    # Download and load datasets
    all_dataframes = {}
    for dataset_name in KAGGLE_DATASETS:
        print(f"Downloading {dataset_name}...")
        dataset_path = download_kaggle_dataset(dataset_name)
        if dataset_path:
            dfs = load_csv_files(dataset_path)
            all_dataframes.update(dfs)
            print(f"  Loaded {len(dfs)} CSV files from {dataset_name}")
    
    if not all_dataframes:
        print("Error: No datasets loaded. Make sure Kaggle credentials are configured.")
        return
    
    print(f"Total datasets loaded: {len(all_dataframes)}")
    
    # Generate plots
    plot_count = 0
    attempts = 0
    max_attempts = n * 3
    
    while plot_count < n and attempts < max_attempts:
        attempts += 1
        
        # Pick random dataset
        dataset_name = random.choice(list(all_dataframes.keys()))
        df = all_dataframes[dataset_name]
        
        # Get numeric columns
        numeric_cols = get_numeric_columns(df)
        if len(numeric_cols) < 2:
            continue
        
        # Pick two random numeric columns
        x_col, y_col = random.sample(numeric_cols, 2)
        
        # Generate plot
        success = generate_plot_from_data(plot_count, df, x_col, y_col, seed=seed)
        
        if success:
            plot_count += 1
            if plot_count % 10 == 0:
                print(f"Generated {plot_count}/{n} plots...")
    
    print(f"Generated {plot_count} plots in: {OUT_IMG}, {OUT_LBL}, {OUT_JSON}")

if __name__ == "__main__":
    main(n=100, seed=1337)
