import matplotlib.pyplot as plt
import numpy as np
import json
import random
from pathlib import Path

OUT_IMG = Path("output/images")
OUT_LBL = Path("output/labels")
OUT_JSON = Path("output/ground_truth")

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)
OUT_JSON.mkdir(parents=True, exist_ok=True)

CLASSES = {
    "plot_area": 0,
    "legend": 1,
    "x_label": 2,
    "y_label": 3,
    "x_ticks": 4,
    "y_ticks": 5,
    "title": 6
}

X_LABELS = ["Distance (m)", "Time (s)", "Frequency (Hz)", "Input", "Samples", "Wavelength (nm)", "Pressure (kPa)"]
Y_LABELS = ["Amplitude", "Velocity", "Voltage", "Output", "Signal", "Power (W)", "Acceleration (m/s^2)"]
TITLES   = ["Experiment Results", "Signal Response", "Velocity Profile", "Sample Data", "Measurement", ""]

LINE_FUNCS = [
    lambda x: np.sin(x),
    lambda x: np.cos(x),
    lambda x: np.tanh(x / 3),
    lambda x: x,
    lambda x: x**2 / 60,
    lambda x: np.exp(x / 10),
    lambda x: np.log(np.abs(x) + 1),
]

STYLES   = ["-", "--", ":", "-."]
MARKERS  = [None, "o", "s", "^", "x", "d"]
LEG_LOCS = ["upper left", "upper right", "lower left", "lower right", "best"]
FONTS    = ["DejaVu Sans", "Arial", "Times New Roman"]

def to_yolo_bbox(x1, y1, x2, y2, W, H):
    # Matplotlib uses bottom-left origin
    # Image/YOLO uses top-left origin
    y1_flipped = H - y2
    y2_flipped = H - y1

    cx = ((x1 + x2) / 2) / W
    cy = ((y1_flipped + y2_flipped) / 2) / H
    w  = (x2 - x1) / W
    h  = (y2_flipped - y1_flipped) / H

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
    # handles unicode minus and scientific notation
    t = text.strip().replace("−", "-")
    return float(t)

def to_list(box):
    return [float(x) for x in box]

def maybe(ax, p):
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



def set_tick_style(ax):
    # rotate some ticks sometimes
    if maybe(ax, 0.25):
        for t in ax.get_xticklabels():
            t.set_rotation(random.choice([20, 30, 45, 60]))
            t.set_ha("right")

    if maybe(ax, 0.25):
        for t in ax.get_yticklabels():
            t.set_rotation(random.choice([0, 0, 90]))

def force_tick_text(ax):
    # Ensure ticks actually render labels (matplotlib sometimes hides)
    ax.tick_params(labelbottom=True, labelleft=True)

from matplotlib.ticker import ScalarFormatter

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


def maybe_log_scale(ax):
    # Only safe if values are positive; apply after plotting if possible.
    # We'll attempt y-log if y is positive; otherwise skip.
    pass

def plot_line(ax, x):
    n = random.randint(1, 4)
    labels = []
    for j in range(n):
        func = random.choice(LINE_FUNCS)
        shift = random.uniform(-2, 2)
        scale = random.uniform(0.5, 2.5)

        y = scale * func(x + shift)

        # noise + occasional outliers
        noise = np.random.normal(0, random.uniform(0.01, 0.25), size=len(x))
        y = y + noise

        if maybe(ax, 0.15):
            k = random.randint(2, 5)
            idx = np.random.choice(len(x), size=k, replace=False)
            y[idx] += np.random.normal(0, random.uniform(0.5, 2.0), size=k)

        ax.plot(
            x, y,
            linestyle=random.choice(STYLES),
            marker=random.choice(MARKERS),
            linewidth=random.uniform(1.0, 3.0),
            alpha=random.uniform(0.75, 1.0),
            markersize=random.uniform(3, 7)
        )
        labels.append(f"Line {j}")
    return labels

def plot_scatter(ax, x):
    n = random.randint(1, 3)
    labels = []
    for j in range(n):
        func = random.choice(LINE_FUNCS)
        shift = random.uniform(-2, 2)
        scale = random.uniform(0.5, 2.5)

        y = scale * func(x + shift)
        y += np.random.normal(0, random.uniform(0.05, 0.4), size=len(x))

        ax.scatter(
            x, y,
            s=random.uniform(8, 35),
            alpha=random.uniform(0.5, 0.9),
            marker=random.choice(["o", "s", "^", "x", "d"])
        )
        labels.append(f"Series {j}")
    return labels

def plot_bar(ax):
    n = random.randint(6, 14)
    x = np.arange(n)
    y = np.abs(np.random.normal(loc=random.uniform(1, 8), scale=random.uniform(0.5, 3), size=n))
    ax.bar(x, y, alpha=random.uniform(0.7, 1.0))
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    return ["Bars"]

def plot_step(ax, x):
    y = np.cumsum(np.random.normal(0, 0.3, size=len(x)))
    ax.step(x, y, where="mid", linewidth=random.uniform(1.0, 2.5))
    return ["Step"]

def plot_stem(ax, x):
    y = np.random.normal(0, 1, size=len(x))
    markerline, stemlines, baseline = ax.stem(x[::10], y[::10])
    plt.setp(stemlines, linewidth=random.uniform(0.8, 2.0))
    plt.setp(markerline, markersize=random.uniform(4, 7))
    return ["Stem"]

def plot_errorbar(ax, x):
    y = np.sin(x / random.uniform(1.5, 3.5)) + np.random.normal(0, 0.1, size=len(x))
    y = y[::10]
    x2 = x[::10]
    err = np.abs(np.random.normal(0.2, 0.1, size=len(x2)))
    ax.errorbar(x2, y, yerr=err, fmt=random.choice(["o-", "s--", "^-", "x:"]), capsize=3)
    return ["Error"]

def plot_hist(ax):
    data = np.random.normal(loc=random.uniform(-2, 2), scale=random.uniform(0.5, 2.0), size=600)
    ax.hist(data, bins=random.randint(10, 30), alpha=random.uniform(0.7, 1.0))
    return ["Histogram"]

def generate_plot(i, seed=None):
    if seed is not None:
        random.seed(seed + i)
        np.random.seed(seed + i)

    fig, ax = plt.subplots(
        figsize=(random.uniform(4.5, 7.5), random.uniform(3.2, 5.4)),
        dpi=random.choice([100, 120, 150])
    )

    # global style knobs
    plt.rcParams["font.family"] = random.choice(FONTS)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Choose plot type
    plot_type = random.choices(
        population=["line", "scatter", "bar", "step", "stem", "errorbar", "hist"],
        weights=[0.45, 0.18, 0.12, 0.08, 0.06, 0.06, 0.05],
        k=1
    )[0]

    # x-range for most plot types
    x = np.linspace(random.uniform(-10, 0), random.uniform(10, 30), random.randint(80, 220))

    legend_labels = []
    if plot_type == "line":
        legend_labels = plot_line(ax, x)
    elif plot_type == "scatter":
        legend_labels = plot_scatter(ax, x)
    elif plot_type == "bar":
        legend_labels = plot_bar(ax)
    elif plot_type == "step":
        legend_labels = plot_step(ax, x)
    elif plot_type == "stem":
        legend_labels = plot_stem(ax, x)
    elif plot_type == "errorbar":
        legend_labels = plot_errorbar(ax, x)
    elif plot_type == "hist":
        legend_labels = plot_hist(ax)

    ax.set_xlabel(random.choice(X_LABELS))
    ax.set_ylabel(random.choice(Y_LABELS))

    title = random.choice(TITLES)
    if title and maybe(ax, 0.7):
        ax.set_title(title)

    # Grid sometimes
    if maybe(ax, 0.6):
        ax.grid(True)

    # scientific notation sometimes
    maybe_sci_notation(ax)

    # ticks formatting and forcing
    force_tick_text(ax)
    set_tick_style(ax)

    # Legend sometimes absent (real-world!)
    has_legend = (len(legend_labels) > 0) and maybe(ax, 0.85)
    if has_legend:
        ax.legend(legend_labels, loc=random.choice(LEG_LOCS), framealpha=random.uniform(0.6, 1.0))

    # IMPORTANT: lock layout BEFORE reading bboxes
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    W, H = fig.canvas.get_width_height()

    # --- pixel bboxes in canvas coordinate space ---
    plot_bbox = ax.get_window_extent(renderer=renderer).extents
    xlabel_bbox = expand_box(ax.xaxis.label.get_window_extent(renderer=renderer).extents, pad=8, W=W, H=H)
    ylabel_bbox = expand_box(ax.yaxis.label.get_window_extent(renderer=renderer).extents, pad=8, W=W, H=H)



    xticks = [t.get_window_extent(renderer=renderer).extents
              for t in ax.get_xticklabels() if t.get_text().strip() != ""]
    yticks = [t.get_window_extent(renderer=renderer).extents
              for t in ax.get_yticklabels() if t.get_text().strip() != ""]

        # Base tick boxes
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

    # Save image (NO bbox_inches="tight" to preserve bbox correctness)
    img_path = OUT_IMG / f"plot_{i:06d}.png"
    fig.savefig(img_path)
    plt.close(fig)

    # YOLO labels (normalized cx,cy,w,h)
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


    label_path = OUT_LBL / f"plot_{i:06d}.txt"
    with open(label_path, "w") as f:
        for cls, cx, cy, w, h in labels:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # ground truth json
    # NOTE: store both pixel boxes and normalized YOLO boxes (useful later)
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

    # Extract tick numeric values safely (some plots have non-numeric categorical ticks)
    def extract_ticks(getter):
        out = []
        for t in getter():
            s = t.get_text().strip()
            if not s:
                continue
            try:
                out.append(safe_float(s))
            except Exception:
                # allow non-numeric ticks (bar/hist)
                pass
        return out

    gt = {
        "image": img_path.name,
        "image_size": [W, H],
        "plot_type": plot_type,
        "boxes_px": boxes_px,
        "boxes_yolo": boxes_yolo,
        "text": {
            "x_label": ax.get_xlabel(),
            "y_label": ax.get_ylabel(),
            "legend_labels": legend_labels if has_legend else [],
            "x_ticks": extract_ticks(ax.get_xticklabels),
            "y_ticks": extract_ticks(ax.get_yticklabels),
        }
    }

    json_path = OUT_JSON / f"plot_{i:06d}.json"
    with open(json_path, "w") as f:
        json.dump(gt, f, indent=2)

def main(n=20, seed=1337):
    for i in range(n):
        generate_plot(i, seed=seed)
    print(f"Generated {n} plots in: {OUT_IMG}, {OUT_LBL}, {OUT_JSON}")

if __name__ == "__main__":
    main(n=20, seed=1337)
