import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH    = "wandb_tokens_per_sec_across_configs.csv"
OUTPUT_FILE = "throughput_across_configs.png"
TITLE       = "Throughput Comparison Across Configurations (1.5B, 4 nodes)"
Y_LABEL     = "Throughput (tokens/sec/gpu)"           
LEGEND_LOC  = "upper right"                           
Y_FROM_ZERO = False                                   # True for throughput, False for loss

# Map legend label to CSV column name 
RUNS = {
    "FP8 Delayed":       "train-1.5b-4n-kernel_fusion-false-fp8-true-cut_cross_entropy-false-FP8_DELAYED - tokens_per_sec_per_gpu",
    "Baseline":          "train-1.5b-4n-kernel_fusion-false-fp8-false-cut_cross_entropy-false-BASELINE - tokens_per_sec_per_gpu",
    "CUDA Graph":        "train-1.5b-4n-kernel_fusion-true-fp8-false-cut_cross_entropy-false-CUDA_GRAPHS - tokens_per_sec_per_gpu",
    "Cut-Cross-Entropy": "train-1.5b-4n-kernel_fusion-false-fp8-false-cut_cross_entropy-true-CUT_LOSS_MBS4 - tokens_per_sec_per_gpu",
}

PALETTE = ["#7fb3d3", "#c0392b", "#27ae60", "#8e44ad"]

# EMA smoothing
EMA_ALPHA = 0.3

# x-axis range
X_MIN = None
X_MAX = None

def ema(values, alpha=EMA_ALPHA):
    smoothed = np.zeros_like(values, dtype=float)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed

def plot_lines(csv_path, output_file):
    df = pd.read_csv(csv_path)

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(TITLE, fontsize=20, fontweight="bold", y=0.95)

    for (label, col), color in zip(RUNS.items(), PALETTE):
        steps  = df["Step"].values
        values = ema(df[col].values)
        ax.plot(steps, values, color=color, linewidth=1.2, label=label, alpha=0.9)

    ax.set_xlabel("Training Step", fontsize=14, fontweight="medium")
    ax.set_ylabel(Y_LABEL, fontsize=14, fontweight="medium")
    ax.legend(fontsize=11, loc=LEGEND_LOC, framealpha=0.85)
    ax.set_xlim(X_MIN or df["Step"].min(), X_MAX or df["Step"].max())
    if Y_FROM_ZERO:
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    plot_lines(CSV_PATH, OUTPUT_FILE)