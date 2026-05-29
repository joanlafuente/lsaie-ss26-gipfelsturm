import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH    = "wandb_resubmission.csv"
OUTPUT_FILE = "resubmission_loss.png"
TITLE       = "Throughput Across Auto-Resubmitted SLURM Jobs (MoE with 2 experts)"
Y_LABEL     = "Throughput (tokens/sec/gpu)"  
LEGEND_LOC  = "lower right"                  
Y_FROM_ZERO = True                           # False for loss plots

# Substring in metric columns
METRIC = "tokens-per-sec-per-gpu"       

# Extract slurm job ID by splitting on prefix in the col name
JOB_ID_SPLIT = "moe-2-"

def plot_resubmission(csv_path, output_file):
    df = pd.read_csv(csv_path)

    cols = [
        c for c in df.columns
        if METRIC in c and "__MIN" not in c and "__MAX" not in c
    ]

    # Sort runs chronologically by their first valid step
    runs = []
    for col in cols:
        valid = df[col].dropna()
        if len(valid) == 0:
            continue
        first_step = df.loc[valid.index[0], "Step"]
        job_id = col.split(JOB_ID_SPLIT)[1].split(" ")[0]
        runs.append((first_step, job_id, col))
    runs.sort(key=lambda x: x[0])

    palette = sns.color_palette("tab10", n_colors=len(runs))

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(TITLE, fontsize=20, fontweight="bold", y=0.95)

    for i, (first_step, job_id, col) in enumerate(runs):
        valid_idx = df[col].dropna().index
        steps  = df.loc[valid_idx, "Step"].values
        values = df.loc[valid_idx, col].values
        ax.plot(
            steps, values,
            color=palette[i], linewidth=1.2, alpha=0.9,
            label=f"Job {job_id} (step {steps[0]}–{steps[-1]})",
        )

    ax.set_xlabel("Training Step", fontsize=14, fontweight="medium")
    ax.set_ylabel(Y_LABEL, fontsize=14, fontweight="medium")
    ax.legend(fontsize=11, loc=LEGEND_LOC, framealpha=0.85)
    ax.set_xlim(0, df["Step"].max())
    if Y_FROM_ZERO:
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    plot_resubmission(CSV_PATH, OUTPUT_FILE)