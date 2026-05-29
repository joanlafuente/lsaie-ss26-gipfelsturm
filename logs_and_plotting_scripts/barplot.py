import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# map raw filenames to display names
NAME_MAPPING = {
    "BASELINE_2_NODES.csv": "Baseline",
    "CUT_LOSS_MBS_4.csv": "Cut Loss (MBS 4)",
    "CUT_LOSS_MBS_8.csv": "Cut Loss (MBS 8)",
    "FP8_DELAYED.csv": "FP8 Delayed",
    "FP8_BLOCKWISE.csv": "FP8 Blockwise",
    "KERNEL_FUSION.csv": "Kernel Fusion"
}

TARGET_CATEGORY_MAPPINGS = {
    "Cut-Cross-Entropy": "CUT_LOSS",
    "FP8": "FP8",
    "Kernel Fusion": "KERNEL_FUSION"
}

def generate_throughput_barplot(json_file_path, target_category, output_filename="plot.png"):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
        
    plot_data = []

    # always grab baseline first
    try:
        baseline_stats = data["BASELINE"]["BASELINE_2_NODES.csv"]
        plot_data.append({
            "Run": NAME_MAPPING.get("BASELINE_2_NODES.csv", "BASELINE_2_NODES.csv"),
            "Mean Throughput": baseline_stats["mean_throughput"],
            "Std Throughput": baseline_stats["std_throughput"]
        })
    except KeyError:
        print("Warning: BASELINE_2_NODES.csv missing from JSON.")

    # pull the requested category
    if target_category in data:
        for run_name, stats in data[target_category].items():
            clean_name = NAME_MAPPING.get(run_name, run_name)
            
            # avoid dupes if baseline somehow ends up in here
            if not any(d['Run'] == clean_name for d in plot_data):
                plot_data.append({
                    "Run": clean_name,
                    "Mean Throughput": stats["mean_throughput"],
                    "Std Throughput": stats["std_throughput"]
                })
    else:
        print(f"Error: Couldn't find category '{target_category}'")
        return

    df = pd.DataFrame(plot_data)

    # styling setup
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    
    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        data=df, 
        x="Run", 
        y="Mean Throughput", 
        palette="viridis", 
        hue="Run",
        legend=False
    )

    # overlay error bars
    ax.errorbar(
        x=range(len(df)), 
        y=df["Mean Throughput"], 
        yerr=df["Std Throughput"], 
        fmt='none',       
        c='black',        
        capsize=8,        
        capthick=1.5,
        elinewidth=1.5
    )

    # formatting & labels
    clean_target = {v:k for k,v in TARGET_CATEGORY_MAPPINGS.items()}.get(target_category, target_category)
    plt.title(f"Throughput Comparison: {clean_target} vs Baseline", pad=20, fontsize=18, fontweight='bold')
    plt.ylabel("Mean Throughput (tokens/sec/gpu)", fontsize=14, fontweight='medium')
    plt.xlabel("Configuration", fontsize=14, fontweight='medium')
    
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    
    print(f"Saved -> {output_filename}")

if __name__ == "__main__":
    generate_throughput_barplot('throughput_metrics_v3.json', 'CUT_LOSS', 'cut_loss_comparison.png')