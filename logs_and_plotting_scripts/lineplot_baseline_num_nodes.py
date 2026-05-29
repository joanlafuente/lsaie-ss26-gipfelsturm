import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from matplotlib.ticker import ScalarFormatter

def generate_side_by_side_scaling(json_file_path, output_filename="baseline_scaling_side_by_side.png"):
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
        
    plot_data = []

    # Extract BASELINE data
    if "BASELINE" in data:
        for run_name, stats in data["BASELINE"].items():
            match = re.search(r'BASELINE_(\d+)_NODE', run_name)
            if match:
                num_nodes = int(match.group(1))
                plot_data.append({
                    "Num Nodes": num_nodes,
                    "Mean Throughput": stats["mean_throughput"],
                    "Std Throughput": stats["std_throughput"]
                })
                
    if not plot_data:
        print("Error: No valid baseline node data found in the JSON.")
        return

    # Convert to DataFrame and sort
    df = pd.DataFrame(plot_data)
    df = df.sort_values(by="Num Nodes")

    # ---------------------------------------------------------
    # Plotting & Font Configuration
    # ---------------------------------------------------------
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    
    # Create a figure with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Add the overarching global figure title
    fig.suptitle(
        "Scaling: Throughput vs Number of Nodes", 
        fontsize=20, 
        fontweight='bold', 
        y=1.05
    )

    # ---------------------------------------------------------
    # --- LEFT PLOT: FULL SCALE ---
    # ---------------------------------------------------------
    ax1.errorbar(
        x=df["Num Nodes"], 
        y=df["Mean Throughput"], 
        yerr=df["Std Throughput"], 
        fmt='-o',               
        color='#2878b5',         
        ecolor='#666666',        
        capsize=5,              
        capthick=1.5,
        elinewidth=1.5,
        markersize=10,           
        linewidth=2.5,
        markeredgecolor='white', 
        markeredgewidth=1.5
    )

    # Updated Subtitle for Left Plot
    ax1.set_title("Throughput Overview", pad=15, fontsize=16, fontweight='medium')
    ax1.set_ylabel("Throughput (tokens/sec/gpu)", fontsize=14, fontweight='medium')
    ax1.set_xlabel("Number of Nodes", fontsize=14, fontweight='medium')
    
    ax1.set_xscale('log', base=2)       
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.set_xticks([1, 2, 4])
    
    # Force Y-axis to start at 0
    ax1.set_ylim(bottom=0)
    current_top = ax1.get_ylim()[1]
    ax1.set_ylim(top=current_top * 1.1)

    # ---------------------------------------------------------
    # --- RIGHT PLOT: ZOOMED IN ---
    # ---------------------------------------------------------
    ax2.errorbar(
        x=df["Num Nodes"], 
        y=df["Mean Throughput"], 
        yerr=df["Std Throughput"], 
        fmt='-o',               
        color='#2878b5',        
        ecolor='#666666',         
        capsize=5,              
        capthick=1.5,
        elinewidth=1.5,
        markersize=10,           
        linewidth=2.5,
        markeredgecolor='white',
        markeredgewidth=1.5
    )

    # Updated Subtitle for Right Plot
    ax2.set_title("Zoomed View", pad=15, fontsize=16, fontweight='medium')
    ax2.set_xlabel("Number of Nodes", fontsize=14, fontweight='medium')
    
    # Dynamic limits based on data variance
    y_min = df["Mean Throughput"].min() - (df["Std Throughput"].max() * 2)
    y_max = df["Mean Throughput"].max() + (df["Std Throughput"].max() * 2)
    ax2.set_ylim(y_min, y_max)
    
    ax2.set_xscale('log', base=2)
    ax2.set_xticks([1, 2, 4])
    ax2.xaxis.set_major_formatter(ScalarFormatter())

    # Note: ax1.indicate_inset_zoom(...) has been successfully removed.

    # Adjust layout to prevent overlap and save
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.show()
    
    print(f"Side-by-side plot without connecting lines successfully saved to {output_filename}")

# --- Example Usage ---
if __name__ == "__main__":
    generate_side_by_side_scaling("throughput_metrics_v2.json", 'baseline_scaling_side_by_side.png')