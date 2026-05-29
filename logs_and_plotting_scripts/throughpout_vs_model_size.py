import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from matplotlib.ticker import FuncFormatter

def generate_model_size_plot(json_file_path, output_filename="throughput_vs_size_log_scale.png"):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
        
    target_key = "Model Size vs Throughpout"
    if target_key not in data:
        print(f"Error: '{target_key}' missing from JSON.")
        return
        
    category_data = data[target_key]
    plot_data = []
    
    # extract numeric sizes from filenames (e.g., _1_5b.csv -> 1.5)
    def parse_size_to_billions(filename):
        match = re.search(r'_(\d+)(?:_(\d+))?([mb])\.csv$', filename.lower())
        if match:
            whole = match.group(1)
            frac = match.group(2)
            unit = match.group(3)
            
            val = float(whole)
            if frac:
                val += float(frac) / (10 ** len(frac))
            
            if unit == 'm':
                return val / 1000.0  
            return val
        return None

    for category, runs in category_data.items():
        for run_name, stats in runs.items():
            size_b = parse_size_to_billions(run_name)
            if size_b is not None:
                plot_data.append({
                    "Category": category,
                    "Model Size (B)": size_b,
                    "Mean Throughput": stats["mean_throughput"],
                    "Std Throughput": stats["std_throughput"]
                })

    df = pd.DataFrame(plot_data).sort_values(by="Model Size (B)")

    # styling setup
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    
    plt.figure(figsize=(11, 7))
    
    colors = ['#2878b5', '#ff7f0e', '#2ca02c', '#d62728'] 
    color_map = dict(zip(df["Category"].unique(), colors))

    # plot each category
    for category in df["Category"].unique():
        subset = df[df["Category"] == category]
        fmt = '-o' if len(subset) > 1 else 'o'
        
        plt.errorbar(
            x=subset["Model Size (B)"], 
            y=subset["Mean Throughput"], 
            yerr=subset["Std Throughput"], 
            fmt=fmt,               
            label=category,
            color=color_map.get(category, '#333333'),        
            ecolor='#666666',         
            capsize=5,              
            capthick=1.5,
            elinewidth=1.5,
            markersize=9,           
            linewidth=2.5,
            markeredgecolor='white',
            markeredgewidth=1.5
        )

    # x-axis: log scale (base 2) with custom string formatting
    plt.xscale('log', base=2)
    plt.xticks([0.125, 0.76, 1.5, 3.0, 8.0])
    
    def format_size(x, pos):
        if x in (0.75, 0.76):
            return "760M" 
        if x < 1.0:
            return f"{int(x * 1000)}M"
        return f"{x:g}B"
        
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_size))
    
    # y-axis: specific ticks for readability, crop bottom empty space
    y_ticks = [10000, 20000, 30000, 40000, 50000, 60000]
    plt.yticks(y_ticks, [f"{y:,}" for y in y_ticks], fontsize=12)
    plt.gca().yaxis.set_minor_formatter(plt.NullFormatter())
    plt.ylim(0, 65000)
    
    # formatting & labels
    plt.title("Throughput vs Model Size", pad=20, fontsize=18, fontweight='bold')
    plt.ylabel("Throughput (tokens/sec/gpu)", fontsize=14, fontweight='medium')
    plt.xlabel("Model Size", fontsize=14, fontweight='medium')
    
    plt.xticks(fontsize=12)
    plt.legend(title="Configuration", fontsize=12, title_fontsize=13)
    plt.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.show()
    
    print(f"Saved -> {output_filename}")

if __name__ == "__main__":
    generate_model_size_plot("throughput_metrics_v3.json", "throughput_vs_size_no_log_scale.png")