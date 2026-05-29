import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_memory_barplot(output_filename="memory_usage.png"):
    data = [
        {"Configuration": "Baseline (MBS 4)", "Memory Usage (%)": 58},
        {"Configuration": "Cut-Cross-Entropy (MBS 4)", "Memory Usage (%)": 48},
        {"Configuration": "Cut-Cross-Entropy (MBS 8)", "Memory Usage (%)": 85}
    ]
    df = pd.DataFrame(data)

    # styling setup
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    
    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        data=df, 
        x="Configuration", 
        y="Memory Usage (%)", 
        palette="viridis", 
        hue="Configuration",
        legend=False
    )

    # float percentage labels just above each bar
    for i, val in enumerate(df["Memory Usage (%)"]):
        ax.text(
            i, val + 1.5, f"{val}%",
            ha='center', 
            fontsize=14,
            fontweight='bold',
            color='#333333'
        )

    # formatting
    plt.title("Peak Memory Usage by Configuration", pad=20, fontsize=18, fontweight='bold')
    plt.ylabel("Memory Usage (%)", fontsize=14, fontweight='medium')
    plt.xlabel("Configuration", fontsize=14, fontweight='medium')
    
    plt.xticks(rotation=0, fontsize=13)
    plt.yticks(fontsize=12)
    
    # lock y-axis to 100% bounds
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    
    print(f"Saved -> {output_filename}")

if __name__ == "__main__":
    generate_memory_barplot('memory_comparison.png')