import json
import numpy as np
import pandas as pd
from pathlib import Path

def get_stable_data(values, is_cce=False):
    n = len(values)
    if n < 20:
        return values  # too short to truncate

    if is_cce:
        # CCE has a long initial spike (gradient skipping), skip first 200
        if n > 200:
            return values[200:]
        else:
            # fallback for short/failed debug runs
            print(f"Warning: CCE run only has {n} steps. Using last 50%.")
            return values[n // 2:]
    else:
        # standard warmup: drop first 25%
        skip = int(n * 0.25)
        return values[skip:]

def process_performance_data(root_dir, output_json):
    root_path = Path(root_dir)
    result_tree = {}

    for csv_path in root_path.rglob('*.csv'):
        try:
            df = pd.read_csv(csv_path)
            
            # find throughput col, ignoring wandb min/max bounds
            cols = [c for c in df.columns if 'tokens-per-sec' in c and '__MIN' not in c and '__MAX' not in c]
            val_col = cols[0] if cols else df.columns[1]
            
            is_cce = 'cut_cross_entropy-true' in val_col or 'cut_cross_entropy' in csv_path.name
            
            values = df[val_col].dropna().values
            stable_values = get_stable_data(values, is_cce=is_cce)
            
            # build nested dict based on folder structure
            rel_parts = csv_path.relative_to(root_path).parts
            current_level = result_tree
            
            for part in rel_parts[:-1]:
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]
            
            current_level[csv_path.name] = {
                "mean_throughput": float(np.mean(stable_values)),
                "std_throughput": float(np.std(stable_values))
            }
            
        except Exception as e:
            print(f"Skipping {csv_path.name}: {e}")

    with open(output_json, 'w') as f:
        json.dump(result_tree, f, indent=4)
    
    print(f"Saved -> {output_json}")

if __name__ == "__main__":
    DATA_DIR = "/Users/jlafuente/Desktop/ETH/2nd Semester/Large Scale AI engineering/Project/" 
    OUTPUT_FILE = "throughput_metrics_v3.json"

    process_performance_data(DATA_DIR, OUTPUT_FILE)