import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# Setup Styling
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

SCRIPT_DIR = Path(__file__).parent.resolve()
FIGURES_DIR = SCRIPT_DIR / "thesis_figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# PhD Timeline definition (Chronological Order for the Thesis Story)
PHASES = [
    {"dir": "exp_v1_baseline", "name": "v1: Initial LCS", "type": "baseline"},
    {"dir": "exp_a_safety_first", "name": "v2: Safety-First", "type": "evolution"},
    {"dir": "exp_b_conservative", "name": "v3: Conservative", "type": "evolution"},
    {"dir": "exp_v4_stacking", "name": "v4: Stacking", "type": "evolution"},
    {"dir": "exp_v5_calibration", "name": "v5: Calibration", "type": "evolution"},
    {"dir": "exp_c_balanced_efficiency", "name": "v6: Balanced", "type": "evolution"},
    {"dir": "exp_d_knowledge_discovery", "name": "v7: Knowledge", "type": "evolution"},
    {"dir": "exp_v8_evidential_uncertainty", "name": "v8: EUQ-LCS", "type": "innovation"}
]

def load_metrics():
    results = []
    for p in PHASES:
        json_path = SCRIPT_DIR / p["dir"] / "results" / "results.json"
        
        # Default empty values in case a run is not yet finished
        parsed_res = {
            "Phase": p["name"],
            "External_BA": 0.0,
            "Type": p["type"]
        }
        
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    res = json.load(f)
                    
                data = res.get('external', res)
                    
                # Handle Phase 8 specific format vs previous phases
                if "balanced_accuracy" in data:
                    parsed_res["External_BA"] = data["balanced_accuracy"]
                elif "External" in data and "Balanced Accuracy" in data["External"]:
                    parsed_res["External_BA"] = data["External"]["Balanced Accuracy"]
                elif "evaluation" in data and "balanced_accuracy" in data["evaluation"]:
                    parsed_res["External_BA"] = data["evaluation"]["balanced_accuracy"]
            except Exception as e:
                print(f"Failed to parse {json_path}: {e}")
                
        results.append(parsed_res)
        
    return pd.DataFrame(results)

def plot_performance_evolution(df):
    plt.figure(figsize=(10, 6))
    
    # Filter out phases that haven't run yet (BA == 0)
    df_valid = df[df["External_BA"] > 0]
    
    if df_valid.empty:
        print("No valid results found yet. Plotting skipped.")
        return
        
    ax = sns.lineplot(
        data=df_valid, 
        x="Phase", 
        y="External_BA", 
        marker="o", 
        linewidth=2.5, 
        markersize=10,
        color="#2c3e50"
    )
    
    # Highlight the innovation phase
    innovation_points = df_valid[df_valid['Type'] == 'innovation']
    if not innovation_points.empty:
        plt.scatter(
            innovation_points['Phase'], 
            innovation_points['External_BA'], 
            color='#e74c3c', 
            s=150, 
            zorder=5, 
            label='Methodological Innovation'
        )
    
    plt.title("PhD Timeline: Evolution of External Balanced Accuracy", pad=20, fontsize=14, fontweight='bold')
    plt.xlabel("Experimental Phase", labelpad=15)
    plt.ylabel("Generalization Score (External BA)", labelpad=15)
    
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.45, 0.85)
    plt.axhline(0.50, color='r', linestyle='--', alpha=0.3, label="Random Guess / Raw DL Baseline")
    
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    
    save_path = FIGURES_DIR / "fig_1_performance_evolution.png"
    plt.savefig(save_path)
    print(f"Saved figure to {save_path}")

if __name__ == "__main__":
    print("--- Generating PhD Thesis Visualizations ---")
    df_metrics = load_metrics()
    plot_performance_evolution(df_metrics)
    print("Complete. Check the 'thesis_figures' directory.")
