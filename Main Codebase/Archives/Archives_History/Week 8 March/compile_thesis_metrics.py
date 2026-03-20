import json
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
THESIS_DIR = SCRIPT_DIR / "thesis_manuscript"

EXP_PATHS_CH4 = {
    "Deep Learning Baseline": SCRIPT_DIR / "dl_baseline" / "results" / "results.json",
    "Phase 1: Basic LCS": SCRIPT_DIR / "exp_v1_baseline" / "results" / "results.json",
    "Phase 2 & 3: Clinical Safety Weighting": SCRIPT_DIR / "exp_a_safety_first" / "results" / "results.json",
    "Phase 4: Stacking Ensemble": SCRIPT_DIR / "exp_v4_stacking" / "results" / "results.json",
    "Phase 6: Global Peak": SCRIPT_DIR / "exp_b_conservative" / "results" / "results.json",
}

EXP_PATHS_CH6 = {
    "Phase 8a: EUQ-LCS (Evidential)": SCRIPT_DIR / "exp_v8_evidential_uncertainty" / "results" / "results.json",
    "Phase 8b: FARM-LCS (Fuzzy)": SCRIPT_DIR / "exp_v8_farm_fuzzy" / "results" / "results.json",
    "Phase 8c: MN-LCS (Neural)": SCRIPT_DIR / "exp_v8_neural_lcs" / "results" / "results.json",
    "Phase 8d: LKH-LCS (Latent)": SCRIPT_DIR / "exp_v8_latent_knowledge" / "results" / "results.json",
}

def load_metrics(exp_dict):
    data = []
    for name, path in exp_dict.items():
        if path.exists():
            with open(path, 'r') as f:
                res = json.load(f)
                
                # Support nested 'external' metrics used by FARM-LCS
                metrics = res.get('external', res)
                
                data.append({
                    "Model Phase": name,
                    "Balanced Accuracy": f"{metrics.get('ba', 0):.4f}",
                    "Sensitivity (Recall)": f"{metrics.get('sens', 0):.4f}",
                    "Specificity": f"{metrics.get('spec', 0):.4f}",
                    "F1-Score": f"{metrics.get('f1', 0):.4f}",
                    "MCC": f"{metrics.get('mcc', 0):.4f}"
                })
    return pd.DataFrame(data)

def inject_markdown(chapter_file, injection_content, header_trigger):
    if not chapter_file.exists():
        return
    with open(chapter_file, 'r') as f:
        lines = f.readlines()
        
    out_lines = []
    ignoring = False
    for i, line in enumerate(lines):
        if line.strip() == "[Draft Content Here]" and header_trigger in "".join(lines[max(0, i-2):i]):
            out_lines.append(f"{injection_content}\n")
        else:
            out_lines.append(line)
            
    with open(chapter_file, 'w') as f:
        f.writelines(out_lines)

if __name__ == "__main__":
    print("Compiling Chapter 4 metrics...")
    df_ch4 = load_metrics(EXP_PATHS_CH4)
    if not df_ch4.empty:
        md_table_ch4 = "\n### Performance Matrix: The Evolutionary Trajectory\n\n" + df_ch4.to_markdown(index=False)
        inject_markdown(THESIS_DIR / "04_Evolutionary_Committee_Architectures.md", md_table_ch4, "## 4.5 Generalization Results (Closing the Gap)")
        print("Injected into Chapter 4.")

    print("Compiling Chapter 6 metrics...")
    df_ch6 = load_metrics(EXP_PATHS_CH6)
    if not df_ch6.empty:
        md_table_ch6 = "\n### Performance Matrix: 2026 Architectural Innovations\n\n" + df_ch6.to_markdown(index=False)
        inject_markdown(THESIS_DIR / "06_Methodological_Innovation_EUQ.md", md_table_ch6, "## 6.4 Results: Correlating Ignorance Mass with False Negatives")
        print("Injected into Chapter 6.")

    print("Thesis metrics compilation completed successfully.")
