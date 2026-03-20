import sys
import os
from pathlib import Path
# Ensure project_paths.py is found at the root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import base64
import requests
import os

# Mermaid diagrams from the PhD documentation
diagrams = {
    "phase2_safety": """graph TD
    A["14 Clinical + 4 Pathological Features"] --> B["SimpleImputer (Strategy=Median)"]
    B --> C["Z-Score Normalization (StandardScaler)"]
    C --> D["LCS Engine: skExSTraCS v2.1"]
    D --> E["100,000 Training Iterations"]
    E --> F["Boolean 'Center-Spread' Interval Matching"]
    F --> G["Target Class: Safety (Balanced Accuracy: 71.79%)"]""",
    
    "phase4_stacking": """graph TD
    A["ResNet-50 (2048) + Clinical (18) Features"] --> B["Combined Feature Matrix (2066-dim)"]
    B --> C["Heterogeneous Ensemble Layer"]
    subgraph Base Models
        C1["ExSTraCS (LCS)"] 
        C2["Random Forest (100 Trees)"]
        C3["XGBoost (Learning Rate 0.1)"]
    end
    C --> C1
    C --> C2
    C --> C3
    C1 --> D["Meta-Level Probability Map"]
    C2 --> D
    C3 --> D
    D --> E["Logistic Regression (L2 Penalty, C=1.0)"]
    E --> F["Stacked Decision Boundary"]
    F --> G["Int. BA: 73.30% | Ext. Spec: 85.09%"]""",
    
    "phase6_consensus": """graph TD
    A["ResNet-50 Features (2048-dim)"] --> B["80/20 Stratified Partitioning (Seed=42)"]
    B --> C["Ens-Initialization: 5x skExSTraCS Experts"]
    subgraph Parallel Convergence
        D["500,000 Max Iterations per Expert"]
    end
    C --> D
    D --> E1["Expert 1 (Path A)"]
    D --> E2["Expert 2 (Path B)"]
    D --> E3["Expert 3 (Path C)"]
    E1 --> F["Majority Voting Mechanism"]
    E2 --> F
    E3 --> F
    F --> G["Mean Rule Probability Calculation"]
    G --> H["Consensus Engine (Ext. Spec: 87.24%)"]""",
    
    "phase8a_uncertainty": """graph TD
    A["Deep Feature Input"] --> B["LCS Rule Discovery Process"]
    B --> C["Evidential Deep Learning Module"]
    subgraph Uncertainty Quantification
        C1["Evidence Accrual Alpha = e + 1"]
        C2["Dirichlet Distribution Mapping"]
    end
    C --> C1
    C --> C2
    C1 --> D["Epistemic Uncertainty Score (u)"]
    C2 --> D
    D --> E{"u < 0.15 Threshold"}
    E -- Pass --> F["Confident Signal (Ext. BA: 73.15%)"]
    E -- Fail --> G["Stochastic Abstinence (Unreliable)"]""",
    
    "phase8b_farm_lcs": """graph LR
    subgraph Data Flow
        A["ResNet-50 Features"] --> B["Impute & Normalization"]
    end
    
    subgraph FARM-Fuzzy Logic [Core Engine]
        B --> C["Gamma Selection (Fixed 10% Slope)"]
        C --> D["Trapezoidal Membership: μ(x; a,b,c,d)"]
        D --> E["Algebraic Product T-Norm: T(a,b) = a*b"]
    end
    
    subgraph Inference Path
        E --> F["5-Expert Parallel Consensus"]
        F --> G["Soft-Match Threshold: μ >= 0.5"]
    end
    
    G --> H["Fuzzy Optimized Prediction"]
    H --> I["**Ext. BA: 76.28%** (PhD Milestone)"]"""
}

output_dir = "C:\\Users\\umair\\Videos\\PhD\\PhD Data\\Week 5 Febuary\\Deep_Features_Experiment\\diagram_images"
os.makedirs(output_dir, exist_ok=True)

def fetch_diagram(name, code):
    # Encode the code to base64 using UTF-8 to handle special characters like μ
    sample_string_bytes = code.encode("utf-8")
    base64_bytes = base64.b64encode(sample_string_bytes)
    base64_string = base64_bytes.decode("ascii")
    
    # Use mermaid.ink to fetch the image
    url = f"https://mermaid.ink/img/{base64_string}?bgColor=white"
    print(f"Fetching {name} from {url}...")
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            filepath = os.path.join(output_dir, f"{name}.png")
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"Success! Saved to {filepath}")
        else:
            print(f"Failed to fetch {name}: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error fetching {name}: {str(e)}")

if __name__ == "__main__":
    for name, code in diagrams.items():
        fetch_diagram(name, code)
    print("\nAll diagrams processed. You can now insert these PNG files into Word.")
