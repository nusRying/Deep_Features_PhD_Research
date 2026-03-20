import sys
import os
from pathlib import Path
# Ensure project_paths.py is found at the root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from project_paths import EXSTRACS_LIB_DIR, HYBRID_MODEL_FILE, HYBRID_IMPORTANCE_FILE, HYBRID_TOP_RULES_FILE

# Setup Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(EXSTRACS_LIB_DIR))
from skExSTraCS.ExSTraCS import ExSTraCS

MODEL_PATH = HYBRID_MODEL_FILE

def analyze_model():
    if not MODEL_PATH.exists():
        print(f"Error: {MODEL_PATH} not found.")
        return

    print("Loading model data...")
    with open(MODEL_PATH, 'rb') as f:
        metrics = pickle.load(f)

    # 1. Define Feature Names
    deep_names = [f"deep_{i}" for i in range(2048)]
    handcrafted_names = [
        "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", 
        "glcm_energy", "glcm_correlation", "glcm_ASM",
        "lbp_bin0", "lbp_bin1", "lbp_bin2", "lbp_bin3", "lbp_bin4",
        "lbp_bin5", "lbp_bin6", "lbp_bin7", "lbp_bin8", "lbp_bin9"
    ]
    metadata_names = [
        "age", "sex_female", "sex_male", "sex_unknown", 
        "site_head_neck", "site_lower_extremity", "site_torso", 
        "site_unknown", "site_upper_extremity"
    ]
    all_feature_names = deep_names + handcrafted_names + metadata_names
    print(f"Total feature names defined: {len(all_feature_names)}")

    # 2. Extract Attribute Importance
    if len(metrics) > 15:
        at_obj = metrics[15]
        env_obj = metrics[16]
        print("Extracting attribute tracking scores...")
        try:
            num_atts = env_obj.formatData.numAttributes
            num_inst = env_obj.formatData.numTrainInstances
            
            globalAttTrack = [0.0 for _ in range(num_atts)]
            for i in range(num_atts):
                for j in range(num_inst):
                    globalAttTrack[i] += at_obj.attAccuracySums[j][i]
            
            importance_df = pd.DataFrame({
                'Feature': all_feature_names[:len(globalAttTrack)],
                'Importance': globalAttTrack
            }).sort_values(by='Importance', ascending=False)

            print("\nTop 20 Features by Importance:")
            print(importance_df.head(20))
            importance_df.to_csv(HYBRID_IMPORTANCE_FILE, index=False)
        except Exception as e:
            print(f"Error extracting importance: {e}")

    # 3. Extract Top Rules
    print("\nExtracting Top Rules...")
    if len(metrics) > 17:
        pop = metrics[17] # list of classifier objects
        print(f"Found population. Rules: {len(pop)}")
        
        try:
            # Sort by fitness
            pop.sort(key=lambda x: x.fitness, reverse=True)
            
            with open(HYBRID_TOP_RULES_FILE, 'w') as f:
                f.write("Top 20 Rules by Fitness\n")
                f.write("========================\n\n")
                for i, cl in enumerate(pop[:20]):
                    f.write(f"Rule {i+1} (Fitness: {cl.fitness:.4f}, Accuracy: {cl.accuracy:.4f}, Numerosity: {cl.numerosity})\n")
                    cond_parts = []
                    for j in range(len(cl.condition)):
                        att_idx = cl.specifiedAttList[j]
                        val = cl.condition[j]
                        name = all_feature_names[att_idx] if att_idx < len(all_feature_names) else f"attr_{att_idx}"
                        cond_parts.append(f"{name}:{val}")
                    f.write(f"  Condition: {', '.join(cond_parts)}\n")
                    f.write(f"  Action (Malignant=1): {cl.phenotype}\n\n")
            print("Top rules saved to hybrid_top_rules.txt")
        except Exception as e:
            print(f"Error sorting/writing rules: {e}")
    else:
        print("Population not found in serialized metrics.")

if __name__ == "__main__":
    analyze_model()
