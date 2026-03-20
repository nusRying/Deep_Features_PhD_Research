import sys
import os
from pathlib import Path
# Ensure project_paths.py is found at the root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

"""
Script to prepare custom binary labels for ISIC 2019.
Mapping:
- Malignant (1): 'MEL', 'BCC', 'SCC'
- Benign (0): 'NV', 'BKL', 'DF', 'VASC'
- Excluded: 'AK', 'UNK'
"""

import pandas as pd
import numpy as np
from pathlib import Path
from project_paths import CUSTOM_LABELS_FILE

# Paths
# Adjust this path if the week folder name is different in your environment, 
# but based on history it matches "Week 15 January"
SOURCE_GT_PATH = Path(r"C:\Users\umair\Videos\PhD\PhD Data\Week 15 January\Code V2\Dataset\clean\CleanData\ISIC2019\ISIC_2019_Training_GroundTruth.csv")
OUTPUT_PATH = CUSTOM_LABELS_FILE

def main():
    print(f"Reading Ground Truth from: {SOURCE_GT_PATH}")
    if not SOURCE_GT_PATH.exists():
        print(f"Error: Source file not found!")
        return

    df = pd.read_csv(SOURCE_GT_PATH)
    print(f"Total rows: {len(df)}")
    
    # Define groups
    malignant_classes = ['MEL', 'BCC', 'SCC']
    benign_classes = ['NV', 'BKL', 'DF', 'VASC']
    
    # Valid columns check
    for c in malignant_classes + benign_classes:
        if c not in df.columns:
            print(f"Error: Column {c} not found in CSV. columns: {df.columns}")
            return

    # Create Label Column
    # 1. Identify Malignant Rows (any of the malignant cols is 1.0)
    # Using max(axis=1) to check if any col is 1
    is_malignant = df[malignant_classes].max(axis=1) == 1
    
    # 2. Identify Benign Rows
    is_benign = df[benign_classes].max(axis=1) == 1
    
    # 3. Filter
    # We only want rows that represent distinct classes in our sets.
    # Note: ISIC rows should be one-hot, so a row shouldn't be both (unless multi-label, which ISIC2019 is generally single)
    
    # Initialize output dataframe
    # We will prioritize explicit matches.
    
    output_rows = []
    
    malignant_count = 0
    benign_count = 0
    
    for idx, row in df.iterrows():
        # Check Malignant
        is_mal = any(row[c] == 1.0 for c in malignant_classes)
        # Check Benign
        is_ben = any(row[c] == 1.0 for c in benign_classes)
        
        # Determine label
        if is_mal and not is_ben:
            lab = 1
            malignant_count += 1
        elif is_ben and not is_mal:
            lab = 0
            benign_count += 1
        else:
            # Either excluded class (AK, UNK) or ambiguous
            continue
            
        output_rows.append({
            'image': row['image'],
            'label': lab
        })
        
    df_out = pd.DataFrame(output_rows)
    
    print("\nGeneration Complete:")
    print(f"  Malignant (1): {malignant_count}")
    print(f"  Benign (0):    {benign_count}")
    print(f"  Total Saved:   {len(df_out)}")
    print(f"  Excluded:      {len(df) - len(df_out)}")
    
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
