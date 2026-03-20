import pickle
import sys
import numpy as np
from pathlib import Path

# Add external lib to path
EXSTRACS_LIB_DIR = Path(r"c:\Users\umair\Videos\PhD\PhD Data\Week 5 Febuary\Deep_Features_Experiment\standalone_hybrid_v2_safety\external\scikit-ExSTraCS-master")
sys.path.insert(0, str(EXSTRACS_LIB_DIR))

# Path to an expert
CKPT_PATH = Path(r"c:\Users\umair\Videos\PhD\PhD Data\Week 5 Febuary\Deep_Features_Experiment\standalone_hybrid_v2_safety\models\checkpoints_opt\expert_0_seed_42.pkl")

print(f"Loading checkpoint: {CKPT_PATH.name}...")
with open(CKPT_PATH, 'rb') as f:
    model = pickle.load(f)

# Extract attribute tracking sums
# Based on ExSTraCS code: self.AT.getSumGlobalAttTrack(self) returns the sums
try:
    attr_sums = model.get_final_attribute_tracking_sums()
    print(f"Number of attributes tracked: {len(attr_sums)}")
    print(f"Top 10 attribute sums: {sorted(attr_sums, reverse=True)[:10]}")
    print(f"Bottom 10 attribute sums: {sorted(attr_sums)[:10]}")
    
    # Save the sums for analysis in the main script implementation later
    np.save("attr_tracking_sums.npy", attr_sums)
    print("Saved attribute sums to attr_tracking_sums.npy")
except Exception as e:
    print(f"Error extracting attribute sums: {e}")
