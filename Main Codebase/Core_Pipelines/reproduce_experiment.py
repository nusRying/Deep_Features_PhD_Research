import sys
import os
from pathlib import Path
# Ensure project_paths.py is found at the root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


"""
Deep Features Experiment: Master Orchestrator
---------------------------------------------
Runs the complete scientific pipeline:
1. Feature Extraction (ResNet50)
2. Feature Selection (Mutual Info)
3. Model Training (ExSTraCS)
"""

import subprocess
import sys
import time
from pathlib import Path

def run_step(script_name):
    print(f"\n{'='*60}")
    print(f" STARTING: {script_name}")
    print(f"{'='*60}")
    
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        print(f"CRITICAL: {script_name} not found!")
        return False
        
    start = time.time()
    # Execute
    try:
        # Use sys.executable to ensure we use the same python env
        result = subprocess.run([sys.executable, str(script_path)], check=False)
        duration = time.time() - start
        
        if result.returncode == 0:
            print(f">>> SUCCESS: {script_name} (Time: {duration:.1f}s)")
            return True
        else:
            print(f">>> FAILED: {script_name} (Exit Code: {result.returncode})")
            return False
    except Exception as e:
        print(f">>> EXECUTION ERROR: {e}")
        return False

def main():
    print("--- DEEP FEATURES EXPERIMENT AUTOMATION ---\n")
    
    # Step 1: Extraction
    # This is the heavy lifting.
    if not run_step("extract_deep_features.py"):
        print("Aborting experiment due to extraction failure.")
        return
        
    # Step 2: Selection
    if not run_step("pipeline_feature_selection.py"):
        print("Aborting experiment due to selection failure.")
        return
        
    # Step 3: Training
    if not run_step("pipeline_retrain_model.py"):
        print("Aborting experiment due to training failure.")
        return
        
    print("\n" + "="*60)
    print(" EXPERIMENT COMPLETE ")
    print("="*60)

if __name__ == "__main__":
    main()
