import subprocess
import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PHASES = [
    "exp_v1_baseline",
    "exp_a_safety_first",
    "exp_b_conservative",
    "exp_v4_stacking",
    "exp_v5_calibration",
    "exp_c_balanced_efficiency",
    "exp_d_knowledge_discovery",
    "exp_v8_evidential_uncertainty"
]

def run_phd_pipeline(smoke_test=True):
    root_dir = Path(__file__).parent.resolve()
    logger.info(f"--- STARTING FULL PHD PIPELINE (Smoke Test: {smoke_test}) ---")
    
    for phase in PHASES:
        phase_dir = root_dir / phase
        if not phase_dir.exists():
            logger.warning(f"Phase folder {phase} not found. Skipping...")
            continue
            
        logger.info(f"\n🚀 EXECUTING: {phase}")
        cmd = [sys.executable, "main.py"]
        if smoke_test:
            cmd.append("--smoke-test")
            
        try:
            # Run within the directory
            subprocess.run(cmd, cwd=str(phase_dir), check=True)
            logger.info(f"✅ COMPLETED: {phase}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ FAILED: {phase} with error {e}")
            sys.exit(1)

    logger.info("\n--- PHD PIPELINE COMPLETE ---")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", default=True)
    parser.add_argument("--full-run", action="store_false", dest="smoke_test")
    args = parser.parse_args()
    
    run_phd_pipeline(smoke_test=args.smoke_test)
