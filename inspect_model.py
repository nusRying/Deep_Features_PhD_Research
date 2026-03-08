import sys
import pickle
from pathlib import Path
from project_paths import HYBRID_MODEL_FILE

# Setup Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_PATH = HYBRID_MODEL_FILE

def inspect_pickle():
    if not MODEL_PATH.exists():
        print("Model file missing.")
        return

    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded Type: {type(data)}")
    if isinstance(data, list):
        print(f"List length: {len(data)}")
        for i, item in enumerate(data):
            print(f"Item {i} type: {type(item)}")
            # If it's the model, check its attributes
            if "ExSTraCS" in str(type(item)):
                print(f"Found ExSTraCS model at index {i}")
                print(f"Attributes: {dir(item)}")
    elif hasattr(data, 'model'):
        print(f"Data has 'model' attribute: {type(data.model)}")
    else:
        print(f"Attributes of data: {dir(data)}")

if __name__ == "__main__":
    inspect_pickle()
