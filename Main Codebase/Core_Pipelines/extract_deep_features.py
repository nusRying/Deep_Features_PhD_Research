import sys
import os
from pathlib import Path
# Ensure project_paths.py is found at the root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


"""
Deep Feature Extractor (ResNet50)
---------------------------------
Extracts 2048-dimensional embeddings from images using a pre-trained ResNet50 model.
The final classification layer is removed to obtain the feature vector.

Target: ISIC 2019 Dataset
Output: deep_features.csv
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2  # Added for Hair Removal
from project_paths import DEEP_FEATURES_FILE

# --- CONFIGURATION ---
# HARDCODED IMAGE PATH AS REQUESTED
IMAGE_DIRECTORY = Path(r"C:\Users\umair\Videos\PhD\PhD Data\Week 15 January\Code V2\Dataset\clean\CleanData\ISIC2019\images_train")
OUTPUT_FILE = DEEP_FEATURES_FILE
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_hair_removal(img_np):
    """
    Remove hair using morphological BlackHat transformation (DullRazor).
    Input: RGB Numpy Array (uint8)
    Output: RGB Numpy Array (uint8) - Hair Removed
    """
    if img_np is None: return None
    
    # 1. Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # 2. Kernel for morphological operations (size tuned for 224x224)
    # Using 9x9 as a robust default
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    
    # 3. BlackHat (Original - Closing) = finds dark details (clean hairs)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # 4. Thresholding to create hair mask
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # 5. Inpaint using the mask (INPAINT_TELEA is fast and good)
    inpainted = cv2.inpaint(img_np, thresh, 1, cv2.INPAINT_TELEA)
    
    return inpainted

def get_resnet_extractor():
    """Returns ResNet50 model with the last fc layer removed."""
    print(f"Loading ResNet50 on {DEVICE}...")
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
    
    # Remove the fully connected layer (classification head)
    # ResNet50 architecture: ... -> avgpool -> fc
    # We want the output of avgpool (2048 dims)
    modules = list(model.children())[:-1]
    model = nn.Sequential(*modules)
    
    model.to(DEVICE)
    model.eval()
    return model

def get_transform():
    """Standard ImageNet preprocessing."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def extract_features():
    if not IMAGE_DIRECTORY.exists():
        print(f"ERROR: Image directory not found at {IMAGE_DIRECTORY}")
        return

    print(f"Scanning for images in {IMAGE_DIRECTORY}...")
    # Support jpg and png
    image_files = list(IMAGE_DIRECTORY.glob("*.jpg")) + list(IMAGE_DIRECTORY.glob("*.png")) + list(IMAGE_DIRECTORY.glob("*.jpeg"))
    
    if not image_files:
        print("No images found! Check path and extensions.")
        return

    print(f"Found {len(image_files)} images.")
    
    # Setup Model
    model = get_resnet_extractor()
    preprocess = get_transform()
    
    features_list = []
    image_ids = []
    
    # Process in batches
    batch_tensor = []
    batch_names = []
    
    print("Starting Extraction with Hair Removal...")
    with torch.no_grad():
        for img_path in tqdm(image_files):
            try:
                # 1. Load PIL
                img_pil = Image.open(img_path).convert("RGB")
                
                # 2. Convert to Numpy for Hair Removal
                img_np = np.array(img_pil)
                
                # 3. Apply Hair Removal
                img_clean_np = apply_hair_removal(img_np)
                
                # 4. Convert back to PIL for Torch Transforms
                img_clean = Image.fromarray(img_clean_np)
                
                # 5. Standard Preprocessing
                tensor = preprocess(img_clean)
                
                batch_tensor.append(tensor)
                # Store ID as filename without extension (common key)
                batch_names.append(img_path.stem) 
                
                if len(batch_tensor) >= BATCH_SIZE:
                    # Run Batch
                    input_batch = torch.stack(batch_tensor).to(DEVICE)
                    output = model(input_batch) 
                    # Output shape: [BATCH, 2048, 1, 1] -> Flatten to [BATCH, 2048]
                    embeddings = output.squeeze(-1).squeeze(-1).cpu().numpy()
                    
                    features_list.append(embeddings)
                    image_ids.extend(batch_names)
                    
                    # Reset
                    batch_tensor = []
                    batch_names = []
                    
            except Exception as e:
                print(f"Skipping {img_path.name}: {e}")

        # Process final partial batch
        if batch_tensor:
            input_batch = torch.stack(batch_tensor).to(DEVICE)
            output = model(input_batch)
            embeddings = output.squeeze(-1).squeeze(-1).cpu().numpy()
            features_list.append(embeddings)
            image_ids.extend(batch_names)

    # Compile DataFrame
    print("Compiling Results...")
    if not features_list:
        print("No features extracted.")
        return

    all_features = np.vstack(features_list)
    
    # Create Columns: deep_0, deep_1, ... deep_2047
    feat_cols = [f"deep_{i}" for i in range(all_features.shape[1])]
    
    df = pd.DataFrame(all_features, columns=feat_cols)
    df.insert(0, "image_id", image_ids)
    
    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"SUCCESS: Saved {len(df)} records to {OUTPUT_FILE}")
    print(f"Feature vector size: {all_features.shape[1]}")

if __name__ == "__main__":
    extract_features()
