
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import gc
from project_paths import (
    CUSTOM_LABELS_FILE,
    DEEP_FEATURES_FILE,
    HYBRID_ISIC_FILE,
    HYBRID_HAM_FILE,
    TEST_HYBRID_ISIC_FILE,
    TEST_HYBRID_HAM_FILE,
)

# --- CONFIGURATION ---
ISIC_IMG_DIR = Path(r"C:\Users\umair\Videos\PhD\PhD Data\Week 15 January\Code V2\Dataset\clean\CleanData\ISIC2019\images_train")
HAM_IMG_DIR = Path(r"C:\Users\umair\Videos\PhD\PhD Data\Week 15 January\Code V2\Dataset\clean\CleanData\HAM10000\images")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

# Labels
ISIC_LABELS = CUSTOM_LABELS_FILE
HAM_METADATA = Path(r"C:\Users\umair\Videos\PhD\PhD Data\Week 15 January\Code V2\Dataset\clean\CleanData\HAM10000\HAM10000_metadata")

# Pre-existing deep features for ISIC to save time
ISIC_DEEP_EXISTING = DEEP_FEATURES_FILE

def apply_hair_removal(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img_np, thresh, 1, cv2.INPAINT_TELEA)

def extract_geometry_features(img_np):
    # Lesion Segmentation
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Robust segmentation with Otsu
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Fallback if masking is obviously wrong
    if np.sum(thresh) < 100 or np.sum(thresh) > (thresh.size * 0.9 * 255):
        _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)

    # Clean mask (morphology)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0.0] * 7
    
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    if area == 0 or perimeter == 0:
        return [0.0] * 7
    
    # 1. Compactness (Isoperimetric Quotient)
    compactness = (4 * np.pi * area) / (perimeter ** 2)
    
    # 2. Asymmetry (Horizontal & Vertical)
    x, y, w, h = cv2.boundingRect(cnt)
    roi_mask = mask[y:y+h, x:x+w]
    h_flip = cv2.flip(roi_mask, 1)
    v_flip = cv2.flip(roi_mask, 0)
    asym_h = np.sum(cv2.absdiff(roi_mask, h_flip)) / (area * 255 + 1e-6)
    asym_v = np.sum(cv2.absdiff(roi_mask, v_flip)) / (area * 255 + 1e-6)
    
    # 3. Eccentricity
    ellipse = cv2.fitEllipse(cnt) if len(cnt) >= 5 else None
    if ellipse:
        (ex, ey), (MA, ma), angle = ellipse
        eccentricity = np.sqrt(1 - (min(MA, ma)**2 / max(MA, ma)**2)) if max(MA, ma) > 0 else 0
    else:
        eccentricity = 0
        
    # 4. Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # 5. Border Irregularity (Perimeter/Hull Perimeter)
    hull_perimeter = cv2.arcLength(hull, True)
    border_irreg = perimeter / hull_perimeter if hull_perimeter > 0 else 1.0
    
    # 6. Extent
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0
    
    return [compactness, asym_h, asym_v, eccentricity, solidity, border_irreg, extent]

def extract_color_features(img_np):
    # img_np is RGB
    # Calculate Mean, Std, and Skew for each channel
    from scipy.stats import skew
    feats = []
    for i in range(3): # R, G, B
        channel = img_np[:,:,i]
        feats.append(np.mean(channel))
        feats.append(np.std(channel))
        feats.append(skew(channel.ravel()))
    return feats

def extract_handcrafted(img_np):
    # 1. Color Features (9: mean, std, skew for R, G, B)
    color_feats = extract_color_features(img_np)
    
    # 2. Geometry Features (7: compactness, asym_h, asym_v, eccentricity, solidity, border_irreg, extent)
    geom_feats = extract_geometry_features(img_np)
    
    # 3. Texture Features
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    scaled = (gray.astype(np.float32) / 255.0) * 15
    scaled = np.clip(scaled, 0, 15).astype(np.uint8)
    glcm = graycomatrix(scaled, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=16, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    glcm_feats = [graycoprops(glcm, p).mean() for p in props]
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    
    # Total: 9 (color) + 7 (geom) + 6 (glcm) + 10 (lbp) = 32 features
    return color_feats + geom_feats + glcm_feats + hist.tolist()

def worker_handcrafted(img_path):
    try:
        # print(f"Processing {img_path.name}...")
        pil_img = Image.open(img_path).convert("RGB")
        img_np = np.array(pil_img)
        img_clean_np = apply_hair_removal(img_np)
        feats = extract_handcrafted(img_clean_np)
        return img_path.stem, feats
    except Exception as e:
        print(f"Error in worker_handcrafted for {img_path}: {e}")
        return img_path.stem, None

def get_resnet_extractor():
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
    model = nn.Sequential(*list(model.children())[:-1])
    model.to(DEVICE).eval()
    return model

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_dataset(image_dir, output_file, is_isic=False, limit=None):
    print(f"\nProcessing {'ISIC' if is_isic else 'HAM10000'} from {image_dir}...")
    image_files = list(image_dir.glob("*.jpg"))
    if limit:
        image_files = image_files[:limit]
        print(f"Limiting to first {limit} images for test run.")
    if not image_files: return
    
    # Check for existing partial file
    existing_records = {}
    if Path(output_file).exists():
        print(f"Loading existing records from {output_file}...")
        df_old = pd.read_csv(output_file)
        existing_records = {row['image_id']: row.tolist()[1:] for _, row in df_old.iterrows()}
        image_files = [f for f in image_files if f.stem not in existing_records]
        print(f"New images to process: {len(image_files)}")

    if not image_files:
        print("All images already processed.")
        return

    # 1. Multi-threaded Handcrafted Extraction
    print(f"Extracting Handcrafted Features (Multiprocessing)...")
    handcrafted_data = {}
    # 1. Multi-threaded Handcrafted Extraction
    print(f"Extracting Handcrafted Features (Multiprocessing)...")
    handcrafted_data = {}
    
    # Use a fixed number of workers to avoid BrokenProcessPool on some systems
    import sys
    # If test run is enabled from __main__, we might need a way to pass it here.
    # For now, I'll just check if 'test-run' is in sys.argv
    max_w = 1 if '--test-run' in sys.argv else min(4, multiprocessing.cpu_count())
    
    if max_w > 1:
        with ProcessPoolExecutor(max_workers=max_w) as executor:
            futures = {executor.submit(worker_handcrafted, f): f for f in image_files}
            for future in tqdm(as_completed(futures), total=len(image_files)):
                try:
                    img_id, feats = future.result()
                    if feats:
                        handcrafted_data[img_id] = feats
                except Exception as e:
                    print(f"Error processing a file: {e}")
    else:
        print("Running sequentially (max_w=1)...")
        for f in tqdm(image_files):
            img_id, feats = worker_handcrafted(f)
            if feats:
                handcrafted_data[img_id] = feats

    # 2. Deep Feature Extraction (Sequential with Batching)
    print(f"Handling Deep Features...")
    df_deep = None
    if is_isic:
        if ISIC_DEEP_EXISTING.exists():
            print("Loading existing ISIC Deep Features...")
            # Optimization: Only load what we need or just load the whole thing if it's manageable
            df_deep = pd.read_csv(ISIC_DEEP_EXISTING).set_index('image_id')
        else:
            print("WARNING: Existing ISIC deep features not found!")
    else:
        print("Extracting Deep Features for HAM (Batched)...")
        model = get_resnet_extractor()
        ham_deep = {}
        batch_ids = []
        batch_tensors = []
        
        # Filter to only images we have handcrafted features for
        images_to_process = [f for f in image_files if f.stem in handcrafted_data]
        
        for img_path in tqdm(images_to_process):
            img_id = img_path.stem
            
            try:
                pil_img = Image.open(img_path).convert("RGB")
                img_np = np.array(pil_img)
                img_clean_np = apply_hair_removal(img_np)
                tensor = preprocess(Image.fromarray(img_clean_np))
                
                batch_tensors.append(tensor)
                batch_ids.append(img_id)
                
                if len(batch_tensors) >= BATCH_SIZE:
                    with torch.no_grad():
                        out = model(torch.stack(batch_tensors).to(DEVICE)).squeeze().cpu().numpy()
                    if out.ndim == 1: # Handle single image batch squeeze
                        out = out.reshape(1, -1)
                    for i, d in enumerate(out):
                        ham_deep[batch_ids[i]] = d.tolist()
                    batch_tensors, batch_ids = [], []
            except Exception as e:
                print(f"Deep Feature Error for {img_id}: {e}")
            
        if batch_tensors:
            with torch.no_grad():
                out = model(torch.stack(batch_tensors).to(DEVICE)).squeeze().cpu().numpy()
            if out.ndim == 1:
                out = out.reshape(1, -1)
            for i, d in enumerate(out):
                ham_deep[batch_ids[i]] = d.tolist()
                
        df_deep = pd.DataFrame.from_dict(ham_deep, orient='index', columns=[f"deep_{i}" for i in range(2048)])
        df_deep.index.name = 'image_id'

    # 3. Compile & Save (Memory Efficient)
    print(f"Compiling results for {len(handcrafted_data)} images...")
    
    # Create handcrafted dataframe
    df_hand = pd.DataFrame.from_dict(handcrafted_data, orient='index', columns=[f"hand_{i}" for i in range(32)])
    df_hand.index.name = 'image_id'
    
    # Free up the dictionary memory
    del handcrafted_data
    gc.collect()

    # Join with deep features
    if df_deep is not None:
        print("Joining with deep features...")
        # Use inner join to ensure we only save images that have both feature sets
        df_new = df_hand.join(df_deep, how='inner').reset_index()
        del df_hand
        del df_deep
        gc.collect()
    else:
        df_new = df_hand.reset_index().rename(columns={'index': 'image_id'})
        del df_hand
        gc.collect()

    print(f"Saving to {output_file} (Shape: {df_new.shape})...")
    # Write in chunks to keep memory low during string conversion
    df_new.to_csv(output_file, index=False, chunksize=1000)
    
    # Final cleanup
    del df_new
    gc.collect()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true", help="Run on small subset")
    args = parser.parse_args()

    num_test = 20 if args.test_run else None
        
    # Force re-extraction for full consistency
    if not args.test_run:
        for f in [HYBRID_ISIC_FILE, HYBRID_HAM_FILE]:
            if f.exists():
                print(f"Removing old {f} for clean extraction...")
                os.remove(f)
        
    # 1. Process ISIC
    out_isic = TEST_HYBRID_ISIC_FILE if args.test_run else HYBRID_ISIC_FILE
    process_dataset(ISIC_IMG_DIR, out_isic, is_isic=True, limit=num_test)
    
    # 2. Process HAM10000
    out_ham = TEST_HYBRID_HAM_FILE if args.test_run else HYBRID_HAM_FILE
    process_dataset(HAM_IMG_DIR, out_ham, is_isic=False, limit=num_test)
