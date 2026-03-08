
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew

def apply_hair_removal(img_np):
    print("Starting hair removal...")
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    res = cv2.inpaint(img_np, thresh, 1, cv2.INPAINT_TELEA)
    print("Hair removal done.")
    return res

def extract_geometry_features(img_np):
    print("Starting geometry features...")
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.sum(thresh) < 100:
        _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0.0] * 7
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    x, y, w, h = cv2.boundingRect(cnt)
    roi_mask = mask[y:y+h, x:x+w]
    h_flip = cv2.flip(roi_mask, 1)
    asym_h = np.sum(cv2.absdiff(roi_mask, h_flip)) / (area * 255 + 1e-6)
    ellipse = cv2.fitEllipse(cnt) if len(cnt) >= 5 else None
    ecc = 0
    if ellipse:
        (ex, ey), (MA, ma), angle = ellipse
        ecc = np.sqrt(1 - (min(MA, ma)**2 / max(MA, ma)**2)) if max(MA, ma) > 0 else 0
    hull = cv2.convexHull(cnt)
    solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
    print("Geometry features done.")
    return [compactness, asym_h, ecc, solidity]

img_dir = Path(r"C:\Users\umair\Videos\PhD\PhD Data\Week 15 January\Code V2\Dataset\clean\CleanData\ISIC2019\images_train")
img_path = next(img_dir.glob("*.jpg"))
print(f"Loading {img_path}")
pil_img = Image.open(img_path).convert("RGB")
img_np = np.array(pil_img)
img_clean = apply_hair_removal(img_np)
geom = extract_geometry_features(img_clean)
print(f"Results: {geom}")
