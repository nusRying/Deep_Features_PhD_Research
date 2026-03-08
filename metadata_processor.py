
import pandas as pd
import numpy as np
from pathlib import Path
from project_paths import CUSTOM_LABELS_FILE, ISIC_METADATA_FILE, HAM_METADATA_FILE

SITE_MAPPING = {
    'head/neck': 'head_neck', 'scalp': 'head_neck', 'ear': 'head_neck', 'face': 'head_neck', 'neck': 'head_neck',
    'anterior torso': 'torso', 'posterior torso': 'torso', 'trunk': 'torso', 'chest': 'torso', 'back': 'torso', 
    'abdomen': 'torso', 'genital': 'torso',
    'upper extremity': 'upper_extremity', 'hand': 'upper_extremity',
    'lower extremity': 'lower_extremity', 'foot': 'lower_extremity',
    'palms/soles': 'lower_extremity', 'lateral torso': 'torso', 'oral/genital': 'torso'
}

def process_isic_metadata(meta_path, label_path):
    df_meta = pd.read_csv(meta_path)
    df_labels = pd.read_csv(label_path)
    df = pd.merge(df_meta, df_labels[['image', 'label']], on='image', how='inner')
    
    df['age_approx'] = df['age_approx'].fillna(df['age_approx'].median())
    df['sex'] = df['sex'].fillna('unknown').str.lower()
    df['site'] = df['anatom_site_general'].fillna('unknown').str.lower().map(SITE_MAPPING).fillna('unknown')
    
    df = pd.get_dummies(df, columns=['sex', 'site'], prefix=['sex', 'site'])
    df = df.rename(columns={'image': 'image_id', 'age_approx': 'age'})
    return df

def process_ham_metadata(meta_path):
    df = pd.read_csv(meta_path)
    malignant_types = ['mel', 'bcc', 'akiec']
    df['label'] = df['dx'].apply(lambda x: 1 if x in malignant_types else 0)
    
    df['age'] = df['age'].fillna(df['age'].median())
    df['sex'] = df['sex'].fillna('unknown').str.lower()
    df['site'] = df['localization'].fillna('unknown').str.lower().map(SITE_MAPPING).fillna('unknown')
    
    df = pd.get_dummies(df, columns=['sex', 'site'], prefix=['sex', 'site'])
    return df

if __name__ == "__main__":
    ISIC_META = Path(r"C:\Users\umair\Videos\PhD\PhD Data\Week 15 January\Code V2\Dataset\clean\CleanData\ISIC2019\ISIC_2019_Training_Metadata.csv")
    ISIC_LABELS = CUSTOM_LABELS_FILE
    HAM_META = Path(r"C:\Users\umair\Videos\PhD\PhD Data\Week 15 January\Code V2\Dataset\clean\CleanData\HAM10000\HAM10000_metadata")
    
    print("Processing ISIC Metadata...")
    df_isic = process_isic_metadata(ISIC_META, ISIC_LABELS)
    df_isic.to_csv(ISIC_METADATA_FILE, index=False)
    
    print("Processing HAM Metadata...")
    df_ham = process_ham_metadata(HAM_META)
    df_ham.to_csv(HAM_METADATA_FILE, index=False)
    
    print("Metadata processing complete.")
