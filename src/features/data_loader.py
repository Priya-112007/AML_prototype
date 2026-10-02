import os
import pandas as pd
import numpy as np
import io
from PIL import Image
from src.preprocessing.image_processing import preprocess_image
from src.features.feature_extraction import extract_lesion_features

def get_features_and_labels(parquet_path, cache_file):
    if os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}...")
        data = np.load(cache_file)
        return data['X'], data['y']
        
    print(f"Processing images from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    features = []
    labels = []
    
    for idx, row in df.iterrows():
        try:
            image_data = row['image']['bytes']
            img = Image.open(io.BytesIO(image_data))
            img_np = np.array(img)
            
            _, y_stretched = preprocess_image(img_np)
            feats = extract_lesion_features(y_stretched)
            
            features.append(feats)
            labels.append(row['label'])
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} images...")
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            
    X = np.array(features)
    y = np.array(labels)
    np.savez(cache_file, X=X, y=y)
    return X, y
