import os
import pandas as pd
import io
from PIL import Image

def main():
    # 0. Create directory structure
    dirs = [
        "src/preprocessing",
        "src/features",
        "src/models",
        "src/evaluation",
        "notebooks",
        "results"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")

    # Load parquets
    print("Loading IDRiD parquets...")
    train_path = "data/idrid/train-00000-of-00001-d81b05cdbfbe95cd.parquet"
    test_path = "data/idrid/test-00000-of-00001-3eaaa0286c6f240e.parquet"
    
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    
    print("Train columns:", df_train.columns)
    
    # Check label distribution
    print("\nTrain label distribution:")
    if 'label' in df_train.columns:
        print(df_train['label'].value_counts().sort_index())
    
    print("\nTest label distribution:")
    if 'label' in df_test.columns:
        print(df_test['label'].value_counts().sort_index())
        
    # Check image decoding
    print("\nDecoding a sample image from train...")
    sample_row = df_train.iloc[0]
    
    if 'image' in sample_row:
        image_data = sample_row['image']
        if isinstance(image_data, dict) and 'bytes' in image_data:
            img = Image.open(io.BytesIO(image_data['bytes']))
            print(f"Image decoded successfully. Size: {img.size}, Mode: {img.mode}")
        else:
            print("Image format different than expected dict with 'bytes'. Type:", type(image_data))
    else:
        print("No 'image' column found.")

if __name__ == "__main__":
    main()
