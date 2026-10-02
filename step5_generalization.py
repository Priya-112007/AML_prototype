import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing.image_processing import preprocess_image
from src.features.feature_extraction import extract_lesion_features

def main():
    print("Loading final model for generalization test...")
    model_path = "results/final_hybrid_model.pkl"
    if not os.path.exists(model_path):
        print("Model not found.")
        return
        
    clf = joblib.load(model_path)
    
    # Load APTOS Data
    csv_path = "data/aptos/train.csv"
    if not os.path.exists(csv_path):
        print(f"APTOS CSV not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Setup CNN extractor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    effnet.classifier = torch.nn.Identity()
    effnet.to(device)
    effnet.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Evaluating on APTOS 2019 dataset...")
    
    y_true = []
    y_pred = []
    
    # Limit to 100 images for time
    df = df.head(100)
    
    for idx, row in df.iterrows():
        id_code = row['id_code']
        label = row['diagnosis']
        
        img_path = f"data/aptos/gaussian_filtered_images/gaussian_filtered_images/{id_code}.png"
        if not os.path.exists(img_path):
            img_path = f"data/aptos/train_images/{id_code}.png"
            if not os.path.exists(img_path):
                continue
                
        try:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)
            
            # Classical
            _, y_stretched = preprocess_image(img_np)
            cls_feats = extract_lesion_features(y_stretched)
            
            # CNN
            img_t = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                cnn_feats = effnet(img_t).cpu().numpy().flatten()
                
            fused_feats = np.concatenate((cls_feats, cnn_feats)).reshape(1, -1)
            pred = clf.predict(fused_feats)[0]
            
            y_true.append(label)
            y_pred.append(pred)
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1} APTOS images...")
        except Exception as e:
            print(f"Error on {id_code}: {e}")
            
    acc = accuracy_score(y_true, y_pred)
    print(f"\nGeneralization Accuracy on APTOS: {acc:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

if __name__ == "__main__":
    main()
