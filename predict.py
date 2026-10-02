import sys
import os
import argparse
import joblib
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing.image_processing import preprocess_image
from src.features.feature_extraction import extract_lesion_features

def predict(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return
        
    print(f"Loading final model...")
    model_path = "results/final_hybrid_model.pkl"
    if not os.path.exists(model_path):
        print("Model not found. Please run the training pipeline first.")
        return
        
    clf = joblib.load(model_path)
    
    # Process image
    print("Extracting classical features...")
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    _, y_stretched = preprocess_image(img_np)
    cls_feats = extract_lesion_features(y_stretched)
    
    print("Extracting CNN features...")
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
    
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        cnn_feats = effnet(img_t).cpu().numpy().flatten()
        
    # Combine
    fused_feats = np.concatenate((cls_feats, cnn_feats)).reshape(1, -1)
    
    # Predict
    pred = clf.predict(fused_feats)[0]
    classes = ["0: No DR", "1: Mild", "2: Moderate", "3: Severe", "4: Proliferative DR"]
    print(f"\n--- Prediction ---")
    print(f"Predicted DR Severity Grade: {classes[pred]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict DR severity from an image")
    parser.add_argument("image_path", type=str, help="Path to the fundus image")
    args = parser.parse_args()
    predict(args.image_path)
