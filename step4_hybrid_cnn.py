import os
import sys
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
import io
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.features.data_loader import get_features_and_labels

def extract_cnn_features(df, model, transform, device):
    features = []
    model.eval()
    with torch.no_grad():
        for idx, row in df.iterrows():
            image_data = row['image']['bytes']
            img = Image.open(io.BytesIO(image_data)).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)
            feat = model(img_t).cpu().numpy().flatten()
            features.append(feat)
            
            if (idx + 1) % 100 == 0:
                print(f"CNN Extracted {idx + 1} images...")
                
    return np.array(features)

def main():
    train_path = "data/idrid/train-00000-of-00001-d81b05cdbfbe95cd.parquet"
    test_path = "data/idrid/test-00000-of-00001-3eaaa0286c6f240e.parquet"
    
    # 1. Classical features
    X_train_cls, y_train = get_features_and_labels(train_path, "data/idrid/train_features.npz")
    X_test_cls, y_test = get_features_and_labels(test_path, "data/idrid/test_features.npz")
    
    # 2. CNN features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    # Strip classifier
    effnet.classifier = torch.nn.Identity()
    effnet.to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2), # Augmentation as requested
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    cnn_train_cache = "data/idrid/train_cnn_features.npz"
    cnn_test_cache = "data/idrid/test_cnn_features.npz"
    
    if os.path.exists(cnn_train_cache):
        print("Loading CNN train features from cache...")
        X_train_cnn = np.load(cnn_train_cache)['X']
    else:
        print("Extracting CNN features for train set...")
        df_train = pd.read_parquet(train_path)
        X_train_cnn = extract_cnn_features(df_train, effnet, transform, device)
        np.savez(cnn_train_cache, X=X_train_cnn)
        
    if os.path.exists(cnn_test_cache):
        print("Loading CNN test features from cache...")
        X_test_cnn = np.load(cnn_test_cache)['X']
    else:
        print("Extracting CNN features for test set...")
        df_test = pd.read_parquet(test_path)
        # Note: we shouldn't augment test set normally, but for simplicity here keeping it same, or change transform
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        X_test_cnn = extract_cnn_features(df_test, effnet, test_transform, device)
        np.savez(cnn_test_cache, X=X_test_cnn)

    # 3. Fusion
    X_train_fused = np.concatenate((X_train_cls, X_train_cnn), axis=1)
    X_test_fused = np.concatenate((X_test_cls, X_test_cnn), axis=1)
    
    # Balance
    smote = SMOTE(random_state=42, k_neighbors=4)
    X_train_res, y_train_res = smote.fit_resample(X_train_fused, y_train)
    
    # 4. Train Best Classifier (Stacking)
    svm_clf = SVC(probability=True, random_state=42)
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    dt_clf = DecisionTreeClassifier(random_state=42)
    rf_clf = RandomForestClassifier(random_state=42)
    xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    
    estimators_5 = [('svm', svm_clf), ('knn', knn_clf), ('dt', dt_clf), ('rf', rf_clf), ('xgb', xgb_clf)]
    
    stacking_clf = StackingClassifier(
        estimators=estimators_5,
        final_estimator=RandomForestClassifier(n_estimators=100, random_state=42)
    )
    
    print("Training Stacking Classifier on Hybrid Features...")
    stacking_clf.fit(X_train_res, y_train_res)
    
    y_pred = stacking_clf.predict(X_test_fused)
    acc = accuracy_score(y_test, y_pred)
    print(f"Hybrid Model Accuracy: {acc:.4f}")
    
    # Save final model
    joblib.dump(stacking_clf, "results/final_hybrid_model.pkl")
    print("Model saved to results/final_hybrid_model.pkl")

if __name__ == "__main__":
    main()
