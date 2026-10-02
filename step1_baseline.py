import pandas as pd
import numpy as np
import io
from PIL import Image
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing.image_processing import preprocess_image
from src.features.feature_extraction import extract_lesion_features

def process_dataset(df):
    features = []
    labels = []
    for idx, row in df.iterrows():
        try:
            image_data = row['image']['bytes']
            img = Image.open(io.BytesIO(image_data))
            img_np = np.array(img)
            
            # Preprocess
            _, y_stretched = preprocess_image(img_np)
            
            # Extract features
            feats = extract_lesion_features(y_stretched)
            
            features.append(feats)
            labels.append(row['label'])
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1} images...")
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            continue
            
    return np.array(features), np.array(labels)

def main():
    print("Loading datasets...")
    train_path = "data/idrid/train-00000-of-00001-d81b05cdbfbe95cd.parquet"
    test_path = "data/idrid/test-00000-of-00001-3eaaa0286c6f240e.parquet"
    
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    
    print("Processing training data...")
    X_train, y_train = process_dataset(df_train)
    
    print("Processing testing data...")
    X_test, y_test = process_dataset(df_test)
    
    # Train classifiers
    print("\nTraining classifiers...")
    svm_clf = SVC(probability=True, random_state=42)
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    dt_clf = DecisionTreeClassifier(random_state=42)
    
    # Majority voting
    voting_clf = VotingClassifier(
        estimators=[('svm', svm_clf), ('knn', knn_clf), ('dt', dt_clf)],
        voting='soft' # using soft to get better metrics if needed, but paper used majority (hard)
    )
    
    voting_clf.fit(X_train, y_train)
    
    print("\nEvaluating on test set...")
    y_pred = voting_clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
