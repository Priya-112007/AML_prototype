import mlflow
import os
import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.features.data_loader import get_features_and_labels

def evaluate_and_log(run_name, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=run_name):
        print(f"\n--- Running: {run_name} ---")
        
        # Train classifiers
        svm_clf = SVC(probability=True, random_state=42)
        knn_clf = KNeighborsClassifier(n_neighbors=5)
        dt_clf = DecisionTreeClassifier(random_state=42)
        
        voting_clf = VotingClassifier(
            estimators=[('svm', svm_clf), ('knn', knn_clf), ('dt', dt_clf)],
            voting='soft'
        )
        
        voting_clf.fit(X_train, y_train)
        y_pred = voting_clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        print(f"Accuracy: {acc:.4f}")
        # Specifically check "Mild DR" which is class 1
        mild_dr_recall = report['1']['recall']
        print(f"Mild DR (Class 1) Recall: {mild_dr_recall:.4f}")
        
        mlflow.log_metric("accuracy", acc)
        for cls in ['0', '1', '2', '3', '4']:
            if cls in report:
                mlflow.log_metric(f"recall_class_{cls}", report[cls]['recall'])
                mlflow.log_metric(f"precision_class_{cls}", report[cls]['precision'])
        
        mlflow.log_metric("mild_dr_recall", mild_dr_recall)
        
def main():
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment("DR_Class_Imbalance")
    
    train_path = "data/idrid/train-00000-of-00001-d81b05cdbfbe95cd.parquet"
    test_path = "data/idrid/test-00000-of-00001-3eaaa0286c6f240e.parquet"
    
    X_train, y_train = get_features_and_labels(train_path, "data/idrid/train_features.npz")
    X_test, y_test = get_features_and_labels(test_path, "data/idrid/test_features.npz")
    
    # 1. Baseline (Unbalanced)
    evaluate_and_log("Baseline_Unbalanced", X_train, y_train, X_test, y_test)
    
    # Check class counts to determine safe k_neighbors for SMOTE
    # Minor class is 1 (20 samples)
    
    # 2. SMOTE
    smote = SMOTE(random_state=42, k_neighbors=4)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    evaluate_and_log("SMOTE", X_smote, y_smote, X_test, y_test)
    
    # 3. Borderline-SMOTE
    bsmote = BorderlineSMOTE(random_state=42, k_neighbors=4, m_neighbors=10)
    X_bsmote, y_bsmote = bsmote.fit_resample(X_train, y_train)
    evaluate_and_log("Borderline-SMOTE", X_bsmote, y_bsmote, X_test, y_test)
    
    # 4. SMOTE-ENN
    smote_enn = SMOTEENN(smote=smote, random_state=42)
    X_smenn, y_smenn = smote_enn.fit_resample(X_train, y_train)
    evaluate_and_log("SMOTE-ENN", X_smenn, y_smenn, X_test, y_test)

if __name__ == "__main__":
    main()
