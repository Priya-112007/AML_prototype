import os
import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import mlflow

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.features.data_loader import get_features_and_labels

def main():
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment("DR_Ensemble_Upgrade")
    
    train_path = "data/idrid/train-00000-of-00001-d81b05cdbfbe95cd.parquet"
    test_path = "data/idrid/test-00000-of-00001-3eaaa0286c6f240e.parquet"
    
    X_train, y_train = get_features_and_labels(train_path, "data/idrid/train_features.npz")
    X_test, y_test = get_features_and_labels(test_path, "data/idrid/test_features.npz")
    
    # Apply SMOTE to balance
    smote = SMOTE(random_state=42, k_neighbors=4)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Base classifiers
    svm_clf = SVC(probability=True, random_state=42)
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    dt_clf = DecisionTreeClassifier(random_state=42)
    rf_clf = RandomForestClassifier(random_state=42)
    xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    
    estimators_3 = [('svm', svm_clf), ('knn', knn_clf), ('dt', dt_clf)]
    estimators_5 = estimators_3 + [('rf', rf_clf), ('xgb', xgb_clf)]
    
    models = {
        "Voting_3_Classifiers_Baseline": VotingClassifier(estimators=estimators_3, voting='soft'),
        "Voting_5_Classifiers": VotingClassifier(estimators=estimators_5, voting='soft'),
        "Stacking_5_Classifiers": StackingClassifier(
            estimators=estimators_5,
            final_estimator=LogisticRegression(max_iter=1000)
        )
    }
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"\nTraining {name}...")
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            print(f"Accuracy: {acc:.4f}")
            mlflow.log_metric("accuracy", acc)
            
            # Save the best model logic could be added here, but for now we just log

if __name__ == "__main__":
    main()
