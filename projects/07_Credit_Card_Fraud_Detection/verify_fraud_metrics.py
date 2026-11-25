import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

base_dir = r"projects/07_Credit_Card_Fraud_Detection"
data_path = os.path.join(base_dir, "data", "creditcard.csv")
models_dir = os.path.join(base_dir, "models")
tables_dir = os.path.join(base_dir, "outputs", "tables")

df = pd.read_csv(data_path)
X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Total rows: {len(df)}, Test rows: {len(X_test)}, Frauds in test: {y_test.sum()}")
total_fraud_dollars = X_test.loc[y_test == 1, "Amount"].sum()
total_legit_dollars = X_test.loc[y_test == 0, "Amount"].sum()
print(f"Total fraud dollars in test set: USD {total_fraud_dollars:.2f}")
print(f"Total legitimate dollars in test set: USD {total_legit_dollars:.2f}")

models = {
    "Logistic Regression": os.path.join(models_dir, "fraud_detection_logistic_regression.joblib"),
    "SMOTE + Logistic Regression": os.path.join(models_dir, "fraud_detection_smote_lr.joblib"),
    "Random Forest": os.path.join(models_dir, "fraud_detection_random_forest.joblib")
}

for name, path in models.items():
    if os.path.exists(path):
        pipe = joblib.load(path)
        y_pred = pipe.predict(X_test)
        if hasattr(pipe, "predict_proba"):
            y_prob = pipe.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
        else:
            roc_auc = None
            pr_auc = None
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        acc = (tp + tn) / (tp + tn + fp + fn)
        
        tp_amounts = X_test.loc[(y_test == 1) & (y_pred == 1), "Amount"].sum()
        fp_amounts = X_test.loc[(y_test == 0) & (y_pred == 1), "Amount"].sum()
        fn_amounts = X_test.loc[(y_test == 1) & (y_pred == 0), "Amount"].sum()
        
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.6f}, TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
        print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
        print(f"TP Fraud Amount: USD {tp_amounts:.2f} ({tp_amounts/total_fraud_dollars*100:.2f}%)")
        print(f"FN Missed Fraud Amount: USD {fn_amounts:.2f} ({fn_amounts/total_fraud_dollars*100:.2f}%)")
        print(f"FP Legitimate Amount: USD {fp_amounts:.2f} ({fp_amounts/total_legit_dollars*100:.4f}%)")

print("\n--- Comparing with saved CSV tables ---")
comp_path = os.path.join(tables_dir, "model_performance_comparison.csv")
if os.path.exists(comp_path):
    print("\nSaved model_performance_comparison.csv:")
    print(pd.read_csv(comp_path).to_string())

fin_path = os.path.join(tables_dir, "financial_fraud_prevention_audit.csv")
if os.path.exists(fin_path):
    print("\nSaved financial_fraud_prevention_audit.csv:")
    print(pd.read_csv(fin_path).to_string())
