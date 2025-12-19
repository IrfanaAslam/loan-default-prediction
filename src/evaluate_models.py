# src/evaluate_models.py
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate(model, X_test, y_test, model_name="model"):
    """
    Evaluate a classification model, print metrics, plot ROC & confusion matrix.
    """
    y_pred = model.predict(X_test)

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    except Exception:
        y_prob = None
        roc_auc = None

    # Print metrics
    print(f"\n===== {model_name} Evaluation =====")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1-score :", f1_score(y_test, y_pred))
    if roc_auc:
        print("ROC-AUC  :", roc_auc)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Create outputs folder
    os.makedirs("outputs", exist_ok=True)

    # Confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'outputs/{model_name}_confusion_matrix.png')
    plt.close()

    # ROC Curve
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0,1],[0,1],'--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(f'outputs/{model_name}_ROC_curve.png')
        plt.close()

    return roc_auc


def save_feature_importance(model, feature_names, model_name):
    """
    Save feature importance to CSV and plot a bar chart.
    """
    if not hasattr(model, "feature_importances_"):
        print(f"{model_name} does not support feature importance.")
        return

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    # Save CSV
    os.makedirs("outputs", exist_ok=True)
    path_csv = f"outputs/{model_name}_feature_importance.csv"
    importance_df.to_csv(path_csv, index=False)
    print(f"Feature importance saved to {path_csv}")

    # Plot bar chart
    plt.figure(figsize=(10,6))
    sns.barplot(x="importance", y="feature", data=importance_df.head(20))  # top 20 features
    plt.title(f"{model_name} Top 20 Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    path_plot = f"outputs/{model_name}_feature_importance.png"
    plt.savefig(path_plot)
    plt.close()
    print(f"Feature importance plot saved to {path_plot}")
