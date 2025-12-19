# main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from src.train_models import train_decision_tree, train_xgboost
from src.evaluate_models import evaluate, save_feature_importance
import re

def sanitize_column_names(df):
    """
    Replace any illegal characters in column names with underscore.
    XGBoost cannot handle [, ], <, >, / in feature names.
    """
    df.columns = [re.sub(r'[^0-9a-zA-Z_]', '_', str(col)) for col in df.columns]
    return df

def main():
    # -----------------------------
    # Step 1: Load Dataset
    # -----------------------------
    data_path = "data/loan_data.csv"  # adjust path if needed
    df = pd.read_csv(data_path)

    # Strip spaces from column names
    df.columns = df.columns.str.strip()

    # -----------------------------
    # Step 2: Preprocessing
    # -----------------------------
    # Drop ID column if exists
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Target column
    target_col = "Status"
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available columns: {df.columns.tolist()}"
        )

    # Fill numeric missing values
    df = df.fillna(df.median(numeric_only=True))

    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Sanitize column names for XGBoost
    df = sanitize_column_names(df)

    # Features & target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # Step 3: Train Models
    # -----------------------------
    print("\nTraining Decision Tree...")
    dt_model = train_decision_tree(X_train, y_train)

    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)

    # -----------------------------
    # Step 4: Evaluate Models
    # -----------------------------
    print("\nEvaluating Decision Tree...")
    evaluate(dt_model, X_test, y_test, model_name="DecisionTree")

    print("\nEvaluating XGBoost...")
    evaluate(xgb_model, X_test, y_test, model_name="XGBoost")

    # -----------------------------
    # Step 5: Save Feature Importance
    # -----------------------------
    save_feature_importance(dt_model, X.columns, "DecisionTree")
    save_feature_importance(xgb_model, X.columns, "XGBoost")

if __name__ == "__main__":
    main()
