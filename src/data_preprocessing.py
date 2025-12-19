import pandas as pd

data_path = "data/loan_data.csv"  # adjust if needed
df = pd.read_csv(data_path)

# Strip column names to remove extra spaces
df.columns = df.columns.str.strip()

# Drop ID column if exists
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

# Set target
target_col = "Status"  # your dataset target

# Fill missing numeric values
df = df.fillna(df.median(numeric_only=True))

# Convert categorical variables to dummy/one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Features & target
X = df.drop(columns=[target_col])
y = df[target_col]
