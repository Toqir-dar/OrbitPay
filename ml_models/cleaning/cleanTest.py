import pandas as pd
import numpy as np

# Load raw test dataset
test_path = r"D:\OrbitPay\OrbitPay\orbit-pay\ml_models\test.csv"
test_data = pd.read_csv(test_path, low_memory=False)  # prevents DtypeWarning

# ---------------------------
# Keep only required columns
# ---------------------------
required_cols = ["Age", "Monthly_Inhand_Salary", "Num_Credit_Card",
                 "Outstanding_Debt", "Credit_Utilization_Ratio",
                 "Monthly_Balance", "Credit_Mix"]  # target included
test_data = test_data[required_cols]

# ---------------------------
# Handle missing values & type conversion
# ---------------------------
num_features = ["Age", "Monthly_Inhand_Salary", "Num_Credit_Card",
                "Outstanding_Debt", "Credit_Utilization_Ratio",
                "Monthly_Balance"]

for col in num_features:
    test_data[col] = pd.to_numeric(test_data[col], errors='coerce')
    median_val = test_data[col].median()
    test_data[col] = test_data[col].fillna(median_val)

# Categorical feature
test_data["Credit_Mix"] = test_data["Credit_Mix"].fillna("Standard")

# ---------------------------
# Remove duplicates
# ---------------------------
test_data.drop_duplicates(inplace=True)

# ---------------------------
# Clip outliers for numerical features
# ---------------------------
for col in num_features:
    Q1 = test_data[col].quantile(0.01)
    Q99 = test_data[col].quantile(0.99)
    test_data[col] = test_data[col].clip(lower=Q1, upper=Q99)

# ---------------------------
# Save cleaned test data
# ---------------------------
clean_test_path = "test_cleaned.csv"
test_data.to_csv(clean_test_path, index=False)
print(f"Cleaned test data saved to {clean_test_path}")
