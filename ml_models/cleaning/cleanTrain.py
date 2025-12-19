import pandas as pd
import numpy as np

# ---------------------------
# Load ORIGINAL data
# ---------------------------
print("Loading original data...")
df = pd.read_csv(r"D:\OrbitPay\OrbitPay\orbit-pay\ml_models\train.csv")

print(f"Original shape: {df.shape}")
print(f"\nOriginal Credit_Score distribution:")
print(df['Credit_Score'].value_counts(dropna=False))

# ---------------------------
# Clean the data PROPERLY
# ---------------------------
features = ["Age", "Monthly_Inhand_Salary", "Num_Credit_Card",
            "Outstanding_Debt", "Credit_Utilization_Ratio",
            "Monthly_Balance", "Credit_Mix", "Credit_Score"]

# Keep only necessary columns
df_clean = df[features].copy()

# Remove rows where Credit_Score is missing or empty
df_clean = df_clean[df_clean['Credit_Score'].notna()]
df_clean = df_clean[df_clean['Credit_Score'].str.strip() != '']  # Remove empty strings

# Standardize Credit_Score values (trim whitespace, fix case)
df_clean['Credit_Score'] = df_clean['Credit_Score'].str.strip().str.title()

print(f"\nAfter removing missing targets: {df_clean.shape}")
print(f"\nCredit_Score distribution after cleaning:")
print(df_clean['Credit_Score'].value_counts())

# Remove rows with missing feature values
df_clean = df_clean.dropna()

print(f"\nFinal cleaned shape: {df_clean.shape}")
print(f"\nFinal Credit_Score distribution:")
print(df_clean['Credit_Score'].value_counts())

# Verify the values are correct
unique_scores = df_clean['Credit_Score'].unique()
print(f"\nUnique Credit_Score values: {unique_scores}")

# Save the properly cleaned data
output_path = r"D:\OrbitPay\OrbitPay\orbit-pay\ml_models\train_cleaned_proper.csv"
df_clean.to_csv(output_path, index=False)
print(f"\n✓ Properly cleaned data saved to: {output_path}")