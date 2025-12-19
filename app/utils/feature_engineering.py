"""
Feature engineering utilities - matches your training script
"""
import pandas as pd
import numpy as np


def create_features(input_data: dict) -> pd.DataFrame:
    """
    Create all engineered features from raw input
    Must match the feature engineering in your training script
    
    Args:
        input_data: Dictionary with keys matching PredictionInput model
    
    Returns:
        DataFrame with all features
    """
    # Create base DataFrame
    df = pd.DataFrame([{
        'Age': input_data['age'],
        'Monthly_Inhand_Salary': input_data['monthly_inhand_salary'],
        'Num_Credit_Card': input_data['num_credit_card'],
        'Outstanding_Debt': input_data['outstanding_debt'],
        'Credit_Utilization_Ratio': input_data['credit_utilization_ratio'],
        'Monthly_Balance': input_data['monthly_balance'],
        'Credit_Mix': input_data['credit_mix']
    }])
    
    # Financial ratios
    df['Debt_to_Income'] = df['Outstanding_Debt'] / (df['Monthly_Inhand_Salary'] + 1)
    df['Savings_Rate'] = df['Monthly_Balance'] / (df['Monthly_Inhand_Salary'] + 1)
    df['Net_Position'] = df['Monthly_Inhand_Salary'] - df['Outstanding_Debt']
    df['Debt_per_Card'] = df['Outstanding_Debt'] / (df['Num_Credit_Card'] + 1)
    df['Income_per_Card'] = df['Monthly_Inhand_Salary'] / (df['Num_Credit_Card'] + 1)
    df['Available_Credit'] = 100 - df['Credit_Utilization_Ratio']
    df['Credit_Capacity'] = df['Monthly_Inhand_Salary'] * (100 - df['Credit_Utilization_Ratio']) / 100
    
    # Risk indicators
    df['High_Debt'] = (df['Outstanding_Debt'] > df['Outstanding_Debt'].quantile(0.75)).astype(int)
    df['High_Utilization'] = (df['Credit_Utilization_Ratio'] > 50).astype(int)
    df['Critical_Utilization'] = (df['Credit_Utilization_Ratio'] > 80).astype(int)
    df['Low_Balance'] = (df['Monthly_Balance'] < df['Monthly_Balance'].quantile(0.25)).astype(int)
    df['Young_Age'] = (df['Age'] < 30).astype(int)
    df['High_Income'] = (df['Monthly_Inhand_Salary'] > df['Monthly_Inhand_Salary'].quantile(0.75)).astype(int)
    
    # Composite scores
    df['Risk_Score'] = (
        (df['Outstanding_Debt'] / df['Outstanding_Debt'].max()) * 0.4 +
        (df['Credit_Utilization_Ratio'] / 100) * 0.4 +
        (1 - df['Monthly_Balance'] / df['Monthly_Balance'].max()) * 0.2
    )
    
    df['Stability_Score'] = (
        (df['Age'] / df['Age'].max()) * 0.3 +
        (df['Monthly_Inhand_Salary'] / df['Monthly_Inhand_Salary'].max()) * 0.5 +
        (df['Monthly_Balance'] / df['Monthly_Balance'].max()) * 0.2
    )
    
    # Interactions
    df['Age_Income'] = df['Age'] * df['Monthly_Inhand_Salary'] / 1000
    df['Debt_Utilization'] = df['Outstanding_Debt'] * df['Credit_Utilization_Ratio'] / 100
    df['Cards_Income'] = df['Num_Credit_Card'] * df['Monthly_Inhand_Salary'] / 1000
    
    # Non-linear transformations
    df['Log_Income'] = np.log1p(df['Monthly_Inhand_Salary'])
    df['Log_Debt'] = np.log1p(df['Outstanding_Debt'])
    df['Log_Balance'] = np.log1p(df['Monthly_Balance'])
    df['Sqrt_Age'] = np.sqrt(df['Age'])
    
    # Binning
    # Fixed bins (define ONCE, same as training)
    # ------------------ FIXED BIN DEFINITIONS ------------------

    AGE_BINS = [0, 25, 35, 45, 100]
    INCOME_BINS = [0, 25000, 50000, 100000, float("inf")]
    DEBT_BINS = [0, 50000, 100000, 200000, float("inf")]

    BIN_LABELS = [0, 1, 2, 3]

    # ------------------ SAFE BINNING ------------------

    df['Age_Bin'] = pd.cut(
        df['Age'],
        bins=AGE_BINS,
        labels=BIN_LABELS,
        include_lowest=True
    ).astype(int)

    df['Income_Bin'] = pd.cut(
        df['Monthly_Inhand_Salary'],
        bins=INCOME_BINS,
        labels=BIN_LABELS,
        include_lowest=True
    ).astype(int)

    df['Debt_Bin'] = pd.cut(
        df['Outstanding_Debt'],
        bins=DEBT_BINS,
        labels=BIN_LABELS,
        include_lowest=True
    ).astype(int)

    # ------------------ SCHEMA LOCK (CRITICAL) ------------------
    for col in ['Age_Bin', 'Income_Bin', 'Debt_Bin', 'Credit_Mix']:
        if col not in df.columns:
            df[col] = 0

    df['Age_Bin'] = df['Age_Bin'].astype(int)
    df['Income_Bin'] = df['Income_Bin'].astype(int)
    df['Debt_Bin'] = df['Debt_Bin'].astype(int)
    df['Credit_Mix'] = df['Credit_Mix'].astype(str)


    
    return df


def get_categorical_columns() -> list:
    """Get list of categorical column names"""
    return ['Credit_Mix', 'Age_Bin', 'Income_Bin', 'Debt_Bin']


def get_numerical_columns(df: pd.DataFrame, cat_cols: list) -> list:
    """Get list of numerical column names"""
    return [col for col in df.columns if col not in cat_cols]