# ============================================================================
# CREDIT SCORE PREDICTION - VS CODE VERSION WITH HYPERPARAMETER TUNING
# Target: 85-90% Accuracy
# Optimized for local machine with smart tuning strategies
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from pathlib import Path

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import ADASYN

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Update this path to your CSV file location
DATA_PATH = (r"D:\OrbitPay\OrbitPay\orbit-pay\ml_models\train_cleaned_proper.csv")  # Change this to your file path
OUTPUT_DIR = (r"D:\OrbitPay\OrbitPay\orbit-pay\ml_models")  # Directory to save models

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(exist_ok=True)

print("=" * 80)
print("CREDIT SCORE PREDICTION - HYPERPARAMETER TUNING")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[STEP 1/9] Loading data...")

try:
    train_data = pd.read_csv(DATA_PATH)
    print(f"✓ Loaded: {DATA_PATH}")
    print(f"Shape: {train_data.shape}")
    print(f"\nClass distribution:")
    print(train_data['Credit_Score'].value_counts())
except FileNotFoundError:
    print(f"❌ Error: File not found at '{DATA_PATH}'")
    print("Please update the DATA_PATH variable with the correct path to your CSV file.")
    exit()

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
print("\n[STEP 2/9] Feature Engineering...")

features = ["Age", "Monthly_Inhand_Salary", "Num_Credit_Card",
            "Outstanding_Debt", "Credit_Utilization_Ratio",
            "Monthly_Balance", "Credit_Mix"]

X = train_data[features].copy()
y = train_data["Credit_Score"].copy()

# Financial ratios
X['Debt_to_Income'] = X['Outstanding_Debt'] / (X['Monthly_Inhand_Salary'] + 1)
X['Savings_Rate'] = X['Monthly_Balance'] / (X['Monthly_Inhand_Salary'] + 1)
X['Net_Position'] = X['Monthly_Inhand_Salary'] - X['Outstanding_Debt']
X['Debt_per_Card'] = X['Outstanding_Debt'] / (X['Num_Credit_Card'] + 1)
X['Income_per_Card'] = X['Monthly_Inhand_Salary'] / (X['Num_Credit_Card'] + 1)
X['Available_Credit'] = 100 - X['Credit_Utilization_Ratio']
X['Credit_Capacity'] = X['Monthly_Inhand_Salary'] * (100 - X['Credit_Utilization_Ratio']) / 100

# Risk indicators
X['High_Debt'] = (X['Outstanding_Debt'] > X['Outstanding_Debt'].quantile(0.75)).astype(int)
X['High_Utilization'] = (X['Credit_Utilization_Ratio'] > 50).astype(int)
X['Critical_Utilization'] = (X['Credit_Utilization_Ratio'] > 80).astype(int)
X['Low_Balance'] = (X['Monthly_Balance'] < X['Monthly_Balance'].quantile(0.25)).astype(int)
X['Young_Age'] = (X['Age'] < 30).astype(int)
X['High_Income'] = (X['Monthly_Inhand_Salary'] > X['Monthly_Inhand_Salary'].quantile(0.75)).astype(int)

# Composite scores
X['Risk_Score'] = (
    (X['Outstanding_Debt'] / X['Outstanding_Debt'].max()) * 0.4 +
    (X['Credit_Utilization_Ratio'] / 100) * 0.4 +
    (1 - X['Monthly_Balance'] / X['Monthly_Balance'].max()) * 0.2
)

X['Stability_Score'] = (
    (X['Age'] / X['Age'].max()) * 0.3 +
    (X['Monthly_Inhand_Salary'] / X['Monthly_Inhand_Salary'].max()) * 0.5 +
    (X['Monthly_Balance'] / X['Monthly_Balance'].max()) * 0.2
)

# Interactions
X['Age_Income'] = X['Age'] * X['Monthly_Inhand_Salary'] / 1000
X['Debt_Utilization'] = X['Outstanding_Debt'] * X['Credit_Utilization_Ratio'] / 100
X['Cards_Income'] = X['Num_Credit_Card'] * X['Monthly_Inhand_Salary'] / 1000

# Non-linear transformations
X['Log_Income'] = np.log1p(X['Monthly_Inhand_Salary'])
X['Log_Debt'] = np.log1p(X['Outstanding_Debt'])
X['Log_Balance'] = np.log1p(X['Monthly_Balance'])
X['Sqrt_Age'] = np.sqrt(X['Age'])

# Binning
X['Age_Bin'] = pd.cut(X['Age'], bins=[0, 25, 35, 45, 100], labels=[0, 1, 2, 3])
X['Income_Bin'] = pd.qcut(X['Monthly_Inhand_Salary'], q=4, labels=[0, 1, 2, 3], duplicates='drop')
X['Debt_Bin'] = pd.qcut(X['Outstanding_Debt'], q=4, labels=[0, 1, 2, 3], duplicates='drop')

print(f"Features created: {len(features)} → {X.shape[1]}")

# ============================================================================
# STEP 3: PREPROCESSING
# ============================================================================
print("\n[STEP 3/9] Preprocessing...")

cat_cols = ['Credit_Mix', 'Age_Bin', 'Income_Bin', 'Debt_Bin']
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(X[cat_cols])

num_cols = [col for col in X.columns if col not in cat_cols]
X_combined = np.concatenate([X[num_cols].values, encoded], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

target_mapping = {"Poor": 0, "Standard": 1, "Good": 2}
y_num = y.map(target_mapping)

print(f"Total features: {X_scaled.shape[1]}")

# ============================================================================
# STEP 4: TRAIN/TEST SPLIT & BALANCING
# ============================================================================
print("\n[STEP 4/9] Splitting and balancing data...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_num, test_size=0.2, random_state=42, stratify=y_num
)

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
print(f"Before ADASYN: {np.bincount(y_train)}")

adasyn = ADASYN(sampling_strategy='auto', random_state=42, n_neighbors=5)
X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train, y_train)
print(f"After ADASYN: {np.bincount(y_train_balanced)}")

# ============================================================================
# STEP 5: HYPERPARAMETER TUNING - XGBoost
# ============================================================================
print("\n[STEP 5/9] Hyperparameter Tuning - XGBoost...")
print("This will take approximately 10-15 minutes...")

xgb_param_dist = {
    'n_estimators': [300, 400, 500],
    'max_depth': [6, 7, 8, 9],
    'learning_rate': [0.03, 0.05, 0.07],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.5, 1.0, 2.0]
}

xgb_random = RandomizedSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',
                  random_state=42, n_jobs=-1, tree_method='hist'),
    xgb_param_dist,
    n_iter=30,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

xgb_random.fit(X_train_balanced, y_train_balanced)

best_xgb = xgb_random.best_estimator_
print(f"\n✓ Best XGBoost params: {xgb_random.best_params_}")
print(f"✓ Best CV score: {xgb_random.best_score_:.4f}")

# ============================================================================
# STEP 6: HYPERPARAMETER TUNING - LightGBM
# ============================================================================
print("\n[STEP 6/9] Hyperparameter Tuning - LightGBM...")
print("This will take approximately 8-12 minutes...")

lgbm_param_dist = {
    'n_estimators': [300, 400, 500],
    'max_depth': [6, 8, 10, 12],
    'learning_rate': [0.03, 0.05, 0.07],
    'num_leaves': [31, 40, 50],
    'min_child_samples': [10, 20, 30],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

lgbm_random = RandomizedSearchCV(
    LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1),
    lgbm_param_dist,
    n_iter=30,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

lgbm_random.fit(X_train_balanced, y_train_balanced)

best_lgbm = lgbm_random.best_estimator_
print(f"\n✓ Best LightGBM params: {lgbm_random.best_params_}")
print(f"✓ Best CV score: {lgbm_random.best_score_:.4f}")

# ============================================================================
# STEP 7: TRAIN ADDITIONAL MODELS
# ============================================================================
print("\n[STEP 7/9] Training additional models...")


# Random Forest
print("  [2/3] Random Forest...")
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_balanced, y_train_balanced)

# Gradient Boosting
print("  [3/3] Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=7,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train_balanced, y_train_balanced)

# ============================================================================
# STEP 8: ADVANCED STACKED ENSEMBLE
# ============================================================================
print("\n[STEP 8/9] Creating advanced stacked ensemble...")

base_models = [
    ('xgb_tuned', best_xgb),
    ('lgbm_tuned', best_lgbm),
    ('rf', rf),
    ('gb', gb)
]

meta_learner = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=20,
    random_state=42,
    verbose=-1
)

stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1
)

print("Training stacked ensemble (this may take 5-10 min)...")
stacking.fit(X_train_balanced, y_train_balanced)

# ============================================================================
# STEP 9: COMPREHENSIVE EVALUATION
# ============================================================================
print("\n[STEP 9/9] Comprehensive Evaluation...")

models = {
    'XGBoost (Tuned)': best_xgb,
    'LightGBM (Tuned)': best_lgbm,
    'Random Forest': rf,
    'Gradient Boosting': gb,
    'Stacked Ensemble': stacking
}

results = []
target_names = ["Poor", "Standard", "Good"]

for name, model in models.items():
    print(f"\n{'='*70}")
    print(f"{name.upper()}")
    print(f"{'='*70}")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({'Model': name, 'Accuracy': acc})

    print(f"✓ Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'{name}\nAccuracy: {acc*100:.2f}%', fontsize=12, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/confusion_matrix_{name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# FINAL RESULTS & COMPARISON
# ============================================================================
print("\n" + "="*80)
print("FINAL RESULTS WITH HYPERPARAMETER TUNING")
print("="*80)

results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
print("\n" + results_df.to_string(index=False))

best_name = results_df.iloc[0]['Model']
best_acc = results_df.iloc[0]['Accuracy']

print(f"\n🏆 BEST MODEL: {best_name}")
print(f"   Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")

# Improvement from baseline
baseline = 0.7176
improvement = (best_acc - baseline) * 100
print(f"\n📈 IMPROVEMENT FROM BASELINE:")
print(f"   Original (XGBoost): 71.76%")
print(f"   Current Best: {best_acc*100:.2f}%")
print(f"   Gain: +{improvement:.2f} percentage points")

# Visual comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart
colors = ['#27ae60' if a == best_acc else '#3498db' for a in results_df['Accuracy']]
bars = ax1.barh(results_df['Model'], results_df['Accuracy'], color=colors)
ax1.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_xlim(0.70, min(0.92, results_df['Accuracy'].max() + 0.03))
ax1.grid(True, alpha=0.3, axis='x')

for bar in bars:
    width = bar.get_width()
    ax1.text(width + 0.002, bar.get_y() + bar.get_height()/2,
             f'{width:.4f}\n({width*100:.2f}%)',
             ha='left', va='center', fontsize=9, fontweight='bold')

ax1.axvline(x=0.85, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='85% Target')
ax1.axvline(x=0.90, color='red', linestyle='--', linewidth=2, alpha=0.6, label='90% Target')
ax1.legend()

# Improvement chart
models_list = ['Baseline\n(Original)', best_name]
accs_list = [baseline, best_acc]
colors_list = ['#e74c3c', '#27ae60']

bars2 = ax2.bar(models_list, accs_list, color=colors_list, alpha=0.8, edgecolor='black', linewidth=2)
ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax2.set_title('Accuracy Improvement', fontsize=14, fontweight='bold')
ax2.set_ylim(0.65, min(0.95, best_acc + 0.05))
ax2.grid(True, alpha=0.3, axis='y')

for bar, acc in zip(bars2, accs_list):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height,
             f'{acc:.4f}\n({acc*100:.2f}%)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add improvement arrow
ax2.annotate('', xy=(1, best_acc), xytext=(0, baseline),
            arrowprops=dict(arrowstyle='->', lw=3, color='green'))
ax2.text(0.5, (baseline + best_acc)/2, f'+{improvement:.2f}pp',
         ha='center', fontsize=12, fontweight='bold', color='green',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', linewidth=2))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Achievement status
print("\n" + "="*80)
print("ACCURACY TARGET STATUS")
print("="*80)
if best_acc >= 0.90:
    print("🎉 OUTSTANDING! Achieved 90%+ accuracy target!")
    print("   Your model is production-ready and highly accurate!")
elif best_acc >= 0.85:
    print("✓ SUCCESS! Achieved 85-90% accuracy target!")
    print("   Excellent performance for credit scoring!")
elif best_acc >= 0.80:
    print("⚠ GOOD PROGRESS! Reached 80%+, close to target.")
    print("   Consider: more features, more data, or ensemble refinement")
else:
    print("⚠ NEEDS IMPROVEMENT. Current accuracy below 80%.")
    print("   Suggestions: check data quality, add more features, collect more samples")

# Feature importance analysis
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

if hasattr(models[best_name], 'feature_importances_'):
    feature_names = num_cols + list(encoder.get_feature_names_out(cat_cols))

    if 'Stacked' in best_name:
        importances = models[best_name].estimators_[0].feature_importances_
    else:
        importances = models[best_name].feature_importances_

    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    print("\nTop 15 Most Important Features:")
    print(imp_df.head(15).to_string(index=False))

    # Plot
    plt.figure(figsize=(10, 8))
    top_features = imp_df.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance Score', fontweight='bold')
    plt.title(f'Top 15 Features - {best_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# SAVE MODELS & PREPROCESSORS
# ============================================================================
print("\n" + "="*80)
print("SAVING MODELS & PREPROCESSORS")
print("="*80)

# Save all models
print("\nSaving models...")
for model_name, model_obj in models.items():
    filename = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    joblib.dump(model_obj, f'{OUTPUT_DIR}/{filename}.pkl')
    print(f"  ✓ Saved: {OUTPUT_DIR}/{filename}.pkl")

# Save preprocessors
joblib.dump(scaler, f'{OUTPUT_DIR}/scaler.pkl')
joblib.dump(encoder, f'{OUTPUT_DIR}/encoder.pkl')
joblib.dump(target_mapping, f'{OUTPUT_DIR}/target_mapping.pkl')
print(f"  ✓ Saved: {OUTPUT_DIR}/scaler.pkl")
print(f"  ✓ Saved: {OUTPUT_DIR}/encoder.pkl")
print(f"  ✓ Saved: {OUTPUT_DIR}/target_mapping.pkl")

# Save hyperparameter tuning results
tuning_results = {
    'xgb_best_params': xgb_random.best_params_,
    'xgb_best_score': xgb_random.best_score_,
    'lgbm_best_params': lgbm_random.best_params_,
    'lgbm_best_score': lgbm_random.best_score_
}
joblib.dump(tuning_results, f'{OUTPUT_DIR}/hyperparameter_tuning_results.pkl')
print(f"  ✓ Saved: {OUTPUT_DIR}/hyperparameter_tuning_results.pkl")

# Save results summary
results_df.to_csv(f'{OUTPUT_DIR}/model_results.csv', index=False)
print(f"  ✓ Saved: {OUTPUT_DIR}/model_results.csv")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nBest Model: {best_name}")
print(f"Final Accuracy: {best_acc*100:.2f}%")
print(f"All models and results saved to '{OUTPUT_DIR}/' directory")
print("\nYou can now use these models in your OrbitPay application!")
print("\nTo load the best model later:")
print(f"  model = joblib.load('{OUTPUT_DIR}/{best_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pkl')")
print(f"  scaler = joblib.load('{OUTPUT_DIR}/scaler.pkl')")
print(f"  encoder = joblib.load('{OUTPUT_DIR}/encoder.pkl')")