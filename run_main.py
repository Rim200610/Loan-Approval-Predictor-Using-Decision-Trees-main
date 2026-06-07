"""
Loan Approval Predictor Using Decision Trees
Trains on train_Dataset.csv, evaluates with cross-validation,
and predicts on test_dataset.csv.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# 1. Load Training Data
# ============================================================
print("=" * 60)
print("LOAN APPROVAL PREDICTOR - Decision Tree Classifier")
print("=" * 60)

train_df = pd.read_csv("train_Dataset.csv")
test_df = pd.read_csv("test_dataset.csv")

print(f"\nTraining data: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
print(f"Test data:     {test_df.shape[0]} rows, {test_df.shape[1]} columns")

print("\n[TRAIN] First 5 rows:")
print(train_df.head())

print("\n[TEST] First 5 rows:")
print(test_df.head())

print("\n[MISSING - TRAIN] Missing values:")
print(train_df.isnull().sum())

print("\n[MISSING - TEST] Missing values:")
print(test_df.isnull().sum())

# ============================================================
# 2. Data Cleaning (both datasets)
# ============================================================
print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

def clean_data(df, is_train=True):
    """Apply the same cleaning steps to train and test data."""
    # Dependents: fill with mode, convert '3+' to 3
    df.Dependents = df.Dependents.fillna(df.Dependents.mode()[0])
    df.Dependents = df.Dependents.replace('3+', 3).astype(int)

    # Gender: fill using proportional ratio from existing data
    male_pct = df['Gender'].value_counts(normalize=True).get('Male', 0)
    num_missing = df['Gender'].isna().sum()
    num_males = int(male_pct * num_missing)
    num_females = num_missing - num_males
    fill_values = ['Male'] * num_males + ['Female'] * num_females
    np.random.shuffle(fill_values)
    df.loc[df['Gender'].isna(), 'Gender'] = fill_values

    # Married: fill with mode
    df.Married = df.Married.fillna(df.Married.mode()[0])

    # Self_Employed: No info treated as 'No'
    df.Self_Employed = df.Self_Employed.fillna('No')

    # Numerical columns: fill with mean/mode
    df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount.mean())
    df.Loan_Amount_Term = df.Loan_Amount_Term.fillna(df.Loan_Amount_Term.mean())
    df.Credit_History = df.Credit_History.fillna(df.Credit_History.mode()[0])

    return df

train_df = clean_data(train_df, is_train=True)
test_df = clean_data(test_df, is_train=False)

print("\n[OK] Missing values after cleaning (TRAIN):")
print(train_df.isnull().sum())
print("\n[OK] Missing values after cleaning (TEST):")
print(test_df.isnull().sum())

# ============================================================
# 3. Data Preprocessing (both datasets)
# ============================================================
print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

def preprocess_data(df, has_target=True):
    """Encode categorical features."""
    df.Education = df.Education.replace({'Graduate': 1, 'Not Graduate': 0}).astype(int)
    df.Self_Employed = df.Self_Employed.replace({'Yes': 1, 'No': 0}).astype(int)
    df.Gender = df.Gender.replace({'Male': 1, 'Female': 0}).astype(int)
    df.Married = df.Married.replace({'Yes': 1, 'No': 0}).astype(int)
    if has_target:
        df.Loan_Status = df.Loan_Status.replace({'Y': 1, 'N': 0}).astype(int)
    df = pd.get_dummies(df, columns=['Property_Area'], drop_first=True)
    df.Credit_History = df.Credit_History.astype(int)
    return df

train_df = preprocess_data(train_df, has_target=True)
test_df = preprocess_data(test_df, has_target=False)

# Ensure both have the same dummy columns
for col in ['Property_Area_Semiurban', 'Property_Area_Urban']:
    if col not in test_df.columns:
        test_df[col] = False

print("\n[INFO] Processed training data:")
train_df.info()

# ============================================================
# 4. Prepare Features and Target
# ============================================================
print("\n" + "=" * 60)
print("MODEL PREPARATION")
print("=" * 60)

feature_cols = [c for c in train_df.columns if c not in ['Loan_ID', 'Loan_Status']]

X_train_full = train_df[feature_cols]
y_train_full = train_df.Loan_Status

X_test_final = test_df[feature_cols]
test_loan_ids = test_df.Loan_ID

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test_final)

print(f"Training features:  {X_train_scaled.shape}")
print(f"Test features:      {X_test_scaled.shape}")
print(f"Feature columns:    {feature_cols}")

# ============================================================
# 5. Train/Validation Split & Evaluate
# ============================================================
print("\n" + "=" * 60)
print("MODEL EVALUATION (Train/Val Split)")
print("=" * 60)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train_full, test_size=0.3, random_state=70
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_tr, y_tr)

val_accuracy = model.score(X_val, y_val)
print(f"\n>>> Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

print("\n[REPORT] Classification Report (Validation):")
y_val_pred = model.predict(X_val)
print(classification_report(y_val, y_val_pred))

print("[MATRIX] Confusion Matrix (Validation):")
print(confusion_matrix(y_val, y_val_pred))

# Cross-validation on full training set
cv_scores = cross_val_score(model, X_train_scaled, y_train_full, cv=5, scoring='accuracy')
print(f"\n[CV] 5-Fold Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"     Fold scores: {[f'{s:.4f}' for s in cv_scores]}")

# ============================================================
# 6. Train Final Model on ALL Training Data & Predict Test Set
# ============================================================
print("\n" + "=" * 60)
print("FINAL MODEL - Predicting on Test Dataset")
print("=" * 60)

final_model = DecisionTreeClassifier(random_state=42)
final_model.fit(X_train_scaled, y_train_full)

test_predictions = final_model.predict(X_test_scaled)
test_pred_labels = np.where(test_predictions == 1, 'Y', 'N')

# Build results DataFrame
results = pd.DataFrame({
    'Loan_ID': test_loan_ids,
    'Loan_Status': test_pred_labels
})

print(f"\nPredictions summary:")
print(f"  Total applications:  {len(results)}")
print(f"  Approved (Y):        {(test_pred_labels == 'Y').sum()}")
print(f"  Rejected (N):        {(test_pred_labels == 'N').sum()}")
print(f"  Approval rate:       {(test_pred_labels == 'Y').sum() / len(results) * 100:.1f}%")

print("\n[RESULTS] First 20 predictions:")
print(results.head(20).to_string(index=False))

# Save predictions to CSV
output_file = "predictions.csv"
results.to_csv(output_file, index=False)
print(f"\n[SAVED] All {len(results)} predictions saved to: {output_file}")

# ============================================================
# 7. Feature Importance
# ============================================================
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

importances = final_model.feature_importances_
feat_imp = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=False)

for _, row in feat_imp.iterrows():
    bar = '#' * int(row['Importance'] * 50)
    print(f"  {row['Feature']:25s} {row['Importance']:.4f}  {bar}")

print("\n" + "=" * 60)
print("DONE - Training, evaluation, and prediction complete!")
print("=" * 60)
