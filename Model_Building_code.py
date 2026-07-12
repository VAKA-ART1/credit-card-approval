import pandas as pd

# -----------------------------
# Load Datasets
# -----------------------------
application = pd.read_csv("/application_record.csv")
credit = pd.read_csv("/credit_record.csv")

# -----------------------------
# Create Target Variable
# -----------------------------

# Bad customer if any status is 2,3,4,5
credit["TARGET"] = credit["STATUS"].apply(
    lambda x: 1 if x in ["2", "3", "4", "5"] else 0
)

# One target per customer
target = credit.groupby("ID")["TARGET"].max().reset_index()

# -----------------------------
# Merge Datasets
# -----------------------------
df = application.merge(target, on="ID", how="inner")

print("Merged Shape :", df.shape)

print("\nTarget Distribution")
print(df["TARGET"].value_counts())

print("\nFirst Five Rows")
print(df.head())

# ==========================================
# TASK 2 : FEATURE ENGINEERING
# ==========================================

import pandas as pd

# Dataset from previous step
# df = merged dataset

# ------------------------------------------
# Check Dataset
# ------------------------------------------
print("Shape :", df.shape)

print("\nColumns")
print(df.columns)

print("\nMissing Values")
print(df.isnull().sum())

# ------------------------------------------
# Remove Duplicate Rows
# ------------------------------------------
print("\nDuplicate Rows :", df.duplicated().sum())

df.drop_duplicates(inplace=True)

# ------------------------------------------
# Convert Age
# ------------------------------------------
df["AGE"] = (-df["DAYS_BIRTH"] / 365).astype(int)

# ------------------------------------------
# Convert Employment Days
# ------------------------------------------
df["YEARS_EMPLOYED"] = (-df["DAYS_EMPLOYED"] / 365).astype(int)

# ------------------------------------------
# Drop Old Columns
# ------------------------------------------
df.drop(["DAYS_BIRTH", "DAYS_EMPLOYED"], axis=1, inplace=True)

# ------------------------------------------
# Remove Constant Columns
# ------------------------------------------
constant_columns = []

for col in df.columns:
    if df[col].nunique() == 1:
        constant_columns.append(col)

print("\nConstant Columns :", constant_columns)

df.drop(columns=constant_columns, inplace=True)

# ------------------------------------------
# Check Outliers (Optional)
# ------------------------------------------
print("\nIncome Summary")
print(df["AMT_INCOME_TOTAL"].describe())

# ------------------------------------------
# Final Information
# ------------------------------------------
print("\nFinal Shape :", df.shape)

print("\nData Types")
print(df.dtypes)

print("\nFirst 5 Rows")
print(df.head())
# ==========================================
# TASK 3 : HANDLING CATEGORICAL VALUES
# ==========================================

import pandas as pd

# ------------------------------------------
# Fill Missing Values
# ------------------------------------------
df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna("Unknown")

# ------------------------------------------
# Binary Encoding
# ------------------------------------------
df["CODE_GENDER"] = df["CODE_GENDER"].map({"M": 1, "F": 0})

df["FLAG_OWN_CAR"] = df["FLAG_OWN_CAR"].map({"Y": 1, "N": 0})

df["FLAG_OWN_REALTY"] = df["FLAG_OWN_REALTY"].map({"Y": 1, "N": 0})

# ------------------------------------------
# One-Hot Encoding
# ------------------------------------------
categorical_columns = [
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE"
]

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)

# ------------------------------------------
# Convert Boolean Columns to Integer
# ------------------------------------------
bool_columns = df.select_dtypes(include="bool").columns

df[bool_columns] = df[bool_columns].astype(int)

# ------------------------------------------
# Check Dataset
# ------------------------------------------
print("="*60)
print("Dataset Shape :", df.shape)

print("\nMissing Values")
print(df.isnull().sum().sum())

print("\nData Types")
print(df.dtypes)

print("\nFirst 5 Rows")
print(df.head())

# ------------------------------------------
# Save Processed Dataset
# ------------------------------------------
df.to_csv("credit_card_processed.csv", index=False)

print("\nDataset Saved Successfully!")

# ==========================================
# TASK 4 : LOGISTIC REGRESSION MODEL
# ==========================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ------------------------------------------
# Load Processed Dataset
# ------------------------------------------

df = pd.read_csv("credit_card_processed.csv")

# Drop ID Column
df.drop("ID", axis=1, inplace=True)

# ------------------------------------------
# Features and Target
# ------------------------------------------

X = df.drop("TARGET", axis=1)

y = df["TARGET"]

# ------------------------------------------
# Train Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ------------------------------------------
# Feature Scaling
# ------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ------------------------------------------
# Logistic Regression Model
# ------------------------------------------

lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr.fit(X_train, y_train)

# ------------------------------------------
# Prediction
# ------------------------------------------

y_pred = lr.predict(X_test)

y_prob = lr.predict_proba(X_test)[:,1]

# ------------------------------------------
# Evaluation
# ------------------------------------------

print("="*60)
print("LOGISTIC REGRESSION RESULTS")
print("="*60)

print("Accuracy :", accuracy_score(y_test,y_pred))

print("Precision :", precision_score(y_test,y_pred))

print("Recall :", recall_score(y_test,y_pred))

print("F1 Score :", f1_score(y_test,y_pred))

print("ROC-AUC :", roc_auc_score(y_test,y_prob))

print("\nConfusion Matrix")
print(confusion_matrix(y_test,y_pred))

print("\nClassification Report")
print(classification_report(y_test,y_pred))

# ==========================================
# TASK 5 : RANDOM FOREST MODEL
# ==========================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ------------------------------------------
# Load Processed Dataset
# ------------------------------------------
df = pd.read_csv("credit_card_processed.csv")

# Drop ID column
df.drop("ID", axis=1, inplace=True)

# ------------------------------------------
# Features and Target
# ------------------------------------------
X = df.drop("TARGET", axis=1)
y = df["TARGET"]

# ------------------------------------------
# Train-Test Split
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ------------------------------------------
# Random Forest Model
# ------------------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# Train Model
rf.fit(X_train, y_train)

# ------------------------------------------
# Prediction
# ------------------------------------------
y_pred = rf.predict(X_test)

y_prob = rf.predict_proba(X_test)[:, 1]

# ------------------------------------------
# Evaluation
# ------------------------------------------
print("=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("=" * 60)
print("TOP 15 IMPORTANT FEATURES")
print("=" * 60)

print(importance.head(15))

# ==========================================
# ROC CURVE COMPARISON
# ==========================================

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score

# Logistic Regression Probability
lr_prob = lr.predict_proba(X_test)[:, 1]

# Random Forest Probability
rf_prob = rf.predict_proba(X_test)[:, 1]

# ROC Values
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)

# AUC Scores
lr_auc = roc_auc_score(y_test, lr_prob)

rf_auc = roc_auc_score(y_test, rf_prob)

# Plot
plt.figure(figsize=(8,6))

plt.plot(lr_fpr, lr_tpr,
         label=f"Logistic Regression (AUC = {lr_auc:.3f})",
         linewidth=2)

plt.plot(rf_fpr, rf_tpr,
         label=f"Random Forest (AUC = {rf_auc:.3f})",
         linewidth=2)

# Random Guess Line
plt.plot([0,1],[0,1],'k--',label="Random Guess")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.grid(True)

plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

lr_prob = lr.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, lr_prob)

auc = roc_auc_score(y_test, lr_prob)

plt.figure(figsize=(7,6))

plt.plot(fpr, tpr, color="blue",
         label=f"AUC = {auc:.3f}")

plt.plot([0,1],[0,1],'r--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Logistic Regression")

plt.legend()

plt.grid(True)

plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

rf_prob = rf.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, rf_prob)

auc = roc_auc_score(y_test, rf_prob)

plt.figure(figsize=(7,6))

plt.plot(fpr, tpr, color="green",
         label=f"AUC = {auc:.3f}")

plt.plot([0,1],[0,1],'r--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Random Forest")

plt.legend()

plt.grid(True)

plt.show()