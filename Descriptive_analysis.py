import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load datasets
app = pd.read_csv("application_record.csv")
credit = pd.read_csv("credit_record.csv")

# Target creation
credit["TARGET"] = credit["STATUS"].apply(lambda x: 1 if x in ["1","2","3","4","5"] else 0)
target = credit.groupby("ID")["TARGET"].max().reset_index()

data = app.merge(target, on="ID", how="inner")

# Clean missing values
data.fillna("Unknown", inplace=True)

# Categorical encoding (FIXED - consistent mapping)
cat_cols = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE"
]

for col in cat_cols:
    data[col] = data[col].astype("category").cat.codes

# Features
X = data.drop(["ID", "TARGET"], axis=1)
y = data["TARGET"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model (better than default)
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
print("Accuracy:", model.score(X_test, y_test))

# Save
joblib.dump(model, "credit_card_model.pkl")

print("Model saved successfully")


# Calculate correlation matrix for numerical features from the 'data' DataFrame
# 'data' is the processed DataFrame after merging and encoding from the previous cell.
numerical_data = data.select_dtypes(include=['number'])
correlation_matrix = numerical_data.corr()

# Display correlation values
print("Correlation Matrix:")
print(correlation_matrix)

# Plot heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5,
    linecolor='black'
)

plt.title("Correlation Heatmap of Numerical Features", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()