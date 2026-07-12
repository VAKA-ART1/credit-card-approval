import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Loading datasets...")

# ---------------- READ DATA ----------------

application = pd.read_csv("dataset/application_record.csv")
credit = pd.read_csv("dataset/credit_record.csv")

# ---------------- CREATE TARGET ----------------

bad_status = ['1','2','3','4','5']

credit["TARGET"] = credit["STATUS"].apply(
    lambda x: 1 if x in bad_status else 0
)

target_df = credit.groupby("ID")["TARGET"].max().reset_index()

# ---------------- MERGE ----------------

data = application.merge(target_df,on="ID",how="inner")

# ---------------- CLEAN ----------------

data["OCCUPATION_TYPE"] = data["OCCUPATION_TYPE"].fillna("Unknown")

data["CNT_FAM_MEMBERS"] = data["CNT_FAM_MEMBERS"].fillna(
    data["CNT_FAM_MEMBERS"].median()
)

# ---------------- ENCODE ----------------

encoders = {}

categorical_columns = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE"
]

for col in categorical_columns:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])
    encoders[col] = encoder

# ---------------- FEATURES ----------------

X = data.drop(columns=["ID","TARGET"])

y = data["TARGET"]

print("\nTraining Features:")
print(X.columns.tolist())

# ---------------- TRAIN TEST SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- MODEL ----------------

print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ---------------- ACCURACY ----------------

pred = model.predict(X_test)

accuracy = accuracy_score(y_test,pred)

print("\nModel Accuracy:", round(accuracy*100,2), "%")

# ---------------- SAVE ----------------

joblib.dump(model,"models/credit_card_model.pkl")

joblib.dump(encoders,"models/encoders.pkl")

print("\nModel Saved Successfully!")

print("credit_card_model.pkl")

print("encoders.pkl")