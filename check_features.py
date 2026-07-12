import joblib

model = joblib.load("models/credit_card_model.pkl")

print(model.feature_names_in_)