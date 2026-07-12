import pandas as pd
import joblib

model = joblib.load("models/credit_card_model.pkl")

application = pd.read_csv("dataset/application_record.csv")
credit = pd.read_csv("dataset/credit_record.csv")

bad_status = ['1','2','3','4','5']
credit["TARGET"] = credit["STATUS"].apply(lambda x: 1 if x in bad_status else 0)
target_df = credit.groupby("ID")["TARGET"].max().reset_index()

data = application.merge(target_df, on="ID")

print(data["TARGET"].value_counts())