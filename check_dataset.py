import pandas as pd

# Read datasets
application = pd.read_csv("dataset/application_record.csv")
credit = pd.read_csv("dataset/credit_record.csv")

print("========== APPLICATION DATASET ==========")
print(application.head())

print("\nApplication Columns:")
print(application.columns.tolist())

print("\nApplication Shape:")
print(application.shape)

print("\n========== CREDIT DATASET ==========")
print(credit.head())

print("\nCredit Columns:")
print(credit.columns.tolist())

print("\nCredit Shape:")
print(credit.shape)