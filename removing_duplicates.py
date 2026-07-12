# ============================================================
# Project: Credit Card Approval Prediction
# Task: Complete Data Preprocessing
# (Missing Values + Duplicate Removal + Data Cleaning)
# ============================================================

import pandas as pd

# ------------------------------------------------------------
# Load Datasets
# ------------------------------------------------------------
application_df = pd.read_csv("application_record.csv")
credit_df = pd.read_csv("credit_record.csv")

print("="*60)
print("ORIGINAL DATASET SHAPES")
print("="*60)
print("Application Dataset :", application_df.shape)
print("Credit Dataset      :", credit_df.shape)

# ============================================================
# STEP 1 : CHECK MISSING VALUES
# ============================================================

print("\n" + "="*60)
print("MISSING VALUES")
print("="*60)

print("\nApplication Dataset")
print(application_df.isnull().sum())

print("\nCredit Dataset")
print(credit_df.isnull().sum())


# Fill missing values in OCCUPATION_TYPE
if "OCCUPATION_TYPE" in application_df.columns:
    application_df["OCCUPATION_TYPE"] = application_df["OCCUPATION_TYPE"].fillna("Unknown")
    print("\nMissing values in OCCUPATION_TYPE replaced with 'Unknown'.")

# ============================================================
# STEP 2 : REMOVE DUPLICATES
# ============================================================

print("\n" + "="*60)
print("REMOVING DUPLICATES")
print("="*60)

print("Before")
print("Application :", application_df.shape)
print("Credit      :", credit_df.shape)

application_df.drop_duplicates(inplace=True)
credit_df.drop_duplicates(inplace=True)

print("\nAfter")
print("Application :", application_df.shape)
print("Credit      :", credit_df.shape)

# ============================================================
# STEP 3 : DATA CLEANING
# ============================================================

# Convert negative values to positive
application_df["DAYS_BIRTH"] = application_df["DAYS_BIRTH"].abs()
application_df["DAYS_EMPLOYED"] = application_df["DAYS_EMPLOYED"].abs()

# Create Total Family Members Feature
application_df["TOTAL_FAMILY_MEMBERS"] = (
    application_df["CNT_FAM_MEMBERS"] +
    application_df["CNT_CHILDREN"]
)

# Remove unnecessary columns
drop_columns = [
    "FLAG_MOBIL",
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "FLAG_EMAIL"
]

application_df.drop(
    columns=[col for col in drop_columns if col in application_df.columns],
    inplace=True
)

# ============================================================
# STEP 4 : FEATURE TRANSFORMATION
# ============================================================

# Gender
application_df["CODE_GENDER"] = application_df["CODE_GENDER"].map({
    "M":1,
    "F":0
})

# Car Ownership
application_df["FLAG_OWN_CAR"] = application_df["FLAG_OWN_CAR"].map({
    "Y":1,
    "N":0
})

# House Ownership
application_df["FLAG_OWN_REALTY"] = application_df["FLAG_OWN_REALTY"].map({
    "Y":1,
    "N":0
})

# Income Type
application_df["NAME_INCOME_TYPE"] = application_df["NAME_INCOME_TYPE"].map({
    "Working":0,
    "Commercial associate":1,
    "Pensioner":2,
    "State servant":3,
    "Student":4
})

# Education
application_df["NAME_EDUCATION_TYPE"] = application_df["NAME_EDUCATION_TYPE"].map({
    "Lower secondary":0,
    "Secondary / secondary special":1,
    "Incomplete higher":2,
    "Higher education":3,
    "Academic degree":4
})

# Family Status
application_df["NAME_FAMILY_STATUS"] = application_df["NAME_FAMILY_STATUS"].map({
    "Single / not married":0,
    "Married":1,
    "Civil marriage":2,
    "Separated":3,
    "Widow":4
})

# Housing Type
application_df["NAME_HOUSING_TYPE"] = application_df["NAME_HOUSING_TYPE"].map({
    "House / apartment":0,
    "With parents":1,
    "Municipal apartment":2,
    "Rented apartment":3,
    "Office apartment":4,
    "Co-op apartment":5
})

# ============================================================
# STEP 5 : CREDIT DATA TRANSFORMATION
# ============================================================

credit_summary = credit_df.groupby("ID").agg(
    open_month=("MONTHS_BALANCE", "min"),
    end_month=("MONTHS_BALANCE", "max")
).reset_index()

credit_summary["window"] = (
    credit_summary["end_month"] -
    credit_summary["open_month"]
)

status_map = {
    "X":"No Loan",
    "C":"Closed",
    "0":"Paid",
    "1":"30 Days Due",
    "2":"60 Days Due",
    "3":"90 Days Due",
    "4":"120 Days Due",
    "5":"Over 150 Days Due"
}

credit_df["STATUS"] = credit_df["STATUS"].map(status_map)

# Get latest status for each applicant
latest_status = credit_df.groupby("ID")["STATUS"].last().reset_index()

# Merge with credit summary
credit_summary = credit_summary.merge(latest_status, on="ID", how="left")

# ============================================================
# STEP 6 : SAVE CLEANED DATASETS
# ============================================================

application_df.to_csv("application_final.csv", index=False)
credit_summary.to_csv("credit_final.csv", index=False)

print("\n" + "="*60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("="*60)

print("\nFinal Application Dataset :", application_df.shape)
print("Final Credit Dataset      :", credit_summary.shape)

print("\nFiles Saved Successfully")
print("application_final.csv")
print("credit_final.csv")