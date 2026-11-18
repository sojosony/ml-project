
import pandas as pd
import numpy as np
import os

# =====================================================
# 1. LOAD DATASETS
# =====================================================

print("\n========== LOADING DATASETS ==========")

train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

print("Training Dataset:", train.shape)
print("Testing Dataset:", test.shape)


# =====================================================
# 2. DISPLAY BASIC INFORMATION
# =====================================================

print("\n========== TRAINING DATA INFORMATION ==========")

print(train.info())

print("\n========== TEST DATA INFORMATION ==========")

print(test.info())


# =====================================================
# 3. CHECK MISSING VALUES
# =====================================================

print("\n========== MISSING VALUES BEFORE CLEANING ==========")

print("Training Data:")
print(train.isnull().sum())

print("\nTesting Data:")
print(test.isnull().sum())


# =====================================================
# 4. REMOVE DUPLICATE RECORDS
# =====================================================

print("\n========== DUPLICATE REMOVAL ==========")

print("Training duplicates:", train.duplicated().sum())
print("Testing duplicates:", test.duplicated().sum())

train = train.drop_duplicates().copy()
test = test.drop_duplicates().copy()

print("Duplicates removed successfully.")


# =====================================================
# 5. CHECK DUPLICATE PROPERTY IDS
# =====================================================

print("\n========== PROPERTY ID CHECK ==========")

if "Property_ID" in train.columns:

    print(
        "Duplicate IDs in training:",
        train["Property_ID"].duplicated().sum()
    )

if "Property_ID" in test.columns:

    print(
        "Duplicate IDs in testing:",
        test["Property_ID"].duplicated().sum()
    )


# =====================================================
# 6. CLEAN TEXT COLUMNS
# =====================================================

print("\n========== TEXT CLEANING ==========")

for df in [train, test]:

    for column in df.select_dtypes(include=["object"]).columns:

        df[column] = df[column].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )

        df[column] = df[column].replace("", np.nan)

print("Text cleaning completed.")


# =====================================================
# 7. CONVERT NUMERICAL COLUMNS
# =====================================================

print("\n========== DATA TYPE CORRECTION ==========")

numeric_columns = [
    "BHK",
    "Bathrooms",
    "Super_Area_SqFt",
    "Carpet_Area_SqFt",
    "Floor_Number",
    "Total_Floors",
    "Age_of_Property",
    "Distance_to_Metro_km",
    "Distance_to_City_Center_km",
    "Price_INR_Lakhs"
]

for df in [train, test]:

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

print("Numerical conversion completed.")


# =====================================================
# 8. HANDLE INVALID VALUES
# =====================================================

print("\n========== INVALID VALUE CHECKING ==========")

# Values that cannot logically be negative
non_negative_columns = [
    "BHK",
    "Bathrooms",
    "Super_Area_SqFt",
    "Carpet_Area_SqFt",
    "Floor_Number",
    "Total_Floors",
    "Age_of_Property",
    "Distance_to_Metro_km",
    "Distance_to_City_Center_km",
    "Price_INR_Lakhs"
]

for df in [train, test]:

    for column in non_negative_columns:

        if column in df.columns:

            invalid_count = (df[column] < 0).sum()

            print(
                f"{column}: {invalid_count} negative values"
            )

            df.loc[df[column] < 0, column] = np.nan

    # Zero is not a valid house price or property area
    positive_columns = [
        "BHK",
        "Bathrooms",
        "Super_Area_SqFt",
        "Carpet_Area_SqFt",
        "Total_Floors",
        "Price_INR_Lakhs"
    ]

    for column in positive_columns:

        if column in df.columns:

            df.loc[df[column] == 0, column] = np.nan

    # Floor number should not exceed total floors
    if (
        "Floor_Number" in df.columns
        and "Total_Floors" in df.columns
    ):

        invalid_floor = (
            df["Floor_Number"] > df["Total_Floors"]
        )

        df.loc[invalid_floor, "Floor_Number"] = np.nan

print("Invalid value checking completed.")


# =====================================================
# 9. REMOVE ROWS WITH MISSING TARGET PRICE
# =====================================================

print("\n========== TARGET COLUMN CHECK ==========")

if "Price_INR_Lakhs" in train.columns:

    before = len(train)

    train = train.dropna(
        subset=["Price_INR_Lakhs"]
    ).copy()

    print(
        "Rows removed due to missing target price:",
        before - len(train)
    )

else:

    raise ValueError(
        "Price_INR_Lakhs column not found in training data."
    )


# =====================================================
# 10. HANDLE MISSING VALUES
# =====================================================

print("\n========== MISSING VALUE HANDLING ==========")

# Calculate statistics using training dataset only
# This prevents information leakage from test data.

for column in train.columns:

    if column == "Price_INR_Lakhs":
        continue

    if pd.api.types.is_numeric_dtype(train[column]):

        median_value = train[column].median()

        if pd.notna(median_value):

            train[column] = train[column].fillna(
                median_value
            )

            if column in test.columns:

                test[column] = test[column].fillna(
                    median_value
                )

    else:

        mode_values = train[column].mode(dropna=True)

        if len(mode_values) > 0:

            mode_value = mode_values[0]

            train[column] = train[column].fillna(
                mode_value
            )

            if column in test.columns:

                test[column] = test[column].fillna(
                    mode_value
                )

print("Missing value handling completed.")


# =====================================================
# 11. CHECK OUTLIERS USING IQR METHOD
# =====================================================

print("\n========== OUTLIER DETECTION ==========")

outlier_columns = [
    "BHK",
    "Bathrooms",
    "Super_Area_SqFt",
    "Carpet_Area_SqFt",
    "Age_of_Property",
    "Distance_to_Metro_km",
    "Distance_to_City_Center_km",
    "Price_INR_Lakhs"
]

for column in outlier_columns:

    if column in train.columns:

        Q1 = train[column].quantile(0.25)

        Q3 = train[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_limit = Q1 - 1.5 * IQR

        upper_limit = Q3 + 1.5 * IQR

        outliers = train[
            (train[column] < lower_limit) |
            (train[column] > upper_limit)
        ]

        print(
            f"{column}: {len(outliers)} possible outliers"
        )

print("Outlier detection completed.")
print("Outliers have been identified, not deleted.")


# =====================================================
# 12. CHECK MISSING VALUES AFTER CLEANING
# =====================================================

print("\n========== FINAL MISSING VALUE CHECK ==========")

print("\nTraining Dataset:")
print(train.isnull().sum())

print("\nTesting Dataset:")
print(test.isnull().sum())


# =====================================================
# 13. SAVE CLEANED DATASETS
# =====================================================

print("\n========== SAVING CLEANED DATA ==========")

os.makedirs("cleaned_data", exist_ok=True)

train.to_csv(
    "cleaned_data/train_cleaned.csv",
    index=False
)

test.to_csv(
    "cleaned_data/test_cleaned.csv",
    index=False
)

print("Training dataset saved successfully.")

print("Testing dataset saved successfully.")


# =====================================================
# 14. FINAL DATASET SUMMARY
# =====================================================

print("\n========== FINAL DATASET SUMMARY ==========")

print("Original training rows: 80000")

print("Cleaned training shape:", train.shape)

print("Cleaned testing shape:", test.shape)

print("\nTraining dataset remaining missing values:",
      train.isnull().sum().sum())

print("Testing dataset remaining missing values:",
      test.isnull().sum().sum())

print("\nDATASET CLEANING COMPLETED SUCCESSFULLY!")
