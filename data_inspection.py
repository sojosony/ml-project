
import pandas as pd
import numpy as np

# Load cleaned training dataset
df = pd.read_csv("cleaned_data/train_cleaned.csv")

# =================================================
# 1. CATEGORICAL VALUE INSPECTION
# =================================================

print("\n========== CATEGORICAL VALUE INSPECTION ==========")

categorical_columns = [
    "City",
    "Locality_Type",
    "Property_Type",
    "Furnishing_Status"
]

for column in categorical_columns:

    print(f"\nUnique values in {column}:")

    print(df[column].value_counts(dropna=False))

    print("Total unique values:", df[column].nunique())


# =================================================
# 2. OUTLIER INSPECTION
# =================================================

print("\n========== OUTLIER INSPECTION ==========")

numerical_columns = [
    "Super_Area_SqFt",
    "Carpet_Area_SqFt",
    "Distance_to_Metro_km",
    "Distance_to_City_Center_km",
    "Price_INR_Lakhs"
]

for column in numerical_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print(f"\nColumn: {column}")

    print("Minimum:", df[column].min())
    print("Maximum:", df[column].max())

    print("Lower limit:", lower_limit)
    print("Upper limit:", upper_limit)

    print("Outlier count:", len(outliers))

    print("\nFirst five possible outliers:")

    print(outliers[[column]].head())


print("\n========== INSPECTION COMPLETED ==========")
