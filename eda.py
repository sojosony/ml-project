
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("cleaned_data/train_cleaned.csv")

os.makedirs("eda_graphs", exist_ok=True)

target = "Price_INR_Lakhs"

print("\n========== DATASET INFORMATION ==========")
print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# ==========================================
# 1. PRICE SUMMARY
# ==========================================

print("\n========== PRICE SUMMARY ==========")
print(df[target].describe())

# ==========================================
# 2. PRICE DISTRIBUTION
# ==========================================

plt.figure(figsize=(9, 5))
sns.histplot(df[target], bins=50, kde=True)
plt.title("House Price Distribution")
plt.xlabel("Price (INR Lakhs)")
plt.tight_layout()
plt.savefig("eda_graphs/price_distribution.png")
plt.show()

# ==========================================
# 3. OUTLIER ANALYSIS
# ==========================================

plt.figure(figsize=(9, 4))
sns.boxplot(x=df[target])
plt.title("House Price Outlier Analysis")
plt.tight_layout()
plt.savefig("eda_graphs/price_boxplot.png")
plt.show()

# ==========================================
# 4. CITY-WISE PRICE COMPARISON
# ==========================================

categorical_cols = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

if "City" in df.columns:
    city_prices = df.groupby("City")[target].mean().sort_values(
        ascending=False
    )

    print("\n========== CITY-WISE AVERAGE PRICE ==========")
    print(city_prices)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=city_prices.index, y=city_prices.values)
    plt.title("Average House Price by City")
    plt.xlabel("City")
    plt.ylabel("Average Price (INR Lakhs)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("eda_graphs/city_price.png")
    plt.show()

# ==========================================
# 5. NUMERICAL FEATURE CORRELATION
# ==========================================

numeric_df = df.select_dtypes(include="number")

correlation = numeric_df.corr()

print("\n========== PRICE CORRELATION ==========")
print(correlation[target].sort_values(ascending=False))

plt.figure(figsize=(12, 8))
sns.heatmap(correlation, cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_graphs/correlation_heatmap.png")
plt.show()

# ==========================================
# 6. AREA VS HOUSE PRICE
# ==========================================

area_candidates = [
    "Super_Area_SqFt",
    "Carpet_Area_SqFt"
]

area_col = next(
    (col for col in area_candidates if col in df.columns),
    None
)

if area_col:
    plt.figure(figsize=(9, 5))
    sns.scatterplot(
        data=df,
        x=area_col,
        y=target,
        alpha=0.4
    )
    plt.title("Area vs House Price")
    plt.xlabel("Area (Square Feet)")
    plt.ylabel("Price (INR Lakhs)")
    plt.tight_layout()
    plt.savefig("eda_graphs/area_price.png")
    plt.show()
else:
    print("\nArea column not found. Skipping area analysis.")

# ==========================================
# COMPLETION
# ==========================================

print("\n========== EDA COMPLETED ==========")
print("All available EDA graphs saved in eda_graphs folder.")
