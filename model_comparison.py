
import pandas as pd
import matplotlib.pyplot as plt
import os

# Model evaluation results
data = {
    "Algorithm": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "MAE": [47.94, 48.36, 34.04],
    "RMSE": [86.03, 101.79, 70.41],
    "R2 Score": [0.6657, 0.5320, 0.7761]
}

df = pd.DataFrame(data)

# Display comparison table
print("\nMODEL PERFORMANCE COMPARISON")
print(df.to_string(index=False))

# Create output folder
os.makedirs("comparison_graphs", exist_ok=True)

# Generate separate graphs
metrics = ["MAE", "RMSE", "R2 Score"]

for metric in metrics:
    plt.figure(figsize=(8, 5))

    plt.bar(df["Algorithm"], df[metric])

    plt.title(f"Algorithm Comparison - {metric}")
    plt.xlabel("Machine Learning Algorithms")
    plt.ylabel(metric)

    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(f"comparison_graphs/{metric}.png")
    plt.show()

print("\nComparison graphs generated successfully!")
