
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("models/best_model.pkl")

# Load dataset to get dropdown options
df = pd.read_csv("cleaned_data/train_cleaned.csv")

# Input columns used during training
input_columns = [
    "City",
    "Locality_Type",
    "Property_Type",
    "BHK",
    "Bathrooms",
    "Super_Area_SqFt",
    "Carpet_Area_SqFt",
    "Floor_Number",
    "Total_Floors",
    "Age_of_Property",
    "Furnishing_Status",
    "Parking",
    "Lift_Available",
    "Gated_Community",
    "Distance_to_Metro_km",
    "Distance_to_City_Center_km"
]

# Dropdown values from dataset
dropdown_options = {
    "City": sorted(df["City"].dropna().unique().tolist()),
    "Locality_Type": sorted(df["Locality_Type"].dropna().unique().tolist()),
    "Property_Type": sorted(df["Property_Type"].dropna().unique().tolist()),
    "Furnishing_Status": sorted(df["Furnishing_Status"].dropna().unique().tolist())
}

@app.route("/")
def home():
    return render_template(
        "home.html",
        options=dropdown_options
    )


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get values from form
        property_data = {
            "City": request.form["City"],
            "Locality_Type": request.form["Locality_Type"],
            "Property_Type": request.form["Property_Type"],
            "BHK": float(request.form["BHK"]),
            "Bathrooms": float(request.form["Bathrooms"]),
            "Super_Area_SqFt": float(request.form["Super_Area_SqFt"]),
            "Carpet_Area_SqFt": float(request.form["Carpet_Area_SqFt"]),
            "Floor_Number": float(request.form["Floor_Number"]),
            "Total_Floors": float(request.form["Total_Floors"]),
            "Age_of_Property": float(request.form["Age_of_Property"]),
            "Furnishing_Status": request.form["Furnishing_Status"],
            "Parking": int(request.form["Parking"]),
            "Lift_Available": int(request.form["Lift_Available"]),
            "Gated_Community": int(request.form["Gated_Community"]),
            "Distance_to_Metro_km": float(request.form["Distance_to_Metro_km"]),
            "Distance_to_City_Center_km": float(request.form["Distance_to_City_Center_km"])
        }

        # Arrange input in the same column order
        input_df = pd.DataFrame([property_data], columns=input_columns)

        # Predict price
        predicted_price = model.predict(input_df)[0]

        return render_template(
            "result.html",
            price=round(predicted_price, 2)
        )

    except (ValueError, KeyError):
        return "Invalid input. Please check all property details.", 400


    
@app.route("/model-comparison")
def model_comparison():
    return render_template("comparison.html")


if __name__ == "__main__":
    app.run(debug=True)