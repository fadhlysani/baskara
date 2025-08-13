import streamlit as st
import streamlit.components.v1 as stc
import pickle
import pandas as pd
import numpy as np

# --- Model and Pipeline Loading ---
MODEL_FILE = 'car_price_prediction_model.pkl'

try:
    with open(MODEL_FILE, 'rb') as file:
        # This object is the entire pipeline (preprocessor + model)
        prediction_pipeline = pickle.load(file)
except FileNotFoundError:
    st.error(f"Error: The model file '{MODEL_FILE}' was not found.")
    st.info("Please make sure the trained model file from your Colab notebook is in the same directory as this script and is named correctly.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()


# --- HTML Templates for UI ---
html_temp = """
<div style="background-color:#2E3D49;padding:10px;border-radius:10px">
    <h1 style="color:#fff;text-align:center">Car Price Prediction App</h1> 
    <h4 style="color:#fff;text-align:center">Predicting Vehicle Selling Prices</h4> 
</div>
"""

desc_temp = """
### Car Price Prediction App
This app uses a Random Forest Regressor model to predict the selling price of a used car based on its features.

#### Data Source
- **Kaggle:** [Vehicle Sales Data](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data)

#### App Sections
- **Home:** You are here. Provides an overview of the app.
- **Machine Learning App:** Enter the car's details to get a price prediction.
"""

# --- Prediction Function ---
def predict(year, condition, odometer, mmr, make, model, trim, body, transmission, state):
    """
    Takes user inputs, creates a DataFrame, and uses the loaded pipeline to make a prediction.
    The pipeline handles all preprocessing (scaling, one-hot encoding).
    """
    # Create a dictionary with the user's input
    input_data = {
        'year': [year],
        'condition': [condition],
        'odometer': [odometer],
        'mmr': [mmr],
        'make': [make],
        'model': [model],
        'trim': [trim],
        'body': [body],
        'transmission': [transmission],
        'state': [state]
    }

    # Convert the dictionary to a pandas DataFrame
    # The column names MUST match the ones used during model training.
    input_df = pd.DataFrame(input_data)

    # Use the loaded pipeline to make a prediction
    # The pipeline will automatically apply the same preprocessing steps.
    prediction = prediction_pipeline.predict(input_df)

    # The prediction is returned as an array, so we get the first element.
    return prediction[0]


# --- UI Function for the ML App ---
def run_ml_app():
    """
    Defines the user interface for the prediction part of the app.
    """
    st.subheader("Enter Car Details for Prediction")

    # Structure Form with columns for better layout
    left, right = st.columns(2)
    
    # Input fields for the car features
    year = left.number_input("Year", min_value=1990, max_value=2025, value=2015)
    odometer = left.number_input("Odometer (miles)", value=50000)
    mmr = right.number_input("Market Value (MMR)", value=20000)
    condition = right.number_input("Condition Rating (1-5)", min_value=1.0, max_value=5.0, step=0.1, value=3.5)
    
    # These dropdowns should contain the options your model was trained on.
    # I've taken these from your Colab notebook.
    make = left.selectbox("Brand", ('Kia', 'BMW', 'Volvo', 'Audi', 'Nissan', 'Hyundai', 'Chevrolet', 'Ford', 'Acura', 'Cadillac', 'Infiniti', 'Lincoln', 'Jeep', 'Mercedes-Benz', 'GMC', 'Dodge', 'Honda', 'Chrysler', 'Ram', 'Lexus', 'Subaru', 'Mazda', 'Toyota', 'Volkswagen', 'Buick', 'Maserati', 'Land Rover', 'Porsche', 'Jaguar', 'Mitsubishi'))
    model = right.selectbox("Model", ('Sorento', '3 Series', 'S60', 'A3', 'Altima', 'Elantra', 'Cruze', 'F-150', 'MDX', 'CTS', 'G37', 'MKZ', 'Grand Cherokee', 'E-Class', 'Acadia', 'Charger', 'Civic', 'Town and Country', '1500', 'IS 250', 'Outback', 'Mazda3', 'Corolla', 'Jetta', 'Enclave', 'Ghibli', 'Range Rover', 'Cayenne', 'XF', 'Outlander Sport'))
    trim = left.selectbox("Trim", ('LX', 'Base', 'T5', 'Premium', '2.5 S', 'SE', '1LT', 'XLT', 'i', 'Luxury', 'Journey', 'Hybrid', 'Laredo', 'E350', 'SLE', 'SXT', 'EX', 'Touring', 'Big Horn', 'Sport', '2.5i Premium', 's Grand Touring', 'L', 'SportWagen SE', 'Convenience', 'Limited', 'LTZ', 'SLT', 'Express', 'SR5', 'ES 350'))
    body = right.selectbox("Body Type", ('SUV', 'Sedan', 'Wagon', 'Convertible', 'Coupe', 'Hatchback', 'Crew Cab', 'Minivan', 'Van', 'SuperCrew', 'SuperCab', 'Quad Cab', 'King Cab', 'Double Cab', 'Extended Cab', 'Access Cab'))
    transmission = left.selectbox("Transmission", ('automatic', 'manual'))
    state = right.selectbox("State", ('fl', 'ca', 'pa', 'tx', 'ga', 'in', 'nj', 'va', 'il', 'tn', 'az', 'oh', 'mi', 'nc', 'co', 'sc', 'mo', 'md', 'wi', 'nv', 'ma', 'pr', 'mn', 'or', 'wa', 'ny', 'la', 'hi', 'ne', 'ut', 'al', 'ms', 'ct'))
    
    button = st.button("Predict Price")

    # If button is clicked, run prediction
    if button:
        # Call the prediction function with the user's input
        result = predict(year, condition, odometer, mmr, make, model, trim, body, transmission, state)
        
        # Display the result formatted as currency
        st.success(f"The predicted selling price of the car is **${result:,.2f}**")


# --- Main App Function ---
def main():
    """
    Main function to run the Streamlit app.
    """
    stc.html(html_temp)
    
    menu = ["Home", "Machine Learning App"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Machine Learning App":
        run_ml_app()


if __name__ == "__main__":
    main()