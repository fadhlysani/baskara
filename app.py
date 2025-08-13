%%writefile streamlit_app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

st.title('Car Price Prediction')

st.write("""
This app predicts the selling price of a car based on its features.
""")

# Load the trained model
try:
    with open('LGBMRegressor_best_model.pkl', 'rb') as f:
        best_model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: Model file not found. Please make sure 'LGBMRegressor_best_model.pkl' is in the correct directory.")
    st.stop()

# Assuming you have the scalers and encoders fitted on the training data
# You would need to save and load these as well, or refit them here
# For demonstration purposes, let's assume they are available or refitted

# --- Load Scaler and Encoders (replace with your actual saved objects) ---
# Example:
# with open('scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)
# with open('label_encoders.pkl', 'rb') as f:
#     label_encoders = pickle.load(f)
# with open('one_hot_encoder.pkl', 'rb') as f:
#     one_hot_encoder = pickle.load(f)

# For this example, we'll create dummy scalers and encoders.
# In a real scenario, you MUST use the ones fitted on your training data.

# Create dummy data to fit scalers and encoders (replace with actual training data loading)
# This is a placeholder! You should load your actual training data (dataset_feature used for training)
# and fit the scalers and encoders on that data.
# For instance, if you saved your processed training data to a CSV:
# training_data = pd.read_csv('processed_training_data.csv')
# X_train_for_fitting = training_data.drop(columns=['sellingprice'])

# Assuming 'dataset_feature' is available in your Colab environment and represents the processed training data
# In a real deployment, you would load this from a file or database
try:
    # Accessing dataset_feature from the Colab environment
    X_train_for_fitting = dataset_feature.drop(columns=['sellingprice'])
except NameError:
    st.error("Error: 'dataset_feature' not found. Please ensure you have run the data preprocessing steps and 'dataset_feature' is available.")
    st.stop()


# Re-fit scalers and encoders on the available processed training data
scaler = StandardScaler()
cols_to_scale = X_train_for_fitting.columns.drop(['make', 'model', 'trim', 'body', 'state', 'color', 'interior', 'seller', 'transmission_automatic', 'transmission_manual', 'transmission_others'])
scaler.fit(X_train_for_fitting[cols_to_scale])


label_encoders = {}
categorical_cols_le = ['make', 'model', 'trim', 'body', 'state', 'color', 'interior', 'seller']
for col in categorical_cols_le:
    le = LabelEncoder()
    # Combine training and a placeholder for new input to fit the encoder robustly
    # In a production app, you'd handle unseen labels differently
    le.fit(X_train_for_fitting[col].astype(str).tolist() + ['placeholder_for_new_input'])
    label_encoders[col] = le

# Assuming 'transmission' was one-hot encoded and the original column was dropped
# Recreate the one-hot encoder based on the columns created
one_hot_encoder_cols = ['transmission_automatic', 'transmission_manual', 'transmission_others']
# Create dummy data that covers all possible categories seen during training
dummy_ohe_data = pd.DataFrame([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
], columns=one_hot_encoder_cols)
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
one_hot_encoder.fit(dummy_ohe_data)



# --- Create Input Fields ---
st.sidebar.header('Car Features')

def user_input_features():
    year = st.sidebar.slider('Year', 1982, 2015, 2010)
    condition = st.sidebar.slider('Condition', 1.0, 49.0, 30.0)
    odometer = st.sidebar.number_input('Odometer', min_value=0.0, max_value=999999.0, value=68000.0)
    mmr = st.sidebar.number_input('MMR (Market Value)', min_value=0.0, max_value=182000.0, value=13000.0)

    # For categorical features, get unique values from your training data
    # In a real app, you would load these lists from saved files
    try:
        # Accessing dataset_feature from the Colab environment to get unique values
        unique_makes = dataset_feature['make'].unique().tolist()
        unique_models = dataset_feature['model'].unique().tolist()
        unique_trims = dataset_feature['trim'].unique().tolist()
        unique_bodies = dataset_feature['body'].unique().tolist()
        unique_transmissions = ['automatic', 'manual', 'others'] # Based on your OHE
        unique_states = dataset_feature['state'].unique().tolist()
        unique_colors = dataset_feature['color'].unique().tolist()
        unique_interiors = dataset_feature['interior'].unique().tolist()
        unique_sellers = dataset_feature['seller'].unique().tolist()

    except NameError:
        st.error("Error: 'dataset_feature' not found. Cannot populate dropdowns.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while getting unique values: {e}")
        st.stop()


    make = st.sidebar.selectbox('Make', unique_makes)
    model = st.sidebar.selectbox('Model', unique_models)
    trim = st.sidebar.selectbox('Trim', unique_trims)
    body = st.sidebar.selectbox('Body', unique_bodies)
    transmission = st.sidebar.selectbox('Transmission', unique_transmissions)
    state = st.sidebar.selectbox('State', unique_states)
    color = st.sidebar.selectbox('Color', unique_colors)
    interior = st.sidebar.selectbox('Interior', unique_interiors)
    seller = st.sidebar.selectbox('Seller', unique_sellers)


    data = {
        'year': year,
        'condition': condition,
        'odometer': odometer,
        'mmr': mmr,
        'make': make,
        'model': model,
        'trim': trim,
        'body': body,
        'transmission': transmission,
        'state': state,
        'color': color,
        'interior': interior,
        'seller': seller
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Preprocess Input Data ---
def preprocess_input(df):
    df_processed = df.copy()

    # Apply Label Encoding
    categorical_cols_le = ['make', 'model', 'trim', 'body', 'state', 'color', 'interior', 'seller']
    for col in categorical_cols_le:
        if col in df_processed.columns:
            # Handle unseen labels by transforming them to the 'others' encoding if it exists,
            # or a default value (e.g., -1) if 'others' wasn't in the training data
            # A robust approach involves fitting the LabelEncoder on combined training and test data
            # or using handle_unknown='ignore' with OneHotEncoder if applicable after LE
            # Here, we'll use a basic approach assuming 'others' was in training or handle errors
            try:
                 # If 'others' was a category in training, use its encoded value
                 if 'others' in label_encoders[col].classes_:
                    df_processed[col] = label_encoders[col].transform(df_processed[col])
                 else:
                     # Handle cases where 'others' was not a training category
                     # This part needs refinement based on how you handled 'others' in training
                     # For now, a simple approach is to use a default value for unseen labels
                     # A better approach is to fit on training data that includes 'others'
                     df_processed[col] = df_processed[col].apply(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)

            except Exception as e:
                 st.warning(f"Could not apply Label Encoding to {col}: {e}")
                 # Fallback or error handling

    # Apply One-Hot Encoding for transmission
    if 'transmission' in df_processed.columns:
        transmission_encoded = one_hot_encoder.transform(df_processed[['transmission']])
        transmission_df = pd.DataFrame(
            transmission_encoded,
            columns=[f"transmission_{cat}" for cat in one_hot_encoder.categories_[0]],
            index=df_processed.index
        )
        df_processed = pd.concat([df_processed.drop(columns=['transmission']), transmission_df], axis=1)


    # Ensure all expected columns from training are present, add missing OHE columns with 0
    for ohe_col in one_hot_encoder_cols:
        if ohe_col not in df_processed.columns:
            df_processed[ohe_col] = 0

    # Apply Standard Scaling
    cols_to_scale = df_processed.columns.drop(categorical_cols_le + one_hot_encoder_cols)
    df_processed[cols_to_scale] = scaler.transform(df_processed[cols_to_scale])

    # Ensure the order of columns matches the training data
    # This is crucial for consistent predictions
    try:
        # Accessing dataset_feature from the Colab environment to get the training column order
        training_columns = dataset_feature.drop(columns=['sellingprice']).columns.tolist()
        df_processed = df_processed[training_columns]
    except NameError:
        st.error("Error: 'dataset_feature' not found. Cannot ensure correct column order for prediction.")
        st.stop()
    except KeyError as e:
        st.error(f"Error ensuring column order: Missing column in input data - {e}")
        st.stop()


    return df_processed

input_df_processed = preprocess_input(input_df)

# --- Display Input and Prediction ---
st.subheader('User Input Features')
st.write(input_df)

if st.button('Predict Selling Price'):
    try:
        prediction = best_model.predict(input_df_processed)
        st.subheader('Predicted Selling Price')
        st.write(f"${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

