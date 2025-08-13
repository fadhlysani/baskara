import streamlit as st
import streamlit.components.v1 as stc
import pickle
import pandas as pd
import numpy as np

# --- Model Loading ---
MODEL_FILE = 'car_price_prediction_model.pkl'

try:
    with open(MODEL_FILE, 'rb') as file:
        # PENTING: Objek ini diasumsikan HANYA model LightGBM, TANPA preprocessor.
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Error: File model '{MODEL_FILE}' tidak ditemukan.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi error saat memuat model: {e}")
    st.stop()

# --- BAGIAN YANG HARUS ANDA ISI ---
# 1. Daftar semua fitur sesuai urutan saat training model
# Ganti dengan urutan kolom yang BENAR-BENAR digunakan saat Anda melatih model.
# Ini sangat krusial!
FINAL_COLUMN_ORDER = [
    'year', 'condition', 'odometer', 'mmr', 'make_Acura', 'make_Audi',
    'make_BMW', 'make_Buick', 'make_Cadillac', 'make_Chevrolet',
    'make_Chrysler', 'make_Dodge', 'make_Ford', 'make_GMC', 'make_Honda',
    'make_Hyundai', 'make_Infiniti', 'make_Jaguar', 'make_Jeep',
    'make_Kia', 'make_Land Rover', 'make_Lexus', 'make_Lincoln',
    'make_Maserati', 'make_Mazda', 'make_Mercedes-Benz',
    'make_Mitsubishi', 'make_Nissan', 'make_Porsche', 'make_Ram',
    'make_Subaru', 'make_Toyota', 'make_Volkswagen', 'make_Volvo',
    'model_1500', 'model_3 Series', 'model_A3', 'model_Acadia',
    'model_Altima', 'model_CTS', 'model_Cayenne', 'model_Charger',

    # ... Lanjutkan daftar ini untuk SEMUA kolom (model, trim, body, dll)
    # ... hingga urutannya sama persis dengan data training Anda.
    # ... Ini akan menjadi daftar yang sangat panjang.
]

# 2. Daftar semua opsi unik untuk setiap fitur kategorikal
# Ambil daftar ini dari notebook training Anda (misal, dengan df['make'].unique())
make_options = ['Kia', 'BMW', 'Volvo', 'Audi', 'Nissan', 'Hyundai', 'Chevrolet', 'Ford', 'Acura', 'Cadillac', 'Infiniti', 'Lincoln', 'Jeep', 'Mercedes-Benz', 'GMC', 'Dodge', 'Honda', 'Chrysler', 'Ram', 'Lexus', 'Subaru', 'Mazda', 'Toyota', 'Volkswagen', 'Buick', 'Maserati', 'Land Rover', 'Porsche', 'Jaguar', 'Mitsubishi']
model_options = ['Sorento', '3 Series', 'S60', 'A3', 'Altima', 'Elantra', 'Cruze', 'F-150', 'MDX', 'CTS', 'G37', 'MKZ', 'Grand Cherokee', 'E-Class', 'Acadia', 'Charger', 'Civic', 'Town and Country', '1500', 'IS 250', 'Outback', 'Mazda3', 'Corolla', 'Jetta', 'Enclave', 'Ghibli', 'Range Rover', 'Cayenne', 'XF', 'Outlander Sport']
trim_options = ['LX', 'Base', 'T5', 'Premium', '2.5 S', 'SE', '1LT', 'XLT', 'i', 'Luxury', 'Journey', 'Hybrid', 'Laredo', 'E350', 'SLE', 'SXT', 'EX', 'Touring', 'Big Horn', 'Sport', '2.5i Premium', 's Grand Touring', 'L', 'SportWagen SE', 'Convenience', 'Limited', 'LTZ', 'SLT', 'Express', 'SR5', 'ES 350']
body_options = ['SUV', 'Sedan', 'Wagon', 'Convertible', 'Coupe', 'Hatchback', 'Crew Cab', 'Minivan', 'Van', 'SuperCrew', 'SuperCab', 'Quad Cab', 'King Cab', 'Double Cab', 'Extended Cab', 'Access Cab']
transmission_options = ['automatic', 'manual']
state_options = ['fl', 'ca', 'pa', 'tx', 'ga', 'in', 'nj', 'va', 'il', 'tn', 'az', 'oh', 'mi', 'nc', 'co', 'sc', 'mo', 'md', 'wi', 'nv', 'ma', 'pr', 'mn', 'or', 'wa', 'ny', 'la', 'hi', 'ne', 'ut', 'al', 'ms', 'ct']


# --- HTML Templates for UI ---
html_temp = """
<div style="background-color:#2E3D49;padding:10px;border-radius:10px">
    <h1 style="color:#fff;text-align:center">Car Price Prediction App</h1>
    <h4 style="color:#fff;text-align:center">Predicting Vehicle Selling Prices</h4>
</div>
"""
desc_temp = "..." # (Sama seperti sebelumnya)

# --- FUNGSI PREDIKSI YANG DIMODIFIKASI ---
def predict(year, condition, odometer, mmr, make, model_input, trim, body, transmission, state):
    # Buat dictionary dari semua input
    input_data = {
        'year': [year], 'condition': [condition], 'odometer': [odometer], 'mmr': [mmr],
        'make': [make], 'model': [model_input], 'trim': [trim], 'body': [body],
        'transmission': [transmission], 'state': [state]
    }
    df = pd.DataFrame(input_data)

    # --- Proses One-Hot Encoding Manual ---
    # Buat kolom untuk setiap kategori dan isi dengan 0 atau 1
    categorical_features = {
        'make': make_options, 'model': model_options, 'trim': trim_options,
        'body': body_options, 'transmission': transmission_options, 'state': state_options
    }

    for feature, options in categorical_features.items():
        for option in options:
            col_name = f"{feature}_{option}"
            df[col_name] = (df[feature] == option).astype(int)

    # Hapus kolom teks asli yang sudah tidak diperlukan
    df = df.drop(columns=list(categorical_features.keys()))

    # --- Pastikan Urutan Kolom Sesuai dengan Training ---
    # Buat DataFrame kosong dengan semua kolom yang mungkin dan urutan yang benar
    processed_df = pd.DataFrame(columns=FINAL_COLUMN_ORDER)
    # Gabungkan dengan data kita, isi kolom yang tidak ada dengan 0
    processed_df = pd.concat([processed_df, df], sort=False).fillna(0)
    # Pilih hanya kolom dengan urutan yang benar
    processed_df = processed_df[FINAL_COLUMN_ORDER]

    # Gunakan model untuk prediksi
    prediction = model.predict(processed_df)
    return prediction[0]

# --- UI Function (Tidak perlu banyak diubah) ---
def run_ml_app():
    st.subheader("Enter Car Details for Prediction")
    left, right = st.columns(2)

    year = left.number_input("Year", min_value=1990, max_value=2025, value=2015)
    odometer = left.number_input("Odometer (miles)", value=50000)
    mmr = right.number_input("Market Value (MMR)", value=20000)
    condition = right.number_input("Condition Rating (1-5)", min_value=1.0, max_value=5.0, step=0.1, value=3.5)

    # Gunakan daftar opsi yang sudah kita definisikan di atas
    make = left.selectbox("Brand", make_options)
    model_input = right.selectbox("Model", model_options) # ganti nama var 'model' agar tidak bentrok
    trim = left.selectbox("Trim", trim_options)
    body = right.selectbox("Body Type", body_options)
    transmission = left.selectbox("Transmission", transmission_options)
    state = right.selectbox("State", state_options)

    if st.button("Predict Price"):
        result = predict(year, condition, odometer, mmr, make, model_input, trim, body, transmission, state)
        st.success(f"The predicted selling price of the car is **${result:,.2f}**")

# --- Main App Function (Tidak diubah) ---
def main():
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