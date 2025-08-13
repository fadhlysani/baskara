import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Pemuatan Model ---
MODEL_FILE = 'car_price_prediction_model.pkl'
try:
    with open(MODEL_FILE, 'rb') as file:
        prediction_model = pickle.load(file)
except Exception as e:
    st.error(f"Error memuat model: {e}")
    st.stop()

# ==============================================================================
# !! LANGKAH 1: LENGKAPI SEMUA INFORMASI DI BAGIAN INI DARI NOTEBOOK ANDA !!
# ==============================================================================

# A. DAFTAR LENGKAP SEMUA KATEGORI UNIK DARI SETIAP FITUR
# (Gunakan .unique() pada kolom di DataFrame training Anda)
make_options = ['Kia', 'BMW', 'Volvo', 'Audi', 'Nissan', 'Hyundai', 'Chevrolet', 'Ford', 'Acura', 'Cadillac', 'Infiniti', 'Lincoln', 'Jeep', 'Mercedes-Benz', 'GMC', 'Dodge', 'Honda', 'Chrysler', 'Ram', 'Lexus', 'Subaru', 'Mazda', 'Toyota', 'Volkswagen', 'Buick', 'Maserati', 'Land Rover', 'Porsche', 'Jaguar', 'Mitsubishi'] # Sesuaikan/lengkapi
transmission_options = ['automatic', 'manual'] # Tambahkan 'others' jika ada
# ... (Tambahkan list options untuk 'model', 'trim', 'body', 'state', 'color', 'interior', 'seller')

# B. NILAI MEAN & SCALE DARI STANDARDSCALER ANDA
# (Anda bisa dapatkan dari objek scaler yang sudah di-fit: scaler.mean_ dan scaler.scale_)
# Urutan nama kolom di sini HARUS SAMA dengan urutan saat Anda scaling.
numerical_features = ['year', 'condition', 'odometer', 'mmr']
# Contoh nilai, GANTI DENGAN NILAI ASLI DARI NOTEBOOK ANDA
scaler_means = {'year': 2010.0, 'condition': 3.0, 'odometer': 60000.0, 'mmr': 15000.0}
scaler_scales = {'year': 2.5, 'condition': 1.2, 'odometer': 35000.0, 'mmr': 8000.0}

# C. URUTAN KOLOM FINAL SETELAH SEMUA PREPROCESSING
# (Gunakan X_train.columns.tolist() di notebook Anda untuk mendapatkan daftar ini)
# Ini adalah bagian paling penting. Urutan harus 100% akurat.
FINAL_COLUMN_ORDER = [
    # Fitur Numerik yang sudah di-scaling
    'year', 'condition', 'odometer', 'mmr',
    # Fitur Kategorikal yang sudah di-one-hot-encode (CONTOH)
    'make_Acura', 'make_Audi', 'make_BMW', #... dan seterusnya untuk semua merek
    # ... (Lanjutkan untuk semua fitur kategorikal lainnya)
    'transmission_automatic', 'transmission_manual'
]

# ==============================================================================
# LANGKAH 2: KODE APLIKASI STREAMLIT (TIDAK PERLU DIUBAH)
# ==============================================================================

st.set_page_config(page_title="Prediksi Harga Mobil", page_icon="🚗", layout="centered")
st.title("🚗 Aplikasi Prediksi Harga Mobil")
st.write("Aplikasi ini menggunakan model yang sudah dilatih untuk memprediksi harga jual mobil bekas.")

def preprocess_and_predict(data_dict):
    df = pd.DataFrame(data_dict)

    # --- 1. Manual Scaling ---
    st.write("Data sebelum di-scaling:")
    st.dataframe(df[numerical_features])
    for col in numerical_features:
        df[col] = (df[col] - scaler_means[col]) / scaler_scales[col]
    st.write("Data setelah di-scaling:")
    st.dataframe(df[numerical_features])

    # --- 2. Manual One-Hot Encoding ---
    # Definisikan semua fitur kategorikal dan opsinya di sini
    categorical_info = {
        'make': make_options,
        'transmission': transmission_options,
        # ... (Tambahkan fitur lain seperti 'model', 'body', 'color', dll.)
    }
    for feature, options in categorical_info.items():
        for option in options:
            col_name = f"{feature}_{option}"
            df[col_name] = (df[feature] == option).astype(int)
    df = df.drop(columns=list(categorical_info.keys()))

    # --- 3. Pastikan Kolom Sesuai ---
    st.write("Fitur yang dihasilkan setelah encoding:")
    st.dataframe(df.T) # Transpose untuk tampilan lebih baik
    
    # Tambah kolom yang hilang dan pastikan urutan benar
    for col in FINAL_COLUMN_ORDER:
        if col not in df.columns:
            df[col] = 0
    processed_df = df[FINAL_COLUMN_ORDER]

    st.write("Data final yang dikirim ke model (pastikan urutan & jumlah kolom benar):")
    st.dataframe(processed_df)

    # Lakukan prediksi
    prediction = prediction_model.predict(processed_df)
    return prediction[0]

with st.form("input_form"):
    st.header("Masukkan Detail Mobil")
    left, right = st.columns(2)
    year = left.number_input("Tahun", 1990, 2025, 2015)
    odometer = left.number_input("Jarak Tempuh (Odometer)", value=50000)
    mmr = right.number_input("Nilai Pasar (MMR)", value=20000)
    condition = right.number_input("Kondisi (1-5)", 1.0, 5.0, 3.5)

    make = st.selectbox("Merek", make_options)
    transmission = st.selectbox("Transmisi", transmission_options)
    # ... (Tambahkan widget input untuk 'model', 'trim', 'body', 'state', 'color', dll.)

    submitted = st.form_submit_button("Prediksi Harga")

    if submitted:
        input_data = {
            'year': [year], 'condition': [condition], 'odometer': [odometer], 'mmr': [mmr],
            'make': [make], 'transmission': [transmission],
            # ... (Tambahkan input lain ke dictionary ini)
        }
        
        try:
            with st.spinner('Memproses data dan menghitung prediksi...'):
                result = preprocess_and_predict(input_data)
            st.success(f"### Estimasi Harga Jual: **${result:,.2f}**")
        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")
            st.warning("Penyebab paling umum adalah **urutan kolom yang salah** di `FINAL_COLUMN_ORDER`. Harap periksa kembali notebook Anda.")