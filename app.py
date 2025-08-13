import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Pemuatan Model ---
# Model ini HANYA berisi regressor, tanpa preprocessor
MODEL_FILE = 'car_price_prediction_model.pkl'
try:
    with open(MODEL_FILE, 'rb') as f:
        prediction_model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: File model 'car_price_prediction_model.pkl' tidak ditemukan.")
    st.stop()

# ==============================================================================
# KONFIGURASI: LENGKAPI SEMUA INFORMASI DI BAGIAN INI DARI NOTEBOOK COLAB ANDA
# ==============================================================================

# 1. OPSI UNTUK SETIAP FITUR KATEGORIKAL
# (Gunakan df['nama_kolom'].unique().tolist() di notebook Anda untuk mendapatkan ini)
make_options = ['Kia', 'BMW', 'Volvo', 'Audi', 'Nissan', 'Hyundai', 'Chevrolet', 'Ford', 'Acura', 'Cadillac', 'Infiniti', 'Lincoln', 'Jeep', 'Mercedes-Benz', 'GMC', 'Dodge', 'Honda', 'Chrysler', 'Ram', 'Lexus', 'Subaru', 'Mazda', 'Toyota', 'Volkswagen', 'Buick', 'Maserati', 'Land Rover', 'Porsche', 'Jaguar', 'Mitsubishi']
model_options = ['Sorento', '3 Series', 'S60', 'A3', 'Altima', 'Elantra', 'Cruze', 'F-150', 'MDX', 'CTS', 'G37', 'MKZ', 'Grand Cherokee', 'E-Class', 'Acadia', 'Charger', 'Civic', 'Town and Country', '1500', 'IS 250', 'Outback', 'Mazda3', 'Corolla', 'Jetta', 'Enclave', 'Ghibli', 'Range Rover', 'Cayenne', 'XF', 'Outlander Sport']
trim_options = ['LX', 'Base', 'T5', 'Premium', '2.5 S', 'SE', '1LT', 'XLT', 'i', 'Luxury', 'Journey', 'Hybrid', 'Laredo', 'E350', 'SLE', 'SXT', 'EX', 'Touring', 'Big Horn', 'Sport', '2.5i Premium', 's Grand Touring', 'L', 'SportWagen SE', 'Convenience', 'Limited', 'LTZ', 'SLT', 'Express', 'SR5', 'ES 350']
body_options = ['SUV', 'Sedan', 'Wagon', 'Convertible', 'Coupe', 'Hatchback', 'Crew Cab', 'Minivan', 'Van', 'SuperCrew', 'SuperCab', 'Quad Cab', 'King Cab', 'Double Cab', 'Extended Cab', 'Access Cab']
state_options = ['fl', 'ca', 'pa', 'tx', 'ga', 'in', 'nj', 'va', 'il', 'tn', 'az', 'oh', 'mi', 'nc', 'co', 'sc', 'mo', 'md', 'wi', 'nv', 'ma', 'pr', 'mn', 'or', 'wa', 'ny', 'la', 'hi', 'ne', 'ut', 'al', 'ms', 'ct']
transmission_options = ['automatic', 'manual', 'others']
color_options = ['black', 'white', 'gray', 'silver', 'blue', 'red', '—'] # HARAP LENGKAPI
interior_options = ['black', 'gray', 'beige', 'tan', 'brown', '—'] # HARAP LENGKAPI
seller_options = ['nissan infiniti of honolulu', 'the hertz corporation', 'ford motor credit company,llc'] # HARAP LENGKAPI

# 2. PEMETAAN LABEL ENCODER
# Buat pemetaan manual dari setiap kategori ke angka (integer)
# Contoh: {'Kia': 0, 'BMW': 1, ...}
# Anda harus membuat ini untuk setiap fitur yang di-LabelEncode di notebook Anda.
make_map = {label: i for i, label in enumerate(make_options)}
model_map = {label: i for i, label in enumerate(model_options)}
trim_map = {label: i for i, label in enumerate(trim_options)}
body_map = {label: i for i, label in enumerate(body_options)}
state_map = {label: i for i, label in enumerate(state_options)}
color_map = {label: i for i, label in enumerate(color_options)}
interior_map = {label: i for i, label in enumerate(interior_options)}
seller_map = {label: i for i, label in enumerate(seller_options)}

# 3. NILAI MEAN & SCALE DARI STANDARDSCALER
# (Ambil dari scaler.mean_ dan scaler.scale_ di notebook Anda)
numerical_features = ['year', 'condition', 'odometer', 'mmr']
scaler_means = {'year': 2010.0, 'condition': 3.0, 'odometer': 60000.0, 'mmr': 15000.0} # GANTI DENGAN NILAI ASLI
scaler_scales = {'year': 2.5, 'condition': 1.2, 'odometer': 35000.0, 'mmr': 8000.0} # GANTI DENGAN NILAI ASLI

# 4. URUTAN KOLOM FINAL
# Saya telah mengekstrak ini langsung dari file .pkl Anda untuk akurasi.
# Anda tidak perlu mengubah bagian ini.
TRAINING_COLUMN_ORDER = [
    'year', 'condition', 'odometer', 'mmr', 'make', 'model', 'trim', 'body',
    'state', 'color', 'interior', 'seller', 'transmission_automatic',
    'transmission_manual', 'transmission_others'
]

# ==============================================================================
# KODE APLIKASI STREAMLIT (JANGAN UBAH BAGIAN INI)
# ==============================================================================

# --- UI dan Fungsi Prediksi ---
st.title('Prediksi Harga Mobil')
st.write("Aplikasi ini memprediksi harga jual mobil berdasarkan fiturnya.")

def preprocess_input(df):
    """Fungsi untuk melakukan preprocessing manual pada input pengguna."""
    df_processed = df.copy()

    # 1. Terapkan Label Encoding menggunakan map yang sudah dibuat
    df_processed['make'] = df_processed['make'].map(make_map)
    df_processed['model'] = df_processed['model'].map(model_map)
    df_processed['trim'] = df_processed['trim'].map(trim_map)
    df_processed['body'] = df_processed['body'].map(body_map)
    df_processed['state'] = df_processed['state'].map(state_map)
    df_processed['color'] = df_processed['color'].map(color_map)
    df_processed['interior'] = df_processed['interior'].map(interior_map)
    df_processed['seller'] = df_processed['seller'].map(seller_map)

    # 2. Terapkan One-Hot Encoding manual untuk transmisi
    for option in transmission_options:
        col_name = f"transmission_{option}"
        df_processed[col_name] = (df_processed['transmission'] == option).astype(int)
    df_processed = df_processed.drop('transmission', axis=1)

    # 3. Terapkan Standard Scaling manual
    for col in numerical_features:
        df_processed[col] = (df_processed[col] - scaler_means[col]) / scaler_scales[col]

    # 4. Pastikan urutan kolom sama persis dengan saat training
    df_processed = df_processed[TRAINING_COLUMN_ORDER]
    
    return df_processed

st.sidebar.header('Fitur Mobil')
def user_input_features():
    year = st.sidebar.slider('Tahun', 1982, 2015, 2010)
    condition = st.sidebar.slider('Kondisi', 1.0, 50.0, 25.0) # Sesuaikan rentang jika perlu
    odometer = st.sidebar.number_input('Odometer', value=68000.0)
    mmr = st.sidebar.number_input('MMR (Nilai Pasar)', value=13000.0)
    
    make = st.sidebar.selectbox('Merek', make_options)
    model = st.sidebar.selectbox('Model', model_options)
    trim = st.sidebar.selectbox('Trim', trim_options)
    body = st.sidebar.selectbox('Tipe Bodi', body_options)
    transmission = st.sidebar.selectbox('Transmisi', transmission_options)
    state = st.sidebar.selectbox('Negara Bagian', state_options)
    color = st.sidebar.selectbox('Warna', color_options)
    interior = st.sidebar.selectbox('Interior', interior_options)
    seller = st.sidebar.selectbox('Penjual', seller_options)

    data = {
        'year': year, 'condition': condition, 'odometer': odometer, 'mmr': mmr,
        'make': make, 'model': model, 'trim': trim, 'body': body,
        'transmission': transmission, 'state': state, 'color': color,
        'interior': interior, 'seller': seller
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('Fitur Pilihan Anda')
st.write(input_df)

if st.button('Prediksi Harga Jual'):
    try:
        input_df_processed = preprocess_input(input_df)
        prediction = prediction_model.predict(input_df_processed)
        
        st.subheader('Prediksi Harga Jual')
        st.write(f"${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
        st.warning("Pastikan semua daftar opsi dan nilai scaler di bagian KONFIGURASI sudah diisi dengan benar.")