import streamlit as st
import streamlit.components.v1 as stc
import pickle
import pandas as pd
import numpy as np

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Harga Mobil",
    page_icon="🚗",
    layout="wide"
)

# --- Pemuatan Model ---
MODEL_FILE = 'car_price_prediction_model.pkl'

try:
    with open(MODEL_FILE, 'rb') as file:
        # Objek ini diasumsikan HANYA model, BUKAN pipeline lengkap.
        prediction_model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Error: File model '{MODEL_FILE}' tidak ditemukan.")
    st.info("Pastikan file .pkl dari notebook Anda berada di folder yang sama dengan skrip ini.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi error saat memuat model: {e}")
    st.stop()


# ---
# !! PENTING: ANDA HARUS MENGISI BAGIAN INI !!
# Buka notebook pelatihan Anda dan dapatkan daftar nilai unik untuk setiap kolom.
# Urutan dan nama harus SAMA PERSIS dengan data training Anda.
# ---

# 1. Daftar semua opsi unik untuk setiap fitur kategorikal
make_options = ['Kia', 'BMW', 'Volvo', 'Audi', 'Nissan', 'Hyundai', 'Chevrolet', 'Ford', 'Acura', 'Cadillac', 'Infiniti', 'Lincoln', 'Jeep', 'Mercedes-Benz', 'GMC', 'Dodge', 'Honda', 'Chrysler', 'Ram', 'Lexus', 'Subaru', 'Mazda', 'Toyota', 'Volkswagen', 'Buick', 'Maserati', 'Land Rover', 'Porsche', 'Jaguar', 'Mitsubishi'] # Contoh, lengkapi/sesuaikan
model_options = ['Sorento', '3 Series', 'S60', 'A3', 'Altima', 'Elantra', 'Cruze', 'F-150', 'MDX', 'CTS', 'G37', 'MKZ', 'Grand Cherokee', 'E-Class', 'Acadia', 'Charger', 'Civic', 'Town and Country', '1500', 'IS 250', 'Outback', 'Mazda3', 'Corolla', 'Jetta', 'Enclave', 'Ghibli', 'Range Rover', 'Cayenne', 'XF', 'Outlander Sport'] # Contoh, lengkapi/sesuaikan
trim_options = ['LX', 'Base', 'T5', 'Premium', '2.5 S', 'SE', '1LT', 'XLT', 'i', 'Luxury', 'Journey', 'Hybrid', 'Laredo', 'E350', 'SLE', 'SXT', 'EX', 'Touring', 'Big Horn', 'Sport', '2.5i Premium', 's Grand Touring', 'L', 'SportWagen SE', 'Convenience', 'Limited', 'LTZ', 'SLT', 'Express', 'SR5', 'ES 350'] # Contoh, lengkapi/sesuaikan
body_options = ['SUV', 'Sedan', 'Wagon', 'Convertible', 'Coupe', 'Hatchback', 'Crew Cab', 'Minivan', 'Van', 'SuperCrew', 'SuperCab', 'Quad Cab', 'King Cab', 'Double Cab', 'Extended Cab', 'Access Cab'] # Contoh, lengkapi/sesuaikan
state_options = ['fl', 'ca', 'pa', 'tx', 'ga', 'in', 'nj', 'va', 'il', 'tn', 'az', 'oh', 'mi', 'nc', 'co', 'sc', 'mo', 'md', 'wi', 'nv', 'ma', 'pr', 'mn', 'or', 'wa', 'ny', 'la', 'hi', 'ne', 'ut', 'al', 'ms', 'ct'] # Contoh, lengkapi/sesuaikan
transmission_options = ['automatic', 'manual', 'others'] # 'others' mungkin ada, cek data Anda

# !! Fitur baru dari analisis model, harap isi dengan nilai dari data Anda !!
color_options = ['black', 'white', 'gray', 'silver', 'blue', 'red', '—'] # GANTI DENGAN OPSI DARI DATA ANDA
interior_options = ['black', 'gray', 'beige', 'tan', 'brown', '—'] # GANTI DENGAN OPSI DARI DATA ANDA
seller_options = ['nissan infiniti of honolulu', 'the hertz corporation', 'ford motor credit company,llc'] # GANTI DENGAN OPSI DARI DATA ANDA

# 2. Daftar URUTAN KOLOM FINAL setelah preprocessing (One-Hot Encoding)
# Ini adalah bagian paling krusial. Urutannya harus sama persis dengan saat model dilatih.
# Anda bisa mendapatkan ini di notebook Anda dengan `X_train.columns.tolist()` setelah preprocessing
FINAL_COLUMN_ORDER = [
    # Fitur Numerik (sesuaikan urutan jika perlu)
    'year', 'condition', 'odometer', 'mmr',
    # One-Hot Encoded Features (INI HANYA CONTOH, HARUS DISESUAIKAN)
    # Anda perlu membuat daftar LENGKAP seperti ini dari notebook Anda
    'make_Acura', 'make_Audi', 'make_BMW', #... dan seterusnya untuk semua merek
    'model_1500', 'model_3 Series', 'model_A3', #... dan seterusnya untuk semua model
    'trim_Base', 'trim_Big Horn', # ... dan seterusnya
    'body_Coupe', 'body_Crew Cab', # ... dan seterusnya
    'state_al', 'state_az', 'state_ca', # ... dan seterusnya
    'color_black', 'color_blue', # ... dan seterusnya
    'interior_beige', 'interior_black', # ... dan seterusnya
    'seller_ford motor credit company,llc', # ... dan seterusnya (perhatikan spasi dan huruf kecil!)
    'transmission_automatic', 'transmission_manual', 'transmission_others'
]

# --- Template UI ---
html_temp = """...""" # (Sama seperti sebelumnya)
desc_temp = """...""" # (Sama seperti sebelumnya)

# --- FUNGSI PREDIKSI (Dengan Preprocessing Manual) ---
def preprocess_and_predict(data_dict):
    """
    Mengambil input, melakukan preprocessing manual, dan memprediksi harga.
    """
    df = pd.DataFrame(data_dict)

    # --- Proses One-Hot Encoding Manual ---
    # Buat dictionary yang berisi semua fitur kategorikal dan opsinya
    categorical_features = {
        'make': make_options, 'model': model_options, 'trim': trim_options,
        'body': body_options, 'state': state_options, 'color': color_options,
        'interior': interior_options, 'seller': seller_options,
        'transmission': transmission_options
    }

    # Loop untuk membuat kolom dummy (0 atau 1)
    for feature, options in categorical_features.items():
        for option in options:
            col_name = f"{feature}_{option}"
            df[col_name] = (df[feature] == option).astype(int)

    # Hapus kolom teks asli
    df = df.drop(columns=list(categorical_features.keys()))

    # --- Pastikan Urutan & Kelengkapan Kolom Sesuai dengan Training ---
    # Tambahkan kolom yang mungkin hilang (jika input tidak mencakup semua kemungkinan)
    for col in FINAL_COLUMN_ORDER:
        if col not in df.columns:
            df[col] = 0

    # Pastikan urutan kolom sama persis
    processed_df = df[FINAL_COLUMN_ORDER]

    # Lakukan prediksi
    prediction = prediction_model.predict(processed_df)
    return prediction[0]

# --- UI Aplikasi ---
def run_ml_app():
    st.subheader("Masukkan Detail Mobil untuk Prediksi")

    # Layout dengan kolom
    left, right = st.columns(2)

    # Input fitur numerik
    year = left.number_input("Tahun", 1990, 2025, 2015)
    odometer = left.number_input("Jarak Tempuh (Odometer)", value=50000, step=1000)
    mmr = right.number_input("Nilai Pasar (MMR)", value=20000, step=500)
    condition = right.number_input("Kondisi (1-5)", 1.0, 5.0, 3.5, 0.1)

    # Input fitur kategorikal
    make = left.selectbox("Merek", make_options)
    model = right.selectbox("Model", model_options)
    trim = left.selectbox("Trim", trim_options)
    body = right.selectbox("Tipe Bodi", body_options)
    transmission = left.selectbox("Transmisi", transmission_options)
    state = right.selectbox("Negara Bagian (State)", state_options)
    color = left.selectbox("Warna (Color)", color_options)
    interior = right.selectbox("Interior", interior_options)
    seller = st.selectbox("Penjual (Seller)", seller_options) # Dibuat full-width

    if st.button("Prediksi Harga", type="primary"):
        with st.spinner('Menghitung prediksi...'):
            input_data = {
                'year': [year], 'condition': [condition], 'odometer': [odometer], 'mmr': [mmr],
                'make': [make], 'model': [model], 'trim': [trim], 'body': [body],
                'transmission': [transmission], 'state': [state], 'color': [color],
                'interior': [interior], 'seller': [seller]
            }
            # Panggil fungsi prediksi
            result = preprocess_and_predict(input_data)
            st.success(f"Prediksi harga jual mobil adalah **${result:,.2f}**")
            st.info("Catatan: Prediksi ini didasarkan pada model yang dilatih dan mungkin tidak mencerminkan harga pasar sesungguhnya.")

# --- Fungsi Utama ---
def main():
    stc.html(html_temp)
    menu = ["Beranda", "Aplikasi Prediksi"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Beranda":
        st.subheader("Beranda")
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Aplikasi Prediksi":
        try:
            run_ml_app()
        except ValueError as e:
            st.error(f"Terjadi error pada data: {e}")
            st.warning("Pastikan daftar `FINAL_COLUMN_ORDER` di dalam kode `app.py` sudah benar dan lengkap sesuai urutan saat training.")
        except Exception as e:
            st.error(f"Terjadi error tak terduga: {e}")

if __name__ == "__main__":
    main()