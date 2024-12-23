import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Judul aplikasi
st.title('Model Prediksi Pembatalan Reservasi Hotel')

# Deskripsi aplikasi
st.write("""
    Aplikasi ini membaca data reservasi hotel dari file Excel lokal dan 
    melakukan prediksi apakah reservasi akan dibatalkan atau tidak berdasarkan data yang ada.
""")

# Lokasi file Excel yang akan dibaca
file_path = "C:/Users/INFINIX/Documents/Semester 5/HotelDs/Hotel-Reservation-Dataset.xlsx"

try:
    # Membaca data dari file Excel
    df = pd.read_excel(file_path)

    # Cek apakah kolom 'booking_status' ada di data
    if 'booking_status' not in df.columns:
        st.error("Kolom 'booking_status' tidak ditemukan dalam dataset. Pastikan dataset memiliki kolom tersebut.")
    else:
        # Pisahkan fitur dan target
        X = df.drop('booking_status', axis=1)
        y = df['booking_status']

        # Identifikasi kolom kategori
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns

        # Encode kolom kategori pada data pelatihan
        label_encoders = {}
        for col in categorical_columns:
            label_encoders[col] = LabelEncoder()
            X[col] = label_encoders[col].fit_transform(X[col])

        # Hapus kolom yang tidak relevan seperti Booking_ID atau arrival_date jika ada
        X = X.drop(columns=['Booking_ID', 'arrival_date'], errors='ignore')

        # Latih model RandomForestClassifier
        rf_model = RandomForestClassifier()
        rf_model.fit(X, y)

        # Inisialisasi DataFrame untuk data baru
        if "data_baru" not in st.session_state:
            st.session_state.data_baru = pd.DataFrame(columns=list(X.columns) + ['booking_status'])

        # Form untuk input data baru
        with st.form("input_form"):
            booking_id = st.text_input("Masukkan Booking ID:")
            no_of_adults = st.number_input("Masukkan Jumlah Dewasa:", min_value=1, max_value=10)
            no_of_children = st.number_input("Masukkan Jumlah Anak-anak:", min_value=0, max_value=10)
            no_of_weekend_nights = st.number_input("Masukkan Jumlah Malam Akhir Pekan:", min_value=0, max_value=7)
            no_of_week_nights = st.number_input("Masukkan Jumlah Malam Mingguan:", min_value=0, max_value=7)
            type_of_meal_plan = st.selectbox("Pilih Jenis Paket Makanan:", df['type_of_meal_plan'].unique())
            room_type_reserved = st.selectbox("Pilih Jenis Kamar:", df['room_type_reserved'].unique())
            lead_time = st.number_input("Masukkan Lead Time:", min_value=0, max_value=365)
            market_segment_type = st.selectbox("Pilih Segmen Pasar:", df['market_segment_type'].unique())
            avg_price_per_room = st.number_input("Masukkan Harga Rata-rata per Kamar:", min_value=0.0)
            submitted = st.form_submit_button("Tambahkan")

            if submitted:
                if booking_id.strip():
                    # Tambahkan data baru ke dalam DataFrame sementara
                    new_data = pd.DataFrame({
                        "Booking_ID": [booking_id],
                        "no_of_adults": [no_of_adults],
                        "no_of_children": [no_of_children],
                        "no_of_weekend_nights": [no_of_weekend_nights],
                        "no_of_week_nights": [no_of_week_nights],
                        "type_of_meal_plan": [type_of_meal_plan],
                        "room_type_reserved": [room_type_reserved],
                        "lead_time": [lead_time],
                        "market_segment_type": [market_segment_type],
                        "avg_price_per_room": [avg_price_per_room]
                    })

                    # Encode data baru menggunakan encoder yang telah dilatih
                    for col in categorical_columns:
                        if col in new_data.columns:
                            if new_data[col].iloc[0] in label_encoders[col].classes_:
                                new_data[col] = label_encoders[col].transform(new_data[col])
                            else:
                                st.warning(f"Label '{new_data[col].iloc[0]}' tidak ditemukan di data pelatihan. Menggunakan nilai default.")
                                new_data[col] = -1  # Nilai default untuk kategori baru

                    # Tambahkan kolom yang hilang di data prediksi
                    missing_columns = set(X.columns) - set(new_data.columns)
                    for col in missing_columns:
                        new_data[col] = 0  # Nilai default untuk fitur yang hilang

                    # Pastikan urutan kolom sama dengan data pelatihan
                    new_data = new_data[X.columns]

                    # Pastikan hanya kolom numerik yang di-scaling
                    numerical_columns = new_data.select_dtypes(include=['int64', 'float64']).columns

                    # Scaling data baru
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X[numerical_columns])  # Fit scaler pada data pelatihan
                    new_data[numerical_columns] = scaler.transform(new_data[numerical_columns])

                    # Prediksi menggunakan model
                    pred_status = rf_model.predict(new_data)

                    # Menentukan hasil prediksi
                    status = 'Canceled' if pred_status[0] == 1 else 'Confirmed'

                    # Tambahkan hasil prediksi pada data baru
                    new_data['booking_status'] = status

                    # Simpan data baru yang telah diprediksi
                    st.session_state.data_baru = pd.concat([st.session_state.data_baru, new_data], ignore_index=True)
                    st.success(f"Data reservasi dengan Booking ID '{booking_id}' berhasil ditambahkan!")
                else:
                    st.error("Booking ID tidak boleh kosong.")

        # Tampilkan data baru yang telah ditambahkan
        st.write("Data Baru yang Ditambahkan:")
        st.write(st.session_state.data_baru)

except FileNotFoundError:
    st.error(f"File Excel '{file_path}' tidak ditemukan. Pastikan file berada di direktori yang benar.")
