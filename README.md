### Sales Forecasting System (Streamlit)

Aplikasi web Streamlit untuk melakukan forecasting penjualan multi-produk berbasis model LSTM. Aplikasi ini membungkus skrip yang sudah ada (train_models.py dan forecast.py) sehingga pengguna non-teknis dapat:

- Upload file Excel data penjualan historis
- Melatih model LSTM per produk
- Menghasilkan prediksi 24 bulan ke depan
- Melihat visualisasi interaktif dan analitik
- Mengunduh hasil (CSV/PNG/ZIP)

### Struktur Folder

- app.py — Halaman Home/Dashboard
- pages/
  - 1_Upload_Data.py
  - 2_Train_Models.py
  - 3_Generate_Forecast.py
  - 4_Results_Analytics.py
- utils/
  - data_handler.py
  - training_wrapper.py
  - forecast_wrapper.py
  - chart_generator.py
  - validators.py
- config/
  - settings.py
- assets/
  - styles.css
- uploads/ — tempat file Excel yang diupload
- trained_models/ — model hasil training (sudah ada pada repo)
- outputs/ — folder umum untuk output tambahan (opsional)

### Persyaratan

Lihat requirements.txt, termasuk:
- streamlit (>=1.30.0)
- pandas, numpy, plotly, openpyxl
- scikit-learn, tensorflow, keras
- python-dateutil

Install dependencies:

```
pip install -r requirements.txt
```

### Menjalankan Aplikasi

```
streamlit run app.py
```

Akses melalui browser:

http://localhost:8501

### Catatan Penting

- Script train_models.py dan forecast.py tidak dimodifikasi, hanya dipanggil via subprocess oleh aplikasi.
- Hasil output CSV yang dihasilkan oleh skrip tetap berada di direktori proyek (root) dan folder trained_models.
- Aplikasi menggunakan bahasa Indonesia untuk teks yang berinteraksi dengan user.

### Troubleshooting

- Jika tombol Generate Forecast atau Train tidak bekerja, cek bahwa Python environment memiliki TensorFlow/Keras versi yang kompatibel.
- Jika file tidak terdeteksi, pastikan nama dan format kolom sesuai dan file berada di folder uploads/ (atau gunakan file default Data_Penjualan_Dengan_ID_Pelanggan.xlsx di root).
- Pada Windows, pastikan aplikasi dijalankan dari direktori proyek ini (D:/tesis).
