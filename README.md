# AI Impact Classifier: Prediksi Masa Depan Pekerjaanmu
Proyek Machine Learning untuk memprediksi apakah sebuah pekerjaan termasuk kategori risiko Low, Medium, atau High terhadap otomatisasi AI di tahun 2030, berdasarkan karakteristik pekerjaannya.

## Isi Folder
- `Klasifikasi_Risiko_Otomatisasi_Pekerjaan_Tahun_2030.ipynb` → notebook lengkap: EDA, preprocessing, training, evaluasi, perbandingan model
- `app.py` → aplikasi Streamlit untuk deployment (prediksi interaktif)
- `model_logreg.pkl`, `scaler.pkl`, `label_encoder.pkl`, `feature_names.pkl` → hasil model yang sudah dilatih (dipakai oleh app.py)

## Cara Menjalankan

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

### 2. Buka & jalankan notebook
```bash
Klasifikasi_Risiko_Otomatisasi_Pekerjaan_Tahun_2030.ipynb
```
Jalankan semua cell (Run All). Ini akan menghasilkan ulang file `.pkl` jika belum ada.

### 3. Jalankan aplikasi Streamlit (Deployment)
Pastikan file `.pkl` ada di folder yang sama dengan `app.py`, lalu:
```bash
streamlit run app.py
```
Browser akan terbuka otomatis (biasanya di `http://localhost:8501`). Isi form, klik "Prediksi Risiko", dan lihat hasilnya.

## Ringkasan Hasil
| Model | Akurasi |
|---|---|
| Logistic Regression | ~99.7% |
| MLPClassifier (Neural Network) | ~93.5% |

Logistic Regression dipilih sebagai model final karena performa lebih tinggi dan lebih sederhana.
