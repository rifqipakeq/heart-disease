# ❤️ HeartGuard AI - Klasifikasi Risiko Penyakit Jantung Koroner

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Sistem prediksi risiko penyakit jantung koroner berbasis *Machine Learning* menggunakan algoritma **Random Forest Classifier**. Aplikasi web interaktif ini dibangun dengan Streamlit untuk membantu tenaga medis dalam mengidentifikasi pasien yang berisiko terkena penyakit jantung.

## 👥 Anggota Kelompok

| Kelas | NIM | Nama |
| :---: | :--- | :--- |
| IF - C | 123230014 | Randra Ferdian Saputra |
| IF - C | 123230030 | Reza Rasendriya Adi Putra |
| IF - C | 123230119 | Rifqi Rahardian |

---

## 📋 Daftar Isi

- [Tentang Proyek](#-tentang-proyek)
- [Fitur Utama](#-fitur-utama)
- [Dataset](#-dataset)
- [Model Machine Learning](#-model-machine-learning)
- [Instalasi](#-instalasi)
- [Cara Penggunaan](#-cara-penggunaan)
- [Struktur Proyek](#-struktur-proyek)
- [Hasil Evaluasi Model](#-hasil-evaluasi-model)
- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
- [Screenshot](#-screenshot)
- [Kontribusi](#-kontribusi)
- [Lisensi](#-lisensi)

---

## 🎯 Tentang Proyek

Penyakit jantung koroner merupakan salah satu penyebab kematian tertinggi di dunia. Deteksi dini sangat penting untuk pencegahan dan penanganan yang tepat. **HeartGuard AI** dikembangkan untuk:

- ✅ Memprediksi risiko penyakit jantung berdasarkan parameter klinis pasien
- ✅ Memberikan visualisasi data yang interaktif dan mudah dipahami
- ✅ Membantu tenaga medis dalam pengambilan keputusan klinis
- ✅ Menyediakan analisis feature importance untuk interpretasi model

---

## ✨ Fitur Utama

### 1. 📊 Prediksi Risiko Real-time
- Input parameter pasien melalui sidebar interaktif
- Prediksi probabilitas risiko penyakit jantung
- Visualisasi gauge chart untuk tingkat risiko
- Rekomendasi tindak lanjut berdasarkan hasil prediksi

### 2. 📈 Dashboard Analytics
- Distribusi target (pasien berisiko vs tidak berisiko)
- Visualisasi berdasarkan jenis kelamin, usia, dan tipe nyeri dada
- Matriks korelasi sebelum dan sesudah preprocessing
- Statistik deskriptif dataset

### 3. 🎯 Feature Importance Analysis
- Visualisasi top 15 fitur paling berpengaruh
- Interpretasi kontribusi setiap fitur terhadap prediksi
- Bar chart interaktif dengan color scale

### 4. 📋 Classification Report
- Confusion Matrix dengan visualisasi heatmap
- Metrik evaluasi: Accuracy, Precision, Recall, F1-Score
- Penjelasan detail setiap metrik evaluasi

### 5. 📜 Riwayat Prediksi
- Pencatatan semua prediksi yang telah dilakukan
- Statistik agregat riwayat prediksi
- Export data riwayat dalam format tabel

### 6. ℹ️ Informasi Medis
- Penjelasan lengkap setiap parameter klinis
- Panduan interpretasi hasil untuk tenaga medis

---

## 📊 Dataset

Dataset yang digunakan adalah **Heart Disease UCI** yang berisi 303 sampel pasien dengan 14 atribut:

### Fitur yang Digunakan (Setelah Feature Selection):
| Fitur | Deskripsi | Tipe |
|-------|-----------|------|
| `age` | Usia pasien (tahun) | Numerik |
| `sex` | Jenis kelamin (1=Laki-laki, 0=Perempuan) | Kategorikal |
| `cp` | Tipe nyeri dada (0-3) | Kategorikal |
| `thalach` | Detak jantung maksimum | Numerik |
| `exang` | Angina akibat olahraga (1=Ya, 0=Tidak) | Kategorikal |
| `oldpeak` | Depresi ST akibat latihan | Numerik |
| `slope` | Kemiringan segmen ST (0-2) | Kategorikal |
| `ca` | Jumlah pembuluh darah utama (0-4) | Kategorikal |
| `thal` | Hasil Thallium stress test (0-3) | Kategorikal |
| `target` | Diagnosis (1=Berisiko, 0=Tidak) | Target |

### Fitur yang Dihapus (Low Correlation):
- `trestbps` (Tekanan darah istirahat)
- `chol` (Kolesterol serum)
- `fbs` (Gula darah puasa)
- `restecg` (Hasil EKG istirahat)

---

## 🚀 Instalasi

### Prerequisites:
- Python 3.8 atau lebih tinggi
- pip (Python package manager)

### Langkah-langkah Instalasi:

1. **Clone Repository**
```bash
git clone https://github.com/reza675/ProjectDS_HeartDisease.git
cd ProjectDS_HeartDisease/heart-disease
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Pastikan File Dataset Ada**
- File `heart.csv` harus berada di folder yang sama dengan `app.py`

4. **Jalankan Aplikasi**
```bash
streamlit run app.py
```

5. **Akses Aplikasi**
- Buka browser dan akses: `http://localhost:8501`

---

## 💻 Cara Penggunaan

### 1. Input Data Pasien
- Buka sidebar di sebelah kiri
- Isi semua parameter klinis pasien:
  - **Data Demografis**: Usia, Jenis Kelamin
  - **Tanda Vital**: Detak Jantung Maksimum, Depresi ST
  - **Riwayat Medis**: Tipe Nyeri Dada, Angina Olahraga, Slope ST, CA, Thalassemia

### 2. Analisis Risiko
- Klik tombol **"🔍 Analisis Risiko"**
- Lihat hasil prediksi berupa:
  - Gauge chart probabilitas risiko
  - Status risiko (Berisiko/Risiko Rendah)
  - Rekomendasi tindak lanjut

### 3. Eksplorasi Dashboard
- Tab **Dashboard Analytics**: Lihat visualisasi data dan statistik
- Tab **Feature Importance**: Analisis fitur yang paling berpengaruh
- Tab **Classification Report**: Evaluasi performa model
- Tab **Riwayat Prediksi**: Lihat semua prediksi yang telah dilakukan
- Tab **Info Medis**: Baca penjelasan parameter klinis

---

## 📁 Struktur Proyek

```
heart-disease/
│
├── app.py                          # Aplikasi Streamlit utama
├── heart.csv                       # Dataset Heart Disease UCI
├── requirements.txt                # Python dependencies
└── README.md                       # Dokumentasi proyek
```

---

## 🛠️ Teknologi yang Digunakan

### Framework & Libraries:
- **[Streamlit](https://streamlit.io/)**: Framework web app interaktif
- **[scikit-learn](https://scikit-learn.org/)**: Machine learning library
- **[Pandas](https://pandas.pydata.org/)**: Data manipulation
- **[NumPy](https://numpy.org/)**: Numerical computing
- **[Plotly](https://plotly.com/)**: Interactive visualizations

---

## 📝 Lisensi

Distributed under the MIT License. See `LICENSE` for more information.

---