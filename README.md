# 🐾 Animal Classification CNN Web Application

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://animal-image-classiication-dl.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/ikjotsingh221/animal-dataset)
[![Google Colab](https://img.shields.io/badge/Training-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/14jaMRPv6oqGLvbhoR9-o7w1puMghlp0z)

Aplikasi web interaktif berbasis **Deep Learning** menggunakan arsitektur **Convolutional Neural Network (CNN)** dengan teknik **Transfer Learning** untuk mengklasifikasikan spesies hewan dari foto yang diunggah pengguna atau melalui tangkapan kamera langsung (*real-time webcam*).

---

## 📌 Sumber Daya & Referensi Proyek

- **Dataset Sumber**: [Animal Dataset by ikjotsingh221 on Kaggle](https://www.kaggle.com/datasets/ikjotsingh221/animal-dataset)
- **Google Colab Notebook (Training & Evaluation)**: [Buka Google Colab](https://colab.research.google.com/drive/14jaMRPv6oqGLvbhoR9-o7w1puMghlp0z)
- **Repositori GitHub**: [azbayou/deep_learning_cnn](https://github.com/azbayou/deep_learning_cnn)
- **Live Deployment**: [animal-image-classiication-dl.streamlit.app](https://animal-image-classiication-dl.streamlit.app/)

---

## 🏷️ Kategori Kelas Hewan

Model CNN ini dilatih secara khusus untuk mengenali **4 kelas hewan**:
1. 🐻 **Bears** (Beruang)
2. 🦅 **Crows** (Burung Gagak)
3. 🐘 **Elephants** (Gajah)
4. 🐀 **Rats** (Tikus)

---

## 🧠 Algoritma & Pipeline Deep Learning

Pipeline Machine Learning dibangun secara modular mulai dari akuisisi data, augmentasi, pemodelan, hingga deployment:

```
[Raw Images] 
     │
     ▼
[Preprocessing & Augmentation] ──► [224x224 RGB, Normalized 0-1]
     │
     ▼
[MobileNetV2 Backbone] ──────────► [Feature Extraction via Inverted Residuals]
     │
     ▼
[Custom Conv2D + BN + GAP] ──────► [Domain-Specific High-Level Features]
     │
     ▼
[Dense + Dropout + Softmax] ─────► [4-Class Probabilities]
     │
     ▼
[Streamlit Web Inference] ───────► [Prediksi + Confidence Score]
```

### 1. Pembagian Data (Data Splitting)
Data dibagi secara **Stratified Split** untuk menjaga keseimbangan rasio kelas di setiap bagian:
- **Training Set (80%)**: Digunakan untuk melatih bobot model.
- **Validation Set (10%)**: Digunakan untuk tuning hyperparameter dan memantau overfitting per epoch.
- **Testing Set (10%)**: Digunakan untuk pengujian independen akhir terhadap data yang belum pernah dilihat model.

### 2. Pra-pemrosesan & Augmentasi Gambar (Preprocessing & Data Augmentation)
- **Resize**: Semua citra diubah seragam menjadi resolusi **224 × 224 piksel** (3 kanal warna RGB).
- **Normalisasi**: Nilai piksel `[0, 255]` diskalakan menjadi rentang `[0, 1]` (`float32`).
- **One-Hot Encoding**: Label kelas dikonversi ke vektor one-hot berdimensi 4 (`depth=4`).
- **Data Augmentation** (diterapkan dinamis pada data training):
  - `RandomFlip("horizontal")`: Membantu invariansi orientasi horizontal objek.
  - `RandomRotation(0.15)`: Rotasi acak hingga ±15% untuk mengatasi sudut pengambilan gambar yang variatif.
  - `RandomZoom(0.15)`: Zoom acak hingga 15% untuk variasi jarak objek hewan dari kamera.
  - `RandomContrast(0.1)`: Penyesuaian kontras acak untuk mengatasi perbedaan pencahayaan.
- **Optimasi Pipeline Data**: Menggunakan `tf.data.Dataset` dengan `shuffle`, `batch(32)`, dan `prefetch(AUTOTUNE)` untuk throughput training GPU maksimal di Google Colab.

---

## 🏗️ Arsitektur Model: Hybrid CNN (Transfer Learning)

Model utama yang digunakan adalah **Hybrid Deep CNN** yang mengombinasikan pre-trained backbone dengan custom convolutional head.

### Mengapa MobileNetV2?
1. **Efisiensi Tinggi**: Menggunakan *Depthwise Separable Convolutions* dan *Inverted Residual Blocks with Linear Bottlenecks*, menghasilkan representasi fitur yang sangat kaya dengan parameter yang relatif ringkas.
2. **Kinerja Akurat**: Bobot pretrained dari ImageNet membawa kemampuan dasar pemahaman tepi, tekstur, bentuk, dan pola umum sebelum dispesialisasikan ke domain hewan.
3. **Fine-Tuning Parsial**: Layer awal dibekukan (*frozen*), sedangkan **30 layer konvolusi teratas dibuka (*unfrozen*)** untuk dilatih bersama head baru agar model mampu menangkap fitur spesifik dataset hewan.

### Rincian Lapisan Model (`Animal_CNN_Classifier`)

| No | Layer Type | Spesifikasi / Hyperparameter | Fungsi Utama |
|---|---|---|---|
| 1 | **InputLayer** | Shape: `(224, 224, 3)` | Menerima citra input RGB berukuran 224×224 |
| 2 | **MobileNetV2 Backbone** | Pre-trained ImageNet (`include_top=False`), 30 top layers fine-tuned | Ekstraksi fitur visual hirarkis berdimensi tinggi |
| 3 | **Conv2D** | 256 Filter, Kernel 3×3, Padding `'same'`, Aktivasi `ReLU` | Mempelajari kombinasi pola spasial spesifik hewan |
| 4 | **BatchNormalization** | Momentum: 0.99, Epsilon: 0.001 | Menstabilkan distribusi aktivasi & mempercepat konvergensi |
| 5 | **GlobalAveragePooling2D** | Pooling spasial menjadi vektor 1D (256-d) | Mengurangi parameter secara drastis & mencegah overfitting |
| 6 | **Dropout** | Rate: `0.4` (40%) | Regularisasi untuk memutus dependensi ko-adaptasi antar neuron |
| 7 | **Dense** | 128 Unit, Aktivasi `ReLU` | Fully Connected Layer untuk klasifikasi representasi fitur |
| 8 | **BatchNormalization** | - | Normalisasi aktivasi internal |
| 9 | **Dropout** | Rate: `0.3` (30%) | Regularisasi tambahan sebelum output |
| 10 | **Dense (Output)** | 4 Unit, Aktivasi `Softmax` | Menghasilkan distribusi probabilitas untuk 4 kelas target |

---

## ⚙️ Pelatihan Model & Strategi Optimasi

- **Optimizer**: `Adam(learning_rate=1e-4)`
- **Loss Function**: `Categorical Crossentropy`
- **Target Kriteria**: Akurasi training dan validasi mencapai **≥ 95%**.
- **Callbacks yang Diterapkan**:
  1. `TargetAccuracyCallback`: Callback kustom yang secara otomatis menghentikan pelatihan ketika akurasi training dan validasi berhasil menembus target 95%.
  2. `EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)`: Menghentikan proses jika tidak ada perbaikan akurasi validasi selama 7 epoch berturut-turut dan mengembalikan bobot terbaik.
  3. `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)`: Menurunkan learning rate secara dinamis saat loss validasi mulai stagnan.

---

## 📊 Evaluasi & Hasil Pengujian

Model diuji pada data testing independen (10% dari total dataset) dan dievaluasi menggunakan metrik standar klasifikasi:
- **Accuracy**: Mengukur proporsi total prediksi yang benar.
- **Precision (Weighted)**: Mengukur ketepatan prediksi positif untuk setiap kelas.
- **Recall (Weighted)**: Mengukur kemampuan model dalam menangkap seluruh sampel positif.
- **F1-Score (Weighted)**: Rata-rata harmonik antara Precision dan Recall.
- **Confusion Matrix**: Matriks kontingensi untuk memvisualisasikan persebaran prediksi benar vs salah pada setiap kelas (Bears, Crows, Elephants, Rats).

---

## 💻 Fitur Aplikasi Streamlit (`app.py`)

1. **Dual Input Mode**:
   - 📁 **Upload Gambar**: Mendukung format `.jpg`, `.jpeg`, `.png`, dan `.webp`.
   - 📷 **Buka Kamera Langsung**: Pengambilan gambar secara instan melalui webcam/kamera smartphone menggunakan `st.camera_input`.
2. **Pra-pemrosesan Otomatis**: Citra di-*resize* ke 224×224 dan dinormalisasi persis sesuai format yang diharapkan model Keras.
3. **Hasil Prediksi Informatif**:
   - Label nama hewan yang terdeteksi dengan format huruf kapital jelas.
   - Nilai *Confidence Score* dalam persentase (%).
   - Indikator badge keyakinan (*Sangat Yakin*, *Cukup Yakin*, atau *Rendah*).
4. **Visualisasi Distribusi Probabilitas**: Grafik batang (*bar chart*) interaktif menampilkan persebaran probabilitas keyakinan model terhadap seluruh kelas hewan.
5. **Caching Model (`@st.cache_resource`)**: Model dimuat ke memori hanya sekali saat inisialisasi, memastikan respon prediksi cepat untuk setiap input baru.

---

## 📁 Struktur Berkas Repositori

```text
deep_learning_cnn/
├── .python-version          # Menentukan versi Python runtime (3.11) untuk Streamlit Cloud
├── animal_cnn_model.keras   # Model biner CNN Keras v3 hasil training Colab (~55 MB)
├── class_names.json         # Array label JSON: ["bears", "crows", "elephants", "rats"]
├── app.py                   # Antarmuka web Streamlit & logika inferensi
├── requirements.txt         # Daftar dependensi Python server Streamlit
└── README.md                # Dokumentasi lengkap proyek & arsitektur
```

---

## 🚀 Panduan Menjalankan Secara Lokal

### 1. Clone Repositori
```bash
git clone https://github.com/azbayou/deep_learning_cnn.git
cd deep_learning_cnn
```

### 2. Siapkan Python Virtual Environment (Disarankan)
```bash
python3 -m venv .venv
# Aktifkan di Linux / WSL / macOS:
source .venv/bin/activate
# Atau aktifkan di Windows Command Prompt / PowerShell:
# .venv\Scripts\activate
```

### 3. Pasang Dependensi
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi
```bash
streamlit run app.py
```
Aplikasi akan otomatis terbuka pada browser Anda di alamat default `http://localhost:8501`.

---

## ☁️ Deployment ke Streamlit Community Cloud

Aplikasi ini telah dikonfigurasi agar dapat langsung di-deploy melalui [Streamlit Community Cloud](https://share.streamlit.io):
1. Masuk ke [share.streamlit.io](https://share.streamlit.io) menggunakan akun GitHub.
2. Klik tombol **New app**.
3. Atur parameter deployment:
   - **Repository**: `azbayou/deep_learning_cnn`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Pastikan di **Advanced settings -> Python version** terpilih **Python 3.11** (didukung langsung oleh file `.python-version`).
5. Klik **Deploy!** 🎉
