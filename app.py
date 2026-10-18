import json
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# Konfigurasi Halaman
st.set_page_config(
    page_title="Animal Classifier CNN",
    page_icon="🐾",
    layout="centered"
)

st.title("🐾 Klasifikasi Gambar Hewan (CNN)")
st.write(
    "Aplikasi Deep Learning berbasis **Convolutional Neural Network (TensorFlow)** "
    "untuk mendeteksi jenis hewan melalui **Upload Gambar** atau **Kamera Langsung**."
)

# Load Model & Daftar Kelas (dengan caching agar cepat)
@st.cache_resource
def load_cnn_model():
    model = tf.keras.models.load_model("animal_cnn_model.keras")
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    return model, class_names

try:
    model, class_names = load_cnn_model()
except Exception as e:
    st.error(
        f"Gagal memuat model atau class_names.json. Pastikan kedua file berada di satu folder dengan app.py. Detail: {e}"
    )
    st.stop()

# Fungsi Pra-pemrosesan Gambar Input User
def preprocess_user_image(image: Image.Image):
    image = image.convert("RGB")
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
    return img_batch

# Pilihan Metode Input
st.subheader("Pilih Sumber Gambar")
input_method = st.radio(
    "Metode Input:",
    ("📁 Upload Gambar", "📷 Buka Kamera"),
    horizontal=True
)

input_image = None

if input_method == "📁 Upload Gambar":
    uploaded_file = st.file_uploader(
        "Unggah foto hewan (JPG, JPEG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"]
    )
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)

elif input_method == "📷 Buka Kamera":
    camera_photo = st.camera_input("Arahkan kamera ke objek hewan lalu ambil foto")
    if camera_photo is not None:
        input_image = Image.open(camera_photo)

# Proses Prediksi jika Gambar Tersedia
if input_image is not None:
    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(input_image, caption="Gambar Input", use_container_width=True)

    with col2:
        with st.spinner("Menganalisis gambar dengan CNN..."):
            processed_tensor = preprocess_user_image(input_image)
            predictions = model.predict(processed_tensor, verbose=0)[0]

            top_idx = int(np.argmax(predictions))
            predicted_label = class_names[top_idx]
            confidence = float(predictions[top_idx]) * 100

        st.subheader("Hasil Prediksi")
        st.metric(label="Prediksi Hewan", value=predicted_label.upper())
        st.metric(label="Tingkat Keyakinan (Confidence)", value=f"{confidence:.2f}%")

        if confidence >= 80:
            st.success("Model sangat yakin dengan hasil prediksi ini.")
        elif confidence >= 50:
            st.warning("Model cukup yakin, namun pastikan pencahayaan gambar jelas.")
        else:
            st.info("Tingkat keyakinan rendah. Coba gunakan sudut foto yang lebih jelas.")

    # Visualisasi Top-5 Probabilitas Kelas
    st.markdown("#### 📊 Distribusi Probabilitas (Top 5 Kelas)")
    top_k = min(5, len(class_names))
    top_indices = np.argsort(predictions)[::-1][:top_k]

    prob_df = pd.DataFrame({
        "Kelas Hewan": [class_names[i] for i in top_indices],
        "Probabilitas (%)": [float(predictions[i]) * 100 for i in top_indices]
    }).set_index("Kelas Hewan")

    st.bar_chart(prob_df)