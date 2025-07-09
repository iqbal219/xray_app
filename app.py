import streamlit as st
import cv2
import numpy as np
import pyfeats
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ========== Konfigurasi App ==========
st.set_page_config(page_title="Sistem Deteksi Penyakit Paru-Paru", layout="wide")
st.title("🩺 Sistem Deteksi Penyakit Paru-Paru pada Citra X-Ray Dada")

st.markdown("""
<style>
.big-font { font-size:20px !important; font-weight:bold; }
.small-font { font-size:14px !important; color: gray; }
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="small-font">Menggunakan ekstraksi fitur <b>FOS + GLCM</b> dan klasifikasi SVM</p>', unsafe_allow_html=True)

# ========== Upload Gambar ==========
st.sidebar.header("📤 Upload Gambar")
uploaded_files = st.sidebar.file_uploader(
    "Pilih gambar X-ray (satu atau beberapa)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ========== Load Model dan Aset ==========
@st.cache_resource
def load_model_and_assets():
    model = joblib.load("model_svm.pkl")
    scaler = joblib.load("scaler.pkl")
    mask = joblib.load("best_mask_kombinasi.pkl")
    labels = ['Normal', 'Tuberculosis', 'Pneumonia', 'Covid19']
    return model, scaler, mask, labels

# ========== Fungsi Preprocessing ==========
def preprocess_image(image_bgr):
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (224, 224))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_resized)
    return img_gray, img_clahe

# ========== Ekstraksi Fitur ==========
@st.cache_data
def extract_fos_glcm(image):
    mask = np.ones_like(image)
    fos_feats, _ = pyfeats.fos(image, mask)
    glcm_feats = pyfeats.glcm_features(image)
    glcm_combined = np.concatenate([glcm_feats[0], glcm_feats[1]])
    drop = [12, 13, 14, 20, 26, 27, 28]
    drop = [i for i in drop if i < len(glcm_combined)]
    glcm_filtered = np.delete(glcm_combined, drop)
    return fos_feats[:10], glcm_filtered

# ========== Main Logic ==========
if uploaded_files:
    model, scaler, mask, labels = load_model_and_assets()

    st.subheader("📋 Hasil Prediksi")
    results = []
    image_dict = {}

    for file in uploaded_files:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_gray, img_clahe = preprocess_image(img_bgr)
        fos, glcm = extract_fos_glcm(img_clahe)
        combined = np.concatenate([fos, glcm])
        masked_feat = combined[mask]
        scaled_feat = scaler.transform([masked_feat])
        pred = model.predict(scaled_feat)[0]

        results.append({
            "Nama File": file.name,
            "Prediksi": labels[pred]
        })

        image_dict[file.name] = {
            "rgb": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
            "clahe": cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB),
            "pred": labels[pred],
            "features": combined,
            "masked": masked_feat,
            "scaled": scaled_feat
        }

    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)

    selected_file = df_results["Nama File"].tolist()[0] if len(df_results) == 1 else st.selectbox("📌 Lihat detail prediksi untuk:", df_results["Nama File"].tolist())
    data = image_dict[selected_file]

    st.subheader(f"🖼️ Visualisasi Gambar: {selected_file}")
    col1, col2 = st.columns(2)
    with col1:
        st.image(data["rgb"], caption="Gambar Asli", use_container_width=True)
    with col2:
        st.image(data["clahe"], caption="Hasil Preprocessing", use_container_width=True)

    st.subheader("🧪 Fitur Hasil Ekstraksi")
    fos_names = ["FOS_Mean", "FOS_Std", "FOS_Median", "FOS_Mode", "FOS_Skewness", "FOS_Kurtosis", "FOS_Energy", "FOS_Entropy", "FOS_Min", "FOS_Max"]
    glcm_names = [
        "GLCM_ASM_Mean", "GLCM_Contrast_Mean", "GLCM_Correlation_Mean", "GLCM_SumOfSquaresVariance_Mean",
        "GLCM_InverseDifferenceMoment_Mean", "GLCM_SumAverage_Mean", "GLCM_SumVariance_Mean", "GLCM_SumEntropy_Mean",
        "GLCM_Entropy_Mean", "GLCM_DifferenceVariance_Mean", "GLCM_DifferenceEntropy_Mean",
        "GLCM_ASM_Range", "GLCM_Contrast_Range", "GLCM_Correlation_Range", "GLCM_SumOfSquaresVariance_Range",
        "GLCM_InverseDifferenceMoment_Range", "GLCM_SumAverage_Range", "GLCM_SumVariance_Range",
        "GLCM_SumEntropy_Range", "GLCM_Entropy_Range", "GLCM_DifferenceVariance_Range", "GLCM_DifferenceEntropy_Range"
    ]
    all_feature_names = fos_names + glcm_names

    df_all = pd.DataFrame([data['features']], columns=all_feature_names[:len(data['features'])])
    st.dataframe(df_all, use_container_width=True)

    st.subheader("✂️ Fitur Setelah Feature Removal")
    df_removed = pd.DataFrame([data['masked']], columns=np.array(all_feature_names)[mask])
    st.dataframe(df_removed, use_container_width=True)

    st.subheader("📏 Fitur Setelah Scaling (Standardisasi)")
    df_scaled = pd.DataFrame(data['scaled'], columns=np.array(all_feature_names)[mask])
    st.dataframe(df_scaled, use_container_width=True)

    st.subheader("📌 Prediksi")
    st.success(f"✅ Kelas Terprediksi: **{data['pred']}**")
