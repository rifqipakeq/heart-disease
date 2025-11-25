import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="HeartGuard AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        height: 3em;
        border-radius: 10px;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def build_model():
    try:
        df = pd.read_csv('heart.csv') 
    except:
        st.warning("File 'heart.csv' tidak ditemukan. Menggunakan data dummy untuk demo UI.")
        return None 

    if df.duplicated().sum() > 0:
        df.drop_duplicates(inplace=True)

    X = df.drop('target', axis=1)
    y = df['target']

    numerical_cols = ['age', 'thalach', 'oldpeak']
    categorical_cols = ['sex', 'cp', 'fbs', 'exang', 'slope', 'ca', 'thal']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])

    model.fit(X, y)
    return model

model = build_model()

col_head1, col_head2 = st.columns([1, 4])
with col_head1:
    st.image("https://img.icons8.com/color/144/heart-with-pulse.png", width=100) 
with col_head2:
    st.title("Klasisfikasi Risiko Penyakit Jantung Koroner dengan Model Random Forest")
    st.markdown("Sistem prediksi risiko penyakit jantung koroner berbasis *Machine Learning*. Masukkan parameter klinis di sidebar untuk memulai.")

st.markdown("---")

st.sidebar.header("📋 Parameter Pasien")

with st.sidebar.form("patient_form"):
    st.subheader("Data Demografis")
    age = st.slider("Usia", 20, 100, 50, help="Usia pasien dalam tahun")
    sex = st.radio("Jenis Kelamin", [1, 0], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan", horizontal=True)
    
    st.subheader("Tanda Vital")
    thalach = st.slider("Detak Jantung Maksimum", 60, 220, 150, help="Detak jantung tertinggi yang dicapai saat tes latihan")
    oldpeak = st.slider("Depresi ST (Oldpeak)", 0.0, 6.0, 1.0, 0.1, help="Penurunan ST yang diinduksi oleh latihan relatif terhadap istirahat")
    
    st.subheader("Riwayat Medis")
    cp = st.selectbox("Tipe Nyeri Dada (CP)", [0, 1, 2, 3], 
                      format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x],
                      help="Jenis nyeri dada yang dirasakan")
    
    col_side1, col_side2 = st.columns(2)
    with col_side1:
        exang = st.checkbox("Angina Olahraga?", help="Nyeri dada saat berolahraga")
        fbs = st.checkbox("Gula Darah > 120?", help="Fasting Blood Sugar > 120 mg/dl")
    
    with col_side2:
        slope = st.selectbox("Slope ST", [0, 1, 2], help="Kemiringan segmen ST latihan puncak")
    
    ca = st.selectbox("Jumlah Pembuluh Besar (CA)", [0, 1, 2, 3, 4], help="Jumlah pembuluh darah besar (0-3) yang diwarnai dengan fluoroskopi")
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3], 
                        format_func=lambda x: ["Unknown", "Normal", "Fixed Defect", "Reversable Defect"][x])

    submit_btn = st.form_submit_button("🔍 Analisis Risiko")

input_data = {
    'age': age, 'sex': sex, 'cp': cp, 'thalach': thalach,
    'exang': 1 if exang else 0, 'oldpeak': oldpeak, 'slope': slope,
    'ca': ca, 'thal': thal, 'fbs': 1 if fbs else 0
}
input_df = pd.DataFrame(input_data, index=[0])

tab1, tab2 = st.tabs(["📊 Hasil Prediksi", "ℹ️ Kamus Medis & Info"])

with tab1:
    if model is None:
        st.error("Model belum siap. Pastikan file 'heart.csv' sudah diupload ke GitHub.")
    elif submit_btn:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        prob_percent = proba[1] * 100
        
        col_res1, col_res2 = st.columns([2, 1])
        
        with col_res1:
            st.subheader("Analisis Tingkat Risiko")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob_percent,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probabilitas Penyakit Jantung"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps' : [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "salmon"}],
                    'threshold' : {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': prob_percent}}))
            st.plotly_chart(fig, use_container_width=True)

        with col_res2:
            st.write("") 
            st.write("")
            if prediction == 1:
                st.error("""
                ### ⚠️ Terindikasi Berisiko
                Berdasarkan pola data, pasien memiliki kemiripan karakteristik dengan penderita penyakit jantung.
                
                **Saran:**
                * Segera konsultasi ke Dokter Jantung.
                * Lakukan tes EKG lanjutan.
                """)
            else:
                st.success("""
                ### ✅ Risiko Rendah
                Berdasarkan data input, pasien cenderung berada dalam kondisi normal.
                
                **Saran:**
                * Pertahankan gaya hidup sehat.
                * Rutin olahraga dan jaga pola makan.
                """)
                
        with st.expander("Lihat Detail Data Pasien yang Dimasukkan"):
            st.dataframe(input_df)

    else:
        st.info("👈 Silakan lengkapi data di sidebar kiri dan klik tombol **'Analisis Risiko'**.")

with tab2:
    st.markdown("""
    ### 📚 Penjelasan Istilah Medis
    Agar hasil prediksi lebih mudah dipahami, berikut adalah penjelasan singkat parameter yang digunakan:
    
    #### 1. Age (Usia)
    Usia pasien. Risiko penyakit jantung meningkat signifikan setelah:
    * **Pria ≥ 45 tahun**
    * **Wanita ≥ 55 tahun**

    #### 2. Sex (Jenis Kelamin)
    * **1 = Pria**
    * **0 = Wanita**
    Pria cenderung memiliki risiko penyakit jantung lebih tinggi pada usia lebih muda.

    #### 3. Nyeri Dada (Chest Pain / CP)
    * **Typical Angina:** Rasa sakit dada klasik akibat jantung (tekanan/berat).
    * **Atypical Angina:** Nyeri dada tetapi tidak memiliki gejala klasik jantung.
    * **Non-anginal Pain:** Nyeri dada yang bukan disebabkan oleh masalah jantung.
    * **Asymptomatic:** Tidak ada gejala nyeri dada.

    #### 4. Thalassemia (Thal)
    Kelainan darah genetik.
    * **Normal:** Aliran darah normal.
    * **Fixed Defect:** Tidak ada aliran darah di beberapa bagian jantung.
    * **Reversible Defect:** Aliran darah diamati tetapi tidak normal.

    #### 5. Depresi ST (Oldpeak) & Slope
    Mengacu pada pembacaan EKG saat berolahraga.
    * **Oldpeak:** Menunjukkan seberapa "stress" jantung saat berolahraga dibandingkan saat istirahat. Semakin tinggi nilainya, semakin besar indikasi masalah.
                
    #### 6. Jumlah Pembuluh Darah Terlihat oleh Fluroskopi (ca)
    Jumlah pembuluh koroner besar yang terlihat menyempit pada fluoroskopi (0–3).
    Semakin banyak pembuluh yang tersumbat → semakin tinggi risiko penyakit jantung.
                
    #### 7. Fasting Blood Sugar (fbs)
    Apakah gula darah puasa > 120 mg/dl?
    * **1 = Ya (tinggi)**
    * **0 = Tidak**
    Gula darah tinggi berhubungan dengan resistensi insulin dan penyakit jantung.

    #### 8. Exercise-Induced Angina (exang)
    Nyeri dada yang muncul saat olahraga:
    * **1 = Ya**
    * **0 = Tidak**
                
    #### 9. Max Heart Rate Achieved (thalach)
    Denyut jantung maksimum yang dicapai saat tes treadmill/olahraga.
    Nilai lebih rendah dari normal dapat mengindikasikan gangguan fungsi jantung.
    """)

st.markdown("---")