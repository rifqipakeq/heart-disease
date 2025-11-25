import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Prediksi Penyakit Jantung Koroner", layout="wide")

st.title("❤️ Aplikasi Prediksi Penyakit Jantung Koroner")
st.write("Model Random Forest")

@st.cache_resource
def load_model():
    try:
        model = joblib.load('heart_disease_model.pkl')
        return model
    except FileNotFoundError:
        st.error("File 'heart_disease_model.pkl' tidak ditemukan. Pastikan file model ada di folder yang sama.")
        return None

model = load_model()

if model is None:
    st.stop() 

st.sidebar.header("Input Data Pasien")

def user_input_features():
    age = st.sidebar.number_input("Usia", 1, 120, 50)
    sex = st.sidebar.selectbox("Jenis Kelamin", [0, 1], format_func=lambda x: "Wanita (0)" if x==0 else "Pria (1)")
    cp = st.sidebar.selectbox("Tipe Nyeri Dada (CP)", [0, 1, 2, 3])
    thalach = st.sidebar.number_input("Detak Jantung Max (Thalach)", 50, 250, 150)
    exang = st.sidebar.selectbox("Angina Olahraga (Exang)", [0, 1], format_func=lambda x: "Tidak (0)" if x==0 else "Ya (1)")
    oldpeak = st.sidebar.number_input("ST Depression", 0.0, 10.0, 1.0)
    slope = st.sidebar.selectbox("Slope", [0, 1, 2])
    ca = st.sidebar.selectbox("Jml Pembuluh Darah (CA)", [0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox("Thalassemia", [0, 1, 2, 3])
    fbs = st.sidebar.selectbox("Gula Darah > 120 (FBS)", [0, 1], format_func=lambda x: "False (0)" if x==0 else "True (1)")

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
        'ca': ca, 'thal': thal, 'fbs': fbs
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader("Data Input")
st.dataframe(input_df)

if st.button("Prediksi"):
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction[0] == 1:
            st.error("⚠️ Terindikasi Penyakit Jantung Koroner")
        else:
            st.success("✅ Normal / Tidak Terindikasi")
            
    with col2:
        st.metric("Probabilitas Sakit", f"{proba[0][1]*100:.1f}%")

if st.checkbox("Tampilkan Feature Importance"):
    try:
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        importances = model.named_steps['classifier'].feature_importances_
        
        feat_df = pd.DataFrame({'Fitur': feature_names, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Importance', y='Fitur', data=feat_df, palette='viridis', ax=ax)
        st.pyplot(fig)
    except:
        st.info("Tidak dapat menampilkan visualisasi feature importance pada model ini.")