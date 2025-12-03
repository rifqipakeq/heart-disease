import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score
from datetime import datetime

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
def load_data():
    try:
        df = pd.read_csv('heart.csv')
        if df.duplicated().sum() > 0:
            df.drop_duplicates(inplace=True)
        return df
    except:
        st.warning("File 'heart.csv' tidak ditemukan.")
        return None

@st.cache_resource
def build_model():
    df = load_data()
    if df is None:
        return None, None, None, None, None

    X = df.drop('target', axis=1)
    y = df['target']

    numerical_cols = ['age', 'thalach', 'oldpeak']
    categorical_cols = ['sex', 'cp', 'exang', 'slope', 'ca', 'thal']

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42,n_jobs=-1,criterion='entropy'))
    ])

    # training model
    model.fit(X_train, y_train)
    
    # prediksi
    y_pred = model.predict(X_test)
    
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    # evaluasi
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    metrics_dict = {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_test': y_test.values,
        'y_pred': y_pred,
        'X_test_shape': X_test.shape,  
        'X_train_shape': X_train.shape,  
        'X_test_transformed_shape': X_test_transformed.shape,  
        'X_train_transformed_shape': X_train_transformed.shape  
    }
    # Feature importance
    feature_names = numerical_cols.copy()
    encoder = preprocessor.named_transformers_['cat']
    cat_features = encoder.get_feature_names_out(categorical_cols)
    feature_names.extend(cat_features)
    
    importance = model.named_steps['classifier'].feature_importances_
    
    X_transformed = preprocessor.transform(X)
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
    X_transformed_df['target'] = y.values
    
    return model, dict(zip(feature_names, importance)), X_transformed_df, metrics_dict, X_test

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

model, feature_importance, df_encoded, metrics, X_test = build_model()
df = load_data()

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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Hasil Prediksi", "📈 Dashboard Analytics", "🎯 Feature Importance", "📋 Classification Report", "📜 Riwayat Prediksi", "ℹ️ Info Medis"])

with tab1:
    if model is None:
        st.error("Model belum siap. Pastikan file 'heart.csv' sudah diupload ke GitHub.")
    elif submit_btn:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        prob_percent = proba[1] * 100
        
        history_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'age': age,
            'sex': 'Laki-laki' if sex == 1 else 'Perempuan',
            'cp': ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'][cp],
            'thalach': thalach,
            'probability': f"{prob_percent:.1f}%",
            'prediction': 'Berisiko ⚠️' if prediction == 1 else 'Risiko Rendah ✅'
        }
        st.session_state.prediction_history.insert(0, history_entry)
        
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
    st.header("📈 Dashboard Analytics")
    
    if df is not None:
        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
        
        with col_metric1:
            st.metric("Total Pasien", len(df))
        with col_metric2:
            pct_disease = (df['target'].sum() / len(df) * 100)
            st.metric("Persentase Berisiko", f"{pct_disease:.1f}%")
        with col_metric3:
            st.metric("Rata-rata Usia", f"{df['age'].mean():.1f} tahun")
        with col_metric4:
            st.metric("Rata-rata HR Max", f"{df['thalach'].mean():.0f} bpm")
        
        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("Distribusi Target (Penyakit Jantung)")
            target_counts = df['target'].value_counts()
            fig_target = go.Figure(data=[go.Pie(
                labels=['Tidak Berisiko', 'Berisiko'],
                values=[target_counts.get(0, 0), target_counts.get(1, 0)],
                hole=0.4,
                marker_colors=['#90EE90', '#FF6B6B']
            )])
            fig_target.update_layout(height=300)
            st.plotly_chart(fig_target, use_container_width=True)
        
        with col_chart2:
            st.subheader("Distribusi Jenis Kelamin")
            sex_counts = df['sex'].value_counts()
            fig_sex = go.Figure(data=[go.Bar(
                x=['Perempuan', 'Laki-laki'],
                y=[sex_counts.get(0, 0), sex_counts.get(1, 0)],
                marker_color=['#FF69B4', '#4169E1']
            )])
            fig_sex.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_sex, use_container_width=True)
        
        col_chart3, col_chart4 = st.columns(2)
        
        with col_chart3:
            st.subheader("Distribusi Usia")
            fig_age = px.histogram(df, x='age', nbins=20, 
                                   color='target',
                                   color_discrete_map={0: '#90EE90', 1: '#FF6B6B'},
                                   labels={'target': 'Status', 'age': 'Usia', 'count': 'Jumlah'})
            fig_age.update_layout(height=300)
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col_chart4:
            st.subheader("Tipe Nyeri Dada vs Target")
            cp_target = df.groupby(['cp', 'target']).size().reset_index(name='count')
            fig_cp = px.bar(cp_target, x='cp', y='count', color='target',
                           barmode='group',
                           color_discrete_map={0: '#90EE90', 1: '#FF6B6B'},
                           labels={'cp': 'Tipe Chest Pain', 'count': 'Jumlah', 'target': 'Status'})
            fig_cp.update_layout(height=300)
            st.plotly_chart(fig_cp, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📊 Korelasi Antar Fitur")
        
        st.markdown("**Sebelum Encoding & Scaling**")
        numeric_cols_raw = df.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix_raw = df[numeric_cols_raw].corr()
        fig_corr_raw = px.imshow(corr_matrix_raw, 
                            text_auto='.2f',
                            color_continuous_scale='RdBu_r',
                            aspect='auto',
                            labels={'color': 'Correlation'})
        fig_corr_raw.update_layout(height=500)
        st.plotly_chart(fig_corr_raw, use_container_width=True)
    
        st.markdown("**Setelah Encoding & Scaling**")
        if df_encoded is not None:
            corr_matrix_encoded = df_encoded.corr()
            
            fig_corr_encoded = px.imshow(corr_matrix_encoded, 
                                text_auto='.2f',
                                color_continuous_scale='RdBu_r',
                                aspect='auto',
                                labels={'color': 'Correlation'})
            fig_corr_encoded.update_layout(height=800) 
            st.plotly_chart(fig_corr_encoded, use_container_width=True)
                
        
    else:
        st.warning("Data tidak tersedia untuk menampilkan analytics.")

with tab3:
    st.header("🎯 Feature Importance")
    st.markdown("""
    Grafik ini menunjukkan seberapa penting setiap fitur dalam menentukan prediksi model Random Forest.
    Semakin tinggi nilai importance, semakin besar pengaruh fitur tersebut terhadap hasil prediksi.
    """)
    
    if feature_importance is not None:
        importance_df = pd.DataFrame(list(feature_importance.items()), 
                                    columns=['Feature', 'Importance'])
        importance_df = importance_df.sort_values('Importance', ascending=True).tail(15)
        
        fig_importance = go.Figure(go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker=dict(
                color=importance_df['Importance'],
                colorscale='Viridis',
                showscale=True
            )
        ))
        
        fig_importance.update_layout(
            title='Top 15 Fitur Paling Berpengaruh',
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            height=600,
            showlegend=False
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        with st.expander("📊 Lihat Semua Feature Importance"):
            full_importance_df = pd.DataFrame(list(feature_importance.items()), 
                                             columns=['Feature', 'Importance'])
            full_importance_df = full_importance_df.sort_values('Importance', ascending=False)
            st.dataframe(full_importance_df, use_container_width=True)
    else:
        st.warning("Feature importance tidak tersedia.")

with tab4:
    st.header("🎯 Evaluasi Model")

    
    if metrics is not None:
        st.info(f"""
        **Informasi Dataset:**
        - Training Set: {metrics['X_train_shape'][0]} samples, {metrics['X_train_transformed_shape'][1]} features
        - Testing Set: {metrics['X_test_shape'][0]} samples, {metrics['X_test_transformed_shape'][1]} features
        """)
        
        col_met1, col_met2, col_met3, col_met4 = st.columns(4)
        
        with col_met1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        with col_met2:
            st.metric("Precision", f"{metrics['precision']:.2%}")
        with col_met3:
            st.metric("Recall", f"{metrics['recall']:.2%}")
        with col_met4:
            st.metric("F1-Score", f"{metrics['f1_score']:.2%}" )
        
        st.markdown("---")
        
        st.subheader("📊 Confusion Matrix")
        
        col_cm1, col_cm2 = st.columns([2, 1])
        
        with col_cm1:
            cm = metrics['confusion_matrix']
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted: No Disease', 'Predicted: Disease'],
                y=['Actual: No Disease', 'Actual: Disease'],
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 20},
                colorscale='Blues',
                showscale=True
            ))
            
            fig_cm.update_layout(
                title='Confusion Matrix - Random Forest',
                xaxis_title='Predicted Label',
                yaxis_title='Actual Label',
                height=400,
                font=dict(size=12)
            )
            
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col_cm2:
            st.write("")
            st.write("")
            st.info("""
            **Penjelasan Confusion Matrix:**
            
            * **True Negative (TN):** Model benar memprediksi "Tidak Sakit"
            * **False Positive (FP):** Model salah prediksi "Sakit" padahal "Tidak Sakit"
            * **False Negative (FN):** Model salah prediksi "Tidak Sakit" padahal "Sakit"
            * **True Positive (TP):** Model benar memprediksi "Sakit"
            """)
        
        col_cm1, col_cm2, col_cm3, col_cm4 = st.columns(4)
        tn, fp, fn, tp = cm.ravel()
        col_cm1.metric("True Negative", tn)
        col_cm2.metric("False Positive", fp)
        col_cm3.metric("False Negative", fn)
        col_cm4.metric("True Positive", tp)
        
        st.markdown("---")
        
        st.subheader("📖 Penjelasan Metrik Evaluasi")
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            st.markdown("""
            **Accuracy (Akurasi)**  
            Proporsi prediksi yang benar dari keseluruhan prediksi.
            
            **Precision (Presisi)**  
            Dari semua prediksi positif, berapa banyak yang benar-benar positif.
            """)
        
        with col_exp2:
            st.markdown("""
            **Recall (Sensitivitas)**  
            Dari semua data positif sebenarnya, berapa banyak yang berhasil diprediksi.
            
            **F1-Score**  
            Rata-rata harmonis antara Precision dan Recall.
            """)
    else:
        st.warning("Metrics tidak tersedia. Pastikan model sudah dibangun dengan benar.")


with tab5:
    st.header("📜 Riwayat Prediksi")
    
    if len(st.session_state.prediction_history) > 0:
        st.success(f"Total prediksi yang telah dilakukan: **{len(st.session_state.prediction_history)}**")
        
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, use_container_width=True)
        col_btn1, col_btn2 = st.columns([1, 5])
        with col_btn1:
            if st.button("🗑️ Hapus Riwayat"):
                st.session_state.prediction_history = []
                st.rerun()
        
        if len(history_df) > 0:
            st.markdown("---")
            st.subheader("Statistik Riwayat")
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                berisiko_count = history_df['prediction'].str.contains('Berisiko').sum()
                st.metric("Total Berisiko", berisiko_count)
            
            with col_stat2:
                aman_count = history_df['prediction'].str.contains('Risiko Rendah').sum()
                st.metric("Total Risiko Rendah", aman_count)
            
            with col_stat3:
                if 'age' in history_df.columns:
                    st.metric("Rata-rata Usia", f"{history_df['age'].mean():.1f} tahun")
    else:
        st.info("Belum ada riwayat prediksi. Silakan lakukan prediksi terlebih dahulu di tab 'Hasil Prediksi'.")

with tab6:
    st.header("ℹ️ Informasi Medis")
    st.markdown("""
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