import streamlit as st
import joblib
import numpy as np
import pandas as pd
import altair as alt

model = joblib.load('model_logreg.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
feature_names = joblib.load('feature_names.pkl')

st.set_page_config(page_title="AI Impact Classifier", page_icon="🤖")

st.title("AI Impact Classifier: Prediksi Masa Depan Pekerjaanmu")
st.write(
    "Aplikasi ini memprediksi apakah sebuah pekerjaan termasuk kategori risiko "
    "**Low**, **Medium**, atau **High** terhadap otomatisasi AI di tahun 2030, "
    "berdasarkan karakteristik pekerjaannya."
)

st.divider()
st.subheader("Masukkan Data Pekerjaan")

col1, col2 = st.columns(2)

with col1:
    USD_TO_IDR = 16000
    INDO_FACTOR = 7
   
    min_idr_month = int(((10000 / INDO_FACTOR) * USD_TO_IDR) / 12)    
    max_idr_month = int(((300000 / INDO_FACTOR) * USD_TO_IDR) / 12)   
    default_idr_month = int(((70000 / INDO_FACTOR) * USD_TO_IDR) / 12)
   
    salary_idr_month = st.number_input(
        "Gajimu rata-rata sebulan berapa nih? (Rp)",
        min_value=min_idr_month, 
        max_value=max_idr_month, 
        value=5000000, 
        step=500000,
        format="%d"
    )
   
    salary_idr_year = salary_idr_month * 12
    
    st.caption(f"Input terdeteksi: **Rp {salary_idr_month:,} / bulan**".replace(",", "."))
    st.caption(f"Total: ≈ Rp {salary_idr_year:,.0f} / tahun".replace(",", "."))
   
    salary = (salary_idr_year * INDO_FACTOR) / USD_TO_IDR

    st.caption(
        f"**Informasi Sistem:** Nilai ini setara dengan **\\${salary:,.0f} / tahun** "
        f"pada standar ekonomi negara maju (Dataset Model)."
    )

    if salary < 30000 or salary > 150000:
        st.warning(
            f"Warning: Gaji yang Anda masukkan (\\$26,250) agak terlalu rendah dari standar sistem kami"
            f"(\\$30,000–$150,000/tahun). Hasil perkiraan di bawah mungkin menjadi kurang pas atau kurang akurat."
        )

    experience = st.number_input("Udah berapa tahun nih nyemplung di dunia kerja?", min_value=0, max_value=40, value=5)
    education_ui = st.selectbox(
            "Pendidikan terakhirmu apa?", 
            ["SMA/SMK", "Sarjana (S1)", "Magister (S2)", "Doktor (S3)"]
        )
    
    edu_mapping = {
        "SMA/SMK": "High School",
        "Sarjana (S1)": "Bachelor's",
        "Magister (S2)": "Master's",
        "Doktor (S3)": "PhD"
    }
    
    education = edu_mapping[education_ui]

    ai_exposure_ui = st.slider(
        "Seberapa sering kerjaanmu dibantu/pakai AI pas kerja sehari-hari? (1 = Gak pernah, 10 = Nempel banget)",
        1, 10, 5
    )
    
    ai_exposure = (ai_exposure_ui - 1) / 9

    tech_growth_ui = st.slider(
        "Menurutmu, seberapa cepat teknologi bakal berkembang di bidang kerjaanmu? (1 = Lambat banget, 10 = Cepat banget)",
        1, 10, 5
    )
  
    tech_growth = 0.5 + ((tech_growth_ui - 1) / 9) * (3.0 - 0.5)

    automation_prob_ui = st.slider(
        "Kira-kira, seberapa besar potensi kerjaanmu bakal digantiin robot/AI di 2030 nanti? (1 = Mustahil, 10 = Pasti banget)",
        1, 10, 5
    )

    automation_prob = (automation_prob_ui - 1) / 9

with col2:
    st.write("**Yuk, rating skill kamu sekarang! (Skala 1-10)**")
    skill_labels = [
        "Creativity (Kreativitas)", "Data Analysis (Analisis Data)", "Communication (Komunikasi)",
        "Problem Solving (Pemecahan Masalah)", "Technical Skills (Kemampuan Teknis)",
        "Critical Thinking (Berpikir Kritis)", "Robotics/Manual Dexterity (Keterampilan Manual/Robotik)",
        "Leadership (Kepemimpinan)", "Emotional Intelligence (Kecerdasan Emosional)",
        "Adaptability (Kemampuan Adaptasi)"
    ]
    skills = []
    for i, label in enumerate(skill_labels, start=1):
        skill_input = st.slider(label, 1, 10, 5, key=f"skill_{i}")
        skills.append(skill_input / 10)  

st.divider()

risk_descriptions = {
    "Low": (
        "🟢 **Risiko Rendah** — Pekerjaan ini kemungkinan kecil tergantikan oleh AI "
        "dalam waktu dekat. Biasanya membutuhkan keterampilan manusia yang sulit "
        "diotomatisasi (kreativitas, interaksi sosial, pengambilan keputusan kompleks)."
    ),
    "Medium": (
        "🟡 **Risiko Sedang** — Sebagian tugas dalam pekerjaan ini berpotensi "
        "diotomatisasi, namun peran manusia masih dibutuhkan untuk tugas-tugas "
        "tertentu. Disarankan untuk terus mengembangkan skill yang relevan dengan AI."
    ),
    "High": (
        "🔴 **Risiko Tinggi** — Pekerjaan ini memiliki kemungkinan besar untuk "
        "sebagian besar tugasnya diotomatisasi oleh AI di tahun 2030. "
        "Disarankan mempertimbangkan upskilling/reskilling ke bidang dengan "
        "risiko lebih rendah."
    ),
}

if st.button("Prediksi Risiko", type="primary"):
    edu_order = {'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3}
    edu_enc = edu_order[education]

    input_dict = {
        'Average_Salary': salary,
        'Years_Experience': experience,
        'AI_Exposure_Index': ai_exposure,
        'Tech_Growth_Factor': tech_growth,
        'Automation_Probability_2030': automation_prob,
    }
    for i, s in enumerate(skills, start=1):
        input_dict[f'Skill_{i}'] = s
    input_dict['Education_Level_Enc'] = edu_enc

    input_df = pd.DataFrame([input_dict])[feature_names]

    input_scaled = scaler.transform(input_df)

    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    label = label_encoder.inverse_transform([pred])[0]

    color_map = {"Low": "green", "Medium": "orange", "High": "red"}
    st.markdown(f"### Hasil Prediksi: :{color_map[label]}[**Risiko {label}**]")
    st.markdown(risk_descriptions[label])

    st.write("**Tingkat Keyakinan AI untuk Setiap Kategori Risiko (%):**")
    
    proba_df = pd.DataFrame({
        'Kategori Risiko': label_encoder.classes_,
        'Tingkat Keyakinan (%)': proba * 100  
    })

    bars = alt.Chart(proba_df).mark_bar().encode(
        x=alt.X('Kategori Risiko:N', sort=['Low', 'Medium', 'High']),
        y=alt.Y('Tingkat Keyakinan (%)', scale=alt.Scale(domain=[0, 100])),
        color=alt.Color(
            'Kategori Risiko:N',
            scale=alt.Scale(
                domain=['Low', 'Medium', 'High'],
                range=['#2ecc71', '#f39c12', '#e74c3c']
            ),
            legend=None
        )
    )

    text = bars.mark_text(
        align='center',
        baseline='bottom',
        dy=-5,  
        fontSize=14, 
        fontWeight='bold' 
    ).encode(        
        text=alt.Text('Tingkat Keyakinan (%)', format='.1f')
    )
    
    chart = (bars + text).properties(
        height=400
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    st.caption(
        "Cara Membaca Grafik:\n"
        "* **Sumbu X (Mendatar):** Menunjukkan urutan Kategori Risiko (Low -> Medium -> High).\n"
        "* **Sumbu Y (Tegak):** Menunjukkan seberapa yakin model AI dalam bentuk persentase (skala pas 0% sampai 100%).\n"
        "* **Angka di atas batang:** Nilai persentase pasti tingkat keyakinan AI untuk masing-masing kategori."
    )

st.divider()
st.caption(
    "Model: Logistic Regression — dilatih pada dataset 'AI Impact on Jobs 2030'. "
    "Catatan: dataset bersifat sintetis, hasil prediksi hanya untuk tujuan pembelajaran."
)