import streamlit as st
import joblib
import numpy as np
from rag import PatientState, build_graph, ask_question_with_rag
from pdf_export import generate_pdf

st.set_page_config(page_title="Heart Risk Predictor", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.stApp { background-color: #0e1117; color: #fafafa; }
h1 { color: #ffffff; text-align: center; margin-bottom: 30px; }
h3 { color: #e0e0e0; border-bottom: 1px solid #333; padding-bottom: 5px; }
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > div {
    background-color: #262730; color: #ffffff; border: 1px solid #4a4a4a;
}
.risk-high { color: #ff4b4b; font-weight: 800; font-size: 32px; }
.risk-low { color: #00cc96; font-weight: 800; font-size: 32px; }
.card {
    background-color: #262730; padding: 30px; border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 20px; border: 1px solid #333;
}
.factor-item {
    font-size: 16px; margin-bottom: 8px; padding-left: 10px;
    border-left: 3px solid #ff4b4b; color: #e0e0e0;
}
.ai-section-title { color: #a9a4ff; font-size: 18px; font-weight: 600; margin: 20px 0 10px 0; }
.source-item {
    color: #7eb8f7; font-size: 13px; margin-bottom: 6px;
    padding-left: 10px; border-left: 3px solid #378ADD;
}
.disclaimer {
    font-size: 13px; color: #b0b0b0; margin-top: 40px; padding: 20px;
    background-color: #1e1e1e; border-left: 5px solid #ffaec9;
    border-radius: 8px; line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "input"

if "input_data" not in st.session_state:
    st.session_state.input_data = {
        "age": 58, "sex": "Male", "trestbps": 140, "chol": 240,
        "fbs": "False", "cp": 0, "thalach": 150, "exang": "No",
        "restecg": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2,
    }

def go_to_input():
    st.session_state.page = "input"

def go_to_result():
    st.session_state.page = "result"

def go_to_result_from_report():
    st.session_state.page = "result"
    st.session_state.pop("ai_report", None)

try:
    model = joblib.load("models/logistic_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def get_risk_factors(data: dict) -> list:
    factors = []
    if not data:
        return factors
    if data.get("chol", 0) > 200:
        factors.append("High Cholesterol (> 200 mg/dl)")
    if data.get("trestbps", 0) > 130:
        factors.append("Elevated Blood Pressure (> 130 mm Hg)")
    if data.get("fbs") == 1:
        factors.append("High Fasting Blood Sugar (> 120 mg/dl)")
    if data.get("age", 0) > 55:
        factors.append("Age above 55")
    if data.get("thalach", 0) > 180:
        factors.append("High Max Heart Rate (> 180)")
    if data.get("cp", 0) != 0:
        factors.append("Chest Pain Reported")
    if data.get("exang") == 1:
        factors.append("Exercise Induced Angina")
    if data.get("oldpeak", 0) > 2.0:
        factors.append("Significant ST Depression (> 2.0)")
    if data.get("ca", 0) > 0:
        factors.append(f"{data['ca']} Major Vessels Detected")
    return factors

if st.session_state.page == "input":
    st.title("Heart Risk Predictor")
    st.write("Enter patient vitals and clinical parameters below to assess cardiovascular risk.")

    with st.form("risk_form"):
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("### Demographics")
            age = st.number_input("Age", 1, 120, st.session_state.input_data["age"])
            sex = st.selectbox("Sex", ["Female", "Male"],
                               index=["Female", "Male"].index(st.session_state.input_data["sex"]))
            st.markdown("### Vitals")
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 50, 250,
                                       st.session_state.input_data["trestbps"])
            chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600,
                                   st.session_state.input_data["chol"])
            fbs_disp = st.radio("Fasting Blood Sugar > 120 mg/dl", ["False", "True"],
                                index=["False", "True"].index(st.session_state.input_data["fbs"]),
                                horizontal=True)

        with col2:
            st.markdown("### Clinical Data")
            cp = st.selectbox("Chest Pain Type", [0,1,2,3],
                format_func=lambda x: ["Typical Angina (0)", "Atypical Angina (1)",
                                       "Non-anginal Pain (2)", "Asymptomatic (3)"][x],
                index=st.session_state.input_data["cp"])
            thalach = st.number_input("Max Heart Rate Achieved", 60, 220,
                                      st.session_state.input_data["thalach"])
            exang_disp = st.radio("Exercise Induced Angina", ["No", "Yes"],
                                  index=["No", "Yes"].index(st.session_state.input_data["exang"]),
                                  horizontal=True)
            restecg = st.selectbox("Resting ECG", [0,1,2],
                format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x],
                index=st.session_state.input_data["restecg"])

            with st.expander("Advanced Clinical Metrics"):
                oldpeak = st.number_input("ST Depression", 0.0, 10.0,
                                          st.session_state.input_data["oldpeak"])
                slope = st.selectbox("ST Slope", [0,1,2],
                    format_func=lambda x: ["Downsloping", "Flat", "Upsloping"][x],
                    index=st.session_state.input_data["slope"])
                ca = st.selectbox("Major Vessels (0-3)", [0,1,2,3],
                                  index=st.session_state.input_data["ca"])
                thal = st.selectbox("Thalassemia", [1,2,3],
                    format_func=lambda x: {1:"Fixed Defect",2:"Normal",3:"Reversable Defect"}[x],
                    index=[1,2,3].index(st.session_state.input_data["thal"]))

        submitted = st.form_submit_button("PREDICT RISK", use_container_width=True)

    if submitted:
        sex_val = 1 if sex == "Male" else 0
        fbs_val = 1 if fbs_disp == "True" else 0
        exang_val = 1 if exang_disp == "Yes" else 0

        st.session_state.input_data = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs_disp, "restecg": restecg, "thalach": thalach,
            "exang": exang_disp, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
        }

        arr = np.array([[age, sex_val, cp, trestbps, chol, fbs_val,
                         restecg, thalach, exang_val, oldpeak, slope, ca, thal]])
        scaled = scaler.transform(arr)
        pred = model.predict(scaled)
        proba = model.predict_proba(scaled)

        st.session_state.prediction = pred[0]
        st.session_state.probability = proba[0]
        go_to_result()
        st.rerun()

elif st.session_state.page == "result":
    st.button("← Back to Predictor", on_click=go_to_input)

    pred = st.session_state.prediction
    prob = st.session_state.probability
    risk_prob = prob[0]
    risk_level = "HIGH" if pred == 0 else "LOW"
    color_class = "risk-high" if pred == 0 else "risk-low"
    bar_color = "#e74c3c" if pred == 0 else "#27ae60"

    st.markdown(f"""
    <div class="card">
    <h3>Risk Summary</h3>
    <div style="display:flex; justify-content:space-between;">
        <span>Heart Disease Risk:</span>
        <span class="{color_class}">{risk_level} ({risk_prob*100:.1f}%)</span>
    </div>
    <div style="height:10px; background:#eee;">
        <div style="width:{risk_prob*100}%; background:{bar_color}; height:100%;"></div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    factors = get_risk_factors(st.session_state.input_data)
    for f in factors:
        st.markdown(f"* {f}")

    if st.button("Generate AI Health Report"):
        st.session_state.page = "report"
        st.rerun()
