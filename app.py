import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Heart Risk Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
# Custom CSS
st.markdown("""
<style>
    /* Global Styles - Dark Mode */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }

    /* Headings */
    h1 {
        color: #ffffff;
        text-align: center;
        margin-bottom: 30px;
    }
    
    h3 {
        color: #e0e0e0;
        border-bottom: 1px solid #333;
        padding-bottom: 5px;
    }

    /* Input Field Visibility */
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input, 
    .stSelectbox > div > div > div {
        background-color: #262730;
        color: #ffffff;
        border: 1px solid #4a4a4a;
    }
    
    /* Result Page Styling */
    .risk-high {
        color: #ff4b4b;
        font-weight: 800;
        font-size: 32px;
    }
    
    .risk-low {
        color: #00cc96;
        font-weight: 800;
        font-size: 32px;
    }

    .card {
        background-color: #262730;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border: 1px solid #333;
    }

    .factor-item {
        font-size: 16px;
        margin-bottom: 8px;
        padding-left: 10px;
        border-left: 3px solid #ff4b4b;
        color: #e0e0e0;
    }

    .disclaimer {
        font-size: 13px;
        color: #b0b0b0;
        margin-top: 40px;
        padding: 20px;
        background-color: #1e1e1e;
        border-left: 5px solid #ffaec9;
        border-radius: 8px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = 'input'

def go_to_result():
    st.session_state.page = 'result'

def go_to_input():
    st.session_state.page = 'input'

# Initialize session state for input data with defaults that result in ~67% risk
if 'input_data' not in st.session_state:
    st.session_state.input_data = {
        'age': 58,
        'sex': 'Male',
        'trestbps': 140,
        'chol': 240,
        'fbs': 'False',
        'cp': 0,
        'thalach': 150,
        'exang': 'No',
        'restecg': 0,
        'oldpeak': 1.0,
        'slope': 1,
        'ca': 0,
        'thal': 2
    }

try:
    model = joblib.load('logistic_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Error loading model resources: {e}")
    st.stop()

def get_risk_factors(data):
    factors = []
    
    if data['chol'] > 200:
        factors.append("High Cholesterol (> 200 mg/dl)")
    if data['trestbps'] > 130:
        factors.append("Elevated Blood Pressure (> 130 mm Hg)")
    if data['fbs'] == 1:
        factors.append("High Fasting Blood Sugar (> 120 mg/dl)")
    if data['age'] > 55:
        factors.append("Age above 55")
    if data['thalach'] > 180:
        factors.append("High Max Heart Rate (> 180)")
    if data['cp'] != 0: 
         factors.append(f"Chest Pain Reported")
    if data['exang'] == 1:
        factors.append("Exercise Induced Angina")
    if data['oldpeak'] > 2.0:
        factors.append("Significant ST Depression (> 2.0)")
    if data['ca'] > 0:
        factors.append(f"{data['ca']} Major Vessels Detected")
        
    return factors

if st.session_state.page == 'input':
    st.title("Heart Risk Predictor")
    st.write("Enter patient vitals and clinical parameters below to assess cardiovascular risk.")
    
    with st.form("risk_form"):
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("### Demographics")
            age = st.number_input("Age", min_value=1, max_value=120, value=st.session_state.input_data['age'])
            sex = st.selectbox("Sex", ["Female", "Male"], index=["Female", "Male"].index(st.session_state.input_data['sex']))
            
            st.markdown("### Vitals")
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 50, 250, st.session_state.input_data['trestbps'], help="Normal is < 120")
            chol = st.number_input("Serum Cholestoral (mg/dl)", 100, 600, st.session_state.input_data['chol'], help="Desirable is < 200")
            fbs_disp = st.radio("Fasting Blood Sugar > 120 mg/dl", ["False", "True"], index=["False", "True"].index(st.session_state.input_data['fbs']), horizontal=True)

        with col2:
            st.markdown("### Clinical Data")
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: [
                "Typical Angina (0)", "Atypical Angina (1)", "Non-anginal Pain (2)", "Asymptomatic (3)"
            ][x], index=st.session_state.input_data['cp'])
            thalach = st.number_input("Max Heart Rate Achieved", 60, 220, st.session_state.input_data['thalach'])
            exang_disp = st.radio("Exercise Induced Angina", ["No", "Yes"], index=["No", "Yes"].index(st.session_state.input_data['exang']), horizontal=True)
            restecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x], index=st.session_state.input_data['restecg'])

            with st.expander("Advanced Clinical Metrics"):
                oldpeak = st.number_input("ST Depression", 0.0, 10.0, st.session_state.input_data['oldpeak'])
                # slope encoding in training data: 0 = Downsloping (worst), 1 = Flat, 2 = Upsloping (best)
                slope = st.selectbox("ST Slope", [0, 1, 2], format_func=lambda x: ["Downsloping", "Flat", "Upsloping"][x], index=st.session_state.input_data['slope'])
                ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3], index=st.session_state.input_data['ca'])
                # thal valid values: 1 = Fixed Defect, 2 = Normal, 3 = Reversable Defect (0 = missing/invalid, never use)
                thal = st.selectbox("Thalassemia", [1, 2, 3], format_func=lambda x: {1: "Fixed Defect", 2: "Normal", 3: "Reversable Defect"}[x], index=[1, 2, 3].index(st.session_state.input_data['thal']))

        st.markdown("<br>", unsafe_allow_html=True)
        # Submit button
        submitted = st.form_submit_button("PREDICT RISK", use_container_width=True)
    
    if submitted:
        # Process inputs
        sex_val = 1 if sex == "Male" else 0
        fbs_val = 1 if fbs_disp == "True" else 0
        exang_val = 1 if exang_disp == "Yes" else 0
        
        st.session_state.input_data = {
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs_disp, 'restecg': restecg, 'thalach': thalach, 'exang': exang_disp,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }
        
        # Prepare for prediction
        input_array = np.array([[
            age, sex_val, cp, trestbps, chol, fbs_val, restecg, thalach, exang_val, oldpeak, slope, ca, thal
        ]])
        
        scaled_features = scaler.transform(input_array)
        prediction = model.predict(scaled_features)
        proba = model.predict_proba(scaled_features)
        
        st.session_state.prediction = prediction[0]
        st.session_state.probability = proba[0]
        
        go_to_result()
        st.rerun()

elif st.session_state.page == 'result':
    st.button("← Back to Predictor", on_click=go_to_input)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Header
    st.markdown("### Risk Summary")
    
    pred = st.session_state.prediction
    prob = st.session_state.probability
    
    # Logic for display
    # NOTE: In this model, class 0 = heart disease (HIGH RISK), class 1 = healthy (LOW RISK)
    # risk_prob always = prob[0] = P(heart disease), so: higher % = higher risk, lower % = lower risk
    risk_prob = prob[0]  # always the probability of class 0 (heart disease)
    if pred == 0:
        risk_level = "HIGH"
        color_class = "risk-high"
    else:
        risk_level = "LOW"
        color_class = "risk-low"

    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
        <span style="font-size: 20px; color: #555;">Heart Disease Risk:</span>
        <span class="{color_class}">{risk_level} ({risk_prob*100:.1f}%)</span>
    </div>
    <div style="height: 10px; width: 100%; background: #eee; border-radius: 5px; overflow: hidden;">
        <div style="height: 100%; width: {risk_prob*100}%; background: {'#e74c3c' if pred == 0 else '#27ae60'};"></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### Key Risk Factors")

    factors = get_risk_factors(st.session_state.input_data)
    
    if factors:
        for factor in factors:
            st.markdown(f'<div class="factor-item">• {factor}</div>', unsafe_allow_html=True)
    else:
        st.info("No specific critical risk factors identified based on standard thresholds.")
        
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Explanation")
    st.write("These factors are strongly associated with increased cardiovascular risk. The model analyzes the complex interaction between your vitals and clinical history to generate this assessment.")
    
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Disclaimer:</strong><br><br>
        This assessment is for informational purposes only and does not replace professional medical advice, diagnosis, or treatment. 
        Calculations are based on statistical models and may not account for individual medical history. 
        Always consult with a qualified healthcare provider for proper evaluation.
    </div>
    """, unsafe_allow_html=True)
