"""
app.py
======
Streamlit entry point — UI only.

Run with:
    .venv/bin/streamlit run app.py

All RAG / LLM logic lives in rag.py.
PDF generation lives in pdf_export.py.
"""

import streamlit as st
import joblib
import numpy as np

from rag import PatientState, build_graph, ask_question_with_rag
from pdf_export import generate_pdf

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Heart Risk Predictor",
    layout="centered",
    initial_sidebar_state="collapsed",
)

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
    .risk-high  { color: #ff4b4b; font-weight: 800; font-size: 32px; }
    .risk-low   { color: #00cc96; font-weight: 800; font-size: 32px; }
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


# ─── Session state defaults ───────────────────────────────────────────────────

if "page" not in st.session_state:
    st.session_state.page = "input"

if "input_data" not in st.session_state:
    st.session_state.input_data = {
        "age": 58, "sex": "Male", "trestbps": 140, "chol": 240,
        "fbs": "False", "cp": 0, "thalach": 150, "exang": "No",
        "restecg": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2,
    }


# ─── Navigation helpers ───────────────────────────────────────────────────────

def go_to_input():
    st.session_state.page = "input"

def go_to_result():
    st.session_state.page = "result"

def go_to_result_from_report():
    st.session_state.page = "result"
    st.session_state.pop("ai_report", None)


# ─── ML model loading ─────────────────────────────────────────────────────────

try:
    model  = joblib.load("models/logistic_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# ─── Risk factor helper ───────────────────────────────────────────────────────

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


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: INPUT
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.page == "input":
    st.title("Heart Risk Predictor")
    st.write("Enter patient vitals and clinical parameters below to assess cardiovascular risk.")

    with st.form("risk_form"):
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("### Demographics")
            age = st.number_input("Age", min_value=1, max_value=120,
                                  value=st.session_state.input_data["age"])
            sex = st.selectbox("Sex", ["Female", "Male"],
                               index=["Female", "Male"].index(st.session_state.input_data["sex"]))
            st.markdown("### Vitals")
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 50, 250,
                                       st.session_state.input_data["trestbps"],
                                       help="Normal is < 120")
            chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600,
                                   st.session_state.input_data["chol"],
                                   help="Desirable is < 200")
            fbs_disp = st.radio("Fasting Blood Sugar > 120 mg/dl", ["False", "True"],
                                index=["False", "True"].index(st.session_state.input_data["fbs"]),
                                horizontal=True)

        with col2:
            st.markdown("### Clinical Data")
            cp = st.selectbox(
                "Chest Pain Type", [0, 1, 2, 3],
                format_func=lambda x: ["Typical Angina (0)", "Atypical Angina (1)",
                                       "Non-anginal Pain (2)", "Asymptomatic (3)"][x],
                index=st.session_state.input_data["cp"],
            )
            thalach = st.number_input("Max Heart Rate Achieved", 60, 220,
                                      st.session_state.input_data["thalach"])
            exang_disp = st.radio("Exercise Induced Angina", ["No", "Yes"],
                                  index=["No", "Yes"].index(st.session_state.input_data["exang"]),
                                  horizontal=True)
            restecg = st.selectbox(
                "Resting ECG", [0, 1, 2],
                format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x],
                index=st.session_state.input_data["restecg"],
            )
            with st.expander("Advanced Clinical Metrics"):
                oldpeak = st.number_input("ST Depression", 0.0, 10.0,
                                          st.session_state.input_data["oldpeak"])
                slope = st.selectbox(
                    "ST Slope", [0, 1, 2],
                    format_func=lambda x: ["Downsloping", "Flat", "Upsloping"][x],
                    index=st.session_state.input_data["slope"],
                )
                ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3],
                                  index=st.session_state.input_data["ca"])
                thal = st.selectbox(
                    "Thalassemia", [1, 2, 3],
                    format_func=lambda x: {1: "Fixed Defect", 2: "Normal", 3: "Reversable Defect"}[x],
                    index=[1, 2, 3].index(st.session_state.input_data["thal"]),
                )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("PREDICT RISK", use_container_width=True)

    if submitted:
        sex_val   = 1 if sex == "Male" else 0
        fbs_val   = 1 if fbs_disp == "True" else 0
        exang_val = 1 if exang_disp == "Yes" else 0

        st.session_state.input_data = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs_disp, "restecg": restecg, "thalach": thalach, "exang": exang_disp,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
        }

        input_array    = np.array([[age, sex_val, cp, trestbps, chol, fbs_val,
                                    restecg, thalach, exang_val, oldpeak, slope, ca, thal]])
        scaled         = scaler.transform(input_array)
        prediction     = model.predict(scaled)
        proba          = model.predict_proba(scaled)

        st.session_state.prediction = prediction[0]
        st.session_state.probability = proba[0]
        go_to_result()
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: RESULT
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "result":
    st.button("← Back to Predictor", on_click=go_to_input)

    pred      = st.session_state.prediction
    prob      = st.session_state.probability
    risk_prob = prob[0]
    risk_level  = "HIGH" if pred == 0 else "LOW"
    color_class = "risk-high" if pred == 0 else "risk-low"
    bar_color   = "#e74c3c" if pred == 0 else "#27ae60"

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Risk Summary")
    st.markdown(f"""
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:20px;">
        <span style="font-size:20px; color:#555;">Heart Disease Risk:</span>
        <span class="{color_class}">{risk_level} ({risk_prob*100:.1f}%)</span>
    </div>
    <div style="height:10px; width:100%; background:#eee; border-radius:5px; overflow:hidden;">
        <div style="height:100%; width:{risk_prob*100}%; background:{bar_color};"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Key Risk Factors")
    factors = get_risk_factors(st.session_state.input_data)
    if factors:
        for factor in factors:
            st.markdown(f'<div class="factor-item">* {factor}</div>', unsafe_allow_html=True)
    else:
        st.info("No specific critical risk factors identified based on standard thresholds.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Explanation")
    st.write(
        "These factors are strongly associated with increased cardiovascular risk. "
        "The model analyzes the complex interaction between your vitals and clinical "
        "history to generate this assessment."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        <strong>Disclaimer:</strong><br><br>
        This assessment is for informational purposes only and does not replace professional
        medical advice, diagnosis, or treatment. Calculations are based on statistical models
        and may not account for individual medical history. Always consult a qualified
        healthcare provider for proper evaluation.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Generate AI Health Report", use_container_width=True, key="gen_report_btn"):
        st.session_state.page = "report"
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: REPORT
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "report":

    # Run the LangGraph agent once per prediction
    if "ai_report" not in st.session_state:
        pred      = st.session_state.prediction
        prob      = st.session_state.probability
        risk_prob = prob[0]
        risk_level = "HIGH" if pred == 0 else "LOW"
        factors    = get_risk_factors(st.session_state.input_data)

        initial_state = PatientState(
            patient_data=st.session_state.input_data,
            risk_score=round(risk_prob * 100, 1),
            risk_level=risk_level,
            risk_factors=factors,
            retrieved_docs=[],
            report={},
            error="",
        )

        with st.spinner("Running AI agent... Analyzing risk -> Retrieving evidence -> Generating report"):
            try:
                final_state = build_graph().invoke(initial_state)
                st.session_state.ai_report     = final_state["report"]
                st.session_state.ai_error      = final_state.get("error", "")
                st.session_state.ai_risk_level = risk_level
                st.session_state.ai_risk_score = round(risk_prob * 100, 1)
            except Exception as e:
                st.session_state.ai_report = None
                st.session_state.ai_error  = str(e)

    # Navigation
    col_back, col_regen = st.columns([1, 3])
    with col_back:
        st.button("← Back to Results", on_click=go_to_result_from_report)
    with col_regen:
        if st.button("Regenerate Report", key="regen"):
            st.session_state.pop("ai_report", None)
            st.rerun()

    # Error state
    report = st.session_state.ai_report
    if not report:
        st.error(f"Could not generate report. Error: {st.session_state.ai_error}")
        st.info("Check that your GROQ_API_KEY in .streamlit/secrets.toml is set. "
                "Get a free key at console.groq.com")
        st.stop()

    risk_level = st.session_state.get("ai_risk_level", "N/A")
    risk_score = st.session_state.get("ai_risk_score", 0.0)

    st.markdown("<h1 style='text-align:center; color:#a9a4ff;'>AI Health Report</h1>",
                unsafe_allow_html=True)

    # Progress badges
    st.markdown("""
    <div style="display:flex; gap:8px; margin-bottom:1.5rem; flex-wrap:wrap;">
      <span style="background:#1a1a2e; color:#6c63ff; border:1px solid #6c63ff;
                   padding:4px 14px; border-radius:20px; font-size:12px;">ML prediction</span>
      <span style="background:#1a1a2e; color:#6c63ff; border:1px solid #6c63ff;
                   padding:4px 14px; border-radius:20px; font-size:12px;">Risk analysis</span>
      <span style="background:#1a1a2e; color:#6c63ff; border:1px solid #6c63ff;
                   padding:4px 14px; border-radius:20px; font-size:12px;">Evidence retrieved</span>
      <span style="background:#6c63ff; color:#fff;
                   padding:4px 14px; border-radius:20px; font-size:12px;">Report ready</span>
    </div>
    """, unsafe_allow_html=True)

    # Patient summary
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="ai-section-title">Your Summary</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Age",        st.session_state.input_data.get("age", "N/A"))
    m2.metric("Sex",        st.session_state.input_data.get("sex", "N/A"))
    m3.metric("Risk Level", risk_level)
    m4.metric("Risk Score", f"{risk_score:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

    # Risk summary
    risk_heading_color = "#ff4b4b" if risk_level == "HIGH" else "#00cc96"
    icon = "!!" if risk_level == "HIGH" else "OK"
    st.markdown(f"""
    <div class="card">
      <div style="color:{risk_heading_color}; font-size:18px; font-weight:600; margin-bottom:10px;">
        [{icon}] Your Risk Summary - {risk_level} RISK
      </div>
      <p style="color:#d0d0d0; line-height:1.7; font-size:15px;">
        {report.get("risk_summary", "No summary available.")}
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Contributing factors
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="ai-section-title">Your Contributing Factors</div>', unsafe_allow_html=True)
    for factor in report.get("contributing_factors", []):
        st.markdown(f'<div class="factor-item">* {factor}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Recommendations
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="ai-section-title">Personalized Recommendations for You</div>', unsafe_allow_html=True)
    for rec in report.get("recommendations", []):
        st.markdown(f"""
        <div style="background:#1e1e2e; border-left:3px solid #6c63ff; border-radius:8px;
                    padding:14px 18px; margin-bottom:10px;">
          <div style="color:#a9a4ff; font-weight:600; font-size:14px; margin-bottom:4px;">
            {rec.get("title", "")}
          </div>
          <div style="color:#c0c0c0; font-size:13px; line-height:1.6;">
            {rec.get("detail", "")}
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Sources
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="ai-section-title">Sources</div>', unsafe_allow_html=True)
    for src in report.get("sources", []):
        st.markdown(f'<div class="source-item">* {src}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>Disclaimer:</strong><br><br>
        This AI-generated report is for educational and informational purposes only.
        It does not constitute medical advice, clinical diagnosis, or treatment recommendations.
        All outputs are generated by a machine learning model and a large language model
        and may contain errors. Always consult a qualified healthcare professional
        before making any health-related decisions.
    </div>
    """, unsafe_allow_html=True)

    # Follow-up questions (click -> go to chat with that question)
    follow_up = report.get("follow_up_questions", [])
    if follow_up:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="ai-section-title">Questions You Can Ask Your Doctor</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#888; font-size:13px; margin-bottom:12px;'>"
            "Click any question to get an AI answer grounded in the medical guidelines:</p>",
            unsafe_allow_html=True,
        )
        for i, question in enumerate(follow_up):
            if st.button(question, key=f"fq_{i}"):
                st.session_state.setdefault("chat_history", [])
                st.session_state.chat_prefill = question
                st.session_state.page = "chat"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Chat and PDF buttons
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Chat with AI About Your Results", use_container_width=True, key="go_chat"):
        st.session_state.page = "chat"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    try:
        pdf_bytes = generate_pdf(
            report=report,
            patient_data=st.session_state.input_data,
            risk_level=risk_level,
            risk_score=risk_score,
        )
        st.download_button(
            label="📄 Export as PDF",
            data=pdf_bytes,
            file_name="heart_risk_ai_report.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="download_pdf",
        )
    except Exception as e:
        st.error(f"PDF generation failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CHAT
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "chat":

    col_back, _ = st.columns([1, 4])
    with col_back:
        if st.button("← Back to Report"):
            st.session_state.page = "report"
            st.rerun()

    st.markdown("<h2 style='text-align:center; color:#a9a4ff;'>AI Health Chat</h2>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#888; font-size:14px;'>"
        "Ask anything about your results. Answers are grounded in AHA, WHO, and NCEP guidelines.</p>",
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("chat_history", [])

    patient_ctx = {
        "risk_level":  st.session_state.get("ai_risk_level", "unknown"),
        "risk_score":  st.session_state.get("ai_risk_score", 0.0),
        "risk_factors": get_risk_factors(st.session_state.get("input_data", {})),
    }

    # Auto-send a prefilled question (from follow-up button)
    if st.session_state.get("chat_prefill"):
        q = st.session_state.chat_prefill
        st.session_state.chat_prefill = None
        st.session_state.chat_history.append({"role": "user", "content": q})
        with st.spinner("Searching guidelines..."):
            answer = ask_question_with_rag(q, patient_ctx)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Render conversation
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div style="display:flex; justify-content:flex-end; margin:8px 0;">
              <div style="background:#2a2a4e; border-radius:12px 12px 2px 12px;
                          padding:12px 16px; max-width:80%; color:#e0e0e0; font-size:14px;">
                {msg["content"]}
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display:flex; justify-content:flex-start; margin:8px 0;">
              <div style="background:#1e1e2e; border:1px solid #333;
                          border-radius:12px 12px 12px 2px; padding:12px 16px;
                          max-width:85%; color:#d0d0d0; font-size:14px; line-height:1.6;">
                <span style="color:#a9a4ff; font-weight:600; font-size:12px;">AI</span><br>
                {msg["content"]}
              </div>
            </div>""", unsafe_allow_html=True)

    # Input form
    st.markdown("<br>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_send = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                "Message",
                placeholder="e.g. What foods should I avoid with high cholesterol?",
                label_visibility="collapsed",
            )
        with col_send:
            send = st.form_submit_button("Send", use_container_width=True)

    if send and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        with st.spinner("Searching guidelines..."):
            answer = ask_question_with_rag(user_input.strip(), patient_ctx)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("""
    <div class="disclaimer" style="margin-top:30px;">
        <strong>Disclaimer:</strong> AI responses are grounded in clinical guidelines
        (AHA, WHO, NCEP) but are not a substitute for professional medical advice.
        Always consult a qualified healthcare provider.
    </div>
    """, unsafe_allow_html=True)
