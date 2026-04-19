"""
rag.py
======
All RAG and LLM logic — zero Streamlit UI code.

Responsibilities:
- Load and query the FAISS vector store
- Call the Groq LLM
- LangGraph pipeline (analyze → generate → finalize)
- Chat Q&A with retrieval
"""

import json
import re
import requests
import streamlit as st
from typing import TypedDict
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END


# ─── State schema ─────────────────────────────────────────────────────────────

class PatientState(TypedDict):
    patient_data:   dict   # raw input dict
    risk_score:     float  # 0–100
    risk_level:     str    # "HIGH" or "LOW"
    risk_factors:   list   # e.g. ["High Cholesterol", ...]
    retrieved_docs: list   # top-k chunks: [{"content": str, "source": str, "page": int}]
    report:         dict   # final structured report
    error:          str    # accumulated error messages


# ─── Vector store ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_vectorstore():
    """Load the FAISS index once and cache it for the app lifetime."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        return FAISS.load_local(
            "data/faiss_index",
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception:
        return None


# ─── LLM client ───────────────────────────────────────────────────────────────

def _call_llm(api_key: str, prompt: str, max_tokens: int = 1200) -> str:
    """
    Call Llama-3.3-70B via Groq's free OpenAI-compatible API.
    Free tier: 14,400 requests/day — https://console.groq.com
    """
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        },
        timeout=60,
    )
    if resp.status_code != 200:
        raise Exception(f"Groq API {resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"].strip()


# ─── JSON parsing ──────────────────────────────────────────────────────────────

def parse_llm_json(text: str):
    """Strip code fences and parse the first JSON object found in text."""
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
    text = re.sub(r'```\s*$', '', text.strip(), flags=re.MULTILINE)
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


# ─── Chat Q&A ─────────────────────────────────────────────────────────────────

def ask_question_with_rag(question: str, patient_context: dict) -> str:
    """Answer a follow-up question: retrieve from FAISS, then call LLM."""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        return "GROQ_API_KEY not configured in secrets."

    db = load_vectorstore()
    retrieved = []
    if db is not None:
        docs = db.similarity_search(question, k=5)
        retrieved = [
            {
                "content": doc.page_content,
                "source":  doc.metadata.get("source", "unknown"),
                "page":    doc.metadata.get("page", "?"),
            }
            for doc in docs
        ]

    docs_str = (
        "\n\n".join(
            f"[{d['source']} p.{d['page']}]: {d['content']}"
            for d in retrieved
        )
        if retrieved else "No specific evidence retrieved."
    )
    risk_level  = patient_context.get("risk_level", "unknown")
    risk_score  = patient_context.get("risk_score", 0)
    factors_str = ", ".join(patient_context.get("risk_factors", [])) or "none identified"

    prompt = f"""You are a clinical AI assistant helping a user understand their cardiovascular health.
Always address the user directly using "you" and "your" (never say "the patient").

User profile:
- Risk Level: {risk_level} ({risk_score:.1f}%)
- Key risk factors: {factors_str}

Relevant medical evidence from clinical guidelines:
{docs_str}

User question: {question}

Answer in 3-5 sentences. Be specific and ground your answer in the evidence above. \
Use only plain ASCII characters. Address the user as "you". Do not suggest you are a doctor."""

    try:
        return _call_llm(api_key, prompt, max_tokens=500)
    except Exception as e:
        return f"Could not get an answer: {str(e)}"


# ─── LangGraph nodes ──────────────────────────────────────────────────────────

def node_analyze_risk(state: PatientState) -> PatientState:
    """Retrieve the top-6 most relevant chunks from FAISS for this patient."""
    query = "cardiovascular risk " + " ".join(state["risk_factors"][:3])
    try:
        db = load_vectorstore()
        if db is None:
            state["retrieved_docs"] = []
            state["error"] = (
                state.get("error", "")
                + " [FAISS index not found -- run: python data/build_vectordb.py]"
            )
        else:
            results = db.similarity_search(query, k=6)
            state["retrieved_docs"] = [
                {
                    "content": doc.page_content,
                    "source":  doc.metadata.get("source", "unknown"),
                    "page":    doc.metadata.get("page", "?"),
                }
                for doc in results
            ]
    except Exception as e:
        state["retrieved_docs"] = []
        state["error"] = state.get("error", "") + f" [FAISS retrieval error: {str(e)}]"
    return state


def _build_dynamic_report_from_docs(
    risk_level: str,
    risk_score: float,
    risk_factors: list,
    retrieved_docs: list,
    api_key: str,
) -> dict:
    """
    Fallback: plain-text LLM call when JSON parsing fails.
    Still uses retrieved evidence — never returns hardcoded strings.
    """
    factors_str = ", ".join(risk_factors) if risk_factors else "none"
    docs_str = "\n\n".join(
        f"[{d['source']} p.{d['page']}]: {d['content'][:400]}"
        for d in retrieved_docs[:4]
    ) if retrieved_docs else ""
    fallback_sources = list(dict.fromkeys(
        f"{d['source'].replace('.pdf','').replace('_',' ').title()} p.{d['page']}"
        for d in retrieved_docs[:4]
    )) if retrieved_docs else ["AHA Prevention Guidelines", "WHO CVD Risk Guide", "NCEP ATP III"]

    prompt = f"""You are a clinical AI assistant. The user has {risk_level} cardiovascular risk ({risk_score:.1f}%).
Key risk factors: {factors_str}.

Using ONLY the clinical evidence below, write:
1. SUMMARY: A 2-sentence plain-English risk summary addressing the user directly (use "you/your").
2. RECOMMENDATIONS: 4 specific recommendations (format: "Title: detail sentence"). Address the user as "you".
3. QUESTIONS: 3 follow-up questions the user should ask their doctor.

Clinical evidence:
{docs_str}

Use only plain ASCII characters. Always use "you" and "your" instead of "the patient"."""

    raw = _call_llm(api_key, prompt, max_tokens=800)
    lines = [l.strip() for l in raw.split('\n') if l.strip()]

    summary = next(
        (l for l in lines if not l.startswith(('1.', '2.', '3.', 'SUMMARY', 'RECOMMENDATIONS', 'QUESTIONS'))),
        f"This patient presents with {risk_level} cardiovascular risk ({risk_score:.1f}%).",
    )

    recs = []
    for line in lines:
        if ':' in line and len(line) < 200:
            parts = line.split(':', 1)
            title  = parts[0].lstrip('0123456789.-) ').strip()
            detail = parts[1].strip()
            if title and detail and len(title) < 80:
                recs.append({"title": title, "detail": detail})
    if not recs:
        recs = [{"title": l[:60], "detail": l} for l in lines if len(l) > 30][:4]

    questions = [l.lstrip('0123456789.-) ').strip() for l in lines if '?' in l][:3]
    if not questions:
        questions = [
            f"What does my {risk_factors[0].lower()} mean for my long-term risk?" if risk_factors
            else "What lifestyle changes would most reduce my risk?",
            "Which medications should I discuss with my doctor?",
            "How often should I have a cardiovascular check-up?",
        ]

    return {
        "risk_summary":        summary,
        "contributing_factors": risk_factors[:4] or ["See clinical assessment"],
        "recommendations":      recs[:4] or [{"title": "Consult your physician", "detail": raw[:300]}],
        "sources":              fallback_sources,
        "follow_up_questions":  questions,
    }


def node_generate_report(state: PatientState) -> PatientState:
    """Generate the full structured report from retrieved evidence via LLM."""
    risk_level     = state["risk_level"]
    risk_score     = state["risk_score"]
    risk_factors   = state["risk_factors"]
    patient_data   = state["patient_data"]
    retrieved_docs = state["retrieved_docs"]

    age      = patient_data.get("age", "N/A")
    sex      = patient_data.get("sex", "N/A")
    chol     = patient_data.get("chol", "N/A")
    trestbps = patient_data.get("trestbps", "N/A")
    thalach  = patient_data.get("thalach", "N/A")
    oldpeak  = patient_data.get("oldpeak", "N/A")
    cp       = patient_data.get("cp", "N/A")
    exang    = patient_data.get("exang", "N/A")

    factors_str = ", ".join(risk_factors) if risk_factors else "None identified"
    docs_str = (
        "\n\n".join(
            f"[Evidence {i+1} | {d['source']} p.{d['page']}]:\n{d['content']}"
            for i, d in enumerate(retrieved_docs[:6])
        ) if retrieved_docs else "No specific evidence retrieved."
    )
    source_refs = list(dict.fromkeys(
        f"{d['source'].replace('.pdf','').replace('_',' ').title()} p.{d['page']}"
        for d in retrieved_docs[:6]
    )) if retrieved_docs else ["AHA Prevention Guidelines", "NCEP ATP III", "WHO CVD Guide"]

    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        state["error"] = "GROQ_API_KEY not found in secrets."
        state["report"] = {}
        return state

    prompt = f"""You are a clinical AI assistant generating a personalized cardiovascular health report.
Respond with ONLY a valid JSON object. No markdown, no code fences, no explanation.
Use only plain ASCII characters. Do not use em dashes, en dashes, or smart quotes.
IMPORTANT: Address the user directly using "you" and "your" (NOT "the patient" or "this patient").

User profile:
- Age: {age}, Sex: {sex}
- Risk Level: {risk_level}, Risk Score: {risk_score:.1f}%
- Cholesterol: {chol} mg/dl, Blood Pressure: {trestbps} mmHg
- Max Heart Rate: {thalach} bpm, ST Depression: {oldpeak}
- Chest Pain Type: {cp}, Exercise Angina: {exang}
- Risk factors: {factors_str}

Retrieved clinical evidence (base ALL recommendations on this):
{docs_str}

Return ONLY this JSON (do NOT include a "sources" field):
{{
  "risk_summary": "2-3 sentence plain-English summary addressing the user directly with 'you/your', grounded in the evidence",
  "contributing_factors": ["factor 1", "factor 2", "factor 3"],
  "recommendations": [
    {{"title": "action title", "detail": "1-2 sentences using 'you/your', citing values from the evidence"}},
    {{"title": "action title", "detail": "1-2 sentences using 'you/your', citing values from the evidence"}},
    {{"title": "action title", "detail": "1-2 sentences using 'you/your', citing values from the evidence"}},
    {{"title": "action title", "detail": "1-2 sentences using 'you/your', citing values from the evidence"}}
  ],
  "follow_up_questions": ["question using 'I/my' e.g. 'What does my cholesterol level mean?'", "question 2?", "question 3?"]
}}

CRITICAL RULE: NEVER use the phrases "the patient", "this patient", or "a patient" anywhere in the output. Always say "you" or "your" instead. Follow-up questions must be phrased in first person (I/my) since the user will ask them."""

    # Primary attempt: structured JSON
    try:
        raw = _call_llm(api_key, prompt, max_tokens=1400)
        parsed = parse_llm_json(raw)
        if parsed is not None and "recommendations" in parsed:
            parsed["sources"] = source_refs
            state["report"] = parsed
            return state
        state["error"] = state.get("error", "") + " [primary JSON malformed, retrying]"
    except Exception as e:
        state["error"] = state.get("error", "") + f" [primary LLM error: {str(e)}]"

    # Dynamic fallback: plain-text LLM call
    try:
        state["report"] = _build_dynamic_report_from_docs(
            risk_level, risk_score, risk_factors, retrieved_docs, api_key
        )
    except Exception as e2:
        state["error"] = state.get("error", "") + f" [fallback LLM error: {str(e2)}]"
        state["report"] = {}

    return state


def node_finalize(state: PatientState) -> PatientState:
    return state


# ─── LangGraph pipeline ────────────────────────────────────────────────────────

@st.cache_resource
def build_graph():
    """Compile the LangGraph pipeline once and cache it."""
    graph = StateGraph(PatientState)
    graph.add_node("analyze",  node_analyze_risk)
    graph.add_node("generate", node_generate_report)
    graph.add_node("finalize", node_finalize)
    graph.set_entry_point("analyze")
    graph.add_edge("analyze",  "generate")
    graph.add_edge("generate", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()
