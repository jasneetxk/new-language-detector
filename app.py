import streamlit as st
import joblib
import os
from pathlib import Path
from typing import List
import io
from PyPDF2 import PdfReader
import docx
import pandas as pd
from dotenv import load_dotenv



MODEL_PATH = "lang_detector_pipeline_v2_papluca_all20.joblib"

LANG_MAP = {
    "en": "English","fr": "French","es": "Spanish","pt": "Portuguese","bg": "Bulgarian",
    "zh": "Chinese","th": "Thai","ru": "Russian","pl": "Polish","ur": "Urdu",
    "sw": "Swahili","tr": "Turkish","ar": "Arabic","it": "Italian","hi": "Hindi",
    "de": "German","el": "Greek","nl": "Dutch","vi": "Vietnamese","ja": "Japanese",
}

EXAMPLES_BY_FULLNAME = {
    "English": "This is an example sentence in English.",
    "French": "Bonjour, je m'appelle Paul et j'aime apprendre.",
    "Spanish": "Hola amigo, espero que tengas un buen día.",
    "Hindi": "नमस्ते! यह एक उदाहरण वाक्य है।",
    "Chinese": "你好！这是一个中文示例句子。",
    "German": "Hallo! Dies ist ein Beispielsatz auf Deutsch.",
}

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="Language Detector", layout="centered")

# ---------------- MODEL HELPERS ----------------
@st.cache_resource(show_spinner=False)
def load_detection_model(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return joblib.load(path)

def extract_labels(model) -> List[str]:
    if hasattr(model, "classes_"):
        return list(model.classes_)
    elif hasattr(model, "steps"):
        final = model.steps[-1][1]
        return list(final.classes_) if hasattr(final, "classes_") else []
    return []

def token_to_full(token: str) -> str:
    return LANG_MAP.get(token, token)

def predict_language(model, text: str):
    """Return predicted token and confidence if available."""
    if not text.strip():
        return None, None
    try:
        if hasattr(model, "predict_proba"):
            pred = model.predict([text])[0]
            probs = model.predict_proba([text])[0]
            confidence = max(probs)
        else:
            pred = model.predict([text])[0]
            confidence = None
        return pred, confidence
    except Exception:
        return None, None

# ---------------- FILE HELPERS ----------------
def extract_text_from_pdf_bytes(b: bytes) -> List[str]:
    try:
        reader = PdfReader(io.BytesIO(b))
        return [p.extract_text() or "" for p in reader.pages]
    except Exception:
        return []

def extract_text_from_docx_bytes(b: bytes) -> str:
    doc = docx.Document(io.BytesIO(b))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Model Info")
st.sidebar.write(f"Model: `{os.path.basename(MODEL_PATH)}`")
st.sidebar.markdown("---")
st.sidebar.subheader("About Model")
st.sidebar.write("TF-IDF (char n-grams) + Logistic Regression model trained to classify 20 languages.")
st.sidebar.markdown("---")
st.sidebar.subheader("Languages Supported")
st.sidebar.write(" > ".join(LANG_MAP.values()))

# ---------------- LOAD MODEL ----------------
with st.spinner("Loading detection model..."):
    model = load_detection_model(MODEL_PATH)

if not model:
    st.error(f"Model not found at `{MODEL_PATH}`. Please place it in the working directory.")
    st.stop()

labels = extract_labels(model)
st.sidebar.caption(f"Total languages: {len(labels)}")

# ---------------- MAIN ----------------
st.title("🌍 Language Detector")
st.caption("Detects the language of text or documents using a trained logistic regression model.")

st.session_state.setdefault("text_input", "")
st.session_state.setdefault("example_select", "— none —")

def populate_example_on_change():
    sel = st.session_state["example_select"]
    if sel and sel != "— none —":
        st.session_state["text_input"] = EXAMPLES_BY_FULLNAME.get(sel, "")

def clear_all():
    st.session_state["text_input"] = ""
    st.session_state["example_select"] = "— none —"

col_main, col_side = st.columns([3, 1])

with col_main:
    input_text = st.text_area("Enter text to classify", key="text_input", height=220)
    col1, col2 = st.columns([1,1])
    classify_pressed = col1.button("Classify")
    col2.button("Clear", on_click=clear_all)

    # ---------- Prediction ----------
    if classify_pressed:
        txt = st.session_state.get("text_input", "").strip()
        if not txt:
            st.warning("Please enter or select some text first.")
        else:
            lang_code, confidence = predict_language(model, txt)
            if lang_code:
                lang_name = token_to_full(lang_code)
                st.markdown("### 🧠 Predicted Language")
                st.success(f"**{lang_name}** (`{lang_code}`)")
                if confidence:
                    st.metric("Confidence", f"{confidence*100:.1f}%")
            else:
                st.warning("Could not classify the given text.")

    # ---------- File Upload ----------
    st.markdown("### 📄 Upload File (PDF / DOCX)")
    uploaded = st.file_uploader("Upload a document", type=["pdf", "docx"])
    process_file = st.button("Classify File")

with col_side:
    st.subheader("Examples")
    example_options = ["— none —"] + list(EXAMPLES_BY_FULLNAME.keys())
    st.selectbox("Choose example", options=example_options, key="example_select", on_change=populate_example_on_change)
    st.caption("Selecting an example fills the input area.")

# ---------- FILE HANDLING ----------
if process_file and uploaded:
    b = uploaded.read()
    file_name = uploaded.name.lower()

    if file_name.endswith(".pdf"):
        pages = extract_text_from_pdf_bytes(b)
        if not pages:
            st.info("No extractable text found in PDF.")
        else:
            st.write(f"Detected {len(pages)} pages. Analyzing first 4000 characters of each page...")
            results = []
            for i, page_text in enumerate(pages, start=1):
                lang_code, _ = predict_language(model, page_text[:4000])
                results.append({"Page": i, "Language": token_to_full(lang_code or "—")})
            st.dataframe(pd.DataFrame(results))
    elif file_name.endswith(".docx"):
        text = extract_text_from_docx_bytes(b)
        if not text:
            st.warning("No readable text found in DOCX.")
        else:
            lang_code, confidence = predict_language(model, text[:4000])
            if lang_code:
                st.success(f"Document language: **{token_to_full(lang_code)}**")
                if confidence:
                    st.metric("Confidence", f"{confidence*100:.1f}%")
            else:
                st.warning("Could not detect language from this document.")
    else:
        st.warning("Unsupported file type. Please upload a PDF or DOCX.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    f"<div style='color:#6b7280; font-size:13px;'>Languages supported: {' > '.join(LANG_MAP.values())}</div>",
    unsafe_allow_html=True
)
