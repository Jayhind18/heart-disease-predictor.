"""
Streamlit frontend for the Heart Disease Classification Model
-------------------------------------------------------------

Run:
1. python flask_api.py
2. streamlit run streamlit_app.py
"""

import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium
import google.generativeai as genai

# ==========================================================
# CONFIG
# ==========================================================

API_URL = "http://localhost:5000"

# Gemini API Key
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-3.5-flash")

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Heart Risk Predictor",
    page_icon="❤️",
    layout="wide",
)

# ==========================================================
# SESSION STATE
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "risk" not in st.session_state:
    st.session_state.risk = None

if "probability" not in st.session_state:
    st.session_state.probability = None

if "reason" not in st.session_state:
    st.session_state.reason = None

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None


# ==========================================================
# HELPER FUNCTION
# ==========================================================

def get_reason(row):
    reasons = []

    if row.get("Age", 0) >= 60:
        reasons.append("Old Age")

    if row.get("BMI", 0) >= 30:
        reasons.append("High BMI")

    if row.get("Hypertension", 0) == 1:
        reasons.append("Hypertension")

    if row.get("Diabetes", 0) == 1:
        reasons.append("Diabetes")

    if row.get("Hyperlipidemia", 0) == 1:
        reasons.append("High Cholesterol")

    if row.get("Family_History", 0) == 1:
        reasons.append("Family History")

    if row.get("Previous_Heart_Attack", 0) == 1:
        reasons.append("Previous Heart Attack")

    if row.get("Blood_Sugar_Fasting", 0) >= 126:
        reasons.append("High Blood Sugar")

    if row.get("Systolic_BP", 0) >= 140:
        reasons.append("High Blood Pressure")

    if row.get("Smoking", 0) == 1:
        reasons.append("Smoking")

    if not reasons:
        return "No major risk factors"

    return ", ".join(reasons)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.header {
    background: linear-gradient(135deg,#0f4c81,#2a6fdb);
    border-radius:25px;
    padding:35px 45px;
    color:white;
    position:relative;
    overflow:hidden;
    box-shadow:0px 8px 30px rgba(0,0,0,.25);
}

/* Background Pattern */
.header::before{
    content:"";
    position:absolute;
    width:500px;
    height:500px;
    right:-150px;
    top:-100px;
    background:rgba(255,255,255,.08);
    border-radius:50%;
}

.header::after{
    content:"✚";
    position:absolute;
    right:70px;
    top:30px;
    font-size:170px;
    color:rgba(255,255,255,.06);
    font-weight:bold;
}

.title{
    font-size:48px;
    font-weight:800;
    margin-bottom:8px;
}

.subtitle{
    font-size:20px;
    color:#dcecff;
    margin-bottom:25px;
}

.features{
    display:flex;
    gap:40px;
    flex-wrap:wrap;
}

.feature{
    font-size:18px;
    color:white;
}

.ecg{
    font-size:22px;
    color:#7CFFB2;
    letter-spacing:3px;
}

</style>

<div class="header">

<div class="title">
❤️ Heart Disease Risk Predictor
</div>

<div class="subtitle">
AI Powered Clinical Decision Support System
</div>

<div class="ecg">
────────────────────⚡────────────────────
</div>

<br>

<div class="features">

<div class="feature">
🤖 <b>AI Powered</b><br>
Smart Prediction
</div>

<div class="feature">
🩺 <b>Clinical Insights</b><br>
Evidence Based
</div>

<div class="feature">
🔒 <b>Secure & Reliable</b><br>
Patient First
</div>

</div>

</div>

""", unsafe_allow_html=True)




# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.title("⚙️ System Status")

    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        api_ok = health.get("status") == "ok"
    except Exception:
        api_ok = False

    if api_ok:
        st.success("✅ Flask API Connected")
    else:
        st.error("❌ Flask API Not Running")
        st.code("python flask_api.py")
        st.stop()
    # ==========================================================
# 🫀 ABOUT PREDICTOR — NO HTML VERSION
# ==========================================================


    st.markdown("""
    <style>

    /* ==============================
    ABOUT PREDICTOR MAIN CARD
    ============================== */

    div[class*="st-key-about_predictor"] {
        background: #ffffff !important;
        background-color: #ffffff !important;

        padding: 22px !important;
        border-radius: 18px !important;

        margin-top: 20px !important;

        border: 1px solid #e4e9f0 !important;

        box-shadow:
            0 8px 25px rgba(31, 41, 55, 0.08) !important;
    }


    /* ==============================
    HEADING
    ============================== */

    div[class*="st-key-about_predictor"] h3 {
        color: #172033 !important;

        font-size: 20px !important;
        font-weight: 700 !important;

        margin-bottom: 12px !important;
    }


    /* ==============================
    DESCRIPTION
    ============================== */

    div[class*="st-key-about_predictor"] p {
        color: #667085 !important;

        font-size: 13px !important;
        line-height: 1.7 !important;

        margin-bottom: 12px !important;
    }


    /* ==============================
    DIVIDER
    ============================== */

    div[class*="st-key-about_predictor"] hr {
        border: none !important;

        border-top: 1px solid #e8edf3 !important;

        margin: 18px 0 !important;
    }


    /* ==============================
    FEATURE TITLES
    ============================== */

    div[class*="st-key-about_predictor"] strong {
        color: #2563eb !important;

        font-size: 13px !important;
        font-weight: 600 !important;
    }


    /* ==============================
    CAPTIONS
    ============================== */

    div[class*="st-key-about_predictor"] small {
        color: #7b8494 !important;

        font-size: 11px !important;
    }


    /* ==============================
    REMOVE EXTRA DARK BACKGROUNDS
    ============================== */

    div[class*="st-key-about_predictor"] [data-testid="stMarkdownContainer"] {
        background: transparent !important;
    }


    /* ==============================
    HOVER EFFECT
    ============================== */

    div[class*="st-key-about_predictor"]:hover {
        box-shadow:
            0 12px 30px rgba(31, 41, 55, 0.12) !important;

        border-color: #d8e1ed !important;

        transition: all 0.25s ease !important;
    }

    </style>
    """, unsafe_allow_html=True)


    with st.container(key="about_predictor"):

        st.markdown("### 🫀 About Predictor")

        st.write(
            "AI-powered heart disease risk prediction system "
            "that analyzes patient health information and "
            "provides an easy-to-understand risk assessment."
        )

        st.markdown("---")

        st.markdown("**🤖 AI Powered**")
        st.caption("Machine Learning Prediction")

        st.markdown("**📊 Risk Analysis**")
        st.caption("Patient Health Assessment")

        st.markdown("**🔗 Flask + ML**")
        st.caption("Backend Prediction API")

        st.markdown("**🛡️ Patient First**")
        st.caption("Simple & Easy Risk Insights")

# ==========================================================
# FEATURE LIST FROM FLASK API
# ==========================================================

try:
    feature_names = requests.get(f"{API_URL}/features", timeout=5).json().get("features", [])
except Exception:
    feature_names = []

    

# ==========================================================
# FEATURE GROUPS
# ==========================================================
st.markdown("""
<style>
/* Tabs container */
.stTabs [data-baseweb="tab-list"] {
    gap: 20px;
}

/* Individual tab */
.stTabs [data-baseweb="tab"] {
    height: 65px;
    padding: 15px 40px;
    font-size: 26px;
    font-weight: 600;
    border-radius: 12px;
}

/* Active tab */
.stTabs [aria-selected="true"] {
    font-size: 24px;
    font-weight: 700;
}

/* Tab text */
.stTabs [data-baseweb="tab"] p {
    font-size: 22px;
}
</style>
""", unsafe_allow_html=True)


GROUPS = {
    "🧍 Personal Information": [
        "Age",
        "Weight",
        "Height",
        "BMI"
    ],
    "🩺 Medical History": [
        "Hypertension",
        "Diabetes",
        "Hyperlipidemia",
        "Family_History",
        "Previous_Heart_Attack"
    ],
    "📊 Vitals & Labs": [
        "Systolic_BP",
        "Diastolic_BP",
        "Heart_Rate",
        "Blood_Sugar_Fasting",
        "Cholesterol_Total"
    ]
}


BINARY_FIELDS = {
    "Hypertension",
    "Diabetes",
    "Hyperlipidemia",
    "Family_History",
    "Previous_Heart_Attack",
    "Smoking"
}



# ==========================================================
# MAIN TABS
# ==========================================================

tab1, tab2, tab3 = st.tabs([
    "🔢 Single Prediction",
    "📄 Batch Prediction",
    "📍 Nearby Hospitals"
])


# ==========================================================
# TAB 1
# ==========================================================

with tab1:
    if feature_names:
        inputs = {}
        remaining = set(feature_names)

        left, right = st.columns([1.5, 1])

        with left:
            for group_name, group_fields in GROUPS.items():
                present = [x for x in group_fields if x in remaining]

                if not present:
                    continue

                st.markdown(
                    f"<div class='card'><div class='section-title'>{group_name}</div>",
                    unsafe_allow_html=True
                )

                cols = st.columns(min(3, len(present)))

                for i, name in enumerate(present):
                    with cols[i % len(cols)]:
                        if name in BINARY_FIELDS:
                            value = st.selectbox(
                                name.replace("_", " "),
                                ["No", "Yes"],
                                key=name
                            )
                            inputs[name] = 1 if value == "Yes" else 0
                        else:
                            inputs[name] = st.number_input(
                                name.replace("_", " "),
                                value=0.0,
                                key=name
                            )

                    remaining.remove(name)

                st.markdown("</div>", unsafe_allow_html=True)

            if remaining:
                st.markdown(
                    "<div class='card'><div class='section-title'>Other Features</div>",
                    unsafe_allow_html=True
                )

                cols = st.columns(3)

                for i, name in enumerate(sorted(remaining)):
                    with cols[i % 3]:
                        inputs[name] = st.number_input(
                            name.replace("_", " "),
                            value=0.0,
                            key=f"other_{name}"
                        )

                st.markdown("</div>", unsafe_allow_html=True)

            predict_button = st.button(
                "🔍 Predict Risk",
                use_container_width=True,
                type="primary"
            )

        # ============================
        # RESULT PANEL
        # ============================

        with right:
            st.markdown(
                "<div class='card'><div class='section-title'>Prediction Result</div>",
                unsafe_allow_html=True
            )

            if predict_button:
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        json={"features": inputs},
                        timeout=10
                    )

                    if response.status_code == 200:
                        result = response.json()
                        prediction = result["prediction"]
                        probabilities = result.get("probabilities", {})
                        disease_probability = float(probabilities.get("1", 0))

                        # Save session
                        st.session_state.prediction = prediction
                        st.session_state.probability = disease_probability * 100
                        st.session_state.reason = get_reason(inputs)

                        if prediction == 1:
                            st.session_state.risk = "High Risk"
                            st.markdown(
                                """
                                <div class='risk-high'>
                                ⚠️ HIGH RISK
                                <br>
                                Heart Disease Likely
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.session_state.risk = "Low Risk"
                            st.markdown(
                                """
                                <div class='risk-low'>
                                ✅ LOW RISK
                                <br>
                                Heart Disease Unlikely
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        st.write("")
                        st.metric("Disease Probability", f"{disease_probability*100:.2f}%")
                        st.progress(disease_probability)
                        st.write("")
                        st.subheader("Risk Factors")
                        st.info(st.session_state.reason)
                    else:
                        st.error(response.json().get("error", "Prediction failed"))

                except Exception as e:
                    st.error(f"API Error : {e}")
            else:
                st.info("Fill patient details and click Predict Risk.")

            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("Feature list not received from Flask API.")

# ==========================================================
# TAB 2 : BATCH PREDICTION
# ==========================================================

with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📄 Batch Prediction</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.success("CSV Uploaded Successfully")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("▶ Run Batch Prediction", type="primary"):
            predictions = []
            probabilities = []

            progress = st.progress(0)
            total = len(df)

            for index, (_, row) in enumerate(df.iterrows()):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        json={"features": row.to_dict()},
                        timeout=10
                    )

                    if response.status_code == 200:
                        result = response.json()
                        pred = result["prediction"]
                        prob = float(result.get("probabilities", {}).get("1", 0))
                    else:
                        pred = None
                        prob = None

                except Exception:
                    pred = None
                    prob = None

                predictions.append(pred)
                probabilities.append(prob)

                progress.progress((index + 1) / total)

            progress.empty()

            df["Prediction"] = predictions
            df["Risk"] = ["High Risk" if x == 1 else "Low Risk" for x in predictions]
            df["Probability (%)"] = [
                round(p * 100, 2) if p is not None else None
                for p in probabilities
            ]
            df["Reason"] = df.apply(get_reason, axis=1)

            st.session_state.batch_results = df.copy()

            st.success("Prediction Completed")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇ Download Prediction CSV",
                csv,
                "prediction_results.csv",
                "text/csv",
                type="primary"
            )

    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================
# 🏥 NEARBY HOSPITALS
# ==========================================================

st.markdown("### 🏥 Nearby Hospitals")

st.success("Location Found ✅")

import urllib.parse

hospital_query = urllib.parse.quote(
    "hospitals near me"
)

google_maps_url = (
    "https://www.google.com/maps/search/"
    + hospital_query
)

st.link_button(
    "🏥 Find Nearby Hospitals",
    google_maps_url
)

st.info(
    "Click the button above to find nearby hospitals "
    "using Google Maps."
)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

col1, col2, col3 = st.columns(
    [1, 3, 1.35],
    gap="medium"
)

with col1:
    st.info("🩺 AI Powered Heart Disease Prediction")

with col2:
    st.info("⚙️ Flask API + Streamlit")

with col3:
    st.info("🤖 Gemini AI Assistant")


# ==========================================================
# ❤️ HEART AI ASSISTANT
# FLOATING RIGHT BUTTON + RIGHT CHAT WINDOW
# ==========================================================


# ==========================================================
# INITIALIZE SESSION STATE
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


if "heart_ai_open" not in st.session_state:
    st.session_state.heart_ai_open = False


# ==========================================================
# CHATBOT CSS
# ==========================================================

st.markdown("""
<style>

/* ==========================================================
   FLOATING BUTTON CONTAINER
   ========================================================== */

div.st-key-heart_ai_toggle {

    position: fixed !important;

    right: 28px !important;

    bottom: 28px !important;

    z-index: 999999 !important;

    width: 180px !important;

}


/* ==========================================================
   FLOATING BUTTON
   ========================================================== */

div.st-key-heart_ai_toggle button {

    width: 180px !important;

    height: 55px !important;

    border-radius: 30px !important;

    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    ) !important;

    color: white !important;

    border: none !important;

    font-size: 15px !important;

    font-weight: 700 !important;

    box-shadow:
        0 8px 25px rgba(37, 99, 235, 0.35) !important;

    transition: all 0.25s ease !important;

}


/* ==========================================================
   BUTTON HOVER
   ========================================================== */

div.st-key-heart_ai_toggle button:hover {

    transform: translateY(-3px) !important;

    box-shadow:
        0 12px 32px rgba(37, 99, 235, 0.50) !important;

}


/* ==========================================================
   CHAT WINDOW
   ========================================================== */

div.st-key-heart_ai_window {

    position: fixed !important;

    right: 28px !important;

    bottom: 95px !important;

    width: 430px !important;

    height: 680px !important;

    max-height: calc(100vh - 120px) !important;

    background: white !important;

    border: 1px solid #e5e7eb !important;

    border-radius: 20px !important;

    box-shadow:
        0 20px 60px rgba(15, 23, 42, 0.30) !important;

    z-index: 999998 !important;

    overflow-y: auto !important;

    overflow-x: hidden !important;

    padding: 0 !important;

}


/* ==========================================================
   CHAT HEADER
   ========================================================== */

div.st-key-heart_chat_header {

    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    ) !important;

    padding: 18px 20px !important;

    border-radius: 20px 20px 0 0 !important;

}


/* Header title */

div.st-key-heart_chat_header h3 {

    color: white !important;

    margin: 0 !important;

    font-size: 21px !important;

}


/* Header status */

div.st-key-heart_chat_header p {

    color: #dcfce7 !important;

    font-size: 12px !important;

    font-weight: 600 !important;

    margin: 3px 0 0 0 !important;

}


/* ==========================================================
   CHAT CONTENT
   ========================================================== */

div.st-key-heart_chat_content {

    padding: 18px !important;

    background: white !important;

}


/* ==========================================================
   WELCOME SCREEN
   ========================================================== */

div.st-key-heart_welcome {

    text-align: center !important;

    padding: 25px 10px 20px 10px !important;

}


/* Heart */

div.st-key-heart_welcome h1 {

    font-size: 50px !important;

    margin-bottom: 8px !important;

}


/* Hello */

div.st-key-heart_welcome h3 {

    color: #111827 !important;

    font-size: 20px !important;

    margin-bottom: 8px !important;

}


/* ==========================================================
   CHAT BUTTONS
   ========================================================== */

div.st-key-heart_chat_content button {

    border-radius: 10px !important;

    min-height: 40px !important;

    font-size: 13px !important;

    font-weight: 600 !important;

}


/* ==========================================================
   SEND BUTTON
   ========================================================== */

div.st-key-heart_chat_content button[kind="primary"] {

    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    ) !important;

    color: white !important;

    border: none !important;

}


/* ==========================================================
   INPUT
   ========================================================== */

div.st-key-heart_chat_content input {

    height: 48px !important;

    border: 1px solid #dbe1ea !important;

    border-radius: 12px !important;

    background: white !important;

    color: #111827 !important;

    font-size: 13px !important;

}


/* ==========================================================
   INPUT FOCUS
   ========================================================== */

div.st-key-heart_chat_content input:focus {

    border-color: #6366f1 !important;

    box-shadow:
        0 0 0 2px rgba(99, 102, 241, 0.10) !important;

}


/* ==========================================================
   CHAT MESSAGES
   ========================================================== */

div.st-key-heart_chat_content
[data-testid="stChatMessage"] {

    border-radius: 12px !important;

}


/* ==========================================================
   DIVIDER
   ========================================================== */

div.st-key-heart_chat_content hr {

    border-color: #e5e7eb !important;

    margin-top: 15px !important;

    margin-bottom: 15px !important;

}


/* ==========================================================
   DISCLAIMER
   ========================================================== */

div.st-key-heart_chat_content
.heart-ai-warning {

    text-align: center;

    color: #6b7280;

    font-size: 10px;

    line-height: 1.5;

    padding: 12px 5px 15px 5px;

}


/* ==========================================================
   MOBILE
   ========================================================== */

@media (max-width: 600px) {

    div.st-key-heart_ai_toggle {

        right: 12px !important;

        bottom: 15px !important;

        width: 165px !important;

    }


    div.st-key-heart_ai_toggle button {

        width: 165px !important;

        height: 52px !important;

        font-size: 14px !important;

    }


    div.st-key-heart_ai_window {

        right: 10px !important;

        bottom: 80px !important;

        width: calc(100vw - 20px) !important;

        height: calc(100vh - 110px) !important;

        max-height: calc(100vh - 110px) !important;

    }

}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# ❤️ FLOATING HEART AI BUTTON
# ==========================================================

with st.container(key="heart_ai_toggle"):

    if st.button(
        "🫀  HeartCare AI",
        key="heart_ai_toggle_button",
        use_container_width=True
    ):

        st.session_state.heart_ai_open = (
            not st.session_state.heart_ai_open
        )

        st.rerun()

# ==========================================================
# SHOW CHATBOT ONLY WHEN BUTTON IS CLICKED
# ==========================================================

if st.session_state.heart_ai_open:

    with st.container(key="heart_ai_window"):


        # ==================================================
        # HEADER
        # ==================================================

        with st.container(key="heart_chat_header"):

            st.markdown(
                "### ❤️ Heart AI Assistant"
            )

            st.markdown(
                "🟢 Online • Ready to help"
            )


        # ==================================================
        # CHAT CONTENT
        # ==================================================

        with st.container(key="heart_chat_content"):


            # ==================================================
            # CLOSE + CLEAR BUTTONS
            # ==================================================

            top_col1, top_col2 = st.columns(
                [5, 1]
            )


            with top_col2:

                if st.button(
                    "✕",
                    key="heart_close_chat",
                    use_container_width=True
                ):

                    st.session_state.heart_ai_open = False

                    st.rerun()


            # ==================================================
            # CLEAR CHAT
            # ==================================================

            clear_col1, clear_col2 = st.columns(
                [5, 1]
            )


            with clear_col2:

                if st.button(
                    "🗑",
                    key="heart_clear_chat",
                    use_container_width=True
                ):

                    st.session_state.messages = []

                    st.rerun()


            # ==================================================
            # WELCOME SCREEN
            # ==================================================

            if not st.session_state.messages:

                with st.container(
                    key="heart_welcome"
                ):

                    st.markdown("# ❤️")

                    st.markdown(
                        "### Hello! 👋"
                    )

                    st.markdown(
                        "**I'm your Heart AI Assistant**"
                    )

                    st.caption(
                        "Ask me anything about heart health, "
                        "risk factors, prevention, diet, "
                        "exercise or your prediction results."
                    )


            # ==================================================
            # CHAT HISTORY
            # ==================================================

            for msg in st.session_state.messages:

                if msg["role"] == "user":

                    with st.chat_message("user"):

                        st.write(
                            msg["content"]
                        )

                else:

                    with st.chat_message(
                        "assistant",
                        avatar="❤️"
                    ):

                        st.write(
                            msg["content"]
                        )


            # ==================================================
            # DIVIDER
            # ==================================================

            st.divider()


            # ==================================================
            # QUICK QUESTIONS
            # ==================================================

            st.markdown(
                "**Quick Questions**"
            )


            quick_col1, quick_col2 = st.columns(
                2
            )


            with quick_col1:

                risk_question = st.button(
                    "❤️ Risk Factors",
                    key="heart_quick_risk",
                    use_container_width=True
                )


                diet_question = st.button(
                    "🥗 Healthy Diet",
                    key="heart_quick_diet",
                    use_container_width=True
                )


            with quick_col2:

                exercise_question = st.button(
                    "🏃 Exercise",
                    key="heart_quick_exercise",
                    use_container_width=True
                )


                prevention_question = st.button(
                    "🫀 Prevention",
                    key="heart_quick_prevention",
                    use_container_width=True
                )


            # ==================================================
            # QUICK QUESTION LOGIC
            # ==================================================

            prompt = None


            if risk_question:

                prompt = (
                    "What are the major risk factors "
                    "for heart disease?"
                )


            elif diet_question:

                prompt = (
                    "What diet is good for heart health?"
                )


            elif exercise_question:

                prompt = (
                    "What exercises are good for heart health?"
                )


            elif prevention_question:

                prompt = (
                    "How can I prevent heart disease "
                    "and keep my heart healthy?"
                )


            # ==================================================
            # USER INPUT
            # ==================================================

            user_input = st.text_input(
                "Ask your question",
                placeholder="Type your question...",
                key="heart_ai_input"
            )


            # ==================================================
            # SEND BUTTON
            # ==================================================

            send_col1, send_col2 = st.columns(
                [4, 1]
            )


            with send_col2:

                send_clicked = st.button(
                    "➤",
                    key="heart_ai_send",
                    type="primary",
                    use_container_width=True
                )


            if send_clicked:

                if user_input.strip():

                    prompt = user_input.strip()


            # ==================================================
            # GEMINI AI PROCESSING
            # ==================================================

            if prompt:


                # ------------------------------------------------
                # SAVE USER MESSAGE
                # ------------------------------------------------

                st.session_state.messages.append({

                    "role": "user",

                    "content": prompt

                })


                # ------------------------------------------------
                # BATCH PATIENT CONTEXT
                # ------------------------------------------------

                patient_context = ""


                if (
                    "batch_results"
                    in st.session_state
                    and
                    st.session_state.batch_results
                    is not None
                ):

                    df = st.session_state.batch_results


                    if "Patient_Name" in df.columns:

                        for _, row in df.iterrows():

                            patient_name = str(
                                row["Patient_Name"]
                            )


                            if (
                                patient_name.lower()
                                in prompt.lower()
                            ):

                                patient_context = f"""
Patient Name : {row['Patient_Name']}
Prediction : {row['Prediction']}
Risk : {row['Risk']}
Probability : {row['Probability (%)']} %
Reason : {row['Reason']}
"""

                                break


                # ------------------------------------------------
                # CURRENT PREDICTION
                # ------------------------------------------------

                probability = st.session_state.get(
                    "probability",
                    None
                )


                if probability is None:

                    probability_text = (
                        "Not Available"
                    )

                else:

                    try:

                        probability_text = (
                            f"{float(probability):.2f}%"
                        )

                    except Exception:

                        probability_text = str(
                            probability
                        )


                # ------------------------------------------------
                # CURRENT PATIENT VALUES
                # ------------------------------------------------

                prediction = (
                    st.session_state.get(
                        "prediction",
                        "Not Available"
                    )
                )


                risk = (
                    st.session_state.get(
                        "risk",
                        "Not Available"
                    )
                )


                reason = (
                    st.session_state.get(
                        "reason",
                        "Not Available"
                    )
                )


                # ------------------------------------------------
                # GEMINI SYSTEM PROMPT
                # ------------------------------------------------

                system_prompt = f"""
You are an expert Heart Disease AI Assistant.

You are part of a Heart Disease Risk Prediction application.

CURRENT PATIENT PREDICTION

Prediction:
{prediction}

Risk:
{risk}

Probability:
{probability_text}

Reason:
{reason}


BATCH PATIENT INFORMATION

{patient_context}


INSTRUCTIONS

- Explain everything in simple English.
- Explain heart disease risk factors.
- Explain the prediction when information is available.
- Answer questions about batch patients when information is available.
- Suggest general healthy diet options.
- Suggest general exercise and lifestyle habits.
- Never prescribe medicines.
- Never claim to replace a doctor.
- For urgent symptoms, advise seeking professional medical attention.
- Keep answers clear and easy to understand.
- Do not make unsupported medical claims.


USER QUESTION

{prompt}
"""


                # ------------------------------------------------
                # GEMINI RESPONSE
                # ------------------------------------------------

                with st.spinner("❤️ Thinking..."):

                    try:

                        response = model.generate_content(
                            system_prompt
                        )

                        answer = response.text


                    except Exception as e:

                        answer = (
                            f"⚠️ Gemini Error: {e}"
                        )


                # ------------------------------------------------
                # SAVE AI RESPONSE
                # ------------------------------------------------

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": answer

                })


                # ------------------------------------------------
                # REFRESH
                # ------------------------------------------------

                st.rerun()


            # ==================================================
            # DISCLAIMER
            # ==================================================

            st.markdown(
                """
                <div class="heart-ai-warning">
                    ⚠️ AI Assistant can make mistakes.
                    Please consult a qualified doctor for medical advice.
                </div>
                """,
                unsafe_allow_html=True
            )
# ==========================================================
# END OF FILE
# ==========================================================
# ==============================
import streamlit.components.v1 as components

# ==========================================
# 🔊 HEART DISEASE PREDICTOR VOICE SYSTEM
# ==========================================

components.html(
    """
    <script>

    // ==========================================
    // 🔊 WELCOME MESSAGE
    // ==========================================

    function welcomeMessage() {

        if ("speechSynthesis" in window) {

            window.speechSynthesis.cancel();

            var welcome = new SpeechSynthesisUtterance(
                "Welcome to my Heart Disease Predictor"
            );

            welcome.lang = "en-US";
            welcome.rate = 0.85;
            welcome.pitch = 1;
            welcome.volume = 1;

            window.speechSynthesis.speak(welcome);
        }
    }


    // ==========================================
    // 🔊 SPEAK FUNCTION
    // ==========================================

    function speak(text) {

        if ("speechSynthesis" in window) {

            window.speechSynthesis.cancel();

            var voice = new SpeechSynthesisUtterance(text);

            voice.lang = "en-US";
            voice.rate = 0.9;
            voice.pitch = 1;
            voice.volume = 1;

            window.speechSynthesis.speak(voice);
        }
    }


    // ==========================================
    // 🔊 FEATURE VOICE
    // ==========================================

    function setupVoice() {

        var buttons =
            window.parent.document.querySelectorAll("button");


        buttons.forEach(function(button) {

            if (button.dataset.voiceAdded === "true") {
                return;
            }

            button.dataset.voiceAdded = "true";


            button.addEventListener("click", function() {

                var text = button.innerText || "";

                text = text.trim();


                if (text.includes("Single Prediction")) {

                    speak("Single Prediction");

                }

                else if (text.includes("Batch Prediction")) {

                    speak("Batch Prediction");

                }

                else if (text.includes("AI Chatbot")) {

                    speak("AI Chatbot");

                }

                else if (text.includes("Nearby Hospitals")) {

                    speak("Nearby Hospitals");

                }

                else if (text.includes("Clear Chat")) {

                    speak("Clear Chat");

                }

                else if (text.includes("Find Nearest Hospital")) {

                    speak("Find Nearest Hospital");

                }

                else if (text.includes("Predict")) {

                    speak("Predict");

                }

                else if (text.includes("Download")) {

                    speak("Download Results");

                }

            });

        });


        // ==========================================
        // 🔊 STREAMLIT TABS
        // ==========================================

        var tabs =
            window.parent.document.querySelectorAll(
                '[role="tab"]'
            );


        tabs.forEach(function(tab) {

            if (tab.dataset.voiceAdded === "true") {
                return;
            }

            tab.dataset.voiceAdded = "true";


            tab.addEventListener("click", function() {

                var text = tab.innerText || "";

                text = text.trim();


                if (text.includes("Single Prediction")) {

                    speak("Single Prediction");

                }

                else if (text.includes("Batch Prediction")) {

                    speak("Batch Prediction");

                }

                else if (text.includes("AI Chatbot")) {

                    speak("AI Chatbot");

                }

                else if (text.includes("Nearby Hospitals")) {

                    speak("Nearby Hospitals");

                }

            });

        });

    }


    // ==========================================
    // 🚀 START WELCOME VOICE
    // ==========================================

    setTimeout(function() {

        welcomeMessage();

        setTimeout(function() {
            setupVoice();
        }, 1000);

    }, 1000);


    // Re-check Streamlit elements
    setInterval(setupVoice, 2000);


    </script>
    """,
    height=0,
    width=0
)