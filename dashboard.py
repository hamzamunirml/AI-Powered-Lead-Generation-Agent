"""
Professional Lead Recommendation Dashboard
AI-Powered Lead Scoring System
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import base64
from streamlit_option_menu import option_menu

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="LeadScout AI | Lead Recommendation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== CUSTOM CSS FOR PROFESSIONAL LOOK ====================
st.markdown(
    """
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }

    /* Card styling */
    .card {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
    }

    /* Metric card styling */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-top: 4px solid #667eea;
    }

    /* Prediction badges */
    .badge-high {
        background-color: #dc3545;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }

    .badge-medium {
        background-color: #ffc107;
        color: #333;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }

    .badge-low {
        background-color: #28a745;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #2c3e50;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        margin-top: 40px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("notebooks/saved_models/best_lead_model.pkl")
        return model
    except:
        try:
            model = joblib.load("../notebooks/saved_models/best_lead_model.pkl")
            return model
        except:
            return None


# ==================== RECOMMENDATIONS ====================
RECOMMENDATIONS = {
    "High": {
        "message": "🔴 Priority Lead - Contact within 24 hours",
        "icon": "🚨",
        "color": "#dc3545",
        "action": "Immediate call/email required",
        "priority": 1,
    },
    "Medium": {
        "message": "🟡 Potential Opportunity - Add to nurture campaign",
        "icon": "📋",
        "color": "#ffc107",
        "action": "Schedule follow-up within week",
        "priority": 2,
    },
    "Low": {
        "message": "🟢 Low Priority - Monitor for future engagement",
        "icon": "📊",
        "color": "#28a745",
        "action": "Add to newsletter list",
        "priority": 3,
    },
}

# ==================== MAIN HEADER ====================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(
        """
    <div class="main-header" style="text-align: center;">
        <h1>🎯 LeadScout AI</h1>
        <p style="font-size: 18px;">Intelligent Lead Recommendation System</p>
        <p style="font-size: 14px; opacity: 0.9;">AI-Powered | Real-Time Predictions | Actionable Insights</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### 🚀 Navigation")

    selected = option_menu(
        menu_title=None,
        options=[
            "Single Prediction",
            "Batch Processing",
            "Analytics Dashboard",
            "Model Insights",
            "API Documentation",
        ],
        icons=["cursor", "upload", "graph-up", "brain", "code"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#2c3e50"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {
                "color": "white",
                "font-size": "14px",
                "text-align": "left",
                "margin": "5px",
            },
            "nav-link-selected": {"background-color": "#667eea"},
        },
    )

    st.markdown("---")
    st.markdown("### 📊 System Status")

    model_status = "✅ Active" if load_model() else "❌ Inactive"
    st.markdown(f"**Model:** {model_status}")
    st.markdown(f"**Version:** v2.0.0")
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d')}")

    st.markdown("---")
    st.markdown("### 🎯 Lead Scoring Criteria")
    st.info("""
    **High Priority (60+ points)**
    - Contact within 24 hours
    - Direct sales outreach

    **Medium Priority (35-59 points)**
    - Nurture campaign
    - Weekly follow-up

    **Low Priority (<35 points)**
    - Monitor engagement
    - Newsletter subscription
    """)

# ==================== SINGLE PREDICTION TAB ====================
if selected == "Single Prediction":
    st.markdown("### 📝 Manual Lead Entry")
    st.markdown("Enter lead information to get instant prediction and recommendations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌐 Website Information")
        website_exists = st.selectbox(
            "Website Status",
            options=[1, 0],
            format_func=lambda x: (
                "✅ Yes, Website Available" if x == 1 else "❌ No Website"
            ),
            help="Does the company have a professional website?",
        )

        contact_form = st.selectbox(
            "Contact Form",
            options=[1, 0],
            format_func=lambda x: (
                "✅ Yes, Contact Form Available" if x == 1 else "❌ No Contact Form"
            ),
            help="Is there a contact form on the website?",
        )

        services_count = st.slider(
            "Number of Services",
            min_value=0,
            max_value=10,
            value=3,
            help="How many services does the company offer?",
        )

    with col2:
        st.markdown("#### 📈 Company Profile")

        # Country selection (country_score)
        country_score = st.selectbox(
            "Target Country",
            options=[3, 2, 1],
            format_func=lambda x: (
                "🇺🇸 United States (Score: 3)"
                if x == 3
                else "🇨🇦 Canada (Score: 2)" if x == 2 else "🇦🇪 UAE (Score: 1)"
            ),
            help="Geographic market priority score",
        )

    st.markdown("---")

    # Calculate Score Preview (with 4 features only)
    score_preview = 0
    score_preview += 10 if website_exists == 1 else 0
    score_preview += 20 if contact_form == 1 else 0
    score_preview += 15 if country_score == 3 else (10 if country_score == 2 else 5)
    score_preview += 15 if services_count > 3 else 0

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔮 Generate Prediction", use_container_width=True):
            model = load_model()
            if model:
                # Use ONLY 4 features (without about_word_count)
                features = [
                    [
                        website_exists,
                        contact_form,
                        services_count,
                        country_score,
                    ]
                ]
                prediction = model.predict(features)[0]
                rec = RECOMMENDATIONS.get(prediction, {})

                # Results Card
                st.markdown(
                    f"""
                <div class="card" style="text-align: center;">
                    <h2>{rec.get('icon', '')} Lead Quality: {prediction}</h2>
                    <div style="background-color: {rec.get('color', '#ccc')}; padding: 15px; border-radius: 10px; margin: 20px 0;">
                        <p style="font-size: 18px; margin: 0; color: {'#333' if prediction == 'Medium' else 'white'}">
                            <strong>{rec.get('message', '')}</strong>
                        </p>
                    </div>
                    <p><strong>Recommended Action:</strong> {rec.get('action', '')}</p>
                    <p><strong>Score Preview:</strong> {score_preview} points</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Save to history
                if "history" not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append(
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "website_exists": website_exists,
                        "contact_form": contact_form,
                        "services_count": services_count,
                        "country_score": country_score,
                        "prediction": prediction,
                        "score": score_preview,
                    }
                )

                st.balloons()
            else:
                st.error("❌ Model not loaded! Please check model file.")

# ==================== BATCH PROCESSING TAB ====================
elif selected == "Batch Processing":
    st.markdown("### 📁 Batch Lead Processing")
    st.markdown("Upload a CSV file with multiple leads to generate predictions in bulk")

    st.info("""
    **Required CSV Columns:**
    - website_exists (0 or 1)
    - contact_form (0 or 1)
    - services_count (integer)
    - country_score (1, 2, or 3)

    Optional: Company Name (for better readability)
    """)

    uploaded_file = st.file_uploader("Choose CSV file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.markdown("#### 📋 Input Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Required columns (4 features only)
            required_cols = [
                "website_exists",
                "contact_form",
                "services_count",
                "country_score",
            ]

            if st.button("🚀 Process Batch", use_container_width=True):
                model = load_model()
                if model:
                    X = df[required_cols]
                    predictions = model.predict(X)

                    df["Prediction"] = predictions
                    df["Recommendation"] = df["Prediction"].apply(
                        lambda x: RECOMMENDATIONS.get(x, {}).get("message", "")
                    )
                    df["Priority Score"] = df.apply(
                        lambda row: (
                            10
                            if row["website_exists"] == 1
                            else (
                                0 + 20
                                if row["contact_form"] == 1
                                else (
                                    0
                                    + (
                                        15
                                        if row["country_score"] == 3
                                        else 10 if row["country_score"] == 2 else 5
                                    )
                                    + 15
                                    if row["services_count"] > 3
                                    else 0
                                )
                            )
                        ),
                        axis=1,
                    )

                    st.success(f"✅ Processed {len(df)} leads successfully!")

                    st.markdown("#### 📊 Results Preview")
                    st.dataframe(
                        (
                            df[
                                [
                                    "Company Name",
                                    "Prediction",
                                    "Recommendation",
                                    "Priority Score",
                                ]
                            ].head(10)
                            if "Company Name" in df.columns
                            else df[
                                ["Prediction", "Recommendation", "Priority Score"]
                            ].head(10)
                        ),
                        use_container_width=True,
                    )

                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions (CSV)",
                        data=csv,
                        file_name=f"lead_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

                    # Distribution chart
                    st.markdown("#### 📈 Prediction Distribution")
                    fig = px.pie(
                        df,
                        names="Prediction",
                        title="Lead Quality Distribution",
                        color="Prediction",
                        color_discrete_map={
                            "High": "#dc3545",
                            "Medium": "#ffc107",
                            "Low": "#28a745",
                        },
                        hole=0.4,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error("❌ Model not loaded!")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# ==================== ANALYTICS DASHBOARD TAB ====================
elif selected == "Analytics Dashboard":
    st.markdown("### 📈 Analytics Dashboard")

    # Load historical data if available
    if "history" in st.session_state and st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                """
            <div class="metric-card">
                <h3>📊 Total</h3>
                <h2>{}</h2>
                <p>Predictions Made</p>
            </div>
            """.format(len(history_df)),
                unsafe_allow_html=True,
            )

        with col2:
            high_count = len(history_df[history_df["prediction"] == "High"])
            st.markdown(
                """
            <div class="metric-card">
                <h3>🔴 High</h3>
                <h2>{}</h2>
                <p>Priority Leads</p>
            </div>
            """.format(high_count),
                unsafe_allow_html=True,
            )

        with col3:
            medium_count = len(history_df[history_df["prediction"] == "Medium"])
            st.markdown(
                """
            <div class="metric-card">
                <h3>🟡 Medium</h3>
                <h2>{}</h2>
                <p>Opportunities</p>
            </div>
            """.format(medium_count),
                unsafe_allow_html=True,
            )

        with col4:
            low_count = len(history_df[history_df["prediction"] == "Low"])
            st.markdown(
                """
            <div class="metric-card">
                <h3>🟢 Low</h3>
                <h2>{}</h2>
                <p>Monitor Leads</p>
            </div>
            """.format(low_count),
                unsafe_allow_html=True,
            )

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.bar(
                history_df["prediction"].value_counts().reset_index(),
                x="prediction",
                y="count",
                title="Prediction Distribution",
                color="prediction",
                color_discrete_map={
                    "High": "#dc3545",
                    "Medium": "#ffc107",
                    "Low": "#28a745",
                },
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.line(
                history_df,
                x="timestamp",
                y="score",
                title="Score Trend Over Time",
                markers=True,
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### 📋 Prediction History")
        st.dataframe(history_df, use_container_width=True)

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info(
            "No prediction history yet. Make some predictions in the Single Prediction tab!"
        )

# ==================== MODEL INSIGHTS TAB ====================
elif selected == "Model Insights":
    st.markdown("### 🧠 Model Insights & Explainability")

    st.markdown(
        """
    <div class="card">
        <h4>🎯 Feature Importance Analysis</h4>
        <p>Understanding which factors most influence lead scoring decisions:</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Feature Weightage")
        feature_importance = pd.DataFrame(
            {
                "Feature": [
                    "Contact Form",
                    "Country Score",
                    "Services Count",
                    "Website Exists",
                ],
                "Importance": [35, 30, 25, 10],
            }
        )
        fig = px.bar(
            feature_importance,
            x="Feature",
            y="Importance",
            title="Feature Impact on Lead Score",
            color="Importance",
            color_continuous_scale="viridis",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📋 Scoring Rules")
        st.markdown("""
        | Feature | Points |
        |---------|--------|
        | Contact Form Available | +20 |
        | USA Company | +15 |
        | Services > 3 | +15 |
        | Website Available | +10 |
        | Canada Company | +10 |
        | UAE Company | +5 |
        """)

    st.markdown("---")
    st.markdown("#### 🎯 Classification Thresholds")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <h3 style="color: #dc3545;">🔴 High</h3>
            <p>Score ≥ 60</p>
            <small>Contact within 24 hours</small>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
        <div class="metric-card">
            <h3 style="color: #ffc107;">🟡 Medium</h3>
            <p>Score 35 - 59</p>
            <small>Nurture campaign</small>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
        <div class="metric-card">
            <h3 style="color: #28a745;">🟢 Low</h3>
            <p>Score &lt; 35</p>
            <small>Monitor only</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

# ==================== API DOCUMENTATION TAB ====================
elif selected == "API Documentation":
    st.markdown("### 🔌 API Documentation")

    st.markdown(
        """
    <div class="card">
        <h4>📡 FastAPI Endpoints</h4>
        <p>The system provides REST API endpoints for programmatic access.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏠 GET /")
        st.code("""
Response:
{
    "message": "Lead Recommendation API is running!",
    "endpoints": {
        "GET /health": "Check API status",
        "POST /predict": "Predict lead quality"
    }
}
        """)

        st.markdown("#### 💚 GET /health")
        st.code("""
Response:
{
    "status": "healthy",
    "model_loaded": true
}
        """)

    with col2:
        st.markdown("#### 🎯 POST /predict")
        st.code("""
Request Body (4 features):
{
    "website_exists": 1,
    "contact_form": 1,
    "services_count": 4,
    "country_score": 3
}

Response:
{
    "prediction": "High",
    "recommendation": "Priority Lead - Contact within 24 hours",
    "timestamp": "2024-01-15T10:30:00"
}
        """)

    st.markdown("#### 🐍 Python Example")
    st.code("""
import requests

url = "http://localhost:8000/predict"
data = {
    "website_exists": 1,
    "contact_form": 1,
    "services_count": 4,
    "country_score": 3
}

response = requests.post(url, json=data)
print(response.json())
    """)

# ==================== FOOTER ====================
st.markdown(
    """
<div class="footer">
    <p>🤖 AI-Powered Lead Recommendation System | Built with Streamlit & FastAPI</p>
    <p style="font-size: 12px;">© 2024 LeadScout AI | Machine Learning Internship Project</p>
</div>
""",
    unsafe_allow_html=True,
)
