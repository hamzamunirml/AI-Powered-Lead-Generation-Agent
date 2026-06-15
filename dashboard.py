"""
Professional Lead Recommendation Dashboard
AI-Powered Lead Scoring System
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime
import numpy as np
from streamlit_option_menu import option_menu

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="LeadScout AI | Lead Recommendation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== CUSTOM CSS ====================
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
        text-align: center;
    }
    .card {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-top: 4px solid #667eea;
    }
    .result-high {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        border-radius: 15px;
        padding: 25px;
        color: white;
        text-align: center;
    }
    .result-medium {
        background: linear-gradient(135deg, #ffc107 0%, #e0a800 100%);
        border-radius: 15px;
        padding: 25px;
        color: #333;
        text-align: center;
    }
    .result-low {
        background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%);
        border-radius: 15px;
        padding: 25px;
        color: white;
        text-align: center;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        margin-top: 40px;
    }

    /* Purple Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 30px;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #8e44ad 0%, #7d3c98 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(155, 89, 182, 0.4);
    }
</style>
""",
    unsafe_allow_html=True,
)


# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    """Load the trained model from various possible paths"""
    paths_to_try = [
        "notebooks/saved_models/best_lead_model.pkl",
        "../notebooks/saved_models/best_lead_model.pkl",
        "best_lead_model.pkl",
        "models/best_lead_model.pkl",
    ]
    for path in paths_to_try:
        try:
            model = joblib.load(path)
            return model
        except:
            continue
    return None


# ==================== SCORE CALCULATION ====================
def calculate_score(website_exists, contact_form, services_count, country_score):
    """Calculate lead score based on features"""
    score = 0
    if website_exists == 1:
        score += 10
    if contact_form == 1:
        score += 20
    if country_score == 3:
        score += 15
    elif country_score == 2:
        score += 10
    elif country_score == 1:
        score += 5
    if services_count > 3:
        score += 15
    return score


def get_priority_from_score(score):
    """Get priority based on score only"""
    if score >= 60:
        return "High"
    elif score >= 35:
        return "Medium"
    else:
        return "Low"


def get_recommendation(priority, score):
    """Get recommendation based on priority"""
    if priority == "High":
        return {
            "label": "HIGH PRIORITY LEAD",
            "message": "Priority Lead - Contact within 24 hours",
            "full_message": "This is a HIGH priority lead. Contact immediately within 24 hours.",
            "icon": "🔴",
            "action": "Immediate call/email required",
            "instruction": "Send to sales team for immediate follow-up",
        }
    elif priority == "Medium":
        return {
            "label": "MEDIUM PRIORITY LEAD",
            "message": "Potential Opportunity - Add to nurture campaign",
            "full_message": "This is a MEDIUM priority lead. Add to nurture campaign for follow-up.",
            "icon": "🟡",
            "action": "Send nurturing email sequence",
            "instruction": "Add to marketing nurture campaign",
        }
    else:
        return {
            "label": "LOW PRIORITY LEAD",
            "message": "Low Priority - Monitor for future engagement",
            "full_message": "This is a LOW priority lead. Monitor for future engagement.",
            "icon": "🟢",
            "action": "Add to newsletter list",
            "instruction": "Track engagement metrics only",
        }


def display_result_card(priority, rec, score):
    """Display result card based on priority"""
    if priority == "High":
        st.markdown(
            f"""
        <div class="result-high">
            <h1>{rec['icon']} {rec['label']}</h1>
            <p style="font-size: 18px; margin: 15px 0;">⚠️ {rec['full_message']}</p>
            <hr style="background: white; margin: 15px 0;">
            <p><strong>⭐ Recommended Action:</strong> {rec['action']}</p>
            <p><strong>📈 Lead Score:</strong> {score} points (Required: 60+ points)</p>
            <p><strong>📝 Instruction:</strong> {rec['instruction']}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    elif priority == "Medium":
        st.markdown(
            f"""
        <div class="result-medium">
            <h1>{rec['icon']} {rec['label']}</h1>
            <p style="font-size: 18px; margin: 15px 0;">📋 {rec['full_message']}</p>
            <hr style="background: #333; margin: 15px 0;">
            <p><strong>⭐ Recommended Action:</strong> {rec['action']}</p>
            <p><strong>📈 Lead Score:</strong> {score} points (Required: 35-59 points)</p>
            <p><strong>📝 Instruction:</strong> {rec['instruction']}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div class="result-low">
            <h1>{rec['icon']} {rec['label']}</h1>
            <p style="font-size: 18px; margin: 15px 0;">📊 {rec['full_message']}</p>
            <hr style="background: white; margin: 15px 0;">
            <p><strong>⭐ Recommended Action:</strong> {rec['action']}</p>
            <p><strong>📈 Lead Score:</strong> {score} points (Required: Less than 35 points)</p>
            <p><strong>📝 Instruction:</strong> {rec['instruction']}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ==================== MAIN HEADER ====================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(
        """
    <div class="main-header">
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

    model = load_model()
    if model:
        st.success("✅ Model: Active")
    else:
        st.warning("⚠️ Model: Using Rule-Based Scoring")

    st.markdown(f"**Version:** v2.0.0")
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d')}")

    st.markdown("---")
    st.markdown("### 🎯 Lead Priority Criteria")
    st.info("""
    🔴 **HIGH Priority (60+ points)**
    - Contact within 24 hours

    🟡 **MEDIUM Priority (35-59 points)**
    - Nurture campaign

    🟢 **LOW Priority (<35 points)**
    - Monitor engagement
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
        )

        contact_form = st.selectbox(
            "Contact Form",
            options=[1, 0],
            format_func=lambda x: (
                "✅ Yes, Contact Form Available" if x == 1 else "❌ No Contact Form"
            ),
        )

        services_count = st.slider(
            "Number of Services",
            min_value=0,
            max_value=10,
            value=3,
        )

    with col2:
        st.markdown("#### 📈 Company Profile")
        country_score = st.selectbox(
            "Target Country",
            options=[3, 2, 1],
            format_func=lambda x: (
                "🇺🇸 United States (Score: 3)"
                if x == 3
                else "🇨🇦 Canada (Score: 2)" if x == 2 else "🇦🇪 UAE (Score: 1)"
            ),
        )

    st.markdown("---")

    # Only button - NO score preview
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔮 Generate Prediction", use_container_width=True):
            # Calculate score and priority
            score = calculate_score(
                website_exists, contact_form, services_count, country_score
            )
            priority = get_priority_from_score(score)

            # Get recommendation based on priority
            rec = get_recommendation(priority, score)

            # Display result
            display_result_card(priority, rec, score)

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
                    "priority": priority,
                    "score": score,
                }
            )

            st.balloons()


# ==================== BATCH PROCESSING TAB ====================
elif selected == "Batch Processing":
    st.markdown("### 📁 Batch Lead Processing")

    st.info("""
    **Required CSV Columns:**
    - website_exists (0 or 1)
    - contact_form (0 or 1)
    - services_count (integer)
    - country_score (1, 2, or 3)
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

            required_cols = [
                "website_exists",
                "contact_form",
                "services_count",
                "country_score",
            ]

            if st.button("🚀 Process Batch", use_container_width=True):
                # Calculate scores and priorities
                scores = []
                priorities = []
                recommendations = []

                for idx, row in df.iterrows():
                    score = calculate_score(
                        row["website_exists"],
                        row["contact_form"],
                        row["services_count"],
                        row["country_score"],
                    )
                    priority = get_priority_from_score(score)
                    rec = get_recommendation(priority, score)

                    scores.append(score)
                    priorities.append(priority)
                    recommendations.append(rec["message"])

                df["Score"] = scores
                df["Priority"] = priorities
                df["Recommendation"] = recommendations

                st.success(f"✅ Processed {len(df)} leads successfully!")

                st.markdown("#### 📊 Results Preview")
                display_cols = ["Priority", "Recommendation", "Score"]
                if "Company Name" in df.columns:
                    display_cols = ["Company Name"] + display_cols
                st.dataframe(df[display_cols].head(10), use_container_width=True)

                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=csv,
                    file_name=f"lead_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

                # Distribution chart
                st.markdown("#### 📈 Priority Distribution")
                priority_counts = df["Priority"].value_counts()
                fig = px.pie(
                    values=priority_counts.values,
                    names=priority_counts.index,
                    title="Lead Priority Distribution",
                    color=priority_counts.index,
                    color_discrete_map={
                        "High": "#dc3545",
                        "Medium": "#ffc107",
                        "Low": "#28a745",
                    },
                    hole=0.4,
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error reading file: {e}")


# ==================== ANALYTICS DASHBOARD TAB ====================
elif selected == "Analytics Dashboard":
    st.markdown("### 📈 Analytics Dashboard")

    if "history" in st.session_state and st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)

        col1, col2, col3 = st.columns(3)

        with col1:
            high_count = len(history_df[history_df["priority"] == "High"])
            st.metric("🔴 High Priority Leads", high_count)
        with col2:
            medium_count = len(history_df[history_df["priority"] == "Medium"])
            st.metric("🟡 Medium Priority Leads", medium_count)
        with col3:
            low_count = len(history_df[history_df["priority"] == "Low"])
            st.metric("🟢 Low Priority Leads", low_count)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                history_df["priority"].value_counts().reset_index(),
                x="priority",
                y="count",
                title="Prediction Distribution",
                color="priority",
                color_discrete_map={
                    "High": "#dc3545",
                    "Medium": "#ffc107",
                    "Low": "#28a745",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
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
        st.info("No prediction history yet. Make some predictions!")


# ==================== MODEL INSIGHTS TAB ====================
elif selected == "Model Insights":
    st.markdown("### 🧠 Model Insights & Scoring Rules")

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
        st.markdown("#### 🏠 GET /health")
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
Request Body:
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
    "timestamp": "2024-06-15T10:30:00"
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
    <p>🤖 AI-Powered Lead Recommendation System | Built with Streamlit</p>
    <p style="font-size: 12px;">© 2025 LeadScout AI | Machine Learning Internship Project</p>
</div>
""",
    unsafe_allow_html=True,
)
