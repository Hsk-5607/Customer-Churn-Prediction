import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F0F4F8; }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #0D9488, #065A82);
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 12px;
        border: none;
        border-radius: 8px;
    }
    .stButton>button:hover { opacity: 0.92; }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
    }
    .churn { background-color: #FEE2E2; color: #991B1B; border: 2px solid #DC2626; }
    .no-churn { background-color: #D1FAE5; color: #065F46; border: 2px solid #059669; }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #0D9488;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ── Label Mappings (must match training) ─────────────────────────────────────
FREQUENT_FLYER   = {"No": 0, "Yes": 1}
INCOME_CLASS     = {"High Income": 0, "Low Income": 1, "Middle Income": 2}
SOCIAL_MEDIA     = {"No": 0, "Yes": 1}
HOTEL_BOOKING    = {"No": 0, "Yes": 1}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/airplane-mode-on.png", width=80)
    st.title("✈️ Churn Predictor")
    st.markdown("**Course:** B.Tech – Gen AI (2nd Semester)")
    st.markdown("**Student:** Rishabh Bheda")
    st.markdown("**Model:** Random Forest (100 trees)")
    st.divider()
    st.markdown("### Model Info")
    st.markdown("- **Accuracy:** 89.01%")
    st.markdown("- **AUC Score:** 0.9585")
    st.markdown("- **Training rows:** 763")
    st.markdown("- **Features used:** 6")
    st.divider()
    st.markdown("### About")
    st.info(
        "This app predicts whether a travel customer will **churn** "
        "based on their demographic and behavioral attributes using a "
        "trained Random Forest model."
    )

# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown("# ✈️ Customer Churn Prediction")
st.markdown("#### *End-to-End ML Deployment — B.Tech Gen AI Final Project*")
st.divider()

# ── Input Form ───────────────────────────────────────────────────────────────
st.markdown("## 📋 Enter Customer Details")
st.markdown("Fill in the customer information below and click **Predict** to see the churn risk.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 👤 Demographics")
    age = st.slider("Age", min_value=27, max_value=38, value=32,
                    help="Customer age (dataset range: 27–38)")
    annual_income = st.selectbox(
        "Annual Income Class",
        options=list(INCOME_CLASS.keys()),
        index=2,
        help="Low / Middle / High Income"
    )

with col2:
    st.markdown("#### 🛫 Travel Behavior")
    frequent_flyer = st.selectbox(
        "Frequent Flyer",
        options=list(FREQUENT_FLYER.keys()),
        index=0,
        help="Is the customer a frequent flyer?"
    )
    services_opted = st.slider(
        "Services Opted",
        min_value=1, max_value=6, value=3,
        help="Number of additional services subscribed (1–6)"
    )

with col3:
    st.markdown("#### 🌐 Digital Engagement")
    social_media = st.selectbox(
        "Account Synced to Social Media",
        options=list(SOCIAL_MEDIA.keys()),
        index=0,
        help="Has the customer linked their account to social media?"
    )
    hotel_booked = st.selectbox(
        "Booked Hotel or Not",
        options=list(HOTEL_BOOKING.keys()),
        index=0,
        help="Has the customer booked a hotel through the platform?"
    )

st.divider()

# ── Predict Button ────────────────────────────────────────────────────────────
predict_clicked = st.button("🔍 Predict Churn Risk", use_container_width=True)

if predict_clicked:
    # Build input dataframe — column order must match training
    input_data = pd.DataFrame([{
        "Age": age,
        "FrequentFlyer": FREQUENT_FLYER[frequent_flyer],
        "AnnualIncomeClass": INCOME_CLASS[annual_income],
        "ServicesOpted": services_opted,
        "AccountSyncedToSocialMedia": SOCIAL_MEDIA[social_media],
        "BookedHotelOrNot": HOTEL_BOOKING[hotel_booked],
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    churn_prob  = probability[1]
    safe_prob   = probability[0]

    st.markdown("## 🎯 Prediction Result")
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if prediction == 1:
            st.markdown(
                f'<div class="result-box churn">⚠️ HIGH CHURN RISK<br>'
                f'<span style="font-size:15px">This customer is likely to churn</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box no-churn">✅ LOW CHURN RISK<br>'
                f'<span style="font-size:15px">This customer is likely to stay</span></div>',
                unsafe_allow_html=True
            )

        st.markdown("### 📊 Probability Breakdown")
        st.progress(float(safe_prob), text=f"Stay Probability: {safe_prob*100:.1f}%")
        st.progress(float(churn_prob), text=f"Churn Probability: {churn_prob*100:.1f}%")

    with res_col2:
        # Donut chart
        fig, ax = plt.subplots(figsize=(4, 4))
        colors = ['#0D9488', '#DC2626']
        wedges, _ = ax.pie(
            [safe_prob, churn_prob],
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2)
        )
        ax.text(0, 0, f"{churn_prob*100:.1f}%\nChurn", ha='center', va='center',
                fontsize=14, fontweight='bold', color='#DC2626')
        ax.set_title("Churn Probability", fontsize=13, fontweight='bold', pad=10)
        legend_elements = [
            mpatches.Patch(color='#0D9488', label=f'Stay ({safe_prob*100:.1f}%)'),
            mpatches.Patch(color='#DC2626', label=f'Churn ({churn_prob*100:.1f}%)')
        ]
        ax.legend(handles=legend_elements, loc='lower center',
                  bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=10)
        st.pyplot(fig)
        plt.close()

    # Input summary
    st.divider()
    st.markdown("### 📝 Input Summary")
    summary_df = pd.DataFrame({
        "Feature": ["Age", "Frequent Flyer", "Annual Income Class",
                    "Services Opted", "Social Media Sync", "Hotel Booked"],
        "Value": [age, frequent_flyer, annual_income,
                  services_opted, social_media, hotel_booked]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Business recommendation
    st.divider()
    st.markdown("### 💡 Business Recommendation")
    if prediction == 1:
        st.warning(
            "**Action Required:** This customer shows signs of churn. "
            "Consider offering a **personalized discount**, **loyalty rewards**, or a "
            "**dedicated support call** to improve retention."
        )
    else:
        st.success(
            "**Customer is engaged!** Continue nurturing this relationship with "
            "**exclusive travel deals** and **loyalty program updates** to maintain satisfaction."
        )

# ── Feature Importance Chart (always visible) ──────────────────────────────
st.divider()
st.markdown("## 📈 Feature Importance (Random Forest)")
st.markdown("These are the features the model considers most important when predicting churn:")

feat_names  = ["Age", "FrequentFlyer", "AnnualIncomeClass",
               "ServicesOpted", "AccountSyncedToSocialMedia", "BookedHotelOrNot"]
importances = model.feature_importances_
sorted_idx  = np.argsort(importances)

fig2, ax2 = plt.subplots(figsize=(9, 4))
colors = ['#0D9488' if i == sorted_idx[-1] else '#5EEAD4' for i in sorted_idx]
bars = ax2.barh([feat_names[i] for i in sorted_idx],
                importances[sorted_idx],
                color=colors, edgecolor='white', height=0.6)
for bar, val in zip(bars, importances[sorted_idx]):
    ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=10, color='#334155')
ax2.set_xlabel("Importance Score", fontsize=12)
ax2.set_facecolor('#F9FAFB')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='x', alpha=0.3)
st.pyplot(fig2)
plt.close()

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#64748B; font-size:13px'>"
    "✈️ Customer Churn Predictor | B.Tech Gen AI Final Project | "
    "Rishabh Bheda | Built with Streamlit + scikit-learn"
    "</div>",
    unsafe_allow_html=True
)
