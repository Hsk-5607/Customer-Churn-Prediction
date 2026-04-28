import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
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
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
    }
    .churn    { background-color:#FEE2E2; color:#991B1B; border:2px solid #DC2626; }
    .no-churn { background-color:#D1FAE5; color:#065F46; border:2px solid #059669; }
</style>
""", unsafe_allow_html=True)

# ── Label mappings — must match training order exactly ────────────────────────
FREQUENT_FLYER = {"No": 0, "Yes": 1}
INCOME_CLASS   = {"High Income": 0, "Low Income": 1, "Middle Income": 2}
SOCIAL_MEDIA   = {"No": 0, "Yes": 1}
HOTEL_BOOKING  = {"No": 0, "Yes": 1}

# ── Train model once, cache for the session ───────────────────────────────────
@st.cache_resource(show_spinner="Training model… please wait.")
def train_model():
    df = pd.read_excel("Customertravel.xlsx")
    df["FrequentFlyer"] = df["FrequentFlyer"].replace("No Record", "No")
    df["FrequentFlyer"]             = df["FrequentFlyer"].map(FREQUENT_FLYER)
    df["AnnualIncomeClass"]         = df["AnnualIncomeClass"].map(INCOME_CLASS)
    df["AccountSyncedToSocialMedia"]= df["AccountSyncedToSocialMedia"].map(SOCIAL_MEDIA)
    df["BookedHotelOrNot"]          = df["BookedHotelOrNot"].map(HOTEL_BOOKING)
    X = df.drop("Target", axis=1)
    y = df["Target"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf

model = train_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("✈️ Churn Predictor")
    st.markdown("**Course:** B.Tech – Gen AI (2nd Semester)")
    st.markdown("**Student:** Harry Sijo")
    st.markdown("**Model:** Random Forest (100 trees)")
    st.divider()
    st.markdown("### Model Info")
    st.markdown("- **Accuracy:** 89.01%")
    st.markdown("- **AUC Score:** 0.9585")
    st.markdown("- **Training rows:** 763")
    st.markdown("- **Features:** 6")
    st.divider()
    st.info(
        "Predicts whether a travel customer will **churn** based on "
        "demographic and behavioral attributes using Random Forest."
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ✈️ Customer Churn Prediction")
st.markdown("#### *End-to-End ML Deployment — B.Tech Gen AI Final Project*")
st.divider()

# ── Input Form ────────────────────────────────────────────────────────────────
st.markdown("## 📋 Enter Customer Details")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 👤 Demographics")
    age           = st.slider("Age", min_value=27, max_value=38, value=32)
    annual_income = st.selectbox("Annual Income Class", list(INCOME_CLASS.keys()), index=2)

with col2:
    st.markdown("#### 🛫 Travel Behavior")
    frequent_flyer = st.selectbox("Frequent Flyer", list(FREQUENT_FLYER.keys()), index=0)
    services_opted = st.slider("Services Opted", min_value=1, max_value=6, value=3)

with col3:
    st.markdown("#### 🌐 Digital Engagement")
    social_media = st.selectbox("Account Synced to Social Media", list(SOCIAL_MEDIA.keys()), index=0)
    hotel_booked = st.selectbox("Booked Hotel or Not", list(HOTEL_BOOKING.keys()), index=0)

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Churn Risk", use_container_width=True):

    input_df = pd.DataFrame([{
        "Age":                         age,
        "FrequentFlyer":               FREQUENT_FLYER[frequent_flyer],
        "AnnualIncomeClass":           INCOME_CLASS[annual_income],
        "ServicesOpted":               services_opted,
        "AccountSyncedToSocialMedia":  SOCIAL_MEDIA[social_media],
        "BookedHotelOrNot":            HOTEL_BOOKING[hotel_booked],
    }])

    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    churn_prob  = float(probability[1])
    safe_prob   = float(probability[0])

    st.markdown("## 🎯 Prediction Result")
    r1, r2 = st.columns(2)

    with r1:
        if prediction == 1:
            st.markdown(
                '<div class="result-box churn">⚠️ HIGH CHURN RISK<br>'
                '<span style="font-size:15px">This customer is likely to churn</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-box no-churn">✅ LOW CHURN RISK<br>'
                '<span style="font-size:15px">This customer is likely to stay</span></div>',
                unsafe_allow_html=True
            )
        st.markdown("### 📊 Probability")
        st.progress(safe_prob,  text=f"Stay probability:  {safe_prob*100:.1f}%")
        st.progress(churn_prob, text=f"Churn probability: {churn_prob*100:.1f}%")

    with r2:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            [safe_prob, churn_prob],
            colors=["#0D9488", "#DC2626"],
            startangle=90,
            wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2)
        )
        ax.text(0, 0, f"{churn_prob*100:.1f}%\nChurn",
                ha="center", va="center", fontsize=14,
                fontweight="bold", color="#DC2626")
        ax.set_title("Churn Probability", fontsize=13, fontweight="bold")
        ax.legend(
            handles=[
                mpatches.Patch(color="#0D9488", label=f"Stay ({safe_prob*100:.1f}%)"),
                mpatches.Patch(color="#DC2626", label=f"Churn ({churn_prob*100:.1f}%)")
            ],
            loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=10
        )
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.markdown("### 📝 Input Summary")
    st.dataframe(pd.DataFrame({
        "Feature": ["Age", "Frequent Flyer", "Annual Income Class",
                    "Services Opted", "Social Media Sync", "Hotel Booked"],
        "Value":   [age, frequent_flyer, annual_income,
                    services_opted, social_media, hotel_booked]
    }), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 💡 Business Recommendation")
    if prediction == 1:
        st.warning(
            "**Action Required:** Consider offering a **personalized discount**, "
            "**loyalty rewards**, or a **dedicated support call** to retain this customer."
        )
    else:
        st.success(
            "**Customer is engaged!** Keep nurturing with **exclusive travel deals** "
            "and **loyalty program updates**."
        )

# ── Feature Importance (always visible) ──────────────────────────────────────
st.divider()
st.markdown("## 📈 Feature Importance")

feat_names  = ["Age", "FrequentFlyer", "AnnualIncomeClass",
               "ServicesOpted", "AccountSyncedToSocialMedia", "BookedHotelOrNot"]
importances = model.feature_importances_
sorted_idx  = np.argsort(importances)

fig2, ax2 = plt.subplots(figsize=(9, 4))
colors = ["#0D9488" if i == sorted_idx[-1] else "#5EEAD4" for i in sorted_idx]
bars   = ax2.barh(
    [feat_names[i] for i in sorted_idx],
    importances[sorted_idx],
    color=colors, edgecolor="white", height=0.6
)
for bar, val in zip(bars, importances[sorted_idx]):
    ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
             f"{val:.4f}", va="center", fontsize=10, color="#334155")
ax2.set_xlabel("Importance Score", fontsize=12)
ax2.set_facecolor("#F9FAFB")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(axis="x", alpha=0.3)
st.pyplot(fig2)
plt.close()

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#64748B;font-size:13px'>"
    "✈️ Customer Churn Predictor | B.Tech Gen AI Final Project | "
    "Harry Sijo | Built with Streamlit + scikit-learn"
    "</div>",
    unsafe_allow_html=True
)
