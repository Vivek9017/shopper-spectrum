import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =====================================================
# ✅ MUST BE FIRST STREAMLIT COMMAND
# =====================================================
st.set_page_config(page_title="Shopper Spectrum", layout="wide")

# =====================================================
# AUTO-CREATE PIVOT TABLE & SIMILARITY (ONLY IF MISSING)
# =====================================================
def build_recommendation_files():
    df = pd.read_csv("online_retail.csv")

    df = df.dropna(subset=["CustomerID", "Description", "Quantity"])

    pivot_table = pd.pivot_table(
        df,
        index="CustomerID",
        columns="Description",
        values="Quantity",
        aggfunc="sum",
        fill_value=0
    )

    similarity_matrix = cosine_similarity(pivot_table.T)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=pivot_table.columns,
        columns=pivot_table.columns
    )

    joblib.dump(pivot_table, "pivot_table.pkl")
    joblib.dump(similarity_df, "similarity_df.pkl")


def build_segmentation_files():
    df = pd.read_csv("online_retail.csv")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (df["InvoiceDate"].max() - x.max()).days,
        "InvoiceNo": "count",
        "Quantity": "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(rfm_scaled)

    segment_map = {
        0: "Champions",
        1: "Loyal Customers",
        2: "Potential Loyalists",
        3: "At Risk",
        4: "Hibernating"
    }

    joblib.dump(kmeans, "kmeans_rfm_model.pkl")
    joblib.dump(scaler, "rfm_scaler.pkl")
    joblib.dump(segment_map, "segment_map.pkl")


required_files = [
    "pivot_table.pkl",
    "similarity_df.pkl",
    "kmeans_rfm_model.pkl",
    "rfm_scaler.pkl",
    "segment_map.pkl"
]

# ✅ NOW SAFE — set_page_config already called
if not all(os.path.exists(f) for f in required_files):
    with st.spinner("🚀 Initializing models for first run..."):
        build_recommendation_files()
        build_segmentation_files()

# =====================================================
# 🔥 FULL ADVANCED 3D + GLASSMORPHIC CSS
# =====================================================
st.markdown("""
<style>

/* GLOBAL BACKGROUND WITH ANIMATED GRADIENT */
html, body {
    background: linear-gradient(-45deg, #0a0a1a, #121230, #0a1a2a, #1a0a2a);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
    min-height: 100vh;
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* FLOATING PARTICLES BACKGROUND */
body::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        radial-gradient(circle at 20% 30%, rgba(120, 119, 198, 0.1) 0%, transparent 20%),
        radial-gradient(circle at 80% 70%, rgba(255, 119, 198, 0.1) 0%, transparent 20%),
        radial-gradient(circle at 40% 80%, rgba(120, 219, 255, 0.1) 0%, transparent 20%);
    z-index: -1;
    animation: floatParticles 20s infinite linear;
}

@keyframes floatParticles {
    0% { transform: translateY(0px) rotate(0deg); }
    100% { transform: translateY(-100px) rotate(360deg); }
}

/* TITLES WITH 3D TEXT EFFECT */
h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
    letter-spacing: 1px;
    text-shadow: 
        0 1px 0 #ccc,
        0 2px 0 #c9c9c9,
        0 3px 0 #bbb,
        0 4px 0 #b9b9b9,
        0 5px 0 #aaa,
        0 6px 1px rgba(0,0,0,.1),
        0 0 5px rgba(0,0,0,.1),
        0 1px 3px rgba(0,0,0,.3),
        0 3px 5px rgba(0,0,0,.2),
        0 5px 10px rgba(0,0,0,.25),
        0 10px 10px rgba(0,0,0,.2),
        0 20px 20px rgba(0,0,0,.15);
    color: #fff;
    position: relative;
    animation: titleGlow 2s ease-in-out infinite alternate;
}

@keyframes titleGlow {
    from { text-shadow: 0 0 10px #00b3ff, 0 0 20px #00b3ff, 0 0 30px #00b3ff; }
    to { text-shadow: 0 0 20px #0080ff, 0 0 30px #0080ff, 0 0 40px #0080ff; }
}

/* 3D INPUT FIELDS WITH NEON BORDERS */
.stTextInput>div>div>input, 
.stNumberInput>div>div>input,
.stSelectbox>div>div>select {
    border-radius: 20px !important;
    padding: 16px 20px !important;
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(20px);
    border: 2px solid transparent !important;
    color: #fff !important;
    font-size: 16px;
    font-weight: 500;
    box-shadow: 
        inset 0 0 20px rgba(0, 0, 0, 0.5),
        0 8px 32px rgba(31, 38, 135, 0.37),
        0 0 30px rgba(0, 200, 255, 0.2) !important;
    transition: all 0.3s ease;
    transform: perspective(500px) translateZ(0);
}

.stTextInput>div>div>input:focus, 
.stNumberInput>div>div>input:focus {
    border: 2px solid #00e5ff !important;
    box-shadow: 
        inset 0 0 30px rgba(0, 229, 255, 0.3),
        0 0 50px rgba(0, 229, 255, 0.4),
        0 10px 40px rgba(0, 0, 0, 0.6) !important;
    transform: perspective(500px) translateZ(20px);
}

/* 3D BUTTON WITH DEPTH AND GLOW */
.stButton>button {
    background: linear-gradient(145deg, #6a11cb 0%, #2575fc 100%);
    color: white;
    border-radius: 20px;
    height: 56px;
    font-size: 18px;
    font-weight: bold;
    border: none;
    position: relative;
    overflow: hidden;
    z-index: 1;
    transform-style: preserve-3d;
    perspective: 1000px;
    box-shadow: 
        0 20px 40px rgba(0, 0, 0, 0.4),
        0 0 30px rgba(37, 117, 252, 0.5),
        inset 0 1px 0 rgba(255, 255, 255, 0.2);
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

.stButton>button::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(145deg, #2575fc 0%, #6a11cb 100%);
    z-index: -1;
    opacity: 0;
    transition: opacity 0.4s;
}

.stButton>button:hover {
    transform: 
        translateY(-8px) 
        scale(1.05) 
        rotateX(15deg) 
        rotateY(5deg);
    box-shadow: 
        0 30px 60px rgba(0, 0, 0, 0.6),
        0 0 50px rgba(106, 17, 203, 0.8),
        0 0 70px rgba(37, 117, 252, 0.6),
        inset 0 1px 0 rgba(255, 255, 255, 0.3);
}

.stButton>button:hover::before {
    opacity: 1;
}

.stButton>button:active {
    transform: translateY(-4px) scale(0.98);
    box-shadow: 
        0 15px 30px rgba(0, 0, 0, 0.3),
        0 0 30px rgba(37, 117, 252, 0.4);
}

/* 3D CARD WITH MULTI-LAYER DEPTH */
.card {
    background: linear-gradient(
        145deg,
        rgba(255, 255, 255, 0.1),
        rgba(255, 255, 255, 0.05)
    );
    padding: 30px;
    border-radius: 25px;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
    transform-style: preserve-3d;
    perspective: 1000px;
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 
        20px 20px 50px rgba(0, 0, 0, 0.5),
        -10px -10px 30px rgba(255, 255, 255, 0.05),
        inset 5px 5px 15px rgba(255, 255, 255, 0.1),
        inset -5px -5px 15px rgba(0, 0, 0, 0.5);
    transition: all 0.5s cubic-bezier(0.23, 1, 0.320, 1);
}

.card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #6a11cb, #2575fc);
    z-index: 2;
}

.card:hover {
    transform: 
        translateY(-15px) 
        rotateX(5deg) 
        rotateY(5deg) 
        scale(1.02);
    box-shadow: 
        40px 40px 80px rgba(0, 0, 0, 0.6),
        -20px -20px 40px rgba(255, 255, 255, 0.05),
        0 0 100px rgba(0, 200, 255, 0.3),
        inset 10px 10px 20px rgba(255, 255, 255, 0.1),
        inset -10px -10px 20px rgba(0, 0, 0, 0.5);
}

/* 3D RECOMMENDATION ITEMS */
.reco-item {
    background: linear-gradient(
        145deg,
        rgba(32, 38, 55, 0.8),
        rgba(20, 25, 40, 0.9)
    );
    padding: 20px 25px;
    border-radius: 18px;
    margin-bottom: 15px;
    font-weight: 600;
    font-size: 16px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    position: relative;
    overflow: hidden;
    transform-style: preserve-3d;
    box-shadow: 
        10px 10px 20px rgba(0, 0, 0, 0.4),
        -5px -5px 15px rgba(255, 255, 255, 0.05),
        0 0 30px rgba(0, 153, 255, 0.2);
    transition: all 0.3s cubic-bezier(0.23, 1, 0.320, 1);
}

.reco-item::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255, 255, 255, 0.1),
        transparent
    );
    transition: 0.5s;
}

.reco-item:hover {
    transform: 
        translateX(15px) 
        translateY(-5px) 
        scale(1.05) 
        rotateY(10deg);
    box-shadow: 
        20px 20px 40px rgba(0, 0, 0, 0.6),
        -10px -10px 20px rgba(255, 255, 255, 0.05),
        0 0 50px rgba(37, 117, 252, 0.8),
        0 0 70px rgba(106, 17, 203, 0.6);
}

.reco-item:hover::before {
    left: 100%;
}

/* 3D SEGMENT BADGE WITH FLOATING EFFECT */
.segment-badge {
    display: inline-block;
    padding: 20px 35px;
    border-radius: 35px;
    font-size: 24px;
    font-weight: bold;
    color: white;
    background: linear-gradient(135deg, #00c6ff, #0072ff, #6a11cb);
    background-size: 200% 200%;
    position: relative;
    overflow: hidden;
    animation: 
        gradientMove 3s ease infinite,
        floatBadge 3s ease-in-out infinite;
    box-shadow: 
        0 20px 40px rgba(0, 0, 0, 0.6),
        0 0 40px rgba(0, 114, 255, 0.6),
        0 0 60px rgba(106, 17, 203, 0.4),
        inset 0 2px 0 rgba(255, 255, 255, 0.3);
    transform-style: preserve-3d;
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes floatBadge {
    0%, 100% { transform: translateY(0px) rotateX(0deg); }
    50% { transform: translateY(-10px) rotateX(5deg); }
}

/* 3D TABS STYLING */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255, 255, 255, 0.05);
    padding: 8px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 
        inset 0 0 20px rgba(0, 0, 0, 0.5),
        0 10px 30px rgba(0, 0, 0, 0.3);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 15px;
    padding: 12px 24px;
    background: rgba(255, 255, 255, 0.1);
    border: none;
    color: rgba(255, 255, 255, 0.7);
    transition: all 0.3s ease;
    transform: perspective(500px) translateZ(0);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6a11cb, #2575fc);
    color: white;
    box-shadow: 
        0 10px 25px rgba(0, 0, 0, 0.4),
        0 0 30px rgba(37, 117, 252, 0.5);
    transform: perspective(500px) translateZ(20px);
}

/* SCROLLBAR STYLING */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(45deg, #6a11cb, #2575fc);
    border-radius: 10px;
    box-shadow: inset 0 0 6px rgba(0, 0, 0, 0.3);
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(45deg, #2575fc, #6a11cb);
}

/* CAPTION STYLING */
.st-caption {
    font-size: 1.1em;
    color: #a0a0ff !important;
    text-shadow: 0 0 10px rgba(160, 160, 255, 0.5);
}

/* RESPONSIVE DESIGN */
@media (max-width: 768px) {
    .card {
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .segment-badge {
        padding: 15px 25px;
        font-size: 20px;
    }
    
    .reco-item {
        padding: 15px 20px;
    }
}

/* SUCCESS, ERROR, WARNING MESSAGE STYLING */
.stAlert {
    border-radius: 15px !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    box-shadow: 
        0 10px 25px rgba(0, 0, 0, 0.4),
        0 0 20px rgba(0, 200, 255, 0.2) !important;
}

/* SPINNER STYLING */
.stSpinner > div {
    border-color: #00e5ff transparent transparent transparent !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_models():
    return (
        joblib.load("kmeans_rfm_model.pkl"),
        joblib.load("rfm_scaler.pkl"),
        joblib.load("segment_map.pkl")
    )

@st.cache_data
def load_recommendation_data():
    return (
        joblib.load("pivot_table.pkl"),
        joblib.load("similarity_df.pkl")
    )

kmeans, scaler, segment_map = load_models()
pivot_table, similarity_df = load_recommendation_data()

# =====================================================
# RECOMMENDATION FUNCTION
# =====================================================
def recommend_products(product_name, similarity_df, top_n=5):
    if product_name not in similarity_df.columns:
        return None
    scores = similarity_df[product_name].sort_values(ascending=False)
    return scores.iloc[1:top_n + 1].index.tolist()

# =====================================================
# ENHANCED 3D UI
# =====================================================
# HEADER WITH 3D EFFECT
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.title("🛒 Shopper Spectrum")
    st.caption("✨ Customer Segmentation & Product Recommendation System with Advanced 3D UI")
    
    # Decorative element
    st.markdown("""
    <div style="text-align: center; margin: 20px 0;">
        <div style="display: inline-block; padding: 10px 30px; background: linear-gradient(90deg, transparent, rgba(0, 200, 255, 0.1), transparent); border-radius: 20px;">
            <span style="color: #00e5ff;">⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯</span>
            <span style="color: white; padding: 0 20px;">PREMIUM ANALYTICS</span>
            <span style="color: #00e5ff;">⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# TABS WITH 3D STYLE
tab1, tab2 = st.tabs(["🎯 **PRODUCT RECOMMENDATION**", "📊 **CUSTOMER SEGMENTATION**"])

# ==================================
# 🎯 PRODUCT RECOMMENDATION MODULE
# ==================================
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Header with icon
    col_title, col_icon = st.columns([4, 1])
    with col_title:
        st.subheader("🔍 Product Recommendation System")
    with col_icon:
        st.markdown("<div style='text-align: right; font-size: 2em;'>🚀</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Input section with label
    st.markdown("<p style='color: #a0a0ff; font-size: 1.1em; margin-bottom: 10px;'>Enter a product name to get similar recommendations:</p>", unsafe_allow_html=True)
    
    product_name = st.text_input("", placeholder="e.g., 'WHITE HANGING HEART T-LIGHT HOLDER'", label_visibility="collapsed")
    
    col_btn, col_space = st.columns([2, 3])
    with col_btn:
        if st.button("🚀 Get Recommendations", use_container_width=True):
            if product_name.strip() == "":
                st.warning("⚠️ Please enter a product name.")
            else:
                with st.spinner("🔮 Finding perfect recommendations..."):
                    recommendations = recommend_products(product_name, similarity_df)
                    if recommendations is None:
                        st.error("❌ Product not found. Please check spelling.")
                    else:
                        st.success(f"✨ Top 5 Recommended Products for **'{product_name}'**")
                        
                        # Display recommendations with enhanced styling
                        for idx, product in enumerate(recommendations, 1):
                            emoji = ["🥇", "🥈", "🥉", "🎯", "💎"][idx-1]
                            # Calculate similarity score
                            similarity_score = similarity_df[product_name][product] if product_name in similarity_df.columns and product in similarity_df.index else 0.0
                            st.markdown(
                                f"""
                                <div class="reco-item">
                                    <div style="display: flex; align-items: center;">
                                        <div style="font-size: 1.5em; margin-right: 15px;">{emoji}</div>
                                        <div>
                                            <div style="font-size: 1.2em; color: #00e5ff;">{product}</div>
                                            <div style="font-size: 0.9em; color: #aaa; margin-top: 5px;">Similarity Score: {similarity_score:.2f}</div>
                                        </div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================
# 📊 CUSTOMER SEGMENTATION MODULE
# ==================================
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Header with icon
    col_title2, col_icon2 = st.columns([4, 1])
    with col_title2:
        st.subheader("🧠 Customer Segmentation (RFM-Based)")
    with col_icon2:
        st.markdown("<div style='text-align: right; font-size: 2em;'>🎯</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Input section in 3 columns
    st.markdown("<p style='color: #a0a0ff; font-size: 1.1em; margin-bottom: 20px;'>Enter customer RFM metrics:</p>", unsafe_allow_html=True)
    
    col_r, col_f, col_m = st.columns(3)
    
    with col_r:
        recency = st.number_input(
            "📅 Recency (Days)", 
            min_value=0, 
            step=1,
            value=30,
            help="How many days since the customer's last purchase"
        )
    
    with col_f:
        frequency = st.number_input(
            "🔄 Frequency", 
            min_value=0, 
            step=1,
            value=5,
            help="Total number of purchases made by the customer"
        )
    
    with col_m:
        monetary = st.number_input(
            "💰 Monetary Value", 
            min_value=0.0, 
            step=10.0,
            value=1000.0,
            help="Total amount spent by the customer"
        )
    
    # Predict button
    if st.button("🔮 Predict Customer Segment", use_container_width=True):
        with st.spinner("🧩 Analyzing customer profile..."):
            rfm_scaled = scaler.transform([[recency, frequency, monetary]])
            cluster = kmeans.predict(rfm_scaled)[0]
            segment = segment_map.get(cluster, "Unknown")
            
            # Display segment with enhanced styling
            st.markdown("<div style='text-align: center; margin: 40px 0;'>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="segment-badge">
                    🎭 Customer Segment: <span style="color: #ffd700; text-shadow: 0 0 10px #ffd700;">{segment}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Add segment insights
            segment_insights = {
                "Champions": "🎯 High-value customers who buy recently and frequently",
                "Loyal Customers": "💎 Frequent buyers but not recent",
                "Potential Loyalists": "🌟 Recent customers with good frequency",
                "At Risk": "🚨 Spent big but haven't purchased lately",
                "Hibernating": "🛌 Last purchase long back and low frequency"
            }
            
            if segment in segment_insights:
                st.info(f"**Insight:** {segment_insights[segment]}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add metrics visualization
            col_v1, col_v2, col_v3 = st.columns(3)
            with col_v1:
                st.metric("Recency", f"{recency} days", delta="Lower is better" if recency < 30 else "Higher needs attention", delta_color="inverse")
            with col_v2:
                st.metric("Frequency", frequency, delta="Higher is better")
            with col_v3:
                st.metric("Monetary", f"${monetary:,.2f}", delta="Higher is better")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================
# FOOTER WITH 3D EFFECT
# ==================================
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 20px; background: rgba(0, 0, 0, 0.2); border-radius: 20px; border-top: 1px solid rgba(0, 200, 255, 0.2);">
    <p style="color: #a0a0ff; font-size: 0.9em;">
        Shopper Spectrum v2.0 • Advanced 3D Analytics Dashboard<br>
        Powered by Machine Learning • Real-time Recommendations
    </p>
</div>
""", unsafe_allow_html=True)

# ==================================
# SIDEBAR WITH INFO
# ==================================
with st.sidebar:
    st.markdown("### 🚀 Quick Tips")
    st.markdown("• Enter exact product names for best results")
    st.markdown("• Lower Recency = More recent purchase")
    st.markdown("• Higher Frequency = More purchases")
    st.markdown("• Higher Monetary = More spending")
    st.markdown("</div>", unsafe_allow_html=True)