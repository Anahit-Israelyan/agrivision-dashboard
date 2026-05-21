import streamlit as st
import numpy as np
from PIL import Image

# AgriVision Backend Modules
from src.config import REGIONAL_DEFAULTS, SUPPORTED_REGIONS
from src.analytics import FieldAnalytics
from src.inference import get_model, run_inference
from src.report import generate_pdf_report

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriVision Platform",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Custom Advanced CSS for Pixel-Perfect Layout
# ---------------------------------------------------------------------------
with open("style.css", "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Global State Management
# ---------------------------------------------------------------------------
if "counts" not in st.session_state:
    st.session_state.counts = []
    st.session_state.annotated_img = None
    st.session_state.original_img = None


# ---------------------------------------------------------------------------
# Model Caching
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Warming up neural engine...")
def load_agrivision_model():
    return get_model("iteration_2_tuned.pt")


# ---------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <span class="sidebar-header-icon">⚙️</span>
        <span class="sidebar-header-text">ENGINE SETTINGS</span>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload Field Drone Imagery (JPG/PNG)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    
    st.write("")
    conf_thresh = st.slider("Confidence Threshold", 0.05, 0.95, 0.45, 0.05)
    iou_thresh = st.slider("IoU Threshold (NMS)", 0.10, 0.90, 0.50, 0.05)
    
    st.write("")
    st.markdown('<p class="arch-text">Architecture: YOLOv8s Multiscale</p>', unsafe_allow_html=True)
    st.markdown('<p class="arch-text">Target Resolution: 1024px</p>', unsafe_allow_html=True)
    st.write("")

    # --- COUNTRY/PARAM OVERRIDE ---
    with st.expander("🌍 Agronomic Variables", expanded=False):
        selected_region = st.selectbox(
            "Load Presets",
            options=SUPPORTED_REGIONS,
            index=SUPPORTED_REGIONS.index("Custom Setup")
        )
        reg_def = REGIONAL_DEFAULTS[selected_region]

        tgw = st.number_input("TGW (g)", value=float(reg_def["tgw_grams"]), step=1.0)
        gph = st.number_input("Grains/Head", value=int(reg_def["grains_per_head"]), step=1)
        price = st.number_input("Price/Tonne", value=float(reg_def["price_per_tonne_usd"]), step=5.0)
        currency = st.text_input("Currency", value=str(reg_def["currency"]))
        
    st.write("")
    trigger_run = st.button("🚀 Analyze Biomass", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-icon">🌾</div>
    <div class="hero-text-container">
        <h1 class="hero-title">AgriVision Platform</h1>
        <p class="hero-subtitle">High-Accuracy Wheat Head Detection Engine</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Main container for statistics (will populate if we have results)
stat_placeholder = st.container()

# ---------------------------------------------------------------------------
# EXECUTION LOOP
# ---------------------------------------------------------------------------
if trigger_run and uploaded_files:
    model = load_agrivision_model()
    counts = []
    
    with st.spinner("Processing spatial data through neural network..."):
        for file in uploaded_files:
            img = Image.open(file).convert("RGB")
            st.session_state.original_img = img  
            
            count, ann_img = run_inference(
                model=model,
                image=img,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                img_size=1024
            )
            counts.append(count)
            st.session_state.annotated_img = ann_img
            
        st.session_state.counts = counts
        st.rerun()

# ---------------------------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------------------------
if st.session_state.counts:
    counts = st.session_state.counts
    engine = FieldAnalytics(counts)
    cv_pct = engine.calculate_cv()
    health = FieldAnalytics.get_health_status(cv_pct)
    mean_heads = float(np.mean(counts))
    
    yield_est = FieldAnalytics.estimate_yield(mean_heads, tgw=tgw, grains_per_head=gph)
    revenue = yield_est * price
    
    # 1. Giant Stat Box
    with stat_placeholder:
        st.markdown(f"""
        <div class="stat-box-container">
            <p class="stat-number">{sum(counts):,}</p>
            <p class="stat-label">TOTAL WHEAT HEADS DETECTED</p>
            <div class="stat-glow-line"></div>
            <div class="stat-success"><span>✨</span> Spatial analysis completed successfully</div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")

    # 2. Side-by-side Images
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown('<div class="panel-container">', unsafe_allow_html=True)
        st.markdown('<p class="panel-header">Raw Telemetry</p>', unsafe_allow_html=True)
        st.image(st.session_state.original_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_img2:
        st.markdown('<div class="panel-container">', unsafe_allow_html=True)
        st.markdown('<p class="panel-header">Neural Detection Overlay</p>', unsafe_allow_html=True)
        st.image(st.session_state.annotated_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Financials & Metrics
    st.write("")
    st.markdown('<div class="panel-container">', unsafe_allow_html=True)
    st.markdown('<p class="panel-header">Agronomic Analytics</p>', unsafe_allow_html=True)
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Estimated Yield", f"{yield_est:.2f} t/ha")
    m_col2.metric("Spatial Uniformity (CV %)", f"{cv_pct:.1f}%", 
                  delta=health["message"], delta_color="inverse")
    m_col3.metric(f"Projected Revenue ({currency})", f"{revenue:,.2f}")
    
    with m_col4:
        pdf_bytes = generate_pdf_report(
            counts=counts, mean_density=mean_heads, cv_pct=cv_pct, 
            yield_est=yield_est, revenue=revenue, health_msg=health["message"],
            currency=currency, conf_thresh=conf_thresh, iou_thresh=iou_thresh
        )
        st.download_button(
            label="📄 Export Briefing",
            data=pdf_bytes,
            file_name="AgriVision_Briefing.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Empty State Images
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown('<p class="panel-header">Raw Telemetry</p>', unsafe_allow_html=True)
        st.markdown('<div class="img-placeholder">Awaiting Upload...</div>', unsafe_allow_html=True)
    with col_img2:
        st.markdown('<p class="panel-header">Neural Detection Overlay</p>', unsafe_allow_html=True)
        st.markdown('<div class="img-placeholder">Awaiting Execution...</div>', unsafe_allow_html=True)
