import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
from fpdf import FPDF
from datetime import datetime

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="AeroGuard AI | Enterprise",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. HIGH-TECH AEROSPACE CSS ---
st.markdown("""
    <style>
    /* GLOBAL THEME: Deep Space */
    .stApp { 
        background-color: #0B1120; 
        color: #F1F5F9; 
    }
    
    /* --- BUTTONS: LIQUID BLUE FILL --- */
    .stButton > button, .stDownloadButton > button {
        position: relative;
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #38BDF8;
        border-radius: 6px;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 0.6rem 1rem;
        overflow: hidden;
        z-index: 1;
        transition: all 0.3s ease-in-out;
        width: 100%;
        text-transform: uppercase;
        font-size: 14px;
    }

    .stButton > button::before, .stDownloadButton > button::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 0%;
        height: 100%;
        background: linear-gradient(90deg, #0EA5E9, #2563EB); 
        z-index: -1;
        transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        color: white;
        border-color: #2563EB;
        box-shadow: 0 0 15px rgba(14, 165, 233, 0.4);
    }

    .stButton > button:hover::before, .stDownloadButton > button:hover::before {
        width: 100%;
    }
    
    .stButton > button:active, .stDownloadButton > button:active {
        transform: scale(0.96);
    }
    
    /* --- CARDS: HOVER GLOW EFFECT --- */
    .metric-card, .winner-card {
        position: relative;
        background-color: #162031;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        overflow: hidden;
        z-index: 1;
        transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card::before, .winner-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 0%;
        height: 100%;
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.1), rgba(37, 99, 235, 0.1));
        z-index: -1;
        transition: width 0.4s ease;
    }

    .metric-card:hover, .winner-card:hover {
        transform: translateY(-5px);
        border-color: #38BDF8;
        box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.3);
    }

    .metric-card:hover::before, .winner-card:hover::before {
        width: 100%;
    }
    
    .winner-card { border: 1px solid #F59E0B; }
    .winner-card:hover { border-color: #FBBF24; box-shadow: 0 0 20px rgba(245, 158, 11, 0.2); }
    
    .metric-value { color: #F1F5F9; font-size: 32px; font-weight: 700; margin: 8px 0; }
    .metric-title { color: #94A3B8; font-size: 12px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase;}
    
    .winner-badge {
        color: #F59E0B; font-weight: 700; font-size: 11px;
        background: rgba(245, 158, 11, 0.1); padding: 4px 10px; border-radius: 4px;
        display: inline-block; margin-bottom: 8px; border: 1px solid rgba(245, 158, 11, 0.2);
    }

    /* --- CENTERED ALIGNMENT --- */
    .centered-status {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 24px;
        background-color: #162031;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        height: 100%; /* Full height for alignment */
    }
    
    /* --- RED SIREN ANIMATION --- */
    @keyframes pulse-red {
        0% { box-shadow: inset 0 0 0 0px rgba(220, 38, 38, 0.5); }
        50% { box-shadow: inset 0 0 50px 20px rgba(220, 38, 38, 0.8); border-color: #ef4444; }
        100% { box-shadow: inset 0 0 0 0px rgba(220, 38, 38, 0.5); }
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    scaler = joblib.load('scaler.gz')
    models = {
        "Simple LSTM": tf.keras.models.load_model("Simple_LSTM.h5"),
        "Stacked LSTM": tf.keras.models.load_model("Stacked_LSTM.h5"),
        "Bi-Directional LSTM": tf.keras.models.load_model("Bi-Directional_LSTM.h5")
    }
    return scaler, models

@st.cache_data
def load_test_data():
    cols = ['engine','cycle'] + [f'op_{i}' for i in range(1,4)] + [f'sensor_{i}' for i in range(1,22)]
    test_df = pd.read_csv("test_FD001.txt", sep=r"\s+", header=None, names=cols)
    true_rul = pd.read_csv("RUL_FD001.txt", header=None, names=['True_RUL'])
    return test_df, true_rul

try:
    scaler, models = load_resources()
    test_df, true_rul = load_test_data()
except Exception as e:
    st.error(f"❌ Error: {e}. Ensure .h5, .gz, and .txt files are in the folder.")
    st.stop()

features = [c for c in test_df.columns if c.startswith('sensor_') or c.startswith('op_')]
RUL_CAP = 125
WINDOW_SIZE = 50

# --- 4. HELPER: PREPARE INPUT ---
def prepare_input(data_list):
    input_arr = np.array(data_list).reshape(1, -1)
    scaled_arr = scaler.transform(input_arr)
    return np.repeat(scaled_arr, WINDOW_SIZE, axis=0).reshape(1, WINDOW_SIZE, 24)

# --- 5. PDF REPORT GENERATOR ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'AeroGuard AI - Diagnostic Report', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf(rul, status, sensor_data):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 8, txt=f"Predicted RUL: {rul:.2f} Cycles", ln=True)
    status_color = (255, 0, 0) if "Critical" in status else (0, 128, 0)
    pdf.set_text_color(*status_color)
    pdf.cell(0, 8, txt=f"Status: {status}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, txt="Sensor Telemetry Scan:", ln=True)
    pdf.set_font("Arial", size=9)
    row_height = 6 
    for i in range(3): 
        pdf.cell(0, row_height, txt=f"Op Setting {i+1}: {sensor_data[i]:.6f}", border=1, ln=True)
    sensor_vals = sensor_data[3:]
    for i, val in enumerate(sensor_vals):
        pdf.cell(0, row_height, txt=f"Sensor {i+1}: {val:.4f}", border=1, ln=True)
    return pdf.output(dest='S').encode('latin-1')

# --- 6. SIDEBAR NAVIGATION ---
st.sidebar.title("✈️ AeroGuard AI")
page = st.sidebar.radio("Navigation", ["Dashboard Overview", "Live Diagnostics", "What-If Simulation"])

# --- BASELINE DEFAULTS ---
BASE_VALUES = [
    -0.0007, -0.0004, 100.0, 518.67, 641.82, 1589.70, 1400.60, 14.62, 21.61, 554.36, 
    2388.06, 9046.19, 1.30, 47.47, 521.66, 2388.02, 8138.62, 8.4195, 0.03, 392, 
    2388, 100.00, 39.06, 23.4190 
]

# --- PAGE 1: DASHBOARD OVERVIEW ---
if page == "Dashboard Overview":
    st.title("📊 Fleet Analytics")
    st.subheader("🏆 Model Architecture Performance")
    
    metrics = { "Simple LSTM": 16.42, "Stacked LSTM": 15.85, "Bi-Directional LSTM": 16.10 }
    best_score = min(metrics.values())
    
    cols = st.columns(3)
    for i, (name, rmse) in enumerate(metrics.items()):
        with cols[i]:
            if rmse == best_score:
                st.markdown(f"""<div class="winner-card"><div class="winner-badge">RECOMMENDED</div><div class="metric-title">{name}</div><div class="metric-value">{rmse}</div><div style="color: #94A3B8; font-size: 11px;">RMSE ACCURACY</div></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="metric-card"><div class="metric-title">{name}</div><div class="metric-value">{rmse}</div><div style="color: #94A3B8; font-size: 11px;">RMSE ACCURACY</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📈 Predictive Analysis")
    
    c_sel, c_btn = st.columns([3, 1])
    with c_sel: model_choice = st.selectbox("Architecture Selection", list(models.keys()), index=1)
    with c_btn: 
        st.write(""); st.write("") 
        generate_btn = st.button("Run Analysis")
    
    if generate_btn:
        with st.spinner("Processing..."):
            seq_array = []
            engine_ids = test_df['engine'].unique()[:50] 
            for id in engine_ids:
                mask = test_df['engine'] == id
                if len(test_df[mask]) >= WINDOW_SIZE:
                    seq_array.append(test_df[mask][features].values[-WINDOW_SIZE:])
            X_test = np.array(seq_array)
            N, T, F = X_test.shape
            X_scaled = scaler.transform(X_test.reshape(N*T, F)).reshape(N, T, F)
            model = models[model_choice]
            preds = model.predict(X_scaled, verbose=0).flatten() * RUL_CAP
            actuals = true_rul['True_RUL'].values[:len(preds)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=actuals, mode='lines', name='Actual RUL', line=dict(color='#0EA5E9', width=2)))
            fig.add_trace(go.Scatter(y=preds, mode='lines', name='Predicted RUL', line=dict(color='#F59E0B', width=2, dash='dash')))
            fig.update_layout(
                xaxis_title="Engine Unit", yaxis_title="Remaining Cycles",
                plot_bgcolor='#0B1120', paper_bgcolor='#0B1120',
                font=dict(color='#E2E8F0'), hovermode="x unified", height=450,
                xaxis=dict(showgrid=True, gridcolor='#1E293B'),
                yaxis=dict(showgrid=True, gridcolor='#1E293B')
            )
            st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: LIVE DIAGNOSTICS (WITH RADAR CHART) ---
elif page == "Live Diagnostics":
    st.title("🛠️ Manual Engine Check")
    
    with st.expander("Sensor Input Panel (24 Parameters)", expanded=True):
        c1, c2, c3 = st.columns(3)
        inputs = []
        for i in range(24):
            col = [c1, c2, c3][i % 3]
            label = f"Sensor {i-2}" if i > 2 else f"Op Setting {i+1}"
            val = col.number_input(label, value=BASE_VALUES[i], format="%.4f", key=f"manual_{i}")
            inputs.append(val)
            
    st.markdown("<br>", unsafe_allow_html=True)
    c_left, c_mid, c_right = st.columns([1, 2, 1])
    with c_mid: run_diag = st.button("Initialize Diagnostics", type="primary")
    
    if run_diag:
        model = models["Stacked LSTM"]
        data = prepare_input(inputs)
        rul = float(model.predict(data, verbose=0)[0][0] * RUL_CAP)
        
        if rul < 50: status, color, status_icon = "CRITICAL FAILURE IMMINENT", "#ef4444", "⚠️"
        elif rul < 80: status, color, status_icon = "MAINTENANCE REQUIRED", "#f59e0b", "🛠️"
        else: status, color, status_icon = "SYSTEM OPTIMAL", "#10b981", "✅"
        
        # --- NEW LAYOUT: STATUS + RADAR CHART ---
        # Using [1, 2] ratio to keep status card tight and radar chart spacious
        col_status, col_radar = st.columns([1, 2])
        
        with col_status:
            st.markdown(f"""
            <div class="centered-status" style="border-left: 6px solid {color};">
                <h2 style="color: {color}; margin:0; font-size: 24px; letter-spacing: 1px;">{status_icon} {status}</h2>
                <p style="font-size: 60px; margin: 10px 0; font-weight: 800; color: #F8FAFC;">{rul:.1f} <span style="font-size:20px; color:#94A3B8;">Cycles</span></p>
                <p style="color: #94A3B8; font-size: 14px;">Estimated Remaining Useful Life (RUL)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Download Button (Margined & Centered via Container Width)
            st.markdown("<br>", unsafe_allow_html=True)
            pdf_bytes = generate_pdf(rul, status, inputs)
            
            # use_container_width=True makes the button fill the column width
            # matching the card above it perfectly
            st.download_button("Download Report (PDF)", pdf_bytes, "engine_report.pdf", "application/pdf", use_container_width=True)

        with col_radar:
            # RADAR CHART LOGIC
            sensors_baseline = BASE_VALUES[3:] # Sensors 1-21
            sensors_current = inputs[3:]
            
            # Calculate Ratio (Handle div by zero)
            ratios = []
            for c, b in zip(sensors_current, sensors_baseline):
                ratios.append(c / b if b != 0 else 0)
            
            sensor_labels = [f"S{i}" for i in range(1, 22)]
            
            fig_radar = go.Figure()
            
            # 1. Nominal Line (The Perfect Circle at 1.0)
            fig_radar.add_trace(go.Scatterpolar(
                r=[1] * len(sensor_labels),
                theta=sensor_labels,
                fill=None,
                name='Nominal Baseline',
                line_color='rgba(255, 255, 255, 0.3)',
                line_dash='dash'
            ))
            
            # 2. Current Scan
            fig_radar.add_trace(go.Scatterpolar(
                r=ratios,
                theta=sensor_labels,
                fill='toself',
                name='Current Scan',
                line_color=color # Matches the status color (Red/Green/Orange)
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0.8, 1.2], showticklabels=False, gridcolor='#334155'),
                    angularaxis=dict(tickfont=dict(size=10, color='#94A3B8'), rotation=90, direction='clockwise')
                ),
                showlegend=True,
                legend=dict(font=dict(color='#E2E8F0')),
                paper_bgcolor='#0B1120',
                plot_bgcolor='#0B1120',
                margin=dict(t=30, b=30, l=30, r=30),
                height=350,
                title=dict(text="Root Cause Analysis (Deviation from Baseline)", font=dict(size=14, color='#94A3B8'), x=0.5)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

# --- PAGE 3: WHAT-IF SIMULATION ---
elif page == "What-If Simulation":
    st.title("🧪 Sensitivity Analysis")
    st.info("Adjust parameters to simulate component degradation.")
    
    if "sim_initialized" not in st.session_state:
        for i, val in enumerate(BASE_VALUES): st.session_state[f"sim_{i}"] = val
        st.session_state["sim_initialized"] = True

    def reset_sliders():
        for i, val in enumerate(BASE_VALUES): st.session_state[f"sim_{i}"] = val

    col_reset, _, _ = st.columns([1, 2, 2])
    with col_reset: st.button("Reset Parameters", on_click=reset_sliders)

    col_main, col_res = st.columns([3, 1])
    sim_inputs = []

    with col_main:
        st.subheader("Parameter Controls")
        sc1, sc2, sc3 = st.columns(3)
        for i in range(24):
            curr_col = [sc1, sc2, sc3][i % 3]
            label = f"Sensor {i-2}" if i > 2 else f"Op Setting {i+1}"
            
            # Smart Ranges
            min_v = float(BASE_VALUES[i] * 0.7) if BASE_VALUES[i] > 0 else float(BASE_VALUES[i] * 1.3)
            max_v = float(BASE_VALUES[i] * 1.3) if BASE_VALUES[i] > 0 else float(BASE_VALUES[i] * 0.7)
            if min_v == max_v: min_v, max_v = -1.0, 1.0

            val = curr_col.slider(label, min_value=min_v, max_value=max_v, key=f"sim_{i}")
            sim_inputs.append(val)
            
    with col_res:
        st.subheader("Real-Time Status")
        model = models["Stacked LSTM"]
        data = prepare_input(sim_inputs)
        rul = float(model.predict(data, verbose=0)[0][0] * RUL_CAP)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = rul,
            gauge = {
                'axis': {'range': [0, 150], 'tickcolor': "#94A3B8"},
                'bar': {'color': "#3B82F6"},
                'bgcolor': "#162031",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 40], 'color': "rgba(239, 68, 68, 0.15)"},
                    {'range': [40, 80], 'color': "rgba(245, 158, 11, 0.15)"},
                    {'range': [80, 150], 'color': "rgba(16, 185, 129, 0.15)"}],
                'threshold': {'line': {'color': "#ef4444", 'width': 4}, 'thickness': 0.75, 'value': 40}}
        ))
        fig.update_layout(paper_bgcolor="#0B1120", font={'color': "#E2E8F0"}, height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        if rul < 40:
            st.markdown("""<style>.stApp { animation: pulse-red 1.5s infinite; border: 5px solid #ef4444; }</style>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="centered-status" style="border-color: #ef4444; background-color: rgba(239, 68, 68, 0.1);"><h3 style="color: #ef4444; margin:0;">⚠️ CRITICAL WARNING</h3><p style="margin:0;">Immediate Maintenance Required</p></div>""", unsafe_allow_html=True)
        elif rul > 100:
            st.markdown(f"""<div class="centered-status" style="border-color: #10b981;"><h3 style="color: #10b981; margin:0;">✅ SYSTEM OPTIMAL</h3><p style="margin:0;">Engine in good health</p></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="centered-status" style="border-color: #f59e0b;"><h3 style="color: #f59e0b; margin:0;">🛠️ ATTENTION NEEDED</h3><p style="margin:0;">Schedule check soon</p></div>""", unsafe_allow_html=True)