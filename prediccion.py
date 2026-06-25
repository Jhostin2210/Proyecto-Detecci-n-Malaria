import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import model_from_json
import base64
import time
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm

# =====================================================================
# 1. CONFIGURACIÓN DE PÁGINA
# =====================================================================
st.set_page_config(
    page_title="Sistema Clínico de IA — Detección de Malaria",
    page_icon="🔬",
    layout="wide"
)

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(image_file):
    bin_str = get_base64(image_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-attachment: fixed;
    }}
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background-color: rgba(12, 16, 22, 0.82);
        z-index: -1;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

try:
    set_background("fondo.png")
except:
    pass

# =====================================================================
# CSS GLOBAL
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*, body, .stApp { font-family: 'Inter', sans-serif !important; }

.top-bar {
    display: flex; align-items: center; justify-content: space-between;
    background: linear-gradient(90deg, rgba(0,43,91,0.95) 0%, rgba(0,82,163,0.95) 100%);
    padding: 12px 28px; border-radius: 12px; margin-bottom: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.top-bar-left { display: flex; align-items: center; gap: 14px; }
.top-bar-icon { font-size: 2rem; }
.top-bar-title { color: #ffffff; font-size: 1.6rem; font-weight: 800; line-height: 1.2; letter-spacing: -0.01em; }
.top-bar-subtitle { color: #90b8e0; font-size: 0.85rem; font-weight: 400; margin-top: 3px; }
.top-bar-badge {
    background: rgba(46,204,113,0.2); border: 1px solid #2ecc71;
    color: #2ecc71; font-size: 0.75rem; font-weight: 600;
    padding: 4px 12px; border-radius: 20px; letter-spacing: 0.05em;
}
.top-bar-badge::before { content: "● "; font-size: 0.6rem; }

.steps-bar {
    display: flex; align-items: center;
    background: rgba(25,35,50,0.85); padding: 14px 24px;
    border-radius: 10px; margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.07); gap: 8px;
}
.step { display: flex; align-items: center; gap: 8px; flex: 1; }
.step-num {
    width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.78rem; font-weight: 700; flex-shrink: 0;
}
.step-num.active  { background: #1a6fb5; color: #fff; }
.step-num.done    { background: #2ecc71; color: #fff; }
.step-num.pending { background: rgba(255,255,255,0.1); color: #666; }
.step-label { font-size: 0.82rem; font-weight: 500; }
.step-label.active  { color: #90b8e0; }
.step-label.done    { color: #2ecc71; }
.step-label.pending { color: #555; }
.step-divider { flex: 0.3; height: 2px; background: rgba(255,255,255,0.1); border-radius: 2px; }

.patient-header {
    display: flex; align-items: center; gap: 16px;
    background: rgba(20,30,45,0.9); padding: 16px 24px;
    border-radius: 10px; margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.08);
}
.patient-avatar {
    width: 52px; height: 52px; border-radius: 50%;
    background: linear-gradient(135deg, #1a6fb5, #0d4a8a);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem; font-weight: 800; color: #fff; flex-shrink: 0;
    border: 2px solid rgba(255,255,255,0.15);
}
.patient-info-label { color: #6b7c93; font-size: 0.72rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; }
.patient-info-value { color: #e2eaf4; font-size: 0.92rem; font-weight: 600; margin-top: 1px; }
.patient-divider { width: 1px; height: 40px; background: rgba(255,255,255,0.1); margin: 0 8px; }

.section-header {
    color: #90b8e0; font-size: 0.75rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em;
    padding: 6px 0 10px 2px;
    border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 14px;
}

.ref-indicator { display: flex; align-items: center; gap: 6px; margin-top: 4px; margin-bottom: 10px; font-size: 0.75rem; }
.ref-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.ref-normal { background: #2ecc71; }
.ref-alto   { background: #e74c3c; }
.ref-text-normal { color: #2ecc71; font-weight: 600; }
.ref-text-alto   { color: #e74c3c; font-weight: 600; }
.ref-range-label { color: #6b7c93; font-size: 0.73rem; }

.result-card {
    background: rgba(20,28,42,0.95); padding: 22px; border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.07); box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}
.result-algo-label { color: #6b7c93; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }
.result-diagnosis  { font-size: 1.35rem; font-weight: 800; margin-top: 10px; margin-bottom: 12px; }
.confidence-label  { color: #6b7c93; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 5px; }
.conf-bar-wrap  { background: rgba(255,255,255,0.08); border-radius: 6px; height: 10px; overflow: hidden; }
.conf-bar-fill  { height: 100%; border-radius: 6px; }

.clinical-warning {
    background: rgba(243,156,18,0.1); border: 1px solid rgba(243,156,18,0.4);
    border-left: 4px solid #f39c12; border-radius: 8px; padding: 14px 18px;
    display: flex; gap: 12px; align-items: flex-start; margin-top: 18px;
}
.warn-icon { font-size: 1.3rem; flex-shrink: 0; margin-top: 1px; }
.warn-text { color: #e8c07a; font-size: 0.85rem; line-height: 1.5; }
.warn-text b { color: #f5d08a; }

.example-box {
    background: rgba(26,111,181,0.12); border: 1px solid rgba(26,111,181,0.35);
    border-left: 4px solid #1a6fb5; border-radius: 8px; padding: 14px 18px;
    margin-bottom: 18px; display: flex; gap: 12px; align-items: flex-start;
}
.example-box-text { color: #90b8e0; font-size: 0.85rem; line-height: 1.6; }
.example-box-text b { color: #aecde8; }

/* ── PANTALLA DE PROCESAMIENTO ── */
.processing-screen {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; min-height: 420px;
    background: rgba(15,22,35,0.95); border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.07);
    padding: 40px;
}
.proc-icon { font-size: 3.5rem; margin-bottom: 20px; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.proc-title { color: #ffffff; font-size: 1.4rem; font-weight: 700; margin-bottom: 8px; text-align:center; }
.proc-subtitle { color: #6b7c93; font-size: 0.9rem; text-align: center; margin-bottom: 30px; }
.proc-steps { width: 100%; max-width: 380px; }
.proc-step-row {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 14px; border-radius: 8px; margin-bottom: 8px;
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06);
}
.proc-step-row.done-step { background: rgba(46,204,113,0.08); border-color: rgba(46,204,113,0.2); }
.proc-step-row.active-step { background: rgba(26,111,181,0.12); border-color: rgba(26,111,181,0.3); }
.proc-step-icon { font-size: 1.1rem; flex-shrink: 0; }
.proc-step-text { font-size: 0.85rem; font-weight: 500; }
.proc-step-text.done-text   { color: #2ecc71; }
.proc-step-text.active-text { color: #90b8e0; }
.proc-step-text.pend-text   { color: #444; }

h1 { font-size: 2.4rem !important; font-weight: 800 !important; color: #ffffff !important; text-shadow: 2px 2px 8px #000; margin-bottom: 4px !important; }
h3 { font-size: 1.5rem !important; font-weight: 600 !important; color: #ffffff !important; }
label p { font-size: 1.1rem !important; font-weight: 600 !important; color: #ffffff !important; text-shadow: 1px 1px 3px #000; margin-bottom: 4px !important; }
.stNumberInput input, .stSelectbox div[data-baseweb="select"] { font-size: 1.1rem !important; height: 46px !important; border-radius: 8px !important; }
.stButton > button {
    font-size: 1.2rem !important; font-weight: 700 !important;
    padding: 14px 0 !important; border-radius: 10px !important;
    background: linear-gradient(135deg, #1a6fb5 0%, #0d4a8a 100%) !important;
    color: white !important; border: none !important;
    box-shadow: 0 4px 18px rgba(26,111,181,0.45);
    transition: all 0.3s ease;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 24px rgba(26,111,181,0.65) !important; }
.stMarkdown p { font-size: 1.05rem !important; color: #c8d8e8 !important; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# CARGA DE MODELOS
# =====================================================================
@st.cache_resource
def cargar_recursos_malaria():
    try:
        arbol = joblib.load("malaria_tree_lab.pkl")
        escalador = joblib.load("scaler_malaria.pkl")
        with open("malaria_rn.json", "r") as f:
            rn_json = f.read()
        rn = model_from_json(rn_json)
        rn.load_weights("malaria_rn.weights.h5")
        return arbol, escalador, rn
    except Exception as e:
        st.error(f"Error critico al cargar los modelos de IA: {e}")
        return None, None, None

arbol_malaria, scaler_malaria, rn_malaria = cargar_recursos_malaria()

# =====================================================================
# HELPERS
# =====================================================================
def ref_label(normal_min, normal_max, unidad=""):
    unidad_str = f" {unidad}" if unidad else ""
    return f"""
    <div class="ref-indicator" style="margin-bottom:2px; margin-top:0;">
        <div class="ref-dot" style="background:#f0c040;"></div>
        <span style="color:#c8c8c8; font-size:0.75rem;">
            Rango normal: <b style="color:#f0c040;">{normal_min}–{normal_max}{unidad_str}</b>
            &nbsp;·&nbsp; <span style="color:#e0e0e0;">Ingrese el valor del hemograma</span>
        </span>
    </div>
    """

def ref_badge(valor, normal_min, normal_max, unidad=""):
    if valor is None:
        return ""
    try:
        valor = float(valor)
    except (TypeError, ValueError):
        return ""
    if valor < normal_min:
        estado, css_dot, css_txt, texto = "BAJO", "ref-alto", "ref-text-alto", "&#8595; Por debajo del rango normal"
    elif valor > normal_max:
        estado, css_dot, css_txt, texto = "ALTO", "ref-alto", "ref-text-alto", "&#8593; Por encima del rango normal"
    else:
        estado, css_dot, css_txt, texto = "NORMAL", "ref-normal", "ref-text-normal", "&#10003; Dentro del rango normal"
    return f"""
    <div class="ref-indicator" style="margin-top:3px;">
        <div class="ref-dot {css_dot}"></div>
        <span class="{css_txt}">{estado}</span>
        <span class="ref-range-label">&nbsp;·&nbsp;{texto}</span>
    </div>
    """

def get_iniciales(sex_input):
    return "M" if sex_input == "Masculino" else "F"

def render_steps(paso):
    pasos = [(1, "Ingreso de datos"), (2, "Procesamiento IA"), (3, "Resultados")]
    steps_html = '<div class="steps-bar">'
    for i, (num, label) in enumerate(pasos):
        if num < paso:
            cls_num, cls_lbl, icon = "done", "done", "✓"
        elif num == paso:
            cls_num, cls_lbl, icon = "active", "active", str(num)
        else:
            cls_num, cls_lbl, icon = "pending", "pending", str(num)
        steps_html += f'<div class="step"><div class="step-num {cls_num}">{icon}</div><span class="step-label {cls_lbl}">{label}</span></div>'
        if i < len(pasos) - 1:
            steps_html += '<div class="step-divider"></div>'
    steps_html += '</div>'
    return steps_html

# =====================================================================
# BARRA SUPERIOR INSTITUCIONAL
# =====================================================================
fecha_hoy = datetime.now().strftime("%d %b %Y — %H:%M")
st.markdown(f"""
<div class="top-bar">
    <div class="top-bar-left">
        <div class="top-bar-icon">🔬</div>
        <div>
            <div class="top-bar-title">Sistema Clínico de Inteligencia Artificial</div>
            <div class="top-bar-subtitle">Módulo de Detección de Malaria por Hemograma · {fecha_hoy}</div>
        </div>
    </div>
    <div class="top-bar-badge">IA ACTIVA</div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# SESSION STATE
# =====================================================================
if "paso_actual" not in st.session_state:
    st.session_state.paso_actual = 1
if "datos_paciente" not in st.session_state:
    st.session_state.datos_paciente = {}

if arbol_malaria is None or scaler_malaria is None or rn_malaria is None:
    st.error("Los componentes de IA no estan listos. Verifica los archivos del modelo.")
    st.stop()

# =====================================================================
# PANTALLA 1 — FORMULARIO
# =====================================================================
if st.session_state.paso_actual == 1:

    st.markdown(render_steps(1), unsafe_allow_html=True)

    paciente_placeholder = st.empty()

    st.markdown("""
    <div class="example-box">
        <span style="font-size:1.3rem; flex-shrink:0;">💡</span>
        <div class="example-box-text">
            <b>¿Cómo usar este sistema?</b> Complete los campos con los valores exactos del hemograma automatizado del paciente.
            Cada campo muestra el <b>rango clínico normal</b> como referencia y un ejemplo de valor esperado.
            Al terminar, presione <b>"Procesar Diagnóstico Médico"</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📋 Datos del Hemograma Completo</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-header">Datos Generales</div>', unsafe_allow_html=True)
        sex_input = st.selectbox("Sexo del Paciente", ["Masculino", "Femenino"],
            help="Seleccione el sexo biológico del paciente")
        st.markdown(ref_label(18, 65, "años"), unsafe_allow_html=True)
        age = st.number_input("Edad (Años)", min_value=0, max_value=120,
            value=None, placeholder="Ej: 35",
            help="Edad del paciente en años")
        if ref_badge(age, 18, 65, "años"): st.markdown(ref_badge(age, 18, 65, "años"), unsafe_allow_html=True)

        hemoglobin_ref = (13.5, 17.5) if sex_input == "Masculino" else (12.0, 15.5)
        st.markdown(ref_label(hemoglobin_ref[0], hemoglobin_ref[1], "g/dL"), unsafe_allow_html=True)
        hemoglobin = st.number_input("Hemoglobina (Hb g/dL)",
            min_value=0.0, max_value=30.0,
            value=None, placeholder=f"Ej: {hemoglobin_ref[0]}", format="%.2f",
            help=f"Rango normal: {hemoglobin_ref[0]}–{hemoglobin_ref[1]} g/dL")
        if ref_badge(hemoglobin, hemoglobin_ref[0], hemoglobin_ref[1], "g/dL"): st.markdown(ref_badge(hemoglobin, hemoglobin_ref[0], hemoglobin_ref[1], "g/dL"), unsafe_allow_html=True)

        st.markdown(ref_label(4500, 11000, "/cumm"), unsafe_allow_html=True)
        wbc = st.number_input("Glóbulos Blancos (/cumm)",
            min_value=0, max_value=50000,
            value=None, placeholder="Ej: 6500",
            help="Leucocitos totales. Rango normal: 4,500–11,000 /cumm")
        if ref_badge(wbc, 4500, 11000, "/cumm"): st.markdown(ref_badge(wbc, 4500, 11000, "/cumm"), unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Fórmula Leucocitaria</div>', unsafe_allow_html=True)
        st.markdown(ref_label(50, 70, "%"), unsafe_allow_html=True)
        neutrophils = st.number_input("Neutrófilos (%)",
            min_value=0, max_value=100,
            value=None, placeholder="Ej: 55",
            help="Porcentaje de neutrófilos. Rango normal: 50–70 %")
        if ref_badge(neutrophils, 50, 70, "%"): st.markdown(ref_badge(neutrophils, 50, 70, "%"), unsafe_allow_html=True)

        st.markdown(ref_label(20, 40, "%"), unsafe_allow_html=True)
        lymphocytes = st.number_input("Linfocitos (%)",
            min_value=0, max_value=100,
            value=None, placeholder="Ej: 32",
            help="Porcentaje de linfocitos. Rango normal: 20–40 %")
        if ref_badge(lymphocytes, 20, 40, "%"): st.markdown(ref_badge(lymphocytes, 20, 40, "%"), unsafe_allow_html=True)

        st.markdown(ref_label(100, 300, ""), unsafe_allow_html=True)
        eosinophils = st.number_input("Eosinófilos Circulantes",
            min_value=0, max_value=1000,
            value=None, placeholder="Ej: 180",
            help="Conteo absoluto de eosinófilos. Rango normal: 100–300")
        if ref_badge(eosinophils, 100, 300, ""): st.markdown(ref_badge(eosinophils, 100, 300, ""), unsafe_allow_html=True)

        htc_ref = (41.0, 53.0) if sex_input == "Masculino" else (36.0, 46.0)
        st.markdown(ref_label(htc_ref[0], htc_ref[1], "%"), unsafe_allow_html=True)
        htc = st.number_input("Hematocrito (HTC %)",
            min_value=0.0, max_value=100.0,
            value=None, placeholder=f"Ej: {htc_ref[0]}",
            help=f"Rango normal: {htc_ref[0]}–{htc_ref[1]} %")
        if ref_badge(htc, htc_ref[0], htc_ref[1], "%"): st.markdown(ref_badge(htc, htc_ref[0], htc_ref[1], "%"), unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="section-header">Índices Eritrocitarios</div>', unsafe_allow_html=True)
        st.markdown(ref_label(27.0, 33.0, "pg"), unsafe_allow_html=True)
        mch = st.number_input("MCH (pg)",
            min_value=0.0, max_value=50.0,
            value=None, placeholder="Ej: 29.5",
            help="Hemoglobina corpuscular media. Rango normal: 27–33 pg")
        if ref_badge(mch, 27.0, 33.0, "pg"): st.markdown(ref_badge(mch, 27.0, 33.0, "pg"), unsafe_allow_html=True)

        st.markdown(ref_label(31.0, 35.0, "g/dL"), unsafe_allow_html=True)
        mchc = st.number_input("MCHC (g/dL)",
            min_value=0.0, max_value=50.0,
            value=None, placeholder="Ej: 33.5",
            help="Concentración de hemoglobina corpuscular media. Rango: 31–35 g/dL")
        if ref_badge(mchc, 31.0, 35.0, "g/dL"): st.markdown(ref_badge(mchc, 31.0, 35.0, "g/dL"), unsafe_allow_html=True)

        st.markdown(ref_label(11.5, 14.5, "%"), unsafe_allow_html=True)
        rdw = st.number_input("RDW-CV (%)",
            min_value=0.0, max_value=30.0,
            value=None, placeholder="Ej: 13.2",
            help="Amplitud de distribución eritrocitaria. Rango normal: 11.5–14.5 %")
        if ref_badge(rdw, 11.5, 14.5, "%"): st.markdown(ref_badge(rdw, 11.5, 14.5, "%"), unsafe_allow_html=True)

        st.markdown(ref_label(150000, 400000, "/mm³"), unsafe_allow_html=True)
        platelets = st.number_input("Recuento de Plaquetas (/mm³)",
            min_value=0, max_value=1000000,
            value=None, placeholder="Ej: 230000",
            help="Rango normal: 150,000–400,000 /mm³")
        if ref_badge(platelets, 150000, 400000, "/mm³"): st.markdown(ref_badge(platelets, 150000, 400000, "/mm³"), unsafe_allow_html=True)

    # Encabezado paciente dinámico
    iniciales = get_iniciales(sex_input)
    paciente_placeholder.markdown(f"""
    <div class="patient-header">
        <div class="patient-avatar">{iniciales}</div>
        <div>
            <div class="patient-info-label">Sexo</div>
            <div class="patient-info-value">{sex_input}</div>
        </div>
        <div class="patient-divider"></div>
        <div>
            <div class="patient-info-label">Edad</div>
            <div class="patient-info-value">{f"{age} años" if age is not None else "—"}</div>
        </div>
        <div class="patient-divider"></div>
        <div>
            <div class="patient-info-label">Fecha de análisis</div>
            <div class="patient-info-value">{datetime.now().strftime("%d/%m/%Y")}</div>
        </div>
        <div class="patient-divider"></div>
        <div>
            <div class="patient-info-label">Estado del sistema</div>
            <div class="patient-info-value" style="color:#2ecc71;">● Listo para procesar</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 PROCESAR DIAGNÓSTICO MÉDICO", use_container_width=True):
        campos_vacios = [
            nombre for nombre, val in [
                ("Edad", age), ("Hemoglobina", hemoglobin), ("Glóbulos Blancos", wbc),
                ("Neutrófilos", neutrophils), ("Linfocitos", lymphocytes),
                ("Eosinófilos", eosinophils), ("Hematocrito", htc),
                ("MCH", mch), ("MCHC", mchc), ("RDW-CV", rdw), ("Plaquetas", platelets)
            ] if val is None
        ]
        if campos_vacios:
            st.error(f"⚠️ Complete los siguientes campos: **{', '.join(campos_vacios)}**")
        else:
            st.session_state.datos_paciente = {
                "sex_input": sex_input, "age": age, "hemoglobin": hemoglobin,
                "wbc": wbc, "neutrophils": neutrophils, "lymphocytes": lymphocytes,
                "eosinophils": eosinophils, "htc": htc, "mch": mch,
                "mchc": mchc, "rdw": rdw, "platelets": platelets
            }
            st.session_state.paso_actual = 2
            st.rerun()

# =====================================================================
# PANTALLA 2 — PROCESAMIENTO
# =====================================================================
elif st.session_state.paso_actual == 2:

    st.markdown(render_steps(2), unsafe_allow_html=True)

    d = st.session_state.datos_paciente
    sex_num = 0 if d["sex_input"] == "Masculino" else 1
    datos_crudos = np.array([[
        sex_num, d["age"], d["hemoglobin"], d["wbc"], d["neutrophils"],
        d["lymphocytes"], d["eosinophils"], d["htc"], d["mch"],
        d["mchc"], d["rdw"], d["platelets"]
    ]])

    proc_placeholder = st.empty()

    # PASO A — Validando datos
    proc_placeholder.markdown("""
    <div class="processing-screen">
        <div class="proc-icon">🔬</div>
        <div class="proc-title">Procesando análisis clínico</div>
        <div class="proc-subtitle">Por favor espere mientras los algoritmos evalúan los parámetros hematológicos</div>
        <div class="proc-steps">
            <div class="proc-step-row active-step">
                <span class="proc-step-icon">⚙️</span>
                <span class="proc-step-text active-text">Validando y escalando parámetros del hemograma...</span>
            </div>
            <div class="proc-step-row">
                <span class="proc-step-icon">🌿</span>
                <span class="proc-step-text pend-text">Ejecutando Árbol de Decisión Clínica...</span>
            </div>
            <div class="proc-step-row">
                <span class="proc-step-icon">🧠</span>
                <span class="proc-step-text pend-text">Ejecutando Red Neuronal Profunda...</span>
            </div>
            <div class="proc-step-row">
                <span class="proc-step-icon">📊</span>
                <span class="proc-step-text pend-text">Generando dictamen del sistema experto...</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(0.8)

    # PASO B — Escalando datos
    datos_escalados = scaler_malaria.transform(datos_crudos)

    proc_placeholder.markdown("""
    <div class="processing-screen">
        <div class="proc-icon">🌿</div>
        <div class="proc-title">Procesando análisis clínico</div>
        <div class="proc-subtitle">Por favor espere mientras los algoritmos evalúan los parámetros hematológicos</div>
        <div class="proc-steps">
            <div class="proc-step-row done-step">
                <span class="proc-step-icon">✅</span>
                <span class="proc-step-text done-text">Parámetros validados y escalados correctamente</span>
            </div>
            <div class="proc-step-row active-step">
                <span class="proc-step-icon">⚙️</span>
                <span class="proc-step-text active-text">Ejecutando Árbol de Decisión Clínica...</span>
            </div>
            <div class="proc-step-row">
                <span class="proc-step-icon">🧠</span>
                <span class="proc-step-text pend-text">Ejecutando Red Neuronal Profunda...</span>
            </div>
            <div class="proc-step-row">
                <span class="proc-step-icon">📊</span>
                <span class="proc-step-text pend-text">Generando dictamen del sistema experto...</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(0.8)

    # PASO C — Árbol de Decisión
    t_inicio = time.time()
    pred_arbol = arbol_malaria.predict(datos_crudos)[0]

    proc_placeholder.markdown("""
    <div class="processing-screen">
        <div class="proc-icon">🧠</div>
        <div class="proc-title">Procesando análisis clínico</div>
        <div class="proc-subtitle">Por favor espere mientras los algoritmos evalúan los parámetros hematológicos</div>
        <div class="proc-steps">
            <div class="proc-step-row done-step">
                <span class="proc-step-icon">✅</span>
                <span class="proc-step-text done-text">Parámetros validados y escalados correctamente</span>
            </div>
            <div class="proc-step-row done-step">
                <span class="proc-step-icon">✅</span>
                <span class="proc-step-text done-text">Árbol de Decisión ejecutado correctamente</span>
            </div>
            <div class="proc-step-row active-step">
                <span class="proc-step-icon">⚙️</span>
                <span class="proc-step-text active-text">Ejecutando Red Neuronal Profunda...</span>
            </div>
            <div class="proc-step-row">
                <span class="proc-step-icon">📊</span>
                <span class="proc-step-text pend-text">Generando dictamen del sistema experto...</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(0.8)

    # PASO D — Red Neuronal
    probabilidades_rn = rn_malaria.predict(datos_escalados, verbose=0)
    pred_rn = np.argmax(probabilidades_rn, axis=1)[0]
    porcentaje_rn = float(np.max(probabilidades_rn) * 100)
    t_proceso = round((time.time() - t_inicio) * 1000, 1)

    proc_placeholder.markdown("""
    <div class="processing-screen">
        <div class="proc-icon">📊</div>
        <div class="proc-title">Procesando análisis clínico</div>
        <div class="proc-subtitle">Por favor espere mientras los algoritmos evalúan los parámetros hematológicos</div>
        <div class="proc-steps">
            <div class="proc-step-row done-step">
                <span class="proc-step-icon">✅</span>
                <span class="proc-step-text done-text">Parámetros validados y escalados correctamente</span>
            </div>
            <div class="proc-step-row done-step">
                <span class="proc-step-icon">✅</span>
                <span class="proc-step-text done-text">Árbol de Decisión ejecutado correctamente</span>
            </div>
            <div class="proc-step-row done-step">
                <span class="proc-step-icon">✅</span>
                <span class="proc-step-text done-text">Red Neuronal Profunda ejecutada correctamente</span>
            </div>
            <div class="proc-step-row active-step">
                <span class="proc-step-icon">⚙️</span>
                <span class="proc-step-text active-text">Generando dictamen del sistema experto...</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(0.6)

    st.session_state.resultados = {
        "pred_arbol": int(pred_arbol),
        "pred_rn": int(pred_rn),
        "porcentaje_rn": porcentaje_rn,
        "t_proceso": t_proceso,
    }
    st.session_state.paso_actual = 3
    st.rerun()

# =====================================================================
# PANTALLA 3 — RESULTADOS
# =====================================================================
elif st.session_state.paso_actual == 3:

    st.markdown(render_steps(3), unsafe_allow_html=True)

    d = st.session_state.datos_paciente
    r = st.session_state.resultados

    pred_arbol    = r["pred_arbol"]
    pred_rn       = r["pred_rn"]
    porcentaje_rn = r["porcentaje_rn"]
    t_proceso     = r["t_proceso"]

    resultado_arbol = "POSITIVO — Infección Detectada" if pred_arbol == 1 else "NEGATIVO — Sin Infección"
    color_arbol     = "#e74c3c" if pred_arbol == 1 else "#2ecc71"
    resultado_rn    = "POSITIVO — Infección Detectada" if pred_rn == 1 else "NEGATIVO — Sin Infección"
    color_rn        = "#e74c3c" if pred_rn == 1 else "#2ecc71"

    iniciales = get_iniciales(d["sex_input"])
    st.markdown(f"""
    <div class="patient-header">
        <div class="patient-avatar">{iniciales}</div>
        <div>
            <div class="patient-info-label">Sexo</div>
            <div class="patient-info-value">{d["sex_input"]}</div>
        </div>
        <div class="patient-divider"></div>
        <div>
            <div class="patient-info-label">Edad</div>
            <div class="patient-info-value">{d["age"]} años</div>
        </div>
        <div class="patient-divider"></div>
        <div>
            <div class="patient-info-label">Fecha de análisis</div>
            <div class="patient-info-value">{datetime.now().strftime("%d/%m/%Y")}</div>
        </div>
        <div class="patient-divider"></div>
        <div>
            <div class="patient-info-label">Estado</div>
            <div class="patient-info-value" style="color:#f39c12;">● Análisis completado</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Dictamen del Sistema Experto")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        # ─── CORRECCIÓN 1: eliminados los meta-chips (⏱ ms, 🧬 Modelo, 📅 fecha) ───
        st.markdown(f"""
        <div class="result-card" style="border-top: 5px solid {color_arbol};">
            <div class="result-algo-label">🌿 Árbol de Decisión Clínica</div>
            <div class="result-diagnosis" style="color:{color_arbol};">{resultado_arbol}</div>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        # ─── CORRECCIÓN 1: eliminados los meta-chips de esta tarjeta también ───
        st.markdown(f"""
        <div class="result-card" style="border-top: 5px solid {color_rn};">
            <div class="result-algo-label">🧠 Red Neuronal Profunda</div>
            <div class="result-diagnosis" style="color:{color_rn};">{resultado_rn}</div>
            <div class="confidence-label">Nivel de confianza del modelo</div>
            <div class="conf-bar-wrap">
                <div class="conf-bar-fill" style="width:{porcentaje_rn:.1f}%; background:{color_rn};"></div>
            </div>
            <div style="color:#8a9bb0; font-size:0.8rem; margin-top:5px; text-align:right;">
                {porcentaje_rn:.2f} %
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="clinical-warning">
        <div class="warn-icon">⚠️</div>
        <div class="warn-text">
            <b>Aviso Clínico Importante:</b> Este sistema es una herramienta de <b>soporte analítico predictivo</b>,
            no un diagnóstico definitivo. La confirmación de malaria requiere
            <b>Gota Gruesa microscópica</b> o <b>prueba molecular PCR</b> realizada por personal calificado.
            El médico tratante es el responsable final del diagnóstico y tratamiento.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── GENERADOR PDF ──
    def generar_pdf(d, res_arbol, res_rn, pct_rn, t_proc, p_arbol, p_rn):
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm)

        COLOR_AZUL  = colors.HexColor("#00529a")
        COLOR_VERDE = colors.HexColor("#1ea050")
        COLOR_ROJO  = colors.HexColor("#c83232")
        COLOR_AMBAR = colors.HexColor("#f39c12")
        COLOR_GRIS  = colors.HexColor("#f0f4f8")

        # ─── CORRECCIÓN 2: estilos del encabezado ajustados para evitar solapamiento ───
        estilo_titulo  = ParagraphStyle(
            "titulo",
            fontSize=13,
            fontName="Helvetica-Bold",
            textColor=colors.white,
            leading=17,
            spaceAfter=0
        )
        estilo_fecha   = ParagraphStyle(
            "fecha",
            fontSize=8.5,
            fontName="Helvetica",
            textColor=colors.HexColor("#cce0ff"),
            alignment=2,   # right
            leading=12,
            spaceAfter=0
        )
        estilo_seccion = ParagraphStyle("seccion", fontSize=11, fontName="Helvetica-Bold", textColor=colors.white, spaceAfter=4)
        estilo_normal  = ParagraphStyle("normal",  fontSize=10, fontName="Helvetica",      spaceAfter=4)
        estilo_aviso   = ParagraphStyle("aviso",   fontSize=8.5, fontName="Helvetica",
                                        textColor=colors.HexColor("#7a5500"), spaceAfter=4, leading=13)

        elementos = []
        ancho = A4[0] - 4*cm

        # ─── CORRECCIÓN 2: encabezado en dos filas para que el título no se corte ───
        encabezado = Table(
            [[
                Paragraph("SISTEMA CLINICO DE IA", estilo_titulo),
                Paragraph(f"Reporte: {datetime.now().strftime('%d/%m/%Y %H:%M')}", estilo_fecha)
            ],
            [
                Paragraph("DETECCION DE MALARIA POR HEMOGRAMA", estilo_titulo),
                Paragraph("", estilo_fecha)
            ]],
            colWidths=[ancho * 0.68, ancho * 0.32]
        )
        encabezado.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, -1), COLOR_AZUL),
            ("ROWPADDING",  (0, 0), (-1, -1), 8),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("SPAN",        (1, 0), (1, 1)),   # celda de fecha ocupa las 2 filas
        ]))
        elementos.append(encabezado)
        elementos.append(Spacer(1, 0.4*cm))

        # Datos del paciente
        sec_px = Table([[Paragraph("DATOS DEL PACIENTE", estilo_seccion)]], colWidths=[ancho])
        sec_px.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), COLOR_AZUL), ("ROWPADDING", (0,0), (-1,-1), 6)]))
        elementos.append(sec_px)
        elementos.append(Spacer(1, 0.2*cm))

        # ─── CORRECCIÓN 3: "años" con tilde correcta ───
        tabla_px = Table(
            [["Sexo",  d["sex_input"]],
             ["Edad",  f"{d['age']} años"],
             ["Fecha", datetime.now().strftime("%d/%m/%Y")]],
            colWidths=[ancho*0.35, ancho*0.65]
        )
        tabla_px.setStyle(TableStyle([
            ("FONTNAME",       (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTNAME",       (1,0), (1,-1), "Helvetica"),
            ("FONTSIZE",       (0,0), (-1,-1), 9.5),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [COLOR_GRIS, colors.white]),
            ("ROWPADDING",     (0,0), (-1,-1), 5),
            ("GRID",           (0,0), (-1,-1), 0.3, colors.HexColor("#d0d8e4")),
        ]))
        elementos.append(tabla_px)
        elementos.append(Spacer(1, 0.4*cm))

        # Hemograma
        sec_hemo = Table([[Paragraph("VALORES DEL HEMOGRAMA COMPLETO", estilo_seccion)]], colWidths=[ancho])
        sec_hemo.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), COLOR_AZUL), ("ROWPADDING", (0,0), (-1,-1), 6)]))
        elementos.append(sec_hemo)
        elementos.append(Spacer(1, 0.2*cm))

        filas_hemo = [
            ["Variable", "Valor", "Rango de Referencia"],
            ["Hemoglobina (Hb)",  f"{d['hemoglobin']:.2f} g/dL",  "13.5-17.5 (M) / 12.0-15.5 (F) g/dL"],
            ["Globulos Blancos",  f"{d['wbc']} /cumm",             "4,500-11,000 /cumm"],
            ["Neutrofilos",       f"{d['neutrophils']} %",         "50-70 %"],
            ["Linfocitos",        f"{d['lymphocytes']} %",         "20-40 %"],
            ["Eosinofilos",       f"{d['eosinophils']}",           "100-300"],
            ["Hematocrito",       f"{d['htc']:.1f} %",             "41-53 % (M) / 36-46 % (F)"],
            ["MCH",               f"{d['mch']:.1f} pg",            "27-33 pg"],
            ["MCHC",              f"{d['mchc']:.1f} g/dL",         "31-35 g/dL"],
            ["RDW-CV",            f"{d['rdw']:.1f} %",             "11.5-14.5 %"],
            ["Plaquetas",         f"{d['platelets']:,} /mm3",       "150,000-400,000 /mm3"],
        ]
        tabla_hemo = Table(filas_hemo, colWidths=[ancho*0.32, ancho*0.28, ancho*0.40])
        tabla_hemo.setStyle(TableStyle([
            ("BACKGROUND",     (0,0), (-1,0), colors.HexColor("#c8d7eb")),
            ("FONTNAME",       (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",       (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [COLOR_GRIS, colors.white]),
            ("ROWPADDING",     (0,0), (-1,-1), 5),
            ("GRID",           (0,0), (-1,-1), 0.3, colors.HexColor("#d0d8e4")),
        ]))
        elementos.append(tabla_hemo)
        elementos.append(Spacer(1, 0.4*cm))

        # Dictamen
        sec_dict = Table([[Paragraph("DICTAMEN DEL SISTEMA EXPERTO", estilo_seccion)]], colWidths=[ancho])
        sec_dict.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), COLOR_AZUL), ("ROWPADDING", (0,0), (-1,-1), 6)]))
        elementos.append(sec_dict)
        elementos.append(Spacer(1, 0.3*cm))

        def safe(t):
            for a, b in [("—","-"),("–","-"),("é","e"),("ó","o"),("í","i"),
                         ("á","a"),("ú","u"),("ñ","n"),("É","E"),("Ó","O"),
                         ("Í","I"),("Á","A"),("Ú","U"),("Ñ","N")]:
                t = t.replace(a, b)
            return t

        c_arbol = COLOR_ROJO if p_arbol == 1 else COLOR_VERDE
        c_rn    = COLOR_ROJO if p_rn    == 1 else COLOR_VERDE

        dictamen = [
            [Paragraph("Arbol de Decision Clinica", ParagraphStyle("l", fontSize=9, fontName="Helvetica-Bold", textColor=colors.HexColor("#444"))),
             Paragraph("Red Neuronal Profunda",     ParagraphStyle("r", fontSize=9, fontName="Helvetica-Bold", textColor=colors.HexColor("#444")))],
            [Paragraph(safe(res_arbol), ParagraphStyle("ra", fontSize=12, fontName="Helvetica-Bold", textColor=c_arbol)),
             Paragraph(safe(res_rn),    ParagraphStyle("rr", fontSize=12, fontName="Helvetica-Bold", textColor=c_rn))],
            [Paragraph("", estilo_normal),
             Paragraph(f"Confianza del modelo: {pct_rn:.2f}%", estilo_normal)],
        ]
        tabla_dict = Table(dictamen, colWidths=[ancho*0.5, ancho*0.5])
        tabla_dict.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), COLOR_GRIS),
            ("ROWPADDING", (0,0), (-1,-1), 8),
            ("GRID",       (0,0), (-1,-1), 0.3, colors.HexColor("#d0d8e4")),
            ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ]))
        elementos.append(tabla_dict)
        elementos.append(Spacer(1, 0.4*cm))

        # Aviso clínico
        aviso_data = [[Paragraph(
            "<b>AVISO CLINICO IMPORTANTE</b><br/>"
            "Este sistema es una herramienta de soporte analitico predictivo, NO un diagnostico definitivo. "
            "La confirmacion de malaria requiere Gota Gruesa microscopica o prueba molecular PCR "
            "por personal calificado. El medico tratante es el responsable final.", estilo_aviso)]]
        tabla_aviso = Table(aviso_data, colWidths=[ancho])
        tabla_aviso.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#fff8e1")),
            ("ROWPADDING", (0,0), (-1,-1), 10),
            ("BOX",        (0,0), (-1,-1), 1.5, COLOR_AMBAR),
            ("LINEBEFORE", (0,0), (0,-1), 4, COLOR_AMBAR),
        ]))
        elementos.append(tabla_aviso)

        doc.build(elementos)
        buf.seek(0)
        return buf.read()

    # Botones
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        pdf_bytes = generar_pdf(
            d, resultado_arbol, resultado_rn, porcentaje_rn,
            t_proceso, pred_arbol, pred_rn
        )
        st.download_button(
            label="📄 Exportar Reporte PDF",
            data=pdf_bytes,
            file_name=f"reporte_malaria_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    with btn_col2:
        if st.button("🔄 Nuevo Análisis", use_container_width=True):
            st.session_state.paso_actual = 1
            st.session_state.datos_paciente = {}
            st.session_state.resultados = {}
            st.rerun()