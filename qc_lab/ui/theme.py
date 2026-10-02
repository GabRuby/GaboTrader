"""
Tema CSS estilo QuantConnect para el dashboard Streamlit.

Se importa desde `app.py` para aplicar el look & feel corporativo (fondo oscuro,
cabeceras, tarjetas, tablas, sidebar y graficas).
"""
from __future__ import annotations

THEME_CSS = """
<style>
:root {
  --qc-bg: #111111;
  --qc-bg-alt: #1a1a1a;
  --qc-surface: #1f1f1f;
  --qc-border: #2f2f2f;
  --qc-primary: #3d9cff;
  --qc-accent: #1aa2ff;
  --qc-success: #4ade80;
  --qc-warning: #f59e0b;
  --qc-danger: #ef4444;
  --qc-text: #f2f2f2;
  --qc-text-muted: #b3b3b3;
  --qc-code: #1e1e1e;
}

div[data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 800px at 50% 0%, #121212 0%, #0b0b0b 100%);
  color: var(--qc-text);
}

header[data-testid="stHeader"] {
  background: rgba(17, 17, 17, 0.92);
  backdrop-filter: blur(10px);
  box-shadow: inset 0 -1px 0 var(--qc-border);
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #141414 0%, #0f0f0f 100%);
  border-right: 1px solid var(--qc-border);
}

section[data-testid="stSidebar"] > div:first-child {
  padding-top: 1rem;
}

h1, h2, h3, h4, h5, h6 {
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--qc-text);
}

p, span, label, small {
  color: var(--qc-text);
}

[data-testid="stMarkdownContainer"] p {
  color: var(--qc-text);
}

.stTabs [data-baseweb="tab-list"] {
  background: var(--qc-bg-alt);
  border: 1px solid var(--qc-border);
  border-radius: 10px;
  padding: 0.25rem;
  width: fit-content;
}

.stTabs [data-baseweb="tab"] {
  border-radius: 8px;
  color: var(--qc-text-muted);
  padding: 0.4rem 0.9rem;
  font-weight: 500;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
  background: linear-gradient(135deg, var(--qc-primary) 0%, var(--qc-accent) 100%);
  color: #0f0f0f;
  box-shadow: 0 10px 30px -18px var(--qc-primary);
}

.stButton > button {
  background: linear-gradient(135deg, var(--qc-primary) 0%, var(--qc-accent) 100%);
  color: #0f0f0f;
  border: 0;
  border-radius: 10px;
  font-weight: 600;
  padding: 0.55rem 1.1rem;
  transition: transform 120ms ease, box-shadow 120ms ease;
}

.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 10px 40px -18px var(--qc-primary);
}

.stButton > button:active {
  transform: translateY(1px);
}

.stButton.secondary > button {
  background: transparent;
  color: var(--qc-text);
  border: 1px solid var(--qc-border);
}

.stDownloadButton > button {
  background: transparent;
  color: var(--qc-text);
  border: 1px solid var(--qc-border);
  border-radius: 10px;
}

.stDataFrame,
[data-testid="stTable"] {
  background: var(--qc-bg-alt);
  border: 1px solid var(--qc-border);
  border-radius: 12px;
}

div[data-testid="stMetric"] {
  background: linear-gradient(180deg, rgba(31, 31, 31, 0.98) 0%, rgba(20, 20, 20, 0.98) 100%);
  border: 1px solid var(--qc-border);
  border-radius: 14px;
  padding: 1.1rem 1.25rem;
  box-shadow: 0 40px 80px -40px rgba(61, 156, 255, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.03);
}

div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
  color: var(--qc-text-muted);
  font-size: 0.95rem;
  letter-spacing: 0.01em;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: var(--qc-text);
  font-weight: 700;
  letter-spacing: -0.01em;
}

div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
  font-weight: 600;
}

[data-testid="stExpander"] > details {
  background: var(--qc-bg-alt);
  border: 1px solid var(--qc-border);
  border-radius: 12px;
}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div,
[data-testid="stMultiSelect"] > div,
[data-testid="stSlider"] > div {
  background: var(--qc-surface) !important;
  border: 1px solid var(--qc-border) !important;
  color: var(--qc-text) !important;
  border-radius: 10px !important;
}

[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stCheckbox"] label {
  color: var(--qc-text) !important;
}

div[data-testid="stFileUploader"] > div,
div[data-testid="stFileUploaderDropzone"] {
  background: var(--qc-bg-alt);
  border: 1px dashed var(--qc-border);
  border-radius: 12px;
}

.stAlert {
  border-radius: 12px;
  border: 1px solid var(--qc-border);
}

.stSuccess {
  background: rgba(74, 222, 128, 0.12);
}

.stWarning {
  background: rgba(245, 158, 11, 0.12);
}

.stError {
  background: rgba(239, 68, 68, 0.12);
}

.stInfo {
  background: rgba(61, 156, 255, 0.12);
}

code, pre, .stCodeBlock {
  background: var(--qc-code) !important;
  border: 1px solid var(--qc-border) !important;
  border-radius: 10px !important;
}

footer {
  display: none !important;
}

div[data-testid="stDecoration"] {
  display: none !important;
}

div[data-testid="stStatusWidget"] {
  background: transparent;
}

hr {
  border-color: var(--qc-border);
}
</style>
"""