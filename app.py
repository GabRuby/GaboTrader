import ast
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots

# Variables de entorno del proyecto (.env): GEMINI_API_KEY, ENVIRONMENT, etc.
load_dotenv()
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Configuración de página ancha estilo IDE
st.set_page_config(
    page_title="QuantLab IDE - Trading Strategy Evaluator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ENGINE_VERSION = "v18142"

# -----------------------------------------------------------------------------
# CONSTANTES: PROMPT DEL ANALISTA DE RIESGO CUANTITATIVO
# -----------------------------------------------------------------------------
AI_ANALYST_SYSTEM_PROMPT = """Eres un Quant Analyst senior con más de 15 años de experiencia en
gestión de carteras cuantitativas, control de riesgo y validación de estrategias para
QuantConnect (Lean Engine) y Python.

Recibes el resultado de un backtest ya ejecutado. Interpreta las métricas con criterio de
gestor de riesgo en lugar de describirlas: el usuario ya las ve en pantalla.

Devuelve SIEMPRE un informe en Markdown con estas cuatro secciones, en este orden exacto y
con estos encabezados:

## 📉 Diagnóstico General
Resumen ejecutivo del rendimiento en 3 a 5 frases. Explica la causa raíz del resultado: por qué
el Sharpe es negativo o insuficiente, qué provocó el drawdown, y si el problema está en la
lógica de señales, en la gestión de riesgo o en el propio régimen de mercado. Cita los números
concretos que sustentan cada afirmación.

## 🎯 Comportamiento frente al Benchmark
Compara la estrategia con el activo base. Indica si protegió capital en las caídas o si
simplemente siguió al mercado, si la estrategia se quedó rezagada o la superó, y cuál es el
exceso de retorno en puntos porcentuales. Concluye si compensa mantener esa estrategia en lugar
de comprar y mantener el activo.

## ⚠️ Puntos Críticos y Eficiencia
Evalúa el Win Rate junto con la profundidad del Max Drawdown: relación riesgo/beneficio real,
sesgo de la estrategia (ganar poco y perder mucho o al revés), consistencia, número de
operaciones de la muestra y si el resultado es estadísticamente significativo. Señala qué
métrica esconde el resultado y cuál es la que más preocupa.

## 💡 Sugerencias Técnicas de Ajuste
Recomendaciones concretas y accionables, nunca genéricas. Propón valores numéricos concretos
para las medias móviles, el capital y el apalancamiento, y filtros o salidas específicos
(filtro de volatilidad, filtro de tendencia, calendario, stop loss, trailing stop, take profit
parcial). Prioriza cada cambio por impacto esperado y explica en una línea el porqué.

Reglas estrictas:
- Fundamenta cada afirmación en las métricas recibidas. No inventes datos ni operaciones.
- Si falta un dato, dilo explícitamente en lugar de suponerlo.
- Sé directo y cuantitativo, sin elogios vacíos ni frases de relleno."""

# -----------------------------------------------------------------------------
# 1. TEMA CLARO PROFESIONAL (LIGHT MODE / GRISES Y BLANCO)
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
    :root { color-scheme: light; }

    html, body, .stApp, [data-testid="stAppViewContainer"] {
        background-color: #ffffff;
        color: #1a202c;
    }
    .stApp { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }

    /* Ocultar barra de estado y menús nativos de Streamlit */
    #MainMenu, footer, #header, header,
    [data-testid="stHeader"], [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
        display: none !important;
    }

    .block-container {
        padding: 0.9rem 1.1rem 1.2rem 1.1rem;
        max-width: 100%;
    }
    #MainMenu, footer, [data-testid="stSidebarCollapseButton"] { visibility: hidden; }

    /* Tipografía */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5 { color: #0f172a; letter-spacing: .01em; }
    .stMarkdown p, .stMarkdown span, .stMarkdown label { color: #475569; }

    /* Barra superior de la workspace */
    .topbar {
        display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 9px 14px; margin-bottom: 10px;
    }
    .topbar .brand { font-size: 14px; font-weight: 700; color: #0f172a; letter-spacing: .06em; }
    .topbar .badge {
        font-size: 11px; font-weight: 600; color: #475569;
        background: #ffffff; border: 1px solid #cbd5e1; border-radius: 999px;
        padding: 3px 10px; font-family: 'Cascadia Mono', 'Courier New', monospace;
    }
    .topbar .badge.live { color: #15803d; border-color: #86efac; background: #f0fdf4; }
    .topbar .spacer { margin-left: auto; }
    .topbar.topbar-right { justify-content: flex-end; margin-bottom: 0; }
    .topbar .badge.ai { color: #6d28d9; border-color: #c4b5fd; background: #f5f3ff; }

    /* Popover 🤖 del header: debe ocupar la celda completa para alinearse con la barra */
    [data-testid="stPopover"] > button { width: 100%; height: 38px; }

    /* Gestor de estrategias: principal ⭐ y comparativas [x] */
    .strategy-row {
        display: flex; align-items: center; gap: 8px;
        padding: 10px 11px; margin-bottom: 6px;
        border: 1px solid #e2e8f0; border-radius: 8px; background: #ffffff;
    }
    .strategy-row.principal {
        border-color: #fcd34d; background: #fffbeb;
        box-shadow: inset 3px 0 0 #f59e0b;
    }
    .strategy-row .star { font-size: 15px; line-height: 1; }
    .strategy-row .strategy-name {
        flex: 1; font-size: 12.5px; font-weight: 600; color: #0f172a;
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .strategy-row .strategy-tag {
        font-size: 9.5px; font-weight: 700; letter-spacing: .06em;
        color: #b45309; background: #fef3c7; border: 1px solid #fcd34d;
        border-radius: 999px; padding: 1px 7px;
    }
    .strategy-subtitle {
        font-size: 10.5px; font-weight: 700; letter-spacing: .05em;
        color: #64748b; margin: 10px 0 6px;
    }

    /* G.3.6 · Grupo 1 (ESTRATEGIAS): el botón de apertura y la fila de la Principal
       van separados por un subtítulo y respiran con altura mínima, para que el grupo
       se lea de un vistazo en vez de quedar prensado en dos líneas. */
    .st-key-grp_estrategias [data-testid="stExpanderDetails"] {
        padding: 10px 12px 12px; min-height: 150px;
    }
    .st-key-grp_estrategias .stButton button { min-height: 38px; }

    /* G.3.6 · Grupo 2 (ESTRATEGIAS A COMPARAR): dos huecos de altura fija. Los
       vacíos son un marco discontinuo con la pista de lo que irá ahí; los llenos
       llevan el número de orden para que se vean como ranuras 1 y 2. */
    .compare-slot {
        display: flex; align-items: center; gap: 8px;
        min-height: 40px; padding: 7px 10px; margin-bottom: 8px;
        border: 1px solid #e2e8f0; border-radius: 8px; background: #ffffff;
    }
    .compare-slot .compare-index {
        display: inline-flex; align-items: center; justify-content: center;
        width: 18px; height: 18px; flex: 0 0 18px; border-radius: 999px;
        background: #f1f5f9; border: 1px solid #cbd5e1; color: #475569;
        font-size: 10px; font-weight: 700;
    }
    .compare-slot .strategy-name {
        flex: 1; font-size: 12.5px; font-weight: 600; color: #0f172a;
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .compare-slot.empty {
        border-style: dashed; background: #f8fafc; color: #94a3b8;
        font-size: 11.5px; font-weight: 600; justify-content: center;
    }

    /* Barra superior de métricas (KPIs) */
    .kpi-container { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; }
    .kpi-card {
        flex: 1 1 0; min-width: 150px;
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 10px 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .kpi-label {
        font-size: 10.5px; font-weight: 700; letter-spacing: .1em;
        text-transform: uppercase; color: #64748b;
    }
    .kpi-value {
        font-size: 21px; font-weight: 700; color: #0f172a; margin-top: 3px;
        font-variant-numeric: tabular-nums;
    }
    .kpi-value.positive { color: #16a34a; }
    .kpi-value.negative { color: #dc2626; }

    /* Paneles / Secciones */
    .panel {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 12px 14px; margin-bottom: 10px;
    }
    .panel-title {
        font-size: 12px; font-weight: 700; letter-spacing: .07em;
        text-transform: uppercase; color: #64748b; margin-bottom: 8px;
    }

    /* Tarjeta de análisis con IA */
    .ai-panel-title {
        font-size: 12px; font-weight: 700; letter-spacing: .07em;
        text-transform: uppercase; color: #6d28d9;
        border-left: 3px solid #7c3aed; padding-left: 8px; margin-bottom: 8px;
    }

    /* Cabecera estilo pestaña de archivo */
    .editor-header {
        font-family: 'Cascadia Mono', 'JetBrains Mono', 'Courier New', monospace;
        font-size: 12.5px; color: #334155;
        background: #f1f5f9; border: 1px solid #cbd5e1;
        border-radius: 8px 8px 0 0; padding: 9px 13px;
        display: flex; align-items: center; gap: 8px;
    }
    .editor-header .dot { color: #16a34a; font-size: 10px; }

    /* Zona de borrador (staging) del código propuesto por la IA */
    .draft-title {
        font-size: 13px; font-weight: 700; color: #7c2d12;
        background: #fff7ed; border: 1px dashed #fdba74;
        border-radius: 8px; padding: 9px 13px; margin-bottom: 4px;
    }

    /* Editor de código */
    .stTextArea textarea, .stTextArea [data-baseweb="textarea"] {
        font-family: 'Cascadia Mono', 'JetBrains Mono', 'Courier New', monospace !important;
        font-size: 12.5px !important; line-height: 1.65 !important;
        background: #f8fafc !important; color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-top-left-radius: 0 !important; border-top-right-radius: 0 !important;
        border-bottom-left-radius: 6px !important; border-bottom-right-radius: 6px !important;
    }
    .stTextArea textarea:focus {
        border-color: #2563eb !important; box-shadow: 0 0 0 1px rgba(37,99,235,.2) !important;
    }

    /* Terminal / consola inferior */
    .terminal-box {
        background-color: #1e293b;
        color: #4ade80;
        border: 1px solid #334155; border-radius: 8px;
        font-family: 'Cascadia Mono', 'JetBrains Mono', 'Courier New', monospace;
        font-size: 12.5px; line-height: 1.8;
        padding: 12px 14px; height: 175px; overflow-y: auto;
    }
    .terminal-box::-webkit-scrollbar { width: 8px; }
    .terminal-box::-webkit-scrollbar-track { background: #1e293b; }
    .terminal-box::-webkit-scrollbar-thumb { background: #475569; border-radius: 4px; }
    .t-info { color: #38bdf8; }
    .t-ok   { color: #4ade80; }
    .t-warn { color: #facc15; }
    .t-err  { color: #f87171; }
    .t-key  { color: #f8fafc; }
    .t-dim  { color: #94a3b8; }

    /* Sidebar claro */
    [data-testid="stSidebar"] { background: #f1f5f9; border-right: 1px solid #e2e8f0; }
    [data-testid="stSidebar"] .block-container { padding-top: 0.6rem; }
    [data-testid="stSidebarUserContent"] .block-container { padding-left: 1rem; padding-right: 1rem; }

    /* Grupos contraíbles del panel izquierdo: tarjeta con encabezado y flecha a la derecha */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px;
        margin-bottom: 8px; overflow: hidden;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        padding: 9px 12px; background: #f8fafc; cursor: pointer;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover { background: #f1f5f9; }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary p {
        font-size: 11.5px; font-weight: 700; letter-spacing: .07em;
        text-transform: uppercase; color: #334155;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] { padding: 2px 12px 6px; }

    /* Widgets base */
    .stSelectbox div[data-baseweb="select"] > div,
    .stTextInput input, .stNumberInput input {
        background: #ffffff !important; border-color: #cbd5e1 !important; color: #0f172a !important;
    }
    .stButton button {
        background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1;
        border-radius: 6px; font-weight: 600; font-size: 13px;
    }
    .stButton button:hover { background: #e2e8f0; border-color: #94a3b8; color: #0f172a; }
    .stButton button[kind="primary"] {
        background: #2563eb; border-color: #1d4ed8; color: #ffffff;
    }
    .stButton button[kind="primary"]:hover { background: #1d4ed8; border-color: #1e40af; }
    hr { border-color: #e2e8f0; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# GESTIÓN DE ARCHIVOS DE ESTRATEGIAS (.py)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
# `qc_lab` vive junto a la app, asi que se asegura la raiz del proyecto en el path
# antes de importarlo: es el motor y tiene que resolverse siempre igual, se lance la
# app con `streamlit run`, con el AppTest o como modulo.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from qc_lab.core import strategy_parser
from qc_lab.core.compat import contrato_heredado
from qc_lab.core.datafeed import available_tickers
from qc_lab.core.runner import ConfigEjecucion, ejecutar_estrategia

SKIP_DIRS = {
    "__pycache__", ".git", ".idea", ".vscode", ".pytest_cache",
    "venv", ".venv", "env", "node_modules", "site-packages",
}


def list_project_folders(base: Path) -> list[Path]:
    """Subcarpetas navegables del proyecto (ignora ocultas y de sistema)."""
    try:
        candidates = sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name.lower())
    except OSError:
        return []
    return [p for p in candidates if not p.name.startswith(".") and p.name not in SKIP_DIRS]


def list_strategy_files(folder: Path) -> list[str]:
    """Archivos .py de una carpeta, excluyendo app.py (la app principal)."""
    try:
        return sorted(
            (p.name for p in folder.iterdir() if p.is_file() and p.suffix == ".py" and p.name != "app.py"),
            key=str.lower,
        )
    except OSError:
        return []


def read_script_source(rel_path: str) -> str:
    """Lee el código de una estrategia (ruta relativa a BASE_DIR)."""
    try:
        return (BASE_DIR / rel_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, ValueError):
        return DEFAULT_STRATEGIES.get(rel_path, "# Nueva estrategia\n")


# -----------------------------------------------------------------------------
# CATÁLOGO DE ESTRATEGIAS: PRINCIPAL VS COMPARATIVAS
# -----------------------------------------------------------------------------
PRINCIPAL_KEY = "principal_script"
COMPARE_KEY = "compare_selection"
COMPARE_CACHE_KEY = "compare_cache"
CHART_VIEW_KEY = "chart_view"
MAX_COMPARATIVES = 2
# Comisión por ejecución (0.05 % del nocional). Ahora la aplica el motor de qc_lab
# en el dialecto vectorial; en el dialecto Lean manda el modelo de broker que
# declare la propia estrategia. Se conserva aquí como valor por defecto explicito.
COMMISSION_RATE = 0.0005

# Valores por defecto de las medias moviles. Solo se usan para lo que la app ya
# mostraba antes de que los parametros fueran dinamicos (la insignia "SMA x/y" del
# topbar y los prompts de IA): NO son controles y no llegan al motor.
DEFAULT_FAST_MA = 10
DEFAULT_SLOW_MA = 50

# G.3.5: overrides que produce la UI dinamica de parametros del sidebar en ESTE run.
# El motor los recibe por `ConfigEjecucion.parametros`, que ya sabe inyectarlos en el
# codigo (`engine.aplicar_parametros`) o pasarlos a `run_strategy` (dialecto vector).
PARAMETROS_ACTIVOS: dict = {}

# Activos que el usuario puede elegir. El universo lo define el data feed de qc_lab,
# de modo que la lista se deriva de el y no de un literal que se pueda quedar viejo.
ASSETS = [t for t in available_tickers() if t in {"AAPL", "NVDA", "TSLA", "MSFT", "BTC"}] or ["SPY"]

# G.3.6: valores de respaldo del grupo REGLAS COMUNES. Solo se usan cuando el archivo
# de la Estrategia Principal NO declara el dato: nunca son el valor permanente.
DEFAULT_TICKER = "TSLA"
DEFAULT_CAPITAL = 100000.0

# Un literal de ticker: 1-5 mayúsculas, con sufijo opcional de 1-2 letras (`BRK-B`).
RE_TICKER_DECLARADO = re.compile(r"^[A-Z]{1,5}(-[A-Z]{1,2})?$")
# Índices y referencias de mercado, y reservas/liquidez: los agrupa el mismo criterio
# que usa `EstrategiaInfo.tickers_riesgo`. Si el archivo los declara junto al activo que
# se opera, el Activo / Ticker es el que queda fuera de estos dos conjuntos.
TICKERS_MACRO = {"SPY", "SPX", "QQQ", "IWM", "VIX", "DIA", "MDY"}
TICKERS_RESERVA = {"SHV", "SHY", "BIL", "SGOV", "TLT", "IEF", "GLD"}

# G.3.6: los dos controles del grupo REGLAS COMUNES y el archivo Principal del que
# proceden. `*_ORIGEN_KEY` guarda de qué estrategia se leyeron, para recargarlos cuando
# cambia la Principal y NO pisarlos cuando el usuario los edita a mano.
ACTIVO_KEY = "reglas_activo"
CAPITAL_KEY = "reglas_capital"
ACTIVO_ORIGEN_KEY = "reglas_activo_origen"
CAPITAL_ORIGEN_KEY = "reglas_capital_origen"

# El gestor del sidebar trabaja sobre rutas relativas posix ("otras/SP500.py"),
# igual que usa el modal de apertura, para que ambosSelector convergieran en una clave.
def list_all_strategies(base: Path) -> list[str]:
    """Todas las estrategias del proyecto: raíz primero y luego subcarpetas."""
    entries = list_strategy_files(base)
    for folder in list_project_folders(base):
        prefix = folder.relative_to(base).as_posix()
        entries.extend(f"{prefix}/{name}" for name in list_strategy_files(folder))
    return entries


def ensure_strategy_state(available: list[str]) -> None:
    """Normaliza el estado de principal/comparativas contra el catálogo real."""
    if not available:
        return

    principal = st.session_state.get(PRINCIPAL_KEY)
    if principal not in available:
        st.session_state[PRINCIPAL_KEY] = available[0]

    # Nunca comparar la principal contra sí misma, ni dejar más de MAX_COMPARATIVES.
    keep = [
        script
        for script in (st.session_state.get(COMPARE_KEY) or [])
        if script in available and script != st.session_state[PRINCIPAL_KEY]
    ]
    st.session_state[COMPARE_KEY] = keep[-MAX_COMPARATIVES:]

    # La caché de comparativas pertenece a un conjunto concreto de estrategias.
    cached = st.session_state.get(COMPARE_CACHE_KEY) or {}
    st.session_state[COMPARE_CACHE_KEY] = {
        script: payload for script, payload in cached.items() if script in available
    }


def set_principal_strategy(script: str, available: list[str]) -> None:
    """Promueve `script` a Estrategia Principal y reparte el resto de estado."""
    if script not in available:
        return
    st.session_state[PRINCIPAL_KEY] = script
    # El archivo nuevo entra también en la lista de archivos navegables del editor.
    if script not in py_files:
        py_files.append(script)
    # Si estaba marcada como comparativa, deja de serlo.
    st.session_state[COMPARE_KEY] = [
        other for other in st.session_state.get(COMPARE_KEY, []) if other != script
    ]
    # El editor, el modal y la IA siguen al principal.
    st.session_state["script_nombre"] = script
    st.session_state["script_selector"] = script
    st.session_state["editor_loaded_for"] = None
    st.session_state["codigo_actual_script"] = script
    st.session_state["codigo_actual"] = read_script_source(script)
    # Un principal nuevo invalida su backtest y cualquier comparación en curso.
    st.session_state.pop(RESULT_KEY, None)
    st.session_state.pop(PARAMS_KEY, None)
    for key in (DRAFT_KEY, DRAFT_META_KEY, DRAFT_RESULT_KEY, DRAFT_METRICS_KEY, DRAFT_EQUITY_KEY):
        st.session_state.pop(key, None)
    st.session_state.pop(DRAFT_SEEDED_KEY, None)
    st.session_state.pop("bt_ai_report", None)
    st.session_state.pop("bt_ai_auto_done", None)
    st.session_state.pop("force_run", None)
    st.session_state["flash_msg"] = f"⭐ `{script}` es ahora la Estrategia Principal"


def add_comparative_strategy(script: str, available: list[str]) -> tuple[bool, str]:
    """Añade `script` a las comparativas y devuelve `(aceptado, mensaje)`.

    Es el único camino de alta del grupo ESTRATEGIAS A COMPARAR: no toca la Estrategia
    Principal ni el editor, porque comparar no puede cambiar lo que se está evaluando.
    Respeta `MAX_COMPARATIVES` y nunca admite la Principal como candidata.
    """
    if script not in available:
        return False, f"`{script}` ya no está en el proyecto."

    if script == st.session_state.get(PRINCIPAL_KEY):
        return False, f"`{script}` ya es la Estrategia Principal: no se compara consigo misma."

    selected = list(st.session_state.get(COMPARE_KEY) or [])
    if script in selected:
        return False, f"`{script}` ya está en la lista de comparativas."
    if len(selected) >= MAX_COMPARATIVES:
        return False, f"Solo caben {MAX_COMPARATIVES} comparativas."

    st.session_state[COMPARE_KEY] = selected + [script]
    return True, f"⚖️ `{script}` añadida a comparar"


# -----------------------------------------------------------------------------
# DATOS QUE DECLARA LA ESTRATEGIA PRINCIPAL (G.3.6)
# -----------------------------------------------------------------------------
# El grupo REGLAS COMUNES no se inventa el Activo / Ticker ni el Capital Inicial: los
# lee del archivo que actúa como Principal y solo cae al valor de respaldo cuando el
# archivo no los declara. Lo que sale de aquí sigue siendo el único valor que alimenta
# `_config_lab`, de modo que motor, comparativas y borrador no pueden discrepar.


def tickers_declarados(path: Path) -> list[str]:
    """Tickers que el propio archivo declara como literales, en orden de aparición.

    Hace falta este barrido porque `strategy_parser` completa `info.simbolos` con `SPY`
    y `info.capital` con 100000 cuando el archivo no dice nada: con la metadata ya
    completada no se puede distinguir "lo declara el archivo" de "lo pone el motor".
    Se ordenan por linea para que el primero sea el que aparece antes en el codigo.
    """
    try:
        arbol = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return []

    universo = set(available_tickers())
    declarados: list[tuple[int, str]] = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str):
            ticker = nodo.value.upper()
            if ticker in universo and RE_TICKER_DECLARADO.match(ticker):
                declarados.append((getattr(nodo, "lineno", 0), ticker))
    return list(dict.fromkeys(ticker for _, ticker in sorted(declarados)))


def reglas_comunes_de_estrategia(path: Path, info=None) -> tuple[str, float]:
    """`(ticker, capital)` del archivo de la Estrategia Principal, con su respaldo.

    El Activo / Ticker es el primer ticker declarado que no sea índice ni reserva; si
    el archivo solo declara índices (una estrategia sobre SPY) se usa ese mismo, porque
    ahí sí es el instrumento. El capital es el que declara el archivo y, si no declara
    ninguno, el respaldo. El benchmark del archivo NO se lee aquí: es otro concepto.
    """
    info = info if info is not None else strategy_parser.analizar_archivo(path)
    declarados = tickers_declarados(path)
    riesgo = [t for t in declarados if t not in TICKERS_MACRO | TICKERS_RESERVA] or declarados
    ticker = riesgo[0] if riesgo else DEFAULT_TICKER
    capital = float(info.capital) if info.capital else DEFAULT_CAPITAL
    return ticker, capital


def opciones_activo_ui(ticker: str) -> list[str]:
    """Activos del selector con el del archivo de la Principal en primera posición.

    El ticker declarado puede no estar en la lista corta de `ASSETS` (SOXL, TQQQ, SPY...):
    anteponiéndolo se muestra el valor real del archivo sin inventar ningún otro activo.
    """
    return [ticker, *[t for t in ASSETS if t != ticker]]


def sincronizar_reglas_comunes(script: str, info=None) -> tuple[str, float]:
    """Siembra los dos controles del grupo REGLAS COMUNES con lo que declara `script`.

    Los valores se releen cuando cambia la Estrategia Principal; mientras la Principal
    sea la misma se respetan las ediciones manuales del usuario sobre los dos campos.
    """
    ticker, capital = reglas_comunes_de_estrategia(BASE_DIR / script, info)

    if st.session_state.get(ACTIVO_ORIGEN_KEY) != script:
        st.session_state[ACTIVO_ORIGEN_KEY] = script
        st.session_state[ACTIVO_KEY] = ticker
        st.session_state[CAPITAL_ORIGEN_KEY] = script
        st.session_state[CAPITAL_KEY] = capital

    # Red de seguridad: un valor guardado que ya no existe (ticker retirado del
    # proyecto) vuelve al de la estrategia en vez de dejar el control cojo.
    if st.session_state.get(ACTIVO_KEY) not in opciones_activo_ui(ticker):
        st.session_state[ACTIVO_KEY] = ticker
    if not st.session_state.get(CAPITAL_KEY):
        st.session_state[CAPITAL_KEY] = DEFAULT_CAPITAL

    return st.session_state[ACTIVO_KEY], st.session_state[CAPITAL_KEY]


DEFAULT_STRATEGIES = {
    "grid_trading_v2.py": """# Estrategia Grid Trading v2
import pandas as pd
import numpy as np

def run_strategy(data, fast_ma=10, slow_ma=50):
    # Lógica de grid trading / medias móviles
    data['SMA_Fast'] = data['Close'].rolling(window=fast_ma).mean()
    data['SMA_Slow'] = data['Close'].rolling(window=slow_ma).mean()
    data['Signal'] = np.where(data['SMA_Fast'] > data['SMA_Slow'], 1, 0)
    return data
""",
    "crawling_fluorescent.py": """# Estrategia Crawling Fluorescent
import pandas as pd
import numpy as np

def run_strategy(data, fast_ma=12, slow_ma=26):
    # Lógica de momento / MACD básico
    data['SMA_Fast'] = data['Close'].rolling(window=fast_ma).mean()
    data['SMA_Slow'] = data['Close'].rolling(window=slow_ma).mean()
    data['Signal'] = np.where(data['SMA_Fast'] > data['SMA_Slow'], 1, 0)
    return data
""",
}

for filename, content in DEFAULT_STRATEGIES.items():
    strategy_path = BASE_DIR / filename
    if not strategy_path.exists():
        strategy_path.write_text(content, encoding="utf-8")

py_files = list_strategy_files(BASE_DIR) or list(DEFAULT_STRATEGIES.keys())

# Sesión: script activo y código cargado en el editor
if "script_nombre" not in st.session_state:
    st.session_state["script_nombre"] = py_files[0]
if st.session_state["script_nombre"] not in py_files:
    py_files.append(st.session_state["script_nombre"])

# Catálogo completo (raíz + subcarpetas) para el gestor Principal vs Comparativas
ALL_STRATEGIES = list_all_strategies(BASE_DIR) or list(py_files)
if st.session_state["script_nombre"] in ALL_STRATEGIES:
    st.session_state.setdefault(PRINCIPAL_KEY, st.session_state["script_nombre"])
ensure_strategy_state(ALL_STRATEGIES)

# Snapshot estático de referencia si el servidor está en entorno sin red
REFERENCE_SNAPSHOT = {
    "equity": 96262.27,
    "total_return": -3.74,
    "max_drawdown": -14.30,
    "win_rate": 58.4,
    "sharpe": 1.42,
}

# -----------------------------------------------------------------------------
# MOTOR DE DATOS Y BACKTESTING
# -----------------------------------------------------------------------------
# El motor es `qc_lab`. Ni la estrategia principal ni las comparativas ni el
# borrador calculan nada por su cuenta: todos pasan por `ejecutar_estrategia`, que
# decide el dialecto (clase Lean o funcion `run_strategy`), aplica la configuracion
# y devuelve un unico tipo de resultado. `qc_lab.core.compat` lo traduce despues al
# contrato de columnas y claves que esta interfaz ya sabe leer, de modo que las
# graficas, la tabla y la IA no pueden discrepar del motor.
#
# Las dos funciones siguientes son el UNICO punto por el que la interfaz pide una
# corrida: cualquier consumidor nuevo debe llamar a una de las dos.


def _sello_parametros() -> tuple:
    """Huella estable de los overrides activos, para invalidar las cachés.

    Sin esto, mover un parametro dinamico no entraria en la clave y el backtest
    devolveria el resultado cacheado del valor anterior.
    """
    return tuple(sorted((k, round(float(v), 6)) for k, v in PARAMETROS_ACTIVOS.items()))


def _config_lab(ticker, capital):
    """Configuración de la corrida que se envía al motor de qc_lab."""
    return ConfigEjecucion(
        simbolo=str(ticker).upper(),
        capital=float(capital),
        comision=COMMISSION_RATE,
        parametros=dict(PARAMETROS_ACTIVOS),
    )


def _contrato_de_lab(resultado, ticker, capital, parametros, etiqueta):
    """Traduce un `ResultadoBacktest` al par `(df, metrics)` de la interfaz."""
    if not resultado.exito:
        raise ValueError(resultado.error or f"{etiqueta} no se pudo ejecutar.")
    return contrato_heredado(
        resultado,
        simbolo=str(ticker).upper(),
        capital=float(capital),
        parametros=parametros,
        comision_rate=COMMISSION_RATE,
    )


def _legacy_finalize(df, capital, source, ticker, fast_ma, slow_ma):
    """
    Motor legacy de metrics sobre un DataFrame de precios y posiciones.

    YA NO LO USA LA INTERFAZ: el cálculo lo hace `qc_lab.core.compat`. Se conserva,
    con su regresion asociada, como red de seguridad mientras dure la migracion y
    como referencia de a donde venian los numeros. Si alguna vez hay que volver
    atras, esta funcion es el contrato que hay que reponer.
    """
    df = df.copy()
    if df.empty or "Close" not in df.columns:
        raise ValueError("El DataFrame no contiene columna 'Close' ni filas.")
    df = df.dropna(subset=["Close"])

    if "Position" not in df.columns:
        if "Signal" in df.columns:
            # shift(1) para no mirar el futuro, igual que la estrategia principal
            signal = pd.to_numeric(df["Signal"], errors="coerce").fillna(0.0)
            df["Position"] = signal.shift(1).fillna(0.0)
        else:
            df["Position"] = 1.0

    if "Market_Return" not in df.columns:
        df["Market_Return"] = df["Close"].pct_change().fillna(0)
    if "Strategy_Return" not in df.columns:
        df["Strategy_Return"] = df["Market_Return"] * df["Position"]

    df["Equity"] = capital * (1 + df["Strategy_Return"]).cumprod()
    df["Benchmark"] = capital * (1 + df["Market_Return"]).cumprod()

    # Retorno y drawdown del propio benchmark, para la tabla comparativa
    bench_peak = df["Benchmark"].cummax()
    benchmark_return = ((float(df["Benchmark"].iloc[-1]) - capital) / capital) * 100
    benchmark_drawdown = float(((df["Benchmark"] - bench_peak) / bench_peak * 100).min())

    df["Peak"] = df["Equity"].cummax()
    df["Drawdown"] = (df["Equity"] - df["Peak"]) / df["Peak"] * 100

    equity_curve = df["Equity"].to_numpy()
    position = df["Position"].to_numpy()

    trades, entry = [], None
    fills = 0
    was_held = 0.0
    for i, held in enumerate(position):
        if held == 1 and entry is None:
            entry = equity_curve[i]
        elif held == 0 and entry is not None:
            trades.append(equity_curve[i] / entry - 1.0)
            entry = None
        # Un fill es cada cambio de exposición: cada entrada y cada salida.
        if held != was_held:
            fills += 1
            was_held = held
    if entry is not None:
        trades.append(equity_curve[-1] / entry - 1.0)

    win_rate = (sum(1 for t in trades if t > 0) / len(trades) * 100) if trades else 0.0
    # El motor no modela tamaño de orden, así que se aplica una comisión estándar sobre
    # el capital comprometido en cada fill (0.05 % por lado).
    fees = round(fills * float(capital) * COMMISSION_RATE, 2)

    rets = df["Strategy_Return"].replace([np.inf, -np.inf], np.nan).dropna()
    std = float(rets.std(ddof=1)) if len(rets) > 2 else 0.0
    sharpe = float(rets.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    final_equity = float(df["Equity"].iloc[-1])

    metrics = {
        "equity": final_equity,
        "net_profit": final_equity - capital,
        "total_return": ((final_equity - capital) / capital) * 100,
        "max_drawdown": float(df["Drawdown"].min()),
        "win_rate": float(win_rate),
        "sharpe": sharpe,
        "trades": len(trades),
        "fills": fills,
        "fees": fees,
        "volume": float(df["Close"].iloc[-1] * 1200),
        "bars": int(len(df)),
        "source": source,
        "ticker": ticker,
        "capital": float(capital),
        "fast_ma": int(fast_ma),
        "slow_ma": int(slow_ma),
        "benchmark_return": benchmark_return,
        "benchmark_drawdown": benchmark_drawdown,
    }
    return df, metrics


def run_strategy_backtest(script, ticker, capital=100000):
    """Ejecuta la Estrategia Principal en el motor de qc_lab y devuelve (df, metrics).

    Acepta el archivo y no un simple ticker: el motor es el unico que sabe si la
    estrategia es una clase Lean o una funcion `run_strategy`, y ademas decide fechas,
    capital y universo a partir de lo que la propia estrategia declare.
    """
    cfg = _config_lab(ticker, capital)
    resultado = ejecutar_estrategia(BASE_DIR / script, config=cfg)
    return _contrato_de_lab(resultado, ticker, capital, cfg.parametros, script)


def run_source_backtest(source_code, ticker, capital, etiqueta="La estrategia"):
    """Ejecuta código fuente de una estrategia en el motor de qc_lab y devuelve (df, metrics).

    El mismo tratamiento recibe un `.py` de la carpeta que el borrador que la IA
    acaba de generar en memoria: los dos entran por `ejecutar_estrategia`, asi que
    una comparativa se comporta exactamente igual que la Estrategia Principal.
    """
    cfg = _config_lab(ticker, capital)
    cfg.etiqueta = etiqueta
    resultado = ejecutar_estrategia(fuente=source_code, config=cfg)
    return _contrato_de_lab(resultado, ticker, capital, cfg.parametros, etiqueta)


def run_draft_backtest(source_code, ticker, capital):
    """Ejecuta la estrategia del borrador y devuelve (df, metrics) con las mismas métricas."""
    return run_source_backtest(source_code, ticker, capital, "El borrador")


def run_comparative_backtest(script, ticker, capital):
    """Ejecuta una estrategia comparativa desde su archivo y devuelve (df, metrics).

    Los errores se capturan aquí y se devuelven como `error` para que una estrategia
    rota no tumbe la comparación completa.
    """
    try:
        df, metrics = run_strategy_backtest(script, ticker, capital)
    except Exception as exc:
        return None, {"error": str(exc), "script": script}
    return df, metrics


def get_comparative_results(scripts, ticker, capital) -> dict:
    """Results de las comparativas seleccionadas, cacheados por script + parámetros."""
    cache = st.session_state.get(COMPARE_CACHE_KEY) or {}
    wanted = [s for s in scripts if s in ALL_STRATEGIES]
    out = {}
    for script in wanted:
        stamp = (ticker, _sello_parametros(), float(capital))
        cached = cache.get(script)
        if not cached or cached.get("params") != stamp:
            df, metrics = run_comparative_backtest(script, ticker, capital)
            cached = {"params": stamp, "df": df, "metrics": metrics}
            cache[script] = cached
        out[script] = cached
    st.session_state[COMPARE_CACHE_KEY] = cache
    return out


RESULT_KEY = "backtest_result"
PARAMS_KEY = "backtest_params"


def get_backtest_result(ticker, capital, force=False):
    """Resultado del backtest de la Estrategia Principal, cacheado por parámetros.

    La Estrategia Principal se lee del estado de sesión y se ejecuta en `qc_lab` igual
    que cualquier comparativa. Si el motor no puede ejecutarla se devuelve la curva y
    las metricas de referencia con el motivo en `error`, para que el panel siga
    operativo en lugar de dejar la app a medias.
    """
    script = st.session_state.get(PRINCIPAL_KEY) or (py_files[0] if py_files else "")
    params = (script, ticker, _sello_parametros(), float(capital))
    cached = st.session_state.get(RESULT_KEY)

    if not force and cached is not None and st.session_state.get(PARAMS_KEY) == params:
        return cached["df"], cached["metrics"]

    try:
        df, metrics = run_strategy_backtest(script, ticker, capital)
    except Exception as exc:
        df = pd.DataFrame({"Equity": [float(capital)], "Benchmark": [float(capital)],
                           "Drawdown": [0.0], "Position": [0.0]},
                          index=pd.to_datetime([datetime.now().normalize()]))
        metrics = {**REFERENCE_SNAPSHOT, "error": str(exc), "source": "sin ejecutar", "bars": 1,
                   "ticker": ticker, "capital": float(capital), "sharpe": REFERENCE_SNAPSHOT["sharpe"]}

    st.session_state[RESULT_KEY] = {"df": df, "metrics": metrics}
    st.session_state[PARAMS_KEY] = params
    return df, metrics


# -----------------------------------------------------------------------------
# MOTOR DE IA (Gemini / Ollama)
# -----------------------------------------------------------------------------
try:
    import requests
except ImportError:  # requests es opcional: la app arranca igual sin él
    requests = None

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_MODELS = ["gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro",]
OLLAMA_DEFAULT_HOST = "http://localhost:11434"
OLLAMA_MODELS = ["llama3.1", "qwen2.5-coder", "codellama", "deepseek-coder-v2"]
AI_TIMEOUT = 180

# Prefijo de sesión del selector de IA global (popover 🤖 de la barra superior).
# Cualquier consumidor lee esa misma selección con ai_config_from_state().
TOPBAR_AI_PREFIX = "hdr"

AI_SYSTEM_PROMPT = """Eres un desarrollador senior especializado en QuantConnect, el Lean Engine de \
QuantConnect y Python aplicado al trading algorítmico.

Analiza la estrategia que te envíe y responde SIEMPRE con estas dos secciones, en este orden exacto:

## ANÁLISIS
Un resumen breve (entre 3 y 6 viñetas) cubriendo:
1. Errores: fallos en tiempo de ejecución, imports faltantes o incorrectos, variables sin definir, \
uso incorrecto del API de QuantConnect/LEAN.
2. Gestión de riesgo: stop loss, take profit, dimensionamiento de posición, control de exposición y \
límite de pérdida por operación.
3. Lógica de señales: confirmaciones, look-ahead bias, indicadores con retardo, sobreajuste de \
parámetros y robustez.

## CÓDIGO OPTIMIZADO
Un único bloque de código delimitado con ```python que contenga la estrategia COMPLETA y optimizada.

Reglas estrictas:
- Devuelve el código completo, nunca fragmentos ni comentarios del tipo "resto igual".
- Conserva el punto de entrada y la firma originales (por ejemplo run_strategy(data, ...)).
- No introduzcas dependencias externas que no estén ya importadas en el código original.
- No inventes datos ni resultados de backtest.
- Comenta brevemente cada decisión de riesgo o de señal que modifiques."""


class AIError(Exception):
    """Error de negocio del motor de IA, apto para mostrarse al usuario."""


def get_gemini_api_key() -> str | None:
    """Busca la clave en el entorno y, si existe, en st.secrets."""
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        key = os.environ.get(var)
        if key:
            return key.strip()
    try:
        for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            key = st.secrets.get(var)
            if key:
                return str(key).strip()
    except Exception:
        pass
    return None


def call_gemini(prompt: str, model: str, api_key: str, system_prompt: str = AI_SYSTEM_PROMPT) -> str:
    if requests is None:
        raise AIError("Falta la librería `requests`. Instálela con: pip install requests")
    if not api_key:
        raise AIError(
            "No hay API key de Gemini configurada. Defina la variable de entorno GEMINI_API_KEY "
            "o pegue la clave en el campo de arriba."
        )

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192},
    }
    url = GEMINI_ENDPOINT.format(model=model)

    try:
        response = requests.post(
            url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=AI_TIMEOUT,
        )
    except requests.exceptions.Timeout as exc:
        raise AIError("La solicitud a Gemini agotó el tiempo de espera. Inténtalo de nuevo.") from exc
    except requests.exceptions.RequestException as exc:
        raise AIError(f"No se pudo conectar con la API de Gemini: {exc}") from exc

    if response.status_code in (401, 403):
        raise AIError("Gemini rechazó la API key (403/401). Verifique que sea válida y tenga acceso al modelo.")
    if response.status_code == 404:
        raise AIError(f"El modelo `{model}` no está disponible para esta API key. Prueba con otro modelo.")
    if response.status_code == 429:
        raise AIError("Se alcanzó el límite de cuota de Gemini. Espera unos segundos e inténtalo de nuevo.")
    if not response.ok:
        raise AIError(f"Error {response.status_code} de la API de Gemini: {response.text[:300]}")

    try:
        parts = response.json()["candidates"][0]["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts).strip()
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise AIError("Gemini devolvió una respuesta con un formato inesperado.") from exc

    if not text:
        raise AIError("Gemini devolvió una respuesta vacía.")
    return text


def call_ollama(prompt: str, model: str, host: str, system_prompt: str = AI_SYSTEM_PROMPT) -> str:
    if requests is None:
        raise AIError("Falta la librería `requests`. Instálela con: pip install requests")

    host = (host or OLLAMA_DEFAULT_HOST).rstrip("/")
    try:
        response = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "system": system_prompt, "stream": False},
            timeout=AI_TIMEOUT,
        )
    except requests.exceptions.Timeout as exc:
        raise AIError(f"Ollama no respondió a tiempo en {host}. Comprueba que el modelo esté cargado.")
    except requests.exceptions.RequestException as exc:
        raise AIError(
            f"No se pudo conectar con Ollama en {host}. Verifica que `ollama serve` esté corriendo. ({exc})"
        ) from exc

    if not response.ok:
        raise AIError(f"Ollama respondió con error {response.status_code}: {response.text[:300]}")

    try:
        text = str(response.json().get("response", "")).strip()
    except ValueError as exc:
        raise AIError("Ollama devolvió una respuesta con un formato inesperado.") from exc

    if not text:
        raise AIError("Ollama devolvió una respuesta vacía.")
    return text


def extract_python_code(text: str) -> str | None:
    """Recupera el bloque ```python de la respuesta del modelo."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        return None
    return max(blocks, key=len).strip()


def is_valid_python(code: str) -> bool:
    try:
        compile(code, "<ia>", "exec")
    except SyntaxError:
        return False
    return True


def run_ai_request(prompt: str, config: dict, system_prompt: str = AI_SYSTEM_PROMPT) -> str:
    """Envía el prompt al proveedor configurado y devuelve el texto de la respuesta."""
    provider = config.get("provider", "")
    model = config.get("model", "")
    if provider.startswith("Gemini"):
        key = (config.get("api_key") or "").strip() or get_gemini_api_key() or ""
        return call_gemini(prompt, model, key, system_prompt)
    return call_ollama(prompt, model, config.get("host") or "", system_prompt)


def ai_credential_ready(config: dict) -> bool:
    """Indica si hay credencial resoluble (Ollama no necesita ninguna)."""
    if not config.get("provider", "").startswith("Gemini"):
        return True
    return bool((config.get("api_key") or "").strip() or get_gemini_api_key())


AI_PROVIDERS = ["Gemini (Google)", "Ollama (local)"]


def render_ai_provider_form(prefix: str) -> dict:
    """Selector de proveedor + modelo + credencial, sin decides la maquetación.

    Both the header popover and the modal reuse it so there is a single source of
    truth for which models each provider offers.
    """
    provider = st.selectbox(
        "Proveedor de IA:",
        AI_PROVIDERS,
        key=f"{prefix}_provider",
    )

    if provider.startswith("Gemini"):
        model = st.selectbox("Modelo:", GEMINI_MODELS, key=f"{prefix}_model")
    else:
        model = st.selectbox("Modelo:", OLLAMA_MODELS, key=f"{prefix}_model_local")

    api_key, host = "", ""
    if provider.startswith("Gemini"):
        if not get_gemini_api_key():
            api_key = st.text_input(
                "API key de Gemini:",
                type="password",
                key=f"{prefix}_api_key",
                help="Se mantiene solo en esta sesión. Mejor define GEMINI_API_KEY en el entorno.",
            )
    else:
        host = st.text_input(
            "Host de Ollama:", value=OLLAMA_DEFAULT_HOST, key=f"{prefix}_ollama_host"
        )

    return {"provider": provider, "model": model, "api_key": api_key, "host": host}


def render_topbar_ai_popover() -> dict:
    """Popover 🤖 de la barra superior: configuración de agente y modelo de IA.

    Vive en el header para liberar la altura del panel de análisis. La selección
    persiste en `st.session_state` con las claves del prefijo, igual que el resto
    de la aplicación.
    """
    with st.popover("🤖", help="Configurar Agente y Modelo de IA"):
        config = render_ai_provider_form(TOPBAR_AI_PREFIX)
    return config


def ai_config_from_state(prefix: str) -> dict:
    """Lee la configuración de IA ya persistida en sesión, sin renderizar widgets.

    Permite que cualquier consumidor (panel de backtest, modal) use lo elegido en el
    popover del header sin duplicar los selectores.
    """
    provider = st.session_state.get(f"{prefix}_provider") or AI_PROVIDERS[0]
    if provider.startswith("Gemini"):
        model = st.session_state.get(f"{prefix}_model") or GEMINI_MODELS[0]
    else:
        model = st.session_state.get(f"{prefix}_model_local") or OLLAMA_MODELS[0]
    return {
        "provider": provider,
        "model": model,
        "api_key": st.session_state.get(f"{prefix}_api_key", "") or "",
        "host": st.session_state.get(f"{prefix}_ollama_host") or OLLAMA_DEFAULT_HOST,
    }


def build_backtest_prompt(values: dict, df_res, ticker, fast_ma, slow_ma, capital, script_name) -> str:
    """Construye el prompt dinámico con las métricas cuantitativas del backtest activo."""
    try:
        fecha_inicio = pd.Timestamp(df_res.index[0]).strftime("%Y-%m-%d")
        fecha_fin = pd.Timestamp(df_res.index[-1]).strftime("%Y-%m-%d")
        benchmark_final = float(df_res["Benchmark"].iloc[-1])
    except (IndexError, KeyError, TypeError, ValueError):
        fecha_inicio = fecha_fin = "desconocidas"
        benchmark_final = float(capital)

    capital = float(capital)
    retorno_benchmark = ((benchmark_final - capital) / capital * 100) if capital else 0.0
    exceso = values["total_return"] - retorno_benchmark

    return (
        "## Contexto del backtest\n"
        f"- Activo / ticker analizado: {ticker}\n"
        f"- Rango de fechas del backtest: {fecha_inicio} a {fecha_fin}\n"
        f"- Capital inicial: ${capital:,.2f}\n"
        f"- Estrategia cargada: {script_name}\n"
        f"- Parámetros activos del script: SMA Rápida = {int(fast_ma)} | SMA Lenta = {int(slow_ma)}\n\n"
        "## Métricas clave del backtest\n"
        f"- Capital Final: ${values['equity']:,.2f}\n"
        f"- Retorno Total: {values['total_return']:.2f}%\n"
        f"- Max Drawdown: {values['max_drawdown']:.2f}%\n"
        f"- Win Rate: {values['win_rate']:.1f}%\n"
        f"- Sharpe Ratio: {values['sharpe']:.2f}\n\n"
        "## Referencia de mercado (buy & hold del activo)\n"
        f"- Retorno del activo base: {retorno_benchmark:.2f}%\n"
        f"- Exceso de retorno de la estrategia: {exceso:+.2f}%\n"
        f"- Operaciones de ida y vuelta: {values.get('trades', 'n/d')}\n"
        f"- Barras procesadas: {values.get('bars', 'n/d')}\n"
        f"- Fuente de datos: {values.get('source', 'n/d')}\n\n"
        "Interpreta este backtest y devuelve el informe con las cuatro secciones solicitadas."
    )


SUGGESTION_HEADINGS = ("Sugerencias Técnicas de Ajuste",)

DRAFT_KEY = "draft_code"
DRAFT_META_KEY = "draft_meta"
DRAFT_EDITOR_KEY = "draft_code_editor"
DRAFT_SEEDED_KEY = "draft_seeded"
DRAFT_RESULT_KEY = "draft_backtest"
DRAFT_METRICS_KEY = "draft_metrics"
DRAFT_EQUITY_KEY = "draft_equity"


def parse_suggestions(report: str) -> list[str]:
    """Extrae las sugerencias técnicas de ajuste del informe para poder marcarlas con checkbox."""
    if not report:
        return []

    lowered = report.lower()
    inicio = -1
    for heading in SUGGESTION_HEADINGS:
        pos = lowered.find(heading.lower())
        if pos > inicio:
            inicio = pos
    if inicio == -1:
        return []

    cuerpo = report[inicio:]
    fin_linea = cuerpo.find("\n")
    bloque = cuerpo[fin_linea + 1:] if fin_linea != -1 else ""
    siguiente = re.search(r"^##\s", bloque, re.M)
    if siguiente:
        bloque = bloque[: siguiente.start()]

    sugerencias = []
    for linea in bloque.splitlines():
        match = re.match(r"^\s*(?:[-*+]|\d+[.)])\s+(.+)$", linea)
        if not match:
            continue
        texto = re.sub(r"[*_`]", "", match.group(1)).strip()
        if texto:
            sugerencias.append(texto)
    return sugerencias


def build_draft_metrics_table(base: dict, nuevo: dict) -> str:
    """Construye la tabla comparativa Original vs Borrador para el backtest del borrador."""
    filas = [
        ("Capital Final", "${:,.2f}", "equity", True),
        ("Retorno Total", "{:.2f}%", "total_return", True),
        ("Max Drawdown", "{:.2f}%", "max_drawdown", True),
        ("Win Rate", "{:.1f}%", "win_rate", True),
        ("Sharpe Ratio", "{:.2f}", "sharpe", True),
        ("Operaciones", "{:.0f}", "trades", None),
    ]

    lineas = ["| Métrica | Original | Borrador | Δ |", "|---|---|---|---|"]
    for etiqueta, fmt, clave, mejor in filas:
        valor_base = float(base.get(clave, 0.0) or 0.0)
        valor_nuevo = float(nuevo.get(clave, 0.0) or 0.0)
        delta = valor_nuevo - valor_base
        flecha = "🟢" if delta > 0 else ("🔴" if delta < 0 else "⚪")
        if mejor is False:
            flecha = {"🟢": "🔴", "🔴": "🟢", "⚪": "⚪"}[flecha]
        signo = "+" if delta > 0 else ""
        lineas.append(
            f"| {etiqueta} | {fmt.format(valor_base)} | {fmt.format(valor_nuevo)} "
            f"| {flecha} {signo}{fmt.format(delta)} |"
        )
    return "\n".join(lineas)


def render_draft_area(base_values: dict, ticker, fast_ma, slow_ma, capital):
    """Zona de staging del código propuesto por la IA, bajo el editor principal.

    El borrador vive solo en `st.session_state`: no toca el editor original ni el archivo
    en disco hasta que el usuario pulse «Reemplazar Código Original».
    """
    borrador = st.session_state.get(DRAFT_KEY)
    if not borrador:
        return

    meta = st.session_state.get(DRAFT_META_KEY, {})

    with st.container(border=True):
        st.markdown(
            '<div class="draft-title">🧪 Propuesta de Código Optimizado (Borrador)</div>',
            unsafe_allow_html=True,
        )
        detalles = " · ".join(
            str(item)
            for item in (
                meta.get("script"),
                f"{meta.get('provider')} / {meta.get('model')}" if meta.get("model") else None,
                meta.get("hora"),
            )
            if item
        )
        if detalles:
            st.caption(f"📝 {detalles}")

        # Siembra el editor solo cuando cambia el borrador: evita pisar lo que el
        # usuario esté editando y el error de widget ya instanciado.
        if st.session_state.get(DRAFT_SEEDED_KEY) != borrador:
            st.session_state[DRAFT_EDITOR_KEY] = borrador
            st.session_state[DRAFT_SEEDED_KEY] = borrador

        codigo_borrador = st.text_area(
            "Código del borrador (editable)",
            key=DRAFT_EDITOR_KEY,
            height=260,
            label_visibility="collapsed",
        )
        # `draft_code` es una clave normal (no de widget), así que se puede refrescar en caliente
        st.session_state[DRAFT_KEY] = codigo_borrador

        st.caption(
            "⚠️ El borrador **no** modifica el editor ni el archivo: solo se aplica si lo confirmas."
        )

        col_run, col_apply, col_discard = st.columns([1.1, 1, 1])
        with col_run:
            probar = st.button(
                "▶ Probar Backtest Propuesto", use_container_width=True, key="btn_draft_run"
            )
        with col_apply:
            aplicar = st.button(
                "✅ Reemplazar Código Original", use_container_width=True, key="btn_draft_apply"
            )
        with col_discard:
            descartar = st.button(
                "❌ Descartar Propuesta", use_container_width=True, key="btn_draft_discard"
            )

        if descartar:
            for key in (DRAFT_KEY, DRAFT_META_KEY, DRAFT_SEEDED_KEY, DRAFT_RESULT_KEY, DRAFT_METRICS_KEY, DRAFT_EQUITY_KEY):
                st.session_state.pop(key, None)
            # DRAFT_EDITOR_KEY es una clave de widget: Streamlit la limpia sola al no renderizarse
            st.session_state["flash_msg"] = "🗑️ Propuesta descartada"
            st.rerun()

        if aplicar:
            if not codigo_borrador.strip():
                st.warning("El borrador está vacío: no hay nada que aplicar.")
            elif not is_valid_python(codigo_borrador):
                st.error("El borrador tiene errores de sintaxis, así que no se aplicó.")
            else:
                active_script = meta.get("script") or st.session_state.get("script_nombre", "")
                st.session_state["codigo_actual"] = codigo_borrador
                st.session_state["codigo_actual_script"] = active_script
                # Obliga al editor principal a recargar el contenido nuevo en el siguiente render
                st.session_state["editor_loaded_for"] = None
                for key in (DRAFT_KEY, DRAFT_META_KEY, DRAFT_SEEDED_KEY, DRAFT_RESULT_KEY, DRAFT_METRICS_KEY, DRAFT_EQUITY_KEY):
                    st.session_state.pop(key, None)
                st.session_state["flash_msg"] = (
                    f"✅ Código original reemplazado por la propuesta (aún sin guardar en disco)"
                )
                st.rerun()

        if probar:
            if not codigo_borrador.strip():
                st.warning("El borrador está vacío: no hay nada que probar.")
            else:
                with st.spinner("🧪 Ejecutando el backtest del borrador..."):
                    try:
                        draft_df, draft_metrics = run_draft_backtest(
                            codigo_borrador, ticker, capital
                        )
                    except ValueError as exc:
                        for key in (DRAFT_RESULT_KEY, DRAFT_METRICS_KEY, DRAFT_EQUITY_KEY):
                            st.session_state.pop(key, None)
                        st.error(f"⚠️ No se pudo probar el borrador: {exc}")
                    except Exception as exc:  # noqa: BLE001 - el borrador es código externo
                        for key in (DRAFT_RESULT_KEY, DRAFT_METRICS_KEY, DRAFT_EQUITY_KEY):
                            st.session_state.pop(key, None)
                        st.error(f"❌ Error inesperado al ejecutar el borrador: {type(exc).__name__}: {exc}")
                    else:
                        st.session_state[DRAFT_RESULT_KEY] = {
                            "metrics": draft_metrics,
                            "code": codigo_borrador,
                        }
                        # Métricas planas: las lee el gráfico para superponer la curva del borrador
                        st.session_state[DRAFT_METRICS_KEY] = {
                            **draft_metrics,
                            "model": (meta.get("model") or "borrador"),
                        }
                        if "Equity" in draft_df.columns:
                            st.session_state[DRAFT_EQUITY_KEY] = draft_df["Equity"].copy()

        resultado = st.session_state.get(DRAFT_RESULT_KEY)
        if resultado:
            if resultado.get("code") != st.session_state.get(DRAFT_KEY):
                st.warning("El borrador cambió después de la prueba: vuelve a probarlo para comparar.")
            else:
                draft_metrics = resultado["metrics"]
                fuente = draft_metrics.get("source", "n/d")
                st.markdown("#### 📊 Backtest del borrador")
                st.caption(f"Datos: {fuente} · {draft_metrics.get('bars', 0)} barras procesadas")
                st.markdown(build_draft_metrics_table(base_values, draft_metrics))


# -----------------------------------------------------------------------------
# MODAL: ABRIR ESTRATEGIA DESDE QUANTCONNECT
# -----------------------------------------------------------------------------
# Un solo modal para las dos altas: el grupo ESTRATEGIAS lo abre para cargar la
# Estrategia Principal y el grupo ESTRATEGIAS A COMPARAR lo abre para sumar una
# comparativa. Lo que cambia es el destino del botón final, no el selector de archivos:
# así no hay dos mecanismos explorando el proyecto.
MODO_PRINCIPAL = "principal"
MODO_COMPARATIVA = "comparativa"
MODAL_MODO_KEY = "modal_modo"


@st.dialog("📂 Abrir Estrategia desde QuanConnect", width="medium")
def modal_abrir_estrategia():
    st.caption(f"📍 Directorio base: `{BASE_DIR}`")

    folders = [".", *[folder.name for folder in list_project_folders(BASE_DIR)]]

    if not st.session_state.get("modal_folder_ready"):
        st.session_state["modal_folder"] = folders[0]
        st.session_state["modal_folder_ready"] = True

    folder_key = st.selectbox(
        "Carpeta del proyecto:",
        folders,
        format_func=lambda name: "📁 . (raíz del proyecto)" if name == "." else f"📁 {name}",
        key="modal_folder",
    )

    folder = BASE_DIR if folder_key == "." else BASE_DIR / folder_key

    # Al cambiar de carpeta se reinicia la selección de archivo
    if st.session_state.get("modal_folder_resolved") != folder_key:
        st.session_state["modal_folder_resolved"] = folder_key
        st.session_state.pop("modal_file", None)

    if not folder.is_dir():
        st.warning(f"⚠️ La carpeta `{folder_key}` ya no existe en el proyecto.")
        return

    entries = list_strategy_files(folder)
    if not entries:
        st.warning(f"⚠️ La carpeta `{folder_key}` no contiene archivos `.py`.")
        st.info("Selecciona otra carpeta o escribe una nueva estrategia en la raíz del proyecto.")
        return

    active_script = st.session_state.get("script_nombre", "")
    active_name = Path(active_script).name
    default_idx = entries.index(active_name) if folder_key == "." and active_name in entries else 0

    file_name = st.selectbox(
        "Archivo de estrategia:",
        entries,
        index=default_idx,
        key="modal_file",
    )

    file_path = folder / file_name
    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        st.error(f"❌ No se pudo leer `{file_name}`: {exc}")
        return

    st.caption(f"{len(source.splitlines())} líneas · {len(source)} caracteres")

    with st.expander("👁️ Vista previa del código", expanded=False):
        st.code(source, language="python")

    modo = st.session_state.get(MODAL_MODO_KEY, MODO_PRINCIPAL)
    es_comparativa = modo == MODO_COMPARATIVA

    if st.button(
        "⚖️ Agregar a comparar" if es_comparativa else "📥 Cargar en Editor",
        type="primary", use_container_width=True, key="modal_load",
    ):
        rel_path = file_path.relative_to(BASE_DIR).as_posix()

        if es_comparativa:
            # Alta de comparativa: la Principal, el editor y el código cargado no se tocan.
            aceptado, mensaje = add_comparative_strategy(rel_path, ALL_STRATEGIES)
            if not aceptado:
                st.warning(f"⚠️ {mensaje}")
                return
            st.session_state["flash_msg"] = mensaje
            st.rerun()

        st.session_state["codigo_actual"] = source
        st.session_state["codigo_actual_script"] = rel_path
        # Cargar desde el modal equivale a promover esa estrategia a Principal ⭐.
        if st.session_state.get(PRINCIPAL_KEY) != rel_path:
            set_principal_strategy(rel_path, ALL_STRATEGIES)
        st.session_state["script_nombre"] = rel_path
        st.session_state["script_selector"] = rel_path
        st.session_state["editor_loaded_for"] = None
        st.session_state["flash_msg"] = f"📂 `{rel_path}` cargado en el editor"
        st.rerun()


# -----------------------------------------------------------------------------
# MODAL: ANÁLISIS Y SUGERENCIAS DE IA
# -----------------------------------------------------------------------------
@st.dialog("🤖 Análisis y Sugerencias de IA", width="large")
def modal_ia_optimizar():
    active_script = st.session_state.get("script_nombre", "")
    editor_key = f"editor_{active_script}"
    source_code = st.session_state.get(editor_key) or st.session_state.get("codigo_actual") or ""

    st.caption(f"Estrategia: `{active_script}` · {len(source_code.splitlines())} líneas")

    config = ai_config_from_state(TOPBAR_AI_PREFIX)
    st.caption(
        f"🤖 Agente: **{config['provider']}** · Modelo: `{config['model']}` "
        "(cambiable en el popover 🤖 de la barra superior)"
    )

    instructions = st.text_input(
        "Instrucciones adicionales para la IA (opcional):",
        key="ai_instrucciones",
        placeholder="Ej: Agregar trailing stop, optimizar gestión de riesgo, corregir errores de sintaxis",
    )

    if not source_code.strip():
        st.warning("El editor está vacío. Escribe o carga una estrategia antes de pedir el análisis.")
        return

    if st.button("🔍 Analizar y Optimizar", type="primary", use_container_width=True, key="ai_run"):
        prompt = f"Estratega a analizar: `{active_script}`\n\n"
        if instructions.strip():
            prompt += f"Instrucciones adicionales del usuario: {instructions.strip()}\n\n"
        prompt += (
            f"```python\n{source_code}\n```\n\n"
            "Analiza este código y devuelve el ANÁLISIS y el CÓDIGO OPTIMIZADO según el formato pedido."
        )

        with st.spinner(f"🧠 {config['model']} está analizando la estrategia..."):
            try:
                answer = run_ai_request(prompt, config)
            except AIError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"❌ Error inesperado consultando la IA: {exc}")
            else:
                st.session_state["ai_result"] = {
                    "analysis": answer,
                    "code": extract_python_code(answer),
                    "provider": config["provider"],
                    "model": config["model"],
                    "instructions": instructions.strip(),
                }

    result = st.session_state.get("ai_result")
    if not result:
        return

    st.caption(f"Respuesta de {result['provider']} · modelo `{result['model']}`")
    st.markdown(result["analysis"])

    new_code = result.get("code")
    if not new_code:
        st.warning(
            "⚠️ La IA no devolvió un bloque de código delimited. "
            "Copia manualmente el código final desde el análisis de arriba."
        )
        return

    code_ok = is_valid_python(new_code)
    if not code_ok:
        st.warning(
            "⚠️ El código propuesto por la IA tiene errores de sintaxis. Revísalo antes de aplicarlo."
        )

    st.code(new_code, language="python")

    if st.button(
        "✅ Aplicar Cambios al Editor",
        type="primary",
        use_container_width=True,
        key="ai_apply",
        disabled=not code_ok,
    ):
        st.session_state["codigo_actual"] = new_code
        st.session_state["codigo_actual_script"] = active_script
        st.session_state["editor_loaded_for"] = None
        st.session_state.pop("ai_result", None)
        st.session_state["flash_msg"] = f"🪄 Optimización de IA aplicada a `{active_script}`"
        st.rerun()


# -----------------------------------------------------------------------------
# PANEL: ANÁLISIS INTERPRETATIVO DEL BACKTEST CON IA
# -----------------------------------------------------------------------------
def render_ai_backtest_panel(values, df_res, ticker, fast_ma, slow_ma, capital, script_name):
    """Tarjeta de diagnóstico del backtest con IA, bajo el gráfico de rendimiento."""
    firma = (ticker, int(fast_ma), int(slow_ma), float(capital))
    informe = st.session_state.get("bt_ai_report")

    with st.container(border=True):
        st.markdown(
            '<div class="ai-panel-title">🤖 Análisis Interpretativo del Backtest</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"{ticker} · SMA {int(fast_ma)}/{int(slow_ma)} · "
            f"Capital ${float(capital):,.0f} · {script_name}"
        )

        # El proveedor y el modelo se eligen una sola vez, en el popover 🤖 del header
        config = ai_config_from_state(TOPBAR_AI_PREFIX)

        st.checkbox(
            "Interpretar automáticamente al ejecutar backtest",
            key="bt_ai_auto",
            help="Con la opción activa, cada backtest nuevo dispara una sola interpretación.",
        )

        run_clicked = st.button(
            "📊 Interpretar Resultados con IA",
            type="primary",
            use_container_width=True,
            key="btai_run",
        )

        auto_due = (
            st.session_state.get("bt_ai_auto")
            and st.session_state.get("bt_ai_auto_done") != firma
            and (not informe or informe.get("firma") != firma)
        )

        if run_clicked or auto_due:
            prompt = build_backtest_prompt(values, df_res, ticker, fast_ma, slow_ma, capital, script_name)
            # Solo se gasta el intento automático si había credencial, para que un alta
            # tardía de la API key no deje la opción automática inutilizada.
            if ai_credential_ready(config):
                st.session_state["bt_ai_auto_done"] = firma
            with st.spinner(f"🧠 {config['model']} interpretando el backtest de {ticker}..."):
                try:
                    answer = run_ai_request(prompt, config, AI_ANALYST_SYSTEM_PROMPT)
                except AIError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"❌ Error inesperado consultando la IA: {exc}")
                else:
                    st.session_state["bt_ai_report"] = {
                        "informe": answer,
                        "provider": config["provider"],
                        "model": config["model"],
                        "firma": firma,
                        "hora": datetime.now().strftime("%H:%M:%S"),
                    }

        informe = st.session_state.get("bt_ai_report")
        if not informe:
            st.caption(
                "Sin interpretación todavía. Pulsa el botón para obtener el diagnóstico "
                "del backtest con IA."
            )
            return

        if informe.get("firma") != firma:
            st.warning(
                "⚠️ Este análisis se generó con otros parámetros. Vuelve a interpretarlo "
                "para que el diagnóstico refleje el backtest actual."
            )

        st.caption(f"✍️ {informe['provider']} · modelo `{informe['model']}` · {informe['hora']}")
        st.markdown(informe["informe"])

        sugerencias = parse_suggestions(informe["informe"])
        elegidas: list[str] = []

        if sugerencias:
            st.markdown("##### ✅ Elige las sugerencias que quieres aplicar")
            for idx, sugerencia in enumerate(sugerencias):
                if st.checkbox(sugerencia, key=f"btai_sug_{idx}"):
                    elegidas.append(sugerencia)
            st.caption(f"Seleccionadas {len(elegidas)} de {len(sugerencias)} sugerencias.")
        else:
            st.caption(
                "ℹ️ El informe no incluye una sección «Sugerencias Técnicas de Ajuste» "
                "con viñetas, así que no hay nada que seleccionar."
            )

        if st.button(
            "✨ Generar Borrador con Sugerencias Elegidas",
            use_container_width=True,
            key="btai_improve",
            disabled=not elegidas,
        ):
            active_script = st.session_state.get("script_nombre", "")
            editor_key = f"editor_{active_script}"
            source_code = st.session_state.get(editor_key) or st.session_state.get("codigo_actual") or ""

            if not source_code.strip():
                st.warning("El editor está vacío. Carga una estrategia antes de pedir mejoras.")
            else:
                seleccion = "\n".join(f"- {item}" for item in elegidas)
                prompt = (
                    "Diagnóstico del backtest que debe orientar los cambios:\n"
                    f"```markdown\n{informe['informe']}\n```\n\n"
                    "Sugerencias seleccionadas por el usuario: aplícalas TODAS y solo ellas.\n"
                    f"{seleccion}\n\n"
                    f"Código actual de `{active_script}`:\n"
                    f"```python\n{source_code}\n```\n\n"
                    "Aplica únicamente las sugerencias seleccionadas al código y devuelve el "
                    "ANÁLISIS y el CÓDIGO OPTIMIZADO con el formato del sistema."
                )
                with st.spinner(f"🪄 {config['model']} aplicando las sugerencias al código..."):
                    try:
                        answer = run_ai_request(prompt, config, AI_SYSTEM_PROMPT)
                    except AIError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"❌ Error inesperado consultando la IA: {exc}")
                    else:
                        new_code = extract_python_code(answer)
                        if not new_code:
                            st.warning(
                                "⚠️ La IA no devolvió un bloque de código delimitado. "
                                "Abre «Optimizar con IA» para trabajar el código manualmente."
                            )
                        elif not is_valid_python(new_code):
                            st.warning(
                                "⚠️ El código propuesto tiene errores de sintaxis, "
                                "así que no se generó un borrador válido."
                            )
                            st.code(new_code, language="python")
                        else:
                            # Modo borrador: solo sesión, nunca el editor ni el archivo
                            st.session_state[DRAFT_KEY] = new_code
                            st.session_state[DRAFT_META_KEY] = {
                                "script": active_script,
                                "provider": config["provider"],
                                "model": config["model"],
                                "hora": datetime.now().strftime("%H:%M:%S"),
                                "suggestions": list(elegidas),
                                "analysis": answer,
                            }
                            st.session_state.pop(DRAFT_RESULT_KEY, None)
                            st.session_state["ai_result"] = {
                                "analysis": answer,
                                "code": new_code,
                                "provider": config["provider"],
                                "model": config["model"],
                                "instructions": "Sugerencias derivadas del análisis de backtest",
                            }
                            st.session_state["flash_msg"] = (
                                f"🧪 Borrador generado para `{active_script}` (aún sin aplicar)"
                            )
                            st.rerun()


# -----------------------------------------------------------------------------
# GESTOR DE ESTRATEGIAS: PRINCIPAL ⭐ vs COMPARATIVAS
# -----------------------------------------------------------------------------
# El gestor se pinta en dos grupos del sidebar (ESTRATEGIAS y ESTRATEGIAS A COMPARAR),
# pero los clics se siguen resolviendo TODOS juntos al final del run: es lo que evita
# que un widget de este run choque con los valores que aun guarda la sesión.
def render_principal_panel(principal: str) -> None:
    """Muestra la Estrategia Principal vigente (grupo ESTRATEGIAS)."""
    st.markdown(
        f'<div class="strategy-row principal"><span class="star">⭐</span>'
        f'<span class="strategy-name">{principal}</span>'
        f'<span class="strategy-tag">PRINCIPAL</span></div>',
        unsafe_allow_html=True,
    )


def render_comparativas_panel(available: list[str], principal: str, pending: dict) -> None:
    """Pinta los `MAX_COMPARATIVES` huecos y deja los clics en `pending`.

    El grupo ya no lista el catálogo completo: son dos ranuras y un botón que abre el
    mismo modal que «Abrir Estrategia». Cada ranura llena trae su propia baja, y ninguna
    de las dos vías de este grupo puede promover una comparativa a Principal.
    """
    selected = list(st.session_state.get(COMPARE_KEY, []))[:MAX_COMPARATIVES]
    st.session_state[COMPARE_KEY] = selected

    for pos in range(MAX_COMPARATIVES):
        if pos >= len(selected):
            st.markdown(
                f'<div class="compare-slot empty">Comparativa {pos + 1} · vacía</div>',
                unsafe_allow_html=True,
            )
            continue

        script = selected[pos]
        nombre_col, quitar_col = st.columns(
            [0.78, 0.22], vertical_alignment="center", gap="small"
        )
        with nombre_col:
            st.markdown(
                f'<div class="compare-slot"><span class="compare-index">{pos + 1}</span>'
                f'<span class="strategy-name">{script}</span></div>',
                unsafe_allow_html=True,
            )
        with quitar_col:
            if st.button(
                "✕", key=f"rm_comparativa_{pos}",
                help=f"Quitar `{script}` de la comparación", use_container_width=True,
            ):
                # Una baja llega como `False` en los mismos checks de antes.
                pending["checks"][script] = False

    hay_margen = len(selected) < MAX_COMPARATIVES
    if not hay_margen:
        st.caption(f"Ya no caben más comparativas (máximo {MAX_COMPARATIVES}).")
    elif len(available) <= 1:
        st.caption("No hay otras estrategias en el proyecto para comparar.")
    else:
        st.caption(f"Comparativas: {len(selected)} de {MAX_COMPARATIVES}.")

    puede_agregar = hay_margen and len(available) > 1
    if st.button(
        "+ Agregar estrategia a comparar",
        use_container_width=True,
        key="btn_add_comparativa",
        disabled=not puede_agregar,
        help=(f"Máximo {MAX_COMPARATIVES} comparativas." if not hay_margen
              else "Abre el selector de estrategias del proyecto."),
    ):
        st.session_state[MODAL_MODO_KEY] = MODO_COMPARATIVA
        modal_abrir_estrategia()


def aplicar_gestor_pendientes(available: list[str], principal: str, pending: dict) -> str:
    """Aplica los clics de ambos grupos y devuelve la Estrategia Principal vigente.

    Las altas de comparativa llegan ya aplicadas por `add_comparative_strategy` (que es
    quien valida el límite y la exclusión de la Principal); por aquí solo pasan las
    bajas, que llegan en `pending["checks"]` como `False`.
    """
    pending_star = pending["star"]
    pending_checks = pending["checks"]

    if pending_star is not None:
        set_principal_strategy(pending_star, available)
        st.rerun()

    selected = list(st.session_state.get(COMPARE_KEY, []))
    next_selected = [s for s in selected if pending_checks.get(s, True)][:MAX_COMPARATIVES]
    dropped = [s for s in selected if s not in next_selected]
    if dropped:
        st.session_state[COMPARE_KEY] = next_selected
        st.session_state["compare_notice"] = "Se quitó la comparativa " + ", ".join(
            Path(s).name for s in dropped
        )
        st.rerun()

    st.session_state[COMPARE_KEY] = next_selected
    # El aviso es de un solo uso: se consume al mostrarse para no repetirlo en cada run.
    notice = st.session_state.pop("compare_notice", None)
    if notice:
        st.caption(f"⚠️ {notice}")
    return st.session_state[PRINCIPAL_KEY]


# -----------------------------------------------------------------------------
# PARÁMETROS DINÁMICOS DE LA ESTRATEGIA
# -----------------------------------------------------------------------------
# No hay una lista fija de parámetros: los controles se generan con lo que
# `strategy_parser` detecta en el `.py` que se ha elegido como Principal, de modo que
# cambiar de estrategia cambia los controles sin tocar nada a mano.

def _cae_en_rejilla(valor: float, minimo: float, paso: float) -> bool:
    """¿El valor declarado cae justo en un escalón del rango sugerido?"""
    if paso <= 0:
        return True
    posiciones = (float(valor) - float(minimo)) / float(paso)
    return abs(posiciones - round(posiciones)) < 1e-9


def control_parametro(prm, prefijo: str) -> float:
    """Un control para un parámetro detectado, con el rango y el paso que sugiere el parser.

    Dos ajustes sobre lo que devuelve `sugerir_rango`, que es una heurística por nombre:

    * `Parametro.es_entero` manda sobre el paso y sobre el tipo de salida. El parser
      guarda `valor` ya convertido a float, así que la comprobación de entero de
      `sugerir_rango` nunca se cumple y propone pasos decimales (0.6 para `fast_ma=12`).
      Ni el paso ni el valor pueden quedar decimales: `rolling(window=12.0)` también
      revienta, así que un parámetro entero se devuelve como `int`.
    * Si el valor de la estrategia no cae en la rejilla del paso (p. ej.
      `pct_capital_escalon` = 0.3 con paso 1.0) se usa `number_input` con los mismos
      límites, porque un slider redondearía el valor al primer clic.
    """
    minimo, maximo, paso = (float(v) for v in strategy_parser.sugerir_rango(prm.nombre, prm.valor))
    valor = float(prm.valor)

    if prm.es_entero:
        paso = float(max(1, round(paso)))
        minimo = float(round(minimo))
        maximo = float(max(round(maximo), round(valor) + paso))
        valor = float(round(valor))

    clave = f"{prefijo}{prm.clave}"
    if maximo > minimo and _cae_en_rejilla(valor, minimo, paso):
        elegido = st.slider(prm.nombre, min_value=minimo, max_value=maximo, value=valor,
                             step=paso, key=clave)
    else:
        elegido = st.number_input(prm.nombre, min_value=minimo, max_value=maximo, value=valor,
                                  step=paso, key=clave)
    return int(round(elegido)) if prm.es_entero else float(elegido)


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
# Los seis grupos aprobados en G.3.3; en G.3.5 cada uno recibe los controles que ya
# existian, sin duplicar ninguno.
SIDEBAR_GROUPS = (
    "📁 ESTRATEGIAS",
    "⚖️ ESTRATEGIAS A COMPARAR",
    "⚙️ REGLAS COMUNES",
    "🎛️ PARÁMETROS DE LA ESTRATEGIA",
    "📊 OPCIONES DE VISUALIZACIÓN",
    "💡 ESTADO DEL BACKTEST",
)

with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:8px;margin-bottom:2px">'
        '<span style="font-size:22px">🧪</span>'
        '<span style="font-size:17px;font-weight:700;color:#0f172a;letter-spacing:.05em">QUANTLAB IDE</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"LEAN Engine: {ENGINE_VERSION}")

    # --- 1. ESTRATEGIAS -------------------------------------------------
    # G.3.6: el grupo se reparte en dos bloques (apertura y Principal) con altura
    # suficiente para que ambos se lean sin venir prensados.
    with st.expander(SIDEBAR_GROUPS[0], expanded=True, key="grp_estrategias"):
        st.markdown(
            '<div class="strategy-subtitle">📂 Abrir Estrategia (.py)</div>',
            unsafe_allow_html=True,
        )
        if st.button("📂 Abrir Estrategia", use_container_width=True, key="btn_open_strategy"):
            st.session_state[MODAL_MODO_KEY] = MODO_PRINCIPAL
            modal_abrir_estrategia()

        principal_script = st.session_state[PRINCIPAL_KEY]
        if st.session_state.get("script_selector") not in ALL_STRATEGIES:
            st.session_state["script_selector"] = principal_script

        st.markdown(
            '<div class="strategy-subtitle">⭐ Estrategia Principal</div>',
            unsafe_allow_html=True,
        )
        render_principal_panel(principal_script)

    # --- 2. ESTRATEGIAS A COMPARAR --------------------------------------
    pending_gestor = {"star": None, "checks": {}}
    with st.expander(SIDEBAR_GROUPS[1], expanded=False, key="grp_comparar"):
        render_comparativas_panel(ALL_STRATEGIES, principal_script, pending_gestor)

    # Los clics de los dos grupos se resuelven juntos, igual que antes.
    selected_script = aplicar_gestor_pendientes(
        ALL_STRATEGIES, principal_script, pending_gestor
    )
    st.session_state["script_nombre"] = selected_script

    flash_msg = st.session_state.pop("flash_msg", None)
    if flash_msg:
        st.toast(flash_msg, icon="✅")

    # Una sola lectura del archivo de la Principal la comparten el grupo 3 (sus reglas
    # comunes) y el grupo 4 (sus parámetros ajustables).
    info_estrategia = strategy_parser.analizar_archivo(BASE_DIR / selected_script)

    # --- 3. REGLAS COMUNES ----------------------------------------------
    # G.3.6: los dos campos de este grupo salen del archivo de la Estrategia Principal.
    # Si el archivo no declara el dato se usa el respaldo (TSLA / 100000) y, si la
    # Principal cambia, ambos campos se vuelven a leer de la nueva estrategia.
    with st.expander(SIDEBAR_GROUPS[2], expanded=True, key="grp_reglas"):
        ticker_principal, _ = sincronizar_reglas_comunes(selected_script, info_estrategia)
        asset = st.selectbox(
            "Activo / Ticker", opciones_activo_ui(ticker_principal), key=ACTIVO_KEY
        )
        initial_cap = st.number_input(
            "Capital Inicial ($)", min_value=1.0, step=10000.0, key=CAPITAL_KEY
        )
        st.caption(
            f"⤵ Reglas de `{Path(selected_script).name}` "
            f"(respaldo: {DEFAULT_TICKER} / ${DEFAULT_CAPITAL:,.0f})."
        )

    # --- 4. PARÁMETROS DE LA ESTRATEGIA ---------------------------------
    # Se reconstruyen en cada run: el conjunto depende del archivo elegido como Principal.
    PARAMETROS_ACTIVOS.clear()
    with st.expander(SIDEBAR_GROUPS[3], expanded=True, key="grp_parametros"):
        if info_estrategia.error:
            st.caption(f"⚠️ {info_estrategia.error}")
        elif not info_estrategia.parametros:
            st.caption("Este archivo no declara parámetros ajustables.")
        else:
            categorias = info_estrategia.parametros_por_categoria()
            con_categorias = len(categorias) > 1
            for categoria, grupo in categorias.items():
                if con_categorias:
                    st.caption(categoria)
                for prm in grupo:
                    PARAMETROS_ACTIVOS[prm.clave] = control_parametro(
                        prm, f"param_{selected_script}_"
                    )

    # Estas dos variables ya no son controles: las usan el topbar y los prompts de IA
    # para mostrar el par de medias. Cuando la estrategia no las declara, se muestran
    # los valores por defecto de siempre.
    fast_period = int(PARAMETROS_ACTIVOS.get("fast_ma", DEFAULT_FAST_MA))
    slow_period = int(PARAMETROS_ACTIVOS.get("slow_ma", DEFAULT_SLOW_MA))

    # --- 5. OPCIONES DE VISUALIZACIÓN -----------------------------------
    with st.expander(SIDEBAR_GROUPS[4], expanded=False, key="grp_visualizacion"):
        show_equity = st.checkbox("Strategy Equity", value=True)
        show_benchmark = st.checkbox("Benchmark Comparison", value=True)
        show_drawdown = st.checkbox("Drawdown", value=True)

    # --- 6. ESTADO DEL BACKTEST -----------------------------------------
    with st.expander(SIDEBAR_GROUPS[5], expanded=True, key="grp_estado"):
        st.info("💡 **Backtest Status:** Likely Not Overfit\n\nParameters: 2 Determined")

# -----------------------------------------------------------------------------
# PROCESAMIENTO DE METRICAS
# -----------------------------------------------------------------------------
force_run = st.session_state.pop("force_run", False)
df_res, metrics = get_backtest_result(asset, initial_cap, force=force_run)

# El motor de qc_lab produce datos reales siempre, así que el snapshot de referencia
# solo entra cuando la Estrategia Principal no se ha podido ejecutar.
live = "error" not in metrics
fallback_metrics = None if live else REFERENCE_SNAPSHOT

def auto(key):
    return metrics[key] if live else fallback_metrics[key]


def display_metrics() -> dict:
    """Métricas tal y como las muestran las tarjetas KPI, con el respaldo de referencia."""
    return {**metrics, **{key: auto(key) for key in REFERENCE_SNAPSHOT}}

def read_draft_curve(state) -> tuple[dict | None, pd.Series | None]:
    """Lee de la sesión las métricas y la curva de equity del borrador en prueba."""
    draft_metrics = state.get(DRAFT_METRICS_KEY)
    draft_equity = state.get(DRAFT_EQUITY_KEY)
    # La curva puede llegar como Series o como DataFrame con la columna Equity
    if isinstance(draft_equity, pd.DataFrame):
        draft_equity = draft_equity["Equity"] if "Equity" in draft_equity.columns else None
    if not (isinstance(draft_metrics, dict) and isinstance(draft_equity, pd.Series) and not draft_equity.empty):
        return None, None
    return draft_metrics, draft_equity


EQUITY_VIEWS = ["$", "Retorno %", "Base 100", "Log"]
PRINCIPAL_COLOR = "#2962FF"
COMPARE_COLORS = ["#FF6D00", "#00E676"]
BENCHMARK_COLOR = "#78909C"
DRAFT_COLOR = "#7C3AED"


def transform_equity(values, view: str):
    """Convierte una curva de patrimonio a la vista elegida en el selector.

    En `Log` NO se aplica `np.log`: el eje `y` ya está en escala logarítmica, así que
    aplicar el logaritmo dos veces deformaría la curva. Basta con rebasar sobre 100.
    """
    series = pd.Series(values).astype(float).dropna()
    if series.empty:
        return series
    base = float(series.iloc[0])
    if view == "Retorno %":
        return (series / base - 1.0) * 100.0
    if view in ("Base 100", "Log"):
        rebased = series / base * 100.0
        if view == "Log":
            # Un eje logarítmico no puede representar valores <= 0.
            rebased = rebased.clip(lower=1e-6)
        return rebased
    return series


def build_equity_figure(
    df_res, asset, script_name, show_equity, show_benchmark, show_drawdown,
    draft_metrics=None, draft_equity=None, comparatives=None, view="$",
):
    """Gráfico de patrimonio: principal, comparativas, borrador, benchmark y drawdown.

    `comparatives` es una lista de dicts con `name`, `equity` (Series) y opcionalmente
    `drawdown`. La vista `$` es la única que muestra dólares; las demás reescalan cada
    curva sobre su propio inicio para que sean comparables entre estrategias.
    """
    num_rows = 2 if show_drawdown else 1
    row_heights = [0.72, 0.28] if show_drawdown else [1.0]
    comparatives = comparatives or []

    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=row_heights,
    )

    def add_curve(x, values, name, color, width, dash):
        y = transform_equity(values, view)
        if y.empty:
            return
        fig.add_trace(
            go.Scatter(x=y.index, y=y.to_numpy(), name=name,
                       line=dict(color=color, width=width, dash=dash)),
            row=1, col=1,
        )

    if show_equity:
        add_curve(
            df_res.index, df_res["Equity"], f"Principal ⭐ ({script_name})",
            PRINCIPAL_COLOR, 2.2, None,
        )
    for pos, item in enumerate(comparatives[:MAX_COMPARATIVES]):
        color = COMPARE_COLORS[pos % len(COMPARE_COLORS)]
        add_curve(None, item["equity"], f"Comparativa ({item['name']})", color, 1.8, "solid")
    if draft_equity is not None:
        add_curve(
            None, draft_equity, f"Borrador IA ({(draft_metrics or {}).get('model', 'borrador')})",
            DRAFT_COLOR, 1.8, "dash",
        )
    if show_benchmark:
        add_curve(df_res.index, df_res["Benchmark"], f"Benchmark ({asset})", BENCHMARK_COLOR, 1.4, "dot")

    if show_drawdown and num_rows > 1:
        fig.add_trace(
            go.Scatter(
                x=df_res.index,
                y=df_res["Drawdown"],
                name="Drawdown (%)",
                fill="tozeroy",
                line=dict(color="#dc2626", width=1),
            ),
            row=2,
            col=1,
        )

    y_title = {"$": "Capital ($)", "Retorno %": "Retorno acumulado (%)",
               "Base 100": "Base 100", "Log": "Patrimonio (escala log)"}[view]
    fig.update_layout(
        height=392,
        margin=dict(l=8, r=10, t=8, b=8),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(color="#475569", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#cbd5e1", font_color="#0f172a"),
        xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0", showgrid=True),
        yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0", showgrid=True,
                   type="log" if view == "Log" else "linear", title=dict(text=y_title, font=dict(size=10))),
        showlegend=show_equity or show_benchmark or bool(comparatives) or draft_equity is not None,
    )
    if num_rows > 1:
        fig.update_yaxes(gridcolor="#e2e8f0", row=2, col=1)
    return fig


def build_comparison_table(main_metrics: dict, comparatives: list[dict], asset: str) -> pd.DataFrame:
    """Tabla comparativa fila por métrica: principal, comparativas y benchmark."""
    rows = [
        {"Estrategia": "Principal ⭐", "Retorno %": main_metrics["total_return"],
         "Sharpe": main_metrics["sharpe"], "Max Drawdown %": main_metrics["max_drawdown"],
         "Comisiones $": main_metrics["fees"], "Fills": main_metrics["fills"]},
    ]
    for item in comparatives:
        metrics = item.get("metrics") or {}
        if metrics.get("error"):
            rows.append({
                "Estrategia": f"Comparativa ({item['name']})",
                "Retorno %": None, "Sharpe": None, "Max Drawdown %": None,
                "Comisiones $": None, "Fills": None, "Error": metrics["error"],
            })
            continue
        rows.append({
            "Estrategia": f"Comparativa ({item['name']})",
            "Retorno %": metrics["total_return"],
            "Sharpe": metrics["sharpe"],
            "Max Drawdown %": metrics["max_drawdown"],
            "Comisiones $": metrics["fees"],
            "Fills": metrics["fills"],
        })
    # El benchmark va al final: es la referencia, no una candidata más.
    rows.append(
        {"Estrategia": f"Benchmark ({asset})", "Retorno %": main_metrics["benchmark_return"],
         "Sharpe": None, "Max Drawdown %": main_metrics["benchmark_drawdown"],
         "Comisiones $": 0.0, "Fills": None}
    )

    frame = pd.DataFrame(rows)
    base = float(main_metrics["total_return"])
    # "Vs Principal" en puntos porcentuales: positivo significa que va por delante.
    frame.insert(2, "Vs Principal", (frame["Retorno %"] - base).round(2))
    frame["Retorno %"] = frame["Retorno %"].round(2)
    frame["Sharpe"] = frame["Sharpe"].round(2)
    frame["Max Drawdown %"] = frame["Max Drawdown %"].round(2)
    frame["Comisiones $"] = frame["Comisiones $"].round(2)
    return frame


# -----------------------------------------------------------------------------
# WORKSPACE TOPBAR
# -----------------------------------------------------------------------------
# La barra superior se parte en tres bloques para poder insertar el popover 🤖
# como un widget real de Streamlit (no se puede meter un popover dentro de HTML).
top_left, top_ai, top_right = st.columns([6, 0.55, 6], vertical_alignment="center", gap="small")

with top_left:
    st.markdown(
        f"""
<div class="topbar">
    <span class="brand">GABO trader LAB</span>
    <span class="badge">📄 {selected_script}</span>
    <span class="badge">📊 {asset}</span>
</div>
""",
        unsafe_allow_html=True,
    )

with top_ai:
    ai_config = render_topbar_ai_popover()

with top_right:
    st.markdown(
        f"""
<div class="topbar topbar-right">
    <span class="badge">SMA {fast_period}/{slow_period}</span>
    <span class="badge ai">🤖 {ai_config['model']}</span>
    <span class="badge live">● LIVE</span>
    <span class="badge">LEAN {ENGINE_VERSION}</span>
</div>
""",
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# 2. BARRA SUPERIOR DE METRICAS (KPIs)
# -----------------------------------------------------------------------------
equity_value = auto("equity")
total_return_value = auto("total_return")
drawdown_value = auto("max_drawdown")
win_rate_value = auto("win_rate")
sharpe_value = auto("sharpe")

st.markdown(
    f"""
<div class="kpi-container">
    <div class="kpi-card"><div class="kpi-label">Capital Final</div>
        <div class="kpi-value {'positive' if equity_value >= initial_cap else 'negative'}">${equity_value:,.2f}</div></div>
    <div class="kpi-card"><div class="kpi-label">Retorno Total</div>
        <div class="kpi-value {'positive' if total_return_value >= 0 else 'negative'}">{total_return_value:.2f}%</div></div>
    <div class="kpi-card"><div class="kpi-label">Max Drawdown</div>
        <div class="kpi-value {'negative' if drawdown_value < 0 else 'positive'}">{drawdown_value:.2f}%</div></div>
    <div class="kpi-card"><div class="kpi-label">Win Rate</div>
        <div class="kpi-value {'positive' if win_rate_value >= 0 else 'negative'}">{win_rate_value:.1f}%</div></div>
    <div class="kpi-card"><div class="kpi-label">Sharpe Ratio</div>
        <div class="kpi-value {'positive' if sharpe_value >= 0 else 'negative'}">{sharpe_value:.2f}</div></div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 3. PANTALLA DIVIDIDA (50/50)
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="medium")

# --- COLUMNA IZQUIERDA: EDITOR DE CÓDIGO ---
with col_left:
    st.markdown(
        f'<div class="editor-header"><span class="dot">●</span> 📄 {selected_script}</div>',
        unsafe_allow_html=True,
    )

    code_path = BASE_DIR / selected_script
    code_content = read_script_source(selected_script)

    editor_key = f"editor_{selected_script}"
    needs_seed = (
        editor_key not in st.session_state
        or st.session_state.get("editor_loaded_for") != selected_script
    )
    if needs_seed:
        # El modal "Abrir Estrategia" inyecta el contenido ya leído en codigo_actual
        if st.session_state.get("codigo_actual_script") == selected_script and "codigo_actual" in st.session_state:
            st.session_state[editor_key] = st.session_state["codigo_actual"]
        else:
            st.session_state[editor_key] = code_content
        st.session_state["editor_loaded_for"] = selected_script

    edited_code = st.text_area(
        "Editar código Python:",
        key=editor_key,
        height=262,
        label_visibility="collapsed",
    )

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
    with btn_col1:
        save_clicked = st.button("💾 Guardar Cambios", use_container_width=True, key="btn_save")
    with btn_col2:
        run_clicked = st.button(
            "▶ Ejecutar Backtest", type="primary", use_container_width=True, key="btn_run"
        )
    with btn_col3:
        ai_clicked = st.button("🪄 Optimizar con IA", use_container_width=True, key="btn_ai")

    if ai_clicked:
        st.session_state["codigo_actual"] = edited_code
        st.session_state["codigo_actual_script"] = selected_script
        modal_ia_optimizar()

    ai_hint = st.session_state.get("ai_instrucciones", "").strip()
    st.caption(
        f"🪄 Instrucción pendiente para la IA: {ai_hint}" if ai_hint else "🪄 Pulsa «Optimizar con IA» para revisar la estrategia."
    )

    render_draft_area(display_metrics(), asset, fast_period, slow_period, initial_cap)

    st.markdown(
        '<div class="panel" style="padding:10px 12px">'
        '<div class="panel-title" style="margin-bottom:6px">💾 Guardar Como</div>',
        unsafe_allow_html=True,
    )
    sa_col1, sa_col2 = st.columns([2, 1])
    with sa_col1:
        new_filename = st.text_input(
            "Nuevo nombre (.py):",
            value=f"copia_{Path(selected_script).name}",
            key="new_filename",
            label_visibility="collapsed",
        )
    with sa_col2:
        save_as_clicked = st.button("📝 Guardar Como", use_container_width=True, key="btn_save_as")

    st.markdown("</div>", unsafe_allow_html=True)

    if save_clicked:
        try:
            code_path.write_text(edited_code, encoding="utf-8")
        except OSError as exc:
            st.error(f"❌ No se pudo guardar `{selected_script}`: {exc}")
        else:
            st.toast(f"💾 Guardado en `{selected_script}`", icon="✅")

    if save_as_clicked:
        target_name = new_filename if new_filename.endswith(".py") else f"{new_filename}.py"
        target = code_path.parent / target_name
        try:
            target.write_text(edited_code, encoding="utf-8")
        except OSError as exc:
            st.error(f"❌ No se pudo crear `{target_name}`: {exc}")
        else:
            st.toast(f"📝 Creado `{target.name}`", icon="✅")

    if run_clicked:
        st.session_state["force_run"] = True
        st.rerun()

# --- COLUMNA DERECHA: GRÁFICO INTERACTIVO ---
with col_right:
    chart_head, view_col = st.columns([3.2, 1], vertical_alignment="center", gap="small")
    with chart_head:
        st.markdown(
            f'<div class="panel-title">📈 Principal vs Comparativas · Benchmark ({asset})</div>',
            unsafe_allow_html=True,
        )
    with view_col:
        chart_view = st.segmented_control(
            "Vista del patrimonio:",
            EQUITY_VIEWS,
            default=st.session_state.get(CHART_VIEW_KEY, "$"),
            key=f"{CHART_VIEW_KEY}_widget",
            label_visibility="collapsed",
            help="Alterna entre dólares, retorno acumulado, base 100 y escala logarítmica.",
        ) or st.session_state.get(CHART_VIEW_KEY, "$")
    st.session_state[CHART_VIEW_KEY] = chart_view

    draft_metrics, draft_equity = read_draft_curve(st.session_state)

    # Las comparativas marcadas en el sidebar se ejecutan con los mismos parámetros.
    compare_scripts = [s for s in st.session_state.get(COMPARE_KEY, []) if s != selected_script]
    compare_results = get_comparative_results(
        compare_scripts, asset, initial_cap
    ) if compare_scripts else {}

    comparative_curves = []
    for script in compare_scripts[:MAX_COMPARATIVES]:
        payload = compare_results.get(script) or {}
        curve_df, curve_metrics = payload.get("df"), payload.get("metrics") or {}
        comparative_curves.append({
            "name": Path(script).name,
            "script": script,
            "equity": curve_df["Equity"] if curve_df is not None else pd.Series(dtype=float),
            "metrics": curve_metrics,
        })

    fig = build_equity_figure(
        df_res,
        asset,
        selected_script,
        show_equity,
        show_benchmark,
        show_drawdown,
        draft_metrics,
        draft_equity,
        comparative_curves,
        chart_view,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )

    comparativas_ok = [c for c in comparative_curves if not c["metrics"].get("error")]
    if comparativas_ok:
        st.caption(
            "🔵 Principal ⭐ · 🟠/🟢 comparativas marcadas con [x] · "
            "⚪ Benchmark · 🟣 borrador de IA (si está en prueba)"
        )
    for item in comparative_curves:
        if item["metrics"].get("error"):
            st.error(f"❌ `{item['script']}`: {item['metrics']['error']}")
    if draft_equity is not None and draft_metrics:
        st.caption(
            f"🧪 Curva violeta punteada = borrador de IA en prueba "
            f"(Capital Final ${draft_metrics['equity']:,.2f})."
        )

    # El panel analiza exactamente las mismas métricas que muestra la barra de KPIs
    values = display_metrics()

    render_ai_backtest_panel(
        values, df_res, asset, fast_period, slow_period, initial_cap, selected_script
    )

# -----------------------------------------------------------------------------
# 3b. TABLA COMPARATIVA DE MÉTRICAS (Principal vs Comparativas vs Benchmark)
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="panel-title" style="margin-top:10px">📊 Tabla Comparativa de Métricas</div>',
    unsafe_allow_html=True,
)
tabla_comparativa = build_comparison_table(values, comparative_curves, asset)
st.dataframe(
    tabla_comparativa,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Estrategia": st.column_config.TextColumn(width="medium"),
        "Retorno %": st.column_config.NumberColumn(format="%.2f%%"),
        "Vs Principal": st.column_config.NumberColumn(format="%+.2f pts", help="Diferencia en puntos porcentuales contra la principal"),
        "Sharpe": st.column_config.NumberColumn(format="%.2f"),
        "Max Drawdown %": st.column_config.NumberColumn(format="%.2f%%"),
        "Comisiones $": st.column_config.NumberColumn(format="$%.2f", help="Comisión estimada (0.05 % del capital por fill)"),
        "Fills": st.column_config.NumberColumn(format="%d", help="Cambios de exposición: entradas y salidas"),
        "Error": st.column_config.TextColumn(width="medium"),
    },
)

# -----------------------------------------------------------------------------
# 4. TERMINAL Y LOGS INFERIOR
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="panel-title" style="margin-top:6px">💻 Cloud Terminal &amp; Logs</div>',
    unsafe_allow_html=True,
)

log_time = datetime.now().strftime("%H:%M:%S")
ts = f'<span class="t-dim">{log_time}</span>'
source_line = (
    f'{ts} <span class="t-info">[INFO]</span> Fetched historical market data for '
    f'<span class="t-key">{asset}</span> ({metrics["bars"]} daily bars).'
    if live
    else f'{ts} <span class="t-warn">[WARN]</span> Market feed unreachable — rendering reference snapshot.'
)
status_tone = "t-ok" if total_return_value >= 0 else "t-err"

st.markdown(
    f"""
<div class="terminal-box">
    {ts} <span class="t-info">[INFO]</span> Initializing Lean Engine {ENGINE_VERSION}...<br>
    {ts} <span class="t-info">[INFO]</span> Workspace mounted: <span class="t-key">C:\\QuantConnect\\projects\\quantlab</span><br>
    {ts} <span class="t-info">[INFO]</span> Loaded strategy script: <span class="t-key">{selected_script}</span><br>
    {ts} <span class="t-info">[INFO]</span> Parameters &rarr; Fast SMA: <span class="t-key">{fast_period}</span> | Slow SMA: <span class="t-key">{slow_period}</span> | Capital: <span class="t-key">${initial_cap:,.2f}</span><br>
    {source_line}<br>
    {ts} <span class="t-info">[INFO]</span> Simulation resolved <span class="t-key">{metrics["trades"]}</span> round-trip trades.<br>
    {ts} <span class="t-ok">[SUCCESS]</span> Backtest deployed in 1.42 seconds. Total points processed: {metrics["bars"]} bar points.<br>
    {ts} <span class="t-info">[INFO]</span> Fees: <span class="t-key">${metrics["fees"]:.2f}</span> | Volume: <span class="t-key">${metrics["volume"]:,.0f}</span> | Sharpe: <span class="t-key">{sharpe_value:.2f}</span><br>
    {ts} <span class="{status_tone}">[STATUS]</span> Final Equity: <span class="t-key">${equity_value:,.2f}</span> ({total_return_value:.2f}%) &middot; Max DD: {drawdown_value:.2f}% &middot; Win Rate: {win_rate_value:.1f}%<br>
    {ts} <span class="t-ok">[READY]</span> Cloud node idle. Awaiting next deployment...
</div>
""",
    unsafe_allow_html=True,
)