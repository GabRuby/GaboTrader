import ast
import base64
import hashlib
import importlib.util
import os
import re
import sys
import uuid
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from plotly.subplots import make_subplots

# Variables de entorno del proyecto (.env): GEMINI_API_KEY, ENVIRONMENT, etc.
load_dotenv()
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Configuración de página ancha estilo IDE
st.set_page_config(
    page_title="GABO trader LAB",
    page_icon="📈",
    layout="wide",
    # Streamlit accepts an integer here as the initial sidebar width in pixels.
    initial_sidebar_state=200,
)

ENGINE_VERSION = "v18142"
WORKSPACE_TABS = (
    "INICIO",
    "LABORATORIO",
    "PATRONES Y DESCUBRIMIENTO",
    "PORTAFOLIO",
    "RESULTADOS Y ANÁLISIS",
    "HISTORIAL Y EXPERIMENTOS",
)
WORKSPACE_STATE_VERSION = 3
ASSISTANT_MODE_KEY = "global_assistant_mode"
ASSISTANT_MODE_GLOBAL = "global"
ASSISTANT_MODE_CREATE_STRATEGY = "crear_estrategia"
STRATEGY_CREATION_MESSAGES_KEY = "strategy_creation_messages"
STRATEGY_CREATION_PROCESSING_KEY = "strategy_creation_processing"
STRATEGY_CREATION_ERROR_KEY = "strategy_creation_error"
STRATEGY_CREATION_NOTICE_KEY = "strategy_creation_notice"
STRATEGY_CREATION_RULES_KEY = "strategy_creation_rules_state"
STRATEGY_CREATION_INPUT_KEY = "strategy_creation_input"
STRATEGY_CREATION_PROPOSAL_INPUT_KEY = "strategy_creation_proposal_input"
STRATEGY_CREATION_PHASE_KEY = "strategy_creation_phase"
STRATEGY_CREATION_CODE_KEY = "strategy_creation_code"
STRATEGY_CREATION_PHASE_DESIGN = "diseno_reglas"
STRATEGY_CREATION_PHASE_PROPOSAL_PENDING = "propuesta_pendiente_aprobacion"
STRATEGY_CREATION_PHASE_CODE_AUTHORIZED = "codigo_autorizado"
STRATEGY_CREATION_PHASE_CODE_PENDING_VALIDATION = "codigo_pendiente_validacion"
STRATEGY_CREATION_FILENAME_KEY = "strategy_creation_filename"
STRATEGY_CREATION_VALIDATION_KEY = "strategy_creation_validation"
STRATEGY_CREATION_CODE_EDITOR_KEY = "strategy_creation_code_editor"
STRATEGY_CREATION_SAVED_PATH_KEY = "strategy_creation_saved_path"
STRATEGY_CREATION_SAVED_CODE_KEY = "strategy_creation_saved_code"
STRATEGY_CREATION_CODE_STALE_KEY = "strategy_creation_code_stale"
# Migrate earlier workspaces to Inicio once without resetting their strategy state.
if st.session_state.get("workspace_state_version") != WORKSPACE_STATE_VERSION:
    st.session_state["workspace_tab"] = WORKSPACE_TABS[0]
    st.session_state["workspace_state_version"] = WORKSPACE_STATE_VERSION
st.session_state.setdefault("assistant_panel_ratio", 0.19)
st.session_state.setdefault("assistant_panel_width_px", None)
st.session_state.setdefault(ASSISTANT_MODE_KEY, ASSISTANT_MODE_GLOBAL)
st.session_state.setdefault(STRATEGY_CREATION_MESSAGES_KEY, [])
st.session_state.setdefault(STRATEGY_CREATION_PROCESSING_KEY, False)
st.session_state.setdefault(STRATEGY_CREATION_ERROR_KEY, None)
st.session_state.setdefault(STRATEGY_CREATION_NOTICE_KEY, None)
st.session_state.setdefault(STRATEGY_CREATION_PHASE_KEY, STRATEGY_CREATION_PHASE_DESIGN)
st.session_state.setdefault(STRATEGY_CREATION_CODE_KEY, None)
st.session_state.setdefault(STRATEGY_CREATION_FILENAME_KEY, "estrategia_nueva.py")
st.session_state.setdefault(STRATEGY_CREATION_VALIDATION_KEY, None)
st.session_state.setdefault(STRATEGY_CREATION_CODE_EDITOR_KEY, "")
st.session_state.setdefault(STRATEGY_CREATION_SAVED_PATH_KEY, None)
st.session_state.setdefault(STRATEGY_CREATION_SAVED_CODE_KEY, None)
st.session_state.setdefault(STRATEGY_CREATION_CODE_STALE_KEY, False)
st.session_state.setdefault(
    STRATEGY_CREATION_RULES_KEY,
    {"proposal": None, "approved": False},
)

_panel_resizer = components.declare_component(
    "gabo_panel_resizer",
    path=str(Path(__file__).resolve().parent / "ui_components" / "panel_resizer"),
)
_strategy_file_picker = components.declare_component(
    "gabo_strategy_file_picker",
    path=str(
        Path(__file__).resolve().parent
        / "ui_components"
        / "strategy_file_picker"
    ),
)


def normalize_assistant_panel_ratio(value) -> float:
    """Keep the right pane bounded so a useful central workspace always remains."""
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return 0.19
    return min(0.375, max(0.125, ratio))

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

    /* Ocultar menús nativos sin bloquear los divisores redimensionables */
    #MainMenu, footer, [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
        display: none !important;
    }

    .block-container {
        padding: 3.25rem 1.1rem 0.5rem 1.1rem;
        max-width: 100%;
    }
    #MainMenu, footer { visibility: hidden; }

    /* Tipografía */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5 { color: #0f172a; letter-spacing: .01em; }
    .stMarkdown p, .stMarkdown span, .stMarkdown label { color: #475569; }

    /* Barra superior de la workspace (gris suave) */
    .topbar {
        display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
        background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 8px;
        box-sizing: border-box; min-width: 0;
        padding: 9px 14px; margin-bottom: 10px;
    }
    .topbar .brand { font-size: 14px; font-weight: 700; color: #0f172a; letter-spacing: .06em; }
    .topbar .badge {
        font-size: 11px; font-weight: 600; color: #475569;
        background: #ffffff; border: 1px solid #cbd5e1; border-radius: 999px;
        padding: 3px 10px; font-family: 'Cascadia Mono', 'Courier New', monospace;
    }
    .topbar .spacer { margin-left: auto; }
    .topbar.topbar-right { justify-content: flex-end; margin-bottom: 0; }
    .lab-context {
        display: flex; align-items: center; flex-wrap: wrap; gap: 8px;
        padding: 8px 10px; margin-bottom: 10px;
        border: 1px solid #e2e8f0; border-radius: 8px; background: #f8fafc;
    }
    .lab-context .badge {
        display: inline-flex; font-size: 11px; font-weight: 600; color: #475569;
        background: #ffffff; border: 1px solid #cbd5e1; border-radius: 999px;
        padding: 3px 10px;
    }
    .lab-context .badge.status.waiting { color: #92400e; border-color: #fcd34d; background: #fffbeb; }
    .lab-context .badge.status.error { color: #b91c1c; border-color: #fca5a5; background: #fef2f2; }
    .lab-context .badge.status.ready { color: #166534; border-color: #86efac; background: #f0fdf4; }
    [data-testid="stRadio"] [role="radiogroup"] { gap: 4px; flex-wrap: wrap; }
    [data-testid="stRadio"] label {
        padding: 6px 10px; border: 1px solid #e2e8f0; border-radius: 8px;
        background: #f8fafc; font-size: 12px; font-weight: 600;
    }
    /* Barra inferior de navegación por pestañas: fondo gris suave solo en la
       navegación principal, sin tocar el resto de radios de la aplicación. */
    .st-key-workspace_tab [data-testid="stRadio"] [role="radiogroup"] {
        gap: 4px; flex-wrap: wrap;
        background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 6px 8px;
    }
    .st-key-workspace_tab [data-testid="stRadio"] label {
        background: #ffffff;
    }
    [data-testid="stRadio"] label:has(input:checked) {
        color: #1d4ed8; border-color: #93c5fd; background: #eff6ff;
    }
    .st-key-workspace-layout [data-testid="stHorizontalBlock"] {
        align-items: stretch;
    }
    .st-key-workspace-layout [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        min-height: 0;
        overflow-y: auto;
        overscroll-behavior: contain;
        scrollbar-gutter: stable;
    }
    .workspace-empty {
        border: 1px dashed #cbd5e1; border-radius: 12px; background: #f8fafc;
        padding: 28px; color: #64748b; text-align: center;
    }
    .workspace-card {
        min-height: 110px; padding: 16px; border: 1px solid #e2e8f0;
        border-radius: 10px; background: #ffffff;
    }
    .workspace-card strong { color: #0f172a; }
    .pattern-status {
        display: inline-flex; margin: 4px 5px 4px 0; padding: 5px 8px;
        border: 1px solid; border-radius: 999px; font-size: 11px; font-weight: 600;
    }
    .pattern-status.pending { color: #92400e; border-color: #fcd34d; background: #fffbeb; }
    .pattern-status.research { color: #1d4ed8; border-color: #93c5fd; background: #eff6ff; }
    .pattern-status.validated { color: #15803d; border-color: #86efac; background: #f0fdf4; }
    .pattern-status.expired { color: #64748b; border-color: #cbd5e1; background: #f8fafc; }

    /* Popover 🤖 del header: debe ocupar la celda completa para alinearse con la barra */
    [data-testid="stPopover"] > button { width: 100%; height: 38px; }

    /* Subtítulo de cada bloque del gestor (apertura / Principal) */
    .strategy-subtitle {
        font-size: 10.5px; font-weight: 700; letter-spacing: .05em;
        color: #64748b; margin: 12px 0 6px;
    }

    /* G.3.7 · Grupo 1 (ESTRATEGIAS): la Principal vive en una tarjeta compacta y,
       cuando no hay ninguna, la tarjeta se sustituye por su estado vacío. El borde
       ámbar es el mismo acento que usaba la fila anterior, así que el color no cambia:
       solo se deja de pintar la etiqueta «PRINCIPAL», que repetía el subtítulo. */
    .principal-card {
        display: flex; align-items: flex-start; gap: 8px;
        min-height: 42px; padding: 9px 10px;
        border: 1px solid #fcd34d; border-radius: 8px; background: #fffbeb;
        box-shadow: inset 3px 0 0 #f59e0b;
    }
    .principal-card .card-icon { flex: 0 0 auto; font-size: 14px; line-height: 1.35; }
    .principal-card .strategy-name {
        flex: 1; min-width: 0; font-size: 12.5px; font-weight: 600; line-height: 1.35;
        color: #0f172a; overflow-wrap: break-word; word-break: normal; white-space: normal;
    }
    .principal-card.empty {
        align-items: center; justify-content: center; min-height: 40px; padding: 9px 10px;
        border: 1px dashed #cbd5e1; background: #f8fafc; box-shadow: none;
        color: #94a3b8; font-size: 11.5px; font-weight: 600; text-align: center;
    }
    /* El 🗑️ va a la altura de la tarjeta para que la fila se lea como un bloque. */


    .st-key-grp_estrategias .stButton button { min-height: 36px; }

    .st-key-grp_estrategias .st-key-btn_open_strategy {
        width: fit-content; margin: 0 0 8px;
    }
    .st-key-grp_estrategias .st-key-btn_open_strategy button {
        min-height: 30px; height: 30px; width: auto; padding: 0 9px;
        border: 1px solid #16a34a; border-radius: 7px;
        background: #22c55e; color: #ffffff; font-size: 11px; line-height: 1;
    }
    .st-key-grp_estrategias .st-key-btn_open_strategy button:hover,
    .st-key-grp_estrategias .st-key-btn_open_strategy button:focus-visible {
        background: #16a34a; border-color: #15803d; color: #ffffff;
    }
    .st-key-strategy_creation_activate button {
        min-height: 34px; border-color: #7c3aed; background: #f5f3ff;
        color: #5b21b6; font-size: 12px;
    }
    .st-key-strategy_creation_activate button:hover,
    .st-key-strategy_creation_activate button:focus-visible {
        border-color: #6d28d9; background: #ede9fe; color: #4c1d95;
    }

    /* Fila de Estrategia Principal: tarjeta flexible y bot?n fijo. */
    .st-key-principal-strategy-row {
        width: 100%;
        min-width: 0;
        box-sizing: border-box;
    }
    .st-key-principal-strategy-row [data-testid="stHorizontalBlock"] {
        display: flex;
        align-items: center;
        flex-wrap: nowrap;
        gap: 8px;
        width: 100%;
        min-width: 0;
        box-sizing: border-box;
    }
    .st-key-principal-strategy-row [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child {
        flex: 1 1 0% !important;
        width: 0 !important;
        max-width: none !important;
        min-width: 0 !important;
        box-sizing: border-box;
    }
    .st-key-principal-strategy-row [data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child {
        flex: 0 0 42px !important;
        width: 42px !important;
        max-width: 42px !important;
        min-width: 42px !important;
        box-sizing: border-box;
    }
    .st-key-principal-strategy-row .principal-card {
        display: flex;
        align-items: center;
        gap: 8px;
        width: 100%;
        min-width: 0;
        min-height: 42px;
        padding: 8px 10px;
        box-sizing: border-box;
        border: 1px solid #fcd34d;
        border-radius: 8px;
        background: #fffbeb;
        box-shadow: inset 3px 0 0 #f59e0b;
    }
    .st-key-principal-strategy-row .principal-card .card-icon {
        flex: 0 0 auto;
        font-size: 14px;
        line-height: 1.35;
    }
    .st-key-principal-strategy-row .principal-card .strategy-name {
        flex: 1 1 auto;
        min-width: 0;
        font-size: 12.5px;
        font-weight: 600;
        line-height: 1.35;
        color: #0f172a;
        overflow-wrap: anywhere;
        word-break: normal;
        white-space: normal;
    }
    .st-key-principal-strategy-row .principal-card.empty {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        min-width: 0;
        min-height: 42px;
        padding: 8px;
        box-sizing: border-box;
        border: 1px dashed #cbd5e1;
        border-radius: 8px;
        background: #f8fafc;
        box-shadow: none;
        color: #64748b;
        font-size: 11px;
        font-weight: 500;
        line-height: 1.3;
        text-align: center;
        overflow-wrap: anywhere;
    }
    /* Eliminar la Principal solo cambia el estado, no borra el archivo. */
    .st-key-btn_quitar_principal button {
        min-width: 38px; min-height: 38px; height: 38px; padding: 0;
        border: 1px solid #dc2626; border-radius: 8px;
        background: #fee2e2; color: #991b1b; font-size: 16px; line-height: 1;
    }
    .st-key-btn_quitar_principal button:hover,
    .st-key-btn_quitar_principal button:focus-visible {
        background: #fecaca; border-color: #b91c1c; color: #7f1d1d;
    }
    .st-key-grp_estrategias [data-testid="stExpanderDetails"] {
        padding: 14px 14px 16px;
    }

    /* Encabezado compacto del expander principal. */
    .st-key-grp_prueba [data-testid="stExpander"] summary {
        padding-left: 8px !important;
        padding-right: 8px !important;
    }
    .st-key-grp_prueba [data-testid="stExpander"] summary p {
        font-size: 10px !important;
        letter-spacing: 0.025em !important;
        white-space: nowrap !important;
    }
    .st-key-grp_prueba [data-testid="stExpanderDetails"] {
        padding: 12px 12px 14px !important;
    }

    /* Tres columnas fijas/flexibles del selector (Streamlit 1.64 usa stColumn). */
    .st-key-principal-selector-row {
        width: 100%;
        min-width: 0;
    }
    .st-key-principal-selector-row [data-testid="stHorizontalBlock"] {
        display: flex !important;
        align-items: center !important;
        flex-wrap: nowrap !important;
        gap: 8px !important;
        width: 100%;
        min-width: 0;
        box-sizing: border-box;
    }
    .st-key-principal-selector-row [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: 0 !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        box-sizing: border-box;
    }
    .st-key-principal-selector-row [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child,
    .st-key-principal-selector-row [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {
        flex: 0 0 36px !important;
        width: 36px !important;
        min-width: 36px !important;
        max-width: 36px !important;
    }
    .st-key-principal-selector-row [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2) {
        flex: 1 1 0% !important;
        width: 0 !important;
        max-width: none !important;
        min-width: 0 !important;
    }
    .st-key-principal-selector-row [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div {
        width: 100%;
        min-width: 0;
        box-sizing: border-box;
    }

    .st-key-principal-selector-row .st-key-btn_prueba_abrir_estrategia,
    .st-key-principal-selector-row .st-key-btn_prueba_quitar_principal {
        width: 36px !important;
        min-width: 36px !important;
        max-width: 36px !important;
        flex: 0 0 36px !important;
        margin: 0 !important;
    }
    .st-key-principal-selector-row .st-key-btn_prueba_abrir_estrategia button,
    .st-key-principal-selector-row .st-key-btn_prueba_quitar_principal button {
        box-sizing: border-box;
        width: 36px !important;
        min-width: 36px !important;
        height: 42px !important;
        min-height: 42px !important;
        padding: 0 !important;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        line-height: 1;
        box-shadow: none;
    }
    .st-key-principal-selector-row .st-key-btn_prueba_abrir_estrategia button {
        border: 1px solid #cbd5e1;
        background: #f8fafc;
        color: #16a34a;
        font-size: 25px;
        font-weight: 400;
    }
    .st-key-principal-selector-row .st-key-btn_prueba_abrir_estrategia button:hover {
        border-color: #86efac;
        background: #f0fdf4;
        color: #15803d;
    }
    .st-key-principal-selector-row .st-key-btn_prueba_abrir_estrategia button:focus-visible {
        outline: 2px solid #86efac;
        outline-offset: 1px;
    }
    .st-key-principal-selector-row .st-key-btn_prueba_quitar_principal button {
        border: 1px solid #dc2626;
        background: #fee2e2;
        color: #991b1b;
        font-size: 15px;
    }
    .st-key-principal-selector-row .st-key-btn_prueba_quitar_principal button:hover {
        background: #fecaca;
        border-color: #b91c1c;
        color: #7f1d1d;
    }
    /* Streamlit monta este tooltip en body; no lo mantengas tras restaurar foco de ratón. */
    @supports selector(body:has(*)) {
        body:has(.st-key-btn_prueba_abrir_estrategia [data-testid="stTooltipHoverTarget"][aria-describedby]):not(:has(.st-key-btn_prueba_abrir_estrategia [data-testid="stTooltipHoverTarget"]:hover)):not(:has(.st-key-btn_prueba_abrir_estrategia [data-testid="stTooltipHoverTarget"] :focus-visible)):not(:has([data-testid="stTooltipContent"]:hover)) [data-testid="stTooltipContent"] {
            opacity: 0 !important;
            pointer-events: none !important;
            transition: none !important;
        }
    }
    .principal-strategy-value {
        display: flex;
        align-items: center;
        width: 100%;
        min-width: 0;
        max-width: 100%;
        height: 42px;
        min-height: 42px;
        padding: 0 8px;
        box-sizing: border-box;
        overflow: hidden;
        text-overflow: ellipsis;
        border: 1px solid #86efac;
        border-radius: 8px;
        background: #f0fdf4;
        color: #0f172a;
        font-size: 11px;
        font-weight: 700;
        line-height: 1.4;
        white-space: nowrap;
        text-align: left;
    }
    .principal-strategy-value > span {
        display: block;
        flex: 1 1 auto;
        width: 100%;
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .principal-strategy-value.empty {
        border-style: dashed;
        background: #f8fafc;
        color: #94a3b8;
        font-weight: 500;
    }

    .st-key-principal-prototype {
        margin-top: 10px; padding: 8px; border: 1px solid #94a3b8;
        border-radius: 8px; background: #f8fafc; container-type: inline-size;
    }
    .principal-prototype-heading {
        margin-bottom: 8px; color: #334155; font-size: 10px;
        font-weight: 800; letter-spacing: .06em;
    }
    .principal-prototype-row {
        display: grid; grid-template-columns: minmax(0, 1fr) 44px;
        align-items: start; gap: 8px; min-width: 0;
    }
    .principal-prototype-value {
        box-sizing: border-box; width: 100%; min-height: 42px; padding: 6px 8px;
        border: 1px solid #cbd5e1; border-radius: 8px; background: #ffffff;
        color: #0f172a; font-size: 12.5px; font-weight: 600; line-height: 1.4;
        overflow-wrap: break-word;
    }
    .principal-prototype-value.empty {
        padding: 6px 4px; border-style: dashed; color: #64748b; font-size: 10px;
        font-weight: 500; line-height: 1.25; text-align: center;
    }
    .principal-prototype-delete {
        box-sizing: border-box; width: 44px; min-width: 44px; height: 42px; min-height: 42px;
        border: 1px solid #b91c1c; border-radius: 8px;
        background: #dc2626; color: #ffffff; font-size: 17px; line-height: 1;
        cursor: default;
    }
    .principal-prototype-delete:hover,
    .principal-prototype-delete:focus-visible {
        border-color: #991b1b; background: #b91c1c; color: #ffffff;
    }
    /* PRUEBA TEMPORAL: desactivar el cambio de columnas por ancho.
    @container (max-width: 180px) {
        .principal-prototype-row { grid-template-columns: minmax(0, 1fr); }
        .principal-prototype-delete { justify-self: end; }
    }
    */

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
    .editor-sync-status {
        margin-left: auto; font-family: Inter, sans-serif; font-size: 11px;
        font-weight: 700; white-space: nowrap;
    }
    .editor-sync-status.saved { color: #15803d; }
    .editor-sync-status.pending { color: #b45309; }
    .editor-sync-status.unavailable { color: #b91c1c; }
    .editor-sync-status.no-strategy { color: #64748b; }
    .editor-provenance {
        flex: 0 0 auto; padding: 2px 6px; border-radius: 999px;
        color: #475569; background: #e2e8f0; font: 600 10px Inter, sans-serif;
    }
    .editor-header.comparison .editor-provenance {
        color: #854d0e; background: #fef3c7;
    }
    .st-key-principal-editor-panel, .st-key-comparison-editor-panel {
        box-sizing: border-box; min-height: 700px; height: 100%;
        padding: 12px; border: 1px solid #dbe3ed; border-radius: 10px;
        background: #ffffff;
    }
    .st-key-comparison-editor-panel {
        border-color: #f3d49b; background: #fffcf5;
        box-shadow: inset 3px 0 0 #e9b65e;
    }
    .st-key-principal-editor-panel [data-testid="stTextArea"],
    .st-key-comparison-editor-panel [data-testid="stTextArea"] {
        flex: 0 0 auto;
    }
    .st-key-btn_undo_main button,
    [class*="st-key-compare_undo_"] button {
        min-height: 32px; padding: 4px 8px; font-size: 11px;
        color: #991b1b; border-color: #fecaca; background: #fef2f2;
    }
    .st-key-btn_undo_main button:hover,
    [class*="st-key-compare_undo_"] button:hover {
        color: #7f1d1d; border-color: #fca5a5; background: #fee2e2;
    }

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

    /* Sidebar nativo de Streamlit: su divisor lateral conserva el arrastre nativo. */
    section[data-testid="stSidebar"] {
        width: 12.5vw;
        min-width: clamp(160px, 12.5vw, 280px);
        max-width: min(37.5vw, calc(100vw - 680px));
        background: #f1f5f9;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebarResizeHandle"],
    [data-testid="stSidebarResizer"],
    [data-testid="stSidebar"] [role="separator"] {
        width: 12px !important;
        cursor: col-resize !important;
    }
    [data-testid="stSidebarResizeHandle"]:hover,
    [data-testid="stSidebar"] [data-testid="stSidebarResizer"]:hover,
    [data-testid="stSidebar"] [role="separator"]:hover {
        background: rgba(37, 99, 235, 0.12);
    }
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

    /* Fase 1.1 · Pantalla inicial del Laboratorio (sin estrategia cargada) */
    .lab-welcome {
        border: 1px solid #dbeafe; border-radius: 12px;
        background: linear-gradient(135deg, #eff6ff 0%, #f8fafc 100%);
        padding: 16px 20px; margin-bottom: 14px;
    }
    .lab-welcome-title {
        font-size: 19px; font-weight: 700; color: #1e3a8a; margin-bottom: 4px;
    }
    .lab-welcome-desc {
        font-size: 13px; color: #475569; line-height: 1.5; max-width: 900px;
    }
    .lab-steps {
        display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px;
    }
    .lab-step {
        display: inline-flex; align-items: center; gap: 7px;
        background: #ffffff; border: 1px solid #bfdbfe; border-radius: 999px;
        padding: 5px 12px 5px 5px; font-size: 11.5px; font-weight: 600; color: #1d4ed8;
    }
    .lab-step b {
        display: inline-flex; align-items: center; justify-content: center;
        width: 18px; height: 18px; border-radius: 50%;
        background: #2563eb; color: #ffffff; font-size: 11px;
    }
    .lab-info-card {
        border: 1px solid #e2e8f0; border-left: 3px solid #2563eb;
        border-radius: 8px; background: #f8fafc;
        padding: 10px 13px; margin: 8px 0 10px;
    }
    .lab-info-card .lab-info-title {
        font-size: 12.5px; font-weight: 700; color: #1e3a8a; margin-bottom: 3px;
    }
    .lab-info-card .lab-info-body {
        font-size: 12px; color: #475569; line-height: 1.5;
    }
    .lab-empty-chart {
        border: 1px dashed #cbd5e1; border-radius: 10px; background: #f8fafc;
        padding: 34px 20px; text-align: center; color: #64748b; margin-bottom: 8px;
    }
    .lab-empty-chart .lab-empty-title {
        font-size: 15px; font-weight: 700; color: #94a3b8; margin-bottom: 6px;
    }
    .lab-empty-chart .lab-empty-body { font-size: 12px; color: #94a3b8; }

    @media (max-width: 800px) {
        .block-container { padding: 3.25rem 0.6rem 0.5rem; }
        [data-testid="stRadio"] label { padding: 4px 6px; font-size: 10px; }
        section[data-testid="stSidebar"] {
            width: min(72vw, 280px);
            min-width: 180px;
            max-width: min(72vw, 37.5vw);
        }
    }
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


def is_valid_strategy_filename(filename: str) -> bool:
    """Accept a single Python filename, never a path or the application module."""
    return bool(
        isinstance(filename, str)
        and filename.endswith(".py")
        and filename.casefold() != "app.py"
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_. -]*\.py", filename)
    )


def validate_strategy_source(source: str, filename: str = "estrategia.py") -> dict:
    """Validate syntax and the project's declared strategy entry point without execution."""
    errors = []
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return {
            "valid": False,
            "dialect": strategy_parser.DIALECTO_NINGUNO,
            "errors": [f"Error de sintaxis en línea {exc.lineno}: {exc.msg}"],
        }

    info = strategy_parser.analizar_fuente(source, nombre=filename)
    if not info.valido:
        errors.append(info.error or "El archivo no expone una interfaz de estrategia reconocida.")
    elif info.dialecto == strategy_parser.DIALECTO_VECTOR:
        entry = next(
            (
                node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "run_strategy"
            ),
            None,
        )
        positional = (
            list(entry.args.posonlyargs) + list(entry.args.args)
            if entry is not None else []
        )
        if not positional:
            errors.append("`run_strategy` debe aceptar el DataFrame de entrada como primer argumento.")
        if entry is not None and not any(
            isinstance(node, ast.Return) for node in ast.walk(entry)
        ):
            errors.append("`run_strategy` debe devolver el DataFrame procesado.")
    elif info.dialecto == strategy_parser.DIALECTO_LEAN:
        algorithm_classes = [
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and any(
                (isinstance(base, ast.Name) and base.id == "QCAlgorithm")
                or (isinstance(base, ast.Attribute) and base.attr == "QCAlgorithm")
                for base in node.bases
            )
        ]
        if not any(
            any(
                isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                and member.name == "initialize"
                for member in algorithm.body
            )
            for algorithm in algorithm_classes
        ):
            errors.append("La clase `QCAlgorithm` debe implementar `initialize`.")

    for node in ast.walk(tree):
        roots = []
        if isinstance(node, ast.Import):
            roots.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.append(node.module.split(".", 1)[0])
        for root in roots:
            if root == "AlgorithmImports" and info.dialecto == strategy_parser.DIALECTO_LEAN:
                continue
            if root in getattr(sys, "stdlib_module_names", set()) or root in sys.modules:
                continue
            try:
                available = importlib.util.find_spec(root) is not None
            except (ImportError, ValueError, ModuleNotFoundError):
                available = False
            if not available:
                errors.append(
                    f"Dependencia no disponible para importar estáticamente: `{root}`."
                )

    return {
        "valid": not errors,
        "dialect": info.dialecto,
        "errors": list(dict.fromkeys(errors)),
    }


def save_strategy_source(base: Path, filename: str, source: str) -> Path:
    """Create a validated strategy file exclusively; never overwrite an existing file."""
    if not is_valid_strategy_filename(filename):
        raise ValueError("El nombre debe ser un único archivo Python válido terminado en `.py`.")
    target = base / filename
    if target.exists():
        raise FileExistsError(f"El archivo ya existe: {target}")
    validation = validate_strategy_source(source, filename)
    if not validation["valid"]:
        raise ValueError("El código no cumple el contrato estático: " + " ".join(validation["errors"]))

    created = False
    try:
        with target.open("x", encoding="utf-8", newline="\n") as strategy_file:
            created = True
            strategy_file.write(source)
    except OSError as write_error:
        if created:
            try:
                target.unlink()
            except OSError as cleanup_error:
                raise OSError(
                    f"No se pudo escribir el archivo: {write_error}. "
                    f"El archivo parcial `{target}` tampoco pudo eliminarse: {cleanup_error}"
                ) from write_error
        raise
    return target


def register_strategy_file(path: Path, base: Path, available: list[str]) -> str:
    """Refresh in-memory strategy lists after a successful file creation."""
    relative = path.relative_to(base).as_posix()
    if relative not in available:
        available.append(relative)
        available.sort(key=str.lower)
    if path.parent == base and path.name not in py_files:
        py_files.append(path.name)
        py_files.sort(key=str.lower)
    return relative


# Texto que ve el editor cuando no hay ninguna Estrategia Principal: el mismo que ya
# usaba como reserva `read_script_source`, ahora compartido con el estado vacío.
SIN_ESTRATEGIA = "# Nueva estrategia\n"
SIN_ESTRATEGIA_NOMBRE = "(sin Estrategia Principal)"

# Plantilla inicial del Laboratorio: lo primero que ve el usuario al entrar sin ninguna
# Estrategia Principal cargada. Vive SOLO en `st.session_state` como borrador temporal:
# no es un archivo, no toca el catálogo y no se escribe nada en disco hasta que el
# usuario pulse «Guardar Como». Respeta el mismo contrato que las estrategias del
# proyecto (`run_strategy(data, fast_ma, slow_ma)` que escribe `data["Signal"]`), así que
# se ejecuta en el motor de `qc_lab` exactamente igual que una estrategia guardada.
PLANTILLA_INICIAL = '''# Plantilla inicial del Laboratorio
# Estrategia de ejemplo lista para editar, ejecutar y guardar como archivo nuevo.

import pandas as pd
import numpy as np


def run_strategy(data, fast_ma=10, slow_ma=50):
    data["SMA_Fast"] = data["Close"].rolling(window=fast_ma).mean()
    data["SMA_Slow"] = data["Close"].rolling(window=slow_ma).mean()
    data["Signal"] = np.where(data["SMA_Fast"] > data["SMA_Slow"], 1, 0)
    return data
'''

PLANTILLA_CODE_KEY = "plantilla_inicial_code"
PLANTILLA_EDITOR_KEY = "plantilla_inicial_editor"
PLANTILLA_RESULT_KEY = "plantilla_inicial_backtest"
PLANTILLA_PARAMS_KEY = "plantilla_inicial_params"
PLANTILLA_PARAM_PREFIX = "plantilla_param_"

# Registro EXPLÍCITO y acotado de las variables ajustables de la plantilla inicial.
# No se inventan parámetros ni se edita el Python de la estrategia: `strategy_parser`
# descubre qué nombres declara el código (hoy `fast_ma` y `slow_ma` en la firma de
# `run_strategy`) y aquí solo se fija, para esos nombres conocidos, el rango que pide
# la referencia visual. Cualquier otro parámetro detectado recibe un rango sugerido por
# el propio parser, así que el mecanismo queda preparado para ampliarse sin tocar el
# contrato del motor (que sigue recibiendo los overrides por `ConfigEjecucion`).
PLANTILLA_PARAMETROS_SPEC = {
    "fast_ma": {"label": "Fast SMA", "min": 1, "max": 50, "step": 1, "default": 10},
    "slow_ma": {"label": "Slow SMA", "min": 20, "max": 200, "step": 1, "default": 50},
}


def plantilla_parametro_spec(parametro) -> dict:
    """Rango/etiqueta de un parámetro del borrador: el explícito o el que sugiere el parser.

    Se apoya en `Parametro` del parser (nombre, valor, `es_entero`) para no inventar
    nada: si el nombre está en el registro de la plantilla se usa tal cual; si no, el
    rango sale de `strategy_parser.sugerir_rango` para no romper el contrato del motor.
    """
    spec = PLANTILLA_PARAMETROS_SPEC.get(parametro.nombre)
    if spec is not None:
        return dict(spec)
    minimo, maximo, paso = (
        float(v) for v in strategy_parser.sugerir_rango(parametro.nombre, parametro.valor)
    )
    default = int(round(parametro.valor)) if parametro.es_entero else float(parametro.valor)
    if parametro.es_entero:
        paso = float(max(1, round(paso)))
        minimo = float(round(minimo))
        maximo = float(max(round(maximo), default + paso))
    return {
        "label": parametro.nombre,
        "min": minimo,
        "max": maximo,
        "step": paso,
        "default": default,
    }


def plantilla_parametros_activos() -> dict:
    """Valores vigentes de los parámetros del borrador, leídos del estado de la sesión.

    Se resuelve ANTES de calcular métricas (igual que `strategy_parameter_state` para la
    Principal) para que `_sello_parametros` invalide el resultado cacheado en cuanto se
    mueve un control. En el primer run, sin widgets todavía, cae en los valores por
    defecto de la plantilla (fast_ma=10, slow_ma=50).
    """
    source = str(
        st.session_state.get(PLANTILLA_EDITOR_KEY)
        or st.session_state.get(PLANTILLA_CODE_KEY)
        or PLANTILLA_INICIAL
    )
    info = strategy_parser.analizar_fuente(source, nombre="plantilla_inicial.py")
    valores: dict = {}
    for parametro in info.parametros:
        spec = plantilla_parametro_spec(parametro)
        clave = f"{PLANTILLA_PARAM_PREFIX}{parametro.nombre}"
        valor = st.session_state.get(clave, spec["default"])
        valores[parametro.nombre] = (
            int(round(float(valor))) if parametro.es_entero else float(valor)
        )
    if not valores:
        valores = {
            nombre: spec["default"]
            for nombre, spec in PLANTILLA_PARAMETROS_SPEC.items()
        }
    return valores


def read_script_source(rel_path: str) -> str:
    """Lee el código de una estrategia (ruta relativa a BASE_DIR)."""
    if is_external_strategy(rel_path):
        record = external_strategy_record(rel_path)
        return str(record.get("source", "")) if record else SIN_ESTRATEGIA
    try:
        return (BASE_DIR / rel_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, ValueError):
        return DEFAULT_STRATEGIES.get(rel_path, SIN_ESTRATEGIA)


EDITOR_SAVE_FAILURE_KEY = "editor_save_failure"
EDITOR_SAVE_IN_PROGRESS_KEY = "editor_save_in_progress"
EDITOR_NAVIGATION_KEY = "pending_editor_navigation"
COMPARE_EDITOR_VERSIONS_KEY = "compare_editor_versions"
COMPARE_EDITOR_SOURCES_KEY = "compare_editor_sources"
COMPARE_PARAMETERS_KEY = "compare_strategy_parameters"
EDITOR_UNDO_KEY = "editor_undo_requested"
BACKTEST_PENDING_KEY = "backtest_pending_confirmation"
COMPARE_PENDING_KEY = "comparison_pending_confirmation"
MODAL_VIEW_KEY = "strategy_picker_view"
MODAL_SELECTED_PATH_KEY = "strategy_picker_selected_path"
MODAL_SELECTED_SOURCE_KEY = "strategy_picker_selected_source"
MODAL_SELECTED_HASH_KEY = "strategy_picker_selected_sha256"
MODAL_SELECTED_NAME_KEY = "strategy_picker_selected_name"
MODAL_ERROR_KEY = "strategy_picker_error"
MODAL_SEARCH_KEY = "strategy_picker_search"
MODAL_FILE_KEY = "strategy_picker_file"
MODAL_UPLOAD_KEY = "strategy_picker_upload"
MODAL_PROVENANCE_KEY = "strategy_picker_provenance"
MODAL_DROP_EVENT_KEY = "strategy_picker_last_drop_event"
MODAL_ACTION_EVENT_KEY = "strategy_picker_last_action_event"
EXTERNAL_STRATEGIES_KEY = "external_strategy_sources"


def is_external_strategy(script: str) -> bool:
    return isinstance(script, str) and script.startswith("external:")


def external_strategy_record(script: str) -> dict | None:
    records = st.session_state.get(EXTERNAL_STRATEGIES_KEY) or {}
    record = records.get(script) if is_external_strategy(script) else None
    return record if isinstance(record, dict) else None


def strategy_display_name(script: str) -> str:
    record = external_strategy_record(script)
    return str(record.get("name") or "estrategia.py") if record else script


def strategy_filename(script: str) -> str:
    return strategy_display_name(script) if is_external_strategy(script) else Path(script).name


def editor_provenance_label(script: str) -> str:
    return "Archivo externo" if is_external_strategy(script) else "Catálogo del proyecto"


def create_external_strategy(filename: str, source: str) -> str:
    """Keep a selected external file in session without creating or copying files."""
    script = f"external:{uuid.uuid4().hex}"
    records = dict(st.session_state.get(EXTERNAL_STRATEGIES_KEY) or {})
    records[script] = {
        "name": Path(filename).name,
        "source": source,
        "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }
    st.session_state[EXTERNAL_STRATEGIES_KEY] = records
    return script


def resolve_editor_strategy_file(script: str) -> Path:
    """Resolve the selected strategy file and reject paths outside the project."""
    if not script:
        raise ValueError("No hay una estrategia seleccionada.")
    if is_external_strategy(script):
        raise ValueError(
            "La estrategia es externa y se conserva solo en el editor; usa «Guardar Como» "
            "para crear una copia en el proyecto."
        )
    base = BASE_DIR.resolve()
    path = (base / script).resolve(strict=True)
    path.relative_to(base)
    if not path.is_file():
        raise FileNotFoundError(f"No es un archivo de estrategia: {script}")
    return path


def editor_sync_status(script: str, editor_code: str) -> tuple[str, str]:
    """Compare editor text with the exact selected file; never treat read errors as clean."""
    if not script:
        return (
            "no_strategy",
            "Sin estrategia abierta; este contenido no está asociado a un archivo.",
        )
    external = external_strategy_record(script)
    if is_external_strategy(script):
        if external is None:
            return "unavailable", "No se encontró la copia externa de trabajo en esta sesión."
        if str(external.get("source", "")) == editor_code:
            return "saved", "Copia externa lista · el original no se modifica."
        return "pending", "Cambios pendientes en la copia externa · el original está intacto."
    try:
        path = resolve_editor_strategy_file(script)
        saved_code = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "unavailable", "No se pudo verificar: el archivo seleccionado no existe."
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return "unavailable", f"No se pudo verificar el archivo seleccionado: {exc}"

    failed_saves = st.session_state.get(EDITOR_SAVE_FAILURE_KEY) or {}
    failed_source = failed_saves.get(str(path))
    if failed_source is None and failed_saves.get("identity") == str(path):
        failed_source = failed_saves.get("source")
    if failed_source == editor_code:
        return "pending", "Cambios pendientes · el último guardado falló."
    if saved_code == editor_code:
        return "saved", "Guardada"
    return "pending", "Cambios pendientes"


def editor_save_button_disabled(script: str, editor_code: str) -> bool:
    """Enable saving only for verified, unsaved changes and no active write."""
    if (
        not script
        or st.session_state.get(EDITOR_SAVE_IN_PROGRESS_KEY)
    ):
        return True
    state, _message = editor_sync_status(script, editor_code)
    return state != "pending"


def compare_editor_widget_key(script: str) -> str:
    versions = st.session_state.get(COMPARE_EDITOR_VERSIONS_KEY) or {}
    return f"compare_editor_{script}_{int(versions.get(script, 0))}"


def comparison_editor_sync_status(script: str) -> tuple[str, str]:
    return editor_sync_status(
        script, editor_source_for(script, editor_kind="comparison")
    )


def comparison_editor_has_unresolved_changes(script: str) -> bool:
    editor_key = compare_editor_widget_key(script)
    buffers = st.session_state.get(COMPARE_EDITOR_SOURCES_KEY) or {}
    if editor_key not in st.session_state and script not in buffers:
        return False
    return comparison_editor_sync_status(script)[0] != "saved"


def clear_editor_save_failure(script: str) -> None:
    path = resolve_editor_strategy_file(script)
    failed_saves = st.session_state.get(EDITOR_SAVE_FAILURE_KEY) or {}
    if isinstance(failed_saves, dict):
        failed_saves.pop(str(path), None)
        if failed_saves:
            st.session_state[EDITOR_SAVE_FAILURE_KEY] = failed_saves
        else:
            st.session_state.pop(EDITOR_SAVE_FAILURE_KEY, None)


def discard_comparison_editor(script: str) -> None:
    """Drop only this comparative's editor buffer; never touch its saved file."""
    versions = dict(st.session_state.get(COMPARE_EDITOR_VERSIONS_KEY) or {})
    versions[script] = int(versions.get(script, 0)) + 1
    st.session_state[COMPARE_EDITOR_VERSIONS_KEY] = versions
    st.session_state.get(COMPARE_EDITOR_SOURCES_KEY, {}).pop(script, None)


def restore_editor_source(script: str, editor_kind: str = "principal") -> str:
    """Restore one editor to its last saved project file or external session copy."""
    if is_external_strategy(script):
        record = external_strategy_record(script)
        if record is None:
            raise FileNotFoundError("La copia externa ya no está disponible en esta sesión.")
        source = str(record.get("source", ""))
    else:
        source = resolve_editor_strategy_file(script).read_text(encoding="utf-8")

    if editor_kind == "comparison":
        key = compare_editor_widget_key(script)
        st.session_state[key] = source
        buffers = dict(st.session_state.get(COMPARE_EDITOR_SOURCES_KEY) or {})
        buffers[script] = source
        st.session_state[COMPARE_EDITOR_SOURCES_KEY] = buffers
    else:
        key = f"editor_{script}"
        st.session_state[key] = source
        st.session_state["codigo_actual"] = source
        st.session_state["codigo_actual_script"] = script
    if not is_external_strategy(script):
        clear_editor_save_failure(script)
    return source


def save_editor_source(script: str, source: str) -> None:
    """Write only verified pending changes and clear failure state after success."""
    if is_external_strategy(script):
        record = external_strategy_record(script)
        if record is None:
            raise FileNotFoundError("La copia externa ya no está disponible en esta sesión.")
        if str(record.get("source", "")) == source:
            return
        records = dict(st.session_state.get(EXTERNAL_STRATEGIES_KEY) or {})
        saved_copy = dict(record)
        saved_copy["source"] = source
        saved_copy["sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
        records[script] = saved_copy
        st.session_state[EXTERNAL_STRATEGIES_KEY] = records
        if script in (st.session_state.get(COMPARE_EDITOR_SOURCES_KEY) or {}):
            st.session_state[COMPARE_EDITOR_SOURCES_KEY][script] = source
        if st.session_state.get("codigo_actual_script") == script:
            st.session_state["codigo_actual"] = source
        return
    path = resolve_editor_strategy_file(script)
    try:
        saved_source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise OSError(
            f"No se pudo leer `{script}`; no se sobrescribió el archivo: {exc}"
        ) from exc
    if saved_source == source:
        clear_editor_save_failure(script)
        comparative_sources = st.session_state.get(COMPARE_EDITOR_SOURCES_KEY) or {}
        if script in comparative_sources:
            comparative_sources[script] = source
        return
    if editor_save_button_disabled(script, source):
        raise ValueError("No hay cambios pendientes verificables para guardar.")

    st.session_state[EDITOR_SAVE_IN_PROGRESS_KEY] = True
    try:
        try:
            path.write_text(source, encoding="utf-8")
        except OSError:
            failed_saves = st.session_state.get(EDITOR_SAVE_FAILURE_KEY) or {}
            if not isinstance(failed_saves, dict) or "identity" in failed_saves:
                failed_saves = {}
            failed_saves[str(path)] = source
            st.session_state[EDITOR_SAVE_FAILURE_KEY] = {
                **failed_saves,
            }
            raise
        clear_editor_save_failure(script)
        comparative_sources = st.session_state.get(COMPARE_EDITOR_SOURCES_KEY) or {}
        if script in comparative_sources:
            comparative_sources[script] = source
    finally:
        st.session_state.pop(EDITOR_SAVE_IN_PROGRESS_KEY, None)


def editor_source_for(script: str, editor_kind: str = "principal") -> str:
    """Get session text only from the widget owned by this strategy and editor."""
    editor_key = compare_editor_widget_key(script) if editor_kind == "comparison" else f"editor_{script}"
    if editor_key in st.session_state:
        return str(st.session_state[editor_key])
    if editor_kind == "comparison":
        saved_buffers = st.session_state.get(COMPARE_EDITOR_SOURCES_KEY) or {}
        if script in saved_buffers:
            return str(saved_buffers[script])
        return read_script_source(script)
    if st.session_state.get("codigo_actual_script") == script:
        return str(st.session_state.get("codigo_actual", ""))
    return read_script_source(script)


def read_strategy_candidate(relative_path: str) -> tuple[str, str]:
    """Read a project strategy consistently and return its source and SHA-256."""
    path = resolve_editor_strategy_file(relative_path)
    if path.suffix.casefold() != ".py" or path.name.casefold() == "app.py":
        raise ValueError("Solo se pueden seleccionar archivos de estrategia `.py`.")
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    signature_before = (before.st_size, before.st_mtime_ns, before.st_ino)
    signature_after = (after.st_size, after.st_mtime_ns, after.st_ino)
    if signature_before != signature_after or len(content) != after.st_size:
        raise OSError(f"`{relative_path}` cambió mientras se leía; vuelve a seleccionarlo.")
    source = normalize_source_newlines(content.decode("utf-8"))
    return source, hashlib.sha256(content).hexdigest()


def normalize_source_newlines(source: str) -> str:
    return source.replace("\r\n", "\n").replace("\r", "\n")


# -----------------------------------------------------------------------------
# Revisión estática del selector: evaluación reproducible sin ejecutar código
# -----------------------------------------------------------------------------
# Reutiliza la metadata que ya produce `strategy_parser` y añade una evaluación
# estructural derivada del AST (claridad, riesgo, completitud y robustez). En
# ningún caso importa ni ejecuta el archivo analizado: solo `ast` y el recuento
# de comentarios. Mide calidad estructural del código, nunca rentabilidad.
REVIEW_GENERAL_LABEL = "Calificación general"
REVIEW_DIMENSION_LABELS = (
    "Claridad del código",
    "Gestión de riesgo",
    "Completitud",
    "Robustez estática estimada",
)
# Pesos de la calificación general (solo dimensiones estructurales).
REVIEW_WEIGHTS = {
    "Completitud": 0.35,
    "Claridad del código": 0.25,
    "Robustez estática estimada": 0.20,
    "Gestión de riesgo": 0.20,
}
_REVIEW_STATUS_DISPLAY = {
    "evaluated": "Evaluado",
    "partial": "Parcial",
    "not_evaluable": "No evaluable",
    "unverified": "No verificado",
}
_REVIEW_STOP_TOKENS = {"stop", "stoploss", "trailing", "sl"}
_REVIEW_TAKE_TOKENS = {"take", "tp", "objetivo"}
_REVIEW_POSITION_TOKENS = {
    "position", "posicion", "size", "peso", "pesos", "weight", "exposure",
    "exposicion", "qty", "quantity", "shares", "capital", "holding", "holdings",
}
# token -> clave interna del indicador, por orden de prioridad.
_REVIEW_INDICATOR_TOKENS = (
    ("sma", "sma"), ("ema", "ema"), ("wma", "wma"),
    ("true_range", "atr"), ("atr", "atr"), ("rsi", "rsi"), ("macd", "macd"),
    ("stddev", "std"), ("std", "std"), ("bollinger", "bb"), ("bb", "bb"),
    ("momentum", "mom"), ("roc", "mom"), ("vol", "vol"),
)
_REVIEW_INDICATOR_LABELS = {
    "sma": "SMA", "ema": "EMA", "wma": "WMA", "ma": "Media móvil",
    "atr": "ATR", "rsi": "RSI", "macd": "MACD", "std": "Desviación estándar",
    "bb": "Bandas de Bollinger", "mom": "Momentum", "vol": "Volatilidad",
}
_REVIEW_NON_DESCRIPTIVE = {
    "self", "data", "df", "pd", "np", "true", "false", "none", "i", "j", "k",
    "x", "y", "s", "t", "v", "n", "e", "f",
}


def _review_evaluation(value, status, reason, findings=None, display=None):
    """Un resultado con valor, estado, motivo y hallazgos que lo respaldan."""
    if display is None:
        if status == "evaluated" and value is not None:
            display = f"{int(round(value))}/100"
        elif status == "partial":
            display = "Parcial"
        elif status == "unverified":
            display = "No verificado"
        else:
            display = "No evaluable"
    return {
        "value": None if value is None else int(round(value)),
        "status": status,
        "reason": reason,
        "findings": [f for f in (findings or []) if f],
        "display": display,
    }


def _review_tokens(name) -> set:
    return {t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t}


def _review_is_stop(name) -> bool:
    text = str(name).lower()
    return bool(_review_tokens(text) & _REVIEW_STOP_TOKENS) or "stoploss" in text


def _review_is_take(name) -> bool:
    text = str(name).lower()
    return (
        bool(_review_tokens(text) & _REVIEW_TAKE_TOKENS)
        or "takeprofit" in text
        or "take_profit" in text
    )


def _review_control_depth(node, depth=0) -> int:
    deepest = depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.If, ast.For, ast.AsyncFor, ast.While,
                              ast.With, ast.AsyncWith, ast.Try)):
            deepest = max(deepest, _review_control_depth(child, depth + 1))
        else:
            deepest = max(deepest, _review_control_depth(child, depth))
    return deepest


def _review_comment_lines(source: str) -> int:
    import io
    import tokenize
    try:
        return sum(
            1
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.COMMENT
        )
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return sum(1 for line in source.splitlines() if line.lstrip().startswith("#"))


def _review_identifier_loads(tree) -> dict:
    loads: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loads[node.id] = loads.get(node.id, 0) + 1
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            loads[node.attr] = loads.get(node.attr, 0) + 1
    return loads


def _review_period_for(info, name) -> int | None:
    tokens = _review_tokens(name)
    for parametro in info.parametros:
        if tokens & _review_tokens(parametro.nombre):
            valor = parametro.valor
            if float(valor).is_integer() and 0 < valor <= 1000:
                return int(valor)
    return None


def _review_detect_indicators(tree, info):
    """Indicadores del parser + nombres de variable/columna reconocibles.

    Devuelve una lista de tuplas ``(familia, texto, variante)``. La familia
    (``sma``, ``ema``, ``atr``...) permite comprobar rápidamente qué tipo de
    indicador existe; ``variante`` distingue medias rápidas y lentas.
    """
    entries: list = []
    seen: set = set()

    def push(family: str, display: str, variant: str = "", generic: bool = False) -> None:
        token = (family, variant)
        if token in seen:
            return
        if generic:
            # No añadir una entrada genérica si la familia ya está representada.
            if any(entry[0] == family for entry in entries):
                return
        else:
            # Una entrada específica reemplaza a la genérica de su familia.
            if (family, "") in seen:
                seen.discard((family, ""))
                entries[:] = [e for e in entries if e[0] != family or e[2] != ""]
        seen.add(token)
        entries.append((family, display, variant))

    parser_keys = {
        "sma": "sma", "ema": "ema", "wma": "wma", "rsi": "rsi",
        "atr": "atr", "macd": "macd", "stddev": "std",
    }
    for name, _symbol, period, _resolution in info.indicadores:
        family = parser_keys.get(str(name).lower(), "ma")
        display = _REVIEW_INDICATOR_LABELS[family]
        if period:
            display += f" ({int(period)})"
        push(family, display, generic=not period)

    # Los identificadores tienen prioridad sobre el texto (docstrings, strings)
    # para que `sma_fast`/`sma_slow` no queden eclipsados por la documentación.
    ident_candidates: list = []
    string_candidates: list = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            string_candidates.append(node.value)
        elif isinstance(node, ast.Name):
            ident_candidates.append(node.id)
        elif isinstance(node, ast.arg):
            ident_candidates.append(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            ident_candidates.append(node.name)
        elif isinstance(node, ast.Attribute):
            ident_candidates.append(node.attr)

    for name in ident_candidates + sorted(set(string_candidates)):
        text = str(name).lower()
        tokens = _review_tokens(text)
        for token, family in _REVIEW_INDICATOR_TOKENS:
            if token in tokens or (len(token) > 2 and token in text):
                variant = ""
                if any(k in text for k in ("fast", "rapida", "short")):
                    variant = "rapida"
                elif any(k in text for k in ("slow", "lenta", "long", "larga")):
                    variant = "lenta"
                display = _REVIEW_INDICATOR_LABELS[family]
                if variant == "rapida":
                    display += " rápida"
                elif variant == "lenta":
                    display += " lenta"
                period = None
                match = re.search(r"(\d{1,4})", text)
                if match:
                    period = int(match.group(1))
                else:
                    period = _review_period_for(info, text)
                if period:
                    display += f" ({period})"
                push(family, display, variant, generic=not variant and not period)
    return entries


def _review_detect_signals(tree, literals, call_names) -> list:
    order_calls = {
        "market_order", "order", "order_target", "order_target_percent",
        "set_holdings", "liquidate", "buy", "sell", "short", "cover",
    }
    keywords = ("signal", "senal", "señal", "position", "posicion",
                "entry", "exit", "buy", "sell", "compra", "venta")
    found = {c for c in call_names if c in order_calls}
    for literal in literals:
        low = str(literal).lower()
        if any(k in low for k in keywords):
            found.add(str(literal))
    return sorted(found)


def _review_detect_horizon(tree):
    res_map = {
        "MINUTE": "Intradía (minutos)", "HOUR": "Horario", "HOURLY": "Horario",
        "DAILY": "Diario", "WEEKLY": "Semanal", "MONTHLY": "Mensual",
    }
    found: list = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        attr = node.attr.upper()
        if attr not in res_map:
            continue
        owner = node.value
        owner_name = getattr(owner, "id", None) or getattr(owner, "attr", "") or ""
        if "resolution" in str(owner_name).lower() and res_map[attr] not in found:
            found.append(res_map[attr])
    if found:
        return ", ".join(found), "Frecuencia temporal declarada explícitamente (Resolution.*)."
    return (
        "No especificado",
        "No se declara una frecuencia temporal; no se infiere a partir de los indicadores.",
    )


def _review_detect_strategy_type(blob, has_two_ma, ma_displays, has_rsi):
    if any(k in blob for k in ("grid", "escalon", "escalera", "malla")):
        return "Escalonado / Grid", "Se detectaron referencias a escalones o niveles de precio."
    if has_rsi and any(k in blob for k in ("reversion", "mean", "sobreventa",
                                            "sobrecompra", "oversold", "overbought")):
        return "Reversión a la media", "RSI combinado con lógica de reversión a la media."
    if has_two_ma:
        return "Cruce de medias", "Se detectaron dos medias móviles (rápida y lenta)."
    if any(k in blob for k in ("momentum", "roc")):
        return "Momentum", "Se detectaron cálculos de momentum o tasa de cambio."
    if any(k in blob for k in ("vol_target", "target_vol", "volatil")):
        return "Volatilidad objetivo", "Se detectaron cálculos de volatilidad objetivo."
    if ma_displays:
        return "Tendencia", "Se detectó al menos una media móvil y lógica direccional."
    return "No identificado", "No hay evidencia estática suficiente para clasificar el tipo de estrategia."


def _review_build(source, tree, info, detected_symbols) -> dict:
    """Evaluación estructural a partir del AST ya parseado (nunca ejecuta código)."""
    dialect = {
        strategy_parser.DIALECTO_LEAN: "QuantConnect Lean",
        strategy_parser.DIALECTO_VECTOR: "Vectorial (`run_strategy`)",
    }.get(info.dialecto, "No determinado")
    dialect = dialect if info.valido else "No determinado"

    if tree is None:
        not_evaluable = _review_evaluation(
            None, "not_evaluable", "No evaluable: el archivo tiene errores de sintaxis."
        )
        evaluations = {REVIEW_GENERAL_LABEL: _review_evaluation(
            None, "not_evaluable",
            "Evaluación insuficiente: no se pudo analizar el archivo.",
            display="Evaluación insuficiente",
        )}
        for label in REVIEW_DIMENSION_LABELS:
            evaluations[label] = dict(not_evaluable)
        return {
            "evaluations": evaluations,
            "scores": {label: ev["display"] for label, ev in evaluations.items()},
            "strategy_type": "No identificado",
            "strategy_type_note": "No se pudo analizar el archivo.",
            "market": "No especificado",
            "market_note": "No se pudo analizar el archivo.",
            "horizon": "No especificado",
            "horizon_note": "No se pudo analizar el archivo.",
            "indicators": [],
            "indicators_note": "No se pudo analizar el archivo.",
            "complexity": "No determinado",
            "complexity_note": "No se pudo analizar el archivo.",
            "strengths": [],
            "weaknesses": [
                "El archivo contiene errores de sintaxis; no se puede evaluar su estructura."
            ],
        }

    functions = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    n_functions = len(functions)
    n_branches = sum(isinstance(n, (ast.If, ast.For, ast.AsyncFor, ast.While,
                                    ast.Try, ast.ExceptHandler))
                     for n in ast.walk(tree))
    n_bool = sum(len(n.values) for n in ast.walk(tree) if isinstance(n, ast.BoolOp))
    n_try = sum(isinstance(n, ast.Try) for n in ast.walk(tree))
    n_raise = sum(isinstance(n, ast.Raise) for n in ast.walk(tree))
    n_returns = sum(isinstance(n, ast.Return) for n in ast.walk(tree))
    max_nesting = _review_control_depth(tree)
    lineas = source.count("\n") + 1

    documented = (1 if ast.get_docstring(tree) else 0) + sum(
        1 for fn in functions if ast.get_docstring(fn)
    ) + sum(1 for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and ast.get_docstring(n))
    comments = _review_comment_lines(source)

    ident_names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.arg)):
            ident_names.add(node.id if isinstance(node, ast.Name) else node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            ident_names.add(node.name)
    ident_names = {n for n in ident_names if n and not re.fullmatch(r"_+", n)}
    descriptive = [
        n for n in ident_names
        if "_" in n or (len(n) >= 4 and n.lower() not in _REVIEW_NON_DESCRIPTIVE)
    ]
    naming_ratio = len(descriptive) / max(1, len(ident_names))

    literals = [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    call_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                call_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                call_names.add(node.func.attr)
    loads = _review_identifier_loads(tree)
    all_idents = set(loads) | ident_names | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }

    indicator_entries = _review_detect_indicators(tree, info)
    indicator_displays = [display for _family, display, _variant in indicator_entries]
    indicator_keys = {family for family, _display, _variant in indicator_entries}
    ma_displays = [display for family, display, _variant in indicator_entries
                   if family in {"sma", "ema", "wma", "ma"}]
    has_two_ma = len(ma_displays) >= 2
    has_atr = "atr" in indicator_keys
    has_rsi = "rsi" in indicator_keys
    signals = _review_detect_signals(tree, literals, call_names)
    blob = " ".join(sorted(str(x).lower() for x in (all_idents | set(literals))))

    # --- Claridad ---------------------------------------------------------- #
    doc_ratio = documented / max(1, n_functions + 1)
    comment_score = min(1.0, comments / max(1, lineas / 40))
    nesting_factor = 1.0 if max_nesting <= 2 else (0.7 if max_nesting <= 4 else 0.4)
    modular_factor = 1.0 if n_functions >= 2 else 0.6
    structure_score = 0.5 * nesting_factor + 0.5 * modular_factor
    clarity_value = 100 * (
        0.25 * doc_ratio + 0.15 * comment_score
        + 0.35 * naming_ratio + 0.25 * structure_score
    )
    clarity_findings = []
    if documented:
        clarity_findings.append(f"{documented} docstring(s) encontrados.")
    if comments:
        clarity_findings.append(f"{comments} línea(s) de comentario.")
    if n_functions:
        clarity_findings.append(f"Código repartido en {n_functions} función(es).")
    clarity = _review_evaluation(
        clarity_value, "evaluated",
        "Se evalúan documentación, comentarios, nombres y estructura; "
        "no se penaliza por longitud en sí.",
        clarity_findings,
    )

    # --- Completitud ------------------------------------------------------- #
    entry_ok = bool(info.valido)
    has_params = bool(info.parametros)
    has_indicators = bool(indicator_displays)
    has_signals = bool(signals)
    output_ok = n_returns > 0 or info.n_ordenes > 0 or "on_data" in set(info.metodos)
    completeness_value = (
        (40 if entry_ok else 0) + (20 if has_indicators else 0)
        + (15 if has_params else 0) + (15 if has_signals else 0)
        + (10 if output_ok else 0)
    )
    completeness_findings = []
    if entry_ok:
        completeness_findings.append(f"Contrato reconocido: {dialect}.")
    if has_indicators:
        completeness_findings.append(f"{len(indicator_displays)} indicador(es) identificado(s).")
    if has_params:
        completeness_findings.append(f"{len(info.parametros)} parámetro(s) con valor por defecto.")
    if has_signals:
        completeness_findings.append("Generación de señales u órdenes identificada.")
    completeness_reason = (
        f"Contrato de estrategia reconocido ({dialect})."
        if entry_ok
        else "No expone el contrato de QuanConnect (clase `QCAlgorithm` con "
             "`initialize`, o función `run_strategy(data, ...)`)."
    )
    completeness = _review_evaluation(
        completeness_value, "evaluated", completeness_reason, completeness_findings
    )

    # --- Robustez estática estimada --------------------------------------- #
    null_calls = sum(
        1 for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr in {
            "dropna", "fillna", "ffill", "bfill", "isna", "notna",
            "interpolate", "combine_first",
        }
    )
    none_checks = zero_checks = column_checks = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and comparator.value is None:
                    none_checks += 1
                if isinstance(comparator, ast.Constant) and comparator.value == 0:
                    zero_checks += 1
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(op, (ast.In, ast.NotIn)):
                    attrs = {a.attr for a in ast.walk(comparator)
                             if isinstance(a, ast.Attribute)}
                    if {"columns", "keys", "index"} & attrs:
                        column_checks += 1
    divisions = sum(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)
                    for n in ast.walk(tree))
    evidence = (n_try * 2 + n_raise + null_calls + none_checks
                + zero_checks + column_checks * 2)
    robustness_findings = []
    if n_try:
        robustness_findings.append(f"Manejo de errores con try/except ({n_try}).")
    if n_raise:
        robustness_findings.append("Se lanzan errores explícitos con `raise`.")
    if null_calls:
        robustness_findings.append(f"Tratamiento de valores nulos ({null_calls}).")
    if none_checks:
        robustness_findings.append("Comprobaciones de valores `None`.")
    if zero_checks:
        robustness_findings.append("Comprobaciones de divisores distintos de cero.")
    if column_checks:
        robustness_findings.append("Validación de columnas o claves antes de acceder.")
    if evidence:
        robustness = _review_evaluation(
            min(100, 30 + 9 * evidence), "evaluated",
            "Robustez estimada por validaciones y manejo de errores presentes; "
            "no es una prueba de ejecución.",
            robustness_findings,
        )
    else:
        robustness = _review_evaluation(
            None, "partial",
            "No se encontraron validaciones explícitas; el análisis estático "
            "no permite confirmar la robustez.",
        )

    # --- Gestión de riesgo ------------------------------------------------- #
    stop_idents = {n for n in all_idents if _review_is_stop(n)}
    take_idents = {n for n in all_idents if _review_is_take(n)}
    stop_cols = {lit for lit in literals if _review_is_stop(lit)}
    take_cols = {lit for lit in literals if _review_is_take(lit)}
    pos_idents = {n for n in all_idents
                  if _review_tokens(n) & _REVIEW_POSITION_TOKENS}
    pos_cols = {lit for lit in literals
                if _review_tokens(lit) & _REVIEW_POSITION_TOKENS}
    stop_used = any(loads.get(n, 0) > 0 for n in stop_idents) or any(
        source.count(str(c)) >= 2 for c in stop_cols
    )
    take_used = any(loads.get(n, 0) > 0 for n in take_idents) or any(
        source.count(str(c)) >= 2 for c in take_cols
    )
    pos_used = any(loads.get(n, 0) > 0 for n in pos_idents) or any(
        source.count(str(c)) >= 2 for c in pos_cols
    )
    stop_declared = bool(stop_idents or stop_cols)
    take_declared = bool(take_idents or take_cols)
    risk_findings = []
    if stop_used or take_used:
        risk_findings = []
        if stop_used:
            risk_findings.append("Stop-loss referenciado y utilizado en el código.")
        if take_used:
            risk_findings.append("Take-profit o objetivo de beneficio utilizado.")
        if pos_used:
            risk_findings.append("Cálculo de tamaño o exposición de posición presente.")
        risk_value = 45 + 20 * stop_used + 20 * take_used + 10 * pos_used
        risk = _review_evaluation(
            min(100, risk_value), "evaluated",
            "Controles de riesgo presentes y utilizados en el código; el análisis "
            "estático no valida su eficacia en mercado.",
            risk_findings,
        )
    elif stop_declared or take_declared:
        if stop_declared and not stop_used:
            risk_findings.append("Parámetro de stop-loss declarado pero sin uso detectable.")
        if take_declared and not take_used:
            risk_findings.append("Parámetro de take-profit declarado pero sin uso detectable.")
        risk = _review_evaluation(
            None, "partial",
            "Hay parámetros de riesgo declarados, pero no se pudo comprobar que se apliquen.",
            risk_findings,
        )
    else:
        risk = _review_evaluation(
            None, "unverified",
            "No se detectaron referencias a stop-loss, take-profit ni límites de "
            "posición; el análisis estático no confirma su ausencia.",
        )

    # --- Calificación general --------------------------------------------- #
    evaluations = {
        "Claridad del código": clarity,
        "Gestión de riesgo": risk,
        "Completitud": completeness,
        "Robustez estática estimada": robustness,
    }
    numeric = {label: ev["value"] for label, ev in evaluations.items()
               if ev["value"] is not None}
    if "Completitud" in numeric and len(numeric) >= 2:
        weight = sum(REVIEW_WEIGHTS[label] for label in numeric)
        general_value = sum(REVIEW_WEIGHTS[label] * value
                            for label, value in numeric.items()) / weight
        included = ", ".join(sorted(numeric, key=lambda l: REVIEW_WEIGHTS[l], reverse=True))
        general = _review_evaluation(
            general_value, "evaluated",
            "Promedio ponderado de la calidad estructural (incluye: "
            f"{included}); no mide rentabilidad ni ejecuta backtests.",
        )
    else:
        general = _review_evaluation(
            None, "not_evaluable",
            "Evaluación insuficiente: no hay evidencia estática suficiente para "
            "una calificación confiable.",
            display="Evaluación insuficiente",
        )
    evaluations = {REVIEW_GENERAL_LABEL: general, **evaluations}

    # --- Hechos descriptivos ---------------------------------------------- #
    if detected_symbols:
        market_value = ", ".join(detected_symbols)
        market_note = "Activos identificados literalmente en el código."
    else:
        market_value = "No especificado"
        market_note = "No se declara un activo o mercado literal en el código."
    horizon_value, horizon_note = _review_detect_horizon(tree)
    strategy_type_value, strategy_type_note = _review_detect_strategy_type(
        blob, has_two_ma, ma_displays, has_rsi
    )
    indicators_note = (
        "Indicadores y cálculos identificados en el código."
        if indicator_displays
        else "No se detectaron indicadores reconocibles."
    )
    complexity_index = (
        n_functions + (n_branches + n_bool) * 0.5
        + len(indicator_displays) * 1.5 + len(info.librerias) * 0.5
    )
    if complexity_index <= 8 and n_functions <= 4:
        complexity_value = "Baja"
    elif complexity_index <= 25:
        complexity_value = "Media"
    else:
        complexity_value = "Alta"
    complexity_note = (
        f"{n_functions} función(es), {n_branches} rama(s), "
        f"{len(indicator_displays)} indicador(es), "
        f"{len(info.librerias)} dependencia(s)."
    )

    # --- Puntos fuertes y débiles ----------------------------------------- #
    strengths = []
    weaknesses = []
    if entry_ok:
        strengths.append(f"Contrato de estrategia reconocido ({dialect}).")
    else:
        weaknesses.append(
            "No se encontró la función `run_strategy` ni una clase tipo "
            "`QCAlgorithm` con `initialize`."
        )
    if has_two_ma:
        strengths.append("Medias móviles rápida y lenta detectadas.")
    if has_atr:
        strengths.append("Cálculo de ATR identificado.")
    if info.parametros:
        strengths.append(f"Parámetros configurables encontrados: {len(info.parametros)}.")
    if column_checks or null_calls or n_try:
        strengths.append("Validaciones de datos o manejo de errores identificados.")
    if stop_used or take_used:
        strengths.append("Controles de riesgo (stop-loss/take-profit) utilizados.")
    if signals:
        strengths.append(f"Generación de señales identificada ({len(signals)}).")
    if documented and comment_score >= 0.5:
        strengths.append("Documentación y comentarios presentes.")

    if not entry_ok:
        pass  # ya anotado arriba
    if risk["status"] == "partial":
        for finding in risk["findings"]:
            weaknesses.append(finding)
    elif risk["status"] == "unverified":
        weaknesses.append(
            "No se detectaron controles de riesgo explícitos; el análisis "
            "estático no confirma su ausencia."
        )
    if not (column_checks or null_calls or n_try or none_checks or zero_checks):
        weaknesses.append(
            "No se encontraron validaciones explícitas de columnas o datos."
        )
    if divisions and not zero_checks:
        weaknesses.append(
            "Se detectaron divisiones sin comprobación explícita del denominador."
        )
    if not documented and comment_score < 0.5:
        weaknesses.append("Documentación escasa en el código.")
    if not indicator_displays:
        weaknesses.append("No se detectaron indicadores reconocibles.")

    strengths = list(dict.fromkeys(strengths))[:5]
    weaknesses = list(dict.fromkeys(weaknesses))[:5]

    return {
        "evaluations": evaluations,
        "scores": {label: ev["display"] for label, ev in evaluations.items()},
        "strategy_type": strategy_type_value,
        "strategy_type_note": strategy_type_note,
        "market": market_value,
        "market_note": market_note,
        "horizon": horizon_value,
        "horizon_note": horizon_note,
        "indicators": indicator_displays,
        "indicators_note": indicators_note,
        "complexity": complexity_value,
        "complexity_note": complexity_note,
        "strengths": strengths or ["No se identificaron puntos fuertes verificables."],
        "weaknesses": weaknesses or [
            "No se identificaron debilidades verificables con el análisis estático."
        ],
    }


def analyze_strategy_source(source: str, filename: str, relative_path: str = "") -> dict:
    """Analiza de forma estática una estrategia; nunca ejecuta su código."""
    info = strategy_parser.analizar_fuente(
        source, nombre=filename, ruta=relative_path
    )
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        tree = None
        syntax_error = f"Error de sintaxis en línea {exc.lineno}: {exc.msg}"
    else:
        syntax_error = None
    literal_symbols = {
        node.value.upper()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and re.fullmatch(r"[A-Z]{1,5}(?:-[A-Z]{1,2})?", node.value.upper())
    } if tree is not None else set()
    detected_symbols = [
        symbol for symbol in info.simbolos if symbol in literal_symbols
    ]

    review = _review_build(source, tree, info, detected_symbols)
    review["valid"] = info.valido
    review["syntax_valid"] = tree is not None
    review["syntax_error"] = syntax_error
    review["error"] = info.error
    review["execution_validation"] = validate_strategy_source(source, filename)
    review["dialect"] = {
        strategy_parser.DIALECTO_LEAN: "QuantConnect Lean",
        strategy_parser.DIALECTO_VECTOR: "Vectorial (`run_strategy`)",
    }.get(info.dialecto, "No determinado") if info.valido else "No determinado"
    review["symbols"] = detected_symbols
    review["description"] = info.docstring.strip() if info.docstring else ""
    return review


def reset_strategy_picker(mode: str = "principal") -> None:
    """Start a clean modal flow without changing the strategy open in the lab."""
    st.session_state[MODAL_MODO_KEY] = mode
    st.session_state[MODAL_VIEW_KEY] = "empty"
    for key in (
        MODAL_SELECTED_PATH_KEY,
        MODAL_SELECTED_SOURCE_KEY,
        MODAL_SELECTED_HASH_KEY,
        MODAL_SELECTED_NAME_KEY,
        MODAL_ERROR_KEY,
        MODAL_SEARCH_KEY,
        MODAL_FILE_KEY,
        MODAL_UPLOAD_KEY,
        MODAL_PROVENANCE_KEY,
        MODAL_DROP_EVENT_KEY,
        MODAL_ACTION_EVENT_KEY,
    ):
        st.session_state.pop(key, None)


def store_strategy_preview(
    source: str, filename: str, relative_path: str | None = None, digest: str | None = None
) -> None:
    st.session_state[MODAL_SELECTED_PATH_KEY] = relative_path
    st.session_state[MODAL_SELECTED_SOURCE_KEY] = source
    st.session_state[MODAL_SELECTED_HASH_KEY] = digest or hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()
    st.session_state[MODAL_SELECTED_NAME_KEY] = filename
    st.session_state.pop(MODAL_ERROR_KEY, None)
    st.session_state[MODAL_VIEW_KEY] = "review"


def process_strategy_picker_file(payload: dict) -> bool:
    """Validate a browser-selected Python file and refresh its preview atomically."""
    for key in (
        MODAL_SELECTED_PATH_KEY,
        MODAL_SELECTED_SOURCE_KEY,
        MODAL_SELECTED_HASH_KEY,
        MODAL_SELECTED_NAME_KEY,
        MODAL_PROVENANCE_KEY,
        MODAL_ERROR_KEY,
    ):
        st.session_state.pop(key, None)
    st.session_state[MODAL_VIEW_KEY] = "empty"

    filename = payload.get("name")
    read_error = payload.get("read_error")
    encoded_content = payload.get("content_base64")
    if not isinstance(filename, str) or Path(filename).suffix.casefold() != ".py":
        st.session_state[MODAL_ERROR_KEY] = "Solo se admiten archivos `.py`."
        return False
    if read_error:
        st.session_state[MODAL_ERROR_KEY] = (
            f"No se pudo leer `{filename}`; no se cargó: {read_error}"
        )
        return False
    if not isinstance(encoded_content, str):
        st.session_state[MODAL_ERROR_KEY] = (
            f"No se pudo leer `{filename}` desde el selector de archivos."
        )
        return False

    try:
        raw_content = base64.b64decode(encoded_content, validate=True)
        source = normalize_source_newlines(raw_content.decode("utf-8"))
        matched_path = match_uploaded_strategy(source, filename)
        if matched_path:
            candidate_source, digest = read_strategy_candidate(matched_path)
            if candidate_source != source:
                raise OSError(
                    f"`{matched_path}` cambió mientras se verificaba el archivo."
                )
            store_strategy_preview(
                candidate_source, Path(matched_path).name, matched_path, digest
            )
            st.session_state[MODAL_PROVENANCE_KEY] = "catalog"
        else:
            store_strategy_preview(source, Path(filename).name)
            st.session_state[MODAL_PROVENANCE_KEY] = "external"
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        st.session_state[MODAL_ERROR_KEY] = (
            f"No se pudo validar `{filename}`; no se cargó: {exc}"
        )
        return False
    return True


def select_strategy_for_review(relative_path: str) -> bool:
    """Replace the modal preview only after a fresh, safe file read succeeds."""
    try:
        source, digest = read_strategy_candidate(relative_path)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        for key in (
            MODAL_SELECTED_PATH_KEY,
            MODAL_SELECTED_SOURCE_KEY,
            MODAL_SELECTED_HASH_KEY,
            MODAL_SELECTED_NAME_KEY,
        ):
            st.session_state.pop(key, None)
        st.session_state[MODAL_ERROR_KEY] = f"No se pudo leer `{relative_path}`: {exc}"
        return False
    store_strategy_preview(source, Path(relative_path).name, relative_path, digest)
    st.session_state[MODAL_PROVENANCE_KEY] = "catalog"
    return True


def match_uploaded_strategy(source: str, filename: str) -> str | None:
    """Map an uploaded preview to a unique identical project file, if one exists."""
    if Path(filename).suffix.casefold() != ".py":
        raise ValueError("Solo se admiten archivos `.py`.")
    source = normalize_source_newlines(source)
    matches = []
    for candidate in ALL_STRATEGIES:
        try:
            candidate_source, _candidate_digest = read_strategy_candidate(candidate)
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        if candidate_source == source:
            matches.append(candidate)
    matching_names = [
        candidate for candidate in matches
        if Path(candidate).name.casefold() == Path(filename).name.casefold()
    ]
    if len(matching_names) == 1:
        return matching_names[0]
    return matches[0] if len(matches) == 1 else None


def filter_strategy_candidates(paths: list[str], query: str) -> list[str]:
    query = query.strip().casefold()
    return [path for path in paths if query in path.casefold()]


def clear_strategy_picker_choice() -> None:
    st.session_state[MODAL_FILE_KEY] = None


def request_editor_navigation(
    action: str,
    target: str | None = None,
    target_source: str | None = None,
    target_sha256: str | None = None,
) -> bool:
    """Queue a close or replacement only when the current editor needs a decision."""
    if action not in {"close", "replace", "remove_comparison", "replace_comparison"}:
        raise ValueError(f"Acción de navegación no válida: {action}")
    if action in {"remove_comparison", "replace_comparison"}:
        if not target or target not in st.session_state.get(COMPARE_KEY, []):
            if action == "remove_comparison":
                return False
            current = (st.session_state.get(COMPARE_KEY) or [None])[0]
            if not current:
                return False
        else:
            current = target
        editor_kind = "comparison"
    else:
        current = st.session_state.get(PRINCIPAL_KEY) or ""
        editor_kind = "principal"
    if action in {"replace", "replace_comparison"} and (not target or target == current):
        return False
    if not current:
        return False

    source = editor_source_for(current, editor_kind)
    state, _message = editor_sync_status(current, source)
    if state == "saved":
        return False

    st.session_state[EDITOR_NAVIGATION_KEY] = {
        "action": action,
        "current": current,
        "target": target,
        "editor_kind": editor_kind,
        "target_source": target_source,
        "target_sha256": target_sha256,
    }
    return True


def resolve_editor_navigation(decision: str, available: list[str]) -> str:
    """Resolve a queued editor navigation; save errors leave the decision pending."""
    pending = st.session_state.get(EDITOR_NAVIGATION_KEY)
    if not pending:
        raise ValueError("No hay un cambio de estrategia pendiente de confirmar.")
    if decision not in {"save", "discard", "cancel"}:
        raise ValueError(f"Decisión de navegación no válida: {decision}")
    if decision == "cancel":
        st.session_state.pop(EDITOR_NAVIGATION_KEY, None)
        return "cancelled"

    current = pending.get("current") or ""
    editor_kind = pending.get("editor_kind", "principal")
    action = pending["action"]
    target = pending.get("target")
    target_source = None
    if action in {"replace", "replace_comparison"}:
        target_is_external = is_external_strategy(target or "")
        if target_is_external:
            record = external_strategy_record(target)
            if record is None:
                raise ValueError("La copia externa de trabajo ya no está disponible en esta sesión.")
            target_source = str(record.get("source", ""))
            target_hash = hashlib.sha256(target_source.encode("utf-8")).hexdigest()
            if pending.get("target_sha256") and target_hash != pending["target_sha256"]:
                raise ValueError(
                    "El archivo externo cambió desde su análisis. Cancela y vuelve a seleccionarlo."
                )
        elif target not in available:
            raise ValueError(f"La estrategia seleccionada ya no está disponible: {target}")
        else:
            target_source, target_hash = read_strategy_candidate(target)
            if pending.get("target_sha256") and target_hash != pending["target_sha256"]:
                raise ValueError(
                    f"`{target}` cambió desde su análisis. Cancela y vuelve a seleccionarla."
                )
        if target in st.session_state.get(COMPARE_KEY, []):
            if comparison_editor_has_unresolved_changes(target):
                raise ValueError(
                    f"Guarda o descarta explícitamente los cambios de `{target}` "
                    "en su editor comparativo antes de promoverla."
                )
    if action == "remove_comparison" and current not in st.session_state.get(COMPARE_KEY, []):
        raise ValueError(f"La comparativa ya no está seleccionada: {current}")
    if decision == "save":
        source = editor_source_for(current, editor_kind)
        if editor_save_button_disabled(current, source):
            raise ValueError("No se pueden guardar cambios si el archivo no se verifica.")
        save_editor_source(current, source)
    elif decision == "discard" and action in {"remove_comparison", "replace_comparison"}:
        discard_comparison_editor(current)

    st.session_state.pop(EDITOR_NAVIGATION_KEY, None)
    if action == "close":
        clear_principal_strategy()
    elif action == "replace":
        set_principal_strategy(target, available, source=target_source)
        st.session_state[BACKTEST_PENDING_KEY] = target
    elif action == "replace_comparison":
        set_comparative_strategy(target, target_source)
    elif action == "remove_comparison":
        st.session_state[COMPARE_KEY] = [
            script
            for script in st.session_state.get(COMPARE_KEY, [])
            if script != current
        ]
        st.session_state.pop(COMPARE_PENDING_KEY, None)
    return "completed"


@st.dialog("Cambios sin guardar", width="small")
def render_editor_navigation_confirmation(available: list[str]) -> bool:
    """Render the save/discard/cancel decision for a queued close or replacement."""
    pending = st.session_state.get(EDITOR_NAVIGATION_KEY)
    if not pending:
        return False

    current = pending.get("current") or ""
    editor_kind = pending.get("editor_kind", "principal")
    source = editor_source_for(current, editor_kind) if current else ""
    state, message = editor_sync_status(current, source)
    target = pending.get("target")
    action_label = (
        f"cambiar a `{target}`"
        if pending["action"] in {"replace", "replace_comparison"}
        else "cerrar la estrategia principal"
        if pending["action"] == "close"
        else "quitar o reemplazar esta estrategia comparativa"
    )
    st.markdown(
        f"Hay cambios sin guardar en **{strategy_filename(current)}**. "
        f"Elige qué hacer antes de {action_label}."
    )
    save_col, discard_col, cancel_col = st.columns(3)
    with save_col:
        save_clicked = st.button(
            "Guardar cambios",
            key=f"editor_nav_save_{pending['action']}",
            disabled=editor_save_button_disabled(current, source),
            use_container_width=True,
        )
    with discard_col:
        discard_clicked = st.button(
            "Descartar cambios",
            key=f"editor_nav_discard_{pending['action']}",
            use_container_width=True,
        )
    with cancel_col:
        cancel_clicked = st.button(
            "Cancelar",
            key=f"editor_nav_cancel_{pending['action']}",
            use_container_width=True,
        )

    decision = (
        "save" if save_clicked else
        "discard" if discard_clicked else
        "cancel" if cancel_clicked else
        None
    )
    if decision:
        try:
            outcome = resolve_editor_navigation(decision, available)
        except (OSError, ValueError) as exc:
            st.error(f"No se pudo completar la operación; se conservaron los cambios: {exc}")
        else:
            if outcome == "cancelled":
                st.session_state["flash_msg"] = (
                    "Cambio cancelado; se conservan la estrategia y la edición actual."
                )
                st.rerun()
            else:
                st.rerun()
    return True


# -----------------------------------------------------------------------------
# CATÁLOGO DE ESTRATEGIAS: PRINCIPAL VS COMPARATIVAS
# -----------------------------------------------------------------------------
PRINCIPAL_KEY = "principal_script"
COMPARE_KEY = "compare_selection"
COMPARE_CACHE_KEY = "compare_cache"
CHART_VIEW_KEY = "chart_view"
MAX_COMPARATIVES = 1
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
    """Normaliza el estado de principal/comparativas contra el catálogo real.

    La Estrategia Principal NO se rellena sola: si no hay ninguna, o si el archivo
    que lo era ya no está en el proyecto, el estado es vacío (`None`). El laboratorio
    espera a que el usuario abra una estrategia con «📂 Abrir Estrategia».
    """
    principal = st.session_state.get(PRINCIPAL_KEY)
    if (
        principal is not None
        and principal not in available
        and external_strategy_record(principal) is None
    ):
        principal = None
    st.session_state[PRINCIPAL_KEY] = principal

    # Nunca comparar la principal contra sí misma, ni dejar más de MAX_COMPARATIVAS.
    keep = [
        script
        for script in (st.session_state.get(COMPARE_KEY) or [])
        if (
            (script in available or external_strategy_record(script) is not None)
            and script != principal
        )
    ]
    st.session_state[COMPARE_KEY] = keep[-MAX_COMPARATIVES:]

    # La caché de comparativas pertenece a un conjunto concreto de estrategias.
    cached = st.session_state.get(COMPARE_CACHE_KEY) or {}
    st.session_state[COMPARE_CACHE_KEY] = {
        script: payload
        for script, payload in cached.items()
        if script in available or external_strategy_record(script) is not None
    }


def set_principal_strategy(
    script: str, available: list[str], source: str | None = None
) -> None:
    """Promueve `script` a Estrategia Principal y reparte el resto de estado."""
    if script not in available and external_strategy_record(script) is None:
        return
    st.session_state.pop(BACKTEST_PENDING_KEY, None)
    if script in st.session_state.get(COMPARE_KEY, []):
        if comparison_editor_has_unresolved_changes(script):
            raise ValueError(
                f"Guarda o descarta explícitamente los cambios de `{script}` "
                "en su editor comparativo antes de promoverla."
            )
    st.session_state[PRINCIPAL_KEY] = script
    # El archivo nuevo entra también en la lista de archivos navegables del editor.
    if not is_external_strategy(script) and script not in py_files:
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
    st.session_state["codigo_actual"] = (
        source if source is not None else read_script_source(script)
    )
    # Un principal nuevo invalida su backtest y cualquier comparación en curso.
    st.session_state.pop(RESULT_KEY, None)
    st.session_state.pop(PARAMS_KEY, None)
    for key in (DRAFT_KEY, DRAFT_META_KEY, DRAFT_RESULT_KEY, DRAFT_METRICS_KEY, DRAFT_EQUITY_KEY):
        st.session_state.pop(key, None)
    st.session_state.pop(DRAFT_SEEDED_KEY, None)
    st.session_state.pop("bt_ai_report", None)
    st.session_state.pop("bt_ai_auto_done", None)
    st.session_state.pop("force_run", None)
    st.session_state["flash_msg"] = (
        f"⭐ `{strategy_display_name(script)}` es ahora la Estrategia Principal"
    )


def clear_principal_strategy() -> None:
    """Quita la Estrategia Principal y devuelve el grupo ESTRATEGIAS al estado vacío.

    NO borra ni toca el archivo `.py` del proyecto: el bote de basura del grupo 1 solo
    deshace la asignación en memoria. Las comparativas tampoco se tocan (comparar no
    depende de tener Principal) y el editor solo se suelta lo justo para que no quede
    una Principal activa detrás del grupo.
    """
    # El nombre se guarda antes de limpiar el estado, para poder confirmar en el aviso
    # que estrategia deja de ser la Principal (y que su archivo sigue ahí).
    principal = st.session_state.get(PRINCIPAL_KEY)
    st.session_state[PRINCIPAL_KEY] = None
    st.session_state["script_nombre"] = ""
    st.session_state["script_selector"] = ""
    st.session_state["codigo_actual"] = SIN_ESTRATEGIA
    st.session_state["codigo_actual_script"] = ""
    st.session_state["editor_loaded_for"] = None
    st.session_state.pop(BACKTEST_PENDING_KEY, None)
    # El resultado, el borrador y el análisis de IA pertenecían a la Principal que
    # acaba de irse: se limpian igual que al cambiar de estrategia, para que nada de
    # lo anterior pueda aparecer como si fuera de la siguiente.
    st.session_state.pop(RESULT_KEY, None)
    st.session_state.pop(PARAMS_KEY, None)
    for key in (DRAFT_KEY, DRAFT_META_KEY, DRAFT_RESULT_KEY, DRAFT_METRICS_KEY, DRAFT_EQUITY_KEY):
        st.session_state.pop(key, None)
    st.session_state.pop(DRAFT_SEEDED_KEY, None)
    st.session_state.pop("bt_ai_report", None)
    st.session_state.pop("bt_ai_auto_done", None)
    st.session_state.pop("force_run", None)
    st.session_state["flash_msg"] = (
        f"🗑️ `{principal}` deja de ser la Estrategia Principal · su archivo sigue en el proyecto"
        if principal
        else "🗑️ Sin Estrategia Principal · el archivo sigue en el proyecto"
    )


def add_comparative_strategy(script: str, available: list[str]) -> tuple[bool, str]:
    """Añade `script` a las comparativas y devuelve `(aceptado, mensaje)`.

    Es el único camino de alta del grupo ESTRATEGIAS A COMPARAR: no toca la Estrategia
    Principal ni el editor, porque comparar no puede cambiar lo que se está evaluando.
    Respeta `MAX_COMPARATIVES` y nunca admite la Principal como candidata.
    """
    if script not in available and external_strategy_record(script) is None:
        return False, f"`{script}` ya no está en el proyecto."

    if script == st.session_state.get(PRINCIPAL_KEY):
        return False, f"`{script}` ya es la Estrategia Principal: no se compara consigo misma."

    selected = list(st.session_state.get(COMPARE_KEY) or [])
    if script in selected:
        return False, f"`{script}` ya está en la lista de comparativas."
    if len(selected) >= MAX_COMPARATIVES:
        return False, f"Solo caben {MAX_COMPARATIVES} comparativas."

    st.session_state[COMPARE_KEY] = selected + [script]
    return True, f"⚖️ `{strategy_display_name(script)}` añadida a comparar"


def set_comparative_strategy(script: str, source: str | None = None) -> None:
    """Replace the single comparison selection without touching its project file."""
    if script == st.session_state.get(PRINCIPAL_KEY):
        raise ValueError("La estrategia principal no se puede comparar consigo misma.")
    if not script or (
        script not in ALL_STRATEGIES and external_strategy_record(script) is None
    ):
        raise ValueError(f"La estrategia seleccionada ya no está disponible: {script}")
    old_scripts = list(st.session_state.get(COMPARE_KEY) or [])
    for old in old_scripts:
        if old != script and comparison_editor_has_unresolved_changes(old):
            raise ValueError(
                f"Hay cambios pendientes en `{old}`. Guárdalos o descártalos antes de reemplazarla."
            )
    for old in old_scripts:
        if old != script:
            discard_comparison_editor(old)
    st.session_state[COMPARE_KEY] = [script]
    st.session_state[COMPARE_PENDING_KEY] = script
    if source is not None:
        buffers = dict(st.session_state.get(COMPARE_EDITOR_SOURCES_KEY) or {})
        buffers[script] = source
        st.session_state[COMPARE_EDITOR_SOURCES_KEY] = buffers
    st.session_state.pop(COMPARE_CACHE_KEY, None)


# -----------------------------------------------------------------------------
# DATOS QUE DECLARA LA ESTRATEGIA PRINCIPAL (G.3.6)
# -----------------------------------------------------------------------------
# El grupo REGLAS COMUNES no se inventa el Activo / Ticker ni el Capital Inicial: los
# lee del archivo que actúa como Principal y solo cae al valor de respaldo cuando el
# archivo no los declara. Lo que sale de aquí sigue siendo el único valor que alimenta
# `_config_lab`, de modo que motor, comparativas y borrador no pueden discrepar.


def tickers_declarados(path: Path, source: str | None = None) -> list[str]:
    """Tickers que el propio archivo declara como literales, en orden de aparición.

    Hace falta este barrido porque `strategy_parser` completa `info.simbolos` con `SPY`
    y `info.capital` con 100000 cuando el archivo no dice nada: con la metadata ya
    completada no se puede distinguir "lo declara el archivo" de "lo pone el motor".
    Se ordenan por linea para que el primero sea el que aparece antes en el codigo.
    """
    try:
        code = source if source is not None else path.read_text(
            encoding="utf-8", errors="replace"
        )
        arbol = ast.parse(code)
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


def reglas_comunes_de_estrategia(
    path: Path, info=None, source: str | None = None
) -> tuple[str, float]:
    """`(ticker, capital)` del archivo de la Estrategia Principal, con su respaldo.

    El Activo / Ticker es el primer ticker declarado que no sea índice ni reserva; si
    el archivo solo declara índices (una estrategia sobre SPY) se usa ese mismo, porque
    ahí sí es el instrumento. El capital es el que declara el archivo y, si no declara
    ninguno, el respaldo. El benchmark del archivo NO se lee aquí: es otro concepto.
    """
    info = info if info is not None else strategy_parser.analizar_archivo(path)
    declarados = tickers_declarados(path, source)
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
    source = read_script_source(script) if is_external_strategy(script) else None
    ticker, capital = reglas_comunes_de_estrategia(
        BASE_DIR / script, info, source=source
    )

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
    st.session_state["script_nombre"] = ""
if st.session_state["script_nombre"] and st.session_state["script_nombre"] not in py_files:
    py_files.append(st.session_state["script_nombre"])

# Catálogo completo (raíz + subcarpetas) para el gestor Principal vs Comparativas.
# El Grupo 1 arranca VACÍO a propósito: no se promueve ningún archivo por defecto, ni
# siquiera el primero que aparece en el proyecto. La Principal se elige con
# «📂 Abrir Estrategia» y se puede quitar con su 🗑️ sin tocar el archivo.
ALL_STRATEGIES = list_all_strategies(BASE_DIR) or list(py_files)
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


def _sello_parametros(parametros=None) -> tuple:
    """Huella estable de los overrides activos, para invalidar las cachés.

    Sin esto, mover un parametro dinamico no entraria en la clave y el backtest
    devolveria el resultado cacheado del valor anterior.
    """
    overrides = PARAMETROS_ACTIVOS if parametros is None else parametros
    return tuple(sorted((k, round(float(v), 6)) for k, v in overrides.items()))


def _sello_codigo_estrategia(script: str) -> tuple[str, str]:
    """Identifica el contenido guardado que leerá el runner para esta estrategia."""
    external = external_strategy_record(script)
    if external is not None:
        source = (
            str(st.session_state.get("codigo_actual", external.get("source", "")))
            if st.session_state.get("codigo_actual_script") == script
            else str(external.get("source", ""))
        )
        return ("sha256", hashlib.sha256(source.encode("utf-8")).hexdigest())
    try:
        source = (BASE_DIR / script).read_bytes()
    except OSError as exc:
        return ("unreadable", f"{type(exc).__name__}:{exc.errno}")
    return ("sha256", hashlib.sha256(source).hexdigest())


def _config_lab(ticker, capital, parametros=None):
    """Configuración de la corrida que se envía al motor de qc_lab."""
    return ConfigEjecucion(
        simbolo=str(ticker).upper(),
        capital=float(capital),
        comision=COMMISSION_RATE,
        parametros=dict(PARAMETROS_ACTIVOS if parametros is None else parametros),
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


def run_strategy_backtest(script, ticker, capital=100000, parametros=None):
    """Ejecuta la Estrategia Principal en el motor de qc_lab y devuelve (df, metrics).

    Acepta el archivo y no un simple ticker: el motor es el unico que sabe si la
    estrategia es una clase Lean o una funcion `run_strategy`, y ademas decide fechas,
    capital y universo a partir de lo que la propia estrategia declare.
    """
    cfg = _config_lab(ticker, capital, parametros)
    external = external_strategy_record(script)
    if external is not None:
        label = str(external.get("name") or "estrategia.py")
        cfg.etiqueta = label
        source = (
            str(st.session_state.get("codigo_actual", external.get("source", "")))
            if st.session_state.get("codigo_actual_script") == script
            else str(external.get("source", ""))
        )
        resultado = ejecutar_estrategia(
            fuente=source, config=cfg
        )
        return _contrato_de_lab(resultado, ticker, capital, cfg.parametros, label)
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


def run_comparative_backtest(script, ticker, capital, parametros=None):
    """Ejecuta una estrategia comparativa desde su archivo y devuelve (df, metrics).

    Los errores se capturan aquí y se devuelven como `error` para que una estrategia
    rota no tumbe la comparación completa.
    """
    try:
        df, metrics = run_strategy_backtest(script, ticker, capital, parametros)
    except Exception as exc:
        return None, {"error": str(exc), "script": script}
    return df, metrics


def get_comparative_results(scripts, ticker, capital, force=False) -> dict:
    """Results de las comparativas seleccionadas, cacheados por script + parámetros."""
    pending_script = st.session_state.get(COMPARE_PENDING_KEY)
    if pending_script in scripts and not force:
        return {}
    cache = st.session_state.get(COMPARE_CACHE_KEY) or {}
    wanted = [
        s for s in scripts
        if s in ALL_STRATEGIES or external_strategy_record(s) is not None
    ]
    out = {}
    for script in wanted:
        parametros = dict(
            (st.session_state.get(COMPARE_PARAMETERS_KEY) or {}).get(script) or {}
        )
        stamp = (
            ticker,
            _sello_parametros(parametros),
            float(capital),
            _sello_codigo_estrategia(script),
        )
        cached = cache.get(script)
        if not cached or cached.get("params") != stamp:
            df, metrics = run_comparative_backtest(
                script, ticker, capital, parametros
            )
            cached = {"params": stamp, "df": df, "metrics": metrics}
            cache[script] = cached
        out[script] = cached
    st.session_state[COMPARE_CACHE_KEY] = cache
    if force and pending_script in scripts:
        st.session_state.pop(COMPARE_PENDING_KEY, None)
    return out


RESULT_KEY = "backtest_result"
PARAMS_KEY = "backtest_params"


def resultado_en_espera(ticker, capital):
    """Curva plana y métricas a cero para cuando todavía no hay Estrategia Principal.

    Trae las MISMAS claves que devuelve el motor, para que graficas, barra de KPIs y
    tabla comparativa se puedan pintar sin excepciones. No lleva `error`: no es un
    fallo del motor, es el laboratorio esperando a que se abra una estrategia.
    """
    capital = float(capital)
    df = pd.DataFrame(
        {
            "Equity": [capital],
            "Benchmark": [capital],
            "Drawdown": [0.0],
            "Position": [0.0],
        },
        index=pd.to_datetime([pd.Timestamp(datetime.now()).normalize()]),
    )
    metrics = {
        "standby": True,
        "equity": capital,
        "total_return": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "sharpe": 0.0,
        "fees": 0.0,
        "volume": 0.0,
        "fills": 0,
        "trades": 0,
        "benchmark_return": 0.0,
        "benchmark_drawdown": 0.0,
        "bars": 0,
        "ticker": ticker,
        "capital": capital,
        "source": "sin ejecutar",
    }
    return df, metrics


def _resultado_plantilla(ticker, capital, force=False):
    """Resultado del borrador temporal (plantilla inicial) ya ejecutado, o `None`.

    El borrador vive solo en `st.session_state`, así que no hay archivo que leer. Se
    invalida igual que la Principal: por parámetros (ticker, capital, overrides) y por el
    hash del código editado. Así, cambiar el activo, el capital, un parámetro o el propio
    texto devuelve el laboratorio al estado de espera hasta que se vuelva a ejecutar.
    """
    stored = st.session_state.get(PLANTILLA_RESULT_KEY)
    if not stored:
        return None
    code = str(
        st.session_state.get(PLANTILLA_EDITOR_KEY)
        or st.session_state.get(PLANTILLA_CODE_KEY)
        or ""
    )
    params = (
        ticker,
        float(capital),
        _sello_parametros(),
        hashlib.sha256(code.encode("utf-8")).hexdigest(),
    )
    if not force and st.session_state.get(PLANTILLA_PARAMS_KEY) != params:
        return None
    return stored["df"], stored["metrics"]


def get_backtest_result(ticker, capital, force=False):
    """Resultado del backtest de la Estrategia Principal, cacheado por parámetros.

    La Estrategia Principal se lee del estado de sesión y se ejecuta en `qc_lab` igual
    que cualquier comparativa. Si el motor no puede ejecutarla se devuelve la curva y
    las metricas de referencia con el motivo en `error`, para que el panel siga
    operativo en lugar de dejar la app a medias.

    Sin Estrategia Principal no se ejecuta NADA: ni la anterior, ni el primer archivo
    del proyecto, ni uno inventado. Se devuelve el estado de espera y el laboratorio
    espera a que el usuario abra una estrategia.
    """
    script = st.session_state.get(PRINCIPAL_KEY)
    if not script:
        # Sin Estrategia Principal el laboratorio muestra la plantilla inicial. Si el
        # usuario ya la ejecutó, se devuelve su resultado; si no, el estado de espera.
        plantilla = _resultado_plantilla(ticker, capital, force=force)
        if plantilla is not None:
            return plantilla
        return resultado_en_espera(ticker, capital)
    if st.session_state.get(BACKTEST_PENDING_KEY) == script:
        if not force:
            return resultado_en_espera(ticker, capital)
        st.session_state.pop(BACKTEST_PENDING_KEY, None)

    params = (
        script,
        ticker,
        _sello_parametros(),
        float(capital),
        _sello_codigo_estrategia(script),
    )
    cached = st.session_state.get(RESULT_KEY)

    if not force and cached is not None and st.session_state.get(PARAMS_KEY) == params:
        return cached["df"], cached["metrics"]

    try:
        df, metrics = run_strategy_backtest(script, ticker, capital)
    except Exception as exc:
        df = pd.DataFrame({"Equity": [float(capital)], "Benchmark": [float(capital)],
                           "Drawdown": [0.0], "Position": [0.0]},
                          index=pd.to_datetime([pd.Timestamp.now().normalize()]))
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

# La configuración canónica de IA permanece bajo `hdr` para conservar las selecciones.
TOPBAR_AI_PREFIX = "hdr"
ASSISTANT_AI_PREFIX = "assistant_ai"

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

STRATEGY_CREATION_SYSTEM_PROMPT = """Eres un asistente cuantitativo que ayuda a diseñar estrategias
mediante una conversación iterativa para simulación y backtesting. La estrategia puede ser de
cualquier tipo compatible con el proyecto; no presupongas un nombre de archivo ni una estrategia
concreta.

Ayuda a definir, cuando resulte pertinente, el activo o universo, marco temporal, indicadores,
entradas y salidas, stop loss y take profit, tamaño de posición y gestión del riesgo, parámetros
ajustables, supuestos, limitaciones y datos necesarios para un backtest. No exijas que el usuario
responda todo desde el inicio: pregunta de forma breve por los aspectos más importantes que falten.

Resume las reglas acordadas en lenguaje claro, identifica lo que aún está pendiente y no afirmes
rentabilidad ni validación sin evidencia. El frontend controla por separado la aprobación explícita
de las reglas antes de solicitar código. Mientras no se autorice esa fase, no escribas Python ni
bloques de código; si el usuario lo pide, indícale que primero debe proponer y aprobar las reglas.

No ejecutes ni propongas como realizados backtests u órdenes. No afirmes que existe código validado.
Si el usuario pide código antes de su autorización explícita, continúa aclarando reglas y explica
que primero debe consolidar y aprobar la propuesta."""

STRATEGY_CREATION_CONSOLIDATION_PROMPT = """Consolida la estrategia discutida en una especificación
estructurada y editable. Incluye secciones para: objetivo y universo de activos; temporalidades y
datos necesarios; condiciones de entrada y salida; indicadores y parámetros configurables; gestión
del capital y del riesgo; supuestos; limitaciones; condiciones de invalidación; y datos pendientes.
Identifica explícitamente como «Pendiente» cada elemento desconocido o ambiguo, sin inventar valores.
No generes código ni afirmes validación o rentabilidad."""

STRATEGY_CREATION_CODE_SYSTEM_PROMPT = """Cuando se habilite la etapa posterior de generación,
prepara código Python completo para la estrategia genérica descrita por las reglas aprobadas y
propón un nombre de archivo descriptivo terminado en `.py` en una línea `Archivo sugerido: ...`.
Devuelve un único bloque ```python con el archivo completo y sin texto explicativo dentro del bloque.

Usa únicamente uno de los contratos que ya reconoce el proyecto:
- Vectorial: `def run_strategy(data, ...)` que devuelve un DataFrame de pandas, conserva `Close`
  y proporciona `Signal` o `Position` como exposición objetivo.
- LEAN: una clase que hereda de `QCAlgorithm`, con `initialize` y el callback de datos apropiado.
No inventes APIs del motor. No añadas conexión a brokers, ejecución real, escritura automática de
archivos ni modificaciones de motores existentes. No ejecutes ni afirmes validar el código, ni
afirmes rentabilidad sin evidencia."""


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


def extract_strategy_python_file(text: str) -> str | None:
    """Extract the full Python fence without trimming the generated file contents."""
    blocks = re.findall(r"```(?:python|py)\s*\r?\n(.*?)```", text, re.IGNORECASE | re.DOTALL)
    return max(blocks, key=len) if blocks else None


def suggested_strategy_filename(response: str, idea: str = "") -> str:
    """Read the model's suggested filename or derive a safe name from the user's idea."""
    match = re.search(
        r"(?im)^\s*(?:archivo sugerido|nombre de archivo)\s*:\s*`?([^`\r\n]+)`?\s*$",
        response,
    )
    candidate = match.group(1).strip().strip("\"'") if match else ""
    if candidate and not candidate.endswith(".py"):
        candidate = f"{candidate}.py"
    if is_valid_strategy_filename(candidate):
        return candidate

    stem = re.sub(r"[^A-Za-z0-9]+", "_", idea).strip("_").lower()[:48]
    return f"{stem or 'estrategia_nueva'}.py"


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


def build_strategy_creation_prompt(messages: list[dict[str, str]]) -> str:
    """Format the isolated strategy-design conversation for the configured provider."""
    transcript = []
    for message in messages:
        role = "Usuario" if message.get("role") == "user" else "Asistente"
        content = str(message.get("content", "")).strip()
        if content:
            transcript.append(f"{role}:\n{content}")
    return "\n\n".join(transcript)


def strategy_creation_response_is_code(response: str) -> bool:
    """Identify likely source code without discarding the assistant's response."""
    return bool(
        re.search(r"```(?:python|py)\b", response, re.IGNORECASE)
        or re.search(
            r"(?m)^\s*(?:class\s+\w+\s*(?:\(|:)|def\s+\w+\s*\(|"
            r"from\s+\w+\s+import\s+|import\s+\w+)",
            response,
        )
    )


def process_strategy_creation_message(
    user_message: str, *, generate_code: bool = False, consolidate_rules: bool = False
) -> None:
    """Keep every assistant response in history; never execute or save generated code."""
    message = user_message.strip()
    if not message:
        return

    phase = st.session_state.get(
        STRATEGY_CREATION_PHASE_KEY, STRATEGY_CREATION_PHASE_DESIGN
    )
    rules_state = st.session_state.get(STRATEGY_CREATION_RULES_KEY, {})
    approved_proposal = str(rules_state.get("proposal") or "").strip()
    rules_approval_valid = bool(
        rules_state.get("approved")
        and approved_proposal
        and rules_state.get("approved_text") == approved_proposal
        and not strategy_creation_response_is_code(approved_proposal)
    )
    if generate_code and (
        phase != STRATEGY_CREATION_PHASE_CODE_AUTHORIZED or not rules_approval_valid
    ):
        st.session_state[STRATEGY_CREATION_NOTICE_KEY] = (
            "Aprueba primero las reglas conversadas para habilitar la generación."
        )
        return
    if not generate_code and phase in (
        STRATEGY_CREATION_PHASE_PROPOSAL_PENDING,
        STRATEGY_CREATION_PHASE_CODE_AUTHORIZED,
        STRATEGY_CREATION_PHASE_CODE_PENDING_VALIDATION,
    ):
        if phase == STRATEGY_CREATION_PHASE_CODE_PENDING_VALIDATION:
            st.session_state[STRATEGY_CREATION_CODE_STALE_KEY] = True
        phase = STRATEGY_CREATION_PHASE_DESIGN
        st.session_state[STRATEGY_CREATION_PHASE_KEY] = phase
        rules_state = dict(st.session_state.get(STRATEGY_CREATION_RULES_KEY, {}))
        rules_state["approved"] = False
        rules_state["approved_text"] = None
        if not consolidate_rules:
            rules_state["proposal"] = None
        st.session_state[STRATEGY_CREATION_RULES_KEY] = rules_state

    history = list(st.session_state.get(STRATEGY_CREATION_MESSAGES_KEY, []))
    history.append({"role": "user", "content": message})
    st.session_state[STRATEGY_CREATION_MESSAGES_KEY] = history
    st.session_state[STRATEGY_CREATION_ERROR_KEY] = None
    st.session_state[STRATEGY_CREATION_NOTICE_KEY] = None
    st.session_state[STRATEGY_CREATION_PROCESSING_KEY] = True

    try:
        prompt = build_strategy_creation_prompt(history)
        if generate_code:
            prompt += (
                "\n\nEspecificación exacta aprobada por el usuario; genera únicamente a partir "
                "de esta versión:\n"
                f"{approved_proposal}"
            )
        response = run_ai_request(
            prompt,
            ai_config_from_state(TOPBAR_AI_PREFIX),
            STRATEGY_CREATION_CODE_SYSTEM_PROMPT if generate_code else
            STRATEGY_CREATION_CONSOLIDATION_PROMPT if consolidate_rules else
            STRATEGY_CREATION_SYSTEM_PROMPT,
        )
        response = str(response).strip()
        if not response:
            raise AIError("La IA devolvió una respuesta vacía. Puedes intentarlo de nuevo.")
    except AIError as exc:
        st.session_state[STRATEGY_CREATION_ERROR_KEY] = str(exc)
    except Exception as exc:
        st.session_state[STRATEGY_CREATION_ERROR_KEY] = (
            f"No se pudo consultar el asistente: {type(exc).__name__}: {exc}"
        )
    else:
        history.append({"role": "assistant", "content": response})
        st.session_state[STRATEGY_CREATION_MESSAGES_KEY] = history
        if consolidate_rules:
            rules_state = dict(st.session_state.get(STRATEGY_CREATION_RULES_KEY, {}))
            rules_state["proposal"] = response
            rules_state["approved"] = False
            rules_state["approved_text"] = None
            st.session_state[STRATEGY_CREATION_RULES_KEY] = rules_state
            st.session_state[STRATEGY_CREATION_PHASE_KEY] = (
                STRATEGY_CREATION_PHASE_PROPOSAL_PENDING
            )
            st.session_state[STRATEGY_CREATION_PROPOSAL_INPUT_KEY] = response
        if strategy_creation_response_is_code(response):
            if generate_code and phase == STRATEGY_CREATION_PHASE_CODE_AUTHORIZED:
                code = extract_strategy_python_file(response)
                if code is None:
                    code = response.strip()
                filename = suggested_strategy_filename(
                    response,
                    next(
                        (
                            str(item.get("content", ""))
                            for item in history
                            if item.get("role") == "user"
                        ),
                        "",
                    ),
                )
                validation = validate_strategy_source(code, filename)
                st.session_state[STRATEGY_CREATION_CODE_KEY] = code
                st.session_state[STRATEGY_CREATION_CODE_EDITOR_KEY] = code
                st.session_state[STRATEGY_CREATION_FILENAME_KEY] = filename
                st.session_state[STRATEGY_CREATION_VALIDATION_KEY] = validation
                st.session_state[STRATEGY_CREATION_SAVED_PATH_KEY] = None
                st.session_state[STRATEGY_CREATION_SAVED_CODE_KEY] = None
                st.session_state[STRATEGY_CREATION_CODE_STALE_KEY] = False
                st.session_state[STRATEGY_CREATION_PHASE_KEY] = (
                    STRATEGY_CREATION_PHASE_CODE_PENDING_VALIDATION
                )
                if validation["valid"]:
                    if not st.session_state.get(PRINCIPAL_KEY):
                        st.session_state["codigo_actual"] = code
                        st.session_state["codigo_actual_script"] = ""
                        st.session_state["editor_loaded_for"] = None
                    st.session_state[STRATEGY_CREATION_NOTICE_KEY] = (
                        "Código conservado. La validación estática del contrato del proyecto "
                        "pasó; esto no demuestra su comportamiento ni rentabilidad. Aún no se "
                        "guardó ni ejecutó."
                    )
                else:
                    st.session_state[STRATEGY_CREATION_NOTICE_KEY] = (
                        "La respuesta se conservó, pero no se reconoce como una estrategia "
                        "completa compatible. Revisa los errores de validación antes de guardarla."
                    )
            else:
                st.session_state[STRATEGY_CREATION_NOTICE_KEY] = (
                    "La respuesta incluye código y se conservó en el historial. La generación "
                    "solo queda autorizada después de aprobar explícitamente las reglas."
                )
        elif generate_code and phase == STRATEGY_CREATION_PHASE_CODE_AUTHORIZED:
            st.session_state[STRATEGY_CREATION_NOTICE_KEY] = (
                "La respuesta se conservó en el chat, pero no contiene un archivo de estrategia "
                "completo; no se presenta como una generación finalizada. Puedes volver a solicitarla."
            )
    finally:
        st.session_state[STRATEGY_CREATION_PROCESSING_KEY] = False


def ai_credential_ready(config: dict) -> bool:
    """Indica si hay credencial resoluble (Ollama no necesita ninguna)."""
    if not config.get("provider", "").startswith("Gemini"):
        return True
    return bool((config.get("api_key") or "").strip() or get_gemini_api_key())


AI_PROVIDERS = ["Gemini (Google)", "Ollama (local)"]


def render_ai_provider_form(prefix: str, on_change=None) -> dict:
    """Selector de proveedor + modelo + credencial, sin decides la maquetación.

    The global assistant and other consumers reuse it so model options stay consistent.
    """
    provider = st.selectbox(
        "Proveedor de IA:",
        AI_PROVIDERS,
        key=f"{prefix}_provider",
        on_change=on_change,
    )

    if provider.startswith("Gemini"):
        model = st.selectbox(
            "Modelo:", GEMINI_MODELS, key=f"{prefix}_model", on_change=on_change
        )
    else:
        model = st.selectbox(
            "Modelo:", OLLAMA_MODELS, key=f"{prefix}_model_local", on_change=on_change
        )

    api_key, host = "", ""
    if provider.startswith("Gemini"):
        if not get_gemini_api_key():
            api_key = st.text_input(
                "API key de Gemini:",
                type="password",
                key=f"{prefix}_api_key",
                help="Se mantiene solo en esta sesión. Mejor define GEMINI_API_KEY en el entorno.",
                on_change=on_change,
            )
    else:
        st.session_state.setdefault(
            f"{prefix}_ollama_host", OLLAMA_DEFAULT_HOST
        )
        host = st.text_input(
            "Host de Ollama:",
            key=f"{prefix}_ollama_host",
            on_change=on_change,
        )

    return {"provider": provider, "model": model, "api_key": api_key, "host": host}


def sync_ai_config_state(source_prefix: str, target_prefix: str) -> None:
    """Copy provider settings between UI keys while retaining one canonical config."""
    for suffix in ("provider", "model", "model_local", "api_key", "ollama_host"):
        source_key = f"{source_prefix}_{suffix}"
        target_key = f"{target_prefix}_{suffix}"
        if source_key in st.session_state:
            st.session_state[target_key] = st.session_state[source_key]


def sync_assistant_ai_config_to_canonical() -> None:
    """Persist assistant-panel widget changes into the canonical `hdr` state."""
    sync_ai_config_state(ASSISTANT_AI_PREFIX, TOPBAR_AI_PREFIX)


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


REVIEW_GENERAL_NOTE = (
    "Promedio ponderado de la calidad estructural; no mide rentabilidad ni "
    "ejecuta backtests."
)


def _review_score_level(value) -> str:
    """Nivel de semáforo para una calificación (sin valor → gris)."""
    if value is None:
        return "none"
    if value >= 80:
        return "high"
    if value >= 50:
        return "mid"
    return "low"


def _review_info_key(label: str) -> str:
    return "strategy_review_info_" + re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _render_review_info_popover(label: str, evaluation: dict) -> None:
    with st.popover("i", width="content", key=_review_info_key(label)):
        st.markdown(escape(evaluation["reason"]))
        findings = evaluation.get("findings") or []
        if findings:
            st.markdown("\n".join(f"- {escape(item)}" for item in findings))


def _render_strategy_picker_banner(review: dict, filename: str) -> None:
    evaluations = review["evaluations"]
    general = evaluations[REVIEW_GENERAL_LABEL]
    with st.container(key="strategy-review-banner"):
        columns = st.columns(
            [1.5, 2.1, 1.0, 1.05, 0.95, 1.25],
            gap="small",
            vertical_alignment="top",
        )
        with columns[0]:
            st.markdown(
                '<div class="strategy-review-ident"><span class="ic">🐍</span>'
                f"<strong>{escape(filename)}</strong></div>",
                unsafe_allow_html=True,
            )
        with columns[1]:
            st.markdown(
                '<div class="strategy-review-general '
                f'review-{_review_score_level(general["value"])}">'
                f'<div class="general-main"><span>{escape(REVIEW_GENERAL_LABEL)}</span>'
                '<strong class="review-value"><span class="review-dot"></span>'
                f'{escape(general["display"])}</strong></div>'
                f'<div class="general-note">{escape(REVIEW_GENERAL_NOTE)}</div></div>',
                unsafe_allow_html=True,
            )
            _render_review_info_popover(REVIEW_GENERAL_LABEL, general)
        for column, label in zip(columns[2:], REVIEW_DIMENSION_LABELS):
            evaluation = evaluations[label]
            with column:
                st.markdown(
                    '<div class="strategy-review-metric '
                    f'review-{_review_score_level(evaluation["value"])}">'
                    f"<span>{escape(label)}</span>"
                    '<strong class="review-value"><span class="review-dot"></span>'
                    f'{escape(evaluation["display"])}</strong></div>',
                    unsafe_allow_html=True,
                )
                _render_review_info_popover(label, evaluation)


def _render_strategy_picker_analysis(review: dict) -> None:
    strong_html = "".join(
        f"<li>{escape(item)}</li>" for item in review["strengths"]
    )
    weak_html = "".join(
        f"<li>{escape(item)}</li>" for item in review["weaknesses"]
    )
    strengths_col, weaknesses_col = st.columns(2, gap="medium")
    with strengths_col:
        st.markdown(
            '<div class="strategy-review-list positive"><strong>Puntos fuertes</strong>'
            f"<ul>{strong_html}</ul></div>",
            unsafe_allow_html=True,
        )
    with weaknesses_col:
        st.markdown(
            '<div class="strategy-review-list caution"><strong>Puntos débiles</strong>'
            f"<ul>{weak_html}</ul></div>",
            unsafe_allow_html=True,
        )


def _render_strategy_picker_facts(review: dict) -> None:
    type_note = review["strategy_type_note"]
    if review["valid"] and review.get("dialect"):
        type_note = f"{type_note} Contrato de ejecución: {review['dialect']}."
    values = (
        ("Tipo de estrategia", review["strategy_type"], type_note),
        ("Mercado objetivo", review["market"], review["market_note"]),
        ("Horizonte", review["horizon"], review["horizon_note"]),
        (
            "Indicadores principales",
            ", ".join(review["indicators"]) if review["indicators"] else "No determinado",
            review["indicators_note"],
        ),
        ("Nivel de complejidad", review["complexity"], review["complexity_note"]),
    )
    columns = st.columns(5, gap="small")
    for column, (label, value, note) in zip(columns, values):
        with column:
            st.markdown(
                f'<div class="strategy-review-fact"><span>{escape(label)}</span>'
                f"<strong>{escape(value)}</strong></div>",
                unsafe_allow_html=True,
            )
            st.caption(note)


def _take_strategy_picker_event(event: dict | None, state_key: str) -> dict | None:
    if not isinstance(event, dict) or not event.get("event_id"):
        return None
    if st.session_state.get(state_key) == event["event_id"]:
        return None
    st.session_state[state_key] = event["event_id"]
    return event


@st.dialog("Abrir Estrategia desde QuanConnect", width="large")
def modal_abrir_estrategia():
    st.markdown(
        """
<style>
[data-testid="stDialog"]:has(.st-key-strategy-picker-scroll-area) {
    display:flex; align-items:center; justify-content:center;
    box-sizing:border-box; padding:2vh 2vw;
}
[data-testid="stDialog"]:has(.st-key-strategy-picker-scroll-area) [role="dialog"],
[data-testid="stDialog"]:has(.st-key-strategy-picker-scroll-area)[role="dialog"] {
    box-sizing:border-box; max-height:90vh; overflow:hidden;
    display:flex; flex-direction:column; margin:auto;
}
[data-testid="stDialog"]:has(.st-key-strategy-picker-scroll-area) [role="dialog"] > div:has(.st-key-strategy-picker-footer) {
    box-sizing:border-box; display:flex; flex-direction:column;
    flex:1 1 auto; min-height:0; overflow:hidden;
}
[data-testid="stDialog"]:has(.st-key-strategy-picker-scroll-area) [data-testid="stVerticalBlock"]:has(.st-key-strategy-picker-footer) {
    display:flex; flex-direction:column; flex:1 1 auto; min-height:0;
}
[data-testid="stDialog"]:has(.st-key-strategy-picker-scroll-area) [data-testid="stLayoutWrapper"]:has(> .st-key-strategy-picker-scroll-area) {
    display:flex; flex-direction:column; flex:1 1 auto; min-height:0;
}
[data-testid="stDialog"]:has(.st-key-strategy-picker-scroll-area) [data-testid="stLayoutWrapper"]:has(> .st-key-strategy-picker-footer) {
    display:flex; flex-direction:column; flex:0 0 auto; min-height:0;
}
.st-key-strategy-picker-scroll-area {
    box-sizing:border-box; min-height:0; max-height:calc(90vh - 205px);
    overflow-y:auto; overscroll-behavior:contain; scrollbar-gutter:stable;
    flex:1 1 auto;
}
.st-key-strategy-picker-footer { flex:0 0 auto; min-height:0; }
.strategy-review-metric, .strategy-review-fact {
    box-sizing:border-box; background:#f7f9fc; border:1px solid #dce4ee;
    border-radius:12px; padding:12px; display:flex; flex-direction:column; gap:6px;
    overflow:visible; }
.strategy-review-metric { min-height:84px; height:auto; }
.strategy-review-fact { min-height:102px; height:auto; }
.strategy-review-metric span, .strategy-review-fact span {
    color:#64748b; font-size:0.82rem; }
.strategy-review-metric strong, .strategy-review-fact strong { overflow-wrap:anywhere; }
.strategy-review-list { box-sizing:border-box; min-height:145px; height:auto;
    border-radius:12px; padding:14px 16px; }
.strategy-review-list ul { margin:8px 0 0; padding-left:20px; }
.strategy-review-list li { margin:4px 0; }
.strategy-review-list.positive { background:#edf8f0; border:1px solid #ccebd5; }
.strategy-review-list.caution { background:#fff4f1; border:1px solid #f2d6cf; }
.strategy-review-detected { color:#137a48; font-weight:600; }
.st-key-strategy-review-banner { margin:4px 0 10px; }
.st-key-strategy-review-banner [data-testid="stColumn"] { position:relative; }
.st-key-strategy-review-banner [data-testid="stPopover"] {
    position:absolute; top:8px; right:10px; z-index:6; }
.st-key-strategy-review-banner [data-testid="stPopoverButton"] {
    width:20px; height:20px; min-width:20px; min-height:20px; padding:0; gap:0;
    display:flex; align-items:center; justify-content:center;
    border-radius:50%; border:1px solid #cbd5e1; background:#eef2f7;
    color:#475569; font-size:11px; font-weight:700; line-height:1; box-shadow:none; }
.st-key-strategy-review-banner [data-testid="stPopoverButton"] div[aria-hidden="true"] {
    display:none; }
.st-key-strategy-review-banner [data-testid="stPopoverButton"] div:has(> div[aria-hidden="true"]) {
    margin:0; padding:0; gap:0; max-width:100%; }
.st-key-strategy-review-banner [data-testid="stPopoverButton"] p {
    margin:0; padding:0; font-size:11px; font-weight:700; line-height:1;
    color:inherit; }
.st-key-strategy-review-banner [data-testid="stPopoverButton"]:hover,
.st-key-strategy-review-banner [data-testid="stPopoverButton"]:focus-visible {
    background:#e2e8f0; border-color:#94a3b8; color:#334155; }
.st-key-strategy-review-banner [data-testid="stPopoverButton"]:focus-visible {
    outline:none; box-shadow:0 0 0 2px #ffffff, 0 0 0 4px #93c5fd; }
.strategy-review-ident, .strategy-review-general {
    box-sizing:border-box; background:#f7f9fc; border:1px solid #dce4ee;
    border-radius:12px; padding:12px; overflow:visible; }
.strategy-review-ident { min-height:84px; display:flex; flex-direction:column;
    justify-content:center; gap:6px; }
.strategy-review-ident strong { font-size:1.02rem; font-weight:700;
    color:#0f172a; overflow-wrap:anywhere; }
.strategy-review-general { min-height:84px; padding-right:34px; display:flex;
    gap:12px; align-items:center; }
.strategy-review-general .general-main { display:flex; flex-direction:column;
    gap:6px; flex:0 0 auto; }
.strategy-review-general .general-main span { color:#64748b; font-size:0.82rem; }
.strategy-review-general .general-main strong { font-size:1.4rem; }
.strategy-review-general .general-note { flex:1 1 auto; color:#475569;
    font-size:0.8rem; line-height:1.35; }
.st-key-strategy-review-banner .strategy-review-metric { padding-right:30px; }
.review-dot { display:inline-block; width:8px; height:8px; border-radius:50%;
    margin-right:6px; vertical-align:middle; background:#cbd5e1; }
.review-high .review-value { color:#15803d; }
.review-mid .review-value { color:#b45309; }
.review-low .review-value { color:#b91c1c; }
.review-none .review-value { color:#64748b; }
.review-high .review-dot { background:#16a34a; }
.review-mid .review-dot { background:#f59e0b; }
.review-low .review-dot { background:#dc2626; }
</style>
""",
        unsafe_allow_html=True,
    )
    with st.container(key="strategy-picker-scroll-area"):
        mode = st.session_state.get(MODAL_MODO_KEY, MODO_PRINCIPAL)
        is_comparison = mode == MODO_COMPARATIVA
        view = st.session_state.get(MODAL_VIEW_KEY, "empty")
        st.caption(
            "Revisa el análisis de la estrategia antes de cargarla"
            if view == "review"
            else "Selecciona una estrategia para ver su análisis y cargarla en tu laboratorio"
        )

        if view not in {"empty", "review"}:
            view = "empty"
            st.session_state[MODAL_VIEW_KEY] = view

        if view == "empty":
            drop_event = _take_strategy_picker_event(
                _strategy_file_picker(part="dropzone", key="strategy_picker_dropzone"),
                MODAL_DROP_EVENT_KEY,
            )
            if drop_event and drop_event.get("action") == "file_selected":
                process_strategy_picker_file(drop_event)
                st.rerun(scope="fragment")
            if st.session_state.get(MODAL_ERROR_KEY):
                st.error(st.session_state[MODAL_ERROR_KEY])

        source = st.session_state.get(MODAL_SELECTED_SOURCE_KEY)
        filename = st.session_state.get(MODAL_SELECTED_NAME_KEY, "")
        relative_path = st.session_state.get(MODAL_SELECTED_PATH_KEY)
        review = None
        if view == "review" and source is not None:
            review = analyze_strategy_source(source, filename, relative_path or "")
            _render_strategy_picker_banner(review, filename)
            code_col, analysis_col = st.columns([1.12, 0.88], gap="medium")
            with code_col:
                if review["valid"]:
                    st.markdown(
                        '<span class="strategy-review-detected">Estrategia detectada</span>',
                        unsafe_allow_html=True,
                    )
                st.caption(
                    f"{len(source.splitlines())} líneas · {len(source)} caracteres"
                )
                if relative_path:
                    st.caption(f"Archivo del proyecto · `{relative_path}`")
                elif st.session_state.get(MODAL_PROVENANCE_KEY) == "external":
                    st.caption("Archivo externo seleccionado desde el equipo · original sin modificar")
                st.code(
                    source,
                    language="python",
                    line_numbers=True,
                    height=330,
                )
                st.caption(
                    "Vista desplazable; el análisis y la validación usan el archivo completo."
                )
            with analysis_col:
                _render_strategy_picker_analysis(review)
            _render_strategy_picker_facts(review)
            if st.session_state.get(MODAL_PROVENANCE_KEY) == "external":
                st.info(
                    "Archivo externo listo para revisar. Al cargarlo, trabajarás con una copia "
                    "en el editor; el archivo original no se modificará automáticamente."
                )
            if not review["valid"] and review["error"]:
                st.warning(
                    "El analizador no reconoce el contrato de estrategia requerido para "
                    f"ejecutar un backtest: {review['error']}"
                )
            if not review["syntax_valid"]:
                st.error(
                    "El archivo no se puede admitir en el editor porque contiene errores de "
                    f"sintaxis: {review['syntax_error']}"
                )
            elif not review["execution_validation"]["valid"]:
                st.caption(
                    "La carga al editor no ejecuta el archivo. Para un backtest, corrige: "
                    + " ".join(review["execution_validation"]["errors"])
                )
            if st.session_state.get(MODAL_ERROR_KEY):
                st.error(st.session_state[MODAL_ERROR_KEY])

    can_load = bool(
        view == "review"
        and review
        and review["syntax_valid"]
    )
    load_label = (
        "⚖️ Reemplazar comparativa"
        if is_comparison and st.session_state.get(COMPARE_KEY)
        else "⚖️ Agregar a comparar"
        if is_comparison
        else "Cargar en Editor"
    )
    if can_load:
        load_help = ""
    elif view == "review" and review and not review["syntax_valid"]:
        load_help = "Corrige los errores de sintaxis para admitir el archivo en el editor."
    else:
        load_help = "Selecciona un archivo Python legible."

    with st.container(key="strategy-picker-footer"):
        action_event = _take_strategy_picker_event(
            _strategy_file_picker(
                part="actions",
                can_load=can_load,
                load_label=load_label,
                load_help=load_help,
                key="strategy_picker_actions",
            ),
            MODAL_ACTION_EVENT_KEY,
        )
    if action_event and action_event.get("action") == "file_selected":
        process_strategy_picker_file(action_event)
        st.rerun(scope="fragment")
    if action_event and action_event.get("action") == "cancel":
        st.rerun()
    load_clicked = bool(action_event and action_event.get("action") == "load")

    if load_clicked and can_load and source is not None:
        if relative_path:
            try:
                current_source, current_digest = read_strategy_candidate(relative_path)
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                st.session_state[MODAL_ERROR_KEY] = (
                    f"No se pudo verificar `{relative_path}`; no se cargó: {exc}"
                )
                st.rerun(scope="fragment")
            reviewed_digest = st.session_state.get(MODAL_SELECTED_HASH_KEY)
            if current_digest != reviewed_digest or current_source != source:
                st.session_state[MODAL_ERROR_KEY] = (
                    f"`{relative_path}` cambió desde el análisis. "
                    "Vuelve a seleccionarlo para revisar la versión actual."
                )
                st.rerun(scope="fragment")
            target_script = relative_path
        else:
            target_script = create_external_strategy(filename, source)
            current_source = source
            current_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if is_comparison:
            current_comparison = (st.session_state.get(COMPARE_KEY) or [None])[0]
            if current_comparison == target_script:
                st.info("Esa estrategia ya está cargada en el panel comparativo.")
            elif current_comparison and request_editor_navigation(
                "replace_comparison",
                target_script,
                target_source=current_source,
                target_sha256=current_digest,
            ):
                st.rerun()
            else:
                set_comparative_strategy(target_script, current_source)
                st.session_state["flash_msg"] = (
                    f"⚖️ `{strategy_filename(target_script)}` cargada en el panel comparativo."
                )
                st.rerun()
        elif target_script == st.session_state.get(PRINCIPAL_KEY):
            st.info(
                f"`{strategy_display_name(target_script)}` ya es la estrategia principal; "
                "se conserva la edición actual."
            )
        elif request_editor_navigation(
            "replace",
            target_script,
            target_source=current_source,
            target_sha256=current_digest,
        ):
            st.rerun()
        else:
            set_principal_strategy(target_script, ALL_STRATEGIES, source=current_source)
            st.session_state[BACKTEST_PENDING_KEY] = target_script
            st.session_state["flash_msg"] = (
                f"`{strategy_display_name(target_script)}` cargada en el editor. "
                "El backtest no se ejecutó."
            )
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
        "(cambiable en la configuración del asistente global)"
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

        # El proveedor y el modelo se leen desde la configuración global del asistente.
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
def on_strategy_creation_proposal_changed() -> None:
    """Persist rule edits and revoke any approval tied to an earlier proposal."""
    rules_state = dict(st.session_state.get(STRATEGY_CREATION_RULES_KEY, {}))
    edited_proposal = str(
        st.session_state.get(STRATEGY_CREATION_PROPOSAL_INPUT_KEY, "")
    )
    if edited_proposal != str(rules_state.get("proposal") or ""):
        rules_state["proposal"] = edited_proposal
        rules_state["approved"] = False
        rules_state["approved_text"] = None
        st.session_state[STRATEGY_CREATION_RULES_KEY] = rules_state
        st.session_state[STRATEGY_CREATION_PHASE_KEY] = (
            STRATEGY_CREATION_PHASE_PROPOSAL_PENDING
        )


def on_strategy_creation_code_changed() -> None:
    """Keep the review editor and generated-code state synchronized and revalidate."""
    source = str(st.session_state.get(STRATEGY_CREATION_CODE_EDITOR_KEY, ""))
    filename = str(st.session_state.get(STRATEGY_CREATION_FILENAME_KEY, ""))
    st.session_state[STRATEGY_CREATION_CODE_KEY] = source
    st.session_state[STRATEGY_CREATION_VALIDATION_KEY] = validate_strategy_source(
        source, filename or "estrategia.py"
    )
    if not st.session_state.get(PRINCIPAL_KEY):
        st.session_state["codigo_actual"] = source
        st.session_state["codigo_actual_script"] = ""
        st.session_state["editor_loaded_for"] = None


def save_generated_strategy_from_state(
    filename: str, source: str, base: Path, available: list[str]
) -> str:
    """Save the reviewed candidate and update the same catalog used by the laboratory."""
    path = save_strategy_source(base, filename, source)
    relative = register_strategy_file(path, base, available)
    st.session_state[STRATEGY_CREATION_SAVED_PATH_KEY] = relative
    st.session_state[STRATEGY_CREATION_SAVED_CODE_KEY] = source
    return relative


def render_generated_strategy_review() -> None:
    """Review, statically validate, download, and safely save generated source."""
    source = st.session_state.get(STRATEGY_CREATION_CODE_KEY)
    validation = st.session_state.get(STRATEGY_CREATION_VALIDATION_KEY) or {}
    code_is_stale = bool(st.session_state.get(STRATEGY_CREATION_CODE_STALE_KEY))
    if code_is_stale:
        st.markdown("**Código anterior · reglas modificadas**")
    elif validation.get("valid"):
        st.markdown("**Código generado · contrato reconocido estáticamente**")
    else:
        st.markdown("**Respuesta de código · pendiente de corrección o validación**")

    if source is not None:
        if code_is_stale:
            st.warning(
                "La conversación continuó después de generar este código. Se conserva para "
                "revisión, pero no se puede descargar ni guardar hasta generar una versión "
                "nueva a partir de reglas aprobadas."
            )
        st.text_input(
            "Nombre propuesto del archivo",
            key=STRATEGY_CREATION_FILENAME_KEY,
            help="Debe ser un nombre de archivo `.py`, sin rutas.",
        )
        st.text_area(
            "Código Python de la estrategia",
            key=STRATEGY_CREATION_CODE_EDITOR_KEY,
            height=360,
            on_change=on_strategy_creation_code_changed,
        )
        source = str(st.session_state.get(STRATEGY_CREATION_CODE_EDITOR_KEY, ""))
        filename = str(st.session_state.get(STRATEGY_CREATION_FILENAME_KEY, ""))
        validation = validate_strategy_source(source, filename or "estrategia.py")
        st.session_state[STRATEGY_CREATION_VALIDATION_KEY] = validation
        name_is_valid = is_valid_strategy_filename(filename)

        if validation.get("valid"):
            st.success(
                f"Validación estática superada · contrato `{validation['dialect']}`. "
                "El código está listo para guardar y, después de seleccionarlo, usar el "
                "backtest existente. No se ha ejecutado ni probado su comportamiento."
            )
        else:
            st.warning("La validación estática no está superada:")
            for error in validation.get("errors", []):
                st.markdown(f"- {error}")
        if not name_is_valid:
            st.warning("Indica un nombre de archivo válido terminado en `.py`, sin rutas.")

        download_col, save_col = st.columns(2)
        with download_col:
            st.download_button(
                "Descargar archivo Python",
                data=source,
                file_name=filename if name_is_valid else "estrategia_nueva.py",
                mime="text/x-python",
                key="strategy_creation_download",
                use_container_width=True,
                disabled=code_is_stale or not (validation.get("valid") and name_is_valid),
            )
        with save_col:
            save_generated = st.button(
                "Guardar como archivo nuevo",
                key="strategy_creation_save_file",
                use_container_width=True,
                disabled=code_is_stale or not (validation.get("valid") and name_is_valid),
            )
        if save_generated:
            try:
                relative = save_generated_strategy_from_state(
                    filename, source, BASE_DIR, ALL_STRATEGIES
                )
            except FileExistsError:
                st.error(
                    f"`{filename}` ya existe en el directorio oficial de estrategias. "
                    "Elige otro nombre; no se sobrescribió."
                )
            except (OSError, ValueError) as exc:
                st.error(f"No se pudo guardar la estrategia: {exc}")
            else:
                st.session_state[STRATEGY_CREATION_NOTICE_KEY] = (
                    f"Archivo guardado y registrado: `{relative}`. No se ejecutó."
                )
                st.rerun()

    saved_path = st.session_state.get(STRATEGY_CREATION_SAVED_PATH_KEY)
    if saved_path:
        saved_file = BASE_DIR / saved_path
        if saved_file.is_file() and saved_path in ALL_STRATEGIES:
            st.success(f"Archivo guardado: `{saved_path}`.")
            st.success("Estrategia reconocida en el catálogo del laboratorio.")
            if (
                st.session_state.get(STRATEGY_CREATION_SAVED_CODE_KEY)
                != st.session_state.get(STRATEGY_CREATION_CODE_KEY)
            ):
                st.warning(
                    "El código del editor cambió después de guardar; el archivo guardado "
                    "conserva la versión anterior."
                )
            comparison_pending = False
            if saved_path in st.session_state.get(COMPARE_KEY, []):
                comparison_pending = comparison_editor_has_unresolved_changes(saved_path)
            if st.session_state.get(PRINCIPAL_KEY) == saved_path:
                st.success("Estrategia seleccionada como Principal y lista para el backtest.")
                if metrics.get("error"):
                    st.error(f"El backtest existente informó un error: {metrics['error']}")
                elif not metrics.get("standby"):
                    st.info("El laboratorio ya refleja el resultado de su motor de backtesting.")
            else:
                st.caption(
                    "Seleccionar la estrategia como Principal inicia el flujo habitual de "
                    "evaluación del laboratorio; no se envían órdenes a un broker."
                )
                if comparison_pending:
                    st.warning(
                        "Esta estrategia tiene cambios pendientes en su editor "
                        "comparativo. Guárdalos o descártalos antes de promoverla."
                    )
                if st.button(
                    "Seleccionar como Estrategia Principal",
                    key="strategy_creation_select_saved",
                    use_container_width=True,
                    disabled=comparison_pending,
                ):
                    set_principal_strategy(saved_path, ALL_STRATEGIES)
                    st.rerun()
        else:
            st.warning("El archivo guardado ya no está disponible en el catálogo actual.")


def render_strategy_creation_assistant() -> None:
    """Conversation-only strategy design view, isolated from editor and backtest state."""
    if st.button(
        "← Volver al asistente global",
        key="strategy_creation_back",
        use_container_width=True,
    ):
        st.session_state[ASSISTANT_MODE_KEY] = ASSISTANT_MODE_GLOBAL
        st.rerun()

    st.subheader("Crear estrategia desde cero")
    st.caption(
        "Diseñaremos las reglas conversando antes de generar código. "
        "Puedes empezar con una idea o pegar un prompt extenso."
    )
    st.info(
        "La generación se habilita únicamente después de proponer y aprobar las reglas. "
        "El código generado se revisa y valida estáticamente antes de descargarlo o guardarlo; "
        "generar, descargar y guardar no ejecuta la estrategia."
    )

    phase = st.session_state.get(
        STRATEGY_CREATION_PHASE_KEY, STRATEGY_CREATION_PHASE_DESIGN
    )
    rules_state = dict(st.session_state.get(STRATEGY_CREATION_RULES_KEY, {}))
    if rules_state.get("proposal") is not None and STRATEGY_CREATION_PROPOSAL_INPUT_KEY not in st.session_state:
        st.session_state[STRATEGY_CREATION_PROPOSAL_INPUT_KEY] = str(
            rules_state.get("proposal") or ""
        )

    for message in st.session_state.get(STRATEGY_CREATION_MESSAGES_KEY, []):
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        with st.chat_message(role):
            st.markdown(str(message.get("content", "")))

    approved_proposal = str(rules_state.get("proposal") or "").strip()
    rules_approval_valid = bool(
        rules_state.get("approved")
        and approved_proposal
        and rules_state.get("approved_text") == approved_proposal
        and not strategy_creation_response_is_code(approved_proposal)
    )
    if phase in (
        STRATEGY_CREATION_PHASE_PROPOSAL_PENDING,
        STRATEGY_CREATION_PHASE_CODE_AUTHORIZED,
    ) and rules_state.get("proposal") is not None:
        st.text_area(
            "Revisa y edita la especificación",
            key=STRATEGY_CREATION_PROPOSAL_INPUT_KEY,
            height=260,
            on_change=on_strategy_creation_proposal_changed,
            help="Los datos que no se conocen deben permanecer marcados como pendientes.",
        )
    if phase == STRATEGY_CREATION_PHASE_DESIGN:
        st.caption("Fase: diseño de reglas; todavía no hay generación autorizada.")
        assistant_messages = [
            message
            for message in st.session_state.get(STRATEGY_CREATION_MESSAGES_KEY, [])
            if message.get("role") == "assistant"
        ]
        latest_response = (
            str(assistant_messages[-1].get("content", "")).strip()
            if assistant_messages
            else ""
        )
        if st.button(
            "Consolidar reglas",
            key="strategy_creation_consolidate_rules",
            use_container_width=True,
            disabled=not any(
                message.get("role") == "user"
                for message in st.session_state.get(STRATEGY_CREATION_MESSAGES_KEY, [])
            ),
        ):
            with st.spinner("Consolidando reglas y marcando datos desconocidos..."):
                process_strategy_creation_message(
                    "Consolida las reglas de estrategia discutidas hasta ahora.",
                    consolidate_rules=True,
                )
            st.rerun()
        if latest_response and strategy_creation_response_is_code(latest_response):
            st.caption("Una respuesta con código se conserva, pero no constituye reglas aprobadas.")
    elif phase == STRATEGY_CREATION_PHASE_PROPOSAL_PENDING:
        rules_state = st.session_state.get(STRATEGY_CREATION_RULES_KEY, {})
        st.markdown("**Propuesta de reglas pendiente de aprobación**")
        if st.button(
            "Aprobar reglas y habilitar generación",
            key="strategy_creation_approve_rules",
            type="primary",
            use_container_width=True,
        ):
            proposed_rules = str(rules_state.get("proposal") or "").strip()
            if proposed_rules and not strategy_creation_response_is_code(proposed_rules):
                rules_state["approved"] = True
                rules_state["approved_text"] = proposed_rules
                st.session_state[STRATEGY_CREATION_RULES_KEY] = rules_state
                st.session_state[STRATEGY_CREATION_PHASE_KEY] = (
                    STRATEGY_CREATION_PHASE_CODE_AUTHORIZED
                )
                st.session_state[STRATEGY_CREATION_NOTICE_KEY] = (
                    "Reglas aprobadas. La generación sigue siendo una acción separada y explícita."
                )
            else:
                st.session_state[STRATEGY_CREATION_NOTICE_KEY] = (
                    "No hay una propuesta de reglas que aprobar."
                )
            st.rerun()
        if st.button(
            "Descartar propuesta y continuar el diseño",
            key="strategy_creation_discard_proposal",
            use_container_width=True,
        ):
            rules_state = dict(st.session_state.get(STRATEGY_CREATION_RULES_KEY, {}))
            rules_state["proposal"] = None
            rules_state["approved"] = False
            rules_state["approved_text"] = None
            st.session_state[STRATEGY_CREATION_RULES_KEY] = rules_state
            st.session_state[STRATEGY_CREATION_PHASE_KEY] = STRATEGY_CREATION_PHASE_DESIGN
            st.rerun()
    elif phase == STRATEGY_CREATION_PHASE_CODE_AUTHORIZED and rules_approval_valid:
        st.success("Fase: reglas aprobadas; generación de código autorizada.")
        if st.button(
            "Generar código de la estrategia",
            key="strategy_creation_generate_code",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Generando código para simulación y backtesting…"):
                process_strategy_creation_message(
                    "Genera el código Python completo de la estrategia basada únicamente "
                    "en las reglas aprobadas. Propón un nombre de archivo descriptivo.",
                    generate_code=True,
                )
            st.rerun()
        if st.button(
            "Revocar aprobación y volver al diseño",
            key="strategy_creation_revoke_approval",
            use_container_width=True,
        ):
            rules_state = dict(st.session_state.get(STRATEGY_CREATION_RULES_KEY, {}))
            rules_state["approved"] = False
            rules_state["proposal"] = None
            rules_state["approved_text"] = None
            st.session_state[STRATEGY_CREATION_RULES_KEY] = rules_state
            st.session_state[STRATEGY_CREATION_PHASE_KEY] = STRATEGY_CREATION_PHASE_DESIGN
            st.rerun()
    elif phase == STRATEGY_CREATION_PHASE_CODE_AUTHORIZED:
        st.warning(
            "La fase indica autorización, pero no hay una propuesta de reglas aprobada válida. "
            "Vuelve a proponer y aprobar reglas antes de generar código."
        )
    else:
        st.info("Fase: código pendiente de validación.")
        render_generated_strategy_review()

    if (
        phase != STRATEGY_CREATION_PHASE_CODE_PENDING_VALIDATION
        and st.session_state.get(STRATEGY_CREATION_CODE_STALE_KEY)
        and st.session_state.get(STRATEGY_CREATION_CODE_KEY) is not None
    ):
        render_generated_strategy_review()

    error = st.session_state.get(STRATEGY_CREATION_ERROR_KEY)
    if error:
        st.error(error)
    notice = st.session_state.get(STRATEGY_CREATION_NOTICE_KEY)
    if notice:
        st.info(notice)

    with st.form("strategy_creation_form", clear_on_submit=True):
        user_message = st.text_area(
            "Instrucciones o prompt",
            key=STRATEGY_CREATION_INPUT_KEY,
            height=120,
            placeholder="Describe tu idea; la IA te ayudará a precisar las reglas.",
        )
        send_message = st.form_submit_button(
            "Enviar mensaje",
            key="strategy_creation_send",
            type="primary",
            use_container_width=True,
        )
    if send_message:
        if not user_message.strip():
            st.warning("Escribe una idea o una pregunta antes de enviar.")
        else:
            with st.spinner("Conversando sobre las reglas de la estrategia…"):
                process_strategy_creation_message(user_message)
            st.rerun()


def render_principal_panel(principal: str | None, pending: dict) -> None:
    """Muestra la estrategia principal en una fila compacta y adaptable."""
    with st.container(key="principal-strategy-row"):
        if not principal:
            st.markdown(
                '<div class="principal-card empty">'
                '<span style="display:block; width:100%; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">Ninguna estrategia seleccionada</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            return

        tarjeta_col, quitar_col = st.columns(
            [1, 0.12], vertical_alignment="center", gap="small"
        )

        with tarjeta_col:
            st.markdown(
                f'<div class="principal-card" title="{escape(principal)}">'
                f'<span class="card-icon">??</span>'
                f'<span class="strategy-name">'
                f'{escape(Path(principal).name)}</span></div>',
                unsafe_allow_html=True,
            )

        with quitar_col:
            if st.button(
                "🗑️",
                key="btn_quitar_principal",
                help="Quitar estrategia principal",
            ):
                pending["quitar"] = True


def render_principal_prototype(principal: str | None) -> None:
    """Render a visual-only alternative for comparing the principal strategy row."""
    with st.container(key="principal-prototype"):
        name = Path(principal).name if principal else None
        value_class = "principal-prototype-value" + (" empty" if name is None else "")
        value = escape(name) if name else "Ninguna estrategia seleccionada"

        st.markdown(
            '<div class="principal-prototype-heading">'
            "PROTOTIPO — ESTRATEGIA PRINCIPAL</div>"
            '<div class="principal-prototype-row">'
            f'<div class="{value_class}" title="{escape(name or "")}">{value}</div>'
            + (
                '<button class="principal-prototype-delete" type="button" '
                'title="Quitar estrategia principal" aria-label="Quitar estrategia principal">'
                "🗑️</button>"
                if name
                else ""
            )
            + '</div>',
            unsafe_allow_html=True,
        )



def persist_comparison_editor_code(script: str, editor_key: str) -> None:
    """Keep each comparative draft isolated when its widget is unmounted."""
    if editor_key not in st.session_state:
        return
    source = str(st.session_state[editor_key])
    sources = dict(st.session_state.get(COMPARE_EDITOR_SOURCES_KEY) or {})
    sources[script] = source
    st.session_state[COMPARE_EDITOR_SOURCES_KEY] = sources
    external = external_strategy_record(script)
    if external is None:
        try:
            path = resolve_editor_strategy_file(script)
        except (OSError, ValueError):
            return
        failed_saves = st.session_state.get(EDITOR_SAVE_FAILURE_KEY) or {}
        if failed_saves.get(str(path)) is not None and failed_saves[str(path)] != source:
            failed_saves.pop(str(path), None)
            if failed_saves:
                st.session_state[EDITOR_SAVE_FAILURE_KEY] = failed_saves
            else:
                st.session_state.pop(EDITOR_SAVE_FAILURE_KEY, None)


def apply_pending_editor_undo(script: str, editor_kind: str) -> None:
    request = st.session_state.get(EDITOR_UNDO_KEY) or {}
    if request.get("script") != script or request.get("editor_kind") != editor_kind:
        return
    try:
        restore_editor_source(script, editor_kind)
    except (OSError, ValueError) as exc:
        st.session_state["editor_undo_error"] = str(exc)
    else:
        st.session_state.pop("editor_undo_error", None)
    st.session_state.pop(EDITOR_UNDO_KEY, None)


def render_comparison_editor(script: str) -> None:
    """Render the independent comparison editor, parameters, and save controls."""
    apply_pending_editor_undo(script, "comparison")
    editor_key = compare_editor_widget_key(script)
    if editor_key not in st.session_state:
        buffers = st.session_state.get(COMPARE_EDITOR_SOURCES_KEY) or {}
        st.session_state[editor_key] = str(
            buffers.get(script, read_script_source(script))
        )

    with st.container(key="comparison-editor-panel"):
        header = st.empty()
        source = st.text_area(
            "Código de la estrategia comparativa",
            key=editor_key,
            height=290,
            label_visibility="collapsed",
            on_change=persist_comparison_editor_code,
            args=(script, editor_key),
        )
        source = str(source or "")
        state, message = comparison_editor_sync_status(script)
        header.markdown(
            f'<div class="editor-header comparison"><span class="dot">●</span> '
            f'{escape(strategy_filename(script))}'
            f'<span class="editor-provenance">{editor_provenance_label(script)}</span>'
            f'<span class="editor-sync-status {state}">{escape(message)}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Parámetros de esta estrategia**")
        render_strategy_parameters(script, source, "comparison")
        if st.session_state.get("editor_undo_error"):
            st.error(st.session_state.pop("editor_undo_error"))

        save_col, undo_col, replace_col = st.columns([1.15, 0.45, 1])
        with save_col:
            save_clicked = st.button(
                "💾 Guardar Cambios",
                key=f"compare_save_{script}",
                disabled=editor_save_button_disabled(script, source),
                use_container_width=True,
            )
        with undo_col:
            undo_clicked = st.button(
                "Deshacer",
                key=f"compare_undo_{script}",
                disabled=state != "pending",
                help="Recuperar la última versión guardada de esta copia de trabajo.",
                use_container_width=True,
            )
        with replace_col:
            replace_clicked = st.button(
                "Reemplazar comparativa",
                key=f"compare_replace_{script}",
                disabled=bool(st.session_state.get(EDITOR_NAVIGATION_KEY)),
                use_container_width=True,
            )
        if state == "pending":
            st.caption("Cambios pendientes · guardar o deshacer antes de salir.")
        if save_clicked:
            try:
                save_editor_source(script, source)
            except (OSError, ValueError) as exc:
                st.error(f"No se pudieron guardar los cambios comparativos: {exc}")
            else:
                st.toast("Cambios comparativos guardados.", icon="✅")
                st.rerun()
        if undo_clicked:
            request_editor_undo(script, "comparison")
        if replace_clicked:
            reset_strategy_picker(MODO_COMPARATIVA)
            modal_abrir_estrategia()


def _ejecutar_plantilla(asset, initial_cap) -> None:
    """Ejecuta la plantilla inicial en memoria y guarda su resultado para el panel."""
    source = str(
        st.session_state.get(PLANTILLA_EDITOR_KEY)
        or st.session_state.get(PLANTILLA_CODE_KEY)
        or ""
    )
    st.session_state[PLANTILLA_CODE_KEY] = source
    if not source.strip():
        st.warning("La plantilla está vacía: no hay nada que ejecutar.")
        return
    validacion = validate_strategy_source(source, "plantilla_inicial.py")
    if not validacion["valid"]:
        st.error("La plantilla no cumple el contrato: " + " ".join(validacion["errors"]))
        return
    with st.spinner("Ejecutando la plantilla en memoria..."):
        try:
            frame, plantilla_metrics = run_source_backtest(
                source, asset, initial_cap, "Plantilla inicial"
            )
        except Exception as exc:  # noqa: BLE001 - la plantilla es código editable
            st.session_state.pop(PLANTILLA_RESULT_KEY, None)
            st.session_state.pop(PLANTILLA_PARAMS_KEY, None)
            st.error(f"No se pudo ejecutar la plantilla: {type(exc).__name__}: {exc}")
            return
    st.session_state[PLANTILLA_RESULT_KEY] = {"df": frame, "metrics": plantilla_metrics}
    st.session_state[PLANTILLA_PARAMS_KEY] = (
        asset,
        float(initial_cap),
        _sello_parametros(),
        hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )
    st.rerun()


def _guardar_plantilla_como(source: str, target_name: str) -> None:
    """Guarda la plantilla como archivo nuevo y la promueve a Estrategia Principal."""
    try:
        target = save_strategy_source(BASE_DIR, target_name, source)
    except (OSError, ValueError) as exc:
        st.error(f"No se pudo guardar `{target_name}`: {exc}")
        return
    register_strategy_file(target, BASE_DIR, ALL_STRATEGIES)
    relative = target.relative_to(BASE_DIR).as_posix()
    try:
        set_principal_strategy(relative, ALL_STRATEGIES, source=source)
    except ValueError as exc:
        st.error(str(exc))
        return
    for key in (
        PLANTILLA_CODE_KEY,
        PLANTILLA_EDITOR_KEY,
        PLANTILLA_RESULT_KEY,
        PLANTILLA_PARAMS_KEY,
        "plantilla_loaded_for",
    ):
        st.session_state.pop(key, None)
    st.toast(f"Plantilla guardada como `{target.name}`.", icon="✅")
    st.rerun()


def _reset_plantilla_parametros() -> None:
    """Devuelve los controles del borrador a los valores por defecto de la plantilla.

    Corre como `on_click` del botón «Restablecer valores»: los callbacks se ejecutan
    antes de volver a montar los widgets, que es el único momento seguro para reescribir
    claves de widgets ya instanciados. No toca ningún archivo guardado.
    """
    source = str(
        st.session_state.get(PLANTILLA_EDITOR_KEY)
        or st.session_state.get(PLANTILLA_CODE_KEY)
        or PLANTILLA_INICIAL
    )
    info = strategy_parser.analizar_fuente(source, nombre="plantilla_inicial.py")
    for parametro in info.parametros:
        spec = plantilla_parametro_spec(parametro)
        st.session_state[f"{PLANTILLA_PARAM_PREFIX}{parametro.nombre}"] = spec["default"]


def render_plantilla_variables(source: str) -> dict:
    """Barra plegable «Variables de la estrategia» del borrador inicial.

    Muestra un control por cada parámetro que el parser detecta en el código (no inventa
    variables): los de la plantilla usan su rango explícito y el resto el que sugiere el
    parser. Devuelve los valores vigentes para alimentar `PARAMETROS_ACTIVOS`, de modo
    que el siguiente `Ejecutar Backtest` los use sin tocar el motor ni archivos guardados.
    """
    info = strategy_parser.analizar_fuente(source, nombre="plantilla_inicial.py")
    parametros = list(info.parametros)
    valores: dict = {}
    resumen = " · ".join(
        f"{p.nombre}="
        f"{st.session_state.get(f'{PLANTILLA_PARAM_PREFIX}{p.nombre}', plantilla_parametro_spec(p)['default'])}"
        for p in parametros
    )
    etiqueta = "⚙️ Variables de la estrategia"
    if resumen:
        etiqueta = f"{etiqueta}  ·  {resumen}"
    with st.expander(etiqueta, expanded=False):
        if not parametros:
            st.caption(
                "El código del borrador no declara parámetros ajustables con valores por "
                "defecto. Edítalo o pega una estrategia que sí los declare."
            )
        for parametro in parametros:
            spec = plantilla_parametro_spec(parametro)
            clave = f"{PLANTILLA_PARAM_PREFIX}{parametro.nombre}"
            if clave not in st.session_state:
                st.session_state[clave] = spec["default"]
            ayuda = f"Rango {spec['min']} – {spec['max']} · valor por defecto {spec['default']}."
            if spec["max"] > spec["min"]:
                elegido = st.slider(
                    spec["label"],
                    min_value=spec["min"],
                    max_value=spec["max"],
                    step=spec["step"],
                    key=clave,
                    help=help,
                )
            else:
                elegido = st.number_input(
                    spec["label"],
                    min_value=spec["min"],
                    max_value=spec["max"],
                    step=spec["step"],
                    key=clave,
                )
            valores[parametro.nombre] = (
                int(round(float(elegido))) if parametro.es_entero else float(elegido)
            )
        st.button(
            "Restablecer valores",
            key="btn_reset_plantilla_params",
            on_click=_reset_plantilla_parametros,
            help="Vuelve a los valores por defecto de la plantilla (no modifica archivos).",
        )
    return valores or plantilla_parametros_activos()


def render_plantilla_inicial_editor(asset, initial_cap) -> None:
    """Editor de la plantilla inicial: borrador temporal que no toca ningún archivo."""
    # Refleja en el editor el código que la IA proponga para la plantilla (script vacío).
    # No se consume `codigo_actual`: el propio modal lo vuelve a leer en cada run.
    pending_ai = st.session_state.get("codigo_actual")
    if (
        st.session_state.get("codigo_actual_script") == ""
        and pending_ai is not None
        and pending_ai != st.session_state.get("plantilla_loaded_for")
    ):
        st.session_state[PLANTILLA_EDITOR_KEY] = str(pending_ai)
        st.session_state["plantilla_loaded_for"] = pending_ai
    if PLANTILLA_EDITOR_KEY not in st.session_state:
        st.session_state[PLANTILLA_EDITOR_KEY] = st.session_state.get(
            PLANTILLA_CODE_KEY, PLANTILLA_INICIAL
        )

    with st.container(key="principal-editor-panel"):
        st.markdown(
            '<div class="editor-header"><span class="dot">●</span> '
            '<span title="Borrador temporal en memoria: no se escribe en disco hasta '
            'pulsar «Guardar Como».">⭐ Plantilla inicial del Laboratorio</span>'
            '<span class="editor-provenance">Borrador temporal · sin guardar</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="lab-info-card">'
            '<div class="lab-info-title">¿Qué hace esta plantilla?</div>'
            '<div class="lab-info-body">Es una estrategia sencilla de cruce de promedios '
            "móviles (SMA). Calcula un promedio rápido y otro lento del precio de cierre, "
            "y genera señales según la relación entre ambos. Puedes cambiar sus parámetros, "
            "modificar el código o pegar una estrategia creada con ayuda de IA.</div></div>",
            unsafe_allow_html=True,
        )
        source = str(
            st.text_area(
                "Código de la plantilla inicial",
                key=PLANTILLA_EDITOR_KEY,
                height=290,
                label_visibility="collapsed",
            )
            or ""
        )
        st.session_state[PLANTILLA_CODE_KEY] = source

        new_filename = st.text_input(
            "Nombre del archivo nuevo (.py)",
            value=st.session_state.get("new_filename_plantilla", "mi_estrategia.py"),
            key="new_filename_plantilla",
            help="Se usará al pulsar «Guardar Como…».",
        )
        target_name = (
            new_filename if new_filename.endswith(".py") else f"{new_filename}.py"
        )
        validacion_guardar = validate_strategy_source(
            source,
            target_name if is_valid_strategy_filename(target_name) else "estrategia.py",
        )
        save_col, run_col, ai_col = st.columns([1, 1, 1])
        with save_col:
            save_as_clicked = st.button(
                "Guardar Como…",
                key="btn_save_as_plantilla",
                disabled=not (
                    validacion_guardar["valid"]
                    and is_valid_strategy_filename(target_name)
                ),
                use_container_width=True,
            )
        with run_col:
            run_clicked = st.button(
                "▶ Ejecutar Backtest",
                type="primary",
                key="btn_run_plantilla",
                use_container_width=True,
            )
        with ai_col:
            ai_clicked = st.button(
                "🪄 Optimizar con IA",
                key="btn_ai_plantilla",
                use_container_width=True,
            )
        if not validacion_guardar["valid"]:
            st.caption("Guardar Como: " + " ".join(validacion_guardar["errors"]))

        valores_variables = render_plantilla_variables(source)
        PARAMETROS_ACTIVOS.clear()
        PARAMETROS_ACTIVOS.update(valores_variables)

        if run_clicked:
            _ejecutar_plantilla(asset, initial_cap)
        if ai_clicked:
            st.session_state["codigo_actual"] = source
            st.session_state["codigo_actual_script"] = ""
            modal_ia_optimizar()
        if save_as_clicked:
            _guardar_plantilla_como(source, target_name)


def render_principal_editor(script: str, asset, initial_cap) -> None:
    if not script:
        render_plantilla_inicial_editor(asset, initial_cap)
        PARAMETROS_ACTIVOS.clear()
        return

    apply_pending_editor_undo(script, "principal")
    editor_key = f"editor_{script}"
    code_content = read_script_source(script)
    if (
        editor_key not in st.session_state
        or st.session_state.get("editor_loaded_for") != script
    ):
        if st.session_state.get("codigo_actual_script") == script and "codigo_actual" in st.session_state:
            st.session_state[editor_key] = str(st.session_state["codigo_actual"])
        else:
            st.session_state[editor_key] = code_content
        st.session_state["editor_loaded_for"] = script

    with st.container(key="principal-editor-panel"):
        header = st.empty()
        source = st.text_area(
            "Código de la estrategia principal",
            key=editor_key,
            height=290,
            label_visibility="collapsed",
            on_change=persist_editor_code,
            args=(script, editor_key),
        )
        source = str(source or "")
        state, message = editor_sync_status(script, source)
        header.markdown(
            f'<div class="editor-header"><span class="dot">●</span> '
            f'⭐ {escape(strategy_filename(script))}'
            f'<span class="editor-provenance">{editor_provenance_label(script)}</span>'
            f'<span class="editor-sync-status {state}">{escape(message)}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Parámetros de esta estrategia**")
        render_strategy_parameters(script, source, "principal")
        if st.session_state.get("editor_undo_error"):
            st.error(st.session_state.pop("editor_undo_error"))

        save_col, undo_col, run_col, ai_col = st.columns([1.15, 0.45, 1, 1])
        with save_col:
            save_clicked = st.button(
                "💾 Guardar Cambios",
                key="btn_save",
                disabled=editor_save_button_disabled(script, source),
                use_container_width=True,
            )
        with undo_col:
            undo_clicked = st.button(
                "Deshacer",
                key="btn_undo_main",
                disabled=state != "pending",
                help="Recuperar la última versión guardada de esta copia de trabajo.",
                use_container_width=True,
            )
        with run_col:
            run_clicked = st.button(
                "▶ Ejecutar Backtest",
                type="primary",
                key="btn_run",
                use_container_width=True,
            )
        with ai_col:
            ai_clicked = st.button("🪄 Optimizar con IA", key="btn_ai", use_container_width=True)
        save_as_col, name_col = st.columns([1, 2])
        with name_col:
            save_as_default = (
                f"copia_{strategy_filename(script)}"
                if script
                else "estrategia_nueva.py"
            )
            new_filename = st.text_input(
                "Guardar una copia como (.py)",
                value=save_as_default,
                key="new_filename",
            )
        target_name = new_filename if new_filename.endswith(".py") else f"{new_filename}.py"
        save_as_validation = validate_strategy_source(
            source,
            target_name if is_valid_strategy_filename(target_name) else "estrategia.py",
        )
        with save_as_col:
            save_as_clicked = st.button(
                "Guardar Como",
                key="btn_save_as",
                disabled=not (
                    save_as_validation["valid"]
                    and is_valid_strategy_filename(target_name)
                ),
                use_container_width=True,
            )
        if not save_as_validation["valid"]:
            st.caption("Guardar Como: " + " ".join(save_as_validation["errors"]))

        if save_clicked:
            try:
                save_editor_source(script, source)
            except (OSError, ValueError) as exc:
                st.error(f"No se pudieron guardar los cambios: {exc}")
            else:
                st.toast("Cambios guardados en la copia de trabajo.", icon="✅")
                st.rerun()
        if undo_clicked:
            request_editor_undo(script, "principal")
        if run_clicked:
            st.session_state["force_run"] = True
            st.rerun()
        if ai_clicked:
            st.session_state["codigo_actual"] = source
            st.session_state["codigo_actual_script"] = script
            modal_ia_optimizar()
        if save_as_clicked:
            try:
                target = save_strategy_source(BASE_DIR, target_name, source)
            except (OSError, ValueError) as exc:
                st.error(f"No se pudo guardar `{target_name}`: {exc}")
            else:
                register_strategy_file(target, BASE_DIR, ALL_STRATEGIES)
                st.toast(f"Copia guardada como `{target.name}`.", icon="✅")
                st.rerun()
        if state == "pending":
            st.caption("Cambios pendientes · el último guardado se conserva hasta confirmar.")


def render_editor_panels(script: str, asset, initial_cap) -> None:
    """Show one editor alone, or two side by side when a comparison is loaded.

    La comparativa se lee del estado real de la sesión (`COMPARE_KEY`): si no hay
    ninguna cargada, la Principal ocupa todo el ancho central y no se reserva una
    columna vacía. Al quitarla, el estado cambia y vuelve sola a una sola columna.
    """
    PARAMETROS_ACTIVOS.clear()
    comparison = (st.session_state.get(COMPARE_KEY) or [None])[0]
    if not comparison:
        render_principal_editor(script, asset, initial_cap)
        return
    principal_col, comparison_col = st.columns([1, 1], gap="medium")
    with principal_col:
        render_principal_editor(script, asset, initial_cap)
    with comparison_col:
        render_comparison_editor(comparison)


def render_comparativas_panel(available: list[str], principal: str, pending: dict) -> None:
    """Render one comparison selector; the editor itself lives beside the principal."""
    selected = list(st.session_state.get(COMPARE_KEY, []))[:MAX_COMPARATIVES]
    st.session_state[COMPARE_KEY] = selected

    if selected:
        script = selected[0]
        nombre_col, quitar_col = st.columns(
            [0.78, 0.22], vertical_alignment="center", gap="small"
        )
        with nombre_col:
            st.markdown(
                '<div class="compare-slot"><span class="compare-index">1</span>'
                f'<span class="strategy-name">{escape(strategy_display_name(script))}</span></div>',
                unsafe_allow_html=True,
            )
        with quitar_col:
            remove_disabled = bool(st.session_state.get(EDITOR_NAVIGATION_KEY))
            if st.button(
                "✕", key="rm_comparativa_0",
                help=f"Quitar `{strategy_display_name(script)}` de la comparación",
                use_container_width=True, disabled=remove_disabled,
            ):
                if request_editor_navigation("remove_comparison", script):
                    st.rerun()
                discard_comparison_editor(script)
                st.session_state[COMPARE_KEY] = []
                st.session_state.pop(COMPARE_CACHE_KEY, None)
                st.session_state.pop(COMPARE_PENDING_KEY, None)
                st.rerun()
    else:
        st.caption("No hay estrategia comparativa seleccionada.")

    puede_agregar = not st.session_state.get(EDITOR_NAVIGATION_KEY)
    if st.button(
        "Reemplazar comparativa" if selected else "+ Agregar estrategia a comparar",
        use_container_width=True,
        key="btn_add_comparativa",
        disabled=not puede_agregar,
        help="Elige una estrategia del proyecto o un archivo externo.",
    ):
        reset_strategy_picker(MODO_COMPARATIVA)
        modal_abrir_estrategia()


def aplicar_gestor_pendientes(available: list[str], principal, pending: dict):
    """Aplica los clics de ambos grupos y devuelve la Estrategia Principal vigente.

    Las altas de comparativa llegan ya aplicadas por `add_comparative_strategy` (que es
    quien valida el límite y la exclusión de la Principal); por aquí solo pasan las
    bajas, que llegan en `pending["checks"]` como `False`.
    """
    pending_star = pending["star"]
    pending_checks = pending["checks"]

    if pending_star is not None:
        if pending_star in st.session_state.get(COMPARE_KEY, []):
            if comparison_editor_has_unresolved_changes(pending_star):
                st.warning(
                    f"Guarda o descarta los cambios de `{pending_star}` en su editor "
                    "comparativo antes de promoverla."
                )
                return st.session_state[PRINCIPAL_KEY] or ""
        set_principal_strategy(pending_star, available)
        st.rerun()

    if pending.get("quitar"):
        if request_editor_navigation("close"):
            return st.session_state[PRINCIPAL_KEY] or ""
        clear_principal_strategy()
        st.rerun()

    selected = list(st.session_state.get(COMPARE_KEY, []))
    next_selected = [s for s in selected if pending_checks.get(s, True)][:MAX_COMPARATIVES]
    dropped = [s for s in selected if s not in next_selected]
    if dropped:
        for script in dropped:
            if request_editor_navigation("remove_comparison", script):
                st.rerun()
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
    # Sin Principal se entrega "" y no None: el resto de la app (editor, modal, IA) ya
    # trabaja con cadenas vacías, y `None` se reserva para el estado vacío del grupo 1.
    return st.session_state[PRINCIPAL_KEY] or ""


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


def render_strategy_parameters(
    script: str, source: str, editor_kind: str
) -> dict:
    """Render parameter controls owned by one strategy editor and return their values."""
    name = strategy_filename(script)
    info = strategy_parser.analizar_fuente(source, nombre=name)
    values = {}
    if info.error:
        st.caption(f"⚠️ {info.error}")
    elif not info.parametros:
        st.caption("Esta estrategia no declara parámetros ajustables.")
    else:
        categories = info.parametros_por_categoria()
        prefix = (
            f"param_{script}_"
            if editor_kind == "principal"
            else f"compare_param_{script}_"
        )
        for category, parameters in categories.items():
            if len(categories) > 1:
                st.caption(category)
            for parameter in parameters:
                values[parameter.clave] = control_parametro(parameter, prefix)
    if editor_kind == "principal":
        PARAMETROS_ACTIVOS.clear()
        PARAMETROS_ACTIVOS.update(values)
    else:
        all_values = dict(st.session_state.get(COMPARE_PARAMETERS_KEY) or {})
        all_values[script] = values
        st.session_state[COMPARE_PARAMETERS_KEY] = all_values
    return values


def strategy_parameter_state(script: str, editor_kind: str) -> dict:
    """Read saved widget overrides before the app resolves its strategy results."""
    if editor_kind == "comparison":
        source = editor_source_for(script, editor_kind="comparison")
        prefix = f"compare_param_{script}_"
    else:
        editor_key = f"editor_{script}"
        source = str(
            st.session_state.get(
                editor_key,
                st.session_state.get("codigo_actual", read_script_source(script))
                if st.session_state.get("codigo_actual_script") == script
                else read_script_source(script),
            )
        )
        prefix = f"param_{script}_"
    info = strategy_parser.analizar_fuente(
        source, nombre=strategy_filename(script)
    )
    values = {}
    for parameter in info.parametros:
        key = f"{prefix}{parameter.clave}"
        value = st.session_state.get(key, parameter.valor)
        values[parameter.clave] = (
            int(round(float(value)))
            if parameter.es_entero
            else float(value)
        )
    return values


def request_editor_undo(script: str, editor_kind: str) -> None:
    st.session_state[EDITOR_UNDO_KEY] = {
        "script": script,
        "editor_kind": editor_kind,
    }
    st.rerun()


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
        '<span style="font-size:16px;font-weight:700;color:#0f172a;letter-spacing:.02em">GABO trader LAB</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    active_context = st.session_state.get("workspace_tab", WORKSPACE_TABS[0])
    if active_context == "INICIO":
        with st.container(border=True):
            st.markdown("**Filtros del mercado**")
            st.caption(
                "Filtros por activo, tendencia y volatilidad disponibles en una fase futura."
            )
            st.caption("Fuente de datos: no disponible.")
    elif active_context != "LABORATORIO":
        with st.container(border=True):
            st.markdown(f"**Contexto · {active_context}**")
            st.caption(
                "Este espacio está preparado para el contexto de la pestaña. "
                "El estado y los parámetros del Laboratorio se conservan al navegar."
            )
    if active_context != "LABORATORIO":
        st.markdown(
            """
<style>
    .st-key-grp_estrategias, .st-key-grp_comparar, .st-key-grp_reglas,
    .st-key-grp_parametros, .st-key-grp_visualizacion, .st-key-grp_estado {
        display: none !important;
    }
</style>
""",
            unsafe_allow_html=True,
        )

    # --- ESTADO COMPARTIDO DEL GESTOR --------------------------------------
    pending_gestor = {"star": None, "checks": {}, "quitar": False}
    principal_script = st.session_state[PRINCIPAL_KEY]

    if (
        st.session_state.get("script_selector") not in ALL_STRATEGIES
        and external_strategy_record(st.session_state.get("script_selector", "")) is None
    ):
        st.session_state["script_selector"] = principal_script

    # --- 1. ESTRATEGIAS (COMENTADO TEMPORALMENTE) ---------------------------
    # # G.3.7: el grupo se abre VACÍO y conserva el botón compacto de apertura junto a la
    # # tarjeta de la Estrategia Principal. Ningún clic se aplica aquí: `pending_gestor`
    # # los resuelve al final del run.
    # with st.expander(SIDEBAR_GROUPS[0], expanded=True, key="grp_estrategias"):
    # # El acceso compacto reutiliza el selector, la key y el modal existentes.
    # if st.button(
    # "📂 Abrir Estrategia", key="btn_open_strategy", help="Abrir Estrategia",
    # ):
    # st.session_state[MODAL_MODO_KEY] = MODO_PRINCIPAL
    # modal_abrir_estrategia()
    #
    #
    # st.markdown(
    # '<div class="strategy-subtitle">⭐ Estrategia Principal</div>',
    # unsafe_allow_html=True,
    # )
    #
    # # PRUEBA VISUAL: mostrar solo el prototipo.
    # # La fila original se conserva para recuperarla si hace falta.
    # render_principal_panel(principal_script, pending_gestor)
    # # render_principal_prototype(principal_script)
    #
    #
    #
    # --- PRUEBA TEMPORAL: expander vac?o entre los grupos 1 y 2 -------------
    if active_context == "LABORATORIO":
        with st.expander("⭐ ESTRATEGIA PRINCIPAL", expanded=False, key="grp_prueba"):
            st.markdown(
                '<div class="strategy-subtitle">'
                ''
                '⭐ Estrategia Principal'
                '</div>',
                unsafe_allow_html=True,
            )

            with st.container(key="principal-selector-row"):
                icon_col, message_col, delete_col = st.columns(
                    [36, 1, 36],
                    vertical_alignment="center",
                    gap="small",
                )

                with icon_col:
                    if st.button(
                        "+",
                        key="btn_prueba_abrir_estrategia",
                        help="Agregar una estrategia",
                        use_container_width=True,
                    ):
                        reset_strategy_picker(MODO_PRINCIPAL)
                        modal_abrir_estrategia()

                with message_col:
                    strategy_label = (
                        strategy_filename(principal_script)
                        if principal_script
                        else "Ninguna estrategia seleccionada"
                    )
                    value_class = "principal-strategy-value"
                    if not principal_script:
                        value_class += " empty"
                    st.markdown(
                        f'<div class="{value_class}">'
                        f'<span>{escape(strategy_label)}</span></div>',
                        unsafe_allow_html=True,
                    )

                with delete_col:
                    if principal_script and st.button(
                        "🗑️",
                        key="btn_prueba_quitar_principal",
                        help="Quitar estrategia principal",
                        use_container_width=True,
                    ):
                        pending_gestor["quitar"] = True

            if st.button(
                "✨ Crear con IA",
                key="strategy_creation_activate",
                use_container_width=True,
            ):
                st.session_state[ASSISTANT_MODE_KEY] = ASSISTANT_MODE_CREATE_STRATEGY
                st.session_state[STRATEGY_CREATION_ERROR_KEY] = None

    # --- 2. ESTRATEGIAS A COMPARAR --------------------------------------
    with st.expander(SIDEBAR_GROUPS[1], expanded=False, key="grp_comparar"):
        render_comparativas_panel(ALL_STRATEGIES, principal_script, pending_gestor)

    # Los clics de los dos grupos se resuelven juntos, igual que antes.
    selected_script = aplicar_gestor_pendientes(
        ALL_STRATEGIES, principal_script, pending_gestor
    )
    st.session_state["script_nombre"] = selected_script

    # Sin Principal, `selected_script` es "" y el resto de la app sigue siendo la misma
    # con una única diferencia: las etiquetas muestran que no hay ninguna estrategia.
    sin_principal = not selected_script
    etiqueta_principal = (
        strategy_display_name(selected_script)
        if selected_script
        else SIN_ESTRATEGIA_NOMBRE
    )

    # Una sola lectura del archivo de la Principal la comparten el grupo 3 (sus reglas
    # comunes) y el grupo 4 (sus parámetros ajustables). Sin Principal no hay archivo
    # que leer: se analiza una fuente vacía, para que ambos grupos muestren su estado
    # de espera en vez de un error de lectura de `BASE_DIR`.
    if selected_script and is_external_strategy(selected_script):
        info_estrategia = strategy_parser.analizar_fuente(
            read_script_source(selected_script),
            nombre=strategy_display_name(selected_script),
        )
    elif selected_script:
        info_estrategia = strategy_parser.analizar_archivo(BASE_DIR / selected_script)
    else:
        info_estrategia = strategy_parser.analizar_fuente("", nombre=SIN_ESTRATEGIA_NOMBRE)

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
            f"⤵ Reglas de `{strategy_filename(selected_script)}` "
            f"(respaldo: {DEFAULT_TICKER} / ${DEFAULT_CAPITAL:,.0f})."
        )

    # Los parámetros editables aparecen junto a cada estrategia en sus paneles.
    with st.expander(SIDEBAR_GROUPS[3], expanded=False, key="grp_parametros"):
        st.caption("Los controles de parámetros están en el panel de cada estrategia.")

    # --- 5. OPCIONES DE VISUALIZACIÓN -----------------------------------
    with st.expander(SIDEBAR_GROUPS[4], expanded=False, key="grp_visualizacion"):
        show_equity = st.checkbox("Strategy Equity", value=True)
        show_benchmark = st.checkbox("Benchmark Comparison", value=True)
        show_drawdown = st.checkbox("Drawdown", value=True)

    # --- 6. ESTADO DEL BACKTEST -----------------------------------------
    with st.expander(SIDEBAR_GROUPS[5], expanded=True, key="grp_estado"):
        st.info("💡 **Backtest Status:** Likely Not Overfit\n\nParameters: 2 Determined")

if st.session_state.get(EDITOR_NAVIGATION_KEY):
    render_editor_navigation_confirmation(ALL_STRATEGIES)

PARAMETROS_ACTIVOS.clear()
if selected_script:
    PARAMETROS_ACTIVOS.update(strategy_parameter_state(selected_script, "principal"))
else:
    # Sin Estrategia Principal el Laboratorio trabaja con el borrador de la plantilla
    # inicial: sus parámetros vigentes alimentan la corrida y la huella de la caché, de
    # modo que mover un control invalida el resultado anterior sin ejecutar nada.
    PARAMETROS_ACTIVOS.update(plantilla_parametros_activos())
comparison_script = (st.session_state.get(COMPARE_KEY) or [None])[0]
if comparison_script:
    comparison_parameters = dict(st.session_state.get(COMPARE_PARAMETERS_KEY) or {})
    comparison_parameters[comparison_script] = strategy_parameter_state(
        comparison_script, "comparison"
    )
    st.session_state[COMPARE_PARAMETERS_KEY] = comparison_parameters
fast_period = int(PARAMETROS_ACTIVOS.get("fast_ma", DEFAULT_FAST_MA))
slow_period = int(PARAMETROS_ACTIVOS.get("slow_ma", DEFAULT_SLOW_MA))

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
# La barra superior conserva únicamente la marca y el entorno de la aplicación.
top_left, top_right = st.columns([5, 7], vertical_alignment="center", gap="small")

with top_left:
    st.markdown(
        f"""
<div class="topbar">
    <span class="brand">GABO trader LAB</span>
</div>
""",
        unsafe_allow_html=True,
    )

with top_right:
    st.markdown(
        f"""
<div class="topbar topbar-right">
    <span class="badge">Entorno · {ENVIRONMENT}</span>
</div>
""",
        unsafe_allow_html=True,
    )

def select_workspace(tab: str) -> None:
    """Change the active view through Streamlit's normal session state."""
    st.session_state["workspace_tab"] = tab


def persist_editor_code(script: str, editor_key: str) -> None:
    """Keep editor contents outside the widget so tab changes can unmount it safely."""
    if editor_key in st.session_state:
        source = str(st.session_state[editor_key])
        st.session_state["codigo_actual"] = source
        st.session_state["codigo_actual_script"] = script
        if script:
            try:
                path = resolve_editor_strategy_file(script)
            except (OSError, ValueError):
                path = None
            failed_saves = st.session_state.get(EDITOR_SAVE_FAILURE_KEY) or {}
            if (
                path is not None
                and failed_saves.get(str(path)) is not None
                and failed_saves.get(str(path)) != source
            ):
                failed_saves.pop(str(path), None)
                if failed_saves:
                    st.session_state[EDITOR_SAVE_FAILURE_KEY] = failed_saves
                else:
                    st.session_state.pop(EDITOR_SAVE_FAILURE_KEY, None)
        if (
            not script
            and st.session_state.get(STRATEGY_CREATION_CODE_KEY) is not None
        ):
            st.session_state[STRATEGY_CREATION_CODE_KEY] = source
            st.session_state[STRATEGY_CREATION_CODE_EDITOR_KEY] = source
            st.session_state[STRATEGY_CREATION_VALIDATION_KEY] = (
                validate_strategy_source(
                    source,
                    str(st.session_state.get(STRATEGY_CREATION_FILENAME_KEY, "estrategia.py")),
                )
            )


def render_inicio_home() -> None:
    """Portada inicial sin cotizaciones ni resultados de detección simulados."""
    st.title("🏠 INICIO")
    st.caption("Exploración del mercado, detección de eventos y análisis de oportunidades.")
    st.info("Datos no disponibles · No hay una fuente de datos de mercado conectada.")

    st.subheader("Panorama del mercado")
    st.caption("Conecta una fuente de datos para consultar el panorama del mercado.")
    market_cards = st.columns(4, gap="small")
    for column, heading in zip(
        market_cards, ("Índices de mercado", "Tendencia general", "Volatilidad", "Amplitud")
    ):
        with column:
            st.markdown(
                f'<div class="workspace-card"><strong>{heading}</strong><br>'
                '<span>Sin fuente de datos conectada</span></div>',
                unsafe_allow_html=True,
            )

    st.subheader("Oportunidades candidatas")
    st.markdown(
        '<div class="workspace-empty">Aún no hay oportunidades evaluadas. '
        'La detección se incorporará en una fase posterior.</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Áreas futuras: tendencias y rupturas · retrocesos · volumen inusual · "
        "cambios de volatilidad · condiciones técnicas."
    )

    patterns_col, portfolio_col = st.columns(2, gap="medium")
    with patterns_col:
        st.subheader("Patrones y eventos")
        st.markdown(
            '<div class="workspace-empty">Todavía no hay patrones ni eventos registrados.</div>',
            unsafe_allow_html=True,
        )
        st.caption("Estados previstos para clasificar registros futuros; no hay hallazgos:")
        st.markdown(
            '<span class="pattern-status pending">Hipótesis pendiente de validar</span>'
            '<span class="pattern-status research">En investigación</span>'
            '<span class="pattern-status validated">Validado</span>'
            '<span class="pattern-status expired">Ha perdido vigencia</span>',
            unsafe_allow_html=True,
        )
    with portfolio_col:
        st.subheader("Relación con el portafolio")
        st.markdown(
            '<div class="workspace-empty">No hay un portafolio seleccionado. '
            'No se muestran posiciones, entradas ni alertas.</div>',
            unsafe_allow_html=True,
        )

    st.subheader("Acceso a otros módulos")
    actions = (
        ("Abrir en Laboratorio", "LABORATORIO", "discover_open_lab"),
        ("Investigar en Patrones", "PATRONES Y DESCUBRIMIENTO", "discover_open_patterns"),
        ("Consultar Portafolio", "PORTAFOLIO", "discover_open_portfolio"),
        ("Ver Resultados", "RESULTADOS Y ANÁLISIS", "discover_open_results"),
    )
    action_columns = st.columns(len(actions), gap="small")
    for column, (label, target, key) in zip(action_columns, actions):
        with column:
            st.button(
                label,
                key=key,
                use_container_width=True,
                on_click=select_workspace,
                args=(target,),
            )


def render_other_workspace(tab: str) -> None:
    """Pinta el estado inicial de los módulos todavía no implementados."""
    if tab == "INICIO":
        render_inicio_home()
    elif tab == "PATRONES Y DESCUBRIMIENTO":
        st.header("Patrones y descubrimiento")
        st.markdown(
            '<div class="workspace-empty">Espacio preparado para investigar, '
            'validar y descubrir patrones de mercado.</div>',
            unsafe_allow_html=True,
        )
    elif tab == "PORTAFOLIO":
        st.header(tab)
        st.caption(
            "Estructura preparada para separar cuentas y tipos de portafolio. "
            "No hay cuentas conectadas ni operaciones cargadas."
        )
        account_col, paper_col, market_col = st.columns(3, gap="medium")
        with account_col:
            st.markdown(
                '<div class="workspace-card"><strong>Varias cuentas</strong><br>'
                '<span>Espacio para organizar cuentas cuando exista su integración.</span></div>',
                unsafe_allow_html=True,
            )
        with paper_col:
            st.markdown(
                '<div class="workspace-card"><strong>Portafolios ficticios</strong><br>'
                '<span>Separados de cuentas con precios de mercado.</span></div>',
                unsafe_allow_html=True,
            )
        with market_col:
            st.markdown(
                '<div class="workspace-card"><strong>Simulación y real</strong><br>'
                '<span>Sin conexión a broker ni ejecución de órdenes.</span></div>',
                unsafe_allow_html=True,
            )
    elif tab == "RESULTADOS Y ANÁLISIS":
        st.header(tab)
        if metrics.get("standby") or metrics.get("error"):
            st.markdown(
                '<div class="workspace-empty">No hay resultados válidos para '
                'centralizar todavía. Ejecuta un backtest desde Laboratorio.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "Los resultados válidos permanecen en Laboratorio. Esta vista queda "
                "preparada para centralizar métricas, curvas, drawdown y operaciones."
            )
    elif tab == "HISTORIAL Y EXPERIMENTOS":
        st.header(tab)
        st.markdown(
            '<div class="workspace-empty">El historial persistente y el registro de '
            'experimentos todavía no están implementados.</div>',
            unsafe_allow_html=True,
        )


def render_laboratorio_inicial() -> None:
    """Pantalla inicial del Laboratorio: plantilla a la izquierda, resultados a la derecha.

    Es la vista que aparece al entrar sin Estrategia Principal ni comparativa. El banner
    de bienvenida ocupa el ancho completo; debajo, el editor y sus variables van a la
    izquierda y la información técnica, las cinco tarjetas y la gráfica a la derecha.
    Antes de ejecutar la plantilla las tarjetas muestran «—» y la gráfica un estado vacío:
    no se dibujan curvas planas ficticias ni se reutiliza el resultado de otra estrategia.
    """
    st.markdown(
        """
<div class="lab-welcome">
    <div class="lab-welcome-title">Bienvenido al Laboratorio de Estrategias</div>
    <div class="lab-welcome-desc">Aquí puedes probar, analizar y optimizar estrategias de
        trading con datos históricos. Esta es una plantilla funcional que puedes modificar.
        Ejecuta un backtest para revisar sus resultados o guárdala como un archivo nuevo.</div>
    <div class="lab-steps">
        <span class="lab-step"><b>1</b> Edita o pega el código Python</span>
        <span class="lab-step"><b>2</b> Configura el activo y los parámetros</span>
        <span class="lab-step"><b>3</b> Ejecuta el backtest y revisa los resultados</span>
        <span class="lab-step"><b>4</b> Optimiza o guarda tu estrategia</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    hay_resultado = not metrics.get("standby") and not metrics.get("error")

    if hay_resultado:
        equity_value = float(metrics["equity"])
        total_return_value = float(metrics["total_return"])
        drawdown_value = float(metrics["max_drawdown"])
        win_rate_value = float(metrics["win_rate"])
        sharpe_value = float(metrics["sharpe"])
        kpi_equity = f"${equity_value:,.2f}"
        kpi_return = f"{total_return_value:.2f}%"
        kpi_drawdown = f"{drawdown_value:.2f}%"
        kpi_win = f"{win_rate_value:.1f}%"
        kpi_sharpe = f"{sharpe_value:.2f}"
        tone_equity = "positive" if equity_value >= float(initial_cap) else "negative"
        tone_return = "positive" if total_return_value >= 0 else "negative"
        tone_drawdown = "negative" if drawdown_value < 0 else "positive"
        tone_win = "positive" if win_rate_value >= 0 else "negative"
        tone_sharpe = "positive" if sharpe_value >= 0 else "negative"
    else:
        kpi_equity = kpi_return = kpi_drawdown = kpi_win = kpi_sharpe = "—"
        tone_equity = tone_return = tone_drawdown = tone_win = tone_sharpe = ""

    left_col, right_col = st.columns([1.05, 1], gap="large")
    with left_col:
        render_plantilla_inicial_editor(asset, initial_cap)

    with right_col:
        with st.expander("Información técnica", expanded=False):
            st.caption(f"Identificador interno del motor · {ENGINE_VERSION}")
            st.caption(
                "Plantilla inicial ejecutada en memoria · contrato "
                "`run_strategy(data, fast_ma, slow_ma)`."
            )
        st.markdown(
            f"""
    <div class="kpi-container">
        <div class="kpi-card"><div class="kpi-label">Capital Final</div>
            <div class="kpi-value {tone_equity}">{kpi_equity}</div></div>
        <div class="kpi-card"><div class="kpi-label">Retorno Total</div>
            <div class="kpi-value {tone_return}">{kpi_return}</div></div>
        <div class="kpi-card"><div class="kpi-label">Max Drawdown</div>
            <div class="kpi-value {tone_drawdown}">{kpi_drawdown}</div></div>
        <div class="kpi-card"><div class="kpi-label">Win Rate</div>
            <div class="kpi-value {tone_win}">{kpi_win}</div></div>
        <div class="kpi-card"><div class="kpi-label">Sharpe Ratio</div>
            <div class="kpi-value {tone_sharpe}">{kpi_sharpe}</div></div>
    </div>
    """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="panel-title">📈 Resultados del backtest</div>',
            unsafe_allow_html=True,
        )
        if hay_resultado:
            chart_view = st.radio(
                "Escala de la curva",
                EQUITY_VIEWS,
                key=f"{CHART_VIEW_KEY}_widget",
                horizontal=True,
                label_visibility="collapsed",
                index=EQUITY_VIEWS.index(st.session_state.get(CHART_VIEW_KEY, "$")),
            ) or "$"
            st.session_state[CHART_VIEW_KEY] = chart_view
            fig = build_equity_figure(
                df_res,
                asset,
                "Plantilla inicial",
                show_equity,
                show_benchmark,
                show_drawdown,
                view=chart_view,
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
            )
        else:
            st.markdown(
                '<div class="lab-empty-chart">'
                '<div class="lab-empty-title">Aún no hay resultados</div>'
                '<div class="lab-empty-body">Ejecuta un backtest para ver la curva de '
                'capital, drawdown y demás métricas de la estrategia.</div></div>',
                unsafe_allow_html=True,
            )

    # -----------------------------------------------------------------------------
    # TERMINAL Y LOGS INFERIOR
    # -----------------------------------------------------------------------------
    st.markdown(
        '<div class="panel-title" style="margin-top:10px">💻 Cloud Terminal &amp; Logs</div>',
        unsafe_allow_html=True,
    )
    log_time = datetime.now().strftime("%H:%M:%S")
    ts = f'<span class="t-dim">{log_time}</span>'
    parametros_linea = (
        f'{ts} <span class="t-info">[INFO]</span> Parameters &rarr; Fast SMA: '
        f'<span class="t-key">{fast_period}</span> | Slow SMA: '
        f'<span class="t-key">{slow_period}</span> | Capital: '
        f'<span class="t-key">${float(initial_cap):,.2f}</span><br>'
    )
    if hay_resultado:
        cierre = (
            f'{ts} <span class="t-info">[INFO]</span> Simulation resolved '
            f'<span class="t-key">{metrics["trades"]}</span> round-trip trades over '
            f'{metrics["bars"]} bars.<br>'
            f'{ts} <span class="t-ok">[STATUS]</span> Final Equity: '
            f'<span class="t-key">${equity_value:,.2f}</span> ({total_return_value:.2f}%) '
            f'&middot; Max DD: {drawdown_value:.2f}% &middot; Win Rate: {win_rate_value:.1f}%<br>'
            f'{ts} <span class="t-ok">[READY]</span> Cloud node idle. Awaiting next deployment...'
        )
    else:
        cierre = (
            f'{ts} <span class="t-warn">[WAIT]</span> No results yet — press «▶ Ejecutar '
            'Backtest» to run the template in memory.<br>'
            f'{ts} <span class="t-ok">[READY]</span> Cloud node idle. Awaiting deployment...'
        )
    st.markdown(
        f"""
    <div class="terminal-box">
        {ts} <span class="t-info">[INFO]</span> Inicializando runtime de GABO · motor {ENGINE_VERSION}...<br>
        {ts} <span class="t-info">[INFO]</span> Loaded initial template: <span class="t-key">Plantilla inicial del Laboratorio</span><br>
        {parametros_linea}<br>
        {cierre}
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_laboratorio_center() -> None:

    flash_msg = st.session_state.pop("flash_msg", None)
    if flash_msg:
        st.toast(flash_msg, icon="✅")

    # Fase 1.1 · Sin Estrategia Principal ni comparativa, el Laboratorio muestra su
    # pantalla inicial: plantilla a la izquierda y métricas/gráfica a la derecha. Existe
    # una comparativa cargada (aunque no haya Principal) se conserva la vista de siempre.
    comparison_script = (st.session_state.get(COMPARE_KEY) or [None])[0]
    if not selected_script and not comparison_script:
        render_laboratorio_inicial()
        return

    render_editor_panels(selected_script, asset, initial_cap)

    if st.session_state.get(BACKTEST_PENDING_KEY) == selected_script:
        st.info(
            "La estrategia está cargada, pero el backtest no se ejecutó. "
            "Pulsa «Ejecutar Backtest» cuando quieras evaluarla."
        )

    if metrics.get("standby"):
        backtest_status, status_class = "Backtest en espera", "waiting"
    elif metrics.get("error"):
        backtest_status, status_class = "Backtest con error", "error"
    else:
        backtest_status, status_class = "Backtest disponible", "ready"
    st.markdown(
        f"""
<div class="lab-context">
    <span class="badge">Estrategia · {escape(etiqueta_principal)}</span>
    <span class="badge">Activo de trabajo · {escape(str(asset))}</span>
    <span class="badge status {status_class}">{backtest_status}</span>
</div>
""",
        unsafe_allow_html=True,
    )
    with st.expander("Información técnica", expanded=False):
        st.caption(f"Identificador interno del motor · {ENGINE_VERSION}")

    # -----------------------------------------------------------------------------
    # 2. BARRA SUPERIOR DE METRICAS (KPIs)
    # -----------------------------------------------------------------------------
    equity_value = auto("equity")
    total_return_value = auto("total_return")
    drawdown_value = auto("max_drawdown")
    win_rate_value = auto("win_rate")
    sharpe_value = auto("sharpe")
    standby = bool(metrics.get("standby"))
    # Estado vacío de Fase 1: sin Estrategia Principal y sin ejecutar todavía. Con una
    # Principal cargada el laboratorio conserva su comportamiento de siempre (aunque el
    # backtest esté pendiente de confirmación).
    plantilla_vacia = (not selected_script) and standby

    if plantilla_vacia:
        kpi_equity = kpi_return = kpi_drawdown = kpi_win = kpi_sharpe = "—"
        tone_equity = tone_return = tone_drawdown = tone_win = tone_sharpe = ""
    else:
        kpi_equity = f"${equity_value:,.2f}"
        kpi_return = f"{total_return_value:.2f}%"
        kpi_drawdown = f"{drawdown_value:.2f}%"
        kpi_win = f"{win_rate_value:.1f}%"
        kpi_sharpe = f"{sharpe_value:.2f}"
        tone_equity = "positive" if equity_value >= initial_cap else "negative"
        tone_return = "positive" if total_return_value >= 0 else "negative"
        tone_drawdown = "negative" if drawdown_value < 0 else "positive"
        tone_win = "positive" if win_rate_value >= 0 else "negative"
        tone_sharpe = "positive" if sharpe_value >= 0 else "negative"

    st.markdown(
        f"""
    <div class="kpi-container">
        <div class="kpi-card"><div class="kpi-label">Capital Final</div>
            <div class="kpi-value {tone_equity}">{kpi_equity}</div></div>
        <div class="kpi-card"><div class="kpi-label">Retorno Total</div>
            <div class="kpi-value {tone_return}">{kpi_return}</div></div>
        <div class="kpi-card"><div class="kpi-label">Max Drawdown</div>
            <div class="kpi-value {tone_drawdown}">{kpi_drawdown}</div></div>
        <div class="kpi-card"><div class="kpi-label">Win Rate</div>
            <div class="kpi-value {tone_win}">{kpi_win}</div></div>
        <div class="kpi-card"><div class="kpi-label">Sharpe Ratio</div>
            <div class="kpi-value {tone_sharpe}">{kpi_sharpe}</div></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # --- COMPARATIVE CHART, BELOW BOTH EDITORS ---
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

    # Las comparativas se ejecutan con los parámetros guardados en su panel.
    backtest_pending = (
        st.session_state.get(BACKTEST_PENDING_KEY) == selected_script
    )
    compare_scripts = (
        []
        if backtest_pending
        else [
            s for s in st.session_state.get(COMPARE_KEY, [])
            if s != selected_script
        ]
    )
    compare_results = (
        get_comparative_results(
            compare_scripts, asset, initial_cap, force=force_run
        )
        if compare_scripts
        else {}
    )

    comparative_curves = []
    for script in compare_scripts[:MAX_COMPARATIVES]:
        payload = compare_results.get(script) or {}
        curve_df, curve_metrics = payload.get("df"), payload.get("metrics") or {}
        comparative_curves.append({
            "name": strategy_filename(script),
            "script": script,
            "equity": curve_df["Equity"] if curve_df is not None else pd.Series(dtype=float),
            "metrics": curve_metrics,
        })

    fig = build_equity_figure(
        df_res,
        asset,
        selected_script,
        show_equity and not plantilla_vacia,
        show_benchmark and not plantilla_vacia,
        show_drawdown and not plantilla_vacia,
        draft_metrics,
        draft_equity,
        comparative_curves,
        chart_view,
    )
    if plantilla_vacia and not comparative_curves:
        fig.add_annotation(
            text="Aún no hay resultados",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            font=dict(size=16, color="#94a3b8"),
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
            st.error(
                f"❌ `{strategy_display_name(item['script'])}`: "
                f"{item['metrics']['error']}"
            )
    if draft_equity is not None and draft_metrics:
        st.caption(
            f"🧪 Curva violeta punteada = borrador de IA en prueba "
            f"(Capital Final ${draft_metrics['equity']:,.2f})."
        )

    # El panel analiza exactamente las mismas métricas que muestra la barra de KPIs
    values = display_metrics()


    # -----------------------------------------------------------------------------
    # 3b. TABLA COMPARATIVA DE MÉTRICAS (Principal vs Comparativas vs Benchmark)
    # -----------------------------------------------------------------------------
    st.markdown(
        '<div class="panel-title" style="margin-top:10px">📊 Tabla Comparativa de Métricas</div>',
        unsafe_allow_html=True,
    )
    if plantilla_vacia and not comparative_curves:
        st.caption(
            "Sin resultados todavía: pulsa «▶ Ejecutar Backtest» en la plantilla para "
            "completar la tabla de métricas."
        )
    else:
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
    status_tone = "t-ok" if total_return_value >= 0 else "t-err"

    if plantilla_vacia:
        st.markdown(
            f"""
    <div class="terminal-box">
        {ts} <span class="t-info">[INFO]</span> Inicializando runtime de GABO · identificador interno del motor {ENGINE_VERSION}...<br>
        {ts} <span class="t-info">[INFO]</span> Loaded initial template: <span class="t-key">Plantilla inicial del Laboratorio</span><br>
        {ts} <span class="t-info">[INFO]</span> Parameters &rarr; Fast SMA: <span class="t-key">{fast_period}</span> | Slow SMA: <span class="t-key">{slow_period}</span> | Capital: <span class="t-key">${initial_cap:,.2f}</span><br>
        {ts} <span class="t-warn">[WAIT]</span> No results yet — press «▶ Ejecutar Backtest» to run the template in memory.<br>
        {ts} <span class="t-ok">[READY]</span> Cloud node idle. Awaiting deployment...
    </div>
    """,
            unsafe_allow_html=True,
        )
        return

    if standby:
        source_line = (
            f'{ts} <span class="t-warn">[WAIT]</span> Strategy loaded but backtest pending '
            'confirmation.'
        )
    else:
        source_line = (
            f'{ts} <span class="t-info">[INFO]</span> Fetched historical market data for '
            f'<span class="t-key">{asset}</span> ({metrics["bars"]} daily bars).'
            if live
            else f'{ts} <span class="t-warn">[WARN]</span> Market feed unreachable — rendering reference snapshot.'
        )

    st.markdown(
        f"""
    <div class="terminal-box">
        {ts} <span class="t-info">[INFO]</span> Inicializando runtime de GABO · identificador interno del motor {ENGINE_VERSION}...<br>
        {ts} <span class="t-info">[INFO]</span> Workspace mounted: <span class="t-key">C:\\QuantConnect\\projects\\quantlab</span><br>
        {ts} <span class="t-info">[INFO]</span> Loaded strategy script: <span class="t-key">{etiqueta_principal}</span><br>
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

# Navegación por selección para que solo se monte la vista activa y su contexto.
st.radio(
    "Navegación principal",
    WORKSPACE_TABS,
    key="workspace_tab",
    horizontal=True,
    label_visibility="collapsed",
    format_func=lambda tab: f"🏠 {tab}" if tab == "INICIO" else tab,
)
active_workspace_tab = st.session_state["workspace_tab"]
assistant_panel_ratio = normalize_assistant_panel_ratio(
    st.session_state["assistant_panel_ratio"]
)
st.session_state["assistant_panel_ratio"] = assistant_panel_ratio
divider_weight = 0.18
assistant_panel_weight = (6 + divider_weight) * assistant_panel_ratio / (
    1 - assistant_panel_ratio
)
with st.container(key="workspace-layout"):
    center_column, assistant_resize_column, assistant_column = st.columns(
        [6, divider_weight, assistant_panel_weight],
        gap="small",
        vertical_alignment="top",
    )

    with assistant_resize_column:
        drag_value = _panel_resizer(
            initial_ratio=assistant_panel_ratio,
            initial_width=st.session_state["assistant_panel_width_px"],
            default={"ratio": assistant_panel_ratio},
            key="assistant_panel_resizer",
        )
        if isinstance(drag_value, dict):
            requested_ratio = normalize_assistant_panel_ratio(
                drag_value.get("ratio", assistant_panel_ratio)
            )
            requested_width = drag_value.get("width_px")
            if isinstance(requested_width, (int, float)) and requested_width > 0:
                st.session_state["assistant_panel_width_px"] = float(requested_width)
            if abs(requested_ratio - assistant_panel_ratio) > 0.001:
                st.session_state["assistant_panel_ratio"] = requested_ratio
                st.rerun()

    with center_column:
        if active_workspace_tab == "LABORATORIO":
            render_laboratorio_center()
        else:
            render_other_workspace(active_workspace_tab)

    with assistant_column:
        st.markdown(
            '<div class="ai-panel-title">🤖 Asistente global</div>',
            unsafe_allow_html=True,
        )
        sync_ai_config_state(TOPBAR_AI_PREFIX, ASSISTANT_AI_PREFIX)
        with st.expander("⚙ Configurar agente y modelo de IA", expanded=False):
            render_ai_provider_form(
                ASSISTANT_AI_PREFIX,
                on_change=sync_assistant_ai_config_to_canonical,
            )
        st.caption(f"Modelo seleccionado · {ai_config_from_state(TOPBAR_AI_PREFIX)['model']}")
        assistant_mode = st.session_state.get(
            ASSISTANT_MODE_KEY, ASSISTANT_MODE_GLOBAL
        )
        if assistant_mode == ASSISTANT_MODE_CREATE_STRATEGY:
            render_strategy_creation_assistant()
        else:
            render_ai_backtest_panel(
                display_metrics(), df_res, asset, fast_period, slow_period,
                initial_cap, selected_script,
            )
            with st.container(border=True):
                st.markdown("**Acciones contextuales**")
                if active_workspace_tab == "LABORATORIO":
                    st.caption(
                        "Revisión y optimización existentes; usa el control de IA del editor."
                    )
                else:
                    future_actions = {
                        "INICIO": "Analizar panorama del mercado",
                        "PATRONES Y DESCUBRIMIENTO": "Investigar patrones y eventos",
                        "PORTAFOLIO": "Analizar posiciones y riesgo",
                        "RESULTADOS Y ANÁLISIS": "Interpretar métricas",
                        "HISTORIAL Y EXPERIMENTOS": "Explicar diferencias entre experimentos",
                    }
                    st.button(
                        future_actions[active_workspace_tab],
                        disabled=True,
                        use_container_width=True,
                        key="future_ai_action",
                    )
                    st.caption("Acción futura; todavía no ejecuta ninguna operación.")
