"""Panel A: distribución compartida del editor y los resultados del Laboratorio.

La función centraliza únicamente la composición visual de dos columnas. La lógica de
edición, ejecución del backtest, métricas y gráfica permanece en app.py para evitar
duplicar el motor o crear dependencias circulares.
"""
import streamlit as st


def render_panel_a():
    """Crear las columnas C.1 (estrategia) y C.2 (resultados) de Panel A."""
    with st.container(key="panel-a-layout"):
        return st.columns([1.05, 1], gap="large")
