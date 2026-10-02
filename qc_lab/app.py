"""
QC Lab — Dashboard Streamlit para estrategias QuantConnect LEAN

Interfaz inspirada en QuantConnect: ejecuta, parametriza, compara y analiza
backtests de estrategias `.py` (sin modificar el codigo fuente del usuario).
"""
from __future__ import annotations

import io
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qc_lab.core import analytics, datafeed, lean_runtime
from qc_lab.core import engine as engine_mod
from qc_lab.core import strategy_parser
from qc_lab.ui.theme import THEME_CSS

ROOT = Path("C:/Projects/QuanConnect")
OTRAS = ROOT / "otras"
CLAUDE = ROOT / "claude"
QC_LAB = ROOT / "qc_lab"


# --------------------------------------------------------------------------- #
# Configuracion
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="QC Lab · QuantConnect Strategy Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(THEME_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _fmt_usd(v: float) -> str:
    try:
        return f"${float(v):,.2f}"
    except Exception:
        return "$0.00"


def _fmt_pct(v: float) -> str:
    try:
        return f"{float(v) * 100:,.2f}%"
    except Exception:
        return "0.00%"


def _fmt_num(v: float, dec: int = 2) -> str:
    try:
        return f"{float(v):,.{dec}f}"
    except Exception:
        return "0.00"


def _load_example_files() -> List[Path]:
    paths: List[Path] = []
    for d in (OTRAS, CLAUDE):
        if d.exists():
            paths.extend(sorted(p for p in d.glob("*.py") if p.is_file()))
    return paths


def _find_all_strategies() -> List[Path]:
    files = _load_example_files()
    for p in ROOT.rglob("*.py"):
        if p in files:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")[:400]
        except Exception:
            continue
        if "QCAlgorithm" in txt or "from AlgorithmImports import" in txt:
            files.append(p)
    return sorted(set(files))


def _estado_color(estado: str) -> str:
    e = (estado or "").lower()
    if e in ("completed", "completado", "ok", "success"):
        return "#4ade80"
    if e in ("in_progress", "running", "ejecutando"):
        return "#3d9cff"
    if e in ("failed", "error", "fallo"):
        return "#ef4444"
    if e in ("warning", "advertencia"):
        return "#f59e0b"
    return "#b3b3b3"


def _lineas_log(re: engine_mod.ResultadoBacktest, max_n: int = 120) -> pd.DataFrame:
    logs = re.logs or []
    if not logs:
        return pd.DataFrame(columns=["Tiempo", "Nivel", "Mensaje"])
    df = pd.DataFrame(logs)
    if "tiempo" in df.columns:
        try:
            df["Tiempo"] = pd.to_datetime(df["tiempo"])
        except Exception:
            df["Tiempo"] = df["tiempo"]
    else:
        df["Tiempo"] = pd.Timestamp.utcnow()
    df["Nivel"] = df.get("nivel", "INFO")
    df["Mensaje"] = df.get("mensaje", "")
    df = df.sort_values("Tiempo", kind="stable").tail(max_n).reset_index(drop=True)
    return df[["Tiempo", "Nivel", "Mensaje"]]


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.image("https://cdn.quantconnect.com/i/tu/qc-logo.svg", width=180)
    st.markdown("## QC Lab")
    st.caption("Laboratorio de backtesting LEAN en Python")

    st.divider()
    st.subheader("Estrategias")
    estrategias = _find_all_strategies()
    if not estrategias:
        st.warning("No se encontraron archivos .py con QCAlgorithm")
        selected_path = None
    else:
        labels = [f"{p.relative_to(ROOT)}" for p in estrategias]
        sel_idx = 0
        selected_label = st.selectbox("Seleccionar estrategia", labels, index=sel_idx)
        try:
            selected_path = next(p for p, l in zip(estrategias, labels) if l == selected_label)
        except StopIteration:
            selected_path = estrategias[0]

    st.divider()
    st.subheader("Parámetros globales")
    periodo_inicio = st.date_input("Fecha inicio", value=pd.Timestamp("2025-09-30").date())
    periodo_fin = st.date_input("Fecha fin", value=pd.Timestamp("2026-09-30").date())
    capital_inicial = st.number_input("Capital inicial ($)", min_value=100, max_value=10_000_000,
                                      value=100_000, step=1000, format="%d")
    ejecutar = st.button("▶️ Ejecutar backtest", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Datos sintéticos")
    with st.expander("Calibración del datafeed", expanded=False):
        st.markdown(
            """
- OHLC deterministas (daily+minute, expansión minuto→diario exacta).
- Factor de mercado común con regímenes, idiosincrático por ticker.
- Volatilidad y correlaciones calibradas. El precio es función pura de (ticker, fecha).
            """
        )
        if st.button("Probar consistencia (rápido)"):
            try:
                dspy = datafeed.daily_bars("SPY", "2025-10-01", "2025-10-31")
                dsoxl = datafeed.daily_bars("SOXL", "2025-10-01", "2025-10-31")
                mspy = datafeed.minute_bars("SPY", "2025-10-01", "2025-10-01")
                st.success(f"SPY {len(dspy)}D / {len(mspy)}M · SOXL {len(dsoxl)}D · OK")
            except Exception as e:
                st.error(f"Error: {e}")

    st.caption("QC Lab · v0.1.0")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if selected_path is None:
    st.info("Selecciona una estrategia en la barra lateral para comenzar.")
    st.stop()

info_parse = None
try:
    info_parse = strategy_parser.analizar_archivo(str(selected_path))
except Exception as e:
    st.error(f"No se pudo analizar {selected_path.name}: {e}")
    st.stop()

st.title("QC Lab — QuantConnect Strategy Lab")
st.caption(f"Estrategia: **{info_parse.clase}** · Archivo: `{selected_path.relative_to(ROOT)}`")

# Tabs
tab_resumen, tab_curva, tab_graficos, tab_trades, tab_logs, tab_explorador, tab_ia = st.tabs(
    ["Resumen", "Curva de Equity", "Gráficos", "Trades & Ejecuciones", "Logs & Debug", "Explorador de Datos", "Asistente IA"]
)

resultado_bt: Optional[engine_mod.ResultadoBacktest] = None

if ejecutar:
    with st.status("Ejecutando backtest...", expanded=True):
        try:
            info_run = strategy_parser.analizar_archivo(str(selected_path))
            resultado_bt = engine_mod.ejecutar_backtest(
                str(selected_path),
                capital=float(capital_inicial),
                fecha_inicio=str(periodo_inicio),
                fecha_fin=str(periodo_fin),
                info=info_run,
            )
            st.session_state["ultimo_resultado"] = resultado_bt
            if resultado_bt.exito:
                st.success("Backtest finalizado correctamente.")
            else:
                st.error(f"Error: {resultado_bt.error}")
        except Exception as e:
            tb = traceback.format_exc(limit=1500)
            st.error(f"Excepción no controlada: {e}")
            st.code(tb)
else:
    if "ultimo_resultado" in st.session_state:
        resultado_bt = st.session_state["ultimo_resultado"]

if resultado_bt is None:
    st.info("Pulsa *Ejecutar backtest* para obtener resultados.")
    st.stop()

re = resultado_bt
e = re.estadisticas or {}
curva_df = re.curva.copy() if re.curva is not None and not re.curva.empty else pd.DataFrame()
fills_df = re.ejecuciones.copy() if re.ejecuciones is not None and not re.ejecuciones.empty else pd.DataFrame()
# El motor expone las operaciones cerradas en `operaciones` (no en `trades`).
trades_df = re.operaciones.copy() if re.operaciones is not None and not re.operaciones.empty else pd.DataFrame()

# --------------------------------------------------------------------------- #
# Tab Resumen
# --------------------------------------------------------------------------- #
with tab_resumen:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equity Final", _fmt_usd(e.get("equity_final", 0)), delta=_fmt_pct(e.get("retorno_total", 0)))
    c2.metric("Beneficio Neto", _fmt_usd(e.get("beneficio_neto", 0)), delta=_fmt_pct(e.get("retorno_total", 0)))
    c3.metric("Sharpe", _fmt_num(e.get("sharpe", 0), 2), delta=None)
    c4.metric("Max Drawdown", _fmt_pct(e.get("dd_max", 0)), delta=None)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Sortino", _fmt_num(e.get("sortino", 0), 2))
    c6.metric("Calmar", _fmt_num(e.get("calmar", 0), 2))
    c7.metric("Win Rate", _fmt_pct(e.get("win_rate", 0)))
    c8.metric("Profit Factor", _fmt_num(e.get("profit_factor", 0), 2))

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("CAGR", _fmt_pct(e.get("cagr", 0)))
    c10.metric("Volatilidad", _fmt_pct(e.get("volatilidad", 0)))
    c11.metric("PSR", _fmt_num(e.get("psr", 0), 3))
    c12.metric("Turnover", _fmt_num(e.get("turnover", 0), 2) + "x")

    st.divider()
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Resumen del Backtest")
        st.dataframe(
            pd.DataFrame(
                [
                    ["Clase", re.info.clase],
                    ["Resolución", re.info.resolucion_principal or "-"],
                    ["Periodo", f"{re.info.fecha_inicio} → {re.info.fecha_fin}"],
                    ["Warm-up", re.info.warmup or 0],
                    ["Warm-up Res.", re.info.warmup_resolucion or "-"],
                    ["Universo", ", ".join(re.info.simbolos) or "-"],
                    ["Benchmark", re.info.benchmark_efectivo or "-"],
                    ["Barras evaluadas", f"{re.barras_evaluadas:,}"],
                    ["Tiempo (s)", f"{re.duracion_s:.2f}"],
                    ["Éxito", "Sí" if re.exito else "No"],
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    with colB:
        st.subheader("Costes & Exposición")
        st.dataframe(
            pd.DataFrame(
                [
                    ["Comisiones", _fmt_usd(e.get("comisiones", 0))],
                    ["Volumen", f"{e.get('volumen', 0):,.0f}"],
                    ["Ejecuciones", f"{e.get('n_ejecuciones', 0):,}"],
                    ["Operaciones cerradas", f"{e.get('n_operaciones', 0):,}"],
                    ["Exposición media", _fmt_pct(e.get("exposicion_media", 0))],
                    ["Margen medio", _fmt_usd(e.get("margen_medio", 0))],
                    ["Ganancia no realizada", _fmt_usd(e.get("sin_realizar", 0))],
                    ["Drawdown Máx.", _fmt_pct(e.get("dd_max", 0))],
                    ["Drawdown Medio", _fmt_pct(e.get("dd_medio", 0))],
                    ["Alfa vs Benchmark", _fmt_pct(e.get("alfa", 0))],
                    ["Beta vs Benchmark", _fmt_num(e.get("beta", 0), 2)],
                    ["Tracking Error", _fmt_pct(e.get("tracking_error", 0))],
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    if re.error:
        st.error(re.error)
    if re.traza:
        with st.expander("Traza de error"):
            st.code(re.traza)

# --------------------------------------------------------------------------- #
# Tab Curva
# --------------------------------------------------------------------------- #
with tab_curva:
    if curva_df.empty:
        st.warning("No hay curva de equity disponible.")
    else:
        fig_eq = go.Figure()
        fig_eq.add_trace(
            go.Scatter(
                x=curva_df.index,
                y=curva_df["equity"],
                mode="lines",
                name="Equity",
                line=dict(color="#3d9cff", width=1.8),
            )
        )
        if "cash" in curva_df.columns:
            fig_eq.add_trace(
                go.Scatter(
                    x=curva_df.index,
                    y=curva_df["cash"],
                    mode="lines",
                    name="Cash",
                    line=dict(color="#f59e0b", width=1.2, dash="dot"),
                )
            )
        if "holdings" in curva_df.columns:
            fig_eq.add_trace(
                go.Scatter(
                    x=curva_df.index,
                    y=curva_df["holdings"],
                    mode="lines",
                    name="Holdings",
                    line=dict(color="#4ade80", width=1.2),
                )
            )
        # Benchmark reexpresado con la misma caja inicial para poder compararlo
        # en la misma escala que la curva de equity (requisito del dashboard).
        bench = getattr(re, "benchmark", None)
        if bench is not None and len(bench) > 1:
            try:
                equity_ini = float(e.get("equity_inicial", 100_000.0))
                b0 = float(bench.iloc[0])
                if b0 > 0:
                    fig_eq.add_trace(
                        go.Scatter(
                            x=bench.index,
                            y=bench * (equity_ini / b0),
                            mode="lines",
                            name="Benchmark",
                            line=dict(color="#a855f7", width=1.6, dash="dash"),
                        )
                    )
            except Exception:
                pass
        fig_eq.update_layout(
            title="Curva de Equity",
            template="plotly_dark",
            xaxis_title="Fecha/Hora",
            yaxis_title="Valor ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig_eq, use_container_width=True, theme="streamlit")

        col_eq, col_dd = st.columns(2)
        with col_eq:
            curva_df["ret_d"] = curva_df["equity"].pct_change().fillna(0)
            fig_ret = px.histogram(curva_df, x="ret_d", nbins=60, title="Distribución de retornos diarios/minuto")
            fig_ret.update_layout(template="plotly_dark", xaxis_title="Retorno", yaxis_title="Frecuencia")
            st.plotly_chart(fig_ret, use_container_width=True, theme="streamlit")
        with col_dd:
            equity = curva_df["equity"]
            dd = equity / equity.cummax() - 1.0
            fig_dd = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", line=dict(color="#ef4444")))
            fig_dd.update_layout(title="Drawdown", template="plotly_dark", yaxis=dict(tickformat=".1%"))
            st.plotly_chart(fig_dd, use_container_width=True, theme="streamlit")

# --------------------------------------------------------------------------- #
# Tab Gráficos
# --------------------------------------------------------------------------- #
with tab_graficos:
    if curva_df.empty:
        st.warning("Sin datos para graficar.")
    else:
        g1, g2 = st.columns(2)
        with g1:
            if "margin" in curva_df.columns:
                fig_m = go.Figure(go.Scatter(x=curva_df.index, y=curva_df["margin"], name="Margen usado", line=dict(color="#f59e0b")))
                fig_m.update_layout(title="Margen usado", template="plotly_dark")
                st.plotly_chart(fig_m, use_container_width=True, theme="streamlit")
        with g2:
            if "fees" in curva_df.columns:
                fig_f = go.Figure(go.Scatter(x=curva_df.index, y=curva_df["fees"].cumsum(), name="Comisiones acumuladas", line=dict(color="#a855f7")))
                fig_f.update_layout(title="Comisiones acumuladas", template="plotly_dark")
                st.plotly_chart(fig_f, use_container_width=True, theme="streamlit")

        if not fills_df.empty:
            fills_df["hora"] = fills_df["time"]
            fig_fills = px.scatter(
                fills_df,
                x="hora",
                y="price",
                color="side",
                size="qty",
                hover_data=["symbol", "qty", "price", "fee"],
                title="Ejecuciones (fills)",
            )
            fig_fills.update_layout(template="plotly_dark")
            st.plotly_chart(fig_fills, use_container_width=True, theme="streamlit")

# --------------------------------------------------------------------------- #
# Tab Trades
# --------------------------------------------------------------------------- #
with tab_trades:
    c_f, c_t = st.columns(2)
    with c_f:
        st.subheader("Ejecuciones (Fills)")
        if fills_df.empty:
            st.info("No hay ejecuciones.")
        else:
            st.dataframe(fills_df, use_container_width=True, hide_index=True)
    with c_t:
        st.subheader("Trades cerrados")
        if trades_df.empty:
            st.info("No hay trades cerrados.")
        else:
            trades_df["PnL"] = trades_df["pnl"]
            st.dataframe(trades_df, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# Tab Logs
# --------------------------------------------------------------------------- #
with tab_logs:
    st.subheader("Logs & Eventos")
    logs_df = _lineas_log(re)
    if logs_df.empty:
        st.info("Sin logs.")
    else:
        st.dataframe(logs_df, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# Tab Explorador
# --------------------------------------------------------------------------- #
with tab_explorador:
    st.subheader("Explorador de Datos")
    universo = list(info_parse.simbolos) if info_parse else []
    if not universo:
        st.info("La estrategia no declara símbolos.")
    else:
        sym = st.selectbox("Símbolo", universo)
        col_r, col_d = st.columns(2)
        with col_r:
            r = st.selectbox("Resolución", ["Daily", "Minute"], index=0)
        with col_d:
            f_ini = st.date_input("Desde", value=pd.Timestamp("2025-09-01").date(), key="ed_ini")
            f_fin = st.date_input("Hasta", value=pd.Timestamp("2026-09-30").date(), key="ed_fin")
        try:
            df_b = datafeed.bars(sym, r, f_ini, f_fin)
        except Exception as e:
            st.error(f"Error cargando barras: {e}")
            df_b = pd.DataFrame()
        if not df_b.empty:
            st.dataframe(df_b.tail(500), use_container_width=True)
            fig_c = go.Figure(data=[go.Candlestick(x=df_b.index, open=df_b["open"], high=df_b["high"], low=df_b["low"], close=df_b["close"])])
            fig_c.update_layout(template="plotly_dark", title=f"{sym} · {r}")
            st.plotly_chart(fig_c, use_container_width=True, theme="streamlit")
        else:
            st.info("Sin datos en el rango seleccionado.")

# --------------------------------------------------------------------------- #
# Tab IA
# --------------------------------------------------------------------------- #
with tab_ia:
    st.subheader("Asistente IA")
    st.info("Este asistente está preparado para analizar el backtest, sugerir mejoras y generar variantes parametrizadas (próxima iteración).")
    consulta = st.text_area("Pregunta sobre el backtest (ej.: ¿Qué parámetro mejora el Sharpe?)", height=100)
    if st.button("Analizar", use_container_width=True):
        with st.spinner("Analizando..."):
            resumen = {
                "clase": re.info.clase,
                "retorno_total": e.get("retorno_total"),
                "sharpe": e.get("sharpe"),
                "sortino": e.get("sortino"),
                "dd_max": e.get("dd_max"),
                "win_rate": e.get("win_rate"),
                "profit_factor": e.get("profit_factor"),
                "n_operaciones": e.get("n_operaciones"),
                "exposicion_media": e.get("exposicion_media"),
            }
            st.success("Análisis listo (stub). Pega aquí tu modelo LLM o conecta MCP para respuestas detalladas.")
            st.json(resumen)

# --------------------------------------------------------------------------- #
# Comparativa multi-estrategia
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("⚔️ Comparar estrategias")

multi_sel = st.multiselect(
    "Estrategias a comparar",
    options=[p for p in estrategias if p != selected_path],
    default=[],
    format_func=lambda p: str(p.relative_to(ROOT)),
)
if multi_sel:
    if st.button("Ejecutar comparación", type="primary"):
        with st.spinner("Ejecutando comparativa..."):
            tablas = []
            fig_cmp = go.Figure()
            with st.status("Comparando estrategias...", expanded=True):
                for p in multi_sel:
                    try:
                        r2 = engine_mod.ejecutar_backtest(
                            str(p),
                            capital=float(capital_inicial),
                            fecha_inicio=str(periodo_inicio),
                            fecha_fin=str(periodo_fin),
                        )
                        if not r2.exito:
                            continue
                        e2 = r2.estadisticas or {}
                        tablas.append({
                            "Estrategia": str(p.relative_to(ROOT)),
                            "Equity Final": e2.get("equity_final", 0.0),
                            "Retorno": e2.get("retorno_total", 0.0),
                            "Sharpe": e2.get("sharpe", 0.0),
                            "Sortino": e2.get("sortino", 0.0),
                            "MaxDD": e2.get("dd_max", 0.0),
                            "CAGR": e2.get("cagr", 0.0),
                            "Win Rate": e2.get("win_rate", 0.0),
                            "PF": e2.get("profit_factor", 0.0),
                            "Ops": e2.get("n_operaciones", 0),
                        })
                        cur = r2.curva
                        if cur is not None and not cur.empty:
                            s = cur["equity"]
                            s = s / float(s.iloc[0]) * float(e.get("equity_inicial", 100_000.0))
                            fig_cmp.add_trace(go.Scatter(
                                x=s.index, y=s, mode="lines",
                                name=str(p.relative_to(ROOT)),
                            ))
                    except Exception as exc:
                        st.warning(f"{p.name}: {exc}")

            if tablas:
                df_cmp = pd.DataFrame(tablas)
                fig_cmp.update_layout(
                    title="Curvas de Equity normalizadas",
                    template="plotly_dark",
                    xaxis_title="Fecha",
                    yaxis_title="Equity ($)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=10, r=10, t=60, b=10),
                )
                st.plotly_chart(fig_cmp, use_container_width=True, theme="streamlit")

                df_view = df_cmp.copy()
                for col in ("Retorno", "MaxDD", "CAGR", "Win Rate"):
                    df_view[col] = df_view[col].map(lambda v: f"{v*100:.2f}%")
                st.dataframe(df_view, use_container_width=True, hide_index=True)

                csv = df_cmp.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Descargar comparativa CSV", csv,
                                   file_name="comparativa.csv", mime="text/csv")
            else:
                st.warning("Ninguna estrategia comparada terminó correctamente.")
else:
    st.caption("Selecciona una o más estrategias arriba para superimponer sus curvas de equity.")

# --------------------------------------------------------------------------- #
# Terminal de logs inferior
# --------------------------------------------------------------------------- #
st.divider()
with st.expander("🖥️ Terminal de ejecución", expanded=False):
    logs_all = re.logs or []
    if logs_all:
        lineas = []
        for lg in logs_all[-300:]:
            ts = lg.get("tiempo")
            try:
                ts = pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                ts = str(ts)
            nivel = str(lg.get("nivel", "INFO")).upper()
            color = {"ERROR": "#ef4444", "WARNING": "#f59e0b",
                     "DEBUG": "#8a8a8a"}.get(nivel, "#9ca3af")
            msg = str(lg.get("mensaje", "")).replace("<", "&lt;")
            lineas.append(
                f"<div style='color:{color};font-family:Consolas,monospace;"
                f"font-size:0.82rem;line-height:1.45'>"
                f"[{ts}] <b>{nivel:8s}</b> {msg}</div>"
            )
        st.markdown("<div style='max-height:340px;overflow-y:auto;background:#0b0b0b;"
                    "border:1px solid #2f2f2f;border-radius:10px;padding:10px'>"
                    + "".join(lineas) + "</div>", unsafe_allow_html=True)
    else:
        st.info("Sin logs de ejecución.")

st.caption(f"Última ejecución: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
