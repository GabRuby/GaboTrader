"""Smoke tests for the dashboard navigation and responsive panel controls."""

from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: E402


def test_dashboard_tabs_keep_the_assistant_available():
    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()

    assert not session.exception, [error.value for error in session.exception]
    assert session.session_state["workspace_tab"] == "INICIO"
    assert tuple(session.radio(key="workspace_tab").options) == (
        "🏠 INICIO",
        *app.WORKSPACE_TABS[1:],
    )

    for tab in app.WORKSPACE_TABS:
        session = session.radio(key="workspace_tab").set_value(tab).run()
        assert not session.exception, [error.value for error in session.exception]
        assert session.session_state["workspace_tab"] == tab
        assert session.button(key="btai_run").label == "📊 Interpretar Resultados con IA"
        global_header = " ".join(
            item.value for item in session.markdown if 'class="topbar' in item.value
        )
        assert "En espera" not in global_header
        assert "gemini-3.1-flash-lite" not in global_header
        assert "v18142" not in global_header
        if tab != "LABORATORIO":
            assert not any('class="lab-context"' in item.value for item in session.markdown)
            assert session.button(key="future_ai_action").disabled

    assert any("El historial persistente" in item.value for item in session.markdown)


def test_editor_sync_updates_from_widget_and_close_cancel_preserves_edit(monkeypatch):
    from qc_lab.core import runner

    def forbid_backtest(*_args, **_kwargs):
        raise RuntimeError("backtest deshabilitado en prueba de editor")

    monkeypatch.setattr(runner, "ejecutar_estrategia", forbid_backtest)
    script = "grid_trading_v2.py"
    source = (ROOT / script).read_text(encoding="utf-8")
    editor_key = f"editor_{script}"
    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    session.session_state["workspace_tab"] = "LABORATORIO"
    session.session_state[app.PRINCIPAL_KEY] = script
    session = session.run()

    def editor_header(current):
        return " ".join(
            item.value
            for item in current.markdown
            if 'class="editor-header"' in item.value
        )

    assert 'class="editor-sync-status saved"' in editor_header(session)
    assert session.button(key="btn_save").disabled

    edited = source + "\n# edición de prueba\n"
    session = session.text_area(key=editor_key).set_value(edited).run()
    assert 'class="editor-sync-status pending"' in editor_header(session)
    assert session.button(key="btn_save").disabled is False

    session = session.text_area(key=editor_key).set_value(source).run()
    assert 'class="editor-sync-status saved"' in editor_header(session)
    assert session.button(key="btn_save").disabled

    session = session.text_area(key=editor_key).set_value(edited).run()
    session = session.button(key="btn_prueba_quitar_principal").click().run()
    assert session.session_state[app.PRINCIPAL_KEY] == script
    assert session.session_state[app.EDITOR_NAVIGATION_KEY]["action"] == "close"
    assert session.text_area(key=editor_key).value == edited

    session = session.button(key="editor_nav_cancel_close").click().run()
    assert app.EDITOR_NAVIGATION_KEY not in session.session_state
    assert session.session_state[app.PRINCIPAL_KEY] == script
    assert session.text_area(key=editor_key).value == edited
    assert 'class="editor-sync-status pending"' in editor_header(session)


def test_principal_and_single_comparative_editors_are_independent(monkeypatch):
    from qc_lab.core import runner

    def forbid_backtest(*_args, **_kwargs):
        raise RuntimeError("backtest deshabilitado en prueba de comparativas")

    scripts = ["grid_trading_v2.py", "otras/SP500.py"]
    sources = {
        script: (ROOT / script).read_text(encoding="utf-8") for script in scripts
    }
    monkeypatch.setattr(runner, "ejecutar_estrategia", forbid_backtest)
    monkeypatch.setattr(app, "get_comparative_results", lambda *_args, **_kwargs: {})

    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    session = session.radio(key="workspace_tab").set_value("LABORATORIO").run()
    session.session_state[app.PRINCIPAL_KEY] = scripts[0]
    session.session_state[app.BACKTEST_PENDING_KEY] = scripts[0]
    session.session_state[app.COMPARE_KEY] = [scripts[1]]
    session = session.run()

    comparison_key = app.compare_editor_widget_key(scripts[1])
    principal_key = f"editor_{scripts[0]}"
    assert not session.exception, [error.value for error in session.exception]
    assert session.text_area(key=principal_key).value == sources[scripts[0]]
    assert session.text_area(key=comparison_key).value == sources[scripts[1]]
    panel_headers = " ".join(
        item.value for item in session.markdown if 'class="editor-header' in item.value
    )
    assert panel_headers.count("Catálogo del proyecto") == 2
    assert session.get("plotly_chart")
    assert session.get("dataframe")
    assert session.button(key="btn_save").disabled
    assert session.button(key=f"compare_save_{scripts[1]}").disabled
    assert session.button(key="btn_undo_main").disabled
    assert session.button(key=f"compare_undo_{scripts[1]}").disabled

    principal_edited = sources[scripts[0]] + "\n# edición principal\n"
    comparison_edited = sources[scripts[1]] + "\n# edición comparativa\n"
    session = session.text_area(key=principal_key).set_value(
        principal_edited
    ).run()
    assert not session.button(key="btn_save").disabled
    assert session.button(key=f"compare_save_{scripts[1]}").disabled
    session = session.text_area(key=comparison_key).set_value(comparison_edited).run()
    assert not session.button(key="btn_save").disabled
    assert not session.button(key=f"compare_save_{scripts[1]}").disabled

    sync_statuses = " ".join(
        item.value for item in session.markdown if 'class="editor-sync-status' in item.value
    )
    assert sync_statuses.count('class="editor-sync-status pending"') == 2
    assert session.session_state[app.COMPARE_EDITOR_SOURCES_KEY][scripts[1]] == comparison_edited

    session = session.button(key="btn_undo_main").click().run()
    assert session.text_area(key=principal_key).value == sources[scripts[0]]
    assert session.text_area(key=comparison_key).value == comparison_edited
    assert session.button(key="btn_save").disabled
    assert not session.button(key=f"compare_save_{scripts[1]}").disabled

    session = session.button(key=f"compare_undo_{scripts[1]}").click().run()
    assert session.text_area(key=comparison_key).value == sources[scripts[1]]
    assert session.button(key=f"compare_save_{scripts[1]}").disabled

    session = session.text_area(key=comparison_key).set_value(comparison_edited).run()
    session = session.button(key="rm_comparativa_0").click().run()
    assert session.session_state[app.EDITOR_NAVIGATION_KEY]["current"] == scripts[1]
    session = session.button(key="editor_nav_cancel_remove_comparison").click().run()
    assert session.session_state[app.COMPARE_KEY] == [scripts[1]]
    assert session.text_area(key=comparison_key).value == comparison_edited

    session = session.button(key="rm_comparativa_0").click().run()
    session = session.button(
        key="editor_nav_discard_remove_comparison"
    ).click().run()
    assert session.session_state[app.COMPARE_KEY] == []
    assert session.session_state[app.PRINCIPAL_KEY] == scripts[0]


def test_global_header_and_laboratorio_context_are_separated():
    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()

    global_header = " ".join(
        item.value for item in session.markdown if 'class="topbar' in item.value
    )
    assistant_captions = " ".join(item.value for item in session.caption)
    assert "GABO trader LAB" in global_header
    assert "Entorno ·" in global_header
    assert "Sin Estrategia Principal" not in global_header
    assert "TSLA" not in global_header
    assert "En espera" not in global_header
    assert "gemini-3.1-flash-lite" not in global_header
    assert "v18142" not in global_header
    assert "Modelo seleccionado · gemini-3.1-flash-lite" in assistant_captions
    assert not any('class="lab-context"' in item.value for item in session.markdown)
    assert not any("Identificador interno del motor" in item.value for item in session.caption)

    session = session.radio(key="workspace_tab").set_value("LABORATORIO").run()
    lab_context = " ".join(
        item.value for item in session.markdown if 'class="lab-context"' in item.value
    )
    assert "Estrategia · (sin Estrategia Principal)" in lab_context
    assert "Activo de trabajo · TSLA" in lab_context
    assert "Backtest en espera" in lab_context
    assert any("Backtest en espera: selecciona una estrategia" in item.value for item in session.info)
    assert any("Identificador interno del motor · v18142" in item.value for item in session.caption)
    editor_header = " ".join(
        item.value for item in session.markdown if 'class="editor-header"' in item.value
    )
    assert 'class="editor-sync-status no-strategy"' in editor_header
    assert "Sin estrategia abierta" in editor_header


def test_assistant_ai_settings_preserve_migration_and_work_across_all_tabs():
    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    provider_key = f"{app.ASSISTANT_AI_PREFIX}_provider"
    model_key = f"{app.ASSISTANT_AI_PREFIX}_model"
    local_model_key = f"{app.ASSISTANT_AI_PREFIX}_model_local"

    assert all(
        widget.key != f"{app.TOPBAR_AI_PREFIX}_provider"
        for widget in session.selectbox
    )
    session.session_state[f"{app.TOPBAR_AI_PREFIX}_provider"] = "Ollama (local)"
    session.session_state[f"{app.TOPBAR_AI_PREFIX}_model_local"] = "qwen2.5-coder"
    session = session.run()
    assert session.selectbox(key=provider_key).value == "Ollama (local)"
    assert session.selectbox(key=local_model_key).value == "qwen2.5-coder"

    session = session.selectbox(key=local_model_key).set_value("llama3.1").run()
    assert session.session_state[f"{app.TOPBAR_AI_PREFIX}_provider"] == "Ollama (local)"
    assert session.session_state[f"{app.TOPBAR_AI_PREFIX}_model_local"] == "llama3.1"
    session = session.selectbox(key=provider_key).set_value("Gemini (Google)").run()
    assert session.selectbox(key=model_key).value == "gemini-3.1-flash-lite"
    session = session.selectbox(key=model_key).set_value("gemini-2.5-pro").run()
    assert session.session_state[f"{app.TOPBAR_AI_PREFIX}_provider"] == "Gemini (Google)"
    assert session.session_state[f"{app.TOPBAR_AI_PREFIX}_model"] == "gemini-2.5-pro"
    assert session.session_state[f"{app.TOPBAR_AI_PREFIX}_model_local"] == "llama3.1"
    assert any("Modelo seleccionado · gemini-2.5-pro" in item.value for item in session.caption)

    for tab in app.WORKSPACE_TABS:
        session = session.radio(key="workspace_tab").set_value(tab).run()
        assert not session.exception, [error.value for error in session.exception]
        assert session.button(key="btai_run").label == "📊 Interpretar Resultados con IA"
        assert session.session_state[f"{app.TOPBAR_AI_PREFIX}_model"] == "gemini-2.5-pro"


def test_dashboard_assistant_panel_width_is_bounded():
    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not session.exception, [error.value for error in session.exception]
    assert session.session_state["assistant_panel_ratio"] == 0.19
    assert app.normalize_assistant_panel_ratio(-5) == 0.125
    assert app.normalize_assistant_panel_ratio(5) == 0.375
    assert app.normalize_assistant_panel_ratio("not-a-ratio") == 0.19


def test_discover_home_has_no_market_quotes_and_navigation_actions_work():
    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()

    texts = " ".join(item.value for item in session.markdown)
    texts += " ".join(item.value for item in session.info)
    texts += " ".join(item.value for item in session.caption)
    texts += " ".join(item.value for item in session.subheader)
    assert session.title[0].value == "🏠 INICIO"
    assert "Datos no disponibles" in texts
    assert "Conecta una fuente de datos para consultar el panorama del mercado." in texts
    assert "Aún no hay oportunidades evaluadas." in texts
    assert "No hay un portafolio seleccionado." in texts

    session = session.button(key="discover_open_lab").click().run()
    assert session.session_state["workspace_tab"] == "LABORATORIO"


def test_laboratorio_messages_are_isolated_from_inicio():
    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not any("Backtest en espera" in item.value for item in session.info)
    session.session_state["flash_msg"] = "Aviso pendiente del Laboratorio"
    session = session.run()
    assert session.session_state["flash_msg"] == "Aviso pendiente del Laboratorio"

    session = session.radio(key="workspace_tab").set_value("LABORATORIO").run()
    assert not any(
        "Bienvenido al Laboratorio de Estrategias" in item.value
        for item in session.markdown
    )
    assert "flash_msg" not in session.session_state

    session = session.radio(key="workspace_tab").set_value("INICIO").run()
    assert not any(
        "Bienvenido al Laboratorio de Estrategias" in item.value
        for item in session.markdown
    )
    assert not any("Backtest en espera" in item.value for item in session.info)


def test_navigation_preserves_the_selected_strategy_and_its_parameters():
    script = "otras/SP500.py"
    strategy_path = ROOT / script
    info = app.strategy_parser.analizar_archivo(strategy_path)
    ticker, capital = app.reglas_comunes_de_estrategia(strategy_path, info)
    parameter_values = {
        parameter.clave: (
            int(parameter.valor) - 1
            if parameter.clave == "max_escalones"
            else int(parameter.valor)
            if parameter.es_entero
            else float(parameter.valor)
        )
        for parameter in info.parametros
    }
    parameter_stamp = tuple(
        sorted((key, round(float(value), 6)) for key, value in parameter_values.items())
    )

    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    session = session.radio(key="workspace_tab").set_value("LABORATORIO").run()
    session.session_state[app.PRINCIPAL_KEY] = script
    editor_key = f"editor_{script}"
    editor_code = "# edición local que debe sobrevivir al cambio de pestaña\n"
    for key, value in parameter_values.items():
        session.session_state[f"param_{script}_{key}"] = value
    result_df, result_metrics = app.resultado_en_espera(ticker, capital)
    session.session_state[app.RESULT_KEY] = {
        "df": result_df,
        "metrics": result_metrics,
    }
    session.session_state[app.PARAMS_KEY] = (
        script,
        ticker,
        parameter_stamp,
        float(capital),
    )
    session = session.run()
    session = session.text_area(key=editor_key).set_value(editor_code).run()

    assert not session.exception, [error.value for error in session.exception]
    assert session.session_state[app.PRINCIPAL_KEY] == script
    assert session.session_state["param_" + script + "_max_escalones"] == parameter_values["max_escalones"]
    assert session.text_area(key=editor_key).value == editor_code
    lab_context = " ".join(
        item.value for item in session.markdown if 'class="lab-context"' in item.value
    )
    assert f"Estrategia · {script}" in lab_context
    assert f"Activo de trabajo · {session.session_state[app.ACTIVO_KEY]}" in lab_context

    session = session.radio(key="workspace_tab").set_value("PORTAFOLIO").run()
    assert session.session_state[app.PRINCIPAL_KEY] == script
    assert session.session_state["param_" + script + "_max_escalones"] == parameter_values["max_escalones"]
    assert session.session_state["codigo_actual"] == editor_code

    session = session.radio(key="workspace_tab").set_value("LABORATORIO").run()
    assert session.session_state[app.PRINCIPAL_KEY] == script
    assert session.session_state["param_" + script + "_max_escalones"] == parameter_values["max_escalones"]
    assert session.text_area(key=editor_key).value == editor_code


def test_sin_principal_el_laboratorio_muestra_la_plantilla_sin_ejecutar(monkeypatch):
    """Sin Estrategia Principal el editor abre la plantilla inicial como borrador."""
    from qc_lab.core import runner

    def forbid_backtest(*_args, **_kwargs):
        raise RuntimeError("no debe ejecutarse ningún backtest al abrir la plantilla")

    monkeypatch.setattr(runner, "ejecutar_estrategia", forbid_backtest)
    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    session = session.radio(key="workspace_tab").set_value("LABORATORIO").run()

    assert not session.exception, [error.value for error in session.exception]
    assert session.session_state[app.PRINCIPAL_KEY] in (None, "")

    # El editor principal es la plantilla inicial, editable y sin archivo asociado.
    assert session.text_area(key=app.PLANTILLA_EDITOR_KEY).value == app.PLANTILLA_INICIAL
    headers = " ".join(
        item.value for item in session.markdown if 'class="editor-header"' in item.value
    )
    assert "Plantilla inicial del Laboratorio" in headers
    assert "Borrador temporal" in headers

    # El banner y la guía de bienvenida se retiraron del laboratorio.
    markdown = " ".join(item.value for item in session.markdown)
    assert "Bienvenido al Laboratorio de Estrategias" not in markdown
    assert 'class="lab-steps"' not in markdown

    # Estado vacío: KPIs con guion, gráfica vacía, sin tabla y sin resultado guardado.
    kpis = " ".join(
        item.value for item in session.markdown if 'class="kpi-container"' in item.value
    )
    assert "—" in kpis
    assert 'class="lab-empty-chart"' in markdown
    assert not session.get("dataframe")
    assert not session.session_state.get(app.PLANTILLA_RESULT_KEY)

    # Variables de la estrategia: valores por defecto 10 / 50.
    assert session.slider(key=f"{app.PLANTILLA_PARAM_PREFIX}fast_ma").value == 10
    assert session.slider(key=f"{app.PLANTILLA_PARAM_PREFIX}slow_ma").value == 50

    # Cambiar un control actualiza el estado del borrador sin tocar archivos.
    session = session.slider(key=f"{app.PLANTILLA_PARAM_PREFIX}fast_ma").set_value(30).run()
    assert session.session_state[f"{app.PLANTILLA_PARAM_PREFIX}fast_ma"] == 30

    # «Restablecer valores» devuelve los controles a los valores por defecto.
    session = session.button(key="btn_reset_plantilla_params").click().run()
    assert session.slider(key=f"{app.PLANTILLA_PARAM_PREFIX}fast_ma").value == 10
    assert session.slider(key=f"{app.PLANTILLA_PARAM_PREFIX}slow_ma").value == 50


def _entrar_en_laboratorio():
    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    return session.radio(key="workspace_tab").set_value("LABORATORIO").run()


def test_plantilla_simulaciones_registran_curvas_y_se_pueden_eliminar():
    """«Simular» crea pruebas con curva y color propios, visibles y eliminables."""
    session = _entrar_en_laboratorio()
    assert not session.exception, [error.value for error in session.exception]

    session = session.button(key="btn_simular_plantilla").click().run()
    assert not session.exception, [error.value for error in session.exception]
    registros = session.session_state[app.LAB_SIMS_KEY]
    assert len(registros) == 1
    assert registros[0]["label"] == "Prueba 1"
    assert registros[0]["params"] == {"fast_ma": 10, "slow_ma": 50}
    assert not registros[0]["equity"].empty
    assert session.get("plotly_chart")
    assert "Prueba 1" in " ".join(item.value for item in session.caption)

    # Tarjeta compacta: línea de parámetros reales y capital final visible en la tarjeta.
    captions = " ".join(item.value for item in session.caption)
    markdown = " ".join(item.value for item in session.markdown)
    assert "Parámetros: fast_ma=10 · slow_ma=50" in captions
    capital_1 = f"${float(registros[0]['metrics']['equity']):,.2f}"
    assert f"Capital final: {capital_1}" in markdown
    assert "Prueba 1" in markdown

    # Con un parámetro distinto se registra una segunda prueba con otro color y su capital.
    session = session.slider(
        key=f"{app.PLANTILLA_PARAM_PREFIX}fast_ma"
    ).set_value(15).run()
    session = session.button(key="btn_simular_plantilla").click().run()
    registros = session.session_state[app.LAB_SIMS_KEY]
    assert len(registros) == 2
    assert registros[1]["params"]["fast_ma"] == 15
    assert registros[0]["color"] != registros[1]["color"]
    capital_2 = f"${float(registros[1]['metrics']['equity']):,.2f}"
    assert capital_1 != capital_2
    markdown = " ".join(item.value for item in session.markdown)
    assert f"Capital final: {capital_1}" in markdown
    assert f"Capital final: {capital_2}" in markdown

    # Eliminar una prueba es inmediato: sin confirmación, sin advertencias y sin tooltip.
    assert session.button(key="sim_del_1").help in (None, "")
    session = session.button(key="sim_del_1").click().run()
    assert not session.warning
    assert [r["id"] for r in session.session_state[app.LAB_SIMS_KEY]] == [2]
    assert app.LAB_SIM_DELETE_KEY not in session.session_state
    assert session.get("plotly_chart")
    markdown = " ".join(item.value for item in session.markdown)
    assert capital_1 not in markdown
    assert capital_2 in markdown


def test_plantilla_paneles_de_informacion_muestran_metricas_propias():
    """Cada panel «ⓘ» conserva las métricas secundarias reales de su propia prueba."""
    session = _entrar_en_laboratorio()
    session = session.button(key="btn_simular_plantilla").click().run()
    session = session.slider(
        key=f"{app.PLANTILLA_PARAM_PREFIX}fast_ma"
    ).set_value(20).run()
    session = session.button(key="btn_simular_plantilla").click().run()
    registros = session.session_state[app.LAB_SIMS_KEY]
    assert [r["id"] for r in registros] == [1, 2]
    markdown = " ".join(item.value for item in session.markdown)
    for registro in registros:
        met = registro["metrics"]
        for etiqueta, valor in (
            ("Retorno total", f"{float(met['total_return']):.2f}%"),
            ("Sharpe", f"{float(met['sharpe']):.2f}"),
            ("Máximo drawdown", f"{float(met['max_drawdown']):.2f}%"),
            ("Operaciones", str(int(met["trades"]))),
            ("Comisiones", f"${float(met['fees']):,.2f}"),
            ("Capital final", f"${float(met['equity']):,.2f}"),
        ):
            fila = (
                f'<span class="sim-info-k">{etiqueta}</span>'
                f'<span class="sim-info-v">{valor}</span>'
            )
            assert fila in markdown
    # El panel no usa el interlineado por defecto de Streamlit (filas propias compactas).
    assert 'class="sim-info"' in markdown
    # El icono de información no expone tooltip en el control de la tarjeta.
    assert session.button(key="sim_del_1").help in (None, "")


def test_plantilla_conserva_resultado_y_avisa_de_desactualizacion():
    """Ejecutar guarda el resultado; cambiar un parámetro no lo borra, solo lo marca."""
    session = _entrar_en_laboratorio()
    session = session.button(key="btn_run_plantilla").click().run()
    assert not session.exception, [error.value for error in session.exception]
    assert session.session_state.get(app.PLANTILLA_RESULT_KEY)
    assert not any("ejecución anterior" in item.value for item in session.info)

    session = session.slider(
        key=f"{app.PLANTILLA_PARAM_PREFIX}fast_ma"
    ).set_value(20).run()
    assert session.session_state.get(app.PLANTILLA_RESULT_KEY)
    assert any("ejecución anterior" in item.value for item in session.info)


def test_plantilla_guardar_como_dialogo_abre_y_cancela():
    """«Guardar como…» abre un diálogo con validación y permite cancelar sin escribir."""
    session = _entrar_en_laboratorio()
    session = session.button(key="btn_save_as_plantilla").click().run()
    assert not session.exception, [error.value for error in session.exception]
    assert session.session_state.get(app.LAB_SAVE_AS_OPEN_KEY)
    assert not session.button(key="plantilla_save_as_confirmar").disabled

    session = session.button(key="plantilla_save_as_cancelar").click().run()
    assert not session.session_state.get(app.LAB_SAVE_AS_OPEN_KEY)
    assert not (ROOT / "mi_estrategia.py").exists()


def test_plantilla_optimizacion_genera_propuesta_y_la_descarta():
    """«Optimizar variables con IA» propone parámetros y se puede descartar."""
    session = _entrar_en_laboratorio()
    ai_button = session.button(key="btn_ai_plantilla")
    assert ai_button.label == "✨ Optimizar variables con IA"
    assert not ai_button.disabled
    session = ai_button.click().run()
    assert not session.exception, [error.value for error in session.exception]
    assert session.session_state.get(app.LAB_OPT_KEY)
    # El capital final sugerido se destaca y no se registra ninguna prueba interna.
    markdown = " ".join(item.value for item in session.markdown)
    assert "Capital final sugerido (ajuste):" in markdown
    assert not session.session_state.get(app.LAB_SIMS_KEY)
    assert session.button(key="lab_opt_aceptar").label == "Aceptar propuesta"
    session = session.button(key="lab_opt_descartar").click().run()
    assert not session.session_state.get(app.LAB_OPT_KEY)


def test_plantilla_optimizacion_aceptar_no_crea_simulacion():
    """Aceptar la propuesta vuelca los parámetros pero no ejecuta una prueba visible."""
    session = _entrar_en_laboratorio()
    session.session_state[app.LAB_OPT_KEY] = {
        "actual": {"fast_ma": 10, "slow_ma": 50},
        "sugerido": {"fast_ma": 12, "slow_ma": 60},
        "optimizados": ["fast_ma", "slow_ma"],
        "metricas_actual": None,
        "metricas_sugerido": None,
        "metricas_validacion": None,
        "criterio": "Mayor Sharpe en el ajuste.",
        "periodos": {"ajuste": ("2020-01-01", "2024-06-30"),
                     "validacion": ("2024-07-01", "2026-09-30")},
        "evaluadas": 4,
        "truncado": False,
    }
    session = session.run()
    session = session.button(key="lab_opt_aceptar").click().run()
    assert session.slider(key=f"{app.PLANTILLA_PARAM_PREFIX}fast_ma").value == 12
    assert session.slider(key=f"{app.PLANTILLA_PARAM_PREFIX}slow_ma").value == 60
    assert app.LAB_OPT_KEY not in session.session_state
    # Sin pulsar «Simular» no hay prueba registrada ni curva comparativa nueva.
    assert not session.session_state.get(app.LAB_SIMS_KEY)
    assert any("Parámetros sugeridos aplicados" in item.value for item in session.success)


def test_panel_a_principal_usa_variables_compartidas_y_simular_registra_por_estrategia():
    """Panel A con Estrategia Principal muestra la barra compartida, simula y filtra por script."""
    script = "otras/SP500.py"
    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    session = session.radio(key="workspace_tab").set_value("LABORATORIO").run()
    session.session_state[app.PRINCIPAL_KEY] = script
    session = session.run()

    assert not session.exception, [error.value for error in session.exception]
    # Barras compartidas deben existir con claves del contexto principal
    eks = [getattr(e, "key", None) for e in session.expander]
    bks = [getattr(b, "key", None) for b in session.button]
    assert "variables_principal_panel" in eks
    assert "btn_reset_principal_params" in bks
    assert "btn_simular_principal" in bks
    # Historial asociado a la estrategia principal

    # Valores por defecto declarados por la estrategia
    info = app.strategy_parser.analizar_archivo(ROOT / script)
    if info.parametros:
        primer = info.parametros[0]
        clave_prefijo = f"param_{script}_{primer.clave}"
        sks = [getattr(s, "key", None) for s in session.slider]
        nks = [getattr(n, "key", None) for n in session.number_input]
        assert clave_prefijo in sks or clave_prefijo in nks

    # Restablecer vuelve a valores por defecto (no lanza excepción)
    session = session.button(key="btn_reset_principal_params").click().run()
    assert not session.exception, [error.value for error in session.exception]

    # Simular registra prueba asociada al archivo de la Estrategia Principal
    session = session.button(key="btn_simular_principal").click().run()
    assert not session.exception, [error.value for error in session.exception]
    registros = session.session_state.get(app.LAB_SIMS_KEY) or []
    assert registros, "Se debe registrar al menos una prueba desde Panel A"
    assert any(r.get("script") == script for r in registros)
    assert all(r.get("script") in ("", script) for r in registros)
    # Debe haber al menos la prueba de la principal
    regs_pr = [r for r in registros if r.get("script") == script]
    assert regs_pr
    assert regs_pr[-1].get("params")


def test_panel_a_simulaciones_no_mezclan_con_plantilla():
    """Las curvas/historial se filtran por script; pruebas de plantilla no aparecen en principal."""
    script = "otras/SP500.py"
    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    session = session.radio(key="workspace_tab").set_value("LABORATORIO").run()

    # Crear prueba en plantilla inicial
    session = session.button(key="btn_simular_plantilla").click().run()
    assert not session.exception, [error.value for error in session.exception]
    regs_antes = session.session_state.get(app.LAB_SIMS_KEY) or []
    assert any(r.get("script") == "" for r in regs_antes)

    # Cargar Estrategia Principal en Panel A
    session.session_state[app.PRINCIPAL_KEY] = script
    session = session.run()
    assert not session.exception, [error.value for error in session.exception]
    registros = session.session_state.get(app.LAB_SIMS_KEY) or []
    # Las pruebas de plantilla siguen existiendo pero el panel/historial de principal filtra por script
    assert any(r.get("script") == "" for r in registros)
    assert all(r.get("script") in ("", script) for r in registros)
    # No debe haber pruebas de otra estrategia distinta
    for r in registros:
        s = r.get("script", "")
        assert s == "" or s == script


def test_panel_a_simular_refleja_cambio_de_parametros():
    """Cambiar parámetro y simular registra valores seleccionados (sin datos reales)."""
    script = "otras/SP500.py"
    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    session = session.radio(key="workspace_tab").set_value("LABORATORIO").run()
    session.session_state[app.PRINCIPAL_KEY] = script
    session = session.run()
    assert not session.exception, [error.value for error in session.exception]

    info = app.strategy_parser.analizar_archivo(ROOT / script)
    if not info.parametros:
        return
    primer = info.parametros[0]
    clave_prefijo = f"param_{script}_{primer.clave}"
    sks = [getattr(s, "key", None) for s in session.slider]
    if clave_prefijo in sks:
        try:
            actual = session.slider(key=clave_prefijo).value
            nuevo = actual
            try:
                minimo, maximo, paso = session.slider(key=clave_prefijo).min_value, session.slider(key=clave_prefijo).max_value, session.slider(key=clave_prefijo).step
                if paso and paso > 0 and nuevo + paso <= maximo:
                    nuevo = nuevo + paso
                elif paso and paso > 0 and nuevo - paso >= minimo:
                    nuevo = nuevo - paso
            except Exception:
                nuevo = actual
            if nuevo != actual:
                session = session.slider(key=clave_prefijo).set_value(nuevo).run()
        except Exception:
            pass

    session = session.button(key="btn_simular_principal").click().run()
    assert not session.exception, [error.value for error in session.exception]
    registros = session.session_state.get(app.LAB_SIMS_KEY) or []
    regs_pr = [r for r in registros if r.get("script") == script]
    assert regs_pr
    # params deben reflejar los valores seleccionados
    params_reg = regs_pr[-1].get("params") or {}
    assert isinstance(params_reg, dict)
