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
    assert any("Backtest en espera" in item.value for item in session.info)
    assert "flash_msg" not in session.session_state

    session = session.radio(key="workspace_tab").set_value("INICIO").run()
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
