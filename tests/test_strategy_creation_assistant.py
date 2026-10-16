"""Tests for the isolated conversational strategy-creation mode."""

from pathlib import Path
import sys

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: E402


def test_creation_mode_activates_without_strategy_and_preserves_conversation():
    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    session = session.radio(key="workspace_tab").set_value("LABORATORIO").run()

    assert session.session_state[app.PRINCIPAL_KEY] is None
    editor_code_before = session.session_state.get("codigo_actual")
    session = session.button(key="strategy_creation_activate").click().run()
    assert not session.exception, [error.value for error in session.exception]
    assert session.session_state[app.PRINCIPAL_KEY] is None
    assert session.session_state.get("codigo_actual") == editor_code_before
    assert (
        session.session_state[app.ASSISTANT_MODE_KEY]
        == app.ASSISTANT_MODE_CREATE_STRATEGY
    )
    assert any(item.value == "Crear estrategia desde cero" for item in session.subheader)

    conversation = [
        {"role": "user", "content": "Quiero una estrategia de tendencia."},
        {"role": "assistant", "content": "¿Qué marco temporal prefieres?"},
    ]
    session.session_state[app.STRATEGY_CREATION_MESSAGES_KEY] = conversation
    session.session_state[app.STRATEGY_CREATION_RULES_KEY] = {
        "proposal": "Usar una media móvil como filtro de tendencia.",
        "approved": True,
        "approved_text": "Usar una media móvil como filtro de tendencia.",
    }
    session.session_state[app.STRATEGY_CREATION_PHASE_KEY] = (
        app.STRATEGY_CREATION_PHASE_CODE_AUTHORIZED
    )
    session = session.button(key="strategy_creation_back").click().run()

    assert session.session_state[app.ASSISTANT_MODE_KEY] == app.ASSISTANT_MODE_GLOBAL
    assert session.session_state[app.STRATEGY_CREATION_MESSAGES_KEY] == conversation
    assert session.session_state[app.STRATEGY_CREATION_RULES_KEY]["approved"] is True
    assert (
        session.session_state[app.STRATEGY_CREATION_PHASE_KEY]
        == app.STRATEGY_CREATION_PHASE_CODE_AUTHORIZED
    )

    session = session.button(key="strategy_creation_activate").click().run()
    assert not session.exception, [error.value for error in session.exception]
    assert any(item.value == "Crear estrategia desde cero" for item in session.subheader)
    transcript = " ".join(item.value for item in session.markdown)
    assert "Quiero una estrategia de tendencia." in transcript
    assert "¿Qué marco temporal prefieres?" in transcript
    assert session.button(key="strategy_creation_generate_code")


def test_creation_input_is_visible_and_keeps_draft_while_assistant_has_history():
    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    session = session.radio(key="workspace_tab").set_value("LABORATORIO").run()
    session.session_state[app.ASSISTANT_MODE_KEY] = app.ASSISTANT_MODE_CREATE_STRATEGY
    session.session_state[app.STRATEGY_CREATION_MESSAGES_KEY] = [
        {"role": "user", "content": "Quiero una estrategia de tendencia."},
        {"role": "assistant", "content": "¿Qué marco temporal prefieres?"},
    ]
    session = session.run()

    assert not session.exception, [error.value for error in session.exception]
    assert session.text_area(key=app.STRATEGY_CREATION_INPUT_KEY).label == "Instrucciones o prompt"
    assert session.button(key="strategy_creation_send").label == "Enviar mensaje"

    draft = "Usa velas diarias y una media móvil de 50 periodos."
    session = session.text_area(key=app.STRATEGY_CREATION_INPUT_KEY).set_value(draft).run()
    assert not session.exception, [error.value for error in session.exception]
    assert session.text_area(key=app.STRATEGY_CREATION_INPUT_KEY).value == draft
    assert session.session_state[app.STRATEGY_CREATION_MESSAGES_KEY][-1]["content"] == (
        "¿Qué marco temporal prefieres?"
    )


def test_empty_creation_form_submission_shows_warning_and_keeps_input():
    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    session = session.radio(key="workspace_tab").set_value("LABORATORIO").run()
    session.session_state[app.ASSISTANT_MODE_KEY] = app.ASSISTANT_MODE_CREATE_STRATEGY
    session = session.run()

    assert session.text_area(key=app.STRATEGY_CREATION_INPUT_KEY)
    session = session.button(key="strategy_creation_send").click().run()

    assert not session.exception, [error.value for error in session.exception]
    assert any("Escribe una idea o una pregunta" in item.value for item in session.warning)
    assert session.text_area(key=app.STRATEGY_CREATION_INPUT_KEY)
    assert session.session_state[app.STRATEGY_CREATION_MESSAGES_KEY] == []


def test_rule_approval_is_required_before_generation_is_exposed():
    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    session = session.radio(key="workspace_tab").set_value("LABORATORIO").run()
    session.session_state[app.ASSISTANT_MODE_KEY] = app.ASSISTANT_MODE_CREATE_STRATEGY
    proposal = (
        "Objetivo: Pendiente\nUniverso: Pendiente\nTemporalidades: Pendiente\n"
        "Datos: Pendiente\nEntradas: Pendiente\nSalidas: Pendiente\n"
        "Indicadores: Pendiente\nParámetros: Pendiente\nRiesgo: Pendiente\n"
        "Supuestos: Pendiente\nLimitaciones: Pendiente\n"
        "Condiciones de invalidación: Pendiente"
    )
    session.session_state[app.STRATEGY_CREATION_MESSAGES_KEY] = [
        {"role": "user", "content": "Diseña una estrategia genérica."},
        {"role": "assistant", "content": proposal},
    ]
    session.session_state[app.STRATEGY_CREATION_RULES_KEY] = {
        "proposal": proposal,
        "approved": False,
        "approved_text": None,
    }
    session.session_state[app.STRATEGY_CREATION_PROPOSAL_INPUT_KEY] = proposal
    session.session_state[app.STRATEGY_CREATION_PHASE_KEY] = (
        app.STRATEGY_CREATION_PHASE_PROPOSAL_PENDING
    )
    session = session.run()

    assert session.text_area(key=app.STRATEGY_CREATION_PROPOSAL_INPUT_KEY).value == proposal
    assert not any(
        button.key == "strategy_creation_generate_code" for button in session.button
    )

    session = session.button(key="strategy_creation_approve_rules").click().run()
    assert not session.exception, [error.value for error in session.exception]
    assert session.session_state[app.STRATEGY_CREATION_RULES_KEY]["approved"] is True
    assert (
        session.session_state[app.STRATEGY_CREATION_PHASE_KEY]
        == app.STRATEGY_CREATION_PHASE_CODE_AUTHORIZED
    )
    assert session.button(key="strategy_creation_generate_code")

    session = session.button(key="strategy_creation_revoke_approval").click().run()
    assert session.session_state[app.STRATEGY_CREATION_RULES_KEY]["approved"] is False
    assert session.session_state[app.STRATEGY_CREATION_RULES_KEY]["proposal"] is None
    assert (
        session.session_state[app.STRATEGY_CREATION_PHASE_KEY]
        == app.STRATEGY_CREATION_PHASE_DESIGN
    )


def test_editing_proposal_invalidates_approval_bound_to_previous_text():
    original = "Entradas: cruce alcista.\nSalidas: cruce bajista."
    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    session = session.radio(key="workspace_tab").set_value("LABORATORIO").run()
    session.session_state[app.ASSISTANT_MODE_KEY] = app.ASSISTANT_MODE_CREATE_STRATEGY
    session.session_state[app.STRATEGY_CREATION_PHASE_KEY] = (
        app.STRATEGY_CREATION_PHASE_CODE_AUTHORIZED
    )
    session.session_state[app.STRATEGY_CREATION_RULES_KEY] = {
        "proposal": original,
        "approved": True,
        "approved_text": original,
    }
    session.session_state[app.STRATEGY_CREATION_PROPOSAL_INPUT_KEY] = original
    session = session.run()

    revised = original + "\nRiesgo: máximo 1% por operación."
    session = session.text_area(
        key=app.STRATEGY_CREATION_PROPOSAL_INPUT_KEY
    ).set_value(revised).run()

    assert session.session_state[app.STRATEGY_CREATION_RULES_KEY]["proposal"] == revised
    assert session.session_state[app.STRATEGY_CREATION_RULES_KEY]["approved"] is False
    assert session.session_state[app.STRATEGY_CREATION_RULES_KEY]["approved_text"] is None
    assert (
        session.session_state[app.STRATEGY_CREATION_PHASE_KEY]
        == app.STRATEGY_CREATION_PHASE_PROPOSAL_PENDING
    )
    assert not any(
        button.key == "strategy_creation_generate_code" for button in session.button
    )


def test_consolidation_requests_structured_generic_rules(monkeypatch):
    state = {
        app.STRATEGY_CREATION_MESSAGES_KEY: [
            {"role": "user", "content": "Quiero operar rupturas de volatilidad."}
        ],
        app.STRATEGY_CREATION_PROCESSING_KEY: False,
        app.STRATEGY_CREATION_ERROR_KEY: None,
        app.STRATEGY_CREATION_NOTICE_KEY: None,
        app.STRATEGY_CREATION_RULES_KEY: {
            "proposal": None,
            "approved": False,
            "approved_text": None,
        },
        app.STRATEGY_CREATION_PHASE_KEY: app.STRATEGY_CREATION_PHASE_DESIGN,
        "hdr_provider": "Ollama (local)",
        "hdr_model_local": "qwen2.5-coder",
    }
    response = (
        "Objetivo y universo: Pendiente\nTemporalidades y datos: Pendiente\n"
        "Entradas: ruptura de volatilidad\nSalidas: Pendiente\n"
        "Indicadores: Pendiente\nParámetros: Pendiente\n"
        "Capital y riesgo: Pendiente\nSupuestos: Pendiente\n"
        "Limitaciones: Pendiente\nInvalidación: Pendiente\nDatos pendientes: Pendiente"
    )
    captured = {}

    def fake_ai_request(prompt, config, system_prompt):
        captured.update(prompt=prompt, config=config, system_prompt=system_prompt)
        return response

    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr(app, "run_ai_request", fake_ai_request)
    app.process_strategy_creation_message(
        "Consolida las reglas de estrategia discutidas hasta ahora.",
        consolidate_rules=True,
    )

    assert captured["system_prompt"] == app.STRATEGY_CREATION_CONSOLIDATION_PROMPT
    assert all(
        section in captured["system_prompt"]
        for section in (
            "objetivo y universo",
            "temporalidades y",
            "condiciones de entrada y salida",
            "indicadores y parámetros",
            "gestión",
            "supuestos",
            "limitaciones",
            "condiciones de invalidación",
            "datos pendientes",
        )
    )
    assert "rupturas de volatilidad" in captured["prompt"]
    assert "escalones_dinamicos.py" not in captured["prompt"]
    assert "escalones_dinamicos.py" not in captured["system_prompt"]
    assert state[app.STRATEGY_CREATION_RULES_KEY]["proposal"] == response
    assert state[app.STRATEGY_CREATION_RULES_KEY]["approved"] is False
    assert state[app.STRATEGY_CREATION_PHASE_KEY] == (
        app.STRATEGY_CREATION_PHASE_PROPOSAL_PENDING
    )
    assert state[app.STRATEGY_CREATION_PROPOSAL_INPUT_KEY] == response
    assert state[app.STRATEGY_CREATION_MESSAGES_KEY][-1] == {
        "role": "assistant",
        "content": response,
    }


def test_continuing_conversation_invalidates_approved_rules_and_preserves_history(monkeypatch):
    approved_text = "Entradas: cruce alcista. Salidas: cruce bajista."
    state = {
        app.STRATEGY_CREATION_MESSAGES_KEY: [
            {"role": "user", "content": "Diseña una estrategia."},
            {"role": "assistant", "content": approved_text},
        ],
        app.STRATEGY_CREATION_PROCESSING_KEY: False,
        app.STRATEGY_CREATION_ERROR_KEY: None,
        app.STRATEGY_CREATION_NOTICE_KEY: None,
        app.STRATEGY_CREATION_PHASE_KEY: app.STRATEGY_CREATION_PHASE_CODE_AUTHORIZED,
        app.STRATEGY_CREATION_RULES_KEY: {
            "proposal": approved_text,
            "approved": True,
            "approved_text": approved_text,
        },
        "hdr_provider": "Ollama (local)",
        "hdr_model_local": "qwen2.5-coder",
    }
    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr(app, "run_ai_request", lambda *_args: "Añade un filtro de volatilidad.")

    app.process_strategy_creation_message("Quiero añadir una condición más.")

    assert state[app.STRATEGY_CREATION_MESSAGES_KEY][-2] == {
        "role": "user",
        "content": "Quiero añadir una condición más.",
    }
    assert state[app.STRATEGY_CREATION_MESSAGES_KEY][-1] == {
        "role": "assistant",
        "content": "Añade un filtro de volatilidad.",
    }
    assert state[app.STRATEGY_CREATION_PHASE_KEY] == app.STRATEGY_CREATION_PHASE_DESIGN
    assert state[app.STRATEGY_CREATION_RULES_KEY]["approved"] is False
    assert state[app.STRATEGY_CREATION_RULES_KEY]["approved_text"] is None
    assert state[app.STRATEGY_CREATION_RULES_KEY]["proposal"] is None


def test_new_conversation_marks_generated_code_stale_without_discarding_it(monkeypatch):
    old_code = VECTOR_STRATEGY
    state = {
        app.STRATEGY_CREATION_MESSAGES_KEY: [
            {"role": "assistant", "content": "Reglas aprobadas."}
        ],
        app.STRATEGY_CREATION_PROCESSING_KEY: False,
        app.STRATEGY_CREATION_ERROR_KEY: None,
        app.STRATEGY_CREATION_NOTICE_KEY: None,
        app.STRATEGY_CREATION_PHASE_KEY: app.STRATEGY_CREATION_PHASE_CODE_PENDING_VALIDATION,
        app.STRATEGY_CREATION_CODE_KEY: old_code,
        app.STRATEGY_CREATION_RULES_KEY: {
            "proposal": "Reglas aprobadas.",
            "approved": True,
            "approved_text": "Reglas aprobadas.",
        },
        "hdr_provider": "Ollama (local)",
        "hdr_model_local": "qwen2.5-coder",
    }
    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr(app, "run_ai_request", lambda *_args: "Añade una salida por volatilidad.")

    app.process_strategy_creation_message("Cambia las condiciones de salida.")

    assert state[app.STRATEGY_CREATION_PHASE_KEY] == app.STRATEGY_CREATION_PHASE_DESIGN
    assert state[app.STRATEGY_CREATION_CODE_STALE_KEY] is True
    assert state[app.STRATEGY_CREATION_CODE_KEY] == old_code
    assert state[app.STRATEGY_CREATION_RULES_KEY]["approved"] is False


def test_inconsistent_authorized_phase_does_not_expose_or_allow_generation(monkeypatch):
    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    session = session.radio(key="workspace_tab").set_value("LABORATORIO").run()
    session.session_state[app.ASSISTANT_MODE_KEY] = app.ASSISTANT_MODE_CREATE_STRATEGY
    session.session_state[app.STRATEGY_CREATION_PHASE_KEY] = (
        app.STRATEGY_CREATION_PHASE_CODE_AUTHORIZED
    )
    session.session_state[app.STRATEGY_CREATION_RULES_KEY] = {
        "proposal": "Reglas no aprobadas.",
        "approved": False,
    }
    session = session.run()
    assert not any(
        button.key == "strategy_creation_generate_code" for button in session.button
    )
    assert any("no hay una propuesta de reglas aprobada válida" in item.value for item in session.warning)

    state = {
        app.STRATEGY_CREATION_MESSAGES_KEY: [],
        app.STRATEGY_CREATION_PROCESSING_KEY: False,
        app.STRATEGY_CREATION_ERROR_KEY: None,
        app.STRATEGY_CREATION_NOTICE_KEY: None,
        app.STRATEGY_CREATION_PHASE_KEY: app.STRATEGY_CREATION_PHASE_CODE_AUTHORIZED,
        app.STRATEGY_CREATION_RULES_KEY: {
            "proposal": "Reglas no aprobadas.",
            "approved": False,
            "approved_text": None,
        },
    }
    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr(
        app,
        "run_ai_request",
        lambda *_args: (_ for _ in ()).throw(AssertionError("AI must not be called")),
    )
    app.process_strategy_creation_message("Genera el código", generate_code=True)
    assert state[app.STRATEGY_CREATION_MESSAGES_KEY] == []
    assert "Aprueba primero" in state[app.STRATEGY_CREATION_NOTICE_KEY]


VECTOR_STRATEGY = """def run_strategy(data):
    data["Position"] = 1
    return data
"""


def test_generated_vector_strategy_is_validated_and_retained_without_execution(
    monkeypatch,
    tmp_path,
):
    marker = tmp_path / "must_not_execute.txt"
    code = (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        + VECTOR_STRATEGY
    )
    response = (
        "Archivo sugerido: ruptura_volatilidad.py\n"
        f"```python\n{code}```"
    )
    state = {
        app.STRATEGY_CREATION_MESSAGES_KEY: [
            {"role": "user", "content": "Estrategia de ruptura de volatilidad."}
        ],
        app.STRATEGY_CREATION_PROCESSING_KEY: False,
        app.STRATEGY_CREATION_ERROR_KEY: None,
        app.STRATEGY_CREATION_NOTICE_KEY: None,
        app.STRATEGY_CREATION_PHASE_KEY: app.STRATEGY_CREATION_PHASE_CODE_AUTHORIZED,
        app.STRATEGY_CREATION_CODE_KEY: None,
        app.STRATEGY_CREATION_RULES_KEY: {
            "proposal": "Entrar ante ruptura de volatilidad.",
            "approved": True,
            "approved_text": "Entrar ante ruptura de volatilidad.",
        },
        "hdr_provider": "Ollama (local)",
        "hdr_model_local": "qwen2.5-coder",
    }
    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr(app, "run_ai_request", lambda *_args: response)
    monkeypatch.setattr(
        app,
        "ejecutar_estrategia",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("generation must not run the strategy")
        ),
    )

    app.process_strategy_creation_message("Genera el código.", generate_code=True)

    assert state[app.STRATEGY_CREATION_PHASE_KEY] == (
        app.STRATEGY_CREATION_PHASE_CODE_PENDING_VALIDATION
    )
    assert state[app.STRATEGY_CREATION_CODE_KEY] == code
    assert state[app.STRATEGY_CREATION_CODE_EDITOR_KEY] == code
    assert state[app.STRATEGY_CREATION_FILENAME_KEY] == "ruptura_volatilidad.py"
    assert state[app.STRATEGY_CREATION_VALIDATION_KEY]["valid"] is True
    assert state[app.STRATEGY_CREATION_VALIDATION_KEY]["dialect"] == "vector"
    assert state["codigo_actual"] == code
    assert state[app.STRATEGY_CREATION_MESSAGES_KEY][-1]["content"] == response
    assert not marker.exists()


def test_static_validation_rejects_python_without_a_supported_strategy_entry_point():
    result = app.validate_strategy_source("def simulate(data):\n    return data\n")

    assert result["valid"] is False
    assert result["dialect"] == app.strategy_parser.DIALECTO_NINGUNO
    assert "run_strategy" in result["errors"][0]


def test_generated_file_save_is_exact_exclusive_and_registered(monkeypatch, tmp_path):
    state = {}
    available = []
    marker = tmp_path / "must_not_run_during_save.txt"
    source = (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        + VECTOR_STRATEGY
    )
    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr(app, "py_files", [])

    relative = app.save_generated_strategy_from_state(
        "ruptura_volatilidad.py", source, tmp_path, available
    )

    assert relative == "ruptura_volatilidad.py"
    assert (tmp_path / relative).read_text(encoding="utf-8") == source
    assert relative in available
    assert "ruptura_volatilidad.py" in app.py_files
    assert relative in app.list_all_strategies(tmp_path)
    assert state[app.STRATEGY_CREATION_SAVED_PATH_KEY] == relative
    assert state[app.STRATEGY_CREATION_SAVED_CODE_KEY] == source
    assert not marker.exists()

    with pytest.raises(FileExistsError):
        app.save_generated_strategy_from_state(
            "ruptura_volatilidad.py", "def replaced(): pass\n", tmp_path, available
        )
    assert (tmp_path / relative).read_text(encoding="utf-8") == source


def test_static_validation_supports_real_lean_entry_point_and_checks_dependencies():
    lean_source = """from AlgorithmImports import *


class TrendAlgorithm(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2020, 1, 1)

    def on_data(self, data):
        return
"""
    result = app.validate_strategy_source(lean_source, "trend.py")

    assert result["valid"] is True
    assert result["dialect"] == "lean"

    missing_dependency = app.validate_strategy_source(
        "import dependency_that_does_not_exist_12345\n" + VECTOR_STRATEGY,
        "vector.py",
    )
    assert missing_dependency["valid"] is False
    assert any("dependency_that_does_not_exist_12345" in error for error in missing_dependency["errors"])


def test_save_write_error_preserves_generated_editor_content(monkeypatch, tmp_path):
    marker = tmp_path / "write_error.py"
    original_open = Path.open
    state = {
        app.STRATEGY_CREATION_CODE_KEY: VECTOR_STRATEGY,
        app.STRATEGY_CREATION_CODE_EDITOR_KEY: VECTOR_STRATEGY,
    }
    monkeypatch.setattr(app.st, "session_state", state)

    def fail_target_open(path, *args, **kwargs):
        if path == marker:
            raise PermissionError("access denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_target_open)
    with pytest.raises(PermissionError, match="access denied"):
        app.save_generated_strategy_from_state(
            marker.name, VECTOR_STRATEGY, tmp_path, []
        )

    assert not marker.exists()
    assert state[app.STRATEGY_CREATION_CODE_EDITOR_KEY] == VECTOR_STRATEGY
    assert state[app.STRATEGY_CREATION_CODE_KEY] == VECTOR_STRATEGY


def test_saved_strategy_can_be_promoted_through_existing_principal_mechanism(
    monkeypatch, tmp_path
):
    strategy = tmp_path / "creada_por_ia.py"
    strategy.write_text(VECTOR_STRATEGY, encoding="utf-8")
    state = {
        app.PRINCIPAL_KEY: None,
        app.COMPARE_KEY: [],
        app.COMPARE_CACHE_KEY: {},
    }
    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr(app, "BASE_DIR", tmp_path)
    monkeypatch.setattr(app, "py_files", [])

    app.set_principal_strategy("creada_por_ia.py", ["creada_por_ia.py"])

    assert state[app.PRINCIPAL_KEY] == "creada_por_ia.py"
    assert state["script_nombre"] == "creada_por_ia.py"
    assert state["codigo_actual"] == VECTOR_STRATEGY
    assert "creada_por_ia.py" in app.py_files


def test_saved_vector_strategy_reaches_existing_backtest_runner(monkeypatch, tmp_path):
    (tmp_path / "generada.py").write_text(VECTOR_STRATEGY, encoding="utf-8")
    monkeypatch.setattr(app, "BASE_DIR", tmp_path)

    dataframe, metrics = app.run_strategy_backtest("generada.py", "SPY", 100_000.0)

    assert dataframe is not None
    assert "error" not in metrics
    assert metrics["ticker"] == "SPY"
    assert metrics["bars"] > 0


def test_backtest_runner_error_is_reported_without_success_state(monkeypatch):
    monkeypatch.setitem(app.st.session_state, app.PRINCIPAL_KEY, "broken.py")
    monkeypatch.delitem(app.st.session_state, app.RESULT_KEY, raising=False)
    monkeypatch.delitem(app.st.session_state, app.PARAMS_KEY, raising=False)
    monkeypatch.setattr(
        app,
        "run_strategy_backtest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("motor no disponible")),
    )

    _dataframe, metrics = app.get_backtest_result("SPY", 100_000.0)

    assert metrics["error"] == "motor no disponible"
    assert metrics["source"] == "sin ejecutar"


def test_generated_code_review_persists_in_assistant_and_download_is_available():
    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    session = session.radio(key="workspace_tab").set_value("LABORATORIO").run()
    session.session_state[app.ASSISTANT_MODE_KEY] = app.ASSISTANT_MODE_CREATE_STRATEGY
    session.session_state[app.STRATEGY_CREATION_PHASE_KEY] = (
        app.STRATEGY_CREATION_PHASE_CODE_PENDING_VALIDATION
    )
    session.session_state[app.STRATEGY_CREATION_CODE_KEY] = VECTOR_STRATEGY
    session.session_state[app.STRATEGY_CREATION_CODE_EDITOR_KEY] = VECTOR_STRATEGY
    session.session_state[app.STRATEGY_CREATION_FILENAME_KEY] = "estrategia_vectorial.py"
    session.session_state[app.STRATEGY_CREATION_VALIDATION_KEY] = (
        app.validate_strategy_source(VECTOR_STRATEGY, "estrategia_vectorial.py")
    )
    session = session.run()

    assert not session.exception, [error.value for error in session.exception]
    assert session.text_area(key=app.STRATEGY_CREATION_CODE_EDITOR_KEY).value == VECTOR_STRATEGY
    assert session.get("download_button")
    assert session.button(key="strategy_creation_save_file").disabled is False


def test_python_only_response_is_not_auto_approved_as_rules():
    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    session = session.radio(key="workspace_tab").set_value("LABORATORIO").run()
    session.session_state[app.ASSISTANT_MODE_KEY] = app.ASSISTANT_MODE_CREATE_STRATEGY
    session.session_state[app.STRATEGY_CREATION_MESSAGES_KEY] = [
        {"role": "assistant", "content": "```python\ndef simulate():\n    pass\n```"}
    ]
    session = session.run()

    assert session.button(key="strategy_creation_consolidate_rules")
    assert not any(
        button.key == "strategy_creation_approve_rules" for button in session.button
    )
    assert session.session_state[app.STRATEGY_CREATION_PHASE_KEY] == (
        app.STRATEGY_CREATION_PHASE_DESIGN
    )
    assert any("no constituye reglas aprobadas" in item.value for item in session.caption)


def test_unapproved_python_response_is_preserved_without_authorizing_generation(monkeypatch):
    code_reply = (
        "Versión preliminar:\n"
        "```python\n"
        "def simulate(data):\n"
        "    return data\n"
        "```"
    )
    state = {
        app.STRATEGY_CREATION_MESSAGES_KEY: [
            {"role": "user", "content": "Diseña reglas para escalones."}
        ],
        app.STRATEGY_CREATION_PROCESSING_KEY: False,
        app.STRATEGY_CREATION_ERROR_KEY: None,
        app.STRATEGY_CREATION_NOTICE_KEY: None,
        app.STRATEGY_CREATION_PHASE_KEY: app.STRATEGY_CREATION_PHASE_DESIGN,
        app.STRATEGY_CREATION_CODE_KEY: None,
        app.STRATEGY_CREATION_RULES_KEY: {"proposal": None, "approved": False},
        "hdr_provider": "Ollama (local)",
        "hdr_model_local": "qwen2.5-coder",
    }
    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr(app, "run_ai_request", lambda *_args: code_reply)

    app.process_strategy_creation_message("Pregunta de diseño")

    assert state[app.STRATEGY_CREATION_MESSAGES_KEY][-1] == {
        "role": "assistant",
        "content": code_reply,
    }
    assert state[app.STRATEGY_CREATION_PHASE_KEY] == app.STRATEGY_CREATION_PHASE_DESIGN
    assert state[app.STRATEGY_CREATION_CODE_KEY] is None
    assert state[app.STRATEGY_CREATION_ERROR_KEY] is None
    assert "se conservó en el historial" in state[app.STRATEGY_CREATION_NOTICE_KEY]


def test_python_response_in_history_is_rendered_in_creation_chat():
    code_reply = "```python\ndef simulate(data):\n    return data\n```"
    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    session = session.radio(key="workspace_tab").set_value("LABORATORIO").run()
    session.session_state[app.ASSISTANT_MODE_KEY] = app.ASSISTANT_MODE_CREATE_STRATEGY
    session.session_state[app.STRATEGY_CREATION_MESSAGES_KEY] = [
        {"role": "assistant", "content": code_reply}
    ]
    session.session_state[app.STRATEGY_CREATION_PHASE_KEY] = (
        app.STRATEGY_CREATION_PHASE_CODE_PENDING_VALIDATION
    )
    session.session_state[app.STRATEGY_CREATION_CODE_KEY] = (
        "def simulate(data):\n    return data"
    )
    session = session.run()

    assert not session.exception, [error.value for error in session.exception]
    assert any("def simulate(data)" in item.value for item in session.markdown)
    assert any("pendiente de validación" in item.value for item in session.info)


def test_authorized_code_response_is_retained_pending_validation_only(monkeypatch):
    code_reply = (
        "Implementación preliminar:\n"
        "```python\n"
        "def simulate(data):\n"
        "    return data\n"
        "```"
    )
    state = {
        app.STRATEGY_CREATION_MESSAGES_KEY: [
            {"role": "assistant", "content": "Reglas aprobadas."}
        ],
        app.STRATEGY_CREATION_PROCESSING_KEY: False,
        app.STRATEGY_CREATION_ERROR_KEY: None,
        app.STRATEGY_CREATION_NOTICE_KEY: None,
        app.STRATEGY_CREATION_PHASE_KEY: app.STRATEGY_CREATION_PHASE_CODE_AUTHORIZED,
        app.STRATEGY_CREATION_CODE_KEY: None,
        app.STRATEGY_CREATION_RULES_KEY: {
            "proposal": "Reglas aprobadas para la simulación.",
            "approved": True,
            "approved_text": "Reglas aprobadas para la simulación.",
        },
        "hdr_provider": "Ollama (local)",
        "hdr_model_local": "qwen2.5-coder",
    }
    requests = []

    def fake_ai_request(prompt, config, system_prompt):
        requests.append((prompt, config, system_prompt))
        return code_reply

    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr(app, "run_ai_request", fake_ai_request)

    app.process_strategy_creation_message(
        "Genera el archivo", generate_code=True
    )

    assert state[app.STRATEGY_CREATION_MESSAGES_KEY][-1] == {
        "role": "assistant",
        "content": code_reply,
    }
    assert state[app.STRATEGY_CREATION_PHASE_KEY] == (
        app.STRATEGY_CREATION_PHASE_CODE_PENDING_VALIDATION
    )
    assert state[app.STRATEGY_CREATION_CODE_KEY] == (
        "def simulate(data):\n    return data\n"
    )
    assert requests[0][2] == app.STRATEGY_CREATION_CODE_SYSTEM_PROMPT
    assert "Reglas aprobadas para la simulación." in requests[0][0]
    assert "escalones_dinamicos.py" not in requests[0][2]
    assert "nombre de archivo descriptivo" in requests[0][2]
    assert state[app.STRATEGY_CREATION_VALIDATION_KEY]["valid"] is False
    assert "no se reconoce como una estrategia completa compatible" in state[
        app.STRATEGY_CREATION_NOTICE_KEY
    ]
    assert "codigo_actual" not in state
    assert "ai_result" not in state


def test_code_generation_cannot_call_ai_before_rules_are_approved(monkeypatch):
    state = {
        app.STRATEGY_CREATION_MESSAGES_KEY: [],
        app.STRATEGY_CREATION_PROCESSING_KEY: False,
        app.STRATEGY_CREATION_ERROR_KEY: None,
        app.STRATEGY_CREATION_NOTICE_KEY: None,
        app.STRATEGY_CREATION_PHASE_KEY: app.STRATEGY_CREATION_PHASE_DESIGN,
        app.STRATEGY_CREATION_RULES_KEY: {"proposal": None, "approved": False},
    }
    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr(
        app,
        "run_ai_request",
        lambda *_args: (_ for _ in ()).throw(AssertionError("AI must not be called")),
    )

    app.process_strategy_creation_message("Genera el código", generate_code=True)

    assert state[app.STRATEGY_CREATION_MESSAGES_KEY] == []
    assert state[app.STRATEGY_CREATION_PHASE_KEY] == app.STRATEGY_CREATION_PHASE_DESIGN
    assert "Aprueba primero" in state[app.STRATEGY_CREATION_NOTICE_KEY]


def test_creation_assistant_remains_available_when_navigating_tabs():
    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    session = session.radio(key="workspace_tab").set_value("LABORATORIO").run()
    session = session.button(key="strategy_creation_activate").click().run()

    for tab in app.WORKSPACE_TABS:
        session = session.radio(key="workspace_tab").set_value(tab).run()
        assert not session.exception, [error.value for error in session.exception]
        assert (
            session.session_state[app.ASSISTANT_MODE_KEY]
            == app.ASSISTANT_MODE_CREATE_STRATEGY
        )
        assert any(
            item.value == "Crear estrategia desde cero" for item in session.subheader
        )


def test_creation_request_uses_canonical_ai_configuration(monkeypatch):
    state = {
        app.STRATEGY_CREATION_MESSAGES_KEY: [],
        app.STRATEGY_CREATION_PROCESSING_KEY: False,
        app.STRATEGY_CREATION_ERROR_KEY: None,
        "hdr_provider": "Ollama (local)",
        "hdr_model_local": "qwen2.5-coder",
        "hdr_ollama_host": "http://localhost:11434",
    }
    request = {}

    responses = iter(
        ["¿Qué marco temporal te interesa?", "¿Qué nivel de riesgo aceptarías?"]
    )

    def fake_ai_request(prompt, config, system_prompt):
        request.setdefault("calls", []).append(
            {"prompt": prompt, "config": config, "system_prompt": system_prompt}
        )
        return next(responses)

    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr(app, "run_ai_request", fake_ai_request)

    app.process_strategy_creation_message("Quiero operar acciones de tecnología.")

    expected_config = {
        "provider": "Ollama (local)",
        "model": "qwen2.5-coder",
        "api_key": "",
        "host": "http://localhost:11434",
    }
    app.process_strategy_creation_message("Prefiero velas diarias.")
    assert len(request["calls"]) == 2
    assert all(call["config"] == expected_config for call in request["calls"])
    assert all(
        call["system_prompt"] == app.STRATEGY_CREATION_SYSTEM_PROMPT
        for call in request["calls"]
    )
    assert "Quiero operar acciones de tecnología." in request["calls"][0]["prompt"]
    assert "Prefiero velas diarias." in request["calls"][1]["prompt"]
    assert state[app.STRATEGY_CREATION_MESSAGES_KEY] == [
        {"role": "user", "content": "Quiero operar acciones de tecnología."},
        {"role": "assistant", "content": "¿Qué marco temporal te interesa?"},
        {"role": "user", "content": "Prefiero velas diarias."},
        {"role": "assistant", "content": "¿Qué nivel de riesgo aceptarías?"},
    ]
    assert state[app.STRATEGY_CREATION_PROCESSING_KEY] is False
    assert state[app.STRATEGY_CREATION_ERROR_KEY] is None
