import base64
import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app


@pytest.fixture
def picker_project(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "BASE_DIR", tmp_path)
    monkeypatch.setattr(app, "ALL_STRATEGIES", ["alpha.py", "beta.py"])
    monkeypatch.setattr(app.st, "session_state", {})
    return tmp_path


def write_strategy(base: Path, name: str, source: str) -> Path:
    path = base / name
    path.write_text(source, encoding="utf-8")
    return path


def test_open_picker_starts_empty_and_does_not_replace_current_strategy():
    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    session = session.radio(key="workspace_tab").set_value("LABORATORIO").run()
    current = "grid_trading_v2.py"
    session.session_state[app.PRINCIPAL_KEY] = current
    session.session_state[app.BACKTEST_PENDING_KEY] = current
    session = session.run()
    session = session.button(key="btn_prueba_abrir_estrategia").click().run()

    assert not session.exception, [error.value for error in session.exception]
    assert session.session_state[app.PRINCIPAL_KEY] == current
    assert any(
        "Selecciona una estrategia para ver su análisis" in item.value
        for item in session.caption
    )
    assert not session.file_uploader
    assert not any(item.key == "modal_load" for item in session.button)
    component_dir = Path(app.__file__).parent / "ui_components" / "strategy_file_picker"
    assert (component_dir / "index.html").is_file()
    assert (component_dir / "picker.js").is_file()


def test_picker_actions_share_one_aligned_footer_without_duplicate_upload():
    component_dir = Path(app.__file__).parent / "ui_components" / "strategy_file_picker"
    script = (component_dir / "picker.js").read_text(encoding="utf-8")
    markup = (component_dir / "index.html").read_text(encoding="utf-8")

    assert '<div class="actions">' in script
    assert script.index('id="search-button"') < script.index('id="cancel-button"')
    assert script.index('id="cancel-button"') < script.index('id="load-button"')
    assert 'class="dropzone"' in script
    assert 'height: 40px;' in markup
    assert 'flex: 1 1 0;' in markup
    assert "Upload" not in script


def file_payload(filename: str, source: bytes) -> dict:
    return {
        "name": filename,
        "content_base64": base64.b64encode(source).decode("ascii"),
    }


def test_browser_file_selection_refreshes_catalog_preview_and_clears_old_error(
    picker_project,
):
    source = b"def run_strategy(data):\n    return data\n"
    write_strategy(picker_project, "alpha.py", source.decode("utf-8"))
    app.st.session_state[app.MODAL_ERROR_KEY] = "obsolete error"

    assert app.process_strategy_picker_file(file_payload("alpha.py", source))

    assert app.st.session_state[app.MODAL_SELECTED_SOURCE_KEY] == source.decode()
    assert app.st.session_state[app.MODAL_SELECTED_PATH_KEY] == "alpha.py"
    assert app.st.session_state[app.MODAL_PROVENANCE_KEY] == "catalog"
    assert app.MODAL_ERROR_KEY not in app.st.session_state


def test_external_python_file_is_previewed_but_not_marked_as_catalog(picker_project):
    source = b"def run_strategy(data):\n    return data\n# section three\n"

    assert app.process_strategy_picker_file(file_payload("outside.py", source))

    assert app.st.session_state[app.MODAL_SELECTED_SOURCE_KEY] == source.decode()
    assert app.st.session_state[app.MODAL_SELECTED_PATH_KEY] is None
    assert app.st.session_state[app.MODAL_PROVENANCE_KEY] == "external"
    assert app.analyze_strategy_source(
        app.st.session_state[app.MODAL_SELECTED_SOURCE_KEY], "outside.py"
    )["valid"]
    assert app.MODAL_ERROR_KEY not in app.st.session_state


def test_catalog_strategy_can_be_loaded_only_after_selection(picker_project):
    source = "def run_strategy(data):\n    return data\n"
    write_strategy(picker_project, "catalog.py", source)
    monkeypatch_available = ["catalog.py"]

    assert app.select_strategy_for_review("catalog.py")
    assert app.st.session_state[app.MODAL_PROVENANCE_KEY] == "catalog"
    review = app.analyze_strategy_source(source, "catalog.py", "catalog.py")
    assert review["syntax_valid"]
    assert review["execution_validation"]["valid"]

    app.set_principal_strategy("catalog.py", monkeypatch_available, source=source)

    assert app.st.session_state[app.PRINCIPAL_KEY] == "catalog.py"
    assert app.read_script_source("catalog.py") == source


def test_external_strategy_is_editor_copy_and_save_as_never_overwrites_original(
    picker_project,
):
    source = "def run_strategy(data):\n    return data\n"
    original = write_strategy(picker_project.parent, "external_alpha.py", source)
    original_bytes = original.read_bytes()
    assert app.process_strategy_picker_file(
        file_payload(original.name, original_bytes)
    )
    assert app.st.session_state[app.MODAL_PROVENANCE_KEY] == "external"
    preview = app.st.session_state[app.MODAL_SELECTED_SOURCE_KEY]
    review = app.analyze_strategy_source(preview, original.name)
    assert review["syntax_valid"] and review["execution_validation"]["valid"]

    external_id = app.create_external_strategy(original.name, preview)
    app.set_principal_strategy(external_id, [], source=preview)
    editor_key = f"editor_{external_id}"
    edited = source + "# editor-only change\n"
    app.st.session_state[editor_key] = edited
    app.persist_editor_code(external_id, editor_key)

    assert app.st.session_state[app.PRINCIPAL_KEY] == external_id
    assert app.strategy_display_name(external_id) == original.name
    assert not app.editor_save_button_disabled(external_id, edited)
    app.save_editor_source(external_id, edited)
    assert app.editor_sync_status(external_id, edited)[0] == "saved"
    assert original.read_bytes() == original_bytes

    copied = app.save_strategy_source(picker_project, "external_copy.py", edited)
    app.register_strategy_file(copied, picker_project, [])
    assert copied.read_text(encoding="utf-8") == edited
    assert original.read_bytes() == original_bytes
    collision = write_strategy(picker_project, "existing.py", "keep this file\n")
    with pytest.raises(FileExistsError):
        app.save_strategy_source(picker_project, collision.name, edited)
    assert collision.read_text(encoding="utf-8") == "keep this file\n"


def test_external_source_metadata_does_not_require_a_catalog_path(
    picker_project, monkeypatch
):
    source = "symbol = 'AAPL'\ndef run_strategy(data):\n    return data\n"
    info = app.strategy_parser.analizar_fuente(source, nombre="outside.py")
    monkeypatch.setattr(app, "available_tickers", lambda: ["AAPL"])

    ticker, capital = app.reglas_comunes_de_estrategia(
        picker_project / "external.py", info, source=source
    )

    assert ticker == "AAPL"
    assert capital == info.capital


def test_external_strategy_can_be_added_as_comparison_only_on_confirmation(
    picker_project,
):
    app.st.session_state[app.PRINCIPAL_KEY] = "catalog.py"
    external_id = app.create_external_strategy(
        "comparison.py", "def run_strategy(data):\n    return data\n"
    )

    accepted, _message = app.add_comparative_strategy(external_id, ["catalog.py"])

    assert accepted
    assert app.st.session_state[app.COMPARE_KEY] == [external_id]
    assert app.st.session_state[app.PRINCIPAL_KEY] == "catalog.py"


def test_python_without_execution_contract_has_specific_nonblocking_analysis(
    picker_project,
):
    source = "answer = 42\n"
    assert app.process_strategy_picker_file(file_payload("notes.py", source.encode()))

    review = app.analyze_strategy_source(source, "notes.py")

    assert review["syntax_valid"]
    assert not review["execution_validation"]["valid"]
    assert review["execution_validation"]["errors"]
    assert review["error"]


def test_cancel_external_load_preserves_active_strategy_and_editor(picker_project):
    current = "def run_strategy(data):\n    return data\n"
    active_path = write_strategy(picker_project, "active.py", current)
    app.st.session_state.update(
        {
            app.PRINCIPAL_KEY: "active.py",
            "editor_active.py": current + "# pending edit\n",
            "codigo_actual_script": "active.py",
            "codigo_actual": current + "# pending edit\n",
        }
    )
    incoming = "def run_strategy(data):\n    return data * 2\n"
    external_id = app.create_external_strategy("new_external.py", incoming)
    digest = app.st.session_state[app.EXTERNAL_STRATEGIES_KEY][external_id]["sha256"]

    assert app.request_editor_navigation(
        "replace",
        external_id,
        target_source=incoming,
        target_sha256=digest,
    )
    assert app.resolve_editor_navigation("cancel", ["active.py"]) == "cancelled"

    assert app.st.session_state[app.PRINCIPAL_KEY] == "active.py"
    assert app.st.session_state["editor_active.py"] == current + "# pending edit\n"
    assert active_path.read_text(encoding="utf-8") == current


@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        (file_payload("notes.txt", b"not python"), "Solo se admiten"),
        (file_payload("broken.py", b"\xff\xfe"), "No se pudo validar"),
        ({"name": "unreadable.py", "read_error": "sin acceso"}, "No se pudo leer"),
    ],
)
def test_invalid_or_unreadable_browser_file_clears_stale_preview(
    picker_project, payload, expected_message
):
    app.st.session_state.update(
        {
            app.MODAL_SELECTED_SOURCE_KEY: "old preview",
            app.MODAL_SELECTED_PATH_KEY: "old.py",
            app.MODAL_ERROR_KEY: "obsolete error",
        }
    )

    assert app.process_strategy_picker_file(payload) is False

    assert app.MODAL_SELECTED_SOURCE_KEY not in app.st.session_state
    assert app.MODAL_SELECTED_PATH_KEY not in app.st.session_state
    assert expected_message in app.st.session_state[app.MODAL_ERROR_KEY]
    assert app.st.session_state[app.MODAL_VIEW_KEY] == "empty"


def test_read_candidate_hashes_and_rejects_missing_or_out_of_scope_files(picker_project):
    source = "def run_strategy(data):\n    return data\n"
    path = write_strategy(picker_project, "alpha.py", source)

    loaded, digest = app.read_strategy_candidate("alpha.py")

    assert loaded == source
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileNotFoundError):
        app.read_strategy_candidate("missing.py")
    with pytest.raises((ValueError, FileNotFoundError)):
        app.read_strategy_candidate("..\\outside.py")


def test_filter_and_select_another_file_refresh_the_analysis_without_loading(
    picker_project,
):
    source_a = 'class Alpha(QCAlgorithm):\n    """Alpha doc."""\n    def initialize(self):\n        pass\n'
    source_b = "def run_strategy(data):\n    return data\n"
    write_strategy(picker_project, "alpha.py", source_a)
    write_strategy(picker_project, "beta.py", source_b)
    app.st.session_state[app.PRINCIPAL_KEY] = "keep.py"

    assert app.filter_strategy_candidates(
        ["alpha.py", "folder/beta.py"], "BETA"
    ) == ["folder/beta.py"]
    assert app.select_strategy_for_review("alpha.py")
    review_a = app.analyze_strategy_source(
        app.st.session_state[app.MODAL_SELECTED_SOURCE_KEY], "alpha.py", "alpha.py"
    )
    assert review_a["dialect"] == "QuantConnect Lean"
    assert review_a["description"] == "Alpha doc."

    assert app.select_strategy_for_review("beta.py")
    review_b = app.analyze_strategy_source(
        app.st.session_state[app.MODAL_SELECTED_SOURCE_KEY], "beta.py", "beta.py"
    )
    assert review_b["dialect"] == "Vectorial (`run_strategy`)"
    assert app.st.session_state[app.MODAL_SELECTED_PATH_KEY] == "beta.py"
    assert app.st.session_state[app.PRINCIPAL_KEY] == "keep.py"


VALID_STATUSES = {"evaluated", "partial", "not_evaluable", "unverified"}
EVALUATION_KEYS = {"value", "status", "reason", "findings", "display"}


def test_analysis_reports_structural_evaluations_without_fabricating_scores(
    picker_project,
):
    source = "def run_strategy(data):\n    return data\n"
    review = app.analyze_strategy_source(source, "beta.py", "beta.py")

    assert review["valid"] is True
    evaluations = review["evaluations"]
    expected_labels = {
        app.REVIEW_GENERAL_LABEL,
        *app.REVIEW_DIMENSION_LABELS,
    }
    assert set(evaluations) == expected_labels
    for evaluation in evaluations.values():
        assert set(evaluation) == EVALUATION_KEYS
        assert evaluation["status"] in VALID_STATUSES
        assert evaluation["reason"]

    # `scores` es solo la vista de presentación de `evaluations`.
    assert review["scores"] == {
        label: ev["display"] for label, ev in evaluations.items()
    }

    general = evaluations[app.REVIEW_GENERAL_LABEL]
    assert general["status"] == "evaluated"
    assert 0 <= general["value"] <= 100

    # La calificación general debe ser el promedio ponderado de las dimensiones
    # con valor, nunca un número inventado.
    numeric = {
        label: ev["value"]
        for label, ev in evaluations.items()
        if label != app.REVIEW_GENERAL_LABEL and ev["value"] is not None
    }
    weight = sum(app.REVIEW_WEIGHTS[label] for label in numeric)
    expected = round(
        sum(app.REVIEW_WEIGHTS[label] * value for label, value in numeric.items())
        / weight
    )
    assert general["value"] == expected


def test_review_score_level_uses_semaphore_thresholds(picker_project):
    assert app._review_score_level(None) == "none"
    assert app._review_score_level(0) == "low"
    assert app._review_score_level(49) == "low"
    assert app._review_score_level(50) == "mid"
    assert app._review_score_level(79) == "mid"
    assert app._review_score_level(80) == "high"
    assert app._review_score_level(100) == "high"


def test_review_info_keys_are_unique_and_stable(picker_project):
    labels = [app.REVIEW_GENERAL_LABEL, *app.REVIEW_DIMENSION_LABELS]
    keys = [app._review_info_key(label) for label in labels]

    assert len(keys) == len(set(keys))
    assert all(key.startswith("strategy_review_info_") for key in keys)


def test_review_general_note_states_structural_scope(picker_project):
    text = app.REVIEW_GENERAL_NOTE.lower()

    assert "calidad estructural" in text
    assert "no mide rentabilidad" in text
    assert "backtest" in text


def test_syntax_error_marks_every_dimension_not_evaluable(picker_project):
    invalid = app.analyze_strategy_source("def broken(:\n", "broken.py")

    assert invalid["valid"] is False
    assert invalid["syntax_valid"] is False
    assert "Error de sintaxis" in invalid["error"]
    assert "Error de sintaxis" in invalid["syntax_error"]
    assert invalid["dialect"] == "No determinado"
    for label, evaluation in invalid["evaluations"].items():
        assert evaluation["status"] == "not_evaluable"
        assert evaluation["value"] is None
    assert (
        invalid["evaluations"][app.REVIEW_GENERAL_LABEL]["display"]
        == "Evaluación insuficiente"
    )
    assert invalid["strengths"] == []


def test_analysis_never_executes_the_reviewed_file(picker_project):
    # Si el analizador ejecutara el archivo, esta línea provocaría una excepción.
    source = (
        'raise RuntimeError("booby trap")\n'
        "def run_strategy(data):\n    return data\n"
    )

    review = app.analyze_strategy_source(source, "booby_trap.py")

    assert isinstance(review, dict)
    assert review["syntax_valid"] is True


@pytest.mark.parametrize(
    ("source", "expected_status"),
    [
        (
            "def run_strategy(data):\n    return data\n",
            "unverified",
        ),
        (
            "def run_strategy(data):\n"
            "    stop_loss_pct = 0.02\n"
            "    return data\n",
            "partial",
        ),
        (
            "def run_strategy(data):\n"
            "    stop_loss_pct = 0.02\n"
            "    take_profit_pct = 0.05\n"
            "    stop = data * (1 - stop_loss_pct)\n"
            "    take = data * (1 + take_profit_pct)\n"
            "    return stop + take\n",
            "evaluated",
        ),
    ],
)
def test_risk_management_distinguishes_declared_and_used_controls(
    picker_project, source, expected_status
):
    review = app.analyze_strategy_source(source, "risk.py")
    risk = review["evaluations"]["Gestión de riesgo"]

    assert risk["status"] == expected_status


def test_declared_but_unused_stop_is_reported_as_a_weakness(picker_project):
    source = "def run_strategy(data):\n    stop_loss_pct = 0.02\n    return data\n"

    review = app.analyze_strategy_source(source, "risk.py")

    risk = review["evaluations"]["Gestión de riesgo"]
    assert risk["status"] == "partial"
    assert any("sin uso detectable" in finding for finding in risk["findings"])
    assert any("sin uso detectable" in item for item in review["weaknesses"])


@pytest.mark.parametrize(
    ("source", "expected_type"),
    [
        (
            "def run_strategy(data):\n"
            "    sma_fast = data.rolling(20).mean()\n"
            "    sma_slow = data.rolling(50).mean()\n"
            "    return sma_fast - sma_slow\n",
            "Cruce de medias",
        ),
        (
            "escalon = 0.01\n"
            "def run_strategy(data):\n    return data\n",
            "Escalonado / Grid",
        ),
        (
            "def run_strategy(data):\n"
            "    momentum = data.diff()\n"
            "    return momentum\n",
            "Momentum",
        ),
    ],
)
def test_strategy_type_is_derived_from_code_evidence(
    picker_project, source, expected_type
):
    review = app.analyze_strategy_source(source, "kind.py")

    assert review["strategy_type"] == expected_type


def test_fast_and_slow_moving_averages_are_both_detected(picker_project):
    source = (
        "def run_strategy(data):\n"
        "    sma_fast = data.rolling(20).mean()\n"
        "    sma_slow = data.rolling(50).mean()\n"
        "    return sma_fast - sma_slow\n"
    )

    review = app.analyze_strategy_source(source, "cross.py")

    assert any("rápida" in item for item in review["indicators"])
    assert any("lenta" in item for item in review["indicators"])


def test_horizon_is_only_taken_from_an_explicit_resolution(picker_project):
    inferred = (
        "def run_strategy(data):\n"
        "    sma = data.rolling(20).mean()\n"
        "    return sma\n"
    )
    review = app.analyze_strategy_source(inferred, "no_res.py")
    assert review["horizon"] == "No especificado"
    assert "no se infiere" in review["horizon_note"]

    explicit = (
        "def run_strategy(data):\n"
        "    window = data.rolling(Resolution.DAILY)\n"
        "    return window\n"
    )
    review = app.analyze_strategy_source(explicit, "daily.py")
    assert review["horizon"] == "Diario"


def test_market_symbols_come_from_literal_detection(picker_project):
    source = 'symbol = "AAPL"\ndef run_strategy(data):\n    return data\n'

    review = app.analyze_strategy_source(source, "symbols.py")

    assert review["symbols"] == ["AAPL"]
    assert review["market"] == "AAPL"


def test_robustness_needs_structural_evidence_to_be_evaluated(picker_project):
    bare = "def run_strategy(data):\n    return data\n"
    review = app.analyze_strategy_source(bare, "bare.py")
    assert review["evaluations"]["Robustez estática estimada"]["status"] == "partial"

    guarded = (
        "def run_strategy(data):\n"
        '    try:\n'
        '        if "close" not in data.columns:\n'
        '            raise ValueError("missing close")\n'
        "    except ValueError:\n"
        "        return data.dropna()\n"
        "    return data\n"
    )
    review = app.analyze_strategy_source(guarded, "guarded.py")
    robustness = review["evaluations"]["Robustez estática estimada"]
    assert robustness["status"] == "evaluated"
    assert robustness["value"] > 50
    assert any("try/except" in finding for finding in robustness["findings"])


def test_minimal_strategy_leaves_risk_and_robustness_unevaluated(picker_project):
    # Sin indicadores, señales ni control de riesgo no se puede confirmar ni la
    # gestión de riesgo ni la robustez: quedan como no verificadas/parciales.
    source = "def run_strategy(data):\n    return data\n"

    review = app.analyze_strategy_source(source, "minimal.py")

    assert review["evaluations"]["Gestión de riesgo"]["status"] == "unverified"
    assert review["evaluations"]["Robustez estática estimada"]["status"] == "partial"


@pytest.mark.parametrize(
    "source",
    [
        "def run_strategy(data):\n    return data\n",
        (
            "def run_strategy(data):\n"
            "    sma_fast = data.rolling(20).mean()\n"
            "    return sma_fast\n"
        ),
        "escalon = 0.01\ndef run_strategy(data):\n    return data\n",
    ],
)
def test_strengths_and_weaknesses_are_bounded_and_evidence_based(
    picker_project, source
):
    review = app.analyze_strategy_source(source, "bounded.py")

    assert 0 < len(review["strengths"]) <= 5
    assert 0 < len(review["weaknesses"]) <= 5
    for item in review["strengths"] + review["weaknesses"]:
        assert item.strip()
    assert len(review["strengths"]) == len(set(review["strengths"]))
    assert len(review["weaknesses"]) == len(set(review["weaknesses"]))


def test_failed_selection_clears_previous_preview_and_reports_error(picker_project):
    write_strategy(picker_project, "alpha.py", "def run_strategy(data):\n    return data\n")
    assert app.select_strategy_for_review("alpha.py")

    assert app.select_strategy_for_review("missing.py") is False

    assert app.MODAL_SELECTED_SOURCE_KEY not in app.st.session_state
    assert "missing.py" in app.st.session_state[app.MODAL_ERROR_KEY]


def test_uploaded_source_only_maps_to_exact_project_file(picker_project):
    source = "def run_strategy(data):\n    return data\n"
    write_strategy(picker_project, "alpha.py", source)

    assert app.match_uploaded_strategy(source, "alpha.py") == "alpha.py"
    assert app.match_uploaded_strategy(source + "# changed\n", "alpha.py") is None
    with pytest.raises(ValueError, match="\\.py"):
        app.match_uploaded_strategy(source, "alpha.txt")


def test_file_changed_after_review_cannot_replace_open_strategy(picker_project):
    original = "def run_strategy(data):\n    return data\n"
    current_source = "edición pendiente\n"
    write_strategy(picker_project, "current.py", "código guardado\n")
    target_path = write_strategy(picker_project, "alpha.py", original)
    app.st.session_state.update(
        {
            app.PRINCIPAL_KEY: "current.py",
            "editor_current.py": current_source,
            "codigo_actual_script": "current.py",
            "codigo_actual": current_source,
        }
    )
    reviewed_source, reviewed_hash = app.read_strategy_candidate("alpha.py")
    assert app.request_editor_navigation(
        "replace",
        "alpha.py",
        target_source=reviewed_source,
        target_sha256=reviewed_hash,
    )
    target_path.write_text(original + "# changed after review\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cambió desde su análisis"):
        app.resolve_editor_navigation("discard", ["current.py", "alpha.py"])

    assert app.st.session_state[app.PRINCIPAL_KEY] == "current.py"
    assert app.st.session_state["editor_current.py"] == current_source
    assert app.EDITOR_NAVIGATION_KEY in app.st.session_state


def test_newly_loaded_strategy_waits_for_explicit_backtest(picker_project, monkeypatch):
    write_strategy(picker_project, "alpha.py", "def run_strategy(data):\n    return data\n")
    app.st.session_state.update(
        {
            app.PRINCIPAL_KEY: "alpha.py",
            app.BACKTEST_PENDING_KEY: "alpha.py",
        }
    )

    def forbid_run(*_args, **_kwargs):
        raise AssertionError("no debe ejecutar el backtest sin solicitud explícita")

    monkeypatch.setattr(app, "run_strategy_backtest", forbid_run)
    _frame, metrics = app.get_backtest_result("SPY", 100_000)

    assert metrics["standby"] is True
    assert app.st.session_state[app.BACKTEST_PENDING_KEY] == "alpha.py"
    assert isinstance(_frame, pd.DataFrame)


def test_pending_load_skips_main_and_comparative_backtests_in_lab(monkeypatch):
    session = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    session = session.radio(key="workspace_tab").set_value("LABORATORIO").run()
    script = "grid_trading_v2.py"
    session.session_state[app.PRINCIPAL_KEY] = script
    session.session_state[app.BACKTEST_PENDING_KEY] = script
    session.session_state[app.COMPARE_KEY] = ["otras/SP500.py"]

    def forbid_run(*_args, **_kwargs):
        raise AssertionError("no debe ejecutarse una estrategia antes de pulsar Run")

    monkeypatch.setattr(app, "run_strategy_backtest", forbid_run)
    monkeypatch.setattr(app, "get_comparative_results", forbid_run)
    session = session.run()

    assert not session.exception, [error.value for error in session.exception]
    assert session.session_state[app.BACKTEST_PENDING_KEY] == script
    assert any(
        "el backtest no se ejecutó" in item.value.lower() for item in session.info
    )
