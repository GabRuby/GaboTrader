import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app


@pytest.fixture
def proyecto_editor(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "BASE_DIR", tmp_path)
    monkeypatch.setattr(app.st, "session_state", {})
    return tmp_path


def escribir(base: Path, script: str, source: str) -> Path:
    path = base / script
    path.write_text(source, encoding="utf-8")
    return path


def test_codigo_identico_al_archivo_se_considera_guardado(proyecto_editor):
    source = "def run_strategy(data):\n    return data\n"
    escribir(proyecto_editor, "a.py", source)

    assert app.editor_sync_status("a.py", source) == ("saved", "Guardada")


def test_cambio_del_editor_se_detecta_al_persistir_el_widget(proyecto_editor):
    escribir(proyecto_editor, "a.py", "versión guardada\n")
    app.st.session_state["editor_a.py"] = "versión editada\n"

    app.persist_editor_code("a.py", "editor_a.py")

    assert app.editor_sync_status("a.py", app.st.session_state["editor_a.py"])[0] == "pending"
    assert app.st.session_state["codigo_actual_script"] == "a.py"


def test_restaurar_exactamente_el_archivo_elimina_cambios_pendientes(proyecto_editor):
    source = "versión guardada\n"
    escribir(proyecto_editor, "a.py", source)
    app.st.session_state["editor_a.py"] = "versión editada\n"
    app.persist_editor_code("a.py", "editor_a.py")
    app.st.session_state["editor_a.py"] = source

    app.persist_editor_code("a.py", "editor_a.py")

    assert app.editor_sync_status("a.py", source) == ("saved", "Guardada")


def test_guardado_correcto_elimina_estado_pendiente(proyecto_editor):
    source = "versión nueva\n"
    escribir(proyecto_editor, "a.py", "versión anterior\n")
    assert app.editor_sync_status("a.py", source)[0] == "pending"

    app.save_editor_source("a.py", source)

    assert app.editor_sync_status("a.py", source) == ("saved", "Guardada")


def test_guardar_contenido_ya_sincronizado_no_escribe(proyecto_editor, monkeypatch):
    source = "ya guardado\n"
    escribir(proyecto_editor, "a.py", source)

    def fail_write(*_args, **_kwargs):
        raise AssertionError("no debe escribir si no hay cambios")

    monkeypatch.setattr(Path, "write_text", fail_write)

    app.save_editor_source("a.py", source)


def test_fallo_de_escritura_conserva_estado_pendiente(proyecto_editor, monkeypatch):
    path = escribir(proyecto_editor, "a.py", "versión anterior\n")
    source = "versión nueva\n"
    original_write_text = Path.write_text

    def fail_selected_write(target, *args, **kwargs):
        if target == path:
            raise OSError("disco de solo lectura")
        return original_write_text(target, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", fail_selected_write)
        with pytest.raises(OSError, match="disco de solo lectura"):
            app.save_editor_source("a.py", source)

    assert app.editor_sync_status("a.py", source) == (
        "pending",
        "Cambios pendientes · el último guardado falló.",
    )
    assert app.EDITOR_SAVE_IN_PROGRESS_KEY not in app.st.session_state
    app.save_editor_source("a.py", source)
    assert app.editor_sync_status("a.py", source) == ("saved", "Guardada")


@pytest.mark.parametrize(
    ("script", "source", "in_progress", "expected_disabled"),
    [
        ("", "sin archivo", False, True),
        ("a.py", "guardado", False, True),
        ("a.py", "pendiente", False, False),
        ("a.py", "pendiente", True, True),
    ],
)
def test_estado_del_boton_guardar(
    proyecto_editor, script, source, in_progress, expected_disabled
):
    if script:
        escribir(proyecto_editor, script, "guardado")
        app.st.session_state[app.PRINCIPAL_KEY] = script
    if in_progress:
        app.st.session_state[app.EDITOR_SAVE_IN_PROGRESS_KEY] = True

    assert app.editor_save_button_disabled(script, source) is expected_disabled


def test_boton_guardar_desactivado_si_no_se_puede_verificar(proyecto_editor):
    app.st.session_state[app.PRINCIPAL_KEY] = "faltante.py"

    assert app.editor_save_button_disabled("faltante.py", "edición") is True


def test_estado_se_compara_con_el_archivo_seleccionado(proyecto_editor):
    source_a = "código de A\n"
    source_b = "código de B\n"
    escribir(proyecto_editor, "a.py", source_a)
    escribir(proyecto_editor, "b.py", source_b)

    assert app.editor_sync_status("a.py", source_b)[0] == "pending"
    assert app.editor_sync_status("b.py", source_b) == ("saved", "Guardada")


def test_reemplazar_estrategia_solicita_decision_y_cancelar_conserva_edicion(
    proyecto_editor, monkeypatch
):
    source = "sin guardar\n"
    escribir(proyecto_editor, "a.py", "guardado A\n")
    escribir(proyecto_editor, "b.py", "guardado B\n")
    monkeypatch.setattr(app, "py_files", [])
    app.st.session_state.update(
        {
            app.PRINCIPAL_KEY: "a.py",
            "editor_a.py": source,
            "codigo_actual_script": "a.py",
            "codigo_actual": source,
        }
    )

    assert app.request_editor_navigation("replace", "b.py") is True
    assert app.resolve_editor_navigation("cancel", ["a.py", "b.py"]) == "cancelled"
    assert app.st.session_state[app.PRINCIPAL_KEY] == "a.py"
    assert app.st.session_state["editor_a.py"] == source
    assert (proyecto_editor / "a.py").read_text(encoding="utf-8") == "guardado A\n"


def test_reemplazo_sin_cambios_no_requiere_confirmacion(proyecto_editor):
    source = "guardado\n"
    escribir(proyecto_editor, "a.py", source)
    escribir(proyecto_editor, "b.py", "otra estrategia\n")
    app.st.session_state.update(
        {
            app.PRINCIPAL_KEY: "a.py",
            "editor_a.py": source,
        }
    )

    assert app.request_editor_navigation("replace", "b.py") is False
    assert app.EDITOR_NAVIGATION_KEY not in app.st.session_state


def test_reemplazar_estrategia_guarda_o_descarta_solo_con_decision_explicita(
    proyecto_editor, monkeypatch
):
    source = "edición A\n"
    path_a = escribir(proyecto_editor, "a.py", "guardado A\n")
    escribir(proyecto_editor, "b.py", "guardado B\n")
    monkeypatch.setattr(app, "py_files", [])
    app.st.session_state.update(
        {
            app.PRINCIPAL_KEY: "a.py",
            "editor_a.py": source,
            "codigo_actual_script": "a.py",
            "codigo_actual": source,
        }
    )

    assert app.request_editor_navigation("replace", "b.py")
    app.resolve_editor_navigation("discard", ["a.py", "b.py"])
    assert app.st.session_state[app.PRINCIPAL_KEY] == "b.py"
    assert path_a.read_text(encoding="utf-8") == "guardado A\n"


def test_fallo_de_guardado_no_sustituye_la_principal_y_permite_reintento(
    proyecto_editor, monkeypatch
):
    source = "edición A\n"
    path_a = escribir(proyecto_editor, "a.py", "guardado A\n")
    escribir(proyecto_editor, "b.py", "guardado B\n")
    monkeypatch.setattr(app, "py_files", [])
    app.st.session_state.update(
        {
            app.PRINCIPAL_KEY: "a.py",
            "editor_a.py": source,
            "codigo_actual_script": "a.py",
            "codigo_actual": source,
        }
    )
    original_write_text = Path.write_text
    fail = True

    def fail_once(target, *args, **kwargs):
        nonlocal fail
        if target == path_a and fail:
            fail = False
            raise OSError("error temporal de escritura")
        return original_write_text(target, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_once)
    assert app.request_editor_navigation("replace", "b.py")
    with pytest.raises(OSError, match="error temporal"):
        app.resolve_editor_navigation("save", ["a.py", "b.py"])
    assert app.st.session_state[app.PRINCIPAL_KEY] == "a.py"
    assert app.EDITOR_NAVIGATION_KEY in app.st.session_state
    assert app.editor_sync_status("a.py", source)[0] == "pending"

    app.resolve_editor_navigation("save", ["a.py", "b.py"])
    assert app.st.session_state[app.PRINCIPAL_KEY] == "b.py"
    assert path_a.read_text(encoding="utf-8") == source


def test_cerrar_con_cambios_se_puede_cancelar_sin_tocar_comparativas(
    proyecto_editor
):
    source = "edición sin guardar\n"
    path = escribir(proyecto_editor, "a.py", "guardado\n")
    app.st.session_state.update(
        {
            app.PRINCIPAL_KEY: "a.py",
            app.COMPARE_KEY: ["comparativa.py"],
            "editor_a.py": source,
            "codigo_actual_script": "a.py",
            "codigo_actual": source,
        }
    )

    assert app.request_editor_navigation("close")
    app.resolve_editor_navigation("cancel", ["a.py"])
    assert app.st.session_state[app.PRINCIPAL_KEY] == "a.py"
    assert app.st.session_state[app.COMPARE_KEY] == ["comparativa.py"]
    assert app.st.session_state["editor_a.py"] == source
    assert path.read_text(encoding="utf-8") == "guardado\n"


def test_cerrar_con_descartar_preserva_archivo_y_comparativas(proyecto_editor):
    source = "edición sin guardar\n"
    path = escribir(proyecto_editor, "a.py", "guardado\n")
    app.st.session_state.update(
        {
            app.PRINCIPAL_KEY: "a.py",
            app.COMPARE_KEY: ["comparativa.py"],
            "editor_a.py": source,
        }
    )

    assert app.request_editor_navigation("close")
    app.resolve_editor_navigation("discard", ["a.py"])
    assert app.st.session_state[app.PRINCIPAL_KEY] is None
    assert app.st.session_state[app.COMPARE_KEY] == ["comparativa.py"]
    assert path.read_text(encoding="utf-8") == "guardado\n"


def test_errores_de_guardado_de_archivos_distintos_no_se_mezclan(proyecto_editor, monkeypatch):
    source_a, source_b = "edit A\n", "edit B\n"
    path_a = escribir(proyecto_editor, "a.py", "saved A\n")
    path_b = escribir(proyecto_editor, "b.py", "saved B\n")
    original_write_text = Path.write_text

    def fail_a(target, *args, **kwargs):
        if target == path_a:
            raise OSError("A no disponible")
        return original_write_text(target, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_a)
    with pytest.raises(OSError):
        app.save_editor_source("a.py", source_a)
    monkeypatch.setattr(Path, "write_text", original_write_text)
    app.save_editor_source("b.py", source_b)

    assert app.editor_sync_status("a.py", source_a)[0] == "pending"
    assert app.editor_sync_status("b.py", source_b)[0] == "saved"
    assert path_b.read_text(encoding="utf-8") == source_b


def test_archivo_inexistente_o_ilegible_no_se_considera_guardado(
    proyecto_editor, monkeypatch
):
    assert app.editor_sync_status("faltante.py", "código\n")[0] == "unavailable"

    path = escribir(proyecto_editor, "ilegible.py", "código\n")
    original_bytes = path.read_bytes()
    original_read_text = Path.read_text

    def fail_selected_read(target, *args, **kwargs):
        if target == path:
            raise PermissionError("acceso denegado")
        return original_read_text(target, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_selected_read)

    state, message = app.editor_sync_status("ilegible.py", "código\n")
    assert state == "unavailable"
    assert "acceso denegado" in message
    with pytest.raises(OSError, match="no se sobrescribió"):
        app.save_editor_source("ilegible.py", "nuevo contenido\n")
    assert path.read_bytes() == original_bytes


def test_sin_estrategia_no_asocia_el_editor_a_un_archivo(proyecto_editor):
    state, message = app.editor_sync_status("", "contenido temporal\n")

    assert state == "no_strategy"
    assert "no está asociado" in message


def test_editores_de_dos_comparativas_conservan_estado_independiente(proyecto_editor):
    saved_a = "código A guardado\n"
    saved_b = "código B guardado\n"
    escribir(proyecto_editor, "a.py", saved_a)
    escribir(proyecto_editor, "b.py", saved_b)
    app.st.session_state[app.COMPARE_KEY] = ["a.py", "b.py"]
    key_a = app.compare_editor_widget_key("a.py")
    key_b = app.compare_editor_widget_key("b.py")
    app.st.session_state[key_a] = "código A editado\n"
    app.st.session_state[key_b] = "código B editado\n"

    app.persist_comparison_editor_code("a.py", key_a)
    app.persist_comparison_editor_code("b.py", key_b)
    app.save_editor_source("a.py", app.editor_source_for("a.py", "comparison"))

    assert key_a != key_b
    assert app.comparison_editor_sync_status("a.py")[0] == "saved"
    assert app.comparison_editor_sync_status("b.py")[0] == "pending"
    assert app.editor_source_for("b.py", "comparison") == "código B editado\n"
    assert (proyecto_editor / "b.py").read_text(encoding="utf-8") == saved_b


def test_quitar_comparativa_pendiente_cancelar_o_descartar_es_explicito(
    proyecto_editor,
):
    saved = "código guardado\n"
    edited = "código editado\n"
    path = escribir(proyecto_editor, "a.py", saved)
    key = app.compare_editor_widget_key("a.py")
    app.st.session_state[app.COMPARE_KEY] = ["a.py"]
    app.st.session_state[key] = edited
    app.persist_comparison_editor_code("a.py", key)

    assert app.request_editor_navigation("remove_comparison", "a.py")
    app.resolve_editor_navigation("cancel", ["a.py"])
    assert app.st.session_state[app.COMPARE_KEY] == ["a.py"]
    assert app.editor_source_for("a.py", "comparison") == edited
    assert path.read_text(encoding="utf-8") == saved

    assert app.request_editor_navigation("remove_comparison", "a.py")
    app.resolve_editor_navigation("discard", ["a.py"])
    assert app.st.session_state[app.COMPARE_KEY] == []
    assert app.editor_source_for("a.py", "comparison") == saved
    assert app.comparison_editor_sync_status("a.py")[0] == "saved"
    assert path.read_text(encoding="utf-8") == saved


def test_error_y_reintento_de_guardado_comparativo_no_afecta_otro_editor(
    proyecto_editor, monkeypatch
):
    path_a = escribir(proyecto_editor, "a.py", "A guardado\n")
    escribir(proyecto_editor, "b.py", "B guardado\n")
    app.st.session_state[app.COMPARE_KEY] = ["a.py", "b.py"]
    key_a = app.compare_editor_widget_key("a.py")
    key_b = app.compare_editor_widget_key("b.py")
    source_a, source_b = "A editado\n", "B editado\n"
    app.st.session_state[key_a] = source_a
    app.st.session_state[key_b] = source_b
    app.persist_comparison_editor_code("a.py", key_a)
    app.persist_comparison_editor_code("b.py", key_b)
    original_write_text = Path.write_text

    def fail_a_once(target, *args, **kwargs):
        if target == path_a:
            monkeypatch.setattr(Path, "write_text", original_write_text)
            raise OSError("A no disponible")
        return original_write_text(target, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_a_once)
    with pytest.raises(OSError, match="A no disponible"):
        app.save_editor_source("a.py", source_a)
    assert app.comparison_editor_sync_status("a.py")[0] == "pending"
    assert app.comparison_editor_sync_status("b.py")[0] == "pending"
    assert app.editor_source_for("b.py", "comparison") == source_b
    assert path_a.read_text(encoding="utf-8") == "A guardado\n"

    app.save_editor_source("a.py", source_a)
    assert app.comparison_editor_sync_status("a.py")[0] == "saved"
    assert app.comparison_editor_sync_status("b.py")[0] == "pending"
    assert path_a.read_text(encoding="utf-8") == source_a


def test_quitar_comparativa_pendiente_con_guardado_guarda_solo_esa_estrategia(
    proyecto_editor,
):
    path_a = escribir(proyecto_editor, "a.py", "A guardado\n")
    path_b = escribir(proyecto_editor, "b.py", "B guardado\n")
    app.st.session_state[app.COMPARE_KEY] = ["a.py", "b.py"]
    key_a = app.compare_editor_widget_key("a.py")
    app.st.session_state[key_a] = "A editado\n"
    app.persist_comparison_editor_code("a.py", key_a)

    assert app.request_editor_navigation("remove_comparison", "a.py")
    app.resolve_editor_navigation("save", ["a.py", "b.py"])

    assert app.st.session_state[app.COMPARE_KEY] == ["b.py"]
    assert path_a.read_text(encoding="utf-8") == "A editado\n"
    assert path_b.read_text(encoding="utf-8") == "B guardado\n"


def test_no_se_promueve_comparativa_con_edicion_pendiente(proyecto_editor):
    saved = "código guardado\n"
    escribir(proyecto_editor, "a.py", saved)
    app.st.session_state[app.COMPARE_KEY] = ["a.py"]
    app.st.session_state[app.compare_editor_widget_key("a.py")] = "código editado\n"

    with pytest.raises(ValueError, match="cambios"):
        app.set_principal_strategy("a.py", ["a.py"])

    assert app.st.session_state.get(app.PRINCIPAL_KEY) is None
    assert app.st.session_state[app.COMPARE_KEY] == ["a.py"]
    assert app.comparison_editor_sync_status("a.py")[0] == "pending"


def test_limite_de_una_comparativa_y_reemplazo_protegido(
    proyecto_editor, monkeypatch
):
    saved = "def run_strategy(data):\n    return data\n"
    first = escribir(proyecto_editor, "a.py", saved)
    escribir(proyecto_editor, "b.py", saved + "# segunda\n")
    monkeypatch.setattr(app, "ALL_STRATEGIES", ["a.py", "b.py"])
    app.st.session_state[app.COMPARE_KEY] = ["a.py"]
    editor_key = app.compare_editor_widget_key("a.py")
    edited = saved + "# cambios pendientes\n"
    app.st.session_state[editor_key] = edited
    app.persist_comparison_editor_code("a.py", editor_key)

    accepted, message = app.add_comparative_strategy("b.py", ["a.py", "b.py"])
    assert not accepted
    assert "Solo caben 1 comparativas" in message
    assert app.request_editor_navigation("replace_comparison", "b.py")
    app.resolve_editor_navigation("cancel", ["a.py", "b.py"])
    assert app.st.session_state[app.COMPARE_KEY] == ["a.py"]
    assert app.editor_source_for("a.py", "comparison") == edited

    assert app.request_editor_navigation("replace_comparison", "b.py")
    app.resolve_editor_navigation("discard", ["a.py", "b.py"])
    assert app.st.session_state[app.COMPARE_KEY] == ["b.py"]
    assert first.read_text(encoding="utf-8") == saved


def test_deshacer_copia_externa_no_modifica_original_ni_otro_editor(
    proyecto_editor, tmp_path
):
    original = tmp_path / "outside.py"
    saved = "def run_strategy(data):\n    return data\n"
    original.write_text(saved, encoding="utf-8")
    external = app.create_external_strategy(original.name, saved)
    app.st.session_state[app.COMPARE_KEY] = [external]
    key = app.compare_editor_widget_key(external)
    edited = saved + "# cambios\n"
    app.st.session_state[key] = edited
    app.persist_comparison_editor_code(external, key)
    app.st.session_state["editor_other.py"] = "editor principal separado\n"

    app.restore_editor_source(external, "comparison")

    assert app.st.session_state[key] == saved
    assert app.st.session_state["editor_other.py"] == "editor principal separado\n"
    assert original.read_text(encoding="utf-8") == saved
    assert app.comparison_editor_sync_status(external)[0] == "saved"


def test_guardar_panel_principal_no_modifica_codigo_comparativo(proyecto_editor):
    saved_main = "main guardado\n"
    saved_comparison = "comparison guardado\n"
    main_path = escribir(proyecto_editor, "main.py", saved_main)
    comparison_path = escribir(proyecto_editor, "comparison.py", saved_comparison)
    app.st.session_state[app.PRINCIPAL_KEY] = "main.py"
    app.st.session_state[app.COMPARE_KEY] = ["comparison.py"]
    app.st.session_state["editor_main.py"] = "main editado\n"
    comparison_key = app.compare_editor_widget_key("comparison.py")
    app.st.session_state[comparison_key] = "comparison editado\n"
    app.persist_editor_code("main.py", "editor_main.py")
    app.persist_comparison_editor_code("comparison.py", comparison_key)

    app.save_editor_source("main.py", app.st.session_state["editor_main.py"])

    assert main_path.read_text(encoding="utf-8") == "main editado\n"
    assert comparison_path.read_text(encoding="utf-8") == saved_comparison
    assert app.editor_sync_status("main.py", "main editado\n")[0] == "saved"
    assert app.comparison_editor_sync_status("comparison.py")[0] == "pending"


def test_comparativa_usa_parametros_independientes_y_no_se_ejecuta_al_cargar(
    proyecto_editor, monkeypatch
):
    script = escribir(
        proyecto_editor,
        "comparison.py",
        "def run_strategy(data):\n    return data\n",
    ).name
    monkeypatch.setattr(app, "ALL_STRATEGIES", [script])
    app.PARAMETROS_ACTIVOS.clear()
    app.PARAMETROS_ACTIVOS.update({"fast_ma": 21})
    cfg = app._config_lab("TSLA", 1000, {"fast_ma": 7})
    assert cfg.parametros == {"fast_ma": 7}

    app.st.session_state[app.COMPARE_KEY] = [script]
    app.st.session_state[app.COMPARE_PENDING_KEY] = script
    app.st.session_state[app.COMPARE_PARAMETERS_KEY] = {script: {"fast_ma": 7}}
    calls = []
    monkeypatch.setattr(
        app,
        "run_comparative_backtest",
        lambda selected, ticker, capital, params=None: (
            calls.append((selected, params)) or (None, {"script": selected})
        ),
    )

    assert app.get_comparative_results([script], "TSLA", 1000) == {}
    assert calls == []
    results = app.get_comparative_results([script], "TSLA", 1000, force=True)

    assert calls == [(script, {"fast_ma": 7})]
    assert results[script]["metrics"]["script"] == script
    assert app.COMPARE_PENDING_KEY not in app.st.session_state
