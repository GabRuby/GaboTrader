"""Fase B · controles numéricos: tipos, rangos, coherencia fast<slow y ejecución.

Cubren la causa raíz del «window must be an integer 0 or greater»: los widgets y los
fallbacks deben producir enteros NATIVOS para fast_ma/slow_ma, y la frontera del motor
convierte floats enteros exactos (10.0 → 10) pero rechaza decimales y fuera de rango
con error visible.
"""

from pathlib import Path
import sys

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: E402


@pytest.fixture
def estado(monkeypatch):
    """Session state vacío tipo dict para probar helpers en bare mode."""
    monkeypatch.setattr(app.st, "session_state", {})
    return app.st.session_state


@pytest.fixture
def laboratorio():
    """Laboratorio abierto sin Estrategia Principal (vista de plantilla inicial)."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    at.session_state["workspace_tab"] = "LABORATORIO"
    at = at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


# -----------------------------------------------------------------------------
# Frontera del motor: 10.0 -> int, decimales y fuera de rango -> error visible
# -----------------------------------------------------------------------------

def test_frontera_convierte_float_entero_exacto_a_int(estado):
    saneado = app._sanear_parametros_motor({"fast_ma": 10.0, "slow_ma": 50.0})

    assert saneado == {"fast_ma": 10, "slow_ma": 50}
    assert type(saneado["fast_ma"]) is int
    assert type(saneado["slow_ma"]) is int


def test_frontera_rechaza_decimal_real(estado):
    with pytest.raises(ValueError, match="debe ser un entero"):
        app._sanear_parametros_motor({"fast_ma": 10.5, "slow_ma": 50})


def test_frontera_rechaza_valores_fuera_de_rango(estado):
    with pytest.raises(ValueError, match="fuera de rango"):
        app._sanear_parametros_motor({"fast_ma": 1, "slow_ma": 50})
    with pytest.raises(ValueError, match="fuera de rango"):
        app._sanear_parametros_motor({"fast_ma": 101, "slow_ma": 50})
    with pytest.raises(ValueError, match="fuera de rango"):
        app._sanear_parametros_motor({"fast_ma": 10, "slow_ma": 2})
    with pytest.raises(ValueError, match="fuera de rango"):
        app._sanear_parametros_motor({"fast_ma": 10, "slow_ma": 201})


def test_frontera_rechaza_fast_mayor_igual_que_slow(estado):
    with pytest.raises(ValueError, match="menor que Slow SMA"):
        app._sanear_parametros_motor({"fast_ma": 50, "slow_ma": 50})
    with pytest.raises(ValueError, match="menor que Slow SMA"):
        app._sanear_parametros_motor({"fast_ma": 60, "slow_ma": 50})


def test_frontera_rechaza_valores_no_numericos(estado):
    with pytest.raises(ValueError, match="no es un n"):
        app._sanear_parametros_motor({"fast_ma": "diez", "slow_ma": 50})
    with pytest.raises(ValueError, match="no es un n"):
        app._sanear_parametros_motor({"fast_ma": True, "slow_ma": 50})


def test_frontera_no_toca_parametros_no_sma(estado):
    saneado = app._sanear_parametros_motor(
        {"fast_ma": 10.0, "slow_ma": 50.0, "sl_pct": 0.10}
    )

    assert saneado["sl_pct"] == 0.10
    assert saneado["fast_ma"] == 10


def test_config_lab_entrega_enteros_al_motor(estado):
    cfg = app._config_lab("TSLA", 100000, {"fast_ma": 12.0, "slow_ma": 26.0})

    assert cfg.parametros == {"fast_ma": 12, "slow_ma": 26}
    assert all(type(v) is int for v in cfg.parametros.values())


# -----------------------------------------------------------------------------
# Ejecución de la estrategia: sin «window must be an integer»
# -----------------------------------------------------------------------------

def test_motor_ejecuta_la_plantilla_con_floats_enteros_inyectados(estado):
    """Reproduce el bug original: 10.0/50.0 en session -> la frontera los hace int."""
    cfg = app._config_lab("TSLA", 100000, {"fast_ma": 12.0, "slow_ma": 26.0})

    resultado = app.ejecutar_estrategia(fuente=app.PLANTILLA_INICIAL, config=cfg)

    assert resultado.exito, resultado.error
    assert "window must be an integer" not in str(resultado.error or "")


def test_run_source_backtest_sin_parametros_ejecuta(estado):
    df, metrics = app.run_source_backtest(app.PLANTILLA_INICIAL, "TSLA", 100000, "t")

    assert not metrics.get("error"), metrics.get("error")
    assert not df.empty


# -----------------------------------------------------------------------------
# Tipos nativos en los widgets del Laboratorio
# -----------------------------------------------------------------------------

def test_sliders_sma_son_enteros_nativos(laboratorio):
    fast = laboratorio.slider(key="plantilla_param_fast_ma")
    slow = laboratorio.slider(key="plantilla_param_slow_ma")

    assert (fast.value, slow.value) == (10, 50)
    assert type(fast.value) is int
    assert type(slow.value) is int

    espejos = {ni.key: ni.value for ni in laboratorio.number_input}
    assert type(espejos["plantilla_param_fast_ma_num"]) is int
    assert type(espejos["plantilla_param_slow_ma_num"]) is int


def test_espejo_numerico_sincroniza_el_slider(laboratorio):
    laboratorio.session_state["plantilla_param_fast_ma_num"] = 30
    laboratorio = laboratorio.run()

    assert not laboratorio.exception, [e.value for e in laboratorio.exception]
    assert laboratorio.slider(key="plantilla_param_fast_ma").value == 30


# -----------------------------------------------------------------------------
# Rangos aprobados: Fast SMA 2–100, Slow SMA 3–200
# -----------------------------------------------------------------------------

def test_valores_fuera_de_rango_se_clampan_a_los_limites():
    # Los valores se inyectan ANTES de que exista el widget: el primer run deja la app
    # en INICIO (el laboratorio aún no monta sus controles), así que en el segundo run
    # el clamp lo aplica `sincronizar_control` (999 -> 100, 1 -> 3). Si la clave ya
    # perteneciera a un widget instanciado, Streamlit ignoraría la escritura entre runs.
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    at.session_state["workspace_tab"] = "LABORATORIO"
    at.session_state["plantilla_param_fast_ma"] = 999
    at.session_state["plantilla_param_slow_ma"] = 1
    at = at.run()

    assert not at.exception, [e.value for e in at.exception]
    assert at.slider(key="plantilla_param_fast_ma").value == 100
    assert at.slider(key="plantilla_param_slow_ma").value == 3


def test_sincronizar_control_clampa_enteros(estado):
    spec = {"label": "fast", "min": 2, "max": 100, "step": 1, "default": 10}
    estado["clave"] = 999

    valor = app.sincronizar_control("clave", spec, True)

    assert valor == 100
    assert estado["clave"] == 100
    assert type(estado["clave"]) is int


# -----------------------------------------------------------------------------
# Coherencia Fast SMA < Slow SMA
# -----------------------------------------------------------------------------

def test_fast_slow_coherencia_muestra_error(estado):
    mensajes = app.validar_parametros(
        {"fast_ma": 50, "slow_ma": 10},
        {"fast_ma": "Fast SMA", "slow_ma": "Slow SMA"},
    )

    assert mensajes
    assert any("Fast SMA" in m and "Slow SMA" in m for m in mensajes)


def test_valores_coherentes_no_producen_error(estado):
    assert app.validar_parametros({"fast_ma": 10, "slow_ma": 50}) == []


def test_valores_seguros_conserva_el_ultimo_valido_entero(estado):
    estado[app.PARAMETROS_VALIDOS_KEY] = {
        "plantilla": {"fast_ma": 10, "slow_ma": 50}
    }

    seguros = app.valores_seguros(
        "plantilla",
        {"fast_ma": 60, "slow_ma": 50},
        {"fast_ma": 10, "slow_ma": 50},
    )

    assert seguros == {"fast_ma": 10, "slow_ma": 50}
    assert type(seguros["fast_ma"]) is int


def test_declarados_nativos_son_enteros(estado):
    info = app.strategy_parser.analizar_fuente(app.PLANTILLA_INICIAL, "plantilla.py")

    declarados = app._declarados_nativos(info)

    assert declarados == {"fast_ma": 10, "slow_ma": 50}
    assert type(declarados["fast_ma"]) is int
    assert type(declarados["slow_ma"]) is int


def test_incoherencia_en_ui_muestra_error_y_conserva_ultimo_valido(laboratorio):
    laboratorio = laboratorio.slider(key="plantilla_param_fast_ma").set_value(60).run()

    assert not laboratorio.exception, [e.value for e in laboratorio.exception]
    errores = " ".join(str(e.value) for e in laboratorio.error)
    assert "Fast SMA" in errores and "Slow SMA" in errores

    registro = laboratorio.session_state[app.PARAMETROS_VALIDOS_KEY]["plantilla"]
    assert registro["fast_ma"] == 10 and registro["slow_ma"] == 50
    assert type(registro["fast_ma"]) is int


def test_simular_bloqueado_con_fast_mayor_que_slow(laboratorio):
    laboratorio = laboratorio.slider(key="plantilla_param_fast_ma").set_value(60).run()

    # AppTest prohíbe pulsar un botón deshabilitado (lanzaría AppTestError), así que
    # el bloqueo se aserta leyendo `disabled`: con fast >= slow el botón no es
    # pulsable y, por tanto, no puede registrarse ninguna simulación.
    boton = laboratorio.button(key="btn_simular_plantilla")
    assert boton.disabled, "Simular debe quedar deshabilitado con fast_ma >= slow_ma"


# -----------------------------------------------------------------------------
# Sin ejecuciones completas al manipular solo controles
# -----------------------------------------------------------------------------

def test_mover_slider_no_dispara_ejecucion_del_motor(monkeypatch, laboratorio):
    from qc_lab.core import runner

    llamadas = []
    original = runner.ejecutar_estrategia

    def spy(*args, **kwargs):
        llamadas.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "ejecutar_estrategia", spy)

    laboratorio = laboratorio.slider(key="plantilla_param_fast_ma").set_value(25).run()

    assert not laboratorio.exception, [e.value for e in laboratorio.exception]
    assert llamadas == [], "mover un control no debe ejecutar el motor"
    assert laboratorio.slider(key="plantilla_param_fast_ma").value == 25


def test_ejecutar_backtest_tras_cambiar_slider_funciona(laboratorio):
    laboratorio = laboratorio.slider(key="plantilla_param_fast_ma").set_value(12).run()
    laboratorio = laboratorio.button(key="btn_run_plantilla").click().run()

    assert not laboratorio.exception, [e.value for e in laboratorio.exception]
    visibles = " ".join(str(m.value) for m in laboratorio.error)
    visibles += " ".join(str(m.value) for m in laboratorio.info)
    assert "window must be an integer" not in visibles
