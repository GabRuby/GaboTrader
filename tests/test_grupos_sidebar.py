"""
Pruebas de los grupos 1, 2 y 3 del sidebar (fase G.3.6).

Se prueba la logica que hay detras de los grupos: la precedencia
ARCHIVO -> valor -> UI -> configuracion del backtest y el ciclo de vida de las
comparativas. Las estrategias que hacen falta se escriben en `tmp_path`, de modo que
no se toca ninguna estrategia del proyecto.

`import app` ejecuta el script entero una vez (Streamlit en modo "bare"), por eso el
import es a nivel de modulo y no por funcion.
"""

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import app  # noqa: E402  (el path del proyecto tiene que estar antes)

# Dialecto vector: `run_strategy(data, ...)` es la otra forma que el motor ejecuta, y no
# depende de Lean, asi que estos archivos se pueden analizar sin instalar nada mas.
VECTOR = """import numpy as np


def run_strategy(data, fast_ma=10, slow_ma=50):
    data["SMA_Fast"] = data["Close"].rolling(window=int(fast_ma)).mean()
    data["SMA_Slow"] = data["Close"].rolling(window=int(slow_ma)).mean()
    data["Signal"] = np.where(data["SMA_Fast"] > data["SMA_Slow"], 1, 0)
    return data
"""


def lean(*lineas: str) -> str:
    """Estrategia Lean que declara exactamente las lineas que se le pasen."""
    cuerpo = "\n".join(f"        {linea}" for linea in lineas)
    return (
        "from AlgorithmImports import *\n\n\nclass Estrategia(QCAlgorithm):\n"
        "    def initialize(self):\n"
        f"{cuerpo}\n"
    )


@pytest.fixture
def proyecto(tmp_path, monkeypatch):
    """Proyecto temporal: `BASE_DIR` y la Principal apuntan dentro de `tmp_path`."""
    monkeypatch.setattr(app, "BASE_DIR", tmp_path)
    monkeypatch.setitem(app.st.session_state, app.PRINCIPAL_KEY, "vector.py")
    monkeypatch.setitem(app.st.session_state, app.COMPARE_KEY, [])
    monkeypatch.setitem(app.st.session_state, app.COMPARE_CACHE_KEY, {})
    return tmp_path


def escribir(base: Path, nombre: str, fuente: str) -> Path:
    ruta = base / nombre
    ruta.write_text(fuente, encoding="utf-8")
    return ruta


# --------------------------------------------------------------------------- #
# CASO A · estrategia con ticker y capital declarados
# --------------------------------------------------------------------------- #
def test_caso_a_muestra_el_ticker_y_el_capital_del_archivo(proyecto):
    ruta = escribir(proyecto, "lean.py", lean(
        "self.set_cash(25000)",
        'self.add_equity("SOXL", Resolution.Daily)',
    ))

    ticker, capital = app.reglas_comunes_de_estrategia(ruta)

    assert ticker == "SOXL"
    assert capital == 25_000.0


def test_caso_a_llega_a_la_configuracion_del_motor(proyecto):
    escribir(proyecto, "lean.py", lean(
        "self.set_cash(25000)",
        'self.add_equity("SOXL", Resolution.Daily)',
    ))

    config = app._config_lab(*app.sincronizar_reglas_comunes("lean.py"))

    assert config.simbolo == "SOXL"
    assert config.capital == 25_000.0


# --------------------------------------------------------------------------- #
# CASO B · estrategia sin ticker -> TSLA
# --------------------------------------------------------------------------- #
def test_caso_b_sin_ticker_muestra_tsla(proyecto):
    ruta = escribir(proyecto, "vector.py", VECTOR)

    assert app.tickers_declarados(ruta) == []
    assert app.reglas_comunes_de_estrategia(ruta)[0] == "TSLA" == app.DEFAULT_TICKER


def test_caso_b_tsla_solo_es_el_respaldo(proyecto):
    """Con ticker declarado el campo muestra el del archivo, no TSLA."""
    ruta = escribir(proyecto, "lean.py", lean('self.add_equity("AAPL", Resolution.Daily)'))

    assert app.reglas_comunes_de_estrategia(ruta)[0] == "AAPL"


# --------------------------------------------------------------------------- #
# CASO C · estrategia sin capital -> 100000
# --------------------------------------------------------------------------- #
def test_caso_c_sin_capital_muestra_100000(proyecto):
    ruta = escribir(proyecto, "vector.py", VECTOR)

    assert app.reglas_comunes_de_estrategia(ruta)[1] == 100_000.0 == app.DEFAULT_CAPITAL


def test_caso_c_el_capital_del_archivo_manda(proyecto):
    """Un capital distinto al de respaldo no se pisa con 100000."""
    ruta = escribir(proyecto, "lean.py", lean("self.set_cash(7500)"))

    assert app.reglas_comunes_de_estrategia(ruta)[1] == 7_500.0


# --------------------------------------------------------------------------- #
# CASO D · cambiar de estrategia principal actualiza ticker y capital
# --------------------------------------------------------------------------- #
def test_caso_d_cambiar_de_principal_reinicia_los_dos_campos(proyecto):
    escribir(proyecto, "vector.py", VECTOR)
    escribir(proyecto, "lean.py", lean(
        "self.set_cash(25000)",
        'self.add_equity("SOXL", Resolution.Daily)',
    ))

    assert app.sincronizar_reglas_comunes("vector.py") == (app.DEFAULT_TICKER, app.DEFAULT_CAPITAL)

    app.st.session_state[app.PRINCIPAL_KEY] = "lean.py"
    assert app.sincronizar_reglas_comunes("lean.py") == ("SOXL", 25_000.0)

    app.st.session_state[app.PRINCIPAL_KEY] = "vector.py"
    assert app.sincronizar_reglas_comunes("vector.py") == (app.DEFAULT_TICKER, app.DEFAULT_CAPITAL)


def test_caso_d_la_edicion_manual_no_se_pisa(proyecto):
    """Sin cambio de Principal, lo que el usuario escriba se respeta."""
    escribir(proyecto, "vector.py", VECTOR)
    app.sincronizar_reglas_comunes("vector.py")

    app.st.session_state[app.ACTIVO_KEY] = "AAPL"
    app.st.session_state[app.CAPITAL_KEY] = 42_000.0

    assert app.sincronizar_reglas_comunes("vector.py") == ("AAPL", 42_000.0)


def test_ticker_del_archivo_siempre_es_elegible(proyecto):
    """Un ticker fuera de la lista corta (SOXL) tiene que poder mostrarse."""
    escribir(proyecto, "lean.py", lean('self.add_equity("SOXL", Resolution.Daily)'))
    ticker, _ = app.sincronizar_reglas_comunes("lean.py")

    opciones = app.opciones_activo_ui(ticker)

    assert ticker == "SOXL" and opciones[0] == "SOXL"
    assert len(opciones) == len(set(opciones))
    assert set(app.ASSETS) <= set(opciones)


def test_activo_no_se_confunde_con_el_benchmark_ni_con_la_reserva(proyecto):
    """SPY es referencia y SHV es la reserva: ninguno puede ser el Activo / Ticker."""
    ruta = escribir(proyecto, "dual.py", lean(
        'self.set_benchmark("SPY")',
        'self.add_equity("TQQQ", Resolution.Daily)',
        'self.add_equity("SHV", Resolution.Daily)',
        "self.set_cash(5000)",
    ))

    ticker, capital = app.reglas_comunes_de_estrategia(ruta)

    assert app.tickers_declarados(ruta) == ["SPY", "TQQQ", "SHV"]
    assert ticker == "TQQQ"
    assert capital == 5_000.0


def test_indice_como_unico_activo_se_conserva(proyecto):
    """Si el archivo solo declara un indice, ese indice es su instrumento."""
    ruta = escribir(proyecto, "sp500.py", lean('self.add_equity("SPY", Resolution.Minute)'))

    assert app.reglas_comunes_de_estrategia(ruta)[0] == "SPY"


def test_archivo_no_analizable_cae_en_el_respaldo(proyecto):
    roto = escribir(proyecto, "roto.py", "def run_strategy(data:\n")

    assert app.tickers_declarados(roto) == []
    assert app.reglas_comunes_de_estrategia(roto) == (app.DEFAULT_TICKER, app.DEFAULT_CAPITAL)


def test_archivo_inexistente_cae_en_el_respaldo(proyecto):
    assert app.tickers_declarados(proyecto / "no_existe.py") == []


# --------------------------------------------------------------------------- #
# GRUPO 2 · ciclo de vida de las comparativas
# --------------------------------------------------------------------------- #
@pytest.fixture
def catalogo(monkeypatch):
    monkeypatch.setattr(app.st, "rerun", lambda *a, **k: None)
    monkeypatch.setitem(app.st.session_state, app.PRINCIPAL_KEY, "vector.py")
    return ["vector.py", "a.py", "b.py", "c.py"]


def test_alta_de_comparativa_no_toca_la_principal(catalogo, monkeypatch):
    monkeypatch.setitem(app.st.session_state, app.COMPARE_KEY, [])

    ok, mensaje = app.add_comparative_strategy("a.py", catalogo)

    assert ok and "a.py" in mensaje
    assert app.st.session_state[app.COMPARE_KEY] == ["a.py"]
    assert app.st.session_state[app.PRINCIPAL_KEY] == "vector.py"


def test_dos_comparativas_caben_una_tercera_no(catalogo, monkeypatch):
    monkeypatch.setitem(app.st.session_state, app.COMPARE_KEY, [])
    app.add_comparative_strategy("a.py", catalogo)
    app.add_comparative_strategy("b.py", catalogo)

    ok, mensaje = app.add_comparative_strategy("c.py", catalogo)

    assert not ok and str(app.MAX_COMPARATIVES) in mensaje
    assert app.st.session_state[app.COMPARE_KEY] == ["a.py", "b.py"]


def test_la_principal_no_se_compara_consigo_misma(catalogo, monkeypatch):
    monkeypatch.setitem(app.st.session_state, app.COMPARE_KEY, [])

    ok, mensaje = app.add_comparative_strategy("vector.py", catalogo)

    assert not ok and "Principal" in mensaje
    assert app.st.session_state[app.COMPARE_KEY] == []


def test_comparativa_repetida_o_inexistente_se_rechaza(catalogo, monkeypatch):
    monkeypatch.setitem(app.st.session_state, app.COMPARE_KEY, ["a.py"])

    assert app.add_comparative_strategy("a.py", catalogo)[0] is False
    assert app.add_comparative_strategy("fuera.py", catalogo)[0] is False
    assert app.st.session_state[app.COMPARE_KEY] == ["a.py"]


def test_baja_individual_por_pending_checks(catalogo, monkeypatch):
    """La baja de un hueco llega por `pending["checks"]` y deja el resto intacto."""
    monkeypatch.setitem(app.st.session_state, app.COMPARE_KEY, ["a.py", "b.py"])
    avisos = []
    monkeypatch.setattr(app.st, "caption", lambda texto=None, *a, **k: avisos.append(texto))

    principal = app.aplicar_gestor_pendientes(
        catalogo, "vector.py", {"star": None, "checks": {"a.py": False}}
    )

    assert principal == "vector.py"
    assert app.st.session_state[app.COMPARE_KEY] == ["b.py"]
    assert any("a.py" in aviso for aviso in avisos if aviso)


def test_sin_clics_el_estado_de_comparativas_no_se_toca(catalogo, monkeypatch):
    monkeypatch.setitem(app.st.session_state, app.COMPARE_KEY, ["a.py"])

    app.aplicar_gestor_pendientes(catalogo, "vector.py", {"star": None, "checks": {}})

    assert app.st.session_state[app.COMPARE_KEY] == ["a.py"]


def test_el_reparto_de_comparativas_respeta_el_tope(monkeypatch):
    """`ensure_strategy_state` sigue recortando a MAX_COMPARATIVES."""
    disponibles = ["vector.py", "a.py", "b.py", "c.py"]
    monkeypatch.setitem(app.st.session_state, app.PRINCIPAL_KEY, "vector.py")
    monkeypatch.setitem(app.st.session_state, app.COMPARE_KEY, ["a.py", "b.py", "c.py"])

    app.ensure_strategy_state(disponibles)

    assert app.st.session_state[app.COMPARE_KEY] == ["b.py", "c.py"]


def test_la_principal_se_saca_de_las_comparativas(monkeypatch):
    disponibles = ["vector.py", "a.py", "b.py"]
    monkeypatch.setitem(app.st.session_state, app.PRINCIPAL_KEY, "vector.py")
    monkeypatch.setitem(app.st.session_state, app.COMPARE_KEY, ["a.py", "b.py"])

    app.ensure_strategy_state(disponibles)
    app.set_principal_strategy("a.py", disponibles)

    assert app.st.session_state[app.PRINCIPAL_KEY] == "a.py"
    assert app.st.session_state[app.COMPARE_KEY] == ["b.py"]


# --------------------------------------------------------------------------- #
# GRUPO 1 · estado vacio, asignacion y baja de la Estrategia Principal
# --------------------------------------------------------------------------- #
def test_el_grupo_1_arranca_vacio(monkeypatch):
    """Sin Principal previa el estado queda vacio: no se promueve el primer archivo."""
    monkeypatch.setitem(app.st.session_state, app.PRINCIPAL_KEY, None)

    app.ensure_strategy_state(["vector.py", "a.py"])

    assert app.st.session_state[app.PRINCIPAL_KEY] is None


def test_una_principal_borrada_del_proyecto_deja_el_grupo_vacio(monkeypatch):
    """Si el archivo de la Principal ya no esta, tampoco se elige otro en su lugar."""
    monkeypatch.setitem(app.st.session_state, app.PRINCIPAL_KEY, "se_fue.py")

    app.ensure_strategy_state(["vector.py", "a.py"])

    assert app.st.session_state[app.PRINCIPAL_KEY] is None


def test_el_bote_de_basura_no_toca_los_archivos_ni_las_comparativas(proyecto, monkeypatch):
    """La baja de la Principal solo deshace la asignacion en memoria."""
    ruta = escribir(proyecto, "vector.py", VECTOR)
    monkeypatch.setitem(app.st.session_state, app.COMPARE_KEY, ["a.py"])
    monkeypatch.setitem(app.st.session_state, "script_nombre", "vector.py")
    monkeypatch.setitem(app.st.session_state, app.RESULT_KEY, {"df": None, "metrics": {}})

    app.clear_principal_strategy()

    # El archivo sigue en el disco, byte a byte.
    assert ruta.exists() and ruta.read_text(encoding="utf-8") == VECTOR
    # El estado de sesion queda vacio y las comparativas intactas.
    assert app.st.session_state[app.PRINCIPAL_KEY] is None
    assert app.st.session_state["script_nombre"] == ""
    assert app.st.session_state[app.COMPARE_KEY] == ["a.py"]
    # Y no queda ningun resultado de la Principal que se quitaba.
    assert app.RESULT_KEY not in app.st.session_state


def test_sin_principal_el_backtest_no_se_ejecuta(monkeypatch):
    """Sin Principal el laboratorio espera: no corre ni la anterior ni ningun archivo."""
    monkeypatch.setitem(app.st.session_state, app.PRINCIPAL_KEY, None)
    ejecutadas = []
    monkeypatch.setattr(app, "run_strategy_backtest", lambda *a, **k: ejecutadas.append(a))

    df, metrics = app.get_backtest_result("TSLA", 100_000.0)

    assert ejecutadas == []
    assert metrics["standby"] is True
    assert metrics["equity"] == 100_000.0
    assert list(df.columns) == ["Equity", "Benchmark", "Drawdown", "Position"]


def test_sin_principal_no_se_toca_la_cache_de_una_ejecucion_anterior(monkeypatch):
    """Quitar la Principal invalida su resultado: no puede reaparecer al reejecutar."""
    monkeypatch.setitem(app.st.session_state, app.PRINCIPAL_KEY, "vector.py")
    monkeypatch.setitem(app.st.session_state, app.RESULT_KEY, {"df": None, "metrics": {}})
    monkeypatch.setitem(app.st.session_state, app.PARAMS_KEY, ("vector.py", "TSLA", (), 1.0))

    app.clear_principal_strategy()

    assert app.RESULT_KEY not in app.st.session_state
    assert app.PARAMS_KEY not in app.st.session_state


def test_cache_principal_se_invalida_al_cambiar_codigo_en_la_misma_ruta(
    proyecto, monkeypatch
):
    ruta = escribir(proyecto, "vector.py", VECTOR)
    monkeypatch.setattr(app, "PARAMETROS_ACTIVOS", {})
    monkeypatch.delitem(app.st.session_state, app.RESULT_KEY, raising=False)
    monkeypatch.delitem(app.st.session_state, app.PARAMS_KEY, raising=False)
    ejecuciones = []

    def ejecutar(script, ticker, capital):
        ejecuciones.append((script, ticker, capital))
        return f"resultado-{len(ejecuciones)}", {"ejecucion": len(ejecuciones)}

    monkeypatch.setattr(app, "run_strategy_backtest", ejecutar)

    primero = app.get_backtest_result("TSLA", 100_000.0)
    segundo = app.get_backtest_result("TSLA", 100_000.0)
    assert primero == segundo
    assert len(ejecuciones) == 1

    ruta.write_text(VECTOR + "\n# nueva version\n", encoding="utf-8")
    tercero = app.get_backtest_result("TSLA", 100_000.0)
    assert tercero[0] == "resultado-2"
    assert len(ejecuciones) == 2

    app.get_backtest_result("TSLA", 100_000.0, force=True)
    assert len(ejecuciones) == 3


def test_cache_comparativa_se_invalida_al_cambiar_codigo_en_la_misma_ruta(
    proyecto, monkeypatch
):
    ruta = escribir(proyecto, "comparativa.py", VECTOR)
    monkeypatch.setattr(app, "ALL_STRATEGIES", ["comparativa.py"])
    monkeypatch.setattr(app, "PARAMETROS_ACTIVOS", {})
    monkeypatch.setitem(app.st.session_state, app.COMPARE_CACHE_KEY, {})
    ejecuciones = []

    def ejecutar(script, ticker, capital):
        ejecuciones.append((script, ticker, capital))
        return f"curva-{len(ejecuciones)}", {"ejecucion": len(ejecuciones)}

    monkeypatch.setattr(app, "run_comparative_backtest", ejecutar)

    primero = app.get_comparative_results(
        ["comparativa.py"], "TSLA", 100_000.0
    )
    segundo = app.get_comparative_results(
        ["comparativa.py"], "TSLA", 100_000.0
    )
    assert primero["comparativa.py"] == segundo["comparativa.py"]
    assert len(ejecuciones) == 1

    ruta.write_text(VECTOR + "\n# nueva version\n", encoding="utf-8")
    tercero = app.get_comparative_results(
        ["comparativa.py"], "TSLA", 100_000.0
    )
    assert tercero["comparativa.py"]["df"] == "curva-2"
    assert len(ejecuciones) == 2