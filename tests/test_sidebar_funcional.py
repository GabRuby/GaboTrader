"""
Prueba funcional de los grupos 1, 2 y 3 tal y como se ven en pantalla (G.3.6).

Recorre los nueve pasos de la fase sobre una sesion real de la app con `AppTest`:
cargar la Principal, verificar ticker y capital, sumar comparativas hasta el tope,
intentar una tercera, quitar una y cambiar de Principal.

Limite conocido de `AppTest`: un `st.dialog` solo se pinta en el run en el que se
pulse su boton de apertura, asi que el segundo clic (el de confirmar dentro del
modal) llega a un run sin dialog y se pierde. Por eso el alta se comprueba sobre el
mismo modal que abre el boton (mismo `modal_folder` / `modal_file`, accion
comparativa) y la mutacion del estado por la funcion que ese modal llama
(`add_comparative_strategy`), que ademas cubren las pruebas unitarias.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import app  # noqa: E402  (el path del proyecto tiene que estar antes)
import pytest  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

# Estrategias reales del proyecto con datos declarados distintos entre si, para que
# cambiar de Principal se note en los dos campos del grupo 3.
DUAL_MOMENTUM = "otras/Dual Momentum + Volatility Targeting.py"
COMPARATIVA_A = "otras/SP500.py"
COMPARATIVA_B = "otras/SOXL.py"


@pytest.fixture(scope="module")
def at():
    sesion = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=180).run()
    assert not sesion.exception, [e.value for e in sesion.exception]
    return sesion


def sin_excepciones(sesion: AppTest) -> None:
    assert not sesion.exception, [e.value for e in sesion.exception]


def abrir_modal(sesion: AppTest, boton: str, carpeta: str, archivo: str) -> AppTest:
    """Abre el modal que reutiliza `boton` con un archivo ya elegido."""
    sesion.session_state["modal_folder_ready"] = True
    sesion.session_state["modal_folder"] = carpeta
    sesion.session_state["modal_folder_resolved"] = carpeta
    sesion.session_state["modal_file"] = archivo
    return sesion.button(key=boton).click().run()


def captions_del_sidebar(sesion: AppTest) -> list[str]:
    return [c.value for c in sesion.sidebar.caption]


def marcadores_del_sidebar(sesion: AppTest) -> list[str]:
    return [m.value for m in sesion.sidebar.markdown]


def poner_comparativas(sesion: AppTest, *scripts: str) -> AppTest:
    """Deja las comparativas que habria tras confirmarlas en el modal y repinta."""
    sesion.session_state[app.COMPARE_KEY] = list(scripts)
    return sesion.run()


def poner_principal(sesion: AppTest, script: str) -> AppTest:
    """Promueve `script` a Principal y repinta el panel entero."""
    sesion.session_state[app.PRINCIPAL_KEY] = script
    return sesion.run()


# --------------------------------------------------------------------------- #
# 1-3 · Principal cargada + ticker y capital del grupo 3
# --------------------------------------------------------------------------- #
def test_pasos_1_a_3_principal_ticker_y_capital(at):
    # 1 · la app arranca con una Principal ya seleccionada
    assert at.session_state["principal_script"]
    assert "compare_selection" in at.session_state
    assert f"{at.session_state['principal_script']}" in " ".join(marcadores_del_sidebar(at))

    # 2 y 3 · el grupo 3 muestra el ticker y el capital que declara el archivo.
    # `crawling_fluorescent.py` no declara ninguno de los dos datos, asi que aqui
    # se ven los respaldos TSLA y 100000.
    selector = at.selectbox(key=app.ACTIVO_KEY)
    capital = at.number_input(key=app.CAPITAL_KEY)
    declarados = app.tickers_declarados(app.BASE_DIR / at.session_state["principal_script"])

    assert selector.value == ("TSLA" if not declarados else declarados[0])
    assert capital.value == 100_000.0
    assert at.session_state[app.ACTIVO_KEY] == selector.value
    assert at.session_state[app.CAPITAL_KEY] == capital.value
    # El respaldo es el unico valor cuando el archivo no declara el dato.
    if not declarados:
        assert selector.value == app.DEFAULT_TICKER
        assert capital.value == app.DEFAULT_CAPITAL


# --------------------------------------------------------------------------- #
# 4-6 · comparativas: una, dos y la tercera rechazada
# --------------------------------------------------------------------------- #
def test_paso_4_el_boton_reutiliza_el_modal_de_estrategias(at):
    sesion = abrir_modal(at, "btn_add_comparativa", "otras", "SP500.py")

    # El modal es el mismo que abre «Abrir Estrategia»: mismos selectores de carpeta y
    # de archivo, unica diferencia la accion final.
    assert sesion.selectbox(key="modal_folder").value == "otras"
    assert sesion.selectbox(key="modal_file").value == "SP500.py"
    assert sesion.button(key="modal_load").label == "⚖️ Agregar a comparar"
    assert at.session_state[app.MODAL_MODO_KEY] == app.MODO_COMPARATIVA
    # Abrir el modal no ha cambiado la Principal.
    assert sesion.session_state[app.PRINCIPAL_KEY] == at.session_state[app.PRINCIPAL_KEY]

    # Una vez confirmada en el modal, la comparativa ocupa el hueco 1 y la Principal
    # sigue siendo la misma.
    principal_antes = at.session_state[app.PRINCIPAL_KEY]
    poner_comparativas(at, COMPARATIVA_A)

    assert at.session_state[app.COMPARE_KEY] == [COMPARATIVA_A]
    assert at.session_state[app.PRINCIPAL_KEY] == principal_antes
    assert any(COMPARATIVA_A in md for md in marcadores_del_sidebar(at))
    assert any("1 de 2" in c for c in captions_del_sidebar(at))
    assert not at.button(key="btn_add_comparativa").disabled


def test_paso_5_segunda_comparativa(at):
    poner_comparativas(at, COMPARATIVA_A, COMPARATIVA_B)

    assert at.session_state[app.COMPARE_KEY] == [COMPARATIVA_A, COMPARATIVA_B]
    assert not at.button(key="rm_comparativa_0").disabled
    assert not at.button(key="rm_comparativa_1").disabled
    assert any("Ya no caben" in c for c in captions_del_sidebar(at))
    assert not any("vacía" in md for md in marcadores_del_sidebar(at))


def test_paso_6_una_tercera_comparativa_no_se_acepta(at):
    """Con los dos huecos llenos el boton de alta esta cerrado y no caben mas."""
    assert at.button(key="btn_add_comparativa").disabled
    assert at.session_state[app.COMPARE_KEY] == [COMPARATIVA_A, COMPARATIVA_B]
    assert len(at.session_state[app.COMPARE_KEY]) == app.MAX_COMPARATIVES


# --------------------------------------------------------------------------- #
# 7 · baja individual
# --------------------------------------------------------------------------- #
def test_paso_7_eliminar_una_comparativa(at):
    at.button(key="rm_comparativa_0").click().run()

    assert at.session_state[app.COMPARE_KEY] == [COMPARATIVA_B]
    assert any(COMPARATIVA_B in md for md in marcadores_del_sidebar(at))
    assert any("1 de 2" in c for c in captions_del_sidebar(at))
    assert not at.button(key="btn_add_comparativa").disabled
    # Principal intacta: quitar una comparativa no cambia lo que se esta evaluando.
    assert at.session_state[app.PRINCIPAL_KEY]


# --------------------------------------------------------------------------- #
# 8-9 · cambio de Principal: ticker y capital se actualizan
# --------------------------------------------------------------------------- #
def test_pasos_8_y_9_cambiar_de_principal_actualiza_ticker_y_capital(at):
    # El boton del grupo 1 sigue siendo el que abre el modal de carga.
    sesion = abrir_modal(at, "btn_open_strategy", "otras", "Dual Momentum + Volatility Targeting.py")
    assert sesion.button(key="modal_load").label == "📥 Cargar en Editor"
    assert at.session_state[app.MODAL_MODO_KEY] == app.MODO_PRINCIPAL

    # Estado tras confirmar la carga en el modal.
    poner_principal(at, DUAL_MOMENTUM)
    sin_excepciones(at)

    assert at.session_state[app.PRINCIPAL_KEY] == DUAL_MOMENTUM
    assert at.session_state[app.ACTIVO_ORIGEN_KEY] == DUAL_MOMENTUM
    # `Dual Momentum` declara ["SOXL", "TQQQ"] como riesgo (SPY es el filtro macro y SHV
    # la reserva) y un capital de 500: los dos campos tienen que reflejarlo.
    assert at.selectbox(key=app.ACTIVO_KEY).value == "SOXL"
    assert at.number_input(key=app.CAPITAL_KEY).value == 500.0
    assert at.session_state[app.CAPITAL_ORIGEN_KEY] == DUAL_MOMENTUM
    # SOXL no esta en la lista corta de ASSETS, asi que tiene que haberse antepuesto.
    assert "SOXL" in at.selectbox(key=app.ACTIVO_KEY).options
    assert at.selectbox(key=app.ACTIVO_KEY).options[0] == "SOXL"
    # Y la comparativa que quedaba sigue en su hueco.
    assert at.session_state[app.COMPARE_KEY] == [COMPARATIVA_B]