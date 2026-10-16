"""
Prueba funcional del grupo 1 · 📁 ESTRATEGIAS tal y como se ve en pantalla (G.3.7).

Recorre las cinco validaciones del grupo sobre una sesion real de la app con `AppTest`:
estado vacio al arrancar, carga por el modal de siempre, baja con el 🗑️, vuelta a abrir
otra estrategia y, sin Principal, ningun backtest en marcha.

Limite conocido de `AppTest`: un `st.dialog` solo se pinta en el run en el que se pulse
su boton de apertura, asi que el segundo clic (el de confirmar dentro del modal) llega a
un run sin dialog y se pierde. Por eso la confirmacion se comprueba sobre el mismo modal
que abre el boton («📥 Cargar en Editor») y la mutacion del estado por la funcion que ese
modal llama (`set_principal_strategy`), que ademas cubren las pruebas unitarias.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import app  # noqa: E402  (el path del proyecto tiene que estar antes)
import pytest  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

DUAL_MOMENTUM = "otras/Dual Momentum + Volatility Targeting.py"
SP500 = "otras/SP500.py"


@pytest.fixture(scope="module")
def at():
    sesion = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=180).run()
    assert not sesion.exception, [e.value for e in sesion.exception]
    return sesion


def sin_excepciones(sesion: AppTest) -> None:
    assert not sesion.exception, [e.value for e in sesion.exception]


def sidebar(sesion: AppTest) -> str:
    return " ".join(m.value for m in sesion.sidebar.markdown)


def tarjetas(sesion: AppTest) -> list[str]:
    return [m.value for m in sesion.sidebar.markdown if "principal-card" in m.value]


def claves_sidebar(sesion: AppTest) -> list[str]:
    return [b.key for b in sesion.sidebar.button]


def abrir_modal(sesion: AppTest, boton: str, carpeta: str, archivo: str) -> AppTest:
    """Abre el modal que reutiliza `boton` con un archivo ya elegido."""
    sesion.session_state["modal_folder_ready"] = True
    sesion.session_state["modal_folder"] = carpeta
    sesion.session_state["modal_folder_resolved"] = carpeta
    sesion.session_state["modal_file"] = archivo
    return sesion.button(key=boton).click().run()


def cargar(sesion: AppTest, script: str) -> AppTest:
    """Estado que queda tras confirmar la carga en el modal de «Abrir Estrategia»."""
    sesion.session_state[app.PRINCIPAL_KEY] = script
    return sesion.run()


# --------------------------------------------------------------------------- #
# A · estado inicial vacio
# --------------------------------------------------------------------------- #
def test_validacion_a_el_grupo_arranca_vacio(at):
    # No hay Principal: ni `crawling_fluorescent.py` ni ninguna otra.
    assert at.session_state[app.PRINCIPAL_KEY] is None
    assert "crawling_fluorescent.py" not in sidebar(at)
    assert not (RAIZ / "crawling_fluorescent.py").name in sidebar(at)

    # Lo que se ve es el marcador del estado vacio, no una tarjeta con nombre.
    assert len(tarjetas(at)) == 1
    assert tarjetas(at)[0] == (
        '<div class="principal-card empty">Ninguna estrategia seleccionada</div>'
    )

    # El acceso compacto de apertura y el subtitulo de la Principal siguen en el grupo.
    assert '<div class="strategy-subtitle">⭐ Estrategia Principal</div>' in sidebar(at)
    assert at.button(key="btn_open_strategy").label == "📂 Abrir Estrategia"
    prototype = " ".join(item.value for item in at.sidebar.markdown)
    assert "PROTOTIPO — ESTRATEGIA PRINCIPAL" in prototype
    assert "Ninguna estrategia seleccionada" in prototype
    assert 'class="principal-prototype-delete" type="button"' in prototype
    assert 'title="Quitar estrategia principal"' in prototype
    assert "principal-prototype-delete" not in claves_sidebar(at)

    # Sin Principal no hay 🗑️ que pulsar todavia.
    assert "btn_quitar_principal" not in claves_sidebar(at)
    sin_excepciones(at)


# --------------------------------------------------------------------------- #
# B · cargar por el modal de siempre
# --------------------------------------------------------------------------- #
def test_validacion_b_abrir_estrategia_sigue_siendo_el_modal_de_siempre(at):
    sesion = abrir_modal(at, "btn_open_strategy", "otras", "Dual Momentum + Volatility Targeting.py")

    # El modal es el de siempre y su accion final sigue siendo cargar en el editor.
    assert sesion.selectbox(key="modal_folder").value == "otras"
    assert sesion.selectbox(key="modal_file").value == "Dual Momentum + Volatility Targeting.py"
    assert sesion.button(key="modal_load").label == "📥 Cargar en Editor"
    assert at.session_state[app.MODAL_MODO_KEY] == app.MODO_PRINCIPAL
    # Abrir el modal no ha elegido nada todavia.
    assert at.session_state[app.PRINCIPAL_KEY] is None
    sin_excepciones(sesion)


def test_validacion_b_la_estrategia_cargada_aparece_como_principal(at):
    sesion = cargar(at, DUAL_MOMENTUM)

    assert sesion.session_state[app.PRINCIPAL_KEY] == DUAL_MOMENTUM
    # Tarjeta compacta con el nombre entero en el DOM (recortado con elipsis) y la ruta
    # completa en el `title`, que es lo que se ve al pasar el cursor.
    tarjeta = tarjetas(sesion)[0]
    assert 'class="principal-card"' in tarjeta
    assert 'title="otras/Dual Momentum + Volatility Targeting.py"' in tarjeta
    assert ">Dual Momentum + Volatility Targeting.py<" in tarjeta
    assert "Ninguna estrategia seleccionada" not in tarjeta
    # El 🗑️ de la tarjeta aparece y el resto del grupo no se duplica.
    assert "btn_quitar_principal" in claves_sidebar(sesion)
    assert sidebar(sesion).count("Estrategia Principal") == 1
    prototype = " ".join(item.value for item in sesion.sidebar.markdown)
    assert "PROTOTIPO — ESTRATEGIA PRINCIPAL" in prototype
    assert "Dual Momentum + Volatility Targeting.py" in prototype
    assert "principal-prototype-delete" not in claves_sidebar(sesion)
    sesion = sesion.run()
    assert sesion.session_state[app.PRINCIPAL_KEY] == DUAL_MOMENTUM
    # El grupo 3 sigue leyendo el archivo nuevo (SOXL y 500 los declara ese .py).
    assert sesion.selectbox(key=app.ACTIVO_KEY).value == "SOXL"
    sin_excepciones(sesion)


# --------------------------------------------------------------------------- #
# C · el 🗑️ quita la Principal y NO borra el archivo
# --------------------------------------------------------------------------- #
def test_validacion_c_el_bote_de_basura_devuelve_el_estado_vacio(at):
    antes = (RAIZ / DUAL_MOMENTUM).read_bytes()
    sesion = at.button(key="btn_quitar_principal").click().run()

    assert sesion.session_state[app.PRINCIPAL_KEY] is None
    assert tarjetas(sesion) == [
        '<div class="principal-card empty">Ninguna estrategia seleccionada</div>'
    ]
    assert "Dual Momentum" not in sidebar(sesion)
    assert "btn_quitar_principal" not in claves_sidebar(sesion)
    # El archivo del proyecto sigue exactamente igual.
    assert (RAIZ / DUAL_MOMENTUM).read_bytes() == antes
    sin_excepciones(sesion)


# --------------------------------------------------------------------------- #
# D · volver a abrir otra estrategia
# --------------------------------------------------------------------------- #
def test_validacion_d_volver_a_abrir_otra_estrategia(at):
    sesion = abrir_modal(at, "btn_open_strategy", "otras", "SP500.py")
    assert sesion.button(key="modal_load").label == "📥 Cargar en Editor"

    sesion = cargar(sesion, SP500)

    assert sesion.session_state[app.PRINCIPAL_KEY] == SP500
    assert ">SP500.py<" in tarjetas(sesion)[0]
    assert "Ninguna estrategia seleccionada" not in tarjetas(sesion)[0]
    assert "btn_quitar_principal" in claves_sidebar(sesion)
    sin_excepciones(sesion)


# --------------------------------------------------------------------------- #
# E · sin Principal no se ejecuta nada
# --------------------------------------------------------------------------- #
def test_validacion_e_sin_principal_el_laboratorio_espera(at):
    at = at.radio(key="workspace_tab").set_value("LABORATORIO").run()

    # Un par de comparativas para comprobar que la baja no las toca.
    at.session_state[app.COMPARE_KEY] = [DUAL_MOMENTUM]
    sesion = at.button(key="btn_quitar_principal").click().run()

    assert sesion.session_state[app.PRINCIPAL_KEY] is None
    # No queda ningun resultado en curso ni parametros de la Principal que se fue.
    assert app.RESULT_KEY not in sesion.session_state
    assert app.PARAMS_KEY not in sesion.session_state
    assert app.DRAFT_KEY not in sesion.session_state
    # Las comparativas siguen ahí: comparar no necesita Principal.
    assert sesion.session_state[app.COMPARE_KEY] == [DUAL_MOMENTUM]
    # El aviso de espera aparece y el terminal no finge haber cargado datos.
    assert any("Backtest en espera" in i.value for i in sesion.info)
    terminal = " ".join(m.value for m in sesion.markdown)
    assert "No strategy loaded" in terminal
    assert "daily bars" not in terminal
    # Guardar queda cerrado: sin archivo al que escribir.
    assert sesion.button(key="btn_save").disabled
    assert sesion.button(key="btn_save_as").disabled
    sin_excepciones(sesion)