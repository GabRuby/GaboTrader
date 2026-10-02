"""
Analisis estatico (AST) de los archivos de estrategia `.py`.

Antes de ejecutar nada hay que responder a varias preguntas:

  * Que clase implementa el algoritmo y cuando se ejecuta (`initialize`).
  * De que fechas y capital parte el backtest.
  * Que valores compra y con que resolucion.
  * Que librerias usa, que indicadores define y a que calendario programa eventos.
  * Que numeros magicosicianos son tunables, para exponerlos como sliders.

El analisis es puramente estatico (no ejecuta el codigo de la estrategia), de
modo que es seguro abrir cualquier archivo del proyecto para inspeccionarlo.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Nombres que parecen tickers: 1-5 letras mayusculas (o con punto/guion corto).
_RE_TICKER = re.compile(r"^[A-Z]{1,5}(-[A-Z]{1,2})?$")

# Palabras clave -> categoria, para agrupar los sliders en la seccion PARAMETERS.
# La comparacion se hace por TOKENS (ver `_categorizar`), no por subcadena.
_CATEGORIAS: List[Tuple[str, str]] = [
    ("stop", "Stop Loss"), ("sl", "Stop Loss"), ("proteccion", "Stop Loss"),
    ("take", "Take Profit"), ("objetivo", "Take Profit"), ("tp", "Take Profit"),
    ("sma", "Medias Moviles"), ("ema", "Medias Moviles"), ("media", "Medias Moviles"),
    ("lookback", "Ventanas"), ("periodo", "Ventanas"), ("window", "Ventanas"),
    ("momentum", "Momentum"), ("vol_target", "Volatilidad"), ("volatil", "Volatilidad"),
    ("peso", "Asignacion"), ("capital", "Capital"), ("banda", "Rebalanceo"),
    ("min_orden", "Rebalanceo"), ("margen", "Rebalanceo"), ("histeresis", "Rebalanceo"),
    ("escalon", "Escalonado"), ("max_escalon", "Escalonado"),
    ("semanal", "Calendario"), ("diario", "Calendario"), ("mensual", "Calendario"),
]

# Identificadores de infraestructura / estado interno: nunca son parametros de
# la estrategia, aunque su valor sea numerico, y no deben aparecer como slider.
_NO_AJUSTABLE = {
    "random_state", "n_estimators", "max_depth", "max_features", "min_samples_leaf",
    "num_fills", "volumen_usd", "conteo_escalones", "precio_entrada", "stop_loss_precio",
    "reserva_fija", "ultimo_precio_compra", "capital_inicial", "peso_minimo_ref",
}


@dataclass
class Parametro:
    """Un numero found en el codigo que el dashboard puede convertir en slider."""

    nombre: str
    valor: float
    categoria: str = "General"
    origen: str = "atributo"     # 'param' si viene de _param(), si no literal
    es_entero: bool = False

    @property
    def clave(self) -> str:
        return self.nombre


@dataclass
class EstrategiaInfo:
    """Todo lo que se puede saber de un archivo de estrategia sin ejecutarlo."""

    ruta: str = ""
    nombre_archivo: str = ""
    clase: str = ""
    docstring: str = ""
    fecha_inicio: Optional[str] = None
    fecha_fin: Optional[str] = None
    capital: Optional[float] = None
    warmup: int = 0
    warmup_resolucion: str = "Daily"
    benchmark: Optional[str] = None
    simbolos: List[str] = field(default_factory=list)
    resolucion_principal: str = "Daily"
    indicadores: List[Tuple[str, str, int, str]] = field(default_factory=list)
    librerias: List[str] = field(default_factory=list)
    parametros: List[Parametro] = field(default_factory=list)
    n_ordenes: int = 0
    n_liquidaciones: int = 0
    n_set_holdings: int = 0
    n_eventos: int = 0
    eventos_calendario: int = 0
    metodos: List[str] = field(default_factory=list)
    usa_ia: bool = False
    lineas: int = 0
    valido: bool = False
    error: str = ""

    # --- derivados -------------------------------------------------- #
    @property
    def tickers_riesgo(self) -> List[str]:
        """
        Activos de riesgo de la estrategia.

        SPY y SHV se clasifican como macro/reserva SOLO si hay otros activos en el
        universo: en una estrategia que opera unicamente SPY, ese SPY es el propio
        activo arriesgado y no debe quedar excluido.
        """
        macro = {"SPY", "SPX", "QQQ", "IWM", "VIX", "DIA", "MDY"}
        refugio = {"SHV", "SHY", "BIL", "SGOV", "TLT", "IEF", "GLD"}
        if len(self.simbolos) <= 1:
            return list(self.simbolos)
        return [s for s in self.simbolos if s not in macro and s not in refugio]

    @property
    def benchmark_efectivo(self) -> str:
        return self.benchmark or "SPY"

    def parametros_por_categoria(self) -> Dict[str, List[Parametro]]:
        grupos: Dict[str, List[Parametro]] = {}
        for p in self.parametros:
            grupos.setdefault(p.categoria, []).append(p)
        return grupos


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def _nodo_nombre(nodo: ast.AST) -> str:
    """Nombre dibujable de un Call/Attribute/Subscript (p.ej. `self.sma`)."""
    partes: List[str] = []
    actual = nodo
    while isinstance(actual, ast.Attribute):
        partes.append(actual.attr)
        actual = actual.value
    if isinstance(actual, ast.Name):
        partes.append(actual.id)
    elif isinstance(actual, ast.Call):
        partes.append(_nodo_nombre(actual.func))
    elif isinstance(actual, ast.Subscript):
        partes.append(_nodo_nombre(actual.value))
    return ".".join(reversed(partes))


def _literal(node: Optional[ast.AST]) -> Any:
    """Evalua un literal de forma segura. Devuelve None si no lo es."""
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _primer_lista_strings(node: Optional[ast.AST]) -> List[str]:
    """Extrae ['SOXL', 'TQQQ'] de una asignacion a self.activos."""
    valor = _literal(node)
    if isinstance(valor, (list, tuple)):
        return [str(v).upper() for v in valor if isinstance(v, (str, int, float))]
    if isinstance(valor, str):
        return [valor.upper()]
    return []


def _categorizar(nombre: str) -> str:
    """
    Clasifica un parametro por categoria semantica.

    Se comparan TOKENS (no subcadenas) porque en español los nombres se solapan:
    `eval_semanal` contiene "ema", lo que haria que un parametro de calendario
    acabara etiquetado como "Medias Moviles".
    """
    tokens = {t for t in re.split(r"[^a-z0-9]+", nombre.lower()) if t}
    for clave, etiqueta in _CATEGORIAS:
        if clave in tokens:
            return etiqueta
    # Segunda pasada: claves compuestas que solo aparecen dentro de un token
    # (por ejemplo "stoploss" escrito sin guion).
    compacto = "".join(sorted(tokens))
    for clave, etiqueta in _CATEGORIAS:
        if len(clave) > 3 and clave in nombre.lower().replace("_", ""):
            return etiqueta
    return "General"


def _es_entero(valor: Any) -> bool:
    return isinstance(valor, int) and not isinstance(valor, bool)


def sugerir_rango(nombre: str, valor: float) -> Tuple[float, float, float]:
    """
    Propone (min, max, step) para el slider de un parametro.

    Heuristica:
      * Porcentajes (<= 1 y > 0)  -> rango 0 .. min(1, 4x valor), paso fino.
      * Numeros pequenos (< 100)  -> +/- 100% del valor, alineado a 2 decimales.
      * Enteros de conteo         -> 0 .. 5x, paso 1.
      * Valores grandes           -> +/- 50%, paso alineado.
    """
    nombre_l = nombre.lower()
    v = float(valor)

    if any(k in nombre_l for k in ("escalon", "max_escalon", "nivel", "hito", "semana", "dia")):
        top = max(1.0, abs(v) * 4)
        return 0.0, float(int(top)), 1.0

    if 0 < abs(v) <= 1.0:                      # parece un porcentaje / ratio
        top = min(1.0, max(abs(v) * 3.5, 0.05))
        paso = 0.0005 if top <= 0.1 else (0.005 if top <= 0.5 else 0.01)
        return 0.0, float(round(top, 4)), paso

    if abs(v) < 100:
        paso = 1.0 if _es_entero(valor) else max(abs(v) * 0.05, 0.01)
        return 0.0, float(round(abs(v) * 3 + paso, 4)), float(paso)

    paso = max(abs(v) * 0.01, 1.0)
    return float(round(abs(v) * 0.5, 2)), float(round(abs(v) * 1.5, 2)), float(round(paso, 4))


# --------------------------------------------------------------------------- #
# Analizador
# --------------------------------------------------------------------------- #
class _Analizador(ast.NodeVisitor):
    """Recorre el ASTtha del archivo y extrae la metadata de la estrategia."""

    def __init__(self) -> None:
        self.info = EstrategiaInfo()
        self._simbolos_vistos: set[str] = set()
        self._alias: Dict[str, str] = {}
        self._clases: List[ast.ClassDef] = []

    # --- utilidades de registro ------------------------------------- #
    def _add_simbolo(self, ticker: Optional[str]) -> None:
        if ticker and ticker.upper() not in self._simbolos_vistos and _RE_TICKER.match(ticker.upper()):
            self._simbolos_vistos.add(ticker.upper())

    # --- clases ------------------------------------------------------ #
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = [_nodo_nombre(b).split(".")[-1] for b in node.bases]
        # La clase principal es la que hereda de QCAlgorithm
        if any(b in ("QCAlgorithm", "AlgorithmImports.QCAlgorithm") for b in bases):
            self._clases.append(node)
            if not self.info.clase:
                self.info.clase = node.name
                doc = ast.get_docstring(node)
                if doc:
                    self.info.docstring = doc.strip()
        for hijo in node.body:
            self.visit(hijo)

    # --- imports ----------------------------------------------------- #
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.info.librerias.append(alias.name)
            if alias.name in ("sklearn", "scipy", "torch", "statsmodels", "xgboost", "lightgbm"):
                self.info.usa_ia = True

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        nombre = node.module or ""
        self.info.librerias.append(nombre)
        if any(k in nombre for k in ("sklearn", "torch", "statsmodels", "xgboost", "lightgbm")):
            self.info.usa_ia = True

    # --- asignaciones ------------------------------------------------ #
    def visit_Assign(self, node: ast.Assign) -> None:
        destino = node.targets[0]
        nombre_destino = _nodo_nombre(destino)
        valor_lit = _literal(node.value)

        # Configuracion del backtest cuando se escribe como asignacion (poco habitual,
        # pero aparece en estrategias que guardan el resultado: `f = self.set_cash(...)`)
        self._config_backtest(nombre_destino, node.value)

        # --- parametros del backtest: self.foo = self._param("foo", 500) ---
        if isinstance(node.value, ast.Call) and _nodo_nombre(node.value.func).endswith("_param"):
            nombre_param = self._string_arg(node.value, 0)
            defecto = _literal(node.value.args[1]) if len(node.value.args) > 1 else None
            if nombre_param and isinstance(defecto, (int, float)) and not isinstance(defecto, bool):
                self._add_param(nombre_param, defecto, origen="param")

        # --- listas de activos: self.activos_riesgo = ["SOXL", "TQQQ"] ---
        for t in _primer_lista_strings(node.value):
            self._add_simbolo(t)

        # --- constantes numericas ajustables ---
        if isinstance(valor_lit, (int, float)) and not isinstance(valor_lit, bool):
            etiqueta = nombre_destino.split(".")[-1]
            if self._es_param_ajustable(etiqueta, node):
                self._add_param(etiqueta, valor_lit, origen="atributo")

        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        """
        Las llamadas de configuracion de LEAN se escriben como sentencias sueltas:

            self.set_start_date(2020, 1, 1)
            self.set_cash(100000)

        Es decir, son `ast.Expr` y NO `ast.Assign`. Sin este visitor el analisis
        pasaria por alto fechas, capital, warm-up y benchmark.
        """
        if isinstance(node.value, ast.Call):
            self._config_backtest(_nodo_nombre(node.value.func), node.value)
        self.generic_visit(node)

    def _config_backtest(self, nombre_destino: str, value: ast.AST) -> None:
        """Interpreta `set_start_date / set_end_date / set_cash / set_warm_up / set_benchmark`."""
        simple = nombre_destino.split(".")[-1]
        if simple == "set_start_date":
            partes = self._argumentos(value)
            if len(partes) >= 3 and all(isinstance(p, (int, float)) for p in partes[:3]):
                self.info.fecha_inicio = f"{int(partes[0]):04d}-{int(partes[1]):02d}-{int(partes[2]):02d}"
        elif simple == "set_end_date":
            partes = self._argumentos(value)
            if len(partes) >= 3 and all(isinstance(p, (int, float)) for p in partes[:3]):
                self.info.fecha_fin = f"{int(partes[0]):04d}-{int(partes[1]):02d}-{int(partes[2]):02d}"
        elif simple == "set_cash":
            capital = self._resolver_capital(value)
            if capital is not None:
                self.info.capital = capital
        elif simple == "set_warm_up":
            partes = self._argumentos(value)
            if partes and isinstance(partes[0], (int, float)):
                self.info.warmup = int(partes[0])
            if len(partes) > 1:
                self.info.warmup_resolucion = Resolution_nombre(partes[1])
        elif simple == "set_benchmark":
            self.info.benchmark = self._resolver_ticker(value)

    @staticmethod
    def _es_param_ajustable(etiqueta: str, node: ast.Assign) -> bool:
        """
        Filtra las constantes numericas que NO son parametros de la estrategia.

        Filtra identificadores de infraestructura y estado interno, que no tiene
        sentido exponer como slider (y que ademas generarian sliders espurios).
        """
        if not etiqueta or etiqueta.startswith("_"):
            return False
        if etiqueta in _NO_AJUSTABLE:
            return False
        # Los indices de bucles y de listas (`s`, `i`, `p`, `j`) tampoco lo son.
        if len(etiqueta) <= 2 and etiqueta.isalpha():
            return False
        return True

    def _argumentos(self, node: Any) -> List[Any]:
        """Argumentos de una llamada, con los `_param("x", 500)` resueltos a su defecto."""
        if not isinstance(node, ast.Call):
            return []
        salida: List[Any] = []
        for arg in node.args:
            if isinstance(arg, ast.Call) and _nodo_nombre(arg.func).endswith("_param"):
                salida.append(_literal(arg.args[1]) if len(arg.args) > 1 else None)
                continue
            salida.append(_literal(arg))
        return salida

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # self.x: float = 0.5
        if node.value is not None:
            valor_lit = _literal(node.value)
            if isinstance(valor_lit, (int, float)) and not isinstance(valor_lit, bool):
                etiqueta = _nodo_nombre(node.target).split(".")[-1]
                if etiqueta and not etiqueta.startswith("_") and etiqueta not in _NO_AJUSTABLE:
                    self._add_param(etiqueta, valor_lit, origen="atributo")
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        # conteo_escalones += 1 : constante de estado, no un parametro
        self.generic_visit(node)

    # --- llamadas --------------------------------------------------- #
    def visit_Call(self, node: ast.Call) -> None:
        func = _nodo_nombre(node.func)
        simple = func.split(".")[-1]

        # self.sma(self.symbol, 200, Resolution.DAILY)
        if simple in ("sma", "ema", "wma", "rsi", "atr", "macd", "stddev"):
            simbolo = self._resolver_ticker(node.args[0]) if node.args else "?"
            periodo = _literal(node.args[1]) if len(node.args) > 1 else None
            res = Resolution_nombre(node.args[2]) if len(node.args) > 2 else "Daily"
            if isinstance(periodo, (int, float)):
                self.info.indicadores.append((simple, simbolo or "?", int(periodo), res))

        # self.add_equity("SPY", Resolution.MINUTE)
        if simple in ("add_equity", "add_index", "add", "add_crypto"):
            ticker = self._resolver_ticker(node.args[0]) if node.args else None
            self._add_simbolo(ticker)
            res = Resolution_nombre(node.args[1]) if len(node.args) > 1 else "Daily"
            if Resolution_nombre(res) == "Minute":
                self.info.resolucion_principal = "Minute"

        # conteo de operaciones
        if simple == "market_order":
            self.info.n_ordenes += 1
        elif simple == "liquidate":
            self.info.n_liquidaciones += 1
        elif simple == "set_holdings":
            self.info.n_set_holdings += 1
        elif simple == "on":
            self.info.eventos_calendario += 1

        # Contabilidad de operaciones y eventos de calendario.
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.info.metodos.append(node.name)
        if node.name.startswith("on_") and node.name != "on_start":
            self.info.n_eventos += 1
        self.generic_visit(node)

    # --- auxiliares ------------------------------------------------- #
    def _add_param(self, nombre: str, valor: Any, origen: str) -> None:
        if any(p.nombre == nombre for p in self.info.parametros):
            return
        self.info.parametros.append(
            Parametro(
                nombre=nombre,
                valor=float(valor),
                categoria=_categorizar(nombre),
                origen=origen,
                es_entero=_es_entero(valor),
            )
        )

    def _llamada_numerica(self, node: ast.Assign) -> List[Any]:
        """Extrae los argumentos de una llamada `self.set_start_date(2020, 1, 1)`."""
        if not isinstance(node.value, ast.Call):
            return []
        out: List[Any] = []
        for arg in node.value.args:
            val = _literal(arg)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                out.append(val)
            else:
                out.append(self._string_arg(arg, 0) if isinstance(arg, ast.Call) else None)
        return out

    @staticmethod
    def _string_arg(node: ast.AST, indice: int) -> Optional[str]:
        if not isinstance(node, ast.Call) or len(node.args) <= indice:
            return None
        val = _literal(node.args[indice])
        return str(val) if val is not None else None

    def _resolver_ticker(self, node: Optional[ast.AST]) -> Optional[str]:
        """
        Deduce el ticker de un argumento.

        Si el argumento es una variable local (`ticker`, `t`) que se recorre en un
        `for` sobre una lista de activos, se toman TODOS los strings en mayusculas
        de esa lista: asi se detectan universos como `["SOXL", "TQQQ"]` aunque el
        `add_equity` reciba una variable.
        """
        if node is None:
            return None
        val = _literal(node)
        if isinstance(val, str):
            return val.upper()
        if isinstance(val, (int, float)):
            return None
        if isinstance(node, ast.Name):
            # El nombre puede ser un alias (self.spy), un ticker o una variable
            # de bucle sobre una lista de activos.
            if node.id in self._alias:
                return self._alias[node.id]
            if node.id.upper() in self._simbolos_vistos:
                return node.id.upper()
            candidatos = self._universos()
            if candidatos:
                return candidatos[0]
        if isinstance(node, ast.Attribute):
            # Caso habitual: `self.sma(self.spy, 50, ...)`
            if node.attr in self._alias:
                return self._alias[node.attr]
            if node.attr.upper() in self._simbolos_vistos:
                return node.attr.upper()
        return None

    def _universos(self) -> List[str]:
        """Universos de activos detectados como listas de literales en mayusculas."""
        return []

    def _resolver_capital(self, node: ast.AST) -> Optional[float]:
        """`self.set_cash(self._param("capital", 500))` -> 500.0"""
        val = _literal(node)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
        if isinstance(node, ast.Call):
            partes = self._argumentos(node)
            if partes and isinstance(partes[0], (int, float)) and not isinstance(partes[0], bool):
                return float(partes[0])
        return None


# Marcador para "el simbolo de este indicador no se pudo resolver".
_SIMBOLO_GEN = "?"


def _mapa_alias(arbol: ast.AST) -> Dict[str, str]:
    """
    Mapa alias -> ticker extraido de asignaciones del tipo:

        self.spy  = self.add_equity("SPY", Resolution.DAILY).symbol
        self.symbol = self.equity.symbol

    Permite que los indicadores se reporten contra su ticker real y no como '?'.
    """
    from qc_lab.core.datafeed import UNIVERSE

    alias: Dict[str, str] = {}
    eq_a_ticker: Dict[str, str] = {}

    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Assign):
            continue
        destino = _nodo_nombre(nodo.targets[0]).split(".")[-1]
        valor = nodo.value

        # Se admite tanto `self.spy = self.add_equity("SPY", ...)`
        # como `self.spy = self.add_equity("SPY", ...).symbol`
        if isinstance(valor, ast.Attribute) and valor.attr == "symbol":
            valor = valor.value
        if not isinstance(valor, ast.Call):
            continue
        if _nodo_nombre(valor.func).split(".")[-1] == "add_equity" and valor.args:
            ticker = _literal(valor.args[0])
            if isinstance(ticker, str) and ticker.upper() in UNIVERSE:
                eq_a_ticker[destino] = ticker.upper()

    # self.<alias> = self.<equity>.symbol
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Assign) and isinstance(nodo.value, ast.Attribute) \
                and nodo.value.attr == "symbol":
            origen = _nodo_nombre(nodo.value.value).split(".")[-1]
            destino = _nodo_nombre(nodo.targets[0]).split(".")[-1]
            if origen in eq_a_ticker:
                alias[destino] = eq_a_ticker[origen]

    return alias


def Resolution_nombre(valor: Any) -> str:
    """Normaliza el argumento de resolucion de una llamada a str canonico."""
    if valor is None:
        return "Daily"
    if isinstance(valor, str):
        return {"MINUTE": "Minute", "DAILY": "Daily", "HOURLY": "Hourly",
                "WEEKLY": "Weekly"}.get(valor.upper(), valor.capitalize())
    if isinstance(valor, ast.Attribute):
        return valor.attr.capitalize()
    if isinstance(valor, ast.Constant) and isinstance(valor.value, str):
        return valor.value.capitalize()
    return "Daily"


# --------------------------------------------------------------------------- #
# API publica
# --------------------------------------------------------------------------- #
def analizar_archivo(ruta: str | Path) -> EstrategiaInfo:
    """Analiza un `.py` de estrategia y devuelve su metadata."""
    ruta = Path(ruta)
    info = EstrategiaInfo(ruta=str(ruta), nombre_archivo=ruta.name)
    try:
        fuente = ruta.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover
        info.error = f"No se pudo leer: {exc}"
        return info

    lineas = fuente.count("\n") + 1
    info.lineas = lineas
    try:
        arbol = ast.parse(fuente, filename=str(ruta))
    except SyntaxError as exc:
        info.error = f"Error de sintaxis linea {exc.lineno}: {exc.msg}"
        return info

    analizador = _Analizador()
    # El mapa de alias se calcula ANTES de recorrer el AST: durante el recorrido
    # se necesita para resolver `self.sma(self.spy, ...)`.
    analizador._alias = _mapa_alias(arbol)
    analizador.visit(arbol)

    # El analizador crea su propia EstrategiaInfo; se conservan los campos de
    # procedencia calculados antes de entrar al AST.
    info = analizador.info
    info.ruta = str(ruta)
    info.nombre_archivo = ruta.name
    info.lineas = lineas

    # Segunda pasada: barrido de literales tipo ticker en todo el archivo.
    # Los ETF apalancados y las reservas aparecen casi siempre dentro de listas,
    # asi que un barrido global es la forma mas robusta de detectarlos.
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str):
            t = nodo.value.upper()
            if _RE_TICKER.match(t) and t in datafeed_universe():
                analizador._add_simbolo(t)

    # Resolucion de alias: `self.spy = self.add_equity("SPY", ...).symbol`
    alias = _mapa_alias(arbol)
    for i, (nombre, simbolo, periodo, res) in enumerate(info.indicadores):
        if simbolo == "?":
            info.indicadores[i] = (nombre, alias.get(_SIMBOLO_GEN, simbolo), periodo, res)

    # Si el indicador sigue sin simbolo y la estrategia opera un solo activo,
    # ese activo es necesariamente el del indicador.
    riesgo = [s for s in analizador._simbolos_vistos
              if s not in {"SPY", "SPX", "QQQ", "IWM", "VIX", "DIA", "MDY"}]
    for i, (nombre, simbolo, periodo, res) in enumerate(info.indicadores):
        if simbolo == "?" and len(riesgo) == 1:
            info.indicadores[i] = (nombre, riesgo[0], periodo, res)

    info.simbolos = sorted(analizador._simbolos_vistos)
    info.librerias = [l for l in dict.fromkeys(info.librerias) if l]
    info.metodos = list(dict.fromkeys(info.metodos))
    info.valido = bool(info.clase)

    if not info.clase:
        info.error = "No se encontro ninguna clase que herede de QCAlgorithm"
    if not info.simbolos:
        info.simbolos = ["SPY"]
    if info.capital is None:
        info.capital = 100000.0
    if not info.fecha_inicio:
        info.fecha_inicio = "2020-01-01"
    if not info.fecha_fin:
        info.fecha_fin = "2026-09-30"
    if info.resolucion_principal != "Minute":
        info.resolucion_principal = "Daily"

    return info


def datafeed_universe() -> set[str]:
    """Tickers reconocidos por el data feed (para el barrido de literales)."""
    from qc_lab.core.datafeed import UNIVERSE
    return set(UNIVERSE.keys())