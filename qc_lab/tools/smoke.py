"""
Prueba de humo del motor: ejecuta todas las estrategias del proyecto.

    python -m qc_lab.tools.smoke
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from qc_lab.core.engine import ejecutar_backtest          # noqa: E402
from qc_lab.core.analytics import (                      # noqa: E402
    fmt_compacto, fmt_dinero, fmt_pct,
)
from qc_lab.core.strategy_parser import analizar_archivo  # noqa: E402

ARCHIVOS = [
    "otras/SP500.py",
    "otras/tesla.py",
    "otras/SOXL.py",
    "otras/Dual Momentum + Volatility Targeting.py",
    "otras/Estrategia IA Diaria - RandomForest + ATR Trailing Stop & Reserve.py",
    "claude/paso1.py",
    "claude/paso 1b.py",
]


def main() -> int:
    fallos = 0
    for rel in ARCHIVOS:
        ruta = RAIZ / rel
        print("=" * 100)
        print(f"{rel}")
        print("-" * 100)
        info = analizar_archivo(ruta)
        if not info.valido:
            print(f"  !! {info.error}")
            fallos += 1
            continue

        try:
            res = ejecutar_backtest(ruta, info=info)
        except Exception:
            print("  !! EXCEPCION NO CONTROLADA")
            traceback.print_exc()
            fallos += 1
            continue

        if not res.exito:
            print(f"  !! FALLO: {res.error}")
            if res.traza:
                print("     " + res.traza[-1200:].replace("\n", "\n     "))
            fallos += 1
            continue

        e = res.estadisticas
        print(f"  clase      {info.clase}   [{info.resolucion_principal}]")
        print(f"  periodo    {info.fecha_inicio} -> {info.fecha_fin}   "
              f"warmup {info.warmup} {info.warmup_resolucion}")
        print(f"  universo   {info.simbolos}  benchmark {info.benchmark_efectivo}")
        print(f"  barras     {res.barras_evaluadas:,}   tiempo {res.duracion_s:.2f}s   "
              f"logs {len(res.logs)}")
        print(f"  equity     {fmt_dinero(e['equity_inicial'])} -> {fmt_dinero(e['equity_final'])}"
              f"   retorno {fmt_pct(e['retorno_total'])}   neto {fmt_dinero(e['beneficio_neto'])}")
        print(f"  riesgo     Sharpe {e['sharpe']:.2f}  Sortino {e['sortino']:.2f}  "
              f"PSR {e['psr']:.3f}  CAGR {fmt_pct(e['cagr'])}  vol {fmt_pct(e['volatilidad'])}")
        print(f"  drawdown   max {fmt_pct(e['dd_max'])}  medio {fmt_pct(e['dd_medio'])}  "
              f"Calmar {e['calmar']:.2f}")
        print(f"  costes     comisiones {fmt_dinero(e['comisiones'])}  "
              f"volumen {fmt_compacto(e['volumen'])}  turnover {e['turnover']:.2f}x")
        print(f"  actividad ejecuciones {e['n_ejecuciones']}  ops {e['n_operaciones']}  "
              f"win {fmt_pct(e['win_rate'],1)}  PF {e['profit_factor']:.2f}")
        print(f"  exposicion media {fmt_pct(e['exposicion_media'],1)}  "
              f"margen medio {fmt_dinero(e['margen_medio'],0)}")
        print(f"  filas curva {len(res.curva)}  ejecuciones {len(res.ejecuciones)}  "
              f"operaciones {len(res.operaciones)}")

        errores = [l for l in res.logs if l["nivel"] == "ERROR"]
        if errores:
            print(f"  ERRORES DE LOG ({len(errores)}):")
            for l in errores[:3]:
                print(f"     {l['mensaje'][:200]}")
    print("=" * 100)
    print(f"fallos: {fallos}/{len(ARCHIVOS)}")
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())