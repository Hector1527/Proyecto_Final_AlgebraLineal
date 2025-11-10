"""
Simulador de escenarios de demanda para redes (Ax = b).

Este módulo genera varios escenarios de demanda y permite ejecutarlos
contra una matriz de conexiones A utilizando un solver ya existente
(`solve_system`) definido en `network_solver.py`.

Funciones públicas:
- create_scenarios(): devuelve escenarios por defecto (3 nodos).
- run_scenario(A, scenario, method="numpy"): ejecuta un escenario.
- run_all_scenarios(A, method="numpy"): ejecuta todos los escenarios
  adaptados al tamaño de A y devuelve resultados estructurados.

Nota: No imprime nada; todas las funciones retornan estructuras de datos
listas para ser usadas por otros módulos (p. ej., `results_analyzer`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any

import numpy as np

# Importación del solver asumido por el enunciado.
try:
    from .network_solver import solve_system  # type: ignore
except Exception:  # pragma: no cover - en ejecución aislada puede no existir
    # Fallback opcional para facilitar validación manual si el solver no está.
    def solve_system(A: np.ndarray, b: np.ndarray, method: str = "numpy") -> np.ndarray:
        """Resolver Ax=b usando NumPy como respaldo si no se encuentra el solver real."""
        return np.linalg.solve(A, b)


@dataclass
class Scenario:
    """Representa un escenario de demanda.

    Atributos:
        name: nombre corto del escenario.
        description: descripción legible.
        b: vector de demanda (shape (n,)).
        meta: metadatos libres (opcional).
    """

    name: str
    description: str
    b: np.ndarray
    meta: Optional[Dict[str, Any]] = None


def _ensure_vector(x: Union[np.ndarray, List[float]]) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    return x_arr.reshape(-1)


def _generate_scenarios_for_size(n: int, base: float = 100.0) -> List[Scenario]:
    """Genera al menos seis escenarios para una red de `n` nodos.

    Los patrones son deterministas y escalan con `base`.
    """
    if n <= 0:
        raise ValueError("El tamaño de la red debe ser positivo")

    ones = np.ones(n)

    # 1) Balanceado: todos los nodos con demanda similar
    b_balance = base * ones

    # 2) Nodo saturado: un nodo con demanda muy superior
    b_sat = base * ones
    b_sat[0] = 3.0 * base

    # 3) Picos mixtos: dos nodos con picos distintos, uno moderado, otro alto
    b_mixed = base * ones
    b_mixed[0] = 0.5 * base
    b_mixed[min(1, n - 1)] = 3.0 * base
    b_mixed[n - 1] = 1.5 * base

    # 4) Baja demanda: cargas mucho menores a lo nominal
    b_low = 0.2 * base * ones

    # 5) Alta demanda: todos muy altos
    b_high = 5.0 * base * ones

    # 6) Extremo: un nodo extremadamente alto y el resto bajos
    b_extreme = 0.25 * base * ones
    b_extreme[n - 1] = 10.0 * base

    scenarios = [
        Scenario(
            name="balanceado",
            description="Demanda uniforme en todos los nodos",
            b=b_balance,
            meta={"type": "baseline"},
        ),
        Scenario(
            name="nodo_saturado",
            description="Un único nodo demanda significativamente más",
            b=b_sat,
            meta={"type": "single_peak", "peak_index": 0},
        ),
        Scenario(
            name="picos_mixtos",
            description="Dos picos de distinta magnitud más un nodo reducido",
            b=b_mixed,
            meta={"type": "mixed_peaks"},
        ),
        Scenario(
            name="baja_demanda",
            description="Escenario conservador de baja demanda",
            b=b_low,
            meta={"type": "low"},
        ),
        Scenario(
            name="alta_demanda",
            description="Escenario agresivo de alta demanda",
            b=b_high,
            meta={"type": "high"},
        ),
        Scenario(
            name="extremo",
            description="Demanda extremadamente desbalanceada en un nodo",
            b=b_extreme,
            meta={"type": "extreme", "peak_index": n - 1},
        ),
    ]

    return scenarios


def create_scenarios() -> List[Scenario]:
    """Crea escenarios por defecto para 3 nodos (útil como plantilla).

    Para ejecutar escenarios del tamaño correcto de `A`, use `run_all_scenarios(A)`
    que ajusta los escenarios automáticamente al número de nodos.
    """
    return _generate_scenarios_for_size(3)


def run_scenario(
    A: np.ndarray, scenario: Union[Scenario, Dict[str, Any]], method: str = "numpy"
) -> Dict[str, Any]:
    """Ejecuta un escenario de demanda sobre la red `A`.

    Parámetros:
        A: Matriz de conexiones (nxn), se asume cuadrada.
        scenario: Instancia `Scenario` o dict con al menos la clave 'b'.
        method: Método del solver a pasar a `solve_system`.

    Retorna:
        Diccionario estructurado con entradas como: name, description, A, b, x,
        method, success, error, residual, residual_norm, condition_number.
    """
    if isinstance(scenario, dict) and not isinstance(scenario, Scenario):
        name = scenario.get("name", "escenario")
        description = scenario.get("description", "")
        b_vec = _ensure_vector(scenario["b"])  # puede lanzar KeyError
        meta = scenario.get("meta")
        scenario_obj = Scenario(name=name, description=description, b=b_vec, meta=meta)
    elif isinstance(scenario, Scenario):
        scenario_obj = scenario
        b_vec = _ensure_vector(scenario_obj.b)
    else:
        raise TypeError("scenario debe ser Scenario o dict con clave 'b'")

    A = np.asarray(A, dtype=float)
    result: Dict[str, Any] = {
        "name": scenario_obj.name,
        "description": scenario_obj.description,
        "A": A,
        "b": b_vec,
        "method": method,
        "meta": scenario_obj.meta or {},
        "success": False,
        "x": None,
        "error": None,
        "residual": None,
        "residual_norm": None,
        "condition_number": None,
    }

    # Validación rápida de dimensiones
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        result["error"] = "A debe ser matriz cuadrada"
        return result
    if b_vec.shape[0] != A.shape[0]:
        result["error"] = f"Dimensión de b ({b_vec.shape[0]}) no coincide con A ({A.shape[0]})"
        return result

    try:
        x = solve_system(A, b_vec, method=method)
        x = _ensure_vector(x)
        result["x"] = x
        result["success"] = True

        # Métricas rápidas útiles para análisis posterior
        residual = A @ x - b_vec
        result["residual"] = residual
        result["residual_norm"] = float(np.linalg.norm(residual, ord=2))
    except Exception as exc:  # captura errores del solver o de linalg
        result["error"] = str(exc)

    # Condición de la matriz (siempre intentamos calcularla)
    try:
        result["condition_number"] = float(np.linalg.cond(A))
    except Exception:
        result["condition_number"] = None

    return result


def run_all_scenarios(A: np.ndarray, method: str = "numpy") -> Dict[str, Dict[str, Any]]:
    """Ejecuta todos los escenarios adaptados al tamaño de `A`.

    Retorna un diccionario indexado por nombre de escenario con el resultado
    de `run_scenario` correspondiente a cada uno.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    scenarios = _generate_scenarios_for_size(n)

    results: Dict[str, Dict[str, Any]] = {}
    for sc in scenarios:
        res = run_scenario(A, sc, method=method)
        results[sc.name] = res
    return results
