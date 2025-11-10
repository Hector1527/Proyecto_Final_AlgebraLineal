"""
Analizador de resultados para escenarios de red (Ax = b).

Este módulo calcula métricas básicas, detecta cuellos de botella y
genera recomendaciones legibles a partir de la estructura de resultados
producida por `scenario_simulator.run_scenario`.

Funciones públicas:
- compute_basic_metrics(result)
- detect_bottlenecks(result, threshold_ratio=0.8)
- generate_recommendation(result, threshold_ratio=0.8)
- analyze_all(results, threshold_ratio=0.8)

No imprime; devuelve estructuras de datos o cadenas.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import numpy as np


def _safe_norm(x: np.ndarray) -> float:
    try:
        return float(np.linalg.norm(x, ord=2))
    except Exception:
        return float("nan")


def _is_solved(result: Dict[str, Any]) -> bool:
    return bool(result.get("success")) and result.get("x") is not None


def compute_basic_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula métricas básicas del resultado de un escenario.

    Retorna un dict con:
        - residual_norm
        - relative_residual
        - x_norm
        - x_max_abs, x_min, x_mean
        - condition_number, rank, determinant
        - solvable (bool), method (str)
    Soporta resultados no resueltos (campos NaN/None adecuados).
    """
    A = result.get("A")
    b = result.get("b")
    x = result.get("x")
    method = result.get("method")

    metrics: Dict[str, Any] = {
        "residual_norm": None,
        "relative_residual": None,
        "x_norm": None,
        "x_max_abs": None,
        "x_min": None,
        "x_mean": None,
        "condition_number": None,
        "rank": None,
        "determinant": None,
        "solvable": bool(result.get("success")),
        "method": method,
    }

    # rank y condicionamiento de A
    try:
        if A is not None:
            metrics["rank"] = int(np.linalg.matrix_rank(A))
            try:
                metrics["condition_number"] = float(np.linalg.cond(A))
            except Exception:
                metrics["condition_number"] = None
            try:
                metrics["determinant"] = float(np.linalg.det(A))
            except Exception:
                metrics["determinant"] = None
    except Exception:
        pass

    # Métricas dependientes de x
    if x is not None and A is not None and b is not None and _is_solved(result):
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        b_arr = np.asarray(b, dtype=float).reshape(-1)
        A_arr = np.asarray(A, dtype=float)

        r = A_arr @ x_arr - b_arr
        r_norm = _safe_norm(r)
        b_norm = _safe_norm(b_arr)
        metrics["residual_norm"] = r_norm
        metrics["relative_residual"] = (r_norm / (b_norm + 1e-12)) if np.isfinite(b_norm) else float("nan")

        x_norm = _safe_norm(x_arr)
        metrics["x_norm"] = x_norm
        metrics["x_max_abs"] = float(np.max(np.abs(x_arr))) if x_arr.size else None
        metrics["x_min"] = float(np.min(x_arr)) if x_arr.size else None
        metrics["x_mean"] = float(np.mean(x_arr)) if x_arr.size else None
    else:
        # Mantener None para indicar no disponible
        pass

    return metrics


def detect_bottlenecks(result: Dict[str, Any], threshold_ratio: float = 0.8) -> Dict[str, Any]:
    """Detecta cuellos de botella a partir del vector de flujos x.

    Criterio principal: nodos cuyo |x_i| sea >= threshold_ratio * max(|x|).
    También informa banderas del sistema (condición alta, error de solución).

    Retorna un dict con:
      - bottleneck_nodes: lista de índices
      - details: lista con dicts {index, flow, relative_load}
      - system_flags: lista de strings (p. ej., 'ill_conditioned', 'unsolved')
    """
    flags: List[str] = []
    x = result.get("x")
    cond = result.get("condition_number")
    success = bool(result.get("success"))

    if not success or x is None:
        flags.append("unsolved")
        return {"bottleneck_nodes": [], "details": [], "system_flags": flags}

    x_arr = np.asarray(x, dtype=float).reshape(-1)

    # Sistema potencialmente mal condicionado
    if cond is not None:
        try:
            if float(cond) > 1e8:
                flags.append("ill_conditioned")
        except Exception:
            pass

    if x_arr.size == 0:
        return {"bottleneck_nodes": [], "details": [], "system_flags": flags}

    max_abs = float(np.max(np.abs(x_arr)))
    if max_abs <= 0.0 or not np.isfinite(max_abs):
        return {"bottleneck_nodes": [], "details": [], "system_flags": flags}

    thr = threshold_ratio * max_abs
    indices = np.where(np.abs(x_arr) >= thr)[0].tolist()
    details = [
        {
            "index": int(i),
            "flow": float(x_arr[i]),
            "relative_load": float(abs(x_arr[i]) / max_abs),
        }
        for i in indices
    ]

    return {"bottleneck_nodes": indices, "details": details, "system_flags": flags}


def generate_recommendation(result: Dict[str, Any], threshold_ratio: float = 0.8) -> str:
    """Genera recomendaciones legibles en base a métricas y cuellos de botella."""
    metrics = compute_basic_metrics(result)
    bottlenecks = detect_bottlenecks(result, threshold_ratio=threshold_ratio)

    parts: List[str] = []

    name = result.get("name", "escenario")
    parts.append(f"Escenario '{name}':")

    if not metrics.get("solvable"):
        parts.append("- No se pudo resolver el sistema. Verifique datos y método de resolución.")
        cond = metrics.get("condition_number")
        if cond is not None and np.isfinite(cond) and cond > 1e8:
            parts.append("- La matriz está muy mal condicionada; revise la conectividad o agregue redundancia.")
        return "\n".join(parts)

    # Residual
    resn = metrics.get("residual_norm")
    relres = metrics.get("relative_residual")
    if resn is not None and relres is not None:
        parts.append(f"- Residual ||Ax-b||: {resn:.4g} (relativo: {relres:.4g}).")

    # Condición
    cond = metrics.get("condition_number")
    if cond is not None and np.isfinite(cond):
        if cond > 1e8:
            parts.append("- Condición alta: resultados sensibles a perturbaciones; considere reforzar enlaces.")
        else:
            parts.append("- Condición de la matriz aceptable para el análisis.")

    # Cuellos de botella
    nodes = bottlenecks.get("bottleneck_nodes", [])
    if nodes:
        parts.append(f"- Cuellos de botella detectados en nodos {nodes}.")
        parts.append("  Sugerencias: redistribuir demanda hacia nodos con menor carga, balancear rutas, o incrementar capacidad en esos nodos/enlaces.")
    else:
        parts.append("- No se detectan cuellos de botella con el umbral actual; la carga parece balanceada.")

    return "\n".join(parts)


def analyze_all(
    results: Union[Dict[str, Dict[str, Any]], List[Dict[str, Any]]],
    threshold_ratio: float = 0.8,
) -> Dict[str, Dict[str, Any]]:
    """Analiza en lote un conjunto de resultados de escenarios.

    Acepta un dict (nombre->resultado) o una lista de resultados (usará la
    clave 'name' si está disponible para indexar). Retorna un dict con las
    llaves de cada escenario y valores con:
      - metrics
      - bottlenecks
      - recommendation (string)
    """
    out: Dict[str, Dict[str, Any]] = {}

    if isinstance(results, dict):
        items = list(results.items())  # (name, result)
    else:
        items = [(r.get("name", f"scenario_{i}"), r) for i, r in enumerate(results)]

    for name, res in items:
        metrics = compute_basic_metrics(res)
        bottlenecks = detect_bottlenecks(res, threshold_ratio=threshold_ratio)
        recommendation = generate_recommendation(res, threshold_ratio=threshold_ratio)
        out[name] = {
            "metrics": metrics,
            "bottlenecks": bottlenecks,
            "recommendation": recommendation,
        }

    return out
