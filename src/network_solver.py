# network_solver.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np

import matrix_operations as mo  # reutilizamos tu módulo


@dataclass
class SolveResult:
    method: str
    success: bool
    x: Optional[np.ndarray]
    residual_rel: Optional[float]
    cond_number: Optional[float]
    notes: List[str]
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "success": self.success,
            "x": None if self.x is None else np.asarray(self.x),
            "residual_rel": self.residual_rel,
            "cond_number": self.cond_number,
            "notes": self.notes,
            "error": self.error,
        }


def _to_array(v) -> np.ndarray:
    return np.array(v, dtype=float)


# ---------------------------------------------------------------------
# MÉTODO 1: Resolver con inversa matricial
# ---------------------------------------------------------------------
def solve_via_inverse(A, b, tol_cond_warn: float = 1e12) -> SolveResult:
    """Resuelve Ax = b usando x = A^{-1} b. Falla si la matriz no es invertible."""
    notes: List[str] = []
    try:
        A = _to_array(A)
        b = _to_array(b)
        mo._check_square(A)

        kappa = mo.condition_number(A, p=2)
        if kappa >= tol_cond_warn:
            notes.append(f"⚠ Advertencia: κ₂(A) = {kappa:.3e}, posible inestabilidad numérica.")

        # Verificamos si es invertible
        if not mo.is_invertible(A):
            msg = "❌ La matriz no es invertible (det(A)≈0)."
            notes.append(msg)
            return SolveResult(
                method="inverse",
                success=False,
                x=None,
                residual_rel=None,
                cond_number=kappa,
                notes=notes,
                error=msg,
            )

        # Calcular inversa y resolver
        A_inv = mo.inverse(A)
        x = A_inv @ b
        val = mo.validate_solution(A, x, b)

        notes.append("✅ Solución calculada mediante inversa A^{-1}·b.")
        return SolveResult(
            method="inverse",
            success=True,
            x=x,
            residual_rel=val.residual_norm_rel,
            cond_number=kappa,
            notes=notes,
        )

    except Exception as e:
        return SolveResult(
            method="inverse",
            success=False,
            x=None,
            residual_rel=None,
            cond_number=None,
            notes=notes,
            error=str(e),
        )


# ---------------------------------------------------------------------
# MÉTODO 2: Resolver con numpy.linalg.solve
# ---------------------------------------------------------------------
def solve_via_linalg_solve(A, b, tol_cond_warn: float = 1e12) -> SolveResult:
    """Resuelve Ax = b usando numpy.linalg.solve (a través de mo.solve)."""
    notes: List[str] = []
    try:
        A = _to_array(A)
        b = _to_array(b)
        mo._check_square(A)

        kappa = mo.condition_number(A, p=2)
        if kappa >= tol_cond_warn:
            notes.append(f"⚠ Advertencia: κ₂(A) = {kappa:.3e}, posible inestabilidad numérica.")

        if not mo.is_invertible(A):
            msg = "❌ La matriz no es invertible, numpy.linalg.solve no puede resolverla."
            notes.append(msg)
            return SolveResult(
                method="linalg_solve",
                success=False,
                x=None,
                residual_rel=None,
                cond_number=kappa,
                notes=notes,
                error=msg,
            )

        x = mo.solve(A, b)
        val = mo.validate_solution(A, x, b)
        notes.append("✅ Solución calculada mediante numpy.linalg.solve.")
        return SolveResult(
            method="linalg_solve",
            success=True,
            x=x,
            residual_rel=val.residual_norm_rel,
            cond_number=kappa,
            notes=notes,
        )

    except np.linalg.LinAlgError as e:
        notes.append(f"Error de álgebra lineal: {e}")
        return SolveResult(
            method="linalg_solve",
            success=False,
            x=None,
            residual_rel=None,
            cond_number=None,
            notes=notes,
            error=str(e),
        )
    except Exception as e:
        return SolveResult(
            method="linalg_solve",
            success=False,
            x=None,
            residual_rel=None,
            cond_number=None,
            notes=notes,
            error=str(e),
        )


# ---------------------------------------------------------------------
# COMPARACIÓN ENTRE AMBOS MÉTODOS
# ---------------------------------------------------------------------
def compare_solutions(A, b, tol_rel_diff: float = 1e-8) -> Dict[str, Any]:
    """Compara los resultados entre ambos métodos si A es invertible."""
    A = _to_array(A)
    b = _to_array(b)

    res_inv = solve_via_inverse(A, b)
    res_solve = solve_via_linalg_solve(A, b)

    compare_notes: List[str] = []
    diff_norm_rel: Optional[float] = None

    if res_inv.success and res_solve.success:
        x_inv, x_solve = res_inv.x, res_solve.x
        denom = np.linalg.norm(x_solve) or 1.0
        diff_norm_rel = float(np.linalg.norm(x_inv - x_solve, ord=2) / denom)

        if diff_norm_rel <= tol_rel_diff:
            compare_notes.append(f"✅ Ambas soluciones coinciden dentro de la tolerancia ({diff_norm_rel:.3e}).")
        else:
            compare_notes.append(f"⚠ Diferencia entre soluciones = {diff_norm_rel:.3e} (> {tol_rel_diff:g}).")
    else:
        compare_notes.append("❌ No se pudo comparar porque al menos un método falló.")

    return {
        "A_shape": A.shape,
        "inverse": res_inv,
        "linalg_solve": res_solve,
        "diff_norm_rel": diff_norm_rel,
        "compare_notes": compare_notes,
    }


# ---------------------------------------------------------------------
# EJEMPLO DE USO
# ---------------------------------------------------------------------
if __name__ == "__main__":
    A = np.array([
        [-3.0, 1.0, 1.0],
        [1.0, -3.0, 1.0],
        [1.0, 1.0, -3.0],
    ], dtype=float)

    b = np.array([1.0, 2.0, 3.0], dtype=float)

    result = compare_solutions(A, b)

    print("=== COMPARACIÓN DE MÉTODOS ===")
    print(f"Dimensión de A: {result['A_shape']}")
    print("\n--- Método por Inversa ---")
    print(result["inverse"].as_dict())
    print("\n--- Método por numpy.linalg.solve ---")
    print(result["linalg_solve"].as_dict())
    print("\n--- Comparación ---")
    for n in result["compare_notes"]:
        print(" -", n)
