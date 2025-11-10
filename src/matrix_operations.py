

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np



Array = np.ndarray

class MatrixShapeError(ValueError):
    pass

def _as_array(M) -> Array:
    return np.array(M, dtype=float)

def _check_square(A: Array) -> None:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise MatrixShapeError(f"La matriz A debe ser cuadrada. Forma recibida: {A.shape}")

def _check_compatibility(A: Array, b: Optional[Array]) -> None:
    _check_square(A)
    if b is not None:
        b = _as_array(b)
        if b.ndim != 1 or b.shape[0] != A.shape[0]:
            raise MatrixShapeError(f"b debe ser vector de tamaño {A.shape[0]}. Forma recibida: {b.shape}")



def determinant(A: Array) -> float:
    
    A = _as_array(A)
    _check_square(A)
    return float(np.linalg.det(A))

def is_invertible(A: Array, tol: float = 1e-12) -> bool:
   
    detA = determinant(A)
    return abs(detA) > tol

def inverse(A: Array) -> Array:
    
    A = _as_array(A)
    _check_square(A)
    return np.linalg.inv(A)

def condition_number(A: Array, p: float | int = 2) -> float:
    
    A = _as_array(A)
    _check_square(A)
    return float(np.linalg.cond(A, p=p))

def solve(A: Array, b: Array) -> Array:
    """Resuelve A x = b"""
    A = _as_array(A)
    b = _as_array(b)
    _check_compatibility(A, b)
    return np.linalg.solve(A, b)



@dataclass
class ValidationReport:
    aa_inv_close_to_I: bool
    inv_a_a_close_to_I: bool
    max_abs_I_error_left: float
    max_abs_I_error_right: float
    residual_norm_rel: Optional[float]
    notes: List[str]

def validate_inverse(A: Array, A_inv: Array, tol: float = 1e-9) -> ValidationReport:
    """
    Valida que A·A^{-1} ≈ I y A^{-1}·A ≈ I.
    """
    A = _as_array(A); _check_square(A)
    A_inv = _as_array(A_inv); _check_square(A_inv)

    I = np.eye(A.shape[0])
    left  = A @ A_inv
    right = A_inv @ A

    err_left  = np.max(np.abs(left - I))
    err_right = np.max(np.abs(right - I))

    ok_left  = err_left  <= tol
    ok_right = err_right <= tol

    notes = []
    if not ok_left:
        notes.append(f"A·A^{-1} difiere de I por máx {err_left:.3e} (> {tol:g})")
    if not ok_right:
        notes.append(f"A^{-1}·A difiere de I por máx {err_right:.3e} (> {tol:g})")
    if ok_left and ok_right:
        notes.append("La inversa pasa la prueba frente a la identidad dentro del umbral.")

    return ValidationReport(
        aa_inv_close_to_I=ok_left,
        inv_a_a_close_to_I=ok_right,
        max_abs_I_error_left=float(err_left),
        max_abs_I_error_right=float(err_right),
        residual_norm_rel=None,
        notes=notes,
    )

def validate_solution(A: Array, x: Array, b: Array, tol: float = 1e-9) -> ValidationReport:
    
    A = _as_array(A); _check_square(A)
    x = _as_array(x)
    b = _as_array(b)

    # Residual relativo con escalamiento estándar
    num = np.linalg.norm(A @ x - b, ord=2)
    den = np.linalg.norm(A, ord=2) * np.linalg.norm(x, ord=2) + np.linalg.norm(b, ord=2)
    rel = float(num / den) if den != 0 else float(num)

    notes = [f"Residual relativo ||Ax-b|| / (||A||||x|| + ||b||) = {rel:.3e}"]
    ok = rel <= tol
    if ok:
        notes.append(f"La solución pasa el umbral de {tol:g}.")
    else:
        notes.append(f"La solución EXCEDE el umbral de {tol:g}.")

    return ValidationReport(
        aa_inv_close_to_I=False,
        inv_a_a_close_to_I=False,
        max_abs_I_error_left=0.0,
        max_abs_I_error_right=0.0,
        residual_norm_rel=rel,
        notes=notes,
    )



def explain_matrix(A: Array, node_labels: Optional[List[str]] = None) -> Dict[Tuple[int,int], str]:
   
    A = _as_array(A)
    _check_square(A)

    n = A.shape[0]
    if node_labels is None:
        node_labels = [f"Nodo {k+1}" for k in range(n)]
    elif len(node_labels) != n:
        raise ValueError("La longitud de node_labels debe coincidir con el tamaño de A.")

    explanations: Dict[Tuple[int,int], str] = {}

    for i in range(n):
        for j in range(n):
            aij = A[i, j]
            if i == j:
                sign = "negativo" if aij < 0 else ("positivo" if aij > 0 else "nulo")
                explanations[(i, j)] = (
                    f"a[{i+1},{j+1}] = {aij:g}: Autoconexión de {node_labels[i]}; "
                    f"representa su flujo propio (típicamente saliente si es negativo). Signo: {sign}."
                )
            else:
                if aij > 0:
                    explanations[(i, j)] = (
                        f"a[{i+1},{j+1}] = {aij:g}: Flujo ENTRANTE a {node_labels[i]} desde {node_labels[j]}."
                    )
                elif aij < 0:
                    explanations[(i, j)] = (
                        f"a[{i+1},{j+1}] = {aij:g}: Flujo SALIENTE de {node_labels[i]} hacia {node_labels[j]}."
                    )
                else:
                    explanations[(i, j)] = (
                        f"a[{i+1},{j+1}] = 0: Sin conexión directa entre {node_labels[i]} y {node_labels[j]}."
                    )
    return explanations



def report(A: Array, b: Optional[Array] = None, tol_inv: float = 1e-9, tol_sol: float = 1e-9) -> str:
    
    A = _as_array(A)
    _check_square(A)

    lines: List[str] = []
    lines.append("=== REPORTE DE OPERACIONES MATRICALES ===")
    lines.append(f"Dimensión de A: {A.shape[0]}x{A.shape[1]}")
    lines.append("")

    # Determinante & condición
    detA = determinant(A)
    lines.append(f"Determinante det(A) = {detA:.6g}")
    inv_flag = is_invertible(A)
    lines.append(f"Invertible: {'Sí' if inv_flag else 'No'}")
    try:
        kappa = condition_number(A, p=2)
        lines.append(f"Número de condición κ₂(A) = {kappa:.6g}")
    except np.linalg.LinAlgError as e:
        lines.append(f"Número de condición: no disponible ({e})")

    # Inversa y validación
    if inv_flag:
        try:
            A_inv = inverse(A)
            val_inv = validate_inverse(A, A_inv, tol=tol_inv)
            lines.append("Validación de inversa (A·A^{-1} e I):")
            lines.append(f"  A·A^{-1} ≈ I?  {val_inv.aa_inv_close_to_I}  (max error = {val_inv.max_abs_I_error_left:.3e})")
            lines.append(f"  A^{-1}·A ≈ I?  {val_inv.inv_a_a_close_to_I}  (max error = {val_inv.max_abs_I_error_right:.3e})")
        except np.linalg.LinAlgError as e:
            lines.append(f"No se pudo invertir A: {e}")
    else:
        lines.append("No se intenta invertir A porque no es invertible.")

    # Resolver Ax = b si b fue provisto
    if b is not None:
        try:
            x = solve(A, b)
            val_sol = validate_solution(A, x, b, tol=tol_sol)
            lines.append("Solución de A x = b:")
            lines.append(f"  x = {np.array2string(x, precision=6, separator=', ')}")
            lines.append(f"  Residual relativo ≈ {val_sol.residual_norm_rel:.3e}")
        except np.linalg.LinAlgError as e:
            lines.append(f"No se pudo resolver A x = b: {e}")
        except MatrixShapeError as e:
            lines.append(f"Error de dimensiones al resolver: {e}")

    return "\n".join(lines)

@dataclass
class RuleCheck:
    diagonal_negatives: int
    diagonal_total: int
    diagonal_ok: bool
    non_negative_diagonals: List[Tuple[int, float]]
    notes: List[str]

def validate_matrix_rules(A: Array) -> RuleCheck:
  
    A = _as_array(A); _check_square(A)
    n = A.shape[0]
    non_negative_diags = []
    for i in range(n):
        if A[i, i] >= 0:
            non_negative_diags.append((i, float(A[i, i])))

    diag_ok = (len(non_negative_diags) == 0)  # estricto; el enunciado dice "en la mayoría", ajusta si quieres

    notes = []
    if diag_ok:
        notes.append("Diagonal: todos los a[ii] son negativos (autoconexión típicamente saliente).")
    else:
        notes.append(
            f"Diagonal: hay {len(non_negative_diags)} entradas no negativas; el enunciado espera "
            "autoconexión NEGATIVA en la mayoría de los casos."
        )

    # Resumen de signos fuera de diagonal (para inspección rápida)
    pos_edges = []
    neg_edges = []
    zero_edges = []
    for i in range(n):
        for j in range(n):
            if i == j: 
                continue
            aij = A[i, j]
            if aij > 0:
                pos_edges.append((i, j, float(aij)))  # entrada a i desde j
            elif aij < 0:
                neg_edges.append((i, j, float(aij)))  # salida de i hacia j
            else:
                zero_edges.append((i, j))

    notes.append(f"Fuera de diagonal: {len(pos_edges)} positivos (entrada), {len(neg_edges)} negativos (salida), {len(zero_edges)} ceros (sin conexión).")

    return RuleCheck(
        diagonal_negatives=(n - len(non_negative_diags)),
        diagonal_total=n,
        diagonal_ok=diag_ok,
        non_negative_diagonals=non_negative_diags,
        notes=notes
    )

def make_diagonal_laplacian(A: Array) -> Array:
   
    A = _as_array(A); _check_square(A)
    B = A.copy()
    n = B.shape[0]
    for i in range(n):
        row_sum_abs = np.sum(np.abs(B[i, :])) - abs(B[i, i])
        B[i, i] = -row_sum_abs
    return B



def print_full_analysis(
    A: Array,
    b: Optional[Array] = None,
    node_labels: Optional[List[str]] = None,
    tol_inv: float = 1e-9,
    tol_sol: float = 1e-9,
    p_norm: int | float = 2,
) -> None:
   
    import numpy as np
    np.set_printoptions(precision=6, suppress=True)

    A = _as_array(A)
    _check_square(A)

    print("=== VALIDACIÓN CONTRA REGLAS DEL ENUNCIADO ===")
    rc = validate_matrix_rules(A)
    print(f"Diagonal negativa (estricto): {rc.diagonal_ok}  "
          f"({rc.diagonal_negatives}/{rc.diagonal_total} a[ii] < 0)")
    if rc.non_negative_diagonals:
        print("Entradas diagonales no negativas:", rc.non_negative_diagonals)
    for note in rc.notes:
        print(" -", note)
    print()

    print("=== VERIFICACIÓN DE INVERTIBILIDAD ===")
        

    detA = determinant(A)
    print(f"det(A) = {detA:.6g}")
    inv_flag = is_invertible(A)
    print(f"Invertible: {'Sí' if inv_flag else 'No'}\n")

    print("=== NÚMERO DE CONDICIÓN ===")
    try:
        kappa = condition_number(A, p=p_norm)
        print(f"κ_{p_norm}(A) = {kappa:.6g}\n")
    except np.linalg.LinAlgError as e:
        print(f"No se pudo calcular el número de condición: {e}\n")
    

    if inv_flag:
        print("=== MATRIZ INVERSA A^{-1} ===")
        try:
            A_inv = inverse(A)
            print(A_inv, "\n")
            print("=== VALIDACIÓN DE LA INVERSA ===")
            v = validate_inverse(A, A_inv, tol=tol_inv)
            print(f"A·A^-1 ≈ I ?  {v.aa_inv_close_to_I}  (error máx = {v.max_abs_I_error_left:.3e})")
            print(f"A^-1·A ≈ I ?  {v.inv_a_a_close_to_I} (error máx = {v.max_abs_I_error_right:.3e})")
            for note in v.notes:
                print(" -", note)
            print()
        except np.linalg.LinAlgError as e:
            print(f"No se pudo invertir A: {e}\n")
    else:
        print("No se imprime A^{-1} porque A no es invertible.\n")

    print("=== EXPLICACIÓN ELEMENTO A ELEMENTO ===")
    try:
        expl = explain_matrix(A, node_labels=node_labels)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                print(expl[(i, j)])
    except Exception as e:
        print(f"No se pudo generar la explicación de A: {e}")
    print()

    if b is not None:
        print("=== SOLUCIÓN Ax=b Y VALIDACIÓN ===")
        try:
            x = solve(A, b)
            print("x =", np.array2string(x, precision=6, separator=', '))
            vsol = validate_solution(A, x, b, tol=tol_sol)
            print(f"Residual relativo ≈ {vsol.residual_norm_rel:.3e}")
            for note in vsol.notes:
                print(" -", note)
        except Exception as e:
            print(f"No se pudo resolver/validar Ax=b: {e}")
    else:
        print("b no fue proporcionado; se omite la resolución Ax=b.")
    print("\n=== FIN DEL ANÁLISIS ===")





def _example():
    import numpy as np

    
    A = np.array([
        [-2.0, 1.0, 1.0],
        [1.0, -2.0, 1.0],
        [1.0, 1.0, -2.0],
    ], dtype=float)

    b = np.array([1.0, 2.0, 3.0], dtype=float)

    
    labels = ["Nodo 1", "Nodo 2", "Nodo 3"]

    
    print_full_analysis(A, b=b, node_labels=labels, tol_inv=1e-9, tol_sol=1e-9, p_norm=2)

    
   



if __name__ == "__main__":
    _example()
