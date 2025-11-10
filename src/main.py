import numpy as np
import matrix_operations as mo
import network_solver as ns
import scenario_simulator as ss
import results_analyzer as ra

def main():
    print("PROYECTO: OPTIMIZACIÓN DE REDES DE COMUNICACIONES")
    print("=" * 60)
    
    # 1. Matriz de conexiones (definida en el proyecto)
    A = np.array([
        [-3, 1, 1],
        [1, -3, 1], 
        [1, 1, -3]
    ])
    
    print("MATRIZ DE CONEXIONES:")
    print(f"A = \n{A}")
    print()
    
    # 2. ANÁLISIS MATRICIAL
    print("ANÁLISIS MATRICIAL")
    print("-" * 40)
    
    det_A = mo.determinant(A)
    is_inv = mo.is_invertible(A)
    cond_num = mo.condition_number(A)
    
    print(f"Determinante: {det_A:.6f}")
    print(f"Matriz invertible: {'Sí' if is_inv else 'No'}")
    print(f"Número de condición: {cond_num:.6f}")
    
    # Validación de reglas de la matriz
    rules = mo.validate_matrix_rules(A)
    print(f"Diagonal negativa: {rules.diagonal_ok} ({rules.diagonal_negatives}/{rules.diagonal_total} elementos < 0)")
    for note in rules.notes:
        print(f"  {note}")
    print()
    
    # 3. RESOLUCIÓN DEL SISTEMA BASE
    print("RESOLUCIÓN DEL SISTEMA BASE")
    print("-" * 40)
    
    # Vector de demanda base
    b_base = np.array([100, 200, 150])
    print(f"Vector de demanda base: {b_base}")
    
    comparison = ns.compare_solutions(A, b_base)
    
    inv_result = comparison["inverse"]
    solve_result = comparison["linalg_solve"]
    
    print("Método por inversa matricial:")
    if inv_result.success:
        print(f"  Solución: {inv_result.x}")
        print(f"  Residual relativo: {inv_result.residual_rel:.6e}")
    else:
        print(f"  Error: {inv_result.error}")
    
    print("Método por solve directo:")
    if solve_result.success:
        print(f"  Solución: {solve_result.x}")
        print(f"  Residual relativo: {solve_result.residual_rel:.6e}")
    else:
        print(f"  Error: {solve_result.error}")
    
    print("Comparación:")
    for note in comparison["compare_notes"]:
        print(f"  {note}")
    print()
    
    # 4. SIMULACIÓN DE ESCENARIOS
    print("SIMULACIÓN DE ESCENARIOS DE DEMANDA")
    print("-" * 40)
    
    all_results = ss.run_all_scenarios(A, method="numpy")
    
    print(f"Escenarios ejecutados: {len(all_results)}")
    for scenario_name, result in all_results.items():
        status = "ÉXITO" if result["success"] else "FALLÓ"
        print(f"  {scenario_name}: {status}")
    print()
    
    # 5. ANÁLISIS DE RESULTADOS
    print("ANÁLISIS DETALLADO DE RESULTADOS")
    print("-" * 40)
    
    analysis_results = ra.analyze_all(all_results, threshold_ratio=0.8)
    
    for scenario_name, analysis in analysis_results.items():
        print(f"ESCENARIO: {scenario_name}")
        print("-" * 30)
        
        metrics = analysis["metrics"]
        bottlenecks = analysis["bottlenecks"]
        recommendation = analysis["recommendation"]
        
        # Métricas básicas
        if metrics["solvable"]:
            print(f"Estado: Resuelto exitosamente")
            print(f"Flujo máximo: {metrics['x_max_abs']:.2f}")
            print(f"Flujo mínimo: {metrics['x_min']:.2f}")
            print(f"Flujo promedio: {metrics['x_mean']:.2f}")
            print(f"Residual relativo: {metrics['relative_residual']:.6e}")
        else:
            print(f"Estado: No se pudo resolver")
        
        # Cuellos de botella
        bottleneck_nodes = bottlenecks["bottleneck_nodes"]
        system_flags = bottlenecks["system_flags"]
        
        if bottleneck_nodes:
            print(f"Cuellos de botella detectados en nodos: {bottleneck_nodes}")
            for detail in bottlenecks["details"]:
                idx = detail["index"]
                flow = detail["flow"]
                load = detail["relative_load"]
                print(f"  Nodo {idx}: flujo={flow:.2f}, carga relativa={load:.2f}")
        else:
            print("No se detectaron cuellos de botella")
        
        if system_flags:
            print(f"Alertas del sistema: {', '.join(system_flags)}")
        
        # Recomendación
        print("Recomendaciones:")
        for line in recommendation.split('\n')[1:]:
            if line.strip():
                print(f"  {line.strip()}")
        
        print()
    
    print("RESUMEN")
    print("-" * 40)
    
    total_scenarios = len(analysis_results)
    successful_scenarios = sum(1 for a in analysis_results.values() 
                             if a["metrics"]["solvable"])
    scenarios_with_bottlenecks = sum(1 for a in analysis_results.values() 
                                   if a["bottlenecks"]["bottleneck_nodes"])
    
    print(f"Total de escenarios analizados: {total_scenarios}")
    print(f"Escenarios resueltos exitosamente: {successful_scenarios}")
    print(f"Escenarios con cuellos de botella: {scenarios_with_bottlenecks}")
    print(f"Escenarios sin problemas: {successful_scenarios - scenarios_with_bottlenecks}")
    
    # Recomendaciones generales
    print("\nRECOMENDACIONES GENERALES:")
    if scenarios_with_bottlenecks > total_scenarios / 2:
        print("- Considerar redistribución general de la carga en la red")
        print("- Evaluar aumento de capacidad en enlaces críticos")
        print("- Implementar políticas de balanceo de carga")
    else:
        print("- La red opera eficientemente en la mayoría de los escenarios")
        print("- Mantener la configuración actual para operación normal")
        print("- Monitorear continuamente para detectar cambios en los patrones de demanda")
    
    print("\n" + "=" * 60)
    print("EJECUCIÓN COMPLETADA - Proyecto de Optimización de Redes")

if __name__ == "__main__":
    main()