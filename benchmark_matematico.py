#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BENCHMARK MATEMÁTICO PURO - Sistema OAM
========================================

Mide tiempos de operaciones matemáticas individuales SIN overhead
de GNU Radio, logging, o I/O. Replica las mediciones del Cap. 3 de la tesis.

Autor: Deiby Fernando Ariza Cadena
Hardware: Medido en /opt/OAM_System
"""

import numpy as np
import time
import json
from datetime import datetime
from scipy.special import genlaguerre, factorial
from multiprocessing import Pool, cpu_count

# ============================================================================
# FUNCIONES MATEMÁTICAS PURAS
# ============================================================================

def generate_lg_beam_pure(ell, p=0, grid_size=512, physical_size=0.02, wavelength=630e-9):
    """Generar haz LG (versión pura, sin logging)"""
    x = np.linspace(-physical_size/2, physical_size/2, grid_size)
    y = np.linspace(-physical_size/2, physical_size/2, grid_size)
    X, Y = np.meshgrid(x, y)

    r = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)

    w0 = 0.002  # 2 mm

    C = np.sqrt(2 * factorial(p) / (np.pi * factorial(p + abs(ell))))
    laguerre = genlaguerre(p, abs(ell))
    arg = 2 * r**2 / w0**2

    amplitude = (C / w0) * (r * np.sqrt(2) / w0)**abs(ell)
    amplitude *= np.exp(-r**2 / w0**2)
    amplitude *= laguerre(arg)

    phase = np.exp(1j * ell * phi)
    field = amplitude * phase

    return field


def generate_phase_screen_pure(grid_size=512, physical_size=0.02,
                                cn2=1e-15, wavelength=630e-9, distance=50):
    """Generar phase screen de Kolmogorov (versión pura)"""
    delta = physical_size / grid_size
    fx = np.fft.fftfreq(grid_size, delta)
    fy = np.fft.fftfreq(grid_size, delta)
    FX, FY = np.meshgrid(fx, fy)

    f = np.sqrt(FX**2 + FY**2)
    f[0, 0] = 1e-10

    k = 2 * np.pi * f
    phi_n = 0.033 * cn2 * k**(-11/3)

    r0 = (0.423 * (2*np.pi/wavelength)**2 * cn2 * distance)**(-3/5)
    phase_variance = 1.03 * (physical_size / r0)**(5/3)

    cn = np.random.randn(grid_size, grid_size) + 1j * np.random.randn(grid_size, grid_size)
    phase_ft = np.sqrt(phi_n) * cn * delta
    phase = np.fft.ifft2(phase_ft).real

    phase = phase * np.sqrt(phase_variance / np.var(phase))

    return np.exp(1j * phase)


def correlate_ncc_pure(field1, field2):
    """Calcular NCC entre dos campos (versión pura)"""
    numerator = np.abs(np.sum(field1 * np.conj(field2)))
    denominator = np.sqrt(np.sum(np.abs(field1)**2) * np.sum(np.abs(field2)**2))
    return numerator / denominator if denominator > 0 else 0


# ============================================================================
# BENCHMARK INDIVIDUAL
# ============================================================================

def benchmark_lg_generation(n_iterations=10):
    """Medir generación de haz LG individual"""
    print("\n1⃣  GENERACIÓN DE HAZ LG (512×512, un modo)")
    print("   " + "="*50)

    times = []
    for i in range(n_iterations):
        start = time.time()
        field = generate_lg_beam_pure(ell=2, grid_size=512)
        end = time.time()
        times.append(end - start)

    avg_time = np.mean(times) * 1000  # convertir a ms
    std_time = np.std(times) * 1000

    print(f"   ️  Promedio: {avg_time:.2f} ms")
    print(f"    Std Dev:  {std_time:.2f} ms")
    print(f"    Rango:    {min(times)*1000:.2f} - {max(times)*1000:.2f} ms")

    return avg_time


def benchmark_phase_screen(n_iterations=10):
    """Medir generación de phase screen (primera vez y con caché)"""
    print("\n2⃣  GENERACIÓN DE PANTALLA DE FASE (512×512)")
    print("   " + "="*50)

    # Primera generación (sin caché)
    times_first = []
    for i in range(n_iterations):
        start = time.time()
        screen = generate_phase_screen_pure(grid_size=512)
        end = time.time()
        times_first.append(end - start)

    avg_first = np.mean(times_first) * 1000

    # Simular caché: guardar y cargar
    screen_cached = generate_phase_screen_pure(grid_size=512)
    times_cached = []
    for i in range(n_iterations):
        start = time.time()
        # Simular lectura de caché (solo copia del array)
        screen_copy = screen_cached.copy()
        end = time.time()
        times_cached.append(end - start)

    avg_cached = np.mean(times_cached) * 1000

    print(f"   ️  Primera vez (sin caché): {avg_first:.2f} ms")
    print(f"    Con caché (array copy):  {avg_cached:.2f} ms")
    print(f"    Mejora con caché: {avg_first/avg_cached:.1f}×")

    return avg_first, avg_cached


def _correlate_mode(args):
    """Helper para paralelizar correlaciones"""
    field_rx, template = args
    return correlate_ncc_pure(field_rx, template)


def benchmark_modal_projection(n_iterations=10, parallel=False):
    """Medir proyección modal completa (8 proyecciones NCC)"""
    print(f"\n3⃣  PROYECCIÓN MODAL COMPLETA (4 canales × 2 signos = 8 proyecciones)")
    print(f"   {'[PARALELO]' if parallel else '[SECUENCIAL]'}")
    print("   " + "="*50)

    # Generar campo recibido
    field_rx = generate_lg_beam_pure(ell=2, grid_size=512)

    # Generar templates (cache de referencia)
    modes = [-4, -3, -2, -1, +1, +2, +3, +4]
    templates = {}
    for mode in modes:
        templates[mode] = generate_lg_beam_pure(ell=mode, grid_size=512)

    # Medir tiempo de las 8 correlaciones
    times = []

    if parallel:
        n_cores = cpu_count()
        print(f"    Usando {n_cores} núcleos")

        for i in range(n_iterations):
            start = time.time()

            # Preparar argumentos para paralelización
            args = [(field_rx, templates[mode]) for mode in modes]

            # Calcular NCC en paralelo
            with Pool(processes=min(8, n_cores)) as pool:
                ncc_list = pool.map(_correlate_mode, args)

            ncc_values = dict(zip(modes, ncc_list))
            detected = max(ncc_values, key=ncc_values.get)

            end = time.time()
            times.append(end - start)
    else:
        for i in range(n_iterations):
            start = time.time()

            # Calcular NCC con todos los modos (secuencial)
            ncc_values = {}
            for mode in modes:
                ncc_values[mode] = correlate_ncc_pure(field_rx, templates[mode])

            # Detectar modo dominante
            detected = max(ncc_values, key=ncc_values.get)

            end = time.time()
            times.append(end - start)

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000

    print(f"   ️  Promedio: {avg_time:.2f} ms")
    print(f"    Std Dev:  {std_time:.2f} ms")
    print(f"    Rango:    {min(times)*1000:.2f} - {max(times)*1000:.2f} ms")
    print(f"    Tiempo por proyección: {avg_time/8:.2f} ms")

    return avg_time


def benchmark_complete_symbol():
    """Medir símbolo completo (encode + channel + decode)"""
    print("\n4⃣  SÍMBOLO COMPLETO (encode + channel + decode)")
    print("   " + "="*50)

    n_iterations = 10
    times = []

    for i in range(n_iterations):
        start = time.time()

        # ENCODER: Generar 4 haces LG en paralelo
        modes = [1, 2, 3, 4]
        field_encoded = np.zeros((512, 512), dtype=complex)
        for mode in modes:
            field_encoded += generate_lg_beam_pure(ell=mode, grid_size=512)

        # CHANNEL: Aplicar turbulencia
        phase_screen = generate_phase_screen_pure(grid_size=512)
        field_propagated = field_encoded * phase_screen

        # DECODER: Proyección modal (8 modos)
        modes_test = [-4, -3, -2, -1, +1, +2, +3, +4]
        templates = {}
        for mode in modes_test:
            templates[mode] = generate_lg_beam_pure(ell=mode, grid_size=512)

        ncc_values = {}
        for mode in modes_test:
            ncc_values[mode] = correlate_ncc_pure(field_propagated, templates[mode])

        detected = max(ncc_values, key=ncc_values.get)

        end = time.time()
        times.append(end - start)

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000

    print(f"   ️  Promedio: {avg_time:.2f} ms")
    print(f"    Std Dev:  {std_time:.2f} ms")
    print(f"    Rango:    {min(times)*1000:.2f} - {max(times)*1000:.2f} ms")

    return avg_time


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_benchmark():
    """Ejecutar benchmark completo"""
    n_cores = cpu_count()

    print("\n" + "="*70)
    print("  BENCHMARK MATEMÁTICO PURO - Sistema OAM")
    print("  (Sin GNU Radio, sin logging, sin I/O)")
    print("="*70)
    print(f"\n⚙  Configuración:")
    print(f"   - Grid: 512×512 píxeles")
    print(f"   - Longitud de onda: 630 nm")
    print(f"   - Modos: 8 (4 canales × 2 signos)")
    print(f"   - Iteraciones por test: 10")
    print(f"   - CPU: {n_cores} núcleos disponibles")

    # Ejecutar benchmarks
    time_lg = benchmark_lg_generation()
    time_phase_first, time_phase_cached = benchmark_phase_screen()

    # Proyección modal: secuencial y paralelo
    time_projection_seq = benchmark_modal_projection(parallel=False)
    time_projection_par = benchmark_modal_projection(parallel=True)

    time_symbol = benchmark_complete_symbol()

    # Proyección a 10 símbolos
    time_10_symbols = time_symbol * 10

    # Resumen
    print("\n" + "="*70)
    print("   RESUMEN DE RESULTADOS")
    print("="*70)
    print(f"\n  Componentes individuales:")
    print(f"  ├─ Haz LG (un modo):              {time_lg:.2f} ms")
    print(f"  ├─ Phase screen (primera vez):    {time_phase_first:.2f} ms")
    print(f"  ├─ Phase screen (con caché):      {time_phase_cached:.2f} ms")
    print(f"  ├─ Proyección modal (secuencial): {time_projection_seq:.2f} ms")
    print(f"  └─ Proyección modal (paralelo):   {time_projection_par:.2f} ms")
    print(f"      └─ Speedup: {time_projection_seq/time_projection_par:.2f}×")
    print(f"\n  Símbolo completo:")
    print(f"  └─ Encode + Channel + Decode:  {time_symbol:.2f} ms")
    print(f"\n  Mensaje completo (10 símbolos):")
    print(f"  └─ Tiempo total estimado:      {time_10_symbols:.2f} ms = {time_10_symbols/1000:.2f} s")
    print(f"  └─ Throughput:                 {10/(time_10_symbols/1000):.2f} símbolos/s")

    print("\n" + "="*70)
    print("   Benchmark completado")
    print("="*70 + "\n")

    # Comparación con tesis
    print(" COMPARACIÓN CON TESIS (Cap. 3):")
    print("="*70)
    print(f"{'Medida':<40} {'Tesis':<15} {'Medido':<15} {'Ratio'}")
    print("-"*70)
    print(f"{'Haz LG (1 modo)':<40} {'~5 ms':<15} {f'{time_lg:.1f} ms':<15} {time_lg/5:.2f}×")
    print(f"{'Phase screen (primera vez)':<40} {'~15 ms':<15} {f'{time_phase_first:.1f} ms':<15} {time_phase_first/15:.2f}×")
    print(f"{'Phase screen (con caché)':<40} {'~0.5 ms':<15} {f'{time_phase_cached:.2f} ms':<15} {time_phase_cached/0.5:.2f}×")
    print(f"{'Proyección modal (secuencial)':<40} {'~40 ms':<15} {f'{time_projection_seq:.1f} ms':<15} {time_projection_seq/40:.2f}×")
    print(f"{'Proyección modal (paralelo)':<40} {'~40 ms':<15} {f'{time_projection_par:.1f} ms':<15} {time_projection_par/40:.2f}×")
    print(f"{'Símbolo completo':<40} {'~80 ms':<15} {f'{time_symbol:.1f} ms':<15} {time_symbol/80:.2f}×")
    print(f"{'10 símbolos':<40} {'~0.8 s':<15} {f'{time_10_symbols/1000:.2f} s':<15} {(time_10_symbols/1000)/0.8:.2f}×")
    print("="*70 + "\n")

    # Guardar resultados en JSON
    results = {
        "timestamp": datetime.now().isoformat(),
        "hardware": {
            "cpu_cores": n_cores,
            "cpu_model": "AMD Ryzen 5 5600H"
        },
        "config": {
            "grid_size": 512,
            "wavelength_nm": 630,
            "n_modes": 8,
            "iterations": 10
        },
        "results_ms": {
            "lg_beam_single": round(time_lg, 2),
            "phase_screen_first": round(time_phase_first, 2),
            "phase_screen_cached": round(time_phase_cached, 2),
            "projection_sequential": round(time_projection_seq, 2),
            "symbol_complete": round(time_symbol, 2),
            "message_10_symbols": round(time_10_symbols, 2)
        },
        "comparison_vs_thesis": {
            "lg_beam_ratio": round(time_lg/5, 2),
            "phase_screen_ratio": round(time_phase_first/15, 2),
            "projection_ratio": round(time_projection_seq/40, 2),
            "symbol_ratio": round(time_symbol/80, 2),
            "message_ratio": round((time_10_symbols/1000)/0.8, 2)
        }
    }

    output_file = "/opt/OAM_System/benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Resultados guardados en: {output_file}\n")


if __name__ == "__main__":
    run_benchmark()
