#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BENCHMARK - Métricas de Desempeño del Sistema OAM
==================================================

Mide tiempos de ejecución de operaciones del sistema OAM:
- Generación de haces Laguerre-Gaussianos
- Generación de pantallas de fase atmosféricas
- Proyección modal para decodificación
- Procesamiento de símbolos completos

Autor: Deiby Fernando Ariza Cadena
"""

import numpy as np
import time
import json
from datetime import datetime
from scipy.special import genlaguerre, factorial


def generate_lg_beam(ell, p=0, grid_size=512, physical_size=0.02, wavelength=630e-9):
    """Generar haz Laguerre-Gaussiano"""
    x = np.linspace(-physical_size/2, physical_size/2, grid_size)
    y = np.linspace(-physical_size/2, physical_size/2, grid_size)
    X, Y = np.meshgrid(x, y)

    r = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)
    w0 = 0.002

    C = np.sqrt(2 * factorial(p) / (np.pi * factorial(p + abs(ell))))
    laguerre = genlaguerre(p, abs(ell))
    arg = 2 * r**2 / w0**2

    amplitude = (C / w0) * (r * np.sqrt(2) / w0)**abs(ell)
    amplitude *= np.exp(-r**2 / w0**2)
    amplitude *= laguerre(arg)

    phase = np.exp(1j * ell * phi)
    return amplitude * phase


def generate_phase_screen(grid_size=512, physical_size=0.02, cn2=1e-15,
                          wavelength=630e-9, distance=50):
    """Generar phase screen de Kolmogorov"""
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

    if np.var(phase) > 0:
        phase = phase * np.sqrt(phase_variance / np.var(phase))

    return np.exp(1j * phase)


def correlate_ncc(field1, field2):
    """Calcular coeficiente de correlación normalizado"""
    numerator = np.abs(np.sum(field1 * np.conj(field2)))
    denominator = np.sqrt(np.sum(np.abs(field1)**2) * np.sum(np.abs(field2)**2))
    return numerator / denominator if denominator > 0 else 0


def benchmark():
    """Ejecutar mediciones de rendimiento del sistema OAM"""

    print("=" * 70)
    print("BENCHMARK - Métricas de Desempeño del Sistema OAM")
    print("=" * 70)
    print()

    N_ITER = 10
    results = {}

    # 1. Generación de haz LG (512x512, un modo)
    print("1. Generación de haz LG (512x512, un modo)...")
    times = []
    for _ in range(N_ITER):
        start = time.time()
        field = generate_lg_beam(ell=2, grid_size=512)
        times.append((time.time() - start) * 1000)

    lg_time = np.mean(times)
    results['lg_beam_ms'] = round(lg_time, 2)
    print(f"   {lg_time:.2f} ms")
    print()

    # 2. Generación de pantalla de fase (primera vez)
    print("2. Generación de pantalla de fase (primera vez)...")
    times = []
    for _ in range(N_ITER):
        start = time.time()
        screen = generate_phase_screen(grid_size=512)
        times.append((time.time() - start) * 1000)

    phase_first = np.mean(times)
    results['phase_screen_first_ms'] = round(phase_first, 2)
    print(f"   {phase_first:.2f} ms")
    print()

    # 2b. Pantalla de fase con caché (simulado)
    print("2b. Generación de pantalla de fase (con caché)...")
    screen_cached = generate_phase_screen(grid_size=512)
    times = []
    for _ in range(N_ITER):
        start = time.time()
        screen_copy = screen_cached.copy()
        times.append((time.time() - start) * 1000)

    phase_cached = np.mean(times)
    results['phase_screen_cached_ms'] = round(phase_cached, 2)
    print(f"   {phase_cached:.2f} ms")
    print()

    # 3. Proyección modal completa (8 proyecciones)
    print("3. Proyección modal completa (4 canales x 2 signos = 8 proyecciones)...")
    field_rx = generate_lg_beam(ell=2, grid_size=512)
    modes = [-4, -3, -2, -1, +1, +2, +3, +4]
    templates = {mode: generate_lg_beam(ell=mode, grid_size=512) for mode in modes}

    times = []
    for _ in range(N_ITER):
        start = time.time()
        ncc_values = {mode: correlate_ncc(field_rx, templates[mode]) for mode in modes}
        detected = max(ncc_values, key=ncc_values.get)
        times.append((time.time() - start) * 1000)

    projection_time = np.mean(times)
    results['projection_8modes_ms'] = round(projection_time, 2)
    print(f"   {projection_time:.2f} ms")
    print()

    # 4. Símbolo completo (encode + channel + decode)
    print("4. Símbolo completo (encode + channel + decode)...")
    times = []
    for _ in range(N_ITER):
        start = time.time()

        # Encode: 4 haces LG
        field_encoded = np.zeros((512, 512), dtype=complex)
        for mode in [1, 2, 3, 4]:
            field_encoded += generate_lg_beam(ell=mode, grid_size=512)

        # Channel: turbulencia
        phase_screen = generate_phase_screen(grid_size=512)
        field_propagated = field_encoded * phase_screen

        # Decode: 8 proyecciones
        ncc_values = {mode: correlate_ncc(field_propagated, templates[mode])
                     for mode in modes}
        detected = max(ncc_values, key=ncc_values.get)

        times.append((time.time() - start) * 1000)

    symbol_time = np.mean(times)
    results['symbol_complete_ms'] = round(symbol_time, 2)
    print(f"   {symbol_time:.2f} ms")
    print()

    # 5. Mensaje de 10 símbolos
    message_time = symbol_time * 10
    results['message_10symbols_ms'] = round(message_time, 2)
    print(f"5. Mensaje de 10 símbolos (sin visualización):")
    print(f"   {message_time:.2f} ms = {message_time/1000:.2f} s")
    print()

    # Resumen
    print("=" * 70)
    print("RESUMEN DE RESULTADOS")
    print("=" * 70)
    print(f"{'Métrica':<50} {'Resultado'}")
    print("-" * 70)
    print(f"{'Haz LG (1 modo)':<50} {lg_time:.2f} ms")
    print(f"{'Phase screen (primera vez)':<50} {phase_first:.2f} ms")
    print(f"{'Phase screen (caché)':<50} {phase_cached:.2f} ms")
    print(f"{'Proyección modal (8 modos)':<50} {projection_time:.2f} ms")
    print(f"{'Símbolo completo':<50} {symbol_time:.2f} ms")
    print(f"{'10 símbolos':<50} {message_time/1000:.2f} s")
    print("=" * 70)
    print()

    # Guardar JSON
    output = {
        "timestamp": datetime.now().isoformat(),
        "hardware": "AMD Ryzen 5 5600H, 12 cores",
        "results_ms": results,
        "thesis_reference_ms": {
            "lg_beam_ms": 5,
            "phase_screen_first_ms": 15,
            "phase_screen_cached_ms": 0.5,
            "projection_8modes_ms": 40,
            "symbol_complete_ms": 80,
            "message_10symbols_ms": 800
        },
        "ratios": {
            "lg_beam": round(lg_time/5, 2),
            "phase_screen": round(phase_first/15, 2),
            "projection": round(projection_time/40, 2),
            "symbol": round(symbol_time/80, 2),
            "message": round((message_time/1000)/0.8, 2)
        }
    }

    with open('/opt/OAM_System/benchmark_tesis_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("Resultados guardados en: /opt/OAM_System/benchmark_tesis_results.json")


if __name__ == "__main__":
    benchmark()
