#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GENERADOR DE GRÁFICAS DE PROFILING
===================================

Genera visualizaciones de los tiempos de ejecución del sistema OAM
a partir del reporte JSON de profiling.

Autor: Deiby Fernando Ariza Cadena
Propósito: Presentar tiempos de ejecución de manera visual para análisis académico
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def load_profiling_report(filepath='current_run/profiling_report.json'):
    """Cargar reporte de profiling desde JSON"""
    if not os.path.exists(filepath):
        print(f"ERROR: No se encontró el archivo de profiling: {filepath}")
        return None

    with open(filepath, 'r') as f:
        report = json.load(f)

    return report

def generate_bar_chart(report, save_path='current_run/profiling_bar_chart.png'):
    """Generar gráfica de barras con tiempos por etapa"""
    stages = ['source', 'encoder', 'channel', 'decoder']
    times = [report['stages'][stage]['total'] for stage in stages]
    percentages = [report['percentages'][stage] for stage in stages]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfica 1: Tiempos absolutos
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
    bars1 = ax1.bar(stages, times, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Tiempo (segundos)', fontsize=12, fontweight='bold')
    ax1.set_title('Tiempo de Ejecución por Etapa', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(times) * 1.15)
    ax1.grid(axis='y', alpha=0.3)

    # Etiquetas de valores sobre las barras
    for bar, time in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.3f}s',
                ha='center', va='bottom', fontweight='bold')

    # Gráfica 2: Porcentajes
    bars2 = ax2.bar(stages, percentages, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Porcentaje del Tiempo Total (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Distribución Porcentual del Tiempo', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)

    # Etiquetas de valores sobre las barras
    for bar, pct in zip(bars2, percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    # Formato de etiquetas del eje X
    for ax in [ax1, ax2]:
        ax.set_xticklabels([s.upper() for s in stages], fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica de barras guardada: {save_path}")
    plt.close()

def generate_pie_chart(report, save_path='current_run/profiling_pie_chart.png'):
    """Generar gráfica circular (pie chart) con distribución de tiempos"""
    stages = ['source', 'encoder', 'channel', 'decoder']
    times = [report['stages'][stage]['total'] for stage in stages]
    percentages = [report['percentages'][stage] for stage in stages]

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
    explode = (0.05, 0.05, 0.05, 0.05)  # Separar ligeramente cada sector

    wedges, texts, autotexts = ax.pie(
        times,
        explode=explode,
        labels=[s.upper() for s in stages],
        autopct=lambda pct: f'{pct:.1f}%\n({times[int(pct/100*len(times))]:.3f}s)',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )

    ax.set_title(f'Distribución de Tiempos de Ejecución\n'
                 f'Total: {report["total_time"]:.3f}s | Throughput: {report["throughput"]:.2f} sym/s',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica circular guardada: {save_path}")
    plt.close()

def generate_summary_table(report, save_path='current_run/profiling_summary_table.png'):
    """Generar tabla resumen con todas las métricas"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Datos de la tabla
    stages = ['SOURCE', 'ENCODER', 'CHANNEL', 'DECODER', 'TOTAL']
    data = []

    for stage in ['source', 'encoder', 'channel', 'decoder']:
        stats = report['stages'][stage]
        pct = report['percentages'][stage]
        data.append([
            stage.upper(),
            f"{stats['total']:.3f}",
            f"{stats['avg']:.6f}",
            f"{stats['min']:.6f}",
            f"{stats['max']:.6f}",
            f"{stats['count']}",
            f"{pct:.1f}%"
        ])

    # Fila de totales
    data.append([
        'TOTAL',
        f"{report['stages_total']:.3f}",
        '-',
        '-',
        '-',
        f"{report['symbol_count']}",
        '100.0%'
    ])

    # Encabezados
    headers = ['Etapa', 'Total (s)', 'Promedio (s)', 'Min (s)', 'Max (s)', 'Conteo', 'Porcentaje']

    # Crear tabla
    table = ax.table(cellText=data, colLabels=headers, cellLoc='center', loc='center',
                    colWidths=[0.15, 0.12, 0.15, 0.12, 0.12, 0.12, 0.12])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Estilo de encabezados
    for i, header in enumerate(headers):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Estilo de filas
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
    for i in range(1, 5):
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(colors[i-1])
            table[(i, j)].set_alpha(0.3)

    # Estilo de fila total
    for j in range(len(headers)):
        table[(5, j)].set_facecolor('#34495e')
        table[(5, j)].set_text_props(weight='bold', color='white')

    # Título
    ax.set_title(f'Reporte de Tiempos de Ejecución - Sistema OAM\n'
                 f'Tiempo Total: {report["total_time"]:.3f}s | Throughput: {report["throughput"]:.2f} símbolos/s | Overhead: {report["overhead"]:.3f}s ({report["overhead"]/report["total_time"]*100:.1f}%)',
                 fontsize=13, fontweight='bold', pad=20)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Tabla resumen guardada: {save_path}")
    plt.close()

def generate_timeline(report, save_path='current_run/profiling_timeline.png'):
    """Generar línea de tiempo con ejecución secuencial"""
    stages = ['source', 'encoder', 'channel', 'decoder']
    times = [report['stages'][stage]['total'] for stage in stages]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Calcular posiciones temporales
    positions = []
    current_pos = 0
    for time in times:
        positions.append((current_pos, current_pos + time))
        current_pos += time

    # Colores
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']

    # Dibujar barras horizontales
    y_pos = 1
    for i, (stage, (start, end)) in enumerate(zip(stages, positions)):
        duration = end - start
        ax.barh(y_pos, duration, left=start, height=0.6, color=colors[i],
                edgecolor='black', linewidth=2, alpha=0.7, label=stage.upper())

        # Etiqueta con tiempo
        ax.text((start + end) / 2, y_pos, f'{duration:.3f}s',
                ha='center', va='center', fontweight='bold', fontsize=11)

    ax.set_xlabel('Tiempo (segundos)', fontsize=12, fontweight='bold')
    ax.set_title(f'Línea de Tiempo de Ejecución Secuencial\nTotal: {report["total_time"]:.3f}s',
                fontsize=14, fontweight='bold')
    ax.set_ylim(0.5, 1.5)
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Línea de tiempo guardada: {save_path}")
    plt.close()

def generate_all_graphs(report_path='current_run/profiling_report.json', output_dir='current_run'):
    """Generar todas las gráficas de profiling"""
    print("=" * 80)
    print(" GENERADOR DE GRÁFICAS DE PROFILING - SISTEMA OAM")
    print("=" * 80)

    # Cargar reporte
    report = load_profiling_report(report_path)
    if report is None:
        return False

    print(f"\n✓ Reporte cargado exitosamente: {report_path}")
    print(f"  - Tiempo total: {report['total_time']:.3f}s")
    print(f"  - Símbolos procesados: {report['symbol_count']}")
    print(f"  - Throughput: {report['throughput']:.2f} símbolos/s")

    print(f"\n📊 Generando gráficas...")

    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Generar todas las gráficas
    generate_bar_chart(report, os.path.join(output_dir, 'profiling_bar_chart.png'))
    generate_pie_chart(report, os.path.join(output_dir, 'profiling_pie_chart.png'))
    generate_summary_table(report, os.path.join(output_dir, 'profiling_summary_table.png'))
    generate_timeline(report, os.path.join(output_dir, 'profiling_timeline.png'))

    print(f"\n✓ Todas las gráficas generadas exitosamente en: {output_dir}/")
    print("=" * 80)

    return True

if __name__ == "__main__":
    # Argumentos de línea de comandos
    if len(sys.argv) > 1:
        report_path = sys.argv[1]
    else:
        report_path = 'current_run/profiling_report.json'

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = 'current_run'

    # Generar gráficas
    success = generate_all_graphs(report_path, output_dir)

    sys.exit(0 if success else 1)
