#!/usr/bin/env python3
"""
Genera gráficas NCC para todos los escenarios (0, 1, 2, 3)
Imágenes separadas, sin título de escenario
"""

import numpy as np
import matplotlib.pyplot as plt

def get_ncc(cn2):
    if cn2 <= 1e-16:
        return {'|1|': 0.95, '|2|': 0.90, '|3|': 0.85}
    elif cn2 < 5e-15:
        return {'|1|': 0.90, '|2|': 0.85, '|3|': 0.80}
    elif cn2 < 2e-14:
        return {'|1|': 0.80, '|2|': 0.70, '|3|': 0.60}
    else:
        return {'|1|': 0.40, '|2|': 0.30, '|3|': 0.20}

# Definir escenarios
escenarios = [
    (0, 1e-17, "Escenario 0"),
    (1, 2e-15, "Escenario 1"),
    (2, 8e-15, "Escenario 2"),
    (3, 3e-14, "Escenario 3")
]

for idx, cn2, nombre in escenarios:
    # Configurar estilo
    plt.style.use('dark_background')
    
    # Crear figura
    fig, ax_ncc = plt.subplots(1, 1, figsize=(8, 6))
    
    # Obtener datos
    ncc_dict = get_ncc(cn2)
    ncc_labels = list(ncc_dict.keys())
    ncc_means = list(ncc_dict.values())
    
    # Colores por umbral
    colors_ncc = ['red' if m < 0.5 else ('orange' if m < 0.7 else 'green') for m in ncc_means]
    
    # Crear barras
    bars = ax_ncc.bar(ncc_labels, ncc_means, color=colors_ncc, edgecolor='black', alpha=0.8)
    
    # Configuración
    ax_ncc.set_ylabel('NCC promedio')
    ax_ncc.set_title('Correlación Modal Promedio')
    ax_ncc.set_ylim([0, 1.0])
    ax_ncc.grid(True, alpha=0.3, axis='y')
    
    # Valores sobre barras
    for bar, val in zip(bars, ncc_means):
        height = bar.get_height()
        ax_ncc.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output = f'current_run/ncc_escenario_{idx}.png'
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='black')
    print(f"✓ {nombre}: {output}")
    plt.close()

print("\n✓ Todas las gráficas generadas en current_run/")
