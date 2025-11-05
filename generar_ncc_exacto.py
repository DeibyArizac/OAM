#!/usr/bin/env python3
"""
Genera gráfica NCC EXACTA como Dashboard E
Líneas 3231-3242 del visualizer
"""

import numpy as np
import matplotlib.pyplot as plt

def get_ncc_values(cn2):
    """NCC hardcodeados por escenario"""
    if cn2 <= 1e-16:
        return [0.95, 0.90, 0.85]
    elif cn2 < 5e-15:
        return [0.90, 0.85, 0.80]
    elif cn2 < 2e-14:
        return [0.80, 0.70, 0.60]
    else:
        return [0.40, 0.30, 0.20]

# Escenarios
escenarios = [
    (1e-17, 'Escenario 0: Sin turbulencia (Cn²=1e-17, Ns=1)'),
    (2e-15, 'Escenario 1: Turbulencia baja (Cn²=2e-15, Ns=3)'),
    (8e-15, 'Escenario 2: Turbulencia media (Cn²=8e-15, Ns=5)'),
    (3e-14, 'Escenario 3: Turbulencia alta (Cn²=3e-14, Ns=7)')
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (cn2, titulo) in enumerate(escenarios):
    ax_ncc = axes[idx]
    
    # Labels y valores
    ncc_labels = ['|1|', '|2|', '|3|']
    ncc_means = get_ncc_values(cn2)
    
    # EXACTO línea 3231: colores según umbral
    colors_ncc = ['red' if m < 0.5 else ('orange' if m < 0.7 else 'green') for m in ncc_means]
    
    # EXACTO línea 3232: crear barras
    bars = ax_ncc.bar(ncc_labels, ncc_means, color=colors_ncc, edgecolor='black', alpha=0.8)
    
    # EXACTO líneas 3233-3236: labels y configuración
    ax_ncc.set_ylabel('NCC promedio')
    ax_ncc.set_title(f'{titulo}\nCorrelación Modal Promedio')
    ax_ncc.set_ylim([0, 1.0])
    ax_ncc.grid(True, alpha=0.3, axis='y')
    
    # EXACTO líneas 3238-3242: valores sobre barras
    for bar, val in zip(bars, ncc_means):
        height = bar.get_height()
        ax_ncc.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

fig.suptitle('Sistema OAM: Degradación por Turbulencia Atmosférica\n6 modos [-3,-2,-1,+1,+2,+3], 340m, λ=1550nm', 
            fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout()
output = '/opt/OAM_System/current_run/ncc_exacto.png'
plt.savefig(output, dpi=150, bbox_inches='tight')
print(f"✓ Guardado: {output}")
plt.show()
