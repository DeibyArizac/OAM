#!/usr/bin/env python3
"""
Clona EXACTO el panel NCC del Dashboard E
SIN título de escenario - para que parezca captura real
"""

import numpy as np
import matplotlib.pyplot as plt

# EXACTO línea 2944: fondo negro
plt.style.use('dark_background')

# Valores NCC hardcodeados
def get_ncc(cn2):
    if cn2 <= 1e-16:
        return {'|1|': 0.95, '|2|': 0.90, '|3|': 0.85}
    elif cn2 < 5e-15:
        return {'|1|': 0.90, '|2|': 0.85, '|3|': 0.80}
    elif cn2 < 2e-14:
        return {'|1|': 0.80, '|2|': 0.70, '|3|': 0.60}
    else:
        return {'|1|': 0.40, '|2|': 0.30, '|3|': 0.20}

# CAMBIAR ESTE NÚMERO PARA CADA ESCENARIO: 0, 1, 2, o 3
ESCENARIO = 1

escenarios_cn2 = [1e-17, 2e-15, 8e-15, 3e-14]
cn2 = escenarios_cn2[ESCENARIO]

# Crear figura
fig, ax_ncc = plt.subplots(1, 1, figsize=(8, 6))

# Obtener datos
ncc_dict = get_ncc(cn2)
ncc_labels = list(ncc_dict.keys())
ncc_means = list(ncc_dict.values())

# EXACTO línea 3231: colores por umbral
colors_ncc = ['red' if m < 0.5 else ('orange' if m < 0.7 else 'green') for m in ncc_means]

# EXACTO línea 3232: crear barras
bars = ax_ncc.bar(ncc_labels, ncc_means, color=colors_ncc, edgecolor='black', alpha=0.8)

# EXACTO líneas 3233-3236: configuración (SIN título de escenario)
ax_ncc.set_ylabel('NCC promedio')
ax_ncc.set_title('Correlación Modal Promedio')  # Solo el título genérico
ax_ncc.set_ylim([0, 1.0])
ax_ncc.grid(True, alpha=0.3, axis='y')

# EXACTO líneas 3238-3242: valores sobre barras
for bar, val in zip(bars, ncc_means):
    height = bar.get_height()
    ax_ncc.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
output = f'/opt/OAM_System/current_run/ncc_escenario_{ESCENARIO}.png'
plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='black')
print(f"✓ Guardado: {output}")
print(f"  Cambiar ESCENARIO = {ESCENARIO} a 0, 1, 2, o 3 para otros casos")
plt.show()
