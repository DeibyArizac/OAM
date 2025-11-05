#!/bin/bash

# ============================================================================
# TEST COMPLETO DEL SISTEMA DE PROFILING
# ============================================================================
# Este script demuestra el uso completo del sistema de medición de tiempos
# para el sistema OAM.
#
# Autor: Deiby Fernando Ariza Cadena
# Propósito: Demostración para el director de tesis
# ============================================================================

echo "================================================================================"
echo " TEST DEL SISTEMA DE PROFILING - OAM"
echo "================================================================================"
echo ""

# Directorio del sistema
OAM_DIR="/opt/OAM_System"
cd "$OAM_DIR"

echo "📁 Directorio de trabajo: $OAM_DIR"
echo ""

# Limpiar ejecuciones anteriores
echo "🧹 Limpiando datos de ejecuciones anteriores..."
rm -rf current_run/*.npz current_run/*.json current_run/*.png current_run/.done 2>/dev/null
echo "✓ Limpieza completada"
echo ""

# Ejecutar sistema en modo headless con profiling
echo "🚀 Ejecutando sistema OAM con profiling activado..."
echo "   (Esto puede tomar 30-60 segundos dependiendo de la configuración)"
echo ""

python3 oam_complete_system.py --headless --config current_run/config_from_grc.json 2>&1 | tee current_run/execution.log

# Verificar si la ejecución fue exitosa
if [ ! -f "current_run/profiling_report.json" ]; then
    echo "❌ ERROR: No se generó el reporte de profiling"
    echo "   Revisar logs en: current_run/execution.log"
    exit 1
fi

echo ""
echo "✓ Sistema ejecutado exitosamente"
echo ""

# Mostrar reporte JSON
echo "================================================================================"
echo " REPORTE JSON GENERADO"
echo "================================================================================"
echo ""
cat current_run/profiling_report.json | python3 -m json.tool
echo ""

# Generar gráficas
echo "================================================================================"
echo " GENERANDO GRÁFICAS DE PROFILING"
echo "================================================================================"
echo ""

python3 generate_profiling_graphs.py current_run/profiling_report.json current_run

echo ""

# Listar archivos generados
echo "================================================================================"
echo " ARCHIVOS GENERADOS"
echo "================================================================================"
echo ""
echo "📊 Reporte JSON:"
ls -lh current_run/profiling_report.json
echo ""
echo "📈 Gráficas generadas:"
ls -lh current_run/profiling_*.png
echo ""

# Resumen final
echo "================================================================================"
echo " RESUMEN FINAL"
echo "================================================================================"
echo ""
echo "✓ Sistema de profiling ejecutado exitosamente"
echo ""
echo "📁 Ubicación de resultados:"
echo "   - Reporte JSON:  current_run/profiling_report.json"
echo "   - Gráficas PNG:  current_run/profiling_*.png"
echo "   - Log completo:  current_run/execution.log"
echo ""
echo "📖 Para más información, consultar: PROFILING_README.md"
echo ""
echo "================================================================================"
