#!/bin/bash
#
# Script de instalación del Sistema OAM para GNU Radio
# Codificación y Decodificación a partir del Momento Angular Orbital de la Luz
#
# Autor: Deiby Fernando Ariza Cadena
# Director: Dr. Omar Javier Tíjaro Rojas
# Institución: Universidad Industrial de Santander (UIS)
# Repositorio: https://github.com/DeibyArizac/OAM

set -e

echo "=========================================="
echo "Sistema OAM - GNU Radio"
echo "Instalación de bloques personalizados"
echo "=========================================="
echo ""

# Detectar directorio del script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLOCKS_DIR="$HOME/.grc_gnuradio/blocks"

# Crear directorio de bloques si no existe
echo "[1/3] Creando directorio de bloques GNU Radio..."
mkdir -p "$BLOCKS_DIR"

# Copiar archivos .block.yml
echo "[2/3] Copiando definiciones de bloques OAM..."
if [ -d "$SCRIPT_DIR/grc" ]; then
    cp -v "$SCRIPT_DIR/grc/"*.block.yml "$BLOCKS_DIR/"
    echo "✓ Bloques copiados a: $BLOCKS_DIR"
else
    echo "ERROR: No se encontró el directorio grc/"
    exit 1
fi

# Verificar instalación
echo "[3/3] Verificando instalación..."
BLOCK_COUNT=$(ls -1 "$BLOCKS_DIR"/oam_*.block.yml 2>/dev/null | wc -l)
if [ "$BLOCK_COUNT" -eq 5 ]; then
    echo "✓ Instalación completada: $BLOCK_COUNT bloques OAM instalados"
else
    echo "⚠ Advertencia: Se encontraron $BLOCK_COUNT bloques (esperados: 5)"
fi

echo ""
echo "=========================================="
echo "Instalación completada"
echo "=========================================="
echo ""
echo "Próximos pasos:"
echo "1. Instalar dependencias Python:"
echo "   pip3 install -r $SCRIPT_DIR/requirements.txt"
echo ""
echo "2. Abrir GNU Radio Companion:"
echo "   gnuradio-companion"
echo ""
echo "3. Buscar bloques [OAM] en la paleta de bloques"
echo ""
echo "4. Abrir flowgraph de ejemplo:"
echo "   $SCRIPT_DIR/oam_complete_flowgraph.grc"
echo ""
echo "Para más información, consulta README.md"
echo "=========================================="