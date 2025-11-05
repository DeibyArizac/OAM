# Sistema de Profiling - Medición de Tiempos de Ejecución

## Descripción

El sistema de profiling permite medir con precisión el tiempo de ejecución de cada etapa del procesamiento OAM (Source, Encoder, Channel, Decoder) para análisis de rendimiento y optimización.

## Arquitectura

El sistema de profiling está integrado en el pipeline principal y mide automáticamente:

1. **Tiempo total de ejecución** del sistema completo
2. **Tiempo por etapa** (Source, Encoder, Channel, Decoder)
3. **Tiempo promedio por símbolo** procesado
4. **Throughput** del sistema (símbolos/segundo)
5. **Distribución porcentual** del tiempo entre etapas

## Archivos del Sistema

### Módulos Principales

- **`oam_profiler.py`** - Módulo de profiling (medición de tiempos)
- **`generate_profiling_graphs.py`** - Generador de gráficas visuales
- **`PROFILING_README.md`** - Este documento

### Archivos Modificados

Los siguientes bloques fueron instrumentados con profiling:

- `oam_encoder.py` - Mide tiempo de codificación
- `oam_channel.py` - Mide tiempo de propagación atmosférica
- `oam_decoder.py` - Mide tiempo de decodificación
- `oam_complete_system.py` - Inicia/finaliza profiling total

## Uso

### Ejecución Automática

El profiling se ejecuta **automáticamente** cuando se ejecuta el sistema:

```bash
cd /opt/OAM_System

# Opción 1: Modo headless (sin GUI)
python3 oam_complete_system.py --headless --config current_run/config_from_grc.json

# Opción 2: Desde GNU Radio Companion
gnuradio-companion oam_complete_flowgraph.grc
# Luego presionar Run (▶)
```

### Ubicación de Resultados

Después de la ejecución, los resultados se guardan en:

- **`current_run/profiling_report.json`** - Reporte JSON con todas las métricas
- **Console output** - Resumen impreso en la terminal al finalizar

## Formato del Reporte JSON

```json
{
  "timestamp": "2025-01-27T10:30:15.123456",
  "total_time": 5.234,
  "symbol_count": 42,
  "throughput": 8.02,
  "stages": {
    "source": {
      "total": 0.123,
      "avg": 0.002929,
      "min": 0.002800,
      "max": 0.003100,
      "count": 42,
      "std": 0.000045
    },
    "encoder": {
      "total": 2.567,
      "avg": 0.061119,
      "min": 0.060500,
      "max": 0.062000,
      "count": 42,
      "std": 0.000234
    },
    "channel": {
      "total": 1.890,
      "avg": 0.045000,
      "min": 0.044800,
      "max": 0.045300,
      "count": 42,
      "std": 0.000098
    },
    "decoder": {
      "total": 0.654,
      "avg": 0.015571,
      "min": 0.015400,
      "max": 0.015800,
      "count": 42,
      "std": 0.000067
    }
  },
  "percentages": {
    "source": 2.35,
    "encoder": 49.05,
    "channel": 36.12,
    "decoder": 12.48
  },
  "stages_total": 5.234,
  "overhead": 0.000
}
```

## Interpretación de Métricas

### Métricas Generales

- **`total_time`**: Tiempo total de ejecución del sistema (segundos)
- **`symbol_count`**: Número total de símbolos procesados
- **`throughput`**: Velocidad de procesamiento (símbolos/segundo)
- **`overhead`**: Tiempo no contabilizado en etapas (overhead del sistema)

### Métricas por Etapa

Para cada etapa (source, encoder, channel, decoder):

- **`total`**: Tiempo total de la etapa (segundos)
- **`avg`**: Tiempo promedio por símbolo (segundos/símbolo)
- **`min`**: Tiempo mínimo registrado (segundos)
- **`max`**: Tiempo máximo registrado (segundos)
- **`count`**: Número de veces que se ejecutó la etapa
- **`std`**: Desviación estándar de los tiempos
- **`percentage`**: Porcentaje del tiempo total

## Generación de Gráficas

Para generar visualizaciones gráficas del reporte:

```bash
cd /opt/OAM_System
python3 generate_profiling_graphs.py current_run/profiling_report.json current_run
```

Esto genera 4 archivos PNG:

1. **`profiling_bar_chart.png`** - Gráfica de barras con tiempos y porcentajes
2. **`profiling_pie_chart.png`** - Gráfica circular con distribución
3. **`profiling_summary_table.png`** - Tabla resumen con todas las métricas
4. **`profiling_timeline.png`** - Línea de tiempo de ejecución secuencial

### Ejemplo de Uso Completo

```bash
# 1. Ejecutar sistema con profiling
cd /opt/OAM_System
python3 oam_complete_system.py --headless --config current_run/config_from_grc.json

# 2. Verificar reporte JSON
cat current_run/profiling_report.json

# 3. Generar gráficas
python3 generate_profiling_graphs.py

# 4. Ver gráficas generadas
ls -lh current_run/profiling_*.png
```

## Salida en Consola

Al finalizar la ejecución, el sistema imprime automáticamente un resumen:

```
================================================================================
 REPORTE DE TIEMPOS DE EJECUCIÓN - SISTEMA OAM
================================================================================

📊 RESUMEN GENERAL:
  • Tiempo total:        5.234 s
  • Símbolos procesados: 42
  • Throughput:          8.02 símbolos/s
  • Overhead sistema:    0.000 s (0.0%)

⏱️  TIEMPOS POR ETAPA:
  Etapa           Total (s)    Promedio (s)    Min (s)    Max (s)    %
  --------------- ------------ --------------- ---------- ---------- --------
  SOURCE               0.123        0.002929   0.002800   0.003100      2.4%
  ENCODER              2.567        0.061119   0.060500   0.062000     49.0%
  CHANNEL              1.890        0.045000   0.044800   0.045300     36.1%
  DECODER              0.654        0.015571   0.015400   0.015800     12.5%

  TOTAL STAGES         5.234                                        100.0%

================================================================================
```

## Análisis de Rendimiento

### Etapa Más Lenta

Identificar la etapa que consume más tiempo permite enfocar esfuerzos de optimización:

- **Encoder alto (>40%)** → Optimizar generación de haces LG
- **Channel alto (>40%)** → Optimizar propagación atmosférica o reducir Ns
- **Decoder alto (>30%)** → Optimizar correlación NCC o usar cache de templates

### Throughput

El throughput indica cuántos símbolos puede procesar el sistema por segundo:

- **> 10 sym/s** → Rendimiento bueno para investigación
- **5-10 sym/s** → Rendimiento aceptable
- **< 5 sym/s** → Considerar optimización

### Overhead

El overhead del sistema debería ser mínimo (<5%):

- **< 1%** → Excelente, casi todo el tiempo es procesamiento útil
- **1-5%** → Aceptable
- **> 5%** → Investigar causas (I/O, sincronización, etc.)

## Optimizaciones Sugeridas

Basado en los tiempos medidos:

### Si Encoder es el cuello de botella:

1. Usar cache de haces LG pre-generados
2. Reducir resolución de grilla (`grid_size=256` en vez de `512`)
3. Vectorizar operaciones NumPy

### Si Channel es el cuello de botella:

1. Reducir número de pantallas de fase (`Ns=1` en vez de `Ns>1`)
2. Optimizar FFT con algoritmos más rápidos
3. Considerar GPU (CUDA/OpenCL)

### Si Decoder es el cuello de botella:

1. Implementar cache de templates LG
2. Usar correlación parcial (sub-sampling)
3. Paralelizar detección de modos

## Información de Contacto

**Autor:** Deiby Fernando Ariza Cadena
**Email:** deibyarizac@gmail.com
**Director:** Dr. Omar Javier Tijaro Rojas
**Institución:** Universidad Industrial de Santander - Escuela E³T

## Versión

- **Versión del Sistema:** OAM 1.0 (Production7)
- **Versión del Profiler:** 1.0.0
- **Fecha:** Enero 2025

---

Para más información sobre el sistema OAM completo, consultar `README.md` en el directorio raíz del proyecto.
