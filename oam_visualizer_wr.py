#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WRAPPER GNU RADIO: OAM VISUALIZER BLOCK
======================================

Este bloque NO ejecuta código - solo edita parámetros de visualización.

Función:
- Recibe parámetros del GUI de GNU Radio relacionados con dashboards
- Actualiza config_manager con estos parámetros
- El bloque OAM Source es el que realmente lanza los dashboards
- Proporciona interfaz visual para controlar dashboards

Parámetros del Visualizer:
- enable_dashboard_a: Habilitar Dashboard A (análisis temporal)
- enable_dashboard_b: Habilitar Dashboard B (métricas QA)
- enable_dashboard_c: Habilitar Dashboard C (snapshot detallado)
- enable_dashboard_d: Habilitar Dashboard D (stream modal)
- dashboard_step_delay: Tiempo entre símbolos en dashboards dinámicos [s]
"""

import os
import numpy as np
from gnuradio import gr
import sys

# Agregar directorio actual al path si no está
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from config_manager import config_manager
from oam_logging import log_info, log_debug, log_warning

class oam_visualizer_wr(gr.sync_block):
    """
    Wrapper GNU Radio para OAM Visualizer

    FUNCIÓN: Solo actualizar parámetros de visualización - NO EJECUTA CÓDIGO

    Este bloque:
    1. Recibe parámetros del GUI
    2. Los valida
    3. Los guarda en config_manager
    4. ¡No lanza dashboards!
    5. OAM Source lanzará los dashboards según esta configuración
    """

    def __init__(self, enable_dashboard_a=True, enable_dashboard_b=True,
                 enable_dashboard_c=True, enable_dashboard_d=True, enable_dashboard_e=False,
                 dashboard_step_delay=3.0):
        """
        Inicializar OAM Visualizer - Solo configuración

        Args:
            enable_dashboard_a: Habilitar Dashboard A - Dynamic temporal analysis
            enable_dashboard_b: Habilitar Dashboard B - Dynamic QA metrics
            enable_dashboard_c: Habilitar Dashboard C - Detailed snapshot
            enable_dashboard_d: Habilitar Dashboard D - Dynamic modal stream
            enable_dashboard_e: Habilitar Dashboard E - Resumen de métricas globales
            dashboard_step_delay: Tiempo entre símbolos [s] (0.5 a 10.0)
        """
        gr.sync_block.__init__(
            self,
            name="oam_visualizer_wr",
            in_sig=[np.complex64],
            out_sig=None
        )

        # Guardar parámetros en el objeto
        self.enable_dashboard_a = enable_dashboard_a
        self.enable_dashboard_b = enable_dashboard_b
        self.enable_dashboard_c = enable_dashboard_c
        self.enable_dashboard_d = enable_dashboard_d
        self.enable_dashboard_e = enable_dashboard_e
        self.dashboard_step_delay = dashboard_step_delay

        log_debug('visualizer_wr', "=== CONFIGURING PARAMETERS VISUALIZER ===")

        # Validar parámetros antes de guardar
        self._validate_visualizer_params(enable_dashboard_a, enable_dashboard_b,
                                       enable_dashboard_c, enable_dashboard_d, enable_dashboard_e,
                                       dashboard_step_delay)

        # Actualizar parámetros en config_manager
        self._update_config()

        log_debug('visualizer_wr', "=== VISUALIZER CONFIGURACIÓN COMPLETADA ===")

    def start(self):
        """Llamado cuando GNU Radio presiona Run - escribir parámetros y esperar artefactos"""
        import time
        import threading
        from gnuradio_cache import write_block_params, get_run_id_from_cache

        # 1. Obtener run_id del caché (generado por Source)
        run_id = get_run_id_from_cache()

        if not run_id:
            log_warning('visualizer_wr', "run_id not found in cache - using timestamp")
            from datetime import datetime
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 2. Escribir parámetros al caché
        visualizer_params = {
            'enable_dashboard_a': self.enable_dashboard_a,
            'enable_dashboard_b': self.enable_dashboard_b,
            'enable_dashboard_c': self.enable_dashboard_c,
            'enable_dashboard_d': self.enable_dashboard_d,
            'enable_dashboard_e': self.enable_dashboard_e,
            'dashboard_step_delay': self.dashboard_step_delay
        }

        write_block_params('oam_visualizer', visualizer_params, run_id)
        log_debug('visualizer_wr', f"Parameters written to cache (run_id: {run_id})")

        # 3. Lanzar thread para esperar artefactos y abrir dashboards
        dashboard_thread = threading.Thread(target=self._wait_and_launch_dashboards, daemon=True)
        dashboard_thread.start()

        return True

    def _wait_and_launch_dashboards(self):
        """Esperar a que existan los artefactos del pipeline y luego abrir dashboards"""
        import time
        import subprocess

        # Delay inicial para asegurar que el pipeline headless haya iniciado
        # Los bloques GNU Radio se ejecutan en paralelo, así que esperamos un poco
        time.sleep(2.0)

        log_debug('visualizer_wr', "Esperando artefactos del pipeline...")

        # Esperar a que el pipeline termine completamente
        # El pipeline crea un archivo .done cuando termina
        # IMPORTANTE: Usar ruta absoluta para evitar problemas con working directory
        done_file = os.path.join(current_dir, 'current_run', '.done')

        timeout = 150.0  # 150 segundos timeout
        start_time = time.time()
        check_interval = 1.0

        log_debug('visualizer_wr', f"Esperando archivo .done en {done_file}")

        while (time.time() - start_time) < timeout:
            # Verificar si existe el archivo .done (indica pipeline completo)
            if os.path.exists(done_file):
                elapsed = time.time() - start_time
                log_debug('visualizer_wr', f"Pipeline completado después de {elapsed:.1f}s")
                time.sleep(0.5)  # Pequeña espera adicional por seguridad
                break

            time.sleep(check_interval)

        # Verificar si el pipeline terminó
        if not os.path.exists(done_file):
            log_warning('visualizer_wr', "====================================================")
            log_warning('visualizer_wr', "TIMEOUT: Pipeline did not complete in expected time")
            log_warning('visualizer_wr', "Dashboards will NOT be launched")
            log_warning('visualizer_wr', "Verify that headless pipeline executed correctly")
            log_warning('visualizer_wr', "Check logs in: pipeline_logs/")
            log_warning('visualizer_wr', "====================================================")
            return

        # Artefactos encontrados - lanzar dashboards habilitados
        log_debug('visualizer_wr', "Lanzando dashboards...")

        enabled_dashboards = []
        if self.enable_dashboard_a: enabled_dashboards.append('A')
        if self.enable_dashboard_b: enabled_dashboards.append('B')
        if self.enable_dashboard_c: enabled_dashboards.append('C')
        if self.enable_dashboard_d: enabled_dashboards.append('D')
        if self.enable_dashboard_e: enabled_dashboards.append('E')

        if not enabled_dashboards:
            log_debug('visualizer_wr', "No hay dashboards habilitados - simulación completa")
            return

        log_debug('visualizer_wr', f"Lanzando dashboards: {enabled_dashboards}")

        # Lanzar cada dashboard
        for dashboard in enabled_dashboards:
            try:
                self._launch_single_dashboard(dashboard)
                time.sleep(0.5)  # Pequeño delay entre lanzamientos
            except Exception as e:
                log_warning('visualizer_wr', f"Error launching Dashboard {dashboard}: {e}")

        log_debug('visualizer_wr', "Todos los dashboards lanzados")

    def _launch_single_dashboard(self, dashboard_type):
        """Lanzar un dashboard específico"""
        import subprocess

        if dashboard_type == 'A':
            # Dashboard A: Temporal analysis
            subprocess.Popen([
                'python3', 'oam_visualizer.py',
                '--mode', 'simple_dynamic',
                '--run', 'current',
                '--gui', 'qt',
                '--step', str(self.dashboard_step_delay)
            ], cwd=current_dir)
            log_debug('visualizer_wr', " Dashboard A (Temporal Analysis) lanzado")

        elif dashboard_type == 'B':
            # Dashboard B: QA metrics
            subprocess.Popen([
                'python3', 'oam_visualizer.py',
                '--mode', 'qa_dynamic',
                '--run', 'current',
                '--gui', 'qt',
                '--step', str(self.dashboard_step_delay),
                '--modalmix'
            ], cwd=current_dir)
            log_debug('visualizer_wr', " Dashboard B (QA Metrics) lanzado")

        elif dashboard_type == 'C':
            # Dashboard C: Detailed snapshot
            subprocess.Popen([
                'python3', 'oam_visualizer.py',
                '--mode', 'snapshot_offline',
                '--run', 'current',
                '--symbol', '13',
                '--gui', 'qt'
            ], cwd=current_dir)
            log_debug('visualizer_wr', " Dashboard C (Snapshot) lanzado")

        elif dashboard_type == 'D':
            # Dashboard D: Modal stream
            subprocess.Popen([
                'python3', 'oam_visualizer.py',
                '--mode', 'modal_stream',
                '--run', 'current',
                '--gui', 'qt',
                '--step', str(self.dashboard_step_delay)
            ], cwd=current_dir)
            log_debug('visualizer_wr', " Dashboard D (Modal Stream) lanzado")

        elif dashboard_type == 'E':
            # Dashboard E: Metrics summary
            subprocess.Popen([
                'python3', 'oam_visualizer.py',
                '--mode', 'metrics_summary',
                '--run', 'current',
                '--gui', 'qt'
            ], cwd=current_dir)
            log_debug('visualizer_wr', " Dashboard E (Metrics Summary) lanzado")

    def _update_config(self):
        """Actualizar config_manager con parámetros actuales"""
        config_manager.update_visualizer_params(
            enable_dashboard_a=self.enable_dashboard_a,
            enable_dashboard_b=self.enable_dashboard_b,
            enable_dashboard_c=self.enable_dashboard_c,
            enable_dashboard_d=self.enable_dashboard_d,
            enable_dashboard_e=self.enable_dashboard_e,
            dashboard_step_delay=self.dashboard_step_delay
        )

        enabled_dashboards = []
        if self.enable_dashboard_a: enabled_dashboards.append('A')
        if self.enable_dashboard_b: enabled_dashboards.append('B')
        if self.enable_dashboard_c: enabled_dashboards.append('C')
        if self.enable_dashboard_d: enabled_dashboards.append('D')

        log_info('visualizer_wr', f" Visualizer configured:")
        log_debug('visualizer_wr', f"  - Dashboards enabled: {enabled_dashboards}")
        log_info('visualizer_wr', f"  - Time between symbols: {self.dashboard_step_delay}s")

        self._describe_enabled_dashboards(self.enable_dashboard_a, self.enable_dashboard_b,
                                        self.enable_dashboard_c, self.enable_dashboard_d)

        self._provide_usage_recommendations(enabled_dashboards, self.dashboard_step_delay)

    def _validate_visualizer_params(self, enable_a, enable_b, enable_c, enable_d, enable_e, step_delay):
        """
        Validar parámetros del visualizer antes de guardar
        """
        errors = []

        # Validar tipos boolean
        for param_name, param_value in [
            ('enable_dashboard_a', enable_a),
            ('enable_dashboard_b', enable_b),
            ('enable_dashboard_c', enable_c),
            ('enable_dashboard_d', enable_d),
            ('enable_dashboard_e', enable_e)
        ]:
            if not isinstance(param_value, bool):
                errors.append(f"{param_name} debe ser boolean (True/False)")

        # Validar dashboard_step_delay
        if not isinstance(step_delay, (int, float)):
            errors.append("dashboard_step_delay debe ser numérico")
        elif step_delay < 0.1 or step_delay > 30.0:
            errors.append("dashboard_step_delay debe estar entre 0.1s y 30s")

        # Validar que al menos un dashboard esté habilitado
        if not (enable_a or enable_b or enable_c or enable_d):
            errors.append("Al menos un dashboard debe estar habilitado")

        if errors:
            error_msg = "Errors in Visualizer parameters: " + "; ".join(errors)
            log_info('visualizer_wr', f" {error_msg}")
            raise ValueError(error_msg)

        log_debug('visualizer_wr', " Visualizer parameters validated correctly")

    def _describe_enabled_dashboards(self, enable_a, enable_b, enable_c, enable_d):
        """
        Describir dashboards habilitados para información del usuario
        """
        log_debug('visualizer_wr', "Dashboards enabled:")

        if enable_a:
            log_debug('visualizer_wr', "   Dashboard A: Dynamic temporal analysis")
            log_debug('visualizer_wr', "      - Symbol-by-symbol visualization")
            log_debug('visualizer_wr', "      - Automatic temporal progression")

        if enable_b:
            log_debug('visualizer_wr', "   Dashboard B: Dynamic QA metrics")
            log_debug('visualizer_wr', "      - SNR, BER, NCC tracking")
            log_debug('visualizer_wr', "      - Modal mixing analysis")

        if enable_c:
            log_debug('visualizer_wr', "   Dashboard C: Detailed snapshot")
            log_debug('visualizer_wr', "      - In-depth analysis of specific symbols")
            log_debug('visualizer_wr', "      - Modal analysis debugging")

        if enable_d:
            log_debug('visualizer_wr', "   Dashboard D: Dynamic modal stream")
            log_debug('visualizer_wr', "      - Real-time modal separation")
            log_debug('visualizer_wr', "      - Dominant sign detection")

    def _provide_usage_recommendations(self, enabled_dashboards, step_delay):
        """
        Proporcionar recomendaciones de uso basadas en configuración
        """
        num_enabled = len(enabled_dashboards)

        # Recomendación por número de dashboards
        if num_enabled >= 4:
            log_warning('visualizer_wr',
                       " Many dashboards enabled - may consume significant resources")
            log_debug('visualizer_wr', " For initial analysis, consider using only A and B")

        elif num_enabled == 1:
            log_debug('visualizer_wr', " Only one dashboard - for full analysis consider enabling more")

        # Recomendación por step delay
        if step_delay < 1.0:
            log_debug('visualizer_wr', " Fast step delay - ideal for dynamic analysis")
        elif step_delay > 5.0:
            log_debug('visualizer_wr', " Slow step delay - ideal for detailed analysis")

        # Recomendaciones específicas por combinación
        if 'A' in enabled_dashboards and 'D' in enabled_dashboards:
            log_debug('visualizer_wr', " Dashboards A+D: Excellent for complete temporal analysis")

        if 'B' in enabled_dashboards:
            log_debug('visualizer_wr', " Dashboard B enabled: Perfect for quality monitoring")

        if 'C' in enabled_dashboards:
            log_debug('visualizer_wr', " Dashboard C enabled: Useful for detailed debugging")

    def work(self, input_items, output_items):
        """
        Función work de GNU Radio - consume datos pero no hace nada
        Los dashboards ya fueron lanzados en start(), este método no procesa señales
        """
        # Este es un sink block, consume todos los inputs disponibles
        return len(input_items[0])

    def get_current_config(self):
        """
        Obtener configuración actual (para debugging)
        """
        current_config = config_manager.get_full_config()
        visualizer_config = {
            'enable_dashboard_a': current_config.get('enable_dashboard_a'),
            'enable_dashboard_b': current_config.get('enable_dashboard_b'),
            'enable_dashboard_c': current_config.get('enable_dashboard_c'),
            'enable_dashboard_d': current_config.get('enable_dashboard_d'),
            'dashboard_step_delay': current_config.get('dashboard_step_delay')
        }
        return visualizer_config

    def get_enabled_dashboard_info(self):
        """
        Obtener información detallada de dashboards habilitados
        """
        config = self.get_current_config()
        dashboard_info = {}

        if config.get('enable_dashboard_a'):
            dashboard_info['A'] = {
                'name': 'Temporal Analysis',
                'type': 'dynamic',
                'description': 'Symbol-by-symbol field visualization'
            }

        if config.get('enable_dashboard_b'):
            dashboard_info['B'] = {
                'name': 'QA Metrics',
                'type': 'dynamic',
                'description': 'SNR, BER, NCC correlation tracking'
            }

        if config.get('enable_dashboard_c'):
            dashboard_info['C'] = {
                'name': 'Detailed Snapshot',
                'type': 'static',
                'description': 'In-depth modal analysis'
            }

        if config.get('enable_dashboard_d'):
            dashboard_info['D'] = {
                'name': 'Modal Stream',
                'type': 'dynamic',
                'description': 'Real-time modal separation'
            }

        return dashboard_info

# ========================================================================
# HELPER FUNCTIONS PARA GNU RADIO
# ========================================================================

def make_oam_visualizer_wr(enable_dashboard_a=True, enable_dashboard_b=True,
                          enable_dashboard_c=True, enable_dashboard_d=True, dashboard_step_delay=3.0):
    """Factory function para GNU Radio block"""
    return oam_visualizer_wr(enable_dashboard_a, enable_dashboard_b,
                           enable_dashboard_c, enable_dashboard_d, dashboard_step_delay)

# ========================================================================
# CONFIGURACIONES PREESTABLECIDAS
# ========================================================================

class VisualizerPresets:
    """Configuraciones preestablecidas para diferentes escenarios"""

    # Análisis completo (todos los dashboards)
    FULL_ANALYSIS = {
        'enable_dashboard_a': True,
        'enable_dashboard_b': True,
        'enable_dashboard_c': True,
        'enable_dashboard_d': True,
        'dashboard_step_delay': 3.0
    }

    # Monitoreo básico (dashboards esenciales)
    BASIC_MONITORING = {
        'enable_dashboard_a': True,
        'enable_dashboard_b': True,
        'enable_dashboard_c': False,
        'enable_dashboard_d': False,
        'dashboard_step_delay': 2.0
    }

    # Solo métricas de calidad
    QA_ONLY = {
        'enable_dashboard_a': False,
        'enable_dashboard_b': True,
        'enable_dashboard_c': False,
        'enable_dashboard_d': False,
        'dashboard_step_delay': 1.0
    }

    # Dynamic temporal analysis
    TEMPORAL_ANALYSIS = {
        'enable_dashboard_a': True,
        'enable_dashboard_b': False,
        'enable_dashboard_c': False,
        'enable_dashboard_d': True,
        'dashboard_step_delay': 2.5
    }

    # Debugging detallado
    DEBUGGING = {
        'enable_dashboard_a': True,
        'enable_dashboard_b': True,
        'enable_dashboard_c': True,
        'enable_dashboard_d': False,
        'dashboard_step_delay': 5.0
    }

    # Modo sin visualización (solo simulación)
    NO_VISUALIZATION = {
        'enable_dashboard_a': False,
        'enable_dashboard_b': False,
        'enable_dashboard_c': False,
        'enable_dashboard_d': False,
        'dashboard_step_delay': 1.0
    }

# ========================================================================
# TEST Y DEMO
# ========================================================================

if __name__ == "__main__":
    print("=== TEST OAM VISUALIZER WRAPPER ===")

    # Test configuración estándar
    print("Test 1: Configuración estándar...")
    try:
        visualizer1 = oam_visualizer_wr()
        print(" Configuración estándar OK")
    except Exception as e:
        print(f" Error en configuración estándar: {e}")

    # Test configuración personalizada
    print("\nTest 2: Configuración personalizada...")
    try:
        visualizer2 = oam_visualizer_wr(
            enable_dashboard_a=True,
            enable_dashboard_b=False,
            enable_dashboard_c=True,
            enable_dashboard_d=False,
            dashboard_step_delay=4.5
        )
        print(" Configuración personalizada OK")
    except Exception as e:
        print(f" Error en configuración personalizada: {e}")

    # Test validación de errores
    print("\nTest 3: Validación de errores...")
    try:
        visualizer3 = oam_visualizer_wr(
            enable_dashboard_a=False,
            enable_dashboard_b=False,
            enable_dashboard_c=False,
            enable_dashboard_d=False  # Todos falsos - debe fallar
        )
        print(" Debería haber fallado con todos los dashboards deshabilitados")
    except ValueError:
        print(" Validación de errores funcionando correctamente")
    except Exception as e:
        print(f" Error inesperado: {e}")

    # Test configuraciones preestablecidas
    print("\nTest 4: Configuraciones preestablecidas...")
    presets_to_test = [
        ("BASIC_MONITORING", VisualizerPresets.BASIC_MONITORING),
        ("QA_ONLY", VisualizerPresets.QA_ONLY),
        ("TEMPORAL_ANALYSIS", VisualizerPresets.TEMPORAL_ANALYSIS),
        ("DEBUGGING", VisualizerPresets.DEBUGGING)
    ]

    for preset_name, preset_config in presets_to_test:
        try:
            visualizer = oam_visualizer_wr(**preset_config)
            info = visualizer.get_enabled_dashboard_info()
            enabled = list(info.keys())
            print(f" Preset {preset_name}: Dashboards {enabled}")
        except Exception as e:
            print(f" Error en preset {preset_name}: {e}")

    # Test información de dashboards
    print("\nTest 5: Información de dashboards...")
    try:
        visualizer5 = oam_visualizer_wr(**VisualizerPresets.FULL_ANALYSIS)
        dashboard_info = visualizer5.get_enabled_dashboard_info()
        print("Información completa de dashboards:")
        for dash_id, info in dashboard_info.items():
            print(f"  Dashboard {dash_id}: {info['name']} ({info['type']})")
    except Exception as e:
        print(f" Error obteniendo info de dashboards: {e}")

    print("\n=== TEST COMPLETADO ===")