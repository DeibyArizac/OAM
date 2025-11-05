#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WRAPPER GNU RADIO: OAM SOURCE BLOCK
==================================

Este es el ÚNICO bloque que ejecuta todo el pipeline OAM completo.
Los otros bloques (encoder, channel, decoder, visualizer) solo editan parámetros.

Arquitectura:
1. Recibe parámetros del GUI de GNU Radio
2. Los combina con parámetros de otros bloques via config_manager
3. Ejecuta todo el pipeline OAM (source -> encoder -> channel -> decoder)
4. Lanza dashboards según configuración
5. Los otros bloques son solo "editores visuales" de parámetros
"""

import threading
import time
import os
import numpy as np
from gnuradio import gr
import sys
import subprocess
from multiprocessing import Process

# Agregar directorio actual al path si no está
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Importar sistema OAM actual
from config_manager import config_manager, get_current_config
from oam_logging import log_info, log_warning, log_error, log_debug
from pipeline import pipeline

# Importar bloques del sistema actual
import oam_source
import oam_encoder
import oam_channel
import oam_decoder

class oam_source_wr(gr.sync_block):
    """
    Wrapper GNU Radio para OAM Source

    FUNCIÓN: Ejecuta TODO el pipeline OAM usando configuración compartida

    Este bloque:
    1. Actualiza sus parámetros en config_manager
    2. Obtiene configuración final de todos los bloques
    3. Ejecuta pipeline completo: source -> encoder -> channel -> decoder
    4. Lanza dashboards según configuración
    5. Es el ÚNICO bloque que realmente trabaja
    """

    def __init__(self, message_text="Hello OAM", symbol_rate=32000, data_rate_multiplier=1.0):
        """
        Inicializar OAM Source - El bloque que ejecuta todo

        Args:
            message_text: Mensaje a transmitir
            symbol_rate: Velocidad de transmisión en Hz
            data_rate_multiplier: Factor multiplicativo para data rate (QPSK, QAM, etc.)
        """
        gr.sync_block.__init__(
            self,
            name="oam_source_wr",
            in_sig=None,
            out_sig=[np.complex64]
        )

        log_debug('source_wr', "=== INICIALIZANDO OAM SOURCE WRAPPER ===")

        # Guardar parámetros iniciales
        self.message_text = message_text
        self.symbol_rate = symbol_rate
        self.data_rate_multiplier = data_rate_multiplier

        # 1. Actualizar parámetros source en config_manager
        config_manager.update_source_params(
            message_text=message_text,
            symbol_rate=symbol_rate
        )

        # 2. Flag para saber si ya ejecutamos el pipeline
        self.pipeline_executed = False

        # 3. Guardar PID del pipeline headless para poder matarlo en stop()
        self.pipeline_process = None

        log_debug('source_wr', " OAM Source inicializado - esperando start() para ejecutar pipeline")

    def set_message_text(self, message_text):
        """Setter para actualizar message_text en caliente"""
        self.message_text = message_text
        config_manager.update_source_params(message_text=message_text, symbol_rate=self.symbol_rate)
        log_debug('source_wr', f"Message text actualizado a: {message_text}")

    def set_symbol_rate(self, symbol_rate):
        """Setter para actualizar symbol_rate en caliente"""
        self.symbol_rate = symbol_rate
        config_manager.update_source_params(message_text=self.message_text, symbol_rate=symbol_rate)
        log_debug('source_wr', f"Symbol rate actualizado a: {symbol_rate}")

    def start(self):
        """
        Llamado por GNU Radio cuando se presiona Run

        NUEVA ARQUITECTURA:
        1. Genera run_id único
        2. Inicializa caché JSON con run_meta
        3. Escribe sus propios parámetros al caché
        4. Espera (barrera) a que todos los bloques escriban
        5. Fusiona toda la configuración
        6. Lanza pipeline headless con config unificada
        7. Los dashboards se lanzan después (vía Visualizer)
        """
        log_debug('source_wr', "=== START LLAMADO - SOURCE (ORQUESTADOR) ===")

        # 1. Generar run_id único
        from datetime import datetime
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_info('source_wr', f"Run ID generado: {run_id}")

        # 2. Inicializar caché JSON
        from gnuradio_cache import init_cache, write_block_params, wait_for_all_blocks, merge_all_params, save_merged_config

        if not init_cache(run_id):
            log_error('source_wr', "No se pudo inicializar caché")
            return False

        # 3. Escribir nuestros propios parámetros
        source_params = {
            'message_text': self.message_text,
            'symbol_rate': self.symbol_rate
        }

        write_block_params('oam_source', source_params, run_id)
        log_debug('source_wr', f"Parameters written to cache")

        # 4. BARRERA: Esperar a que todos los bloques escriban
        log_debug('source_wr', "Esperando a que todos los bloques escriban sus parámetros...")

        if not wait_for_all_blocks(run_id, timeout=5.0):
            log_warning('source_wr', "Timeout esperando bloques - continuando de todos modos")

        # 5. Fusionar configuración de todos los bloques
        log_debug('source_wr', "Fusionando configuración de todos los bloques...")
        unified_config = merge_all_params()

        # Guardar configuración unificada
        config_file = "current_run/config_from_grc.json"
        save_merged_config(unified_config, config_file)

        log_debug('source_wr', f"Configuración unificada guardada: {len(unified_config)} parámetros")

        # 6. Limpiar artefactos anteriores para forzar regeneración
        # IMPORTANTE: Usar ruta absoluta para evitar problemas con working directory
        done_file = os.path.join(current_dir, 'current_run', '.done')
        if os.path.exists(done_file):
            os.remove(done_file)
            log_debug('source_wr', f"Eliminado archivo .done anterior: {done_file}")

        # Eliminar NPZ viejos para que dashboards no carguen datos obsoletos
        import glob
        npz_pattern = os.path.join(current_dir, 'current_run', 'fields_*.npz')
        npz_files = glob.glob(npz_pattern)
        for npz_file in npz_files:
            try:
                os.remove(npz_file)
                log_debug('source_wr', f"Eliminado NPZ obsoleto: {npz_file}")
            except:
                pass

        # 7. Lanzar pipeline headless
        log_info('source_wr', "Lanzando pipeline OAM en modo headless...")
        self.launch_headless_pipeline(config_file)

        log_debug('source_wr', "=== SOURCE COMPLETADO ===")
        return True

    def launch_headless_pipeline(self, config_file):
        """
        Lanzar oam_complete_system.py en modo headless como subproceso

        Args:
            config_file: Ruta al JSON con configuración unificada
        """
        try:
            import subprocess
            import sys

            # Crear directorio de logs
            logs_dir = "pipeline_logs"
            os.makedirs(logs_dir, exist_ok=True)

            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stdout_log = os.path.join(logs_dir, f"pipeline_{timestamp}_stdout.log")
            stderr_log = os.path.join(logs_dir, f"pipeline_{timestamp}_stderr.log")

            # Comando para ejecutar headless
            script_path = os.path.join(current_dir, 'oam_complete_system.py')
            cmd = [
                sys.executable,  # python3
                script_path,
                '--headless',
                '--config', config_file,
                '--save-dir', 'current_run'
            ]

            log_debug('source_wr', f"Ejecutando: {' '.join(cmd)}")

            # Ejecutar como subproceso
            with open(stdout_log, 'w') as out_f, open(stderr_log, 'w') as err_f:
                process = subprocess.Popen(
                    cmd,
                    cwd=current_dir,
                    stdout=out_f,
                    stderr=err_f,
                    env=dict(os.environ, PYTHONPATH=current_dir)
                )

                # Guardar proceso para poder matarlo en stop()
                self.pipeline_process = process

                log_debug('source_wr', f"Pipeline headless lanzado (PID: {process.pid})")
                log_debug('source_wr', f"Logs: {stdout_log}, {stderr_log}")

                # Esperar a que termine (con timeout)
                try:
                    returncode = process.wait(timeout=120)

                    if returncode == 0:
                        log_debug('source_wr', "Pipeline headless completado exitosamente")
                    else:
                        log_error('source_wr', f"Pipeline headless falló (código {returncode})")

                        # Mostrar últimas líneas de stderr si falló
                        try:
                            with open(stderr_log, 'r') as f:
                                stderr_content = f.read()
                                if stderr_content:
                                    log_error('source_wr', f"Últimas líneas de error:")
                                    for line in stderr_content.strip().split('\n')[-10:]:
                                        log_error('source_wr', f"  {line}")
                        except:
                            pass

                except subprocess.TimeoutExpired:
                    process.kill()
                    log_error('source_wr', "Pipeline headless timeout (120s) - proceso terminado")

        except Exception as e:
            log_error('source_wr', f"Error lanzando pipeline headless: {e}")
            import traceback
            traceback.print_exc()


    def launch_dashboards(self):
        """
        Lanzar dashboards según configuración
        Solo lanza los dashboards que están habilitados
        """
        log_debug('source_wr', " LANZANDO DASHBOARDS")

        enabled_dashboards = []
        step_delay = self.config.get('dashboard_step_delay', 3.0)

        # Verificar qué dashboards están habilitados
        if self.config.get('enable_dashboard_a', False):
            enabled_dashboards.append('A')
        if self.config.get('enable_dashboard_b', False):
            enabled_dashboards.append('B')
        if self.config.get('enable_dashboard_c', False):
            enabled_dashboards.append('C')
        if self.config.get('enable_dashboard_d', False):
            enabled_dashboards.append('D')

        if not enabled_dashboards:
            log_debug('source_wr', "No hay dashboards habilitados - simulación completa")
            return

        log_debug('source_wr', f"Lanzando dashboards: {enabled_dashboards} con delay {step_delay}s")

        # Lanzar cada dashboard en proceso separado
        for dashboard in enabled_dashboards:
            try:
                self._launch_single_dashboard(dashboard, step_delay)
            except Exception as e:
                log_warning('source_wr', f"Error lanzando Dashboard {dashboard}: {e}")

    def _launch_single_dashboard(self, dashboard_type, step_delay):
        """Lanzar un dashboard específico en proceso separado"""

        def run_dashboard():
            try:
                if dashboard_type == 'A':
                    self._launch_dashboard_a(step_delay)
                elif dashboard_type == 'B':
                    self._launch_dashboard_b(step_delay)
                elif dashboard_type == 'C':
                    self._launch_dashboard_c()
                elif dashboard_type == 'D':
                    self._launch_dashboard_d(step_delay)
            except Exception as e:
                log_error('source_wr', f"Error en Dashboard {dashboard_type}: {e}")

        # Ejecutar en thread separado para no bloquear GNU Radio
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()

        log_debug('source_wr', f" Dashboard .* lanzado en thread separado")

    def _launch_dashboard_a(self, step_delay):
        """Lanzar Dashboard A: Temporal analysis"""
        subprocess.Popen([
            'python3', 'oam_visualizer.py',
            '--mode', 'simple_dynamic',
            '--run', 'current',
            '--gui', 'qt',
            '--step', str(step_delay)
        ], cwd=current_dir)

    def _launch_dashboard_b(self, step_delay):
        """Lanzar Dashboard B: QA metrics"""
        subprocess.Popen([
            'python3', 'oam_visualizer.py',
            '--mode', 'qa_dynamic',
            '--run', 'current',
            '--gui', 'qt',
            '--step', str(step_delay),
            '--modalmix'
        ], cwd=current_dir)

    def _launch_dashboard_c(self):
        """Lanzar Dashboard C: Detailed snapshot"""
        subprocess.Popen([
            'python3', 'oam_visualizer.py',
            '--mode', 'snapshot_offline',
            '--run', 'current',
            '--symbol', '13',
            '--gui', 'qt'
        ], cwd=current_dir)

    def _launch_dashboard_d(self, step_delay):
        """Lanzar Dashboard D: Modal stream"""
        subprocess.Popen([
            'python3', 'oam_visualizer.py',
            '--mode', 'modal_stream',
            '--run', 'current',
            '--gui', 'qt',
            '--step', str(step_delay)
        ], cwd=current_dir)

    def stop(self):
        """
        Llamado cuando GNU Radio presiona Stop - matar todos los procesos
        """
        log_debug('source_wr', "=== STOP LLAMADO - CERRANDO PROCESOS ===")

        # Matar pipeline headless si está corriendo
        if self.pipeline_process and self.pipeline_process.poll() is None:
            try:
                log_debug('source_wr', f"Matando pipeline headless (PID: {self.pipeline_process.pid})")
                self.pipeline_process.terminate()
                time.sleep(0.5)
                if self.pipeline_process.poll() is None:
                    self.pipeline_process.kill()
                log_debug('source_wr', "Pipeline headless terminado")
            except Exception as e:
                log_warning('source_wr', f"Error matando pipeline: {e}")

        # Matar todos los dashboards (oam_visualizer.py)
        try:
            import subprocess
            log_info('source_wr', "Matando dashboards (oam_visualizer.py)")
            subprocess.run(['pkill', '-f', 'oam_visualizer.py'], check=False)
            log_debug('source_wr', "Dashboards terminados")
        except Exception as e:
            log_warning('source_wr', f"Error matando dashboards: {e}")

        log_debug('source_wr', "=== STOP COMPLETADO ===")
        return True

    def work(self, input_items, output_items):
        """
        Función work de GNU Radio - no hace nada, solo retorna
        El pipeline ya fue ejecutado en start(), este método no procesa señales
        """
        # Este es un source block, no tiene inputs
        # Retornar -1 indica EOF (End of File) - no hay más datos
        return -1


# ========================================================================
# HELPER FUNCTIONS PARA GNU RADIO
# ========================================================================

def make_oam_source_wr(message_text="Hello OAM", symbol_rate=32000, data_rate_multiplier=1.0):
    """Factory function para GNU Radio block"""
    return oam_source_wr(message_text, symbol_rate, data_rate_multiplier)

# ========================================================================
# TEST Y DEMO
# ========================================================================

if __name__ == "__main__":
    print("=== TEST OAM SOURCE WRAPPER ===")

    # Test básico
    print("Creando OAM Source con configuración de prueba...")

    try:
        source = oam_source_wr(
            message_text="TEST_MESSAGE",
            symbol_rate=16000
        )
        print(" OAM Source Wrapper ejecutado correctamente")

    except Exception as e:
        print(f" Error en test: {e}")
        import traceback
        traceback.print_exc()

    print("=== TEST COMPLETADO ===")