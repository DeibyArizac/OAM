
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sistema completo OAM con interfaz gráfica para transmisión y recepción.
Integra encoder, canal, decoder y visualización en pipeline unificado.

Este módulo implementa el sistema completo de comunicación óptica basado en
Momento Angular Orbital (OAM) de la luz, desarrollado como parte del trabajo
de grado en Ingeniería Electrónica, Eléctrica y de Telecomunicaciones.
"""

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import gr
import sys
import signal
import subprocess
import os
import tempfile
from multiprocessing import Process


# Sistema de logging unificado
from oam_logging import log_info, log_warning, log_error, log_debug
from pipeline import pipeline
# Configuración centralizada del sistema
from oam_system_config import OAMConfig, get_system_config

# Funciones auxiliares for Dashboard D integration
def _pickle_default_path():
    return os.path.join(tempfile.gettempdir(), "oam", "pipeline_result.pkl")

def _last_completed_run(runs_dir):
    if not (os.path.exists(runs_dir) and os.listdir(runs_dir)):
        return None
    for run in sorted(os.listdir(runs_dir), reverse=True):
        path = os.path.join(runs_dir, run)
        if os.path.exists(os.path.join(path, '.done')):
            # opción: verifica que exista al menos 1 source NPZ con tamaño razonable
            for src in ["at_decoder","after_channel","before_channel","after_encoder"]:
                p = os.path.join(path, f"fields_{src}.npz")
                if os.path.exists(p) and os.path.getsize(p) > 1024:
                    return run
    return None

def cleanup_corrupted_files():
    """Limpiar archivos NPZ corruptos para evitar crashes"""
    import numpy as np

    current_run_dir = "current_run"
    if not os.path.exists(current_run_dir):
        return

    print("[MAIN] Verificando archivos NPZ...")
    for filename in os.listdir(current_run_dir):
        if filename.endswith('.npz'):
            filepath = os.path.join(current_run_dir, filename)
            try:
                # Intentar cargar el archivo para verificar integridad
                data = np.load(filepath)
                data.close()
            except Exception as e:
                print(f"[MAIN] Archivo corrupto eliminado: {filename} - {e}")
                try:
                    os.remove(filepath)
                except:
                    pass

def check_simulation_cache():
    """
    Verificar si existe un cache válido y debe ejecutarse la simulación.

    Returns:
        tuple: (run_simulation, reason) donde run_simulation es bool y reason es string
    """
    try:
        # Cargar configuración desde GUI antes de validar el cache
        config_file = os.path.join(os.path.dirname(__file__), "current_run", "config_from_grc.json")
        if os.path.exists(config_file):
            import json
            import oam_system_config
            with open(config_file, 'r') as f:
                config_overrides = json.load(f)
            # Actualizar SYSTEM_CONFIG con valores de GUI
            oam_system_config.SYSTEM_CONFIG.update(config_overrides)
            log_info('cache', f"Config GUI aplicada: {config_overrides.get('num_oam_modes', '?')} modos")

        # Obtener configuración actual del sistema desde configuración centralizada
        current_config = get_system_config()
        current_config['oam_channels'] = OAMConfig.get_oam_channels()  # Añadir modos generados

        # Verificar si el cache es válido
        if pipeline.is_cache_valid(current_config):
            # Verificar que existen archivos de datos
            current_run_dir = os.path.join(os.path.dirname(__file__), "current_run")
            required_files = [
                "fields_before_channel.npz",
                "fields_after_channel.npz",
                "env.json",
                "config_hash.txt"
            ]

            missing_files = []
            for filename in required_files:
                file_path = os.path.join(current_run_dir, filename)
                if not os.path.exists(file_path):
                    missing_files.append(filename)
                elif os.path.getsize(file_path) == 0:
                    missing_files.append(f"{filename} (vacío)")

            if missing_files:
                return True, f"Cache válido pero faltan archivos: {missing_files}"
            else:
                return False, "Cache válido - usando datos existentes"
        else:
            return True, "Parámetros de simulación cambiaron - regenerar"

    except Exception as e:
        log_warning('cache', f"Error verificando cache: {e}")
        return True, f"Error en cache - regenerar por seguridad: {e}"

def launch_dashboard_d(pickle_path=None, step=0.4, style="dark"):
    try:
        from oam_visualizer import render_dashboard_modal_stream
        render_dashboard_modal_stream(
            pickle_path=pickle_path or _pickle_default_path(),
            style=style,
            gui_mode='qt',
            step_delay=step,
            show_sign_detection=True,
            low_mode_thresh=0.01
        )
    except Exception as e:
        print(f"[MAIN] Dashboard D error: {e}")

# Modulos locales del sistema OAM
import oam_channel
import oam_decoder
import oam_encoder
import oam_source
from gnuradio import blocks



class oam_complete_system(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Sistema OAM Completo", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Sistema OAM Completo")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            log_warning('main', f"Qt GUI: No se pudo establecer icono: {str(exc)}")
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "oam_complete_system")

        # Configurar tamaño de ventana MUCHO más pequeño
        self.resize(400, 300)  # Ventana pequeña (ancho x alto)
        self.setMinimumSize(300, 200)  # Tamaño mínimo
        self.setMaximumSize(500, 400)  # Tamaño máximo
        self.move(100, 100)  # Posición en pantalla (x, y)

        try:
            # FORZAR tamaño pequeño - NO restaurar geometría guardada
            self.resize(400, 300)
            log_info('main', "Qt GUI: Forzando ventana pequeña (400x300)")
        except BaseException as exc:
            log_error('main', f"Qt GUI: No se pudo establecer geometría: {str(exc)}")
            # En caso de error, asegurar tamaño pequeño
            self.resize(400, 300)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 32000

        ##################################################
        # Blocks
        ##################################################

        # self.oam_visualizer_0 = oam_visualizer(visualization_type='intensity_phase', grid_size=256, save_plots=True, show_plots=True)
        # Payload de prueba conocido para test de sanidad: 0xAA, 0x55, 0xCC, 0x33
        # Usando caracteres que produzcan estos bytes: ª (0xAA), U (0x55), Ì (0xCC), 3 (0x33)

        # Obtener configuración automáticamente del sistema centralizado
        config = get_system_config()
        system_channels = OAMConfig.get_oam_channels()
        self.modes_per_symbol = OAMConfig.get_modes_per_symbol()

        self.oam_source_0 = oam_source.oam_source(
            message=config['message_text'][:10],  # Usar primeros 10 chars del mensaje configurado
            packet_size=32
        )
        self.oam_encoder_0 = oam_encoder.oam_encoder(
            grid_size=config['grid_size'],
            wavelength=config['wavelength'],
            tx_aperture_size=config['tx_aperture_size'],
            tx_beam_waist=OAMConfig.get_tx_beam_waist(),
            oam_channels=system_channels
        )
        self.oam_decoder_0 = oam_decoder.oam_decoder(
            grid_size=config['grid_size'],
            wavelength=config['wavelength'],
            propagation_distance=config['propagation_distance'],
            tx_aperture_size=config['tx_aperture_size'],
            tx_beam_waist=OAMConfig.get_tx_beam_waist(),
            oam_channels=system_channels,
            aperture_mode="none"
        )
        self.oam_channel_0 = oam_channel.oam_channel(
            grid_size=config['grid_size'],
            wavelength=config['wavelength'],
            propagation_distance=config['propagation_distance'],  # Usar distancia de configuración centralizada
            tx_aperture_size=config['tx_aperture_size'],
            tx_beam_waist=OAMConfig.get_tx_beam_waist(),
            oam_channels=system_channels,
            cn2=config['cn2'],
            snr_db=config['snr_target']  # Usar SNR de configuración centralizada
        )
        # Crear directorio para resultados si no existe
        output_dir = os.path.join(os.path.dirname(__file__), 'current_run')
        os.makedirs(output_dir, exist_ok=True)
        pipeline_result_path = os.path.join(output_dir, 'pipeline_result.txt')

        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_char*1, pipeline_result_path, False)
        self.blocks_file_sink_0.set_unbuffered(True)

        # VISUALIZER ELIMINADO - Sistema sin gráficas

        # self.oam_visualizer_0 = oam_visualizer.OAMUnifiedVisualizer(...)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.oam_channel_0, 0), (self.oam_decoder_0, 0))
        self.connect((self.oam_decoder_0, 0), (self.blocks_file_sink_0, 0))
        # self.connect((self.oam_decoder_0, 0), (self.oam_visualizer_0, 0))  # VISUALIZER ELIMINADO
        self.connect((self.oam_encoder_0, 0), (self.oam_channel_0, 0))
        self.connect((self.oam_source_0, 0), (self.oam_encoder_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "oam_complete_system")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate




def main(top_block_cls=oam_complete_system):

    # LIMPIAR ARCHIVOS CORRUPTOS Y RESET PIPELINE AL INICIAR
    cleanup_corrupted_files()
    from pipeline import pipeline
    pipeline.reset()
    log_info('main', "Pipeline reset - sistema listo para nueva ejecución")

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()
    tb.raise_()  # Forzar ventana al frente
    tb.activateWindow()  # Activar ventana

    def sig_handler(*_):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # Timer para mantener la aplicación responsive
    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    # Timer para terminar automáticamente después del procesamiento
    def shutdown_gnu_radio():
        tb.stop()
        tb.wait()
        qapp.quit()
        log_info('main', "GNU Radio finished - launching dashboards...")

    shutdown_timer = Qt.QTimer()
    shutdown_timer.setSingleShot(True)
    shutdown_timer.timeout.connect(shutdown_gnu_radio)
    shutdown_timer.start(10000)  # 10 segundos - luego dashboard offline

    log_info('main', "Sistema terminará en 10 segundos y activará dashboard offline")
    qapp.exec_()

    # Orquestación A→B fuera del proceso Qt
    log_info('main', "Launching Dashboard sequence A → B...")
    

    import subprocess
    import os
    import time
    from oam_visualizer import get_portable_pickle_path

    try:
        pickle_path = get_portable_pickle_path()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        visualizer_path = os.path.join(script_dir, "oam_visualizer.py")

        # Esperar a que el pipeline complete el guardado de datos verificando archivos en disco
        log_info('main', "Esperando a que pipeline guarde todos los datos...")

        # Verificación de existencia de archivos críticos con espera adaptativa
        current_run_dir = os.path.join(script_dir, "current_run")
        if os.path.exists(current_run_dir):
            required_files = [
                "fields_before_channel.npz",
                "fields_after_channel.npz",
                "symbol_metadata.json"
            ]

            log_info('main', f"Verificando archivos en: current_run/")

            # Esperar hasta 30 segundos a que todos los archivos existan
            max_wait_time = 30.0
            check_interval = 0.1
            elapsed_time = 0.0

            while elapsed_time < max_wait_time:
                missing_files = []
                for filename in required_files:
                    file_path = os.path.join(current_run_dir, filename)
                    if not os.path.exists(file_path):
                        missing_files.append(filename)

                if not missing_files:
                    log_info('main', f" Todos los archivos críticos verificados en {elapsed_time:.1f}s: {required_files}")
                    break

                time.sleep(check_interval)
                elapsed_time += check_interval

            if missing_files:
                log_warning('main', f" Archivos aún faltantes después de {max_wait_time}s: {missing_files}")

        log_info('main', f"Datos del run actual listos para visualización")

        # Usar pickle del current_run si existe, sino usar temporal
        current_pickle_path = os.path.join(current_run_dir, "pipeline_result.pkl")
        if os.path.exists(current_pickle_path) and os.path.getsize(current_pickle_path) > 1000:
            pickle_path = current_pickle_path
            log_info('main', f"Usando pickle del run actual: {current_pickle_path}")
        else:
            log_info('main', f"Run actual sin pickle, usando temporal: {pickle_path}")

        # Verificar que el archivo final existe
        if os.path.exists(pickle_path):
            log_info('main', f"Pickle confirmado: {pickle_path} ({os.path.getsize(pickle_path)} bytes)")
        else:
            log_warning('main', f"Pickle no disponible: {pickle_path}")

        # Preparar comandos para dashboards A, B, C usando current_run
        if os.path.exists(current_run_dir):
            log_info('main', f"Usando dashboards offline con current_run")
            # Comandos offline que usan current_run
            cmd_a = [sys.executable, visualizer_path, "--mode", "simple_dynamic", "--run", "current", "--gui", "qt", "--step", "3.0"]
            cmd_b = [sys.executable, visualizer_path, "--mode", "qa_dynamic", "--run", "current", "--gui", "qt", "--step", "3.0", "--modalmix"]
            cmd_c = [sys.executable, visualizer_path, "--mode", "snapshot_offline", "--run", "current", "--symbol", "13", "--gui", "qt"]

            # Dashboard D (modal) usando current_run también
            cmd_d = [sys.executable, visualizer_path, "--mode", "modal_stream", "--run", "current", "--gui", "qt", "--step", "3.0", "--show-sign-detection"]

            # Dashboard E (metrics summary) usando current_run
            cmd_e = [sys.executable, visualizer_path, "--mode", "metrics_summary", "--run", "current", "--gui", "qt"]
        else:
            log_warning('main', "No hay current_run disponible, usando pickle temporal")
            cmd_a = [sys.executable, visualizer_path, "--mode", "simple_dynamic", "--gui", "qt",
                     "--pickle", pickle_path, "--step", "0.6"]
            cmd_b = [sys.executable, visualizer_path, "--mode", "qa_dynamic", "--gui", "qt",
                     "--pickle", pickle_path, "--step", "0.8", "--modalmix"]
            cmd_c = [sys.executable, visualizer_path, "--mode", "snapshot_offline", "--symbol", "13", "--gui", "qt",
                     "--pickle", pickle_path]
            cmd_d = [sys.executable, visualizer_path, "--mode", "modal_stream", "--gui", "qt",
                     "--pickle", pickle_path, "--step", "0.4"]

            # Dashboard E (metrics summary) usando pickle
            cmd_e = [sys.executable, visualizer_path, "--mode", "metrics_summary", "--gui", "qt",
                     "--pickle", pickle_path]

        # Verificar qué dashboards están habilitados en la configuración
        from oam_system_config import SYSTEM_CONFIG
        enable_a = SYSTEM_CONFIG.get('enable_dashboard_a', True)
        enable_b = SYSTEM_CONFIG.get('enable_dashboard_b', True)
        enable_c = SYSTEM_CONFIG.get('enable_dashboard_c', True)
        enable_d = SYSTEM_CONFIG.get('enable_dashboard_d', True)
        enable_e = SYSTEM_CONFIG.get('enable_dashboard_e', True)

        enabled_dashboards = []
        if enable_a: enabled_dashboards.append('A')
        if enable_b: enabled_dashboards.append('B')
        if enable_c: enabled_dashboards.append('C')
        if enable_d and cmd_d is not None: enabled_dashboards.append('D')
        if enable_e: enabled_dashboards.append('E')

        # Lanzar dashboards según configuración
        log_info('main', f"Lanzando {len(enabled_dashboards)} dashboards enabled: {', '.join(enabled_dashboards)}")
        if enable_a: log_info('main', "  Dashboard A: temporal analysis (continuous loop)")
        if enable_b: log_info('main', "  Dashboard B: QA metrics (continuous loop)")
        if enable_c: log_info('main', "  Dashboard C: snapshot symbol 10 (single shot)")
        if enable_d and cmd_d is not None: log_info('main', "  Dashboard D: modal stream - datos reales con ruido y turbulencia (continuous loop)")
        if enable_e: log_info('main', "  Dashboard E: global metrics de simulación (static)")

        # Lanzar procesos habilitados
        proc_a = subprocess.Popen(cmd_a) if enable_a else None
        proc_b = subprocess.Popen(cmd_b) if enable_b else None
        proc_c = subprocess.Popen(cmd_c) if enable_c else None
        proc_d = subprocess.Popen(cmd_d) if (enable_d and cmd_d is not None) else None
        proc_e = subprocess.Popen(cmd_e) if enable_e else None

        log_info('main', "Todos los dashboards iniciados - aparecerán cuando estén listos")
        log_info('main', "El sistema terminará cuando todas las ventanas se hayan cerrado")

        # CONTROL MEJORADO: Esperar a que todos los dashboards terminen
        try:
            log_info('main', "Monitoreando todos los dashboards - el sistema terminará cuando todos se cierren...")

            # Monitorear continuamente todos los procesos
            import time
            while True:
                # Verificar cuáles procesos siguen activos
                active_processes = []
                if proc_a is not None and proc_a.poll() is None:
                    active_processes.append("A")
                if proc_b is not None and proc_b.poll() is None:
                    active_processes.append("B")
                if proc_c is not None and proc_c.poll() is None:
                    active_processes.append("C")
                if proc_d is not None and proc_d.poll() is None:
                    active_processes.append("D")
                if proc_e is not None and proc_e.poll() is None:
                    active_processes.append("E")

                # Si no hay procesos activos, terminar
                if not active_processes:
                    log_info('main', "Todos los dashboards cerrados - ejecutando limpieza automática")
                    # Limpieza automática de procesos colgados
                    try:
                        import subprocess
                        subprocess.run(['pkill', '-f', 'oam_visualizer.py'], check=False)
                        log_info('main', "Limpieza automática completada - procesos OAM cerrados")
                    except Exception as e:
                        log_warning('main', f"Error en limpieza automática: {e}")

                    log_info('main', "Sistema terminando limpiamente")
                    break

                # Log cada 5 segundos para mostrar estado
                log_info('main', f"Dashboards activos: {', '.join(active_processes)}")
                time.sleep(5)

            log_info('main', "Todos los dashboards terminaron naturalmente")

        except KeyboardInterrupt:
            log_info('main', "Interrupción detectada - cerrando todos los procesos")
            try:
                if proc_a is not None: proc_a.terminate()
                if proc_b is not None: proc_b.terminate()
                if proc_c is not None: proc_c.terminate()
                if proc_d is not None: proc_d.terminate()
                if proc_e is not None: proc_e.terminate()
                time.sleep(1)
                if proc_a is not None: proc_a.kill()
                if proc_b is not None: proc_b.kill()
                if proc_c is not None: proc_c.kill()
                if proc_d is not None: proc_d.kill()
                if proc_e is not None: proc_e.kill()
            except:
                pass

        log_info('main', "Todos los procesos cerrados automáticamente")
    except Exception as e:
        log_error('main', f"Error en secuencia A→B: {e}")
        log_info('main', "Para lanzar manualmente:")
        log_info('main', f"       cd {script_dir}")
        log_info('main', "       python3 oam_visualizer.py --mode simple_offline --run <run_name> --gui qt --step 0.6")
        log_info('main', "       python3 oam_visualizer.py --mode qa_offline --run <run_name> --gui qt")
        log_info('main', "       python3 oam_visualizer.py --mode snapshot_offline --run <run_name> --symbol 10 --gui qt")
        log_info('main', "       python3 oam_visualizer.py --mode modal_stream --run <run_name> --gui qt           # Datos reales con ruido")

def launch_cached_dashboards():
    """
    Lanzar dashboards directamente usando datos del cache sin ejecutar simulación.
    """
    try:
        # Cargar datos del cache
        if not pipeline.load_run("current"):
            log_error('main', "Error loading cache data")
            return False

        log_info('main', "Cache data loaded - launching dashboards...")

        # Preparar comandos para dashboards usando current_run
        script_dir = os.path.dirname(os.path.abspath(__file__))
        visualizer_path = os.path.join(script_dir, "oam_visualizer.py")

        cmd_a = [sys.executable, visualizer_path, "--mode", "simple_dynamic", "--run", "current", "--gui", "qt", "--step", "3.0"]
        cmd_b = [sys.executable, visualizer_path, "--mode", "qa_dynamic", "--run", "current", "--gui", "qt", "--step", "3.0", "--modalmix"]
        cmd_c = [sys.executable, visualizer_path, "--mode", "snapshot_offline", "--run", "current", "--symbol", "13", "--gui", "qt"]

        # Verificar qué dashboards están habilitados
        from oam_system_config import SYSTEM_CONFIG
        enable_a = SYSTEM_CONFIG.get('enable_dashboard_a', True)
        enable_b = SYSTEM_CONFIG.get('enable_dashboard_b', True)
        enable_c = SYSTEM_CONFIG.get('enable_dashboard_c', True)

        # Lanzar dashboards enabled en secuencia
        import subprocess
        import time

        procs = []
        enabled_names = []

        if enable_a:
            log_info('main', "Launching Dashboard A (cache)...")
            proc_a = subprocess.Popen(cmd_a)
            procs.append(proc_a)
            enabled_names.append('A')
            time.sleep(2)

        if enable_b:
            log_info('main', "Launching Dashboard B (cache)...")
            proc_b = subprocess.Popen(cmd_b)
            procs.append(proc_b)
            enabled_names.append('B')
            time.sleep(2)

        if enable_c:
            log_info('main', "Launching Dashboard C (cache)...")
            proc_c = subprocess.Popen(cmd_c)
            procs.append(proc_c)
            enabled_names.append('C')

        log_info('main', f"Lanzados {len(procs)} dashboards desde cache: {', '.join(enabled_names)}")

        # Esperar a que terminen todos los procesos lanzados
        for proc in procs:
            proc.wait()

        # Limpieza automática después de que terminen los dashboards del cache
        log_info('main', "Dashboards del cache terminados - ejecutando limpieza automática")
        try:
            subprocess.run(['pkill', '-f', 'oam_visualizer.py'], check=False)
            log_info('main', "Limpieza automática completada - procesos OAM cerrados")
        except Exception as e:
            log_warning('main', f"Error en limpieza automática: {e}")

        return True

    except Exception as e:
        log_error('main', f"Error en dashboards del cache: {e}")
        return False

def run_headless_pipeline(config_file=None, save_dir="current_run"):
    """
    Ejecutar pipeline OAM en modo headless (sin GUI)

    Args:
        config_file: Ruta a JSON con configuración (overrides de SYSTEM_CONFIG)
        save_dir: Directorio donde guardar artefactos

    Returns:
        bool: True si exitoso, False si falló
    """
    try:
        log_info('headless', "=== MODO HEADLESS ACTIVADO ===")

        # Limpiar archivos corruptos
        cleanup_corrupted_files()

        # Reset pipeline
        from pipeline import pipeline
        pipeline.reset()
        log_info('headless', "Pipeline reset completed")

        # Cargar configuración desde JSON si existe
        if config_file and os.path.exists(config_file):
            import json
            log_info('headless', f"Loading configuration from: {config_file}")
            with open(config_file, 'r') as f:
                config_overrides = json.load(f)

            # Fusionar con SYSTEM_CONFIG
            import oam_system_config
            oam_system_config.SYSTEM_CONFIG.update(config_overrides)
            log_info('headless', f"Configuración actualizada: {config_overrides.get('num_oam_modes', '?')} modos OAM")

        # Crear directorio de salida
        os.makedirs(save_dir, exist_ok=True)

        # Borrar archivo .done anterior si existe
        done_file = os.path.join(save_dir, '.done')
        if os.path.exists(done_file):
            os.remove(done_file)
            log_info('headless', "Archivo .done anterior eliminado")

        # Crear top_block sin GUI
        log_info('headless', "Creando flowgraph GNU Radio (sin GUI)...")

        # No podemos usar Qt en headless, así que usamos gr.top_block directamente
        # Importar bloques individuales y forzar recarga para que lean SYSTEM_CONFIG actualizado
        import importlib
        import oam_source
        import oam_encoder
        import oam_channel
        import oam_decoder

        # Recargar módulos para que lean la configuración actualizada
        importlib.reload(oam_source)
        importlib.reload(oam_encoder)
        importlib.reload(oam_channel)
        importlib.reload(oam_decoder)

        # Crear top_block básico
        tb = gr.top_block("OAM Headless")

        # Crear bloques con parámetros del JSON
        log_info('headless', "Instanciando bloques...")

        # Extraer parámetros del config actualizado
        num_channels = oam_system_config.SYSTEM_CONFIG.get('num_oam_modes')
        wavelength = oam_system_config.SYSTEM_CONFIG.get('wavelength')
        tx_aperture = oam_system_config.SYSTEM_CONFIG.get('tx_aperture_size')
        distance = oam_system_config.SYSTEM_CONFIG.get('propagation_distance')
        cn2 = oam_system_config.SYSTEM_CONFIG.get('cn2')
        snr = oam_system_config.SYSTEM_CONFIG.get('snr_target')
        rx_aperture = oam_system_config.SYSTEM_CONFIG.get('rx_aperture_size')
        grid_size = oam_system_config.SYSTEM_CONFIG.get('grid_size')

        # Convertir num_channels a lista de modos OAM
        # Si num_channels=2 → [-1, 1]
        # Si num_channels=4 → [-2, -1, 1, 2]
        # Si num_channels=6 → [-3, -2, -1, 1, 2, 3]
        from oam_system_config import get_oam_channels
        oam_modes = get_oam_channels()  # Esto lee de SYSTEM_CONFIG que ya fue actualizado

        log_info('headless', f"Parámetros: {num_channels} modos → {oam_modes}, grid={grid_size}, dist={distance}m")

        source = oam_source.oam_source()
        encoder = oam_encoder.oam_encoder(
            grid_size=grid_size,
            wavelength=wavelength,
            tx_aperture_size=tx_aperture,
            oam_channels=oam_modes
        )
        channel = oam_channel.oam_channel(
            propagation_distance=distance,
            cn2=cn2,
            snr_db=snr,
            oam_channels=oam_modes
        )
        decoder = oam_decoder.oam_decoder(
            grid_size=grid_size,
            wavelength=wavelength,
            oam_channels=oam_modes,
            propagation_distance=distance
        )

        # Null sink para terminar el flowgraph (decoder output es bytes)
        from gnuradio import blocks
        null_sink = blocks.null_sink(gr.sizeof_char)

        # Conectar bloques
        tb.connect(source, encoder)
        tb.connect(encoder, channel)
        tb.connect(channel, decoder)
        tb.connect(decoder, null_sink)

        log_info('headless', "Flowgraph created, executing pipeline...")

        # Ejecutar
        tb.start()
        tb.wait()

        log_info('headless', "Flowgraph completed, saving artifacts...")

        # Guardar datos del pipeline al finalizar
        from pipeline import pipeline
        pipeline._save_pipeline_data()

        # Crear archivo .done para indicar que el pipeline terminó
        done_file = os.path.join(save_dir, '.done')
        with open(done_file, 'w') as f:
            import time
            f.write(f"Pipeline completado: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        log_info('headless', "Pipeline completed successfully")
        log_info('headless', f"Artifacts saved in: {save_dir}/")
        log_info('headless', f"Done file created: {done_file}")

        return True

    except Exception as e:
        log_error('headless', f"Error en modo headless: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import argparse

    # Parsear argumentos
    parser = argparse.ArgumentParser(description='Sistema OAM Completo')
    parser.add_argument('--headless', action='store_true', help='Ejecutar sin GUI')
    parser.add_argument('--config', type=str, help='Archivo JSON con configuración')
    parser.add_argument('--save-dir', type=str, default='current_run', help='Directorio para artefactos')
    args = parser.parse_args()

    # Modo headless
    if args.headless:
        log_info('main', "=== INICIANDO EN MODO HEADLESS ===")
        success = run_headless_pipeline(config_file=args.config, save_dir=args.save_dir)
        sys.exit(0 if success else 1)

    # Modo GUI normal
    # VERIFICACIÓN DE CACHE INTELIGENTE
    run_simulation, cache_reason = check_simulation_cache()

    if not run_simulation:
        log_info('main', f"USANDO CACHE: {cache_reason}")
        try:
            if launch_cached_dashboards():
                log_info('main', "Simulacion completa usando cache - tiempo ahorrado")
                exit(0)
            else:
                log_warning('main', "Error con cache - ejecutando simulacion completa")
        except Exception as e:
            log_error('main', f"Error usando cache: {e} - ejecutando simulacion")

    # Ejecutar simulacion normal si cache no funciona o no es valido
    log_info('main', f"EJECUTANDO SIMULACION: {cache_reason}")

    try:
        main()
    except KeyboardInterrupt:
        log_info('main', "Programa interrumpido por el usuario")
        # Limpiar procesos en segundo plano si existen
        import subprocess
        try:
            subprocess.run(['pkill', '-f', 'oam_visualizer.py'], check=False)
            log_info('main', "Procesos de visualización cerrados")
        except:
            pass
    except Exception as e:
        log_error('main', f"Error en el programa principal: {e}")
    finally:
        log_info('main', "Programa terminado - todos los procesos cerrados")