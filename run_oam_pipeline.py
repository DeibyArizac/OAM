#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script standalone para ejecutar pipeline OAM sin GNU Radio
Lee configuración de config_manager y ejecuta procesamiento completo
"""

import sys
import os

# Agregar directorio actual al path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from oam_logging import log_info, log_error
from config_manager import get_current_config
from oam_system_config import SYSTEM_CONFIG
import oam_system_config

def main():
    """Ejecutar pipeline OAM completo"""
    try:
        # Leer configuración actualizada desde config_manager
        config = get_current_config()

        log_info('pipeline_runner', "=== EJECUTANDO PIPELINE OAM STANDALONE ===")
        log_info('pipeline_runner', f"Configuración: {config['num_oam_channels']} canales OAM")

        # Actualizar SYSTEM_CONFIG con valores de config_manager
        oam_system_config.SYSTEM_CONFIG.update(config)

        log_info('pipeline_runner', f"SYSTEM_CONFIG actualizado en memoria")

        # Ahora ejecutar el script completo que ya existe
        from oam_complete_system import check_simulation_cache, cleanup_corrupted_files

        # Limpiar archivos corruptos
        cleanup_corrupted_files()

        # Verificar si necesitamos regenerar
        run_sim, reason = check_simulation_cache()

        if run_sim:
            log_info('pipeline_runner', f"Regenerando simulación: {reason}")

            # Importar y ejecutar bloques en modo standalone
            # Esto requiere que los bloques tengan métodos que no dependan de GNU Radio
            # Por ahora, ejecutamos el sistema completo en modo headless

            import subprocess
            script_path = os.path.join(current_dir, 'oam_complete_system.py')

            # Ejecutar sin GUI
            result = subprocess.run(
                [sys.executable, script_path, '--no-gui'],
                cwd=current_dir,
                capture_output=True,
                timeout=120
            )

            if result.returncode == 0:
                log_info('pipeline_runner', "Pipeline ejecutado correctamente")
                return 0
            else:
                log_error('pipeline_runner', f"Pipeline falló con código {result.returncode}")
                if result.stderr:
                    log_error('pipeline_runner', result.stderr.decode()[:500])
                return 1
        else:
            log_info('pipeline_runner', f"Usando caché: {reason}")
            return 0

    except Exception as e:
        log_error('pipeline_runner', f"Error ejecutando pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())