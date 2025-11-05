#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WRAPPER GNU RADIO: OAM ENCODER BLOCK
===================================

Este bloque NO ejecuta código - solo edita parámetros del encoder.

Función:
- Recibe parámetros del GUI de GNU Radio relacionados con codificación OAM
- Actualiza config_manager con estos parámetros
- El bloque OAM Source es el que realmente usa estos parámetros
- Proporciona interfaz visual para configurar el encoder

Parámetros del Encoder:
- num_oam_channels: Número de canales OAM (2, 4, 6, 8, 10, 12)
- wavelength: Longitud de onda del láser [m]
- tx_power: Potencia del transmisor [W]
- tx_aperture_size: Diámetro de apertura TX [m]
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
from oam_logging import log_info, log_debug

class oam_encoder_wr(gr.sync_block):
    """
    Wrapper GNU Radio para OAM Encoder

    FUNCIÓN: Solo actualizar parámetros del encoder - NO EJECUTA CÓDIGO

    Este bloque:
    1. Recibe parámetros del GUI
    2. Los valida
    3. Los guarda en config_manager
    4. ¡No hace nada más!
    5. OAM Source usará estos parámetros cuando ejecute
    """

    def __init__(self, num_oam_modes=2, wavelength=630e-9, tx_power=0.01, tx_aperture_size=35e-3, grid_size=512):
        """
        Inicializar OAM Encoder - Solo configuración

        Args:
            num_oam_modes: Número de modos OAM (2, 4, 6, 8)
                          2 modos → ±1
                          4 modos → ±1, ±2
                          6 modos → ±1, ±2, ±3
                          8 modos → ±1, ±2, ±3, ±4
            wavelength: Longitud de onda [m] (532e-9, 630e-9, 850e-9, 1550e-9)
            tx_power: Potencia del transmisor [W] (0.001 a 1.0)
            tx_aperture_size: Diámetro apertura TX [m] (10e-3 a 50e-3)
            grid_size: Tamaño de la grilla computacional (256, 512, 1024)
        """
        gr.sync_block.__init__(
            self,
            name="oam_encoder_wr",
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )

        # Guardar parámetros en el objeto
        self.num_oam_modes = num_oam_modes
        self.wavelength = wavelength
        self.tx_power = tx_power
        self.tx_aperture_size = tx_aperture_size
        self.grid_size = grid_size

        log_debug('encoder_wr', "=== CONFIGURING PARAMETERS ENCODER ===")

        # Validar parámetros antes de guardar
        self._validate_encoder_params(num_oam_modes, wavelength, tx_power, tx_aperture_size, grid_size)

        # Actualizar parámetros en config_manager
        self._update_config()

        log_debug('encoder_wr', "=== ENCODER CONFIGURACIÓN COMPLETADA ===")

    def start(self):
        """Llamado cuando GNU Radio presiona Run - escribir parámetros al caché JSON"""
        from gnuradio_cache import write_block_params, get_run_id_from_cache
        from oam_logging import log_warning

        # Obtener run_id del caché (generado por Source)
        run_id = get_run_id_from_cache()

        if not run_id:
            log_warning('encoder_wr', "No se encontró run_id en caché - usando timestamp")
            from datetime import datetime
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Escribir parámetros al caché
        encoder_params = {
            'num_oam_modes': self.num_oam_modes,
            'wavelength': self.wavelength,
            'tx_power': self.tx_power,
            'tx_aperture_size': self.tx_aperture_size,
            'grid_size': self.grid_size
        }

        write_block_params('oam_encoder', encoder_params, run_id)
        log_debug('encoder_wr', f"Parameters written to cache (run_id: {run_id})")

        return True

    def _update_config(self):
        """Actualizar config_manager con parámetros actuales"""
        config_manager.update_encoder_params(
            num_oam_modes=self.num_oam_modes,
            wavelength=self.wavelength,
            tx_power=self.tx_power,
            tx_aperture_size=self.tx_aperture_size,
            grid_size=self.grid_size
        )

        log_debug('encoder_wr', f" Encoder configured:")
        log_debug('encoder_wr', f"  - Modos OAM: {self.num_oam_modes}")
        log_debug('encoder_wr', f"  - Longitud de onda: {self.wavelength*1e9:.0f} nm")
        log_debug('encoder_wr', f"  - Potencia TX: {self.tx_power*1000:.1f} mW")
        log_debug('encoder_wr', f"  - Apertura TX: {self.tx_aperture_size*1000:.1f} mm")
        log_debug('encoder_wr', f"  - Grid Size: {self.grid_size}")

        self._log_derived_properties(self.num_oam_modes, self.wavelength, self.tx_aperture_size)

    def _validate_encoder_params(self, num_oam_modes, wavelength, tx_power, tx_aperture_size, grid_size):
        """
        Validar parámetros del encoder antes de guardar
        """
        errors = []

        # Validar num_oam_modes
        if not isinstance(num_oam_modes, int):
            errors.append("num_oam_modes debe ser entero")
        elif num_oam_modes not in [2, 4, 6, 8, 10, 12]:
            errors.append("num_oam_modes debe ser 2, 4, 6, 8, 10 o 12")
        elif num_oam_modes % 2 != 0:
            errors.append("num_oam_modes debe ser par")

        # Validar wavelength
        if not isinstance(wavelength, (int, float)):
            errors.append("wavelength debe ser numérico")
        elif wavelength < 400e-9 or wavelength > 2000e-9:
            errors.append("wavelength debe estar entre 400nm y 2000nm")

        # Validar tx_power
        if not isinstance(tx_power, (int, float)):
            errors.append("tx_power debe ser numérico")
        elif tx_power < 0.001 or tx_power > 10.0:
            errors.append("tx_power debe estar entre 0.001W y 10W")

        # Validar tx_aperture_size
        if not isinstance(tx_aperture_size, (int, float)):
            errors.append("tx_aperture_size debe ser numérico")
        elif tx_aperture_size < 5e-3 or tx_aperture_size > 100e-3:
            errors.append("tx_aperture_size debe estar entre 5mm y 100mm")

        # Validar grid_size
        if not isinstance(grid_size, int):
            errors.append("grid_size debe ser entero")
        elif grid_size not in [256, 512, 1024, 2048]:
            errors.append("grid_size debe ser 256, 512, 1024 o 2048")

        if errors:
            error_msg = "Errores en parámetros del Encoder: " + "; ".join(errors)
            log_info('encoder_wr', f" {error_msg}")
            raise ValueError(error_msg)

        log_debug('encoder_wr', " Parameters validated correctly")

    def _log_derived_properties(self, num_oam_modes, wavelength, tx_aperture_size):
        """
        Calcular y mostrar propiedades derivadas para información
        """
        try:
            # Canales OAM generados
            max_mode = num_oam_channels // 2
            oam_modes = list(range(-max_mode, 0)) + list(range(1, max_mode + 1))

            # Cintura del haz (beam waist)
            beam_waist = tx_aperture_size * 0.5

            # Información adicional
            log_debug('encoder_wr', f"Propiedades derivadas:")
            log_debug('encoder_wr', f"  - Modos OAM: {oam_modes}")
            log_debug('encoder_wr', f"  - Modos por símbolo: {len(oam_modes)//2}")
            log_debug('encoder_wr', f"  - Cintura del haz: {beam_waist*1000:.1f} mm")
            log_debug('encoder_wr', f"  - Frecuencia óptica: {3e8/wavelength:.2e} Hz")

            # Warnings para configuraciones extremas
            if num_oam_channels > 8:
                log_info('encoder_wr', " Configuración con muchos canales OAM - puede afectar robustez")

            if wavelength > 1000e-9:
                log_info('encoder_wr', " Longitud de onda infrarroja - verificar compatibilidad atmosférica")

            if tx_power > 0.1:
                log_info('encoder_wr', " Alta potencia láser - considerar seguridad ocular")

        except Exception as e:
            log_debug('encoder_wr', f"Error calculando propiedades derivadas: {e}")

    def work(self, input_items, output_items):
        """
        Función work de GNU Radio - simplemente pasa los datos sin modificar
        Este bloque solo configura parámetros, no procesa señales
        """
        output_items[0][:] = input_items[0]
        return len(output_items[0])

    def get_current_config(self):
        """
        Obtener configuración actual (para debugging)
        Normalmente no se usa - solo para testing
        """
        current_config = config_manager.get_full_config()
        encoder_config = {
            'num_oam_channels': current_config.get('num_oam_channels'),
            'wavelength': current_config.get('wavelength'),
            'tx_power': current_config.get('tx_power'),
            'tx_aperture_size': current_config.get('tx_aperture_size')
        }
        return encoder_config

# ========================================================================
# HELPER FUNCTIONS PARA GNU RADIO
# ========================================================================

def make_oam_encoder_wr(num_oam_channels=6, wavelength=630e-9, tx_power=0.01, tx_aperture_size=35e-3):
    """Factory function para GNU Radio block"""
    return oam_encoder_wr(num_oam_channels, wavelength, tx_power, tx_aperture_size)

# ========================================================================
# CONFIGURACIONES PREESTABLECIDAS
# ========================================================================

class EncoderPresets:
    """Configuraciones preestablecidas para diferentes escenarios"""

    # Configuración conservadora (alta confiabilidad)
    CONSERVATIVE = {
        'num_oam_channels': 4,
        'wavelength': 630e-9,
        'tx_power': 0.01,
        'tx_aperture_size': 50e-3
    }

    # Configuración estándar (balance rendimiento-robustez)
    STANDARD = {
        'num_oam_channels': 6,
        'wavelength': 630e-9,
        'tx_power': 0.01,
        'tx_aperture_size': 35e-3
    }

    # Configuración alta capacidad (máximo throughput)
    HIGH_CAPACITY = {
        'num_oam_channels': 8,
        'wavelength': 532e-9,
        'tx_power': 0.05,
        'tx_aperture_size': 40e-3
    }

    # Configuración infrarroja (condiciones específicas)
    INFRARED = {
        'num_oam_channels': 6,
        'wavelength': 1550e-9,
        'tx_power': 0.001,
        'tx_aperture_size': 25e-3
    }

# ========================================================================
# TEST Y DEMO
# ========================================================================

if __name__ == "__main__":
    print("=== TEST OAM ENCODER WRAPPER ===")

    # Test configuración estándar
    print("Test 1: Configuración estándar...")
    try:
        encoder1 = oam_encoder_wr()
        print(" Configuración estándar OK")
    except Exception as e:
        print(f" Error en configuración estándar: {e}")

    # Test configuración personalizada
    print("\nTest 2: Configuración personalizada...")
    try:
        encoder2 = oam_encoder_wr(
            num_oam_channels=8,
            wavelength=532e-9,
            tx_power=0.05,
            tx_aperture_size=40e-3
        )
        print(" Configuración personalizada OK")
    except Exception as e:
        print(f" Error en configuración personalizada: {e}")

    # Test validación de errores
    print("\nTest 3: Validación de errores...")
    try:
        encoder3 = oam_encoder_wr(
            num_oam_channels=3,  # Impar - debe fallar
            wavelength=532e-9,
            tx_power=0.05,
            tx_aperture_size=40e-3
        )
        print(" Debería haber fallado con num_oam_channels impar")
    except ValueError:
        print(" Validación de errores funcionando correctamente")
    except Exception as e:
        print(f" Error inesperado: {e}")

    # Test presets
    print("\nTest 4: Configuraciones preestablecidas...")
    try:
        encoder4 = oam_encoder_wr(**EncoderPresets.HIGH_CAPACITY)
        print(" Preset alta capacidad OK")
    except Exception as e:
        print(f" Error en preset: {e}")

    print("\n=== TEST COMPLETADO ===")