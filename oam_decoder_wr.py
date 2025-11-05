#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WRAPPER GNU RADIO: OAM DECODER BLOCK
===================================

Este bloque NO ejecuta código - solo edita parámetros del decoder/receptor.

Función:
- Recibe parámetros del GUI de GNU Radio relacionados con el receptor/decoder
- Actualiza config_manager con estos parámetros
- El bloque OAM Source es el que realmente usa estos parámetros
- Proporciona interfaz visual para configurar el receptor OAM

Parámetros del Decoder:
- rx_aperture_size: Diámetro de apertura RX [m]
- grid_size: Resolución de cálculo (256, 512, 1024)
- correlation_threshold: Umbral para detección modal
- detection_method: Método de detección ("ncc", "correlation", "ml")
"""

import os
import numpy as np
from gnuradio import gr
import sys
import math

# Agregar directorio actual al path si no está
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from config_manager import config_manager
from oam_logging import log_info, log_debug, log_warning

class oam_decoder_wr(gr.sync_block):
    """
    Wrapper GNU Radio para OAM Decoder

    FUNCIÓN: Solo actualizar parámetros del decoder - NO EJECUTA CÓDIGO

    Este bloque:
    1. Recibe parámetros del GUI
    2. Los valida
    3. Calcula métricas de rendimiento esperado
    4. Los guarda en config_manager
    5. ¡No hace nada más!
    6. OAM Source usará estos parámetros cuando ejecute
    """

    def __init__(self, rx_aperture_size=35e-3, grid_size=512, correlation_threshold=0.1, detection_method="ncc"):
        """
        Inicializar OAM Decoder - Solo configuración

        Args:
            rx_aperture_size: Diámetro apertura RX [m] (10e-3 a 100e-3)
            grid_size: Resolución computacional (256, 512, 1024)
            correlation_threshold: Umbral detección modal (0.01 a 0.9)
            detection_method: Método detección ("ncc", "correlation", "ml")
        """
        gr.sync_block.__init__(
            self,
            name="oam_decoder_wr",
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )

        # Guardar parámetros en el objeto
        self.rx_aperture_size = rx_aperture_size
        self.grid_size = grid_size
        self.correlation_threshold = correlation_threshold
        self.detection_method = detection_method

        log_debug('decoder_wr', "=== CONFIGURING PARAMETERS DECODER ===")

        # Validar parámetros antes de guardar
        self._validate_decoder_params(rx_aperture_size, grid_size, correlation_threshold, detection_method)

        # Actualizar parámetros en config_manager
        self._update_config()

        log_debug('decoder_wr', "=== DECODER CONFIGURACIÓN COMPLETADA ===")

    def start(self):
        """Llamado cuando GNU Radio presiona Run - escribir parámetros al caché JSON"""
        from gnuradio_cache import write_block_params, get_run_id_from_cache

        # Obtener run_id del caché (generado por Source)
        run_id = get_run_id_from_cache()

        if not run_id:
            log_warning('decoder_wr', "No se encontró run_id en caché - usando timestamp")
            from datetime import datetime
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Escribir parámetros al caché
        decoder_params = {
            'rx_aperture_size': self.rx_aperture_size,
            'grid_size': self.grid_size,
            'correlation_threshold': self.correlation_threshold,
            'detection_method': self.detection_method
        }

        write_block_params('oam_decoder', decoder_params, run_id)
        log_debug('decoder_wr', f"Parameters written to cache (run_id: {run_id})")

        return True

    def _update_config(self):
        """Actualizar config_manager con parámetros actuales"""
        config_manager.update_decoder_params(
            rx_aperture_size=self.rx_aperture_size,
            grid_size=self.grid_size,
            correlation_threshold=self.correlation_threshold,
            detection_method=self.detection_method
        )

        log_info('decoder_wr', f" Decoder configured:")
        log_debug('decoder_wr', f"  - Apertura RX: {self.rx_aperture_size*1000:.1f} mm")
        log_debug('decoder_wr', f"  - Resolución grid: {self.grid_size}x{self.grid_size}")
        log_debug('decoder_wr', f"  - Umbral correlación: {self.correlation_threshold}")
        log_debug('decoder_wr', f"  - Método detección: {self.detection_method}")

        self._calculate_decoder_metrics(self.rx_aperture_size, self.grid_size, self.correlation_threshold)

    def _validate_decoder_params(self, rx_aperture_size, grid_size, correlation_threshold, detection_method):
        """
        Validar parámetros del decoder antes de guardar
        """
        errors = []

        # Validar rx_aperture_size
        if not isinstance(rx_aperture_size, (int, float)):
            errors.append("rx_aperture_size debe ser numérico")
        elif rx_aperture_size < 5e-3 or rx_aperture_size > 200e-3:
            errors.append("rx_aperture_size debe estar entre 5mm y 200mm")

        # Validar grid_size
        valid_grid_sizes = [256, 512, 1024]
        if not isinstance(grid_size, int):
            errors.append("grid_size debe ser entero")
        elif grid_size not in valid_grid_sizes:
            errors.append(f"grid_size debe ser uno de {valid_grid_sizes}")

        # Validar correlation_threshold
        if not isinstance(correlation_threshold, (int, float)):
            errors.append("correlation_threshold debe ser numérico")
        elif correlation_threshold < 0.001 or correlation_threshold > 0.9:
            errors.append("correlation_threshold debe estar entre 0.001 y 0.9")

        # Validar detection_method
        valid_methods = ["ncc", "correlation", "ml", "peak_detection"]
        if detection_method not in valid_methods:
            errors.append(f"detection_method debe ser uno de {valid_methods}")

        if errors:
            error_msg = "Errores en parámetros del Decoder: " + "; ".join(errors)
            log_info('decoder_wr', f" {error_msg}")
            raise ValueError(error_msg)

        log_debug('decoder_wr', " Parameters validated correctly")

    def _calculate_decoder_metrics(self, rx_aperture_size, grid_size, correlation_threshold):
        """
        Calcular métricas de rendimiento esperado del decoder
        """
        try:
            # Obtener configuración actual para calcular métricas completas
            current_config = config_manager.get_full_config()

            # Resolución angular (limitada por difracción)
            wavelength = current_config.get('wavelength', 630e-9)
            angular_resolution = 1.22 * wavelength / rx_aperture_size  # radianes

            # Resolución espacial en el plano del receptor
            propagation_distance = current_config.get('propagation_distance', 50)
            spatial_resolution = angular_resolution * propagation_distance  # metros

            # Número de elementos resolvibles en la apertura
            resolvable_elements = (rx_aperture_size / spatial_resolution)**2

            # Área efectiva de captura
            capture_area = math.pi * (rx_aperture_size/2)**2  # m²

            # Factor de calidad del grid computacional
            grid_quality = "Alta" if grid_size >= 512 else "Media" if grid_size >= 256 else "Baja"

            # Estimación de rendimiento esperado
            expected_snr_loss = self._estimate_snr_loss(rx_aperture_size, grid_size, correlation_threshold)

            # Log métricas
            log_info('decoder_wr', f"Métricas del decoder:")
            log_debug('decoder_wr', f"  - Resolución angular: {angular_resolution*1e6:.1f} μrad")
            log_debug('decoder_wr', f"  - Resolución espacial @ {propagation_distance}m: {spatial_resolution*1000:.1f} mm")
            log_debug('decoder_wr', f"  - Elementos resolvibles: {resolvable_elements:.0f}")
            log_debug('decoder_wr', f"  - Área de captura: {capture_area*1e4:.1f} cm²")
            log_debug('decoder_wr', f"  - Calidad grid computacional: {grid_quality}")
            log_debug('decoder_wr', f"  - Pérdida SNR estimada: {expected_snr_loss:.1f} dB")

            # Recomendaciones basadas en métricas
            self._provide_decoder_recommendations(rx_aperture_size, grid_size, correlation_threshold,
                                                 current_config, resolvable_elements)

        except Exception as e:
            log_debug('decoder_wr', f"Error calculando métricas del decoder: {e}")

    def _estimate_snr_loss(self, rx_aperture_size, grid_size, correlation_threshold):
        """
        Estimar pérdida de SNR debido a configuración del decoder
        """
        snr_loss = 0.0

        # Pérdida por tamaño de apertura (menor apertura = menos luz capturada)
        if rx_aperture_size < 20e-3:
            snr_loss += 3.0  # 3 dB loss for small aperture

        # Pérdida por resolución de grid (menor grid = menos precisión)
        if grid_size < 512:
            snr_loss += 1.5  # 1.5 dB loss for low grid resolution

        # Pérdida por umbral de correlación (umbral alto = más false negatives)
        if correlation_threshold > 0.5:
            snr_loss += 2.0  # 2 dB effective loss due to high threshold

        return snr_loss

    def _provide_decoder_recommendations(self, rx_aperture_size, grid_size, correlation_threshold,
                                       current_config, resolvable_elements):
        """
        Proporcionar recomendaciones basadas en la configuración
        """
        # Obtener número de canales OAM
        num_channels = current_config.get('num_oam_channels', 6)
        max_mode = num_channels // 2

        # Recomendación: elementos resolvibles vs modos OAM
        if resolvable_elements < max_mode * 10:
            log_warning('decoder_wr',
                       f" Pocos elementos resolvibles ({resolvable_elements:.0f}) para "
                       f"{num_channels} canales OAM. Considera aumentar apertura RX")

        # Recomendación: grid size vs aperture
        optimal_grid = max(256, int(rx_aperture_size * 10000))  # Heurística simple
        if grid_size < optimal_grid:
            log_warning('decoder_wr',
                       f" Resolución de grid ({grid_size}) podría ser insuficiente. "
                       f"Considera usar {optimal_grid} o superior")

        # Recomendación: correlation threshold
        if correlation_threshold < 0.05:
            log_warning('decoder_wr', " Umbral de correlación muy bajo - posibles falsos positivos")
        elif correlation_threshold > 0.7:
            log_warning('decoder_wr', " Umbral de correlación muy alto - posibles pérdidas de señal")

        # Recomendación: balance TX/RX aperture
        tx_aperture = current_config.get('tx_aperture_size', 35e-3)
        if abs(rx_aperture_size - tx_aperture) > 20e-3:
            log_info('decoder_wr',
                    f" Diferencia significativa TX/RX aperture "
                    f"({tx_aperture*1000:.1f}mm vs {rx_aperture_size*1000:.1f}mm)")

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
        """
        current_config = config_manager.get_full_config()
        decoder_config = {
            'rx_aperture_size': current_config.get('rx_aperture_size'),
            'grid_size': current_config.get('grid_size'),
            'correlation_threshold': current_config.get('correlation_threshold'),
            'detection_method': current_config.get('detection_method')
        }
        return decoder_config

# ========================================================================
# HELPER FUNCTIONS PARA GNU RADIO
# ========================================================================

def make_oam_decoder_wr(rx_aperture_size=35e-3, grid_size=512, correlation_threshold=0.1, detection_method="ncc"):
    """Factory function para GNU Radio block"""
    return oam_decoder_wr(rx_aperture_size, grid_size, correlation_threshold, detection_method)

# ========================================================================
# CONFIGURACIONES PREESTABLECIDAS
# ========================================================================

class DecoderPresets:
    """Configuraciones preestablecidas para diferentes escenarios"""

    # Alta precisión (máxima calidad)
    HIGH_PRECISION = {
        'rx_aperture_size': 50e-3,
        'grid_size': 1024,
        'correlation_threshold': 0.15,
        'detection_method': 'ncc'
    }

    # Configuración estándar (balance rendimiento-calidad)
    STANDARD = {
        'rx_aperture_size': 35e-3,
        'grid_size': 512,
        'correlation_threshold': 0.1,
        'detection_method': 'ncc'
    }

    # Alta velocidad (optimizado para speed)
    HIGH_SPEED = {
        'rx_aperture_size': 25e-3,
        'grid_size': 256,
        'correlation_threshold': 0.2,
        'detection_method': 'peak_detection'
    }

    # Condiciones desafiantes (mayor robustez)
    ROBUST = {
        'rx_aperture_size': 60e-3,
        'grid_size': 512,
        'correlation_threshold': 0.05,
        'detection_method': 'correlation'
    }

    # Modo experimental (para testing)
    EXPERIMENTAL = {
        'rx_aperture_size': 40e-3,
        'grid_size': 1024,
        'correlation_threshold': 0.3,
        'detection_method': 'ml'
    }

# ========================================================================
# TEST Y DEMO
# ========================================================================

if __name__ == "__main__":
    print("=== TEST OAM DECODER WRAPPER ===")

    # Test configuración estándar
    print("Test 1: Configuración estándar...")
    try:
        decoder1 = oam_decoder_wr()
        print(" Configuración estándar OK")
    except Exception as e:
        print(f" Error en configuración estándar: {e}")

    # Test configuración personalizada
    print("\nTest 2: Configuración personalizada...")
    try:
        decoder2 = oam_decoder_wr(
            rx_aperture_size=45e-3,
            grid_size=1024,
            correlation_threshold=0.15,
            detection_method="correlation"
        )
        print(" Configuración personalizada OK")
    except Exception as e:
        print(f" Error en configuración personalizada: {e}")

    # Test validación de errores
    print("\nTest 3: Validación de errores...")
    try:
        decoder3 = oam_decoder_wr(
            grid_size=600  # No válido - debe fallar
        )
        print(" Debería haber fallado con grid_size inválido")
    except ValueError:
        print(" Validación de errores funcionando correctamente")
    except Exception as e:
        print(f" Error inesperado: {e}")

    # Test configuraciones preestablecidas
    print("\nTest 4: Configuraciones preestablecidas...")
    presets_to_test = [
        ("HIGH_PRECISION", DecoderPresets.HIGH_PRECISION),
        ("HIGH_SPEED", DecoderPresets.HIGH_SPEED),
        ("ROBUST", DecoderPresets.ROBUST)
    ]

    for preset_name, preset_config in presets_to_test:
        try:
            decoder = oam_decoder_wr(**preset_config)
            print(f" Preset {preset_name} OK")
        except Exception as e:
            print(f" Error en preset {preset_name}: {e}")

    print("\n=== TEST COMPLETADO ===")