#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WRAPPER GNU RADIO: OAM CHANNEL BLOCK
===================================

Este bloque NO ejecuta código - solo edita parámetros del canal atmosférico.

Función:
- Recibe parámetros del GUI de GNU Radio relacionados con el canal de propagación
- Actualiza config_manager con estos parámetros
- El bloque OAM Source es el que realmente usa estos parámetros
- Proporciona interfaz visual para configurar el canal atmosférico

Parámetros del Canal:
- propagation_distance: Distancia TX-RX [m]
- cn2: Turbulencia atmosférica [m^-2/3]
- snr_target: SNR objetivo [dB]
- atmospheric_conditions: Condiciones atmosféricas preset
"""

import os
import sys
import math
import numpy as np
from gnuradio import gr

# Agregar directorio actual al path si no está
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from config_manager import config_manager
from oam_logging import log_info, log_debug, log_warning

class oam_channel_wr(gr.sync_block):
    """
    Wrapper GNU Radio para OAM Channel

    FUNCIÓN: Solo actualizar parámetros del canal - NO EJECUTA CÓDIGO

    Este bloque:
    1. Recibe parámetros del GUI
    2. Los valida
    3. Calcula métricas derivadas
    4. Los guarda en config_manager
    5. ¡No hace nada más!
    6. OAM Source usará estos parámetros cuando ejecute
    """

    def __init__(self, propagation_distance=50, cn2=1e-15, snr_target=30, atmospheric_conditions="clear"):
        """
        Inicializar OAM Channel - Solo configuración

        Args:
            propagation_distance: Distancia TX-RX [m] (1 a 10000)
            cn2: Índice de turbulencia [m^-2/3] (1e-17 a 1e-13)
            snr_target: SNR objetivo [dB] (10 a 40)
            atmospheric_conditions: Preset de condiciones ("clear", "light_turb", "heavy_turb")
        """
        gr.sync_block.__init__(
            self,
            name="oam_channel_wr",
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )

        # Guardar parámetros en el objeto
        self.propagation_distance = propagation_distance
        self.cn2 = cn2
        self.snr_target = snr_target
        self.atmospheric_conditions = atmospheric_conditions

        log_debug('channel_wr', "=== CONFIGURING PARAMETERS CANAL ===")

        # Aplicar preset si se especifica
        if atmospheric_conditions != "clear":
            self.propagation_distance, self.cn2, self.snr_target = self._apply_atmospheric_preset(
                atmospheric_conditions, propagation_distance, cn2, snr_target
            )

        # Validar parámetros antes de guardar
        self._validate_channel_params(self.propagation_distance, self.cn2, self.snr_target, self.atmospheric_conditions)

        # Actualizar parámetros en config_manager
        self._update_config()

        log_debug('channel_wr', "=== CANAL CONFIGURACIÓN COMPLETADA ===")

    def start(self):
        """Llamado cuando GNU Radio presiona Run - escribir parámetros al caché JSON"""
        from gnuradio_cache import write_block_params, get_run_id_from_cache

        run_id = get_run_id_from_cache()
        if not run_id:
            from datetime import datetime
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        channel_params = {
            'propagation_distance': self.propagation_distance,
            'cn2': self.cn2,
            'snr_target': self.snr_target,
            'atmospheric_conditions': self.atmospheric_conditions
        }

        write_block_params('oam_channel', channel_params, run_id)
        log_debug('channel_wr', f"Parameters written to cache (run_id: {run_id})")
        return True

    def _update_config(self):
        """Actualizar config_manager con parámetros actuales"""
        config_manager.update_channel_params(
            propagation_distance=self.propagation_distance,
            cn2=self.cn2,
            snr_target=self.snr_target,
            atmospheric_conditions=self.atmospheric_conditions
        )

        log_info('channel_wr', f" Canal configured:")
        log_debug('channel_wr', f"  - Distancia: {self.propagation_distance} m")
        log_debug('channel_wr', f"  - Turbulencia Cn²: {self.cn2:.2e} m^-2/3")
        log_debug('channel_wr', f"  - SNR objetivo: {self.snr_target} dB")
        log_debug('channel_wr', f"  - Condiciones: {self.atmospheric_conditions}")

        self._calculate_channel_metrics(self.propagation_distance, self.cn2, self.snr_target)

    def _apply_atmospheric_preset(self, conditions, distance, cn2, snr):
        """
        Aplicar presets de condiciones atmosféricas
        """
        log_info('channel_wr', f"Aplicando preset atmosférico: {conditions}")

        if conditions == "light_turb":
            # Turbulencia ligera
            cn2 = 1e-16
            snr = max(25, snr - 5)  # Reducir SNR ligeramente
            log_info('channel_wr', "Preset: Turbulencia ligera aplicado")

        elif conditions == "heavy_turb":
            # Turbulencia pesada
            cn2 = 1e-14
            snr = max(15, snr - 10)  # Reducir SNR significativamente
            log_info('channel_wr', "Preset: Turbulencia pesada aplicado")

        elif conditions == "clear":
            # Condiciones claras (valores por defecto)
            pass

        else:
            log_warning('channel_wr', f"Condición atmosférica desconocida: {conditions}")

        return distance, cn2, snr

    def _validate_channel_params(self, propagation_distance, cn2, snr_target, atmospheric_conditions):
        """
        Validar parámetros del canal antes de guardar
        """
        # Log valores recibidos para debugging
        log_debug('channel_wr', f"Validando: distance={propagation_distance}, cn2={cn2} ({cn2:.2e}), snr={snr_target}")

        errors = []

        # Validar propagation_distance
        if not isinstance(propagation_distance, (int, float)):
            errors.append("propagation_distance debe ser numérico")
        elif propagation_distance < 1 or propagation_distance > 50000:
            errors.append("propagation_distance debe estar entre 1m y 50km")

        # Validar cn2
        if not isinstance(cn2, (int, float)):
            errors.append("cn2 debe ser numérico")
        elif cn2 < 1e-18 or cn2 > 1e-11:
            errors.append("cn2 debe estar entre 1e-18 y 1e-11 m^-2/3 (log10: -18 a -11)")

        # Validar snr_target
        if not isinstance(snr_target, (int, float)):
            errors.append("snr_target debe ser numérico")
        elif snr_target < 0 or snr_target > 50:
            errors.append("snr_target debe estar entre 0 y 50 dB")

        # Validar atmospheric_conditions
        valid_conditions = ["clear", "light_turb", "heavy_turb"]
        if atmospheric_conditions not in valid_conditions:
            errors.append(f"atmospheric_conditions debe ser uno de {valid_conditions}")

        if errors:
            error_msg = "Errores en parámetros del Canal: " + "; ".join(errors)
            log_info('channel_wr', f" {error_msg}")
            raise ValueError(error_msg)

        log_debug('channel_wr', " Parameters validated correctly")

    def _calculate_channel_metrics(self, distance, cn2, snr_target):
        """
        Calcular métricas derivadas del canal para información
        """
        try:
            # Parámetro de Fried (coherence diameter)
            wavelength = 630e-9  # Asumiendo wavelength típico
            r0 = (0.423 * (2*math.pi/wavelength)**2 * cn2 * distance)**(-3/5)

            # Clasificación de turbulencia según Cn²
            if cn2 < 1e-17:
                turb_class = "Muy débil"
            elif cn2 < 1e-16:
                turb_class = "Débil"
            elif cn2 < 1e-15:
                turb_class = "Moderada"
            elif cn2 < 1e-14:
                turb_class = "Fuerte"
            else:
                turb_class = "Muy fuerte"

            # Scintillation index (simplified)
            scint_index = 1.23 * cn2 * (2*math.pi/wavelength)**(7/6) * distance**(11/6)

            # Path loss (free space)
            path_loss_db = 20 * math.log10(distance) + 20 * math.log10(4*math.pi/wavelength) / math.log10(10)

            # Log métricas
            log_info('channel_wr', f"Métricas del canal:")
            log_debug('channel_wr', f"  - Parámetro de Fried (r₀): {r0*100:.1f} cm")
            log_debug('channel_wr', f"  - Clasificación turbulencia: {turb_class}")
            log_debug('channel_wr', f"  - Índice de scintilación: {scint_index:.4f}")
            log_debug('channel_wr', f"  - Path loss (espacio libre): {path_loss_db:.1f} dB")

            # Warnings basados en métricas
            if scint_index > 1.0:
                log_warning('channel_wr', " Alto índice de scintilación - esperar fluctuaciones de señal")

            if r0 < 0.01:  # < 1 cm
                log_warning('channel_wr', " Diámetro de coherencia muy pequeño - turbulencia severa")

            if distance > 1000 and cn2 > 1e-15:
                log_warning('channel_wr', " Combinación de larga distancia y alta turbulencia - canal desafiante")

        except Exception as e:
            log_debug('channel_wr', f"Error calculando métricas del canal: {e}")

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
        channel_config = {
            'propagation_distance': current_config.get('propagation_distance'),
            'cn2': current_config.get('cn2'),
            'snr_target': current_config.get('snr_target'),
            'atmospheric_conditions': current_config.get('atmospheric_conditions')
        }
        return channel_config

# ========================================================================
# HELPER FUNCTIONS PARA GNU RADIO
# ========================================================================

def make_oam_channel_wr(propagation_distance=50, cn2=1e-15, snr_target=30, atmospheric_conditions="clear"):
    """Factory function para GNU Radio block"""
    return oam_channel_wr(propagation_distance, cn2, snr_target, atmospheric_conditions)

# ========================================================================
# CONFIGURACIONES PREESTABLECIDAS
# ========================================================================

class ChannelPresets:
    """Configuraciones preestablecidas para diferentes escenarios"""

    # Laboratorio (condiciones ideales)
    LABORATORY = {
        'propagation_distance': 5,
        'cn2': 1e-17,
        'snr_target': 35,
        'atmospheric_conditions': 'clear'
    }

    # Interior (distancia corta, buenas condiciones)
    INDOOR = {
        'propagation_distance': 20,
        'cn2': 5e-17,
        'snr_target': 32,
        'atmospheric_conditions': 'clear'
    }

    # Exterior moderado
    OUTDOOR_MODERATE = {
        'propagation_distance': 100,
        'cn2': 1e-15,
        'snr_target': 25,
        'atmospheric_conditions': 'light_turb'
    }

    # Exterior desafiante
    OUTDOOR_CHALLENGING = {
        'propagation_distance': 500,
        'cn2': 5e-15,
        'snr_target': 20,
        'atmospheric_conditions': 'heavy_turb'
    }

    # Largo alcance
    LONG_RANGE = {
        'propagation_distance': 2000,
        'cn2': 1e-14,
        'snr_target': 15,
        'atmospheric_conditions': 'heavy_turb'
    }

# ========================================================================
# TEST Y DEMO
# ========================================================================

if __name__ == "__main__":
    print("=== TEST OAM CHANNEL WRAPPER ===")

    # Test configuración estándar
    print("Test 1: Configuración estándar...")
    try:
        channel1 = oam_channel_wr()
        print(" Configuración estándar OK")
    except Exception as e:
        print(f" Error en configuración estándar: {e}")

    # Test configuración personalizada
    print("\nTest 2: Configuración personalizada...")
    try:
        channel2 = oam_channel_wr(
            propagation_distance=200,
            cn2=5e-15,
            snr_target=22,
            atmospheric_conditions="light_turb"
        )
        print(" Configuración personalizada OK")
    except Exception as e:
        print(f" Error en configuración personalizada: {e}")

    # Test presets atmosféricos
    print("\nTest 3: Presets atmosféricos...")
    try:
        # Turbulencia pesada
        channel3 = oam_channel_wr(atmospheric_conditions="heavy_turb")
        config = channel3.get_current_config()
        print(f" Preset turbulencia pesada: Cn²={config['cn2']:.2e}")
    except Exception as e:
        print(f" Error en preset: {e}")

    # Test validación de errores
    print("\nTest 4: Validación de errores...")
    try:
        channel4 = oam_channel_wr(
            propagation_distance=-10,  # Negativo - debe fallar
            cn2=1e-15,
            snr_target=30
        )
        print(" Debería haber fallado con distancia negativa")
    except ValueError:
        print(" Validación de errores funcionando correctamente")
    except Exception as e:
        print(f" Error inesperado: {e}")

    # Test configuraciones preestablecidas
    print("\nTest 5: Configuraciones preestablecidas...")
    presets_to_test = [
        ("LABORATORY", ChannelPresets.LABORATORY),
        ("OUTDOOR_MODERATE", ChannelPresets.OUTDOOR_MODERATE),
        ("LONG_RANGE", ChannelPresets.LONG_RANGE)
    ]

    for preset_name, preset_config in presets_to_test:
        try:
            channel = oam_channel_wr(**preset_config)
            print(f" Preset {preset_name} OK")
        except Exception as e:
            print(f" Error en preset {preset_name}: {e}")

    print("\n=== TEST COMPLETADO ===")