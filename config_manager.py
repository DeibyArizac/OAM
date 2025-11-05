#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MANEJADOR DE CONFIGURACIÓN COMPARTIDA PARA BLOQUES GNU RADIO OAM
================================================================

Este módulo maneja la configuración compartida entre los 5 bloques GNU Radio:
- OAM Source (ejecuta todo el pipeline)
- OAM Encoder (solo edita parámetros)
- OAM Channel (solo edita parámetros)
- OAM Decoder (solo edita parámetros)
- OAM Visualizer (solo edita parámetros)

Arquitectura:
- Archivo temporal JSON para comunicación entre bloques
- Base: SYSTEM_CONFIG de oam_system_config.py
- Sincronización bidireccional: JSON ↔ oam_system_config.py
- Solo OAM Source ejecuta el pipeline completo

Flujo de Configuración:
1. GNU Radio GUI → Bloques Wrapper → config_manager.update_*()
2. config_manager guarda en JSON temporal
3. OAM Source lee JSON y ejecuta pipeline
4. Dashboards usan oam_system_config.py actualizado
"""

import json
import tempfile
import os
import logging
from pathlib import Path
import importlib

# Importar configuración base del sistema
try:
    from .oam_system_config import SYSTEM_CONFIG, OAMConfig
    import oam_system_config
except ImportError:
    from oam_system_config import SYSTEM_CONFIG, OAMConfig
    import oam_system_config

class OAMConfigManager:
    """Manejador centralizado de configuración para bloques GNU Radio"""

    def __init__(self):
        # Archivo temporal para compartir config entre bloques
        self.temp_dir = Path(tempfile.gettempdir()) / "oam_gnuradio"
        self.temp_dir.mkdir(exist_ok=True)
        self.config_file = self.temp_dir / "oam_config.json"

        # Configuración base desde oam_system_config.py
        self.config = SYSTEM_CONFIG.copy()

        # Logging
        self.logger = logging.getLogger(__name__)

        # Cargar configuración existente si existe
        self._load_config()

    # ========================================================================
    # MÉTODOS DE ACTUALIZACIÓN POR BLOQUE
    # ========================================================================

    def update_source_params(self, **kwargs):
        """Actualizado por OAM Source block"""
        self.logger.info(f"OAM Source updating params: {kwargs}")
        self.config.update(kwargs)
        self._save_config()

    def update_encoder_params(self, **kwargs):
        """Actualizado por OAM Encoder block"""
        self.logger.info(f"OAM Encoder updating params: {kwargs}")

        # Validar parámetros del encoder
        encoder_params = {}
        if 'num_oam_modes' in kwargs:
            encoder_params['num_oam_modes'] = int(kwargs['num_oam_modes'])
        if 'wavelength' in kwargs:
            encoder_params['wavelength'] = float(kwargs['wavelength'])
        if 'tx_power' in kwargs:
            encoder_params['tx_power'] = float(kwargs['tx_power'])
        if 'tx_aperture_size' in kwargs:
            encoder_params['tx_aperture_size'] = float(kwargs['tx_aperture_size'])
        if 'grid_size' in kwargs:
            encoder_params['grid_size'] = int(kwargs['grid_size'])

        self.config.update(encoder_params)
        self._save_config()

    def update_channel_params(self, **kwargs):
        """Actualizado por OAM Channel block"""
        self.logger.info(f"OAM Channel updating params: {kwargs}")

        # Validar parámetros del canal
        channel_params = {}
        if 'propagation_distance' in kwargs:
            channel_params['propagation_distance'] = float(kwargs['propagation_distance'])
        if 'cn2' in kwargs:
            channel_params['cn2'] = float(kwargs['cn2'])
        if 'snr_target' in kwargs:
            channel_params['snr_target'] = float(kwargs['snr_target'])
        if 'atmospheric_conditions' in kwargs:
            channel_params['atmospheric_conditions'] = str(kwargs['atmospheric_conditions'])

        self.config.update(channel_params)
        self._save_config()

    def update_decoder_params(self, **kwargs):
        """Actualizado por OAM Decoder block"""
        self.logger.info(f"OAM Decoder updating params: {kwargs}")

        # Validar parámetros del decoder
        decoder_params = {}
        if 'rx_aperture_size' in kwargs:
            decoder_params['rx_aperture_size'] = float(kwargs['rx_aperture_size'])
        if 'grid_size' in kwargs:
            decoder_params['grid_size'] = int(kwargs['grid_size'])

        self.config.update(decoder_params)
        self._save_config()

    def update_visualizer_params(self, **kwargs):
        """Actualizado por OAM Visualizer block"""
        self.logger.info(f"OAM Visualizer updating params: {kwargs}")

        # Validar parámetros del visualizer
        visualizer_params = {}
        if 'enable_dashboard_a' in kwargs:
            visualizer_params['enable_dashboard_a'] = bool(kwargs['enable_dashboard_a'])
        if 'enable_dashboard_b' in kwargs:
            visualizer_params['enable_dashboard_b'] = bool(kwargs['enable_dashboard_b'])
        if 'enable_dashboard_c' in kwargs:
            visualizer_params['enable_dashboard_c'] = bool(kwargs['enable_dashboard_c'])
        if 'enable_dashboard_d' in kwargs:
            visualizer_params['enable_dashboard_d'] = bool(kwargs['enable_dashboard_d'])
        if 'dashboard_step_delay' in kwargs:
            visualizer_params['dashboard_step_delay'] = float(kwargs['dashboard_step_delay'])

        self.config.update(visualizer_params)
        self._save_config()

    # ========================================================================
    # MÉTODOS DE ACCESO
    # ========================================================================

    def get_full_config(self):
        """
        Solo OAM Source usa esto para ejecutar pipeline
        Retorna configuración completa y actualizada
        """
        self._load_config()

        # Log configuración final (solo en debug)
        self.logger.debug("=== CONFIGURACIÓN FINAL PARA PIPELINE ===")
        for key, value in self.config.items():
            self.logger.debug(f"  {key}: {value}")

        return self.config.copy()

    def get_config_summary(self):
        """Resumen de configuración para logs"""
        self._load_config()
        try:
            channels = list(range(-self.config['num_oam_modes']//2, 0)) + \
                      list(range(1, self.config['num_oam_modes']//2 + 1))
            return (f"OAM Config: {len(channels)} modos {channels}, "
                   f"grid {self.config['grid_size']}x{self.config['grid_size']}, "
                   f"distancia {self.config['propagation_distance']}m")
        except:
            return "OAM Config: Error en resumen"

    # ========================================================================
    # MÉTODOS INTERNOS
    # ========================================================================

    def _save_config(self):
        """Guardar configuración a archivo temporal Y sincronizar con oam_system_config.py"""
        try:
            # Guardar en JSON temporal
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.debug(f"Configuración guardada en {self.config_file}")

            # Sincronizar con oam_system_config.py
            self._sync_to_system_config()

        except Exception as e:
            self.logger.error(f"Error guardando configuración: {e}")

    def _sync_to_system_config(self):
        """
        Sincronizar configuración actual con oam_system_config.py
        Actualiza el diccionario SYSTEM_CONFIG en memoria
        """
        try:
            # Actualizar SYSTEM_CONFIG en memoria (para módulos ya importados)
            oam_system_config.SYSTEM_CONFIG.update(self.config)
            self.logger.debug("SYSTEM_CONFIG sincronizado en memoria")
        except Exception as e:
            self.logger.warning(f"No se pudo sincronizar SYSTEM_CONFIG: {e}")

    def _load_config(self):
        """Cargar configuración desde archivo temporal"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    saved_config = json.load(f)
                self.config.update(saved_config)
                self.logger.debug(f"Configuración cargada desde {self.config_file}")
            except Exception as e:
                self.logger.error(f"Error cargando configuración: {e}")

    def reset_to_default(self):
        """Resetear a configuración por defecto"""
        self.config = SYSTEM_CONFIG.copy()
        self._save_config()
        self.logger.info("Configuración reseteada a valores por defecto")

    def validate_config(self):
        """Validar que la configuración sea válida"""
        errors = []

        # Validar num_oam_modes
        if self.config['num_oam_modes'] % 2 != 0:
            errors.append("num_oam_modes debe ser par")
        if self.config['num_oam_modes'] < 2 or self.config['num_oam_modes'] > 12:
            errors.append("num_oam_modes debe estar entre 2 y 12")

        # Validar grid_size
        valid_grids = [256, 512, 1024]
        if self.config['grid_size'] not in valid_grids:
            errors.append(f"grid_size debe ser uno de {valid_grids}")

        # Validar wavelength
        if self.config['wavelength'] < 400e-9 or self.config['wavelength'] > 2000e-9:
            errors.append("wavelength debe estar entre 400nm y 2000nm")

        if errors:
            raise ValueError("Errores de configuración: " + "; ".join(errors))

        return True

# ========================================================================
# INSTANCIA GLOBAL
# ========================================================================

# Instancia global compartida entre todos los bloques
config_manager = OAMConfigManager()

# ========================================================================
# FUNCIONES HELPER PARA BLOQUES
# ========================================================================

def get_current_config():
    """Función auxiliar: Obtener configuración actual completa"""
    return config_manager.get_full_config()

def update_source_config(**kwargs):
    """Función auxiliar: Actualizar desde Source block"""
    config_manager.update_source_params(**kwargs)

def update_encoder_config(**kwargs):
    """Función auxiliar: Actualizar desde Encoder block"""
    config_manager.update_encoder_params(**kwargs)

def update_channel_config(**kwargs):
    """Función auxiliar: Actualizar desde Channel block"""
    config_manager.update_channel_params(**kwargs)

def update_decoder_config(**kwargs):
    """Función auxiliar: Actualizar desde Decoder block"""
    config_manager.update_decoder_params(**kwargs)

def update_visualizer_config(**kwargs):
    """Función auxiliar: Actualizar desde Visualizer block"""
    config_manager.update_visualizer_params(**kwargs)

# ========================================================================
# TEST Y DEMO
# ========================================================================

if __name__ == "__main__":
    print("=== TEST CONFIG MANAGER ===")

    # Test básico
    print("Configuración inicial:")
    print(config_manager.get_config_summary())

    # Test updates
    print("\nTest updates...")
    config_manager.update_encoder_params(num_oam_modes=8, wavelength=532e-9)
    config_manager.update_channel_params(propagation_distance=100, snr_target=25)
    config_manager.update_visualizer_params(enable_dashboard_a=False, dashboard_step_delay=2.0)

    # Ver configuración final
    print("\nConfiguración final:")
    final_config = config_manager.get_full_config()
    for key, value in final_config.items():
        print(f"  {key}: {value}")

    # Test validación
    print("\nTest validación...")
    try:
        config_manager.validate_config()
        print(" Configuración válida")
    except ValueError as e:
        print(f" Error de configuración: {e}")