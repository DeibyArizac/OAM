#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modulo central de configuracion de logging para el sistema OAM.
Proporciona configuracion unificada, prefijos consistentes y utilidades de control.
"""

import logging
import sys
from typing import Dict, Set, Optional

# Variables de control globales
LOG_LEVEL = logging.INFO  # INFO o DEBUG
LOG_SAMPLE_RATE = 0       # Si > 0, registrar 1 de cada N simbolos en DEBUG
LOG_HEX_MAXLEN = 32       # Longitud maxima para cadenas hex
LOG_ONCE_KEYS: Set[str] = set()  # Claves para LOG_ONCE

# Estado para evitar mensajes duplicados
_last_channel_config: Dict = {}
_last_decoder_stage: Optional[str] = None

# Prefijos de modulo unificados
PREFIXES = {
    'main': '[MAIN]',
    'source': '[OAM_SOURCE]',
    'encoder': '[OAM_ENCODER]',
    'channel': '[OAM_CHANNEL]',
    'decoder': '[GNU_RADIO_DECODER]',
    'pipeline': '[PIPELINE]',
    'lab': '[LAB]',
    'snr': '[SNR]',
    'ber': '[BER]',
    'channel_config': '[CHANNEL_CONFIG]'
}

def setup_logging(level: int = LOG_LEVEL) -> None:
    """Configurar el sistema de logging global."""
    logging.basicConfig(
        level=level,
        format='%(message)s',  # Solo el mensaje, sin timestamp
        stream=sys.stdout
    )

def get_logger(module_name: str) -> logging.Logger:
    """Obtener logger con prefijo para un modulo especifico."""
    logger = logging.getLogger(module_name)
    return logger

def log_info(module: str, message: str) -> None:
    """Registrar mensaje INFO con prefijo de modulo."""
    prefix = PREFIXES.get(module, f'[{module.upper()}]')
    logging.info(f"{prefix} {message}")

def log_warning(module: str, message: str) -> None:
    """Registrar mensaje WARNING con prefijo de modulo."""
    prefix = PREFIXES.get(module, f'[{module.upper()}]')
    logging.warning(f"{prefix} {message}")

def log_error(module: str, message: str) -> None:
    """Registrar mensaje ERROR con prefijo de modulo."""
    prefix = PREFIXES.get(module, f'[{module.upper()}]')
    logging.error(f"{prefix} {message}")

def log_debug(module: str, message: str) -> None:
    """Registrar mensaje DEBUG con prefijo de modulo."""
    prefix = PREFIXES.get(module, f'[{module.upper()}]')
    logging.debug(f"{prefix} {message}")

def log_debug_sampled(module: str, message: str, index: int) -> None:
    """Registrar mensaje DEBUG con muestreo basado en LOG_SAMPLE_RATE."""
    if LOG_SAMPLE_RATE <= 0:
        return
    if index % LOG_SAMPLE_RATE == 0:
        log_debug(module, message)

def log_once(module: str, key: str, message: str, level: str = 'info') -> None:
    """Registrar mensaje solo la primera vez para una clave dada."""
    full_key = f"{module}:{key}"
    if full_key not in LOG_ONCE_KEYS:
        LOG_ONCE_KEYS.add(full_key)
        if level == 'info':
            log_info(module, message)
        elif level == 'warning':
            log_warning(module, message)
        elif level == 'error':
            log_error(module, message)
        elif level == 'debug':
            log_debug(module, message)

def truncate_hex(hex_string: str) -> str:
    """Truncar cadena hexadecimal a LOG_HEX_MAXLEN caracteres."""
    if len(hex_string) <= LOG_HEX_MAXLEN:
        return hex_string
    return hex_string[:LOG_HEX_MAXLEN] + "... (truncado)"

def should_log_channel_config(cn2: float, snr_db: float, ns: int, turbulence: bool) -> bool:
    """Verificar si se debe registrar configuracion del canal (solo cuando cambie)."""
    global _last_channel_config
    current_config = {
        'cn2': cn2,
        'snr_db': snr_db,
        'ns': ns,
        'turbulence': turbulence
    }

    if current_config != _last_channel_config:
        _last_channel_config = current_config.copy()
        return True
    return False

def should_log_decoder_stage(stage: str) -> bool:
    """Verificar si se debe registrar cambio de estado del decoder."""
    global _last_decoder_stage
    if stage != _last_decoder_stage:
        _last_decoder_stage = stage
        return True
    return False

def format_byte_list(byte_list: list, max_items: int = 10) -> str:
    """Formatear lista de bytes para logging, truncando si es necesario."""
    if len(byte_list) <= max_items:
        return str(byte_list)
    return str(byte_list[:max_items]) + f"... (+{len(byte_list)-max_items} mas)"

def format_modes_list(modes: list) -> str:
    """Formatear lista de modos OAM para logging."""
    return "[" + ",".join([f"{m:+d}" if m >= 0 else str(m) for m in modes]) + "]"

def format_speed(bps: float) -> str:
    """Formatear velocidad en formato legible."""
    if bps >= 1e9:
        return f"{bps/1e9:.3f} Gb/s"
    elif bps >= 1e6:
        return f"{bps/1e6:.3f} Mb/s"
    elif bps >= 1e3:
        return f"{bps/1e3:.3f} kb/s"
    else:
        return f"{bps:.1f} bps"

# Configurar logging al importar el modulo
setup_logging()