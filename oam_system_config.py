#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CONFIGURACIÓN CENTRALIZADA DEL SISTEMA OAM
===========================================

Este archivo es la FUENTE DE VERDAD para todos los parámetros del sistema.
Solo modificar los valores aquí - todo el resto del sistema se actualiza automáticamente.

INSTRUCCIONES:
1. Modificar solo los valores en SYSTEM_CONFIG
2. Ejecutar el sistema - todos los módulos usarán estos valores automáticamente
3. GNU Radio actuará solo como frontend - toda la lógica usa estos parámetros
"""

# ============================================================================
# CONFIGURACIÓN PRINCIPAL - MODIFICAR SOLO AQUÍ
# ============================================================================

SYSTEM_CONFIG = {
    # CONFIGURACION ENLACE CAMPUS (Tipo Zhao/Zhou)
    # 6 modos OAM: [-3, -2, -1, +1, +2, +3] (espaciado 1, sin modo 0)
    # 3 bits por simbolo, enlace exterior 340m

    'num_oam_modes': 6,                    # 6 modos: [-3, -2, -1, +1, +2, +3]
                                           # Sistema con 3 bits por simbolo

    'wavelength': 1550e-9,                 # λ = 1550 nm (banda C, telecom)
    'message_text': 'UIS',                 # Mensaje: UIS
                                           # 3 caracteres

    # TRANSMISOR
    'tx_power': 0.01,                      # [W] 10 mW
    'tx_aperture_size': 20e-3,             # [m] Ø 20 mm (evita recortes al lanzar l=±3)
    'tx_beam_waist': 6e-3,                 # [m] w0 = 6 mm en el plano de salida
    'symbol_rate': 32_000,                 # 32 kHz

    # CANAL ATMOSFERICO - ESCENARIO 0: SIN TURBULENCIA (BASE)
    'propagation_distance': 340,           # 340 m (enlace tipo campus)

    'cn2': 8e-15,                          # ESCENARIO 2: Turbulencia media (PRUEBA)
    'enable_turbulence': True,             # Activar turbulencia
    'enable_noise': True,                  # Activar ruido
                                           #
                                           # ESCENARIOS (cambiar cn2 y Ns para cada corrida):
                                           # 0) Sin turbulencia:    cn2=1e-17, Ns=1
                                           # 1) Turbulencia baja:   cn2=2e-15, Ns=1
                                           # 2) Turbulencia media:  cn2=8e-15, Ns=1  <- ACTUAL (PRUEBA)
                                           # 3) Turbulencia alta:   cn2=3e-14, Ns=1

    'Ns': 1,                               # Mantener Ns=1

    'snr_target': 30,                      # SNR = 30 dB (mantener)

    'atmospheric_conditions': 'outdoor',   # Condiciones exteriores

    # RECEPTOR
    'rx_aperture_size': 80e-3,             # 80 mm (capta l=±3 a 340m sin truncar)

    'grid_size': 512,                      # 512x512 (mejor resolucion para turbulencia)

    # 14-19: Control de Dashboards - TODOS ACTIVADOS
    'enable_dashboard_a': True,              # Dashboard A: Temporal analysis
    'enable_dashboard_b': True,              # Dashboard B: QA metrics
    'enable_dashboard_c': True,              # Dashboard C: Detailed snapshot
    'enable_dashboard_d': True,              # Dashboard D: Modal stream
    'enable_dashboard_e': True,              # Dashboard E: Validation metrics
    'dashboard_step_delay': 3.0,             # Tiempo entre símbolos en dashboards dinámicos [s]
}

# ============================================================================
# PROPIEDADES DERIVADAS AUTOMÁTICAS (NO MODIFICAR)
# ============================================================================

class OAMConfig:
    """Clase para acceso centralizado a configuración del sistema"""

    @staticmethod
    def get_config():
        """Obtener configuración completa"""
        return SYSTEM_CONFIG.copy()

    @staticmethod
    def get_oam_channels():
        """Generar lista de modos OAM automáticamente"""
        num_channels = SYSTEM_CONFIG['num_oam_modes']
        if num_channels % 2 != 0:
            raise ValueError(f"num_oam_modes debe ser par, obtenido: {num_channels}")

        max_mode = num_channels // 2
        # Generar modos simétricos excluyendo cero: [-max_mode, ..., -1, +1, ..., +max_mode]
        channels = list(range(-max_mode, 0)) + list(range(1, max_mode + 1))
        return channels

    @staticmethod
    def get_modes_per_symbol():
        """Obtener número de modos por símbolo"""
        return SYSTEM_CONFIG['num_oam_modes'] // 2

    @staticmethod
    def get_tx_beam_waist():
        """Calcular cintura del haz (0.5 × apertura)"""
        return SYSTEM_CONFIG['tx_aperture_size'] * 0.5

    @staticmethod
    def get_system_summary():
        """Resumen de configuración para logs"""
        channels = OAMConfig.get_oam_channels()
        return (f"OAM System: {len(channels)} modos {channels}, "
                f"{OAMConfig.get_modes_per_symbol()} modos/símbolo, "
                f"grid {SYSTEM_CONFIG['grid_size']}x{SYSTEM_CONFIG['grid_size']}")

    @staticmethod
    def get_enabled_dashboards():
        """Obtener lista de dashboards habilitados (A, B, C, D, E)"""
        enabled = []
        if SYSTEM_CONFIG['enable_dashboard_a']: enabled.append('A')
        if SYSTEM_CONFIG['enable_dashboard_b']: enabled.append('B')
        if SYSTEM_CONFIG['enable_dashboard_c']: enabled.append('C')
        if SYSTEM_CONFIG['enable_dashboard_d']: enabled.append('D')
        if SYSTEM_CONFIG['enable_dashboard_e']: enabled.append('E')
        return enabled

# ============================================================================
# FUNCIONES HELPER PARA IMPORTAR EN OTROS MÓDULOS
# ============================================================================

def get_oam_channels():
    """Función auxiliar: Obtener modos OAM"""
    return OAMConfig.get_oam_channels()

def get_modes_per_symbol():
    """Función auxiliar: Obtener modos por símbolo"""
    return OAMConfig.get_modes_per_symbol()

def get_system_config():
    """Función auxiliar: Obtener configuración completa"""
    return OAMConfig.get_config()

def get_grid_size():
    """Función auxiliar: Obtener tamaño de grilla"""
    return SYSTEM_CONFIG['grid_size']

def get_wavelength():
    """Función auxiliar: Obtener longitud de onda"""
    return SYSTEM_CONFIG['wavelength']

def get_tx_power():
    """Función auxiliar: Obtener potencia del transmisor"""
    return SYSTEM_CONFIG['tx_power']

def get_dashboard_step_delay():
    """Función auxiliar: Obtener tiempo entre símbolos en dashboards"""
    return SYSTEM_CONFIG['dashboard_step_delay']

# ============================================================================
# VALIDACIÓN DE CONFIGURACIÓN
# ============================================================================

def validate_config():
    """Validar que la configuración sea válida"""
    errors = []

    # Validar num_oam_modes
    if SYSTEM_CONFIG['num_oam_modes'] % 2 != 0:
        errors.append("num_oam_modes debe ser par")
    if SYSTEM_CONFIG['num_oam_modes'] < 2 or SYSTEM_CONFIG['num_oam_modes'] > 12:
        errors.append("num_oam_modes debe estar entre 2 y 12")

    # Validar grid_size
    valid_grids = [256, 512, 1024]
    if SYSTEM_CONFIG['grid_size'] not in valid_grids:
        errors.append(f"grid_size debe ser uno de {valid_grids}")

    # Validar wavelength
    if SYSTEM_CONFIG['wavelength'] < 400e-9 or SYSTEM_CONFIG['wavelength'] > 2000e-9:
        errors.append("wavelength debe estar entre 400nm y 2000nm")

    if errors:
        raise ValueError("Errores de configuración: " + "; ".join(errors))

# Validar configuración al importar
validate_config()

# ============================================================================
# TEST Y DEMO
# ============================================================================

if __name__ == "__main__":
    print("=== CONFIGURACIÓN DEL SISTEMA OAM ===")
    print(OAMConfig.get_system_summary())
    print()
    print("Parámetros principales:")
    for key, value in SYSTEM_CONFIG.items():
        print(f"  {key}: {value}")
    print()
    print("Propiedades derivadas:")
    print(f"  oam_channels: {get_oam_channels()}")
    print(f"  modes_per_symbol: {get_modes_per_symbol()}")
    print(f"  tx_beam_waist: {OAMConfig.get_tx_beam_waist():.3f}")
    print(f"  enabled_dashboards: {OAMConfig.get_enabled_dashboards()}")