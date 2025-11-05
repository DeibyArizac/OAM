#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo: oam_encoder.py
Propósito: Codificación de mensajes a símbolos OAM mediante mapeo directo bit-a-modo.

Este módulo implementa el encoder del sistema de comunicación óptica basado en OAM,
realizando la conversión de bits de información a modos de momento angular orbital.
Soporta configuraciones de 2, 4, 6 y 8 modos OAM con detección automática de
parámetros y generación de haces Laguerre-Gaussianos.

Desarrollo: Codificación y Decodificación a partir del Momento Angular Orbital de la Luz en GNU Radio

Autor: Deiby Fernando Ariza Cadena
Director: Dr. Omar Javier Tíjaro Rojas (ojtijaro@uis.edu.co)
Institución: Escuela de Ingenierías Eléctrica, Electrónica y de Telecomunicaciones (E3T)
             Universidad Industrial de Santander
"""

import numpy as np
from gnuradio import gr
from scipy import special
from dataclasses import dataclass
from typing import List
from pipeline import pipeline

# Configuración centralizada del sistema
from oam_system_config import get_system_config, OAMConfig

# Sistema de logging unificado
from oam_logging import log_info, log_warning, log_error, log_debug

@dataclass
class SystemConfig:
    """
    Configuracion de parametros del sistema OAM.

    Parametros:
        grid_size: Tamano de la grilla de simulacion.
        wavelength: Longitud de onda en metros.
        propagation_distance: Distancia de propagacion.
        tx_aperture_size: Tamano de apertura del transmisor.
        tx_beam_waist: Cintura del haz transmisor.
        oam_channels: Lista de modos OAM a utilizar.
        modulation: Tipo de modulacion (OAMPSK por defecto).
        symbol_rate: Velocidad de simbolos.

    Notas:
        - Validacion automatica de parametros fisicos.
        - Configuracion por defecto usa 8 modos simetricos sin modo 0.
    """
    grid_size: int = None
    wavelength: float = None
    propagation_distance: float = None
    tx_aperture_size: float = None
    tx_beam_waist: float = None
    oam_channels: List[int] = None
    modulation: str = 'OAMPSK'
    symbol_rate: float = None

    def __post_init__(self):
        # Obtener configuración centralizada para valores None
        config = get_system_config()

        if self.grid_size is None:
            self.grid_size = config['grid_size']
        if self.wavelength is None:
            self.wavelength = config['wavelength']
        if self.propagation_distance is None:
            self.propagation_distance = config['propagation_distance']
        if self.tx_aperture_size is None:
            self.tx_aperture_size = config['tx_aperture_size']
        if self.tx_beam_waist is None:
            self.tx_beam_waist = OAMConfig.get_tx_beam_waist()
        if self.symbol_rate is None:
            self.symbol_rate = config['symbol_rate']
        if self.oam_channels is None:
            self.oam_channels = OAMConfig.get_oam_channels()

        # Validacion de parametros fisicos
        self._validate_physical_parameters()

    @classmethod
    def get_default_oam_channels(cls):
        """FUENTE DE VERDAD: Obtener modos OAM por defecto del sistema"""
        return OAMConfig.get_oam_channels()

    @classmethod
    def get_default_modes_per_symbol(cls):
        """Obtener número de modos por símbolo basado en canales por defecto"""
        channels = cls.get_default_oam_channels()
        magnitudes = sorted(list(set([abs(l) for l in channels if l != 0])))
        return len(magnitudes)

    @classmethod
    def get_default_system_config(cls):
        """Obtener configuración completa del sistema por defecto"""
        return {
            'grid_size': 512,
            'wavelength': 630e-9,
            'tx_aperture_size': 35e-3,
            'tx_beam_waist': 17.5e-3,
            'oam_channels': cls.get_default_oam_channels(),
            'propagation_distance': 50,
            'cn2': 1e-15,
            'snr_db': 30,
            'modulation': 'OAMPSK'
        }

    @classmethod
    def from_max_mode(cls, max_mode: int, include_zero: bool = False, **kwargs):
        """
        Crea configuracion especificando el modo maximo.

        Parametros:
            max_mode: Valor absoluto maximo del modo OAM.
            include_zero: Si incluir el modo 0.

        Retorna:
            Instancia de SystemConfig con modos simetricos.
        """
        channels = []
        for l in range(-max_mode, max_mode + 1):
            if l == 0 and not include_zero:
                continue
            channels.append(l)
        return cls(oam_channels=channels, **kwargs)

    @classmethod
    def from_num_modes(cls, num_modes: int, **kwargs):
        """
        Crea configuracion especificando el numero total de modos.

        Parametros:
            num_modes: Numero total de modos (debe ser par).

        Retorna:
            Instancia de SystemConfig.

        Notas:
            - Requiere numero par para modos simetricos sin modo 0.
        """
        if num_modes <= 0:
            raise ValueError("num_modes debe ser > 0 y par (modos simetricos ±l)")
        if num_modes % 2 != 0:
            raise ValueError("num_modes debe ser par (modos +/- simetricos sin modo 0)")

        max_mode = num_modes // 2
        channels = []
        for l in range(-max_mode, max_mode + 1):
            if l != 0:  # Excluir modo 0
                channels.append(l)

        return cls(oam_channels=channels, **kwargs)

    def _validate_physical_parameters(self):
        """Complete parameter validation"""

        # Validar grid_size
        if not isinstance(self.grid_size, int) or self.grid_size <= 0:
            raise ValueError(f"grid_size must be a positive integer, got: {self.grid_size}")
        if self.grid_size < 64 or self.grid_size > 2048:
            raise ValueError(f"grid_size must be between 64 and 2048, got: {self.grid_size}")
        if not (self.grid_size & (self.grid_size - 1)) == 0:
            raise ValueError(f"grid_size must be power of 2, got: {self.grid_size}")

        # Validar longitud de onda (visible/infrarrojo cercano)
        if not isinstance(self.wavelength, (int, float)) or self.wavelength <= 0:
            raise ValueError(f"wavelength must be a positive number, got: {self.wavelength}")
        if self.wavelength < 400e-9 or self.wavelength > 2000e-9:
            raise ValueError(
                f"wavelength must be between 400nm and 2000nm, got: {self.wavelength*1e9:.0f}nm"
            )

        # Validar distancia de propagación
        if not isinstance(self.propagation_distance, (int, float)) or self.propagation_distance <= 0:
            raise ValueError(
                f"propagation_distance must be positive, got: {self.propagation_distance}"
            )
        if self.propagation_distance < 1 or self.propagation_distance > 1e6:
            raise ValueError(
                f"propagation_distance must be between 1m and 1Mm, got: {self.propagation_distance}m"
            )

        # Validar tamaño de apertura del transmisor
        if not isinstance(self.tx_aperture_size, (int, float)) or self.tx_aperture_size <= 0:
            raise ValueError(
                f"tx_aperture_size must be positive, got: {self.tx_aperture_size}"
            )
        if self.tx_aperture_size < 1e-3 or self.tx_aperture_size > 1:
            raise ValueError(
                f"tx_aperture_size must be between 1mm and 1m, got: {self.tx_aperture_size*1000:.1f}mm"
            )

        # Validar cintura del haz
        if not isinstance(self.tx_beam_waist, (int, float)) or self.tx_beam_waist <= 0:
            raise ValueError(f"tx_beam_waist must be positive, got: {self.tx_beam_waist}")
        if self.tx_beam_waist < 0.1e-3 or self.tx_beam_waist > 0.1:
            raise ValueError(
                f"tx_beam_waist must be between 0.1mm and 100mm, got: {self.tx_beam_waist*1000:.1f}mm"
            )

        # Validar que beam_waist sea menor que aperture_size
        if self.tx_beam_waist >= self.tx_aperture_size:
            raise ValueError(
                f"tx_beam_waist ({self.tx_beam_waist*1000:.1f}mm) must be smaller than "
                f"tx_aperture_size ({self.tx_aperture_size*1000:.1f}mm)"
            )

        # Validar modos OAM
        if not isinstance(self.oam_channels, list) or len(self.oam_channels) == 0:
            raise ValueError(
                f"oam_channels must be a non-empty list, got: {self.oam_channels}"
            )
        for channel in self.oam_channels:
            if not isinstance(channel, int):
                raise ValueError(f"All OAM channels must be integers, found: {channel}")
            if abs(channel) > 10:
                raise ValueError(f"OAM channel |l| > 10 is not practical, found: l={channel}")

        # Verificar canales únicos
        if len(self.oam_channels) != len(set(self.oam_channels)):
            raise ValueError(f"oam_channels contains duplicates: {self.oam_channels}")

        # Validar esquema de modulación
        valid_modulations = ['OAMPSK']
        if self.modulation not in valid_modulations:
            raise ValueError(
                f"modulation must be one of {valid_modulations}, got: {self.modulation}"
            )

        # Validar tasa de símbolos
        if not isinstance(self.symbol_rate, (int, float)) or self.symbol_rate <= 0:
            raise ValueError(f"symbol_rate must be positive, got: {self.symbol_rate}")
        if self.symbol_rate < 1.0 or self.symbol_rate > 1e6:
            raise ValueError(
                f"symbol_rate must be between 1Hz and 1MHz for SLM/camera hardware, "
                f"got: {self.symbol_rate:.1f}Hz"
            )

class oam_encoder(gr.sync_block):
    """
    OAM Encoder Block - Complete OAM encoding implementation
    Laguerre-Gaussian beam generation and data modulation
    """
    def __init__(self, grid_size=None, wavelength=None, tx_aperture_size=None, tx_beam_waist=None, oam_channels=None):
        # Obtener configuración centralizada para valores None
        config = get_system_config()

        if grid_size is None:
            grid_size = config['grid_size']
        if wavelength is None:
            wavelength = config['wavelength']
        if tx_aperture_size is None:
            tx_aperture_size = config['tx_aperture_size']
        if tx_beam_waist is None:
            tx_beam_waist = OAMConfig.get_tx_beam_waist()
        if oam_channels is None:
            oam_channels = OAMConfig.get_oam_channels()

        # ALTERNATIVE APPROACH: Use smaller chunks instead of full frame
        # GNU Radio tiene problemas con vectores grandes, dividir en fragmentos manejables
        total_elements = grid_size * grid_size  # 65536 for 256x256

        # ENFOQUE PIPELINE: Usar GNU Radio como frontend, vectores Python como backend
        self.total_elements = total_elements

        gr.sync_block.__init__(self,
            name="oam_encoder",
            in_sig=[np.uint8],  # Input: byte stream
            out_sig=[np.uint8])  # Output: trigger signal (not actual data)

        # PIPELINE state management - store results in memory for next block
        self.encoded_symbols = []  # Complete symbols stored here
        self.processing_complete = False

        log_info('encoder', "Usando enfoque PIPELINE: GNU Radio frontend + Python backend")

        # Create system configuration from parameters
        if isinstance(oam_channels, str):
            oam_channels = eval(oam_channels)  # Convert string to list if needed

        self.config = SystemConfig(
            grid_size=int(grid_size),
            wavelength=float(wavelength),
            tx_aperture_size=float(tx_aperture_size),
            tx_beam_waist=float(tx_beam_waist),
            oam_channels=list(oam_channels),
            modulation='OAMPSK'
        )

        # Initialize OAM encoder with system parameters
        self.modulations = {
            'OAMPSK': self._modulate_oampsk
        }
        self.setup_grids()
        self.generate_modes()

        # HYBRID STREAMING specific state
        # Variables will be created dynamically in work() method

        log_info('encoder', "Bloque inicializado correctamente")
        log_info('encoder', f"SystemConfig: {self.config.grid_size}x{self.config.grid_size}")
        log_info('encoder', f"Modos OAM: {self.config.oam_channels}")
        log_info('encoder', f"Modulación: {self.config.modulation}")
        log_info('encoder', "Modo: HYBRID STREAMING (estructura de codificación + transmisión streaming)")
        log_info('encoder', "Configuración completa")

    def setup_grids(self):
        """Configure spatial grids"""
        self.x_tx = np.linspace(
            -self.config.tx_aperture_size/2,
            self.config.tx_aperture_size/2,
            self.config.grid_size
        )
        self.y_tx = np.linspace(
            -self.config.tx_aperture_size/2,
            self.config.tx_aperture_size/2,
            self.config.grid_size
        )
        self.X_tx, self.Y_tx = np.meshgrid(self.x_tx, self.y_tx)

    def generate_modes(self):
        """Generate all OAM modes"""
        self.modes = {}
        for l in self.config.oam_channels:
            m = self.generate_laguerre_gaussian(l, 0)
            # Normalize to power 1
            p = np.sqrt(np.sum(np.abs(m)**2))
            if p > 0:
                m = m / p
            final_power = np.mean(np.abs(m)**2)
            max_val = np.max(np.abs(m))
            self.modes[l] = m
            # log_debug('encoder', f"Modo l={l} generado: potencia_media={final_power:.6f}, max={max_val:.6f}")

    def generate_laguerre_gaussian(self, l, p, z=0):
        """Generate Laguerre-Gaussian mode"""
        k = 2 * np.pi / self.config.wavelength
        zR = np.pi * self.config.tx_beam_waist**2 / self.config.wavelength

        if z == 0:
            w_z = self.config.tx_beam_waist
            R_z = np.inf
            gouy_phase = 0
        else:
            w_z = self.config.tx_beam_waist * np.sqrt(1 + (z/zR)**2)
            R_z = z * (1 + (zR/z)**2)
            gouy_phase = (abs(l) + 2*p + 1) * np.arctan(z/zR)

        r = np.sqrt(self.X_tx**2 + self.Y_tx**2)
        phi = np.arctan2(self.Y_tx, self.X_tx)
        norm = np.sqrt(2 * special.factorial(p) / (np.pi * special.factorial(p + abs(l))))
        rho = np.sqrt(2) * r / w_z
        radial = rho**abs(l) * np.exp(-rho**2/2)
        laguerre = special.eval_genlaguerre(p, abs(l), rho**2)

        if np.isfinite(R_z):
            curvature = np.exp(1j * k * r**2 / (2 * R_z))
        else:
            curvature = 1

        azimuthal = np.exp(1j * l * phi)
        gouy = np.exp(-1j * gouy_phase)

        return norm * radial * laguerre * curvature * azimuthal * gouy / w_z

    def _setup_wide_grids(self):
        """Configurar grillas más amplias para diagnóstico"""
        self.N_diag = self.config.grid_size  # Usar resolución configurada
        aperture_wide = self.config.tx_aperture_size * 2  # Doble apertura para ver más contenido

        self.x_wide = np.linspace(-aperture_wide/2, aperture_wide/2, self.N_diag)
        self.y_wide = np.linspace(-aperture_wide/2, aperture_wide/2, self.N_diag)
        self.X_wide, self.Y_wide = np.meshgrid(self.x_wide, self.y_wide)

    def _generate_laguerre_gaussian_wide(self, l, p, z=0):
        """Generar modo Laguerre-Gaussian en grilla amplia"""
        k = 2 * np.pi / self.config.wavelength
        zR = np.pi * self.config.tx_beam_waist**2 / self.config.wavelength

        if z == 0:
            w_z = self.config.tx_beam_waist
            R_z = np.inf
            gouy_phase = 0
        else:
            w_z = self.config.tx_beam_waist * np.sqrt(1 + (z/zR)**2)
            R_z = z * (1 + (zR/z)**2)
            gouy_phase = (abs(l) + 2*p + 1) * np.arctan(z/zR)

        r = np.sqrt(self.X_wide**2 + self.Y_wide**2)
        phi = np.arctan2(self.Y_wide, self.X_wide)
        norm = np.sqrt(2 * special.factorial(p) / (np.pi * special.factorial(p + abs(l))))
        rho = np.sqrt(2) * r / w_z
        radial = rho**abs(l) * np.exp(-rho**2/2)
        laguerre = special.eval_genlaguerre(p, abs(l), rho**2)

        if np.isfinite(R_z):
            curvature = np.exp(1j * k * r**2 / (2 * R_z))
        else:
            curvature = 1

        azimuthal = np.exp(1j * l * phi)
        gouy = np.exp(-1j * gouy_phase)

        # Normalizar igual que el modo regular
        mode_wide = norm * radial * laguerre * curvature * azimuthal * gouy / w_z
        p_wide = np.sqrt(np.sum(np.abs(mode_wide)**2))
        if p_wide > 0:
            mode_wide = mode_wide / p_wide

        return mode_wide

    # === API: Flujo directo ASCII a bits ===

    def bytes_from_ascii(self, msg: str):
        """
        Convierte cadena ASCII a lista de valores byte.

        Parametros:
            msg: Cadena de texto ASCII.

        Retorna:
            Lista de valores enteros (0-255).
        """
        return [ord(c) for c in msg]

    def bitstream_from_bytes(self, bytes_list):
        """
        Convierte bytes a flujo de bits en orden LSB→MSB (LSB-first).

        Parametros:
            bytes_list: Lista de valores byte.

        Retorna:
            Lista de bits (0 o 1) en orden LSB primero (derecha a izquierda).

        Ejemplo:
            byte 0x4D = 0b01001101
                        76543210 (posiciones)
            LSB-first: [bit0, bit1, bit2, ..., bit7] = [1,0,1,1,0,0,1,0]
        """
        bits = []
        for b in bytes_list:
            # LSB primero: bit 0, 1, 2, 3, 4, 5, 6, 7
            for i in range(0, 8):
                bits.append((b >> i) & 1)
        return bits

    def symbols_from_bits(self, bits, modes_per_symbol: int):
        """
        Segmenta flujo de bits en bloques de modes_per_symbol.

        Parametros:
            bits: Lista de bits.
            modes_per_symbol: Numero de bits por simbolo.

        Retorna:
            Lista de simbolos, cada simbolo es lista de bits.

        Notas:
            - Rellena con ceros si el ultimo simbolo es incompleto.
        """
        symbols = []
        for i in range(0, len(bits), modes_per_symbol):
            symbol_bits = bits[i:i+modes_per_symbol]
            # Rellenar con ceros si es incompleto
            while len(symbol_bits) < modes_per_symbol:
                symbol_bits.append(0)
            symbols.append(symbol_bits)
        return symbols

    def modes_from_symbol_bits(self, sym_bits):
        """
        Convierte bits de simbolo a modos OAM.

        Parametros:
            sym_bits: Lista de bits del simbolo.

        Retorna:
            Lista de modos OAM correspondientes.

        Notas:
            - Mapeo dinámico basado en configuración: magnitudes desde oam_channels
            - Signo por bit: 0 → negativo, 1 → positivo.
        """
        modes = []
        # Obtener magnitudes desde configuración (no hardcodeado)
        magnitudes = sorted(list(set([abs(l) for l in self.config.oam_channels if l != 0])))

        for i, bit in enumerate(sym_bits):
            if i < len(magnitudes):
                magnitude = magnitudes[i]
                sign = 1 if bit == 1 else -1
                modes.append(sign * magnitude)

        return modes

    def encode(self, data, modulation=None, modes_per_symbol=None):
        """
        Codifica datos a modos OAM usando flujo directo de bits.

        Parametros:
            data: Datos binarios a codificar.
            modulation: Tipo de modulacion (opcional).
            modes_per_symbol: Numero de modos por simbolo (1-4).

        Retorna:
            Tupla (campo_simbolos, campo_amplio) con arrays complejos.

        Notas:
            - Agrega marcadores STX/ETX automaticamente.
            - Soporta M=1,2,3,4 modos por simbolo.
            - Genera símbolos de preámbulo y piloto adicionales.
        """
        if modulation is None:
            modulation = self.config.modulation
        
        # Determinar numero de modos por simbolo desde modos OAM disponibles
        magnitudes = sorted(list(set([abs(l) for l in self.config.oam_channels if l != 0])))
        if modes_per_symbol is None:
            modes_per_symbol = len(magnitudes)  # Por defecto: usar todos los modos disponibles

        # Configuración de modos por símbolo determinada

        # Agregar marcadores STX/ETX
        STX = bytes([0x02])  # Inicio de texto
        ETX = bytes([0x03])  # Fin de texto
        data_with_markers = STX + data + ETX

        # Datos de entrada preparados con marcadores STX/ETX
        
        # Flujo directo de bits
        from pipeline import pipeline
        
        # Convertir datos a bytes
        pipeline.original_bytes = list(data_with_markers)
        
        # Convertir bytes a flujo de bits (MSB primero)
        bitstream = self.bitstream_from_bytes(pipeline.original_bytes)
        
        # Segmentar flujo de bits en simbolos de M bits cada uno
        symbol_bits_list = self.symbols_from_bits(bitstream, modes_per_symbol)

        # === BUILD SYMBOLS WITH LABELS ===
        symbols_data = []
        pipeline.symbol_metadata = []
        
        # 1) FRAME DE PREÁMBULO
        dummy_bits = [0] * modes_per_symbol
        symbols_data.append({
            "kind": "piloto_cero",
            "bits": dummy_bits.copy(),
            "byte_index": None
        })

        # 2) PILOT SIGN FRAME
        pilot_bits = [1] * modes_per_symbol
        symbols_data.append({
            "kind": "piloto_signo",
            "bits": pilot_bits.copy(),
            "byte_index": None
        })

        # 3) DATA FRAMES (using direct bit flow)
        for sym_bits in symbol_bits_list:
            symbols_data.append({
                "kind": "datos",
                "bits": sym_bits.copy(),
                "byte_index": None  # Could track which byte this came from if needed
            })

        # Build frame per symbol, summing all modes simultaneously
        n_symbols = len(symbols_data)
        field = np.zeros((n_symbols, self.config.grid_size, self.config.grid_size), dtype=complex)

        # DIAGNÓSTICO: Generar versión "wide" para visualización
        self._setup_wide_grids()
        field_wide = np.zeros((n_symbols, self.N_diag, self.N_diag), dtype=complex)

        # Generacion de campo fisico para cada simbolo
        for s, sym_data in enumerate(symbols_data):
            frame = np.zeros_like(self.modes[magnitudes[0]], dtype=complex)
            frame_wide = np.zeros((self.N_diag, self.N_diag), dtype=complex)
            bits = sym_data['bits']

            # Generate modes from symbol bits using new API
            modes_list = self.modes_from_symbol_bits(bits)

            # Importante: Guardar modos OAM en metadata para Dashboard D
            symbols_data[s]['oam_modes'] = modes_list.copy()

            # Usar solo los primeros modes_per_symbol modos
            for j in range(min(len(modes_list), modes_per_symbol)):
                l_use = modes_list[j]
                if l_use in self.modes:  # Check if mode is available
                    frame += self.modes[l_use]
                    frame_wide += self._generate_laguerre_gaussian_wide(l_use, 0)

            # Normalize by number of active modes
            active_modes = min(len(modes_list), modes_per_symbol)
            if active_modes > 0:
                frame /= np.sqrt(active_modes)
                frame_wide /= np.sqrt(active_modes)

            # NORMALIZACION A TX_POWER (en lugar de 1.0 W)
            # Garantiza que np.sum(|frame|^2) = tx_power (e.g., 0.01 W = 10 mW)
            tx_power = get_system_config()['tx_power']
            power_current = np.sum(np.abs(frame)**2)
            if power_current > 0:
                frame = frame * np.sqrt(tx_power / power_current)
            power_current_wide = np.sum(np.abs(frame_wide)**2)
            if power_current_wide > 0:
                frame_wide = frame_wide * np.sqrt(tx_power / power_current_wide)

            field[s] = frame
            field_wide[s] = frame_wide
            
            bits_str = ''.join(str(b) for b in bits)
            modes_str = [f"{m:+d}" for m in modes_list[:modes_per_symbol]]
            log_debug('encoder', f"Símbolo {s} ({sym_data['kind']}): [{bits_str}] -> modos: {modes_str}")

        # Save wide version in pipeline for visualization
        pipeline.field_wide = field_wide
        pipeline.field_tx = field.copy()
        # Guardar cada símbolo individualmente para consistencia de formas
        for i, symbol_field in enumerate(field):
            pipeline.push_field("before_channel", symbol_field.copy())
        # Símbolos guardados en pipeline

        # Importante: Guardar metadatos de simbolos en pipeline para persistencia
        pipeline.push_symbol_metadata(symbols_data)

        # For compatibility, return simple dict with count per channel (optional)
        channel_symbols = {l: np.full(n_symbols, 1+0j, dtype=complex) for l in self.config.oam_channels}

        return field, channel_symbols

    def _modulate_oampsk(self, bits):
        """OAMPSK modulation implementation"""
        # Get magnitudes of configured modes (no duplicates, sorted)
        magnitudes = sorted(list(set([abs(l) for l in self.config.oam_channels if l != 0])))
        n_modes = len(magnitudes)

        log_debug('encoder', f"_modulate_oampsk(): Bits de entrada: {bits}")
        log_debug('encoder', f"_modulate_oampsk(): Magnitudes: {magnitudes}")
        log_debug('encoder', f"_modulate_oampsk(): n_modes: {n_modes}")

        # Padding to complete symbols
        while len(bits) % n_modes != 0:
            bits = np.append(bits, 0)  # Padding with zeros

        log_debug('encoder', f"_modulate_oampsk(): Bits después del padding: {bits}")

        symbols_per_mode = {}

        # Initialize lists for each magnitude
        for mag in magnitudes:
            symbols_per_mode[mag] = []

        # Process bits in groups of n_modes
        for i in range(0, len(bits), n_modes):
            bit_group = bits[i:i+n_modes]

            # Each bit in group goes to different magnitude by position
            for j, bit in enumerate(bit_group):
                magnitude = magnitudes[j]  # Magnitude by bit position

                # Determine sign based on bit value
                if bit == 1:
                    oam_mode = +magnitude  # bit 1 → positive mode
                else:
                    oam_mode = -magnitude  # bit 0 → negative mode

                # Symbol always unitary - information is in mode sign
                symbol = complex(1, 0)
                symbols_per_mode[magnitude].append((oam_mode, symbol))

        return symbols_per_mode

    def work(self, input_items, output_items):
        """HYBRID STREAMING: Use encode() structure but transmit symbol-by-symbol"""
        input_data = input_items[0]
        output = output_items[0]

        # Handle empty input
        if len(input_data) == 0:
            return 0

        # FASE 0: Si ya procesamos, solo enviar señal de completado
        if hasattr(self, 'processing_complete') and self.processing_complete:
            output[:] = 1  # Signal that processing is complete
            return 1

        # FASE 1: Acumular mensaje completo del source
        if not hasattr(self, 'symbols_queue'):
            # Acumular datos hasta detectar mensaje completo
            if not hasattr(self, 'message_buffer'):
                self.message_buffer = []

            # Agregar nuevos bytes al buffer
            for byte_val in input_data:
                if byte_val == 0:  # Skip null padding
                    continue
                self.message_buffer.append(byte_val)

            # Detectar mensaje completo: STX + datos + ETX
            if len(self.message_buffer) >= 2:  # Mínimo STX + ETX
                message_bytes = bytes(self.message_buffer)

                # Si detectamos ETX (0x03), tenemos mensaje completo
                if 0x03 in self.message_buffer:
                    # Extraer hasta ETX (inclusive)
                    etx_pos = self.message_buffer.index(0x03)
                    complete_message = bytes(self.message_buffer[:etx_pos+1])

                    log_info('encoder', f"HYBRID: Mensaje completo detectado: {complete_message.hex()} ('{complete_message.decode('ascii', errors='replace')}')")

                    # Use encode() method to generate correct structure
                    # But extract solo el contenido sin STX/ETX (encode los agregará)
                    if len(complete_message) >= 2 and complete_message[0] == 0x02 and complete_message[-1] == 0x03:
                        content_only = complete_message[1:-1]  # Remove STX/ETX
                        log_debug('encoder', f"HYBRID: Solo contenido: {content_only.hex()} ('{content_only.decode('ascii', errors='replace')}')")
                    else:
                        content_only = complete_message

                    # Pipeline: Process complete message at once
                    encoded_fields, _ = self.encode(content_only)

                    # Store in pipeline GLOBAL so channel can pick it up
                    pipeline.encoder_symbols = encoded_fields
                    pipeline.processing_stage = "encoding_complete"
                    self.processing_complete = True

                    # Mensaje procesado y almacenado en pipeline

                    # Metricas de velocidad teorica
                    K = len({abs(l) for l in self.config.oam_channels if l != 0})
                    symbol_rate_hz = getattr(self.config, "symbol_rate_hz", 32000)

                    # Contar simbolos de datos usando metadatos
                    symbols_data_count = sum(1 for s in pipeline.symbol_metadata if s.get("kind") == "datos")

                    pipeline.set_run_meta({
                        "K_bits_per_symbol": int(K),
                        "symbol_rate_hz": float(symbol_rate_hz),
                        "symbols_total": int(len(encoded_fields)),
                        "symbols_data": int(symbols_data_count),
                    })
                    log_info('encoder', f"Velocidad teórica: K={K} bits/símbolo, Rs={symbol_rate_hz} Hz")

                    # Clear buffer to avoid infinite reprocessing
                    self.message_buffer = []

                    # Continuar para enviar señal
                else:
                    # Mensaje aún incompleto, esperar más datos
                    return len(input_data)
            else:
                # Muy pocos datos, esperar más
                return len(input_data)

        # FASE 2: Enviar señal cuando processing esté completo
        if self.processing_complete:
            # Seguir enviando señal de "listo" al siguiente bloque
            output[:] = 1  # Signal that processing is complete
            return 1

        # If transmission finished, signal end
        elif hasattr(self, 'symbols_queue'):
            return -1  # End of stream

        # Si no hay símbolos listos, consumir input
        return len(input_data)

    def encode_message(self, message_data):
        """
        Codifica mensaje directamente a simbolos OAM.

        Parametros:
            message_data: Cadena de texto a codificar.

        Retorna:
            Array de simbolos OAM codificados.
        """
        log_info('encoder', f"Codificación directa: '{message_data}'")

        # Usar la logica de encode() existente
        encoded_symbols = self.encode(message_data)
        log_info('encoder', f"Generados {len(encoded_symbols)} símbolos OAM")
        return encoded_symbols