# Modulo: oam_channel.py
# Proposito: Simulacion de canal de propagacion atmosferico para modos OAM
# Dependencias clave: numpy, gnuradio, scipy
# Notas: Incluye efectos de turbulencia atmosferica, ruido y distorsion

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from gnuradio import gr
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass
from typing import List
from pipeline import pipeline  # Import global pipeline

# Configuración centralizada del sistema
from oam_system_config import get_system_config, OAMConfig

# Sistema de logging unificado
from oam_logging import log_info, log_warning, log_error, log_debug

# MIGRADO DE 2.py: Variable global para sincronizar turbulencia coherente entre bloques
_coherent_turbulence_manager = {
    'initialized': False,
    'message_id': None,
    'n_symbols': 0,
    'current_symbol': 0
}

# ChannelConfig - Configuración del canal óptico
@dataclass
class ChannelConfig:
    """Configuración del canal óptico con parámetros atmosféricos"""
    cn2: float = None
    wind_speed: float = 5.0
    snr_db: float = None
    enable_turbulence: bool = True
    enable_noise: bool = True

    def __post_init__(self):
        # Obtener configuración centralizada para valores None
        config = get_system_config()

        if self.cn2 is None:
            self.cn2 = config['cn2']
        if self.snr_db is None:
            self.snr_db = config['snr_target']
    enable_crosstalk: bool = True
    outer_scale: float = 20.0
    inner_scale: float = 0.002
    frame_rate: float = 100.0

    def __post_init__(self):
        # Validación de parámetros del canal
        self._validate_channel_parameters()

    def _validate_channel_parameters(self):
        """Validación de parámetros del canal óptico"""

        # Validar Cn²
        if not isinstance(self.cn2, (int, float)) or self.cn2 < 0:
            raise ValueError(f"cn2 debe ser no negativo, obtenido: {self.cn2}")
        if self.cn2 > 1e-12:
            raise ValueError(f"cn2 > 1e-12 representa turbulencia extrema, obtenido: {self.cn2}")

        # Validar velocidad del viento
        if not isinstance(self.wind_speed, (int, float)) or self.wind_speed < 0:
            raise ValueError(f"wind_speed debe ser no negativo, obtenido: {self.wind_speed}")
        if self.wind_speed > 50:
            raise ValueError(f"wind_speed > 50 m/s es poco realista, obtenido: {self.wind_speed}")

        # Validar SNR
        if not isinstance(self.snr_db, (int, float)):
            raise ValueError(f"snr_db debe ser numérico, obtenido: {self.snr_db}")
        if self.snr_db < -10 or self.snr_db > 60:
            raise ValueError(f"snr_db debe estar entre -10 y 60 dB, obtenido: {self.snr_db}")

        # Validar escalas de turbulencia
        if not isinstance(self.outer_scale, (int, float)) or self.outer_scale <= 0:
            raise ValueError(f"outer_scale debe ser positivo, obtenido: {self.outer_scale}")
        if not isinstance(self.inner_scale, (int, float)) or self.inner_scale <= 0:
            raise ValueError(f"inner_scale debe ser positivo, obtenido: {self.inner_scale}")
        if self.inner_scale >= self.outer_scale:
            raise ValueError(f"inner_scale debe ser menor que outer_scale")

        # Validar frame rate
        if not isinstance(self.frame_rate, (int, float)) or self.frame_rate <= 0:
            raise ValueError(f"frame_rate debe ser positivo, obtenido: {self.frame_rate}")
        if self.frame_rate > 1000:
            raise ValueError(f"frame_rate > 1000 Hz puede ser excesivo, obtenido: {self.frame_rate}")

# SystemConfig - Configuración completa del sistema
@dataclass
class SystemConfig:
    """Parámetros del sistema OAM con validación física completa"""
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

        # Validación completa de parámetros físicos
        self._validate_physical_parameters()

    def _validate_physical_parameters(self):
        """Validación rigurosa de parámetros físicos del sistema"""
        # (Se puede reutilizar la misma validación de los otros bloques)
        # Por brevedad, incluyo solo las validaciones más críticas

        # Validar grid_size
        if not isinstance(self.grid_size, int) or self.grid_size <= 0:
            raise ValueError(f"grid_size debe ser un entero positivo, obtenido: {self.grid_size}")
        if not (self.grid_size & (self.grid_size - 1)) == 0:
            raise ValueError(f"grid_size debe ser potencia de 2, obtenido: {self.grid_size}")

        # Validar parámetros físicos críticos
        if self.wavelength <= 0 or self.wavelength < 400e-9 or self.wavelength > 2000e-9:
            raise ValueError(f"wavelength inválida: {self.wavelength*1e9:.0f}nm")

        if self.tx_beam_waist >= self.tx_aperture_size:
            raise ValueError(f"tx_beam_waist debe ser menor que tx_aperture_size")

        # Validar modos OAM
        if not isinstance(self.oam_channels, list) or len(self.oam_channels) == 0:
            raise ValueError("oam_channels debe ser una lista no vacía")

        for channel in self.oam_channels:
            if not isinstance(channel, int) or abs(channel) > 10:
                raise ValueError(f"Canal OAM inválido: {channel}")

class oam_channel(gr.sync_block):
    """
    OAM Channel Block - Canal de propagación óptica atmosférica
    Simula efectos de turbulencia, ruido y crosstalk en propagación de espacio libre
    """
    def __init__(self, grid_size=None, wavelength=None, propagation_distance=None,
                 tx_aperture_size=None, tx_beam_waist=None, oam_channels=None,
                 cn2=None, wind_speed=5.0, snr_db=None, output_factor=1):

        # Obtener configuración centralizada para valores None
        config = get_system_config()

        if grid_size is None:
            grid_size = config['grid_size']
        if wavelength is None:
            wavelength = config['wavelength']
        if propagation_distance is None:
            propagation_distance = config['propagation_distance']
        if tx_aperture_size is None:
            tx_aperture_size = config['tx_aperture_size']
        if tx_beam_waist is None:
            tx_beam_waist = OAMConfig.get_tx_beam_waist()
        if oam_channels is None:
            oam_channels = OAMConfig.get_oam_channels()
        if cn2 is None:
            cn2 = config['cn2']
        if snr_db is None:
            snr_db = config['snr_target']

        # Input/Output: complex field vector
        vlen = grid_size * grid_size

        gr.sync_block.__init__(self,
            name="oam_channel",
            in_sig=[np.uint8],  # Input: trigger signal
            out_sig=[np.uint8])  # Output: trigger signal

        try:
            # Guardar instancia para acceso desde otros bloques
            oam_channel._instance = self

            # Control de logging de configuración del canal
            self._last_channel_cfg = None

            # Convertir string a lista si es necesario (para compatibilidad GNU Radio)
            if isinstance(oam_channels, str):
                oam_channels = eval(oam_channels)

            # Crear SystemConfig y ChannelConfig con validación completa
            self.sys_config = SystemConfig(
                grid_size=int(grid_size),
                wavelength=float(wavelength),
                propagation_distance=float(propagation_distance),
                tx_aperture_size=float(tx_aperture_size),
                tx_beam_waist=float(tx_beam_waist),
                oam_channels=list(oam_channels),
                modulation='OAMPSK'
            )

            self.ch_config = ChannelConfig(
                cn2=float(cn2),
                wind_speed=float(wind_speed),
                snr_db=float(snr_db),
                enable_turbulence=True,
                enable_noise=True,
                enable_crosstalk=True
            )

            # Inicialización del canal óptico
            self.setup_receiver_params()
            self.calculate_channel_params()

            # Contador específico de GNU Radio
            self.fields_processed = 0

            # Log de SNR medido por símbolo
            self.snr_log = []

            # Potencia de referencia del transmisor para ruido realista
            self.reference_signal_power = None

            # MIGRADO DE 2.py: Coherencia temporal de turbulencia
            self.message_phase_screens = None
            self.message_scintillation_screens = None
            self.message_symbol_count = 0
            self.current_message_id = None
            self.total_symbols_in_message = None

            # Logging del canal configurado
            log_info('channel', f"Configuración: {self.sys_config.grid_size}x{self.sys_config.grid_size}, distancia={self.sys_config.propagation_distance}m")
            log_info('channel', f"Canal: Cn²={self.ch_config.cn2:.2e}, SNR={self.ch_config.snr_db}dB, viento={self.ch_config.wind_speed}m/s")
            log_info('channel', f"Calculated parameters: r0={self.r0 if np.isfinite(self.r0) else 'inf'}m, Rytov={self.rytov:.3f}")

        except Exception as e:
            log_error('channel', f"Error en inicialización: {e}")
            raise RuntimeError(f"No se pudo inicializar OAM Channel: {e}")

    def setup_receiver_params(self):
        """Parámetros del receptor con dimensionado basado en l_max"""
        self.divergence = self.sys_config.wavelength / (np.pi * self.sys_config.tx_beam_waist)
        self.rx_beam_size = (
            self.sys_config.tx_beam_waist +
            self.sys_config.propagation_distance * self.divergence
        )

        # Implementación: elegir tamaño de RX para que quepa el anillo de |l|=l_max ---
        l_max = max(abs(l) for l in self.sys_config.oam_channels if l != 0)
        r_peak_max = self.rx_beam_size * np.sqrt(l_max / 2.0)          # (m)
        margin = 1.35                                                  # 30–40% de margen
        required_radius = margin * r_peak_max                          # (m)
        required_diameter = 2.0 * required_radius

        # RX aperture desde configuración central
        self.rx_aperture_size = get_system_config()['rx_aperture_size']

        # Marcar si RX y TX tienen tamaños similares
        tx_aperture = self.sys_config.tx_aperture_size
        self._aperture_matched = abs(self.rx_aperture_size - tx_aperture) / tx_aperture < 0.02

        # Grillas del receptor
        x_rx = np.linspace(
            -self.rx_aperture_size/2,
            self.rx_aperture_size/2,
            self.sys_config.grid_size
        )
        y_rx = np.linspace(
            -self.rx_aperture_size/2,
            self.rx_aperture_size/2,
            self.sys_config.grid_size
        )
        self.X_rx, self.Y_rx = np.meshgrid(x_rx, y_rx)

    def calculate_channel_params(self):
        """Cálculo de parámetros del canal atmosférico"""
        k = 2 * np.pi / self.sys_config.wavelength

        # Canal ideal
        if self.ch_config.cn2 == 0 or not self.ch_config.enable_turbulence:
            # Sin turbulencia
            self.r0 = float('inf')
            self.rytov = 0.0
            self.turbulence_loss_db = 0.0
            log_info('channel', "Canal ideal: sin efectos de turbulencia")
        else:
            # Con turbulencia
            self.r0 = (
                0.423 * k**2 * self.ch_config.cn2 * self.sys_config.propagation_distance
            )**(-3/5)
            self.rytov = (
                1.23 * self.ch_config.cn2 * k**(7/6) *
                self.sys_config.propagation_distance**(11/6)
            )
            self.turbulence_loss_db = 4.3 * self.rytov

        # Pérdidas geométricas
        self.geometric_loss_db = 20 * np.log10(
            self.rx_beam_size / self.sys_config.tx_beam_waist
        )

        # Total
        self.total_loss_db = self.geometric_loss_db + self.turbulence_loss_db

    def angular_spectrum_propagation(self, field):
        """Propagación por espectro angular con FFT"""
        k = 2 * np.pi / self.sys_config.wavelength
        N = field.shape[0]

        # Espaciado espacial consistente con malla linspace
        dx = self.sys_config.tx_aperture_size / (N - 1)
        fx = np.fft.fftfreq(N, dx)
        fy = np.fft.fftfreq(N, dx)
        FX, FY = np.meshgrid(fx, fy)

        field_spectrum = np.fft.fft2(field)
        kz = np.sqrt(k**2 - (2*np.pi*FX)**2 - (2*np.pi*FY)**2 + 0j)
        H = np.exp(1j * kz * self.sys_config.propagation_distance)
        # Máscara evanescente corregida: |f| <= 1/λ
        evanescent_mask = (FX**2 + FY**2) <= (1/self.sys_config.wavelength)**2
        H = H * evanescent_mask

        propagated_spectrum = field_spectrum * H
        propagated_field = np.fft.ifft2(propagated_spectrum)

        # Implementación: Siempre usar resampling mejorado si hay propagación > 1m
        # Esto activa el preservado de fases azimutales
        size_diff_ratio = abs(self.rx_aperture_size - self.sys_config.tx_aperture_size) / self.sys_config.tx_aperture_size
        force_resampling = self.sys_config.propagation_distance > 1.0  # Forzar si d > 1m
        if size_diff_ratio > 0.01 or force_resampling:  # Umbral reducido + forzado
            propagated_field = self.resample_field(propagated_field)

        return propagated_field

    def resample_field(self, field):
        """Optimizado: Evita interpolación cuando TX/RX coinciden"""
        # Si las aperturas coinciden, no es necesario remuestrear
        if hasattr(self, '_aperture_matched') and self._aperture_matched:
            return field.copy()

        # Método rápido para casos simples con factor entero
        tx_size = self.sys_config.tx_aperture_size
        rx_size = self.rx_aperture_size
        size_ratio = rx_size / tx_size

        # Implementación: Siempre usar resampling mejorado si hay propagación significativa
        # Si el ratio es muy cercano a 1 Y no hay propagación, usar campo directo
        if abs(size_ratio - 1.0) < 0.005 and self.sys_config.propagation_distance < 1.0:  # Solo para casos muy cercanos sin propagación
            return field.copy()

        # Para otros casos, usar método optimizado con FFT cuando sea posible
        if abs(size_ratio - round(size_ratio)) < 0.1:  # Factor casi entero
            return self._fast_resample_integer(field, round(size_ratio))

        # Método alternativo: RegularGridInterpolator
        return self._slow_resample_interpolation(field)

    def _fast_resample_integer(self, field, factor):
        """Resampling rápido para factores enteros usando FFT"""
        N = field.shape[0]

        if factor == 1:
            return field.copy()
        elif factor > 1:
            # Upsampling con fftshift/ifftshift para evitar corrimientos
            F = np.fft.fftshift(np.fft.fft2(field))
            F_padded = np.zeros((N * factor, N * factor), dtype=complex)

            # Copiar frecuencias al centro
            start = (N * factor - N) // 2
            end = start + N
            F_padded[start:end, start:end] = F

            resampled = np.fft.ifft2(np.fft.ifftshift(F_padded)) * (factor**2)
        else:
            # Downsampling con fftshift/ifftshift
            F = np.fft.fftshift(np.fft.fft2(field))
            new_N = int(N / factor)
            start = (N - new_N) // 2
            end = start + new_N
            F_cropped = F[start:end, start:end]
            resampled = np.fft.ifft2(np.fft.ifftshift(F_cropped)) / (factor**2)

        # Conservar energía
        original_power = np.sum(np.abs(field)**2)
        resampled_power = np.sum(np.abs(resampled)**2)

        if resampled_power > 0:
            energy_factor = np.sqrt(original_power / resampled_power)
            resampled *= energy_factor

        return resampled

    def _slow_resample_interpolation(self, field):
        """Interpolar COMPLEJO tal cual (sin detectar l, sin purificar fase)"""
        N = field.shape[0]
        x_old = np.linspace(-self.sys_config.tx_aperture_size/2,  self.sys_config.tx_aperture_size/2,  N)
        y_old = x_old.copy()
        x_new = np.linspace(-self.rx_aperture_size/2,            self.rx_aperture_size/2,            N)
        y_new = x_new.copy()

        interp_re = RegularGridInterpolator((y_old, x_old), np.real(field), method='cubic', bounds_error=False, fill_value=0.0)
        interp_im = RegularGridInterpolator((y_old, x_old), np.imag(field), method='cubic', bounds_error=False, fill_value=0.0)

        Y_new, X_new = np.meshgrid(y_new, x_new, indexing='ij')
        P = np.column_stack([Y_new.ravel(), X_new.ravel()])

        re = interp_re(P).reshape(N, N)
        im = interp_im(P).reshape(N, N)
        out = re + 1j*im

        # conservar energía global
        e0 = np.sum(np.abs(field)**2); e1 = np.sum(np.abs(out)**2)
        if e1 > 0: out *= np.sqrt(e0/e1)
        return out

    def _detect_topological_charge(self, field, X, Y):
        """Detectar carga topológica dominante del campo automáticamente"""
        # Evitar el centro (singularidad)
        r = np.sqrt(X**2 + Y**2)
        mask = r > 0.1 * np.max(r)  # Anillo externo

        if np.sum(mask) < 10:  # Si no hay suficientes puntos
            return 0

        # Extraer fase
        phase = np.angle(field)
        phi = np.arctan2(Y, X)

        # Análisis en anillo
        phase_ring = phase[mask]
        phi_ring = phi[mask]

        # Ordenar por ángulo
        sort_idx = np.argsort(phi_ring)
        phi_sorted = phi_ring[sort_idx]
        phase_sorted = phase_ring[sort_idx]

        # Desarrollar fase (unwrap)
        try:
            phase_unwrapped = np.unwrap(phase_sorted)

            # Ajuste lineal: phase = l * phi + offset
            if len(phi_sorted) > 5:
                A = np.vstack([phi_sorted, np.ones(len(phi_sorted))]).T
                slope, _ = np.linalg.lstsq(A, phase_unwrapped, rcond=None)[0]

                # Redondear a entero más cercano
                l_detected = int(round(slope))

                # Limitar a rango razonable
                l_detected = max(-8, min(8, l_detected))

                return l_detected
        except:
            pass

        return 0  # Default si falla detección

    def generate_phase_screens(self, n_screens, field_size=None):
        """Genera pantallas de fase turbulentas con scintillation"""
        # Usar el tamaño del campo si se especifica, sino usar grid_size por defecto
        N = field_size if field_size is not None else self.sys_config.grid_size

        # Espaciado espacial coherente con linspace
        dx = self.rx_aperture_size / (N - 1)
        fx = np.fft.fftfreq(N, d=dx)
        fy = np.fft.fftfreq(N, d=dx)
        FX, FY = np.meshgrid(fx, fy)
        f_mag = np.sqrt(FX**2 + FY**2) + 1e-12

        # Parámetros de Kolmogorov más realistas
        # Escalas atmosféricas típicas para comunicaciones ópticas
        L0 = getattr(self.ch_config, 'outer_scale', 20.0)  # 20m escala externa típica
        l0 = getattr(self.ch_config, 'inner_scale', 0.002)  # 2mm escala interna típica

        spectrum = (
            0.033 * self.ch_config.cn2 * self.sys_config.propagation_distance *
            np.exp(-(f_mag*l0/5.92)**2) /
            (f_mag**2 + (1/L0)**2)**(11/6)
        )

        # Generar pantallas con frozen flow
        screens = []

        # Intervalo temporal más realista para frozen flow
        # Para sistemas de comunicación típicos: ~1-10 ms entre frames
        frame_rate = getattr(self.ch_config, 'frame_rate', 100)  # 100 fps por defecto
        dt = 1.0 / frame_rate
        pixel_shift = self.ch_config.wind_speed * dt / dx

        # Pantalla base de fase
        random_phase = np.random.randn(N, N) + 1j * np.random.randn(N, N)
        base_screen = np.real(
            np.fft.ifft2(
                np.fft.ifftshift(np.sqrt(spectrum) * random_phase)
            )
        )

        for i in range(n_screens):
            shift = int(i * pixel_shift) % N
            screens.append(np.roll(base_screen, shift, axis=1))

        return screens

    def initialize_coherent_turbulence(self, n_symbols, field_size=None, message_id=None):
        """MIGRADO DE 2.py: Inicializa pantallas coherentes para un mensaje completo"""
        if message_id is None:
            message_id = f"msg_{np.random.randint(10000)}"

        # Solo generar nuevas pantallas si es un mensaje diferente
        if self.current_message_id != message_id:
            # SEMILLA FIJA para reproducibilidad como 2.py
            # np.random.seed(42)  # Desactivado: permitir variabilidad realista

            self.current_message_id = message_id
            self.total_symbols_in_message = n_symbols
            self.message_symbol_count = 0

            # Generar pantallas coherentes para todo el mensaje
            self.message_phase_screens = self.generate_phase_screens(n_symbols, field_size)
            self.message_scintillation_screens = self.generate_scintillation_screens(n_symbols, field_size)

            log_info('channel', f"COHERENT_TURBULENCE: Inicializadas {n_symbols} pantallas coherentes para mensaje {message_id}")

        return True

    def get_coherent_turbulence_for_symbol(self):
        """MIGRADO DE 2.py: Obtiene turbulencia coherente para el símbolo actual"""
        if (self.message_phase_screens is None or
            self.message_scintillation_screens is None or
            self.message_symbol_count >= len(self.message_phase_screens)):
            # Usar turbulencia por defecto si no hay coherencia disponible
            return None, None

        phase_screen = self.message_phase_screens[self.message_symbol_count]
        scintillation_screen = self.message_scintillation_screens[self.message_symbol_count]

        self.message_symbol_count += 1

        return phase_screen, scintillation_screen

    def generate_scintillation_screens(self, n_screens, field_size=None):
        """Genera pantallas de scintillation (centelleo de intensidad)"""
        # Usar el tamaño del campo si se especifica, sino usar grid_size por defecto
        N = field_size if field_size is not None else self.sys_config.grid_size

        if self.rytov < 0.1:
            # Turbulencia débil - scintillation despreciable
            return [
                np.ones((N, N))
                for _ in range(n_screens)
            ]

        # Varianza de scintillation según teoría de Rytov
        # Para onda plana: σ²_I ≈ 1.23 * Cn² * k^(7/6) * L^(11/6)
        k = 2 * np.pi / self.sys_config.wavelength
        sigma_I_squared = (
            1.23 * self.ch_config.cn2 * (k**(7/6)) *
            (self.sys_config.propagation_distance**(11/6))
        )

        # Limitar varianza para evitar valores no físicos
        sigma_I_squared = min(sigma_I_squared, 2.0)
        sigma_I = np.sqrt(sigma_I_squared)

        # Generar fluctuaciones log-normales de intensidad
        scint_screens = []
        for i in range(n_screens):
            # Fluctuaciones Gaussianas
            log_amplitude = np.random.normal(0, sigma_I/2, (N, N))

            # Convertir a fluctuaciones de intensidad log-normales
            # I = I₀ * exp(2*χ) donde χ es la fluctuación log-amplitud
            intensity_fluctuation = np.exp(2 * log_amplitude)

            # Normalizar para conservar potencia promedio
            intensity_fluctuation = intensity_fluctuation / np.mean(intensity_fluctuation)

            # Aplicar suavizado espacial para mayor realismo
            intensity_fluctuation = gaussian_filter(intensity_fluctuation, sigma=2.0)

            scint_screens.append(intensity_fluctuation)

        return scint_screens

    def calculate_crosstalk_matrix(self):
        """
        Calcula matriz de crosstalk físicamente realista entre modos OAM

        Modelo empírico para simulación, no derivado exacto.
        Basado en correlaciones fenomenológicas observadas experimentalmente
        con dependencia en |Δl| y parámetro Rytov. Para comparación rigurosa
        con literatura, usar modelos teóricos específicos (ej. Paterson 2005).
        """
        modes = self.sys_config.oam_channels
        n = len(modes)
        matrix = np.zeros((n, n), dtype=complex)

        for i, l1 in enumerate(modes):
            for j, l2 in enumerate(modes):
                if i == j:
                    # Término diagonal: atenuación por turbulencia
                    # Basado en Paterson et al. (2005) - Pure mode transmission
                    if self.rytov > 0.1:
                        mode_attenuation = np.exp(-self.rytov * (abs(l1) + 1)**0.5)
                    else:
                        mode_attenuation = 1.0 - self.rytov * (abs(l1) + 1)**2 / 20
                    matrix[i, j] = mode_attenuation
                else:
                    # Términos off-diagonal: coupling entre modos
                    delta_l = l1 - l2

                    # Modelo físico basado en:
                    # 1. Diferencia de carga topológica
                    # 2. Intensidad de turbulencia (Rytov)
                    # 3. Efectos de apertura finita
                    if abs(delta_l) == 1:
                        # Coupling fuerte para modos adyacentes
                        coupling_strength = 0.3 * np.sqrt(self.rytov)
                    elif abs(delta_l) == 2:
                        # Coupling medio para modos separados por 2
                        coupling_strength = 0.1 * self.rytov
                    else:
                        # Coupling débil para modos muy separados
                        coupling_strength = 0.05 * self.rytov / (abs(delta_l)**0.5)

                    # Limitar coupling para evitar valores no físicos
                    coupling_strength = min(coupling_strength, 0.3)

                    # Fase aleatoria por efectos atmosféricos
                    phase = np.random.uniform(0, 2*np.pi)
                    matrix[i, j] = coupling_strength * np.exp(1j * phase)

        # Normalización física para conservar energía
        # Método: Make matrix row-stochastic para conservar potencia
        for i in range(n):
            row_power = np.sum(np.abs(matrix[i, :])**2)
            if row_power > 1.0:
                # Solo normalizar si excede la potencia unitaria
                matrix[i, :] /= np.sqrt(row_power)

        # Añadir pérdidas por turbulencia en la diagonal si es necesario
        trace_power = np.sum([np.abs(matrix[i, i])**2 for i in range(n)])
        if trace_power > n:
            # Normalización adicional si es necesario
            factor = np.sqrt(n / trace_power)
            for i in range(n):
                matrix[i, i] *= factor

        return matrix

    def apply_crosstalk(self, field, matrix):
        """Aplica crosstalk entre modos OAM usando descomposición modal"""
        # Crear grillas del tamaño del campo propagado
        N = field.shape[0]
        x = np.linspace(-self.rx_aperture_size/2, self.rx_aperture_size/2, N)
        y = np.linspace(-self.rx_aperture_size/2, self.rx_aperture_size/2, N)
        X_field, Y_field = np.meshgrid(x, y)

        # Descomponer campo en modos OAM
        mode_coefficients = []
        modes = self.sys_config.oam_channels
        for l in modes:
            # Obtener modo de referencia (debería estar en encoder)
            # Para simplicidad, usamos proyección directa
            norm_factor = np.sum(np.abs(field)**2) / len(modes)
            if norm_factor > 0:
                # Proyección aproximada en modo l usando grillas del tamaño correcto
                phase_l = np.exp(1j * l * np.arctan2(Y_field, X_field + 1e-12))
                r = np.sqrt(X_field**2 + Y_field**2)
                gaussian = np.exp(-r**2 / (self.rx_aperture_size/4)**2)
                mode_pattern = phase_l * gaussian
                coeff = np.sum(np.conj(mode_pattern) * field) / np.sum(np.abs(mode_pattern)**2)
                mode_coefficients.append(coeff)
            else:
                mode_coefficients.append(0)

        # Aplicar matriz de crosstalk
        mode_coefficients = np.array(mode_coefficients)
        mixed_coefficients = matrix @ mode_coefficients

        # Reconstruir campo con crosstalk aplicado
        reconstructed_field = np.zeros_like(field, dtype=complex)
        for i, l in enumerate(modes):
            phase_l = np.exp(1j * l * np.arctan2(Y_field, X_field + 1e-12))
            r_field = np.sqrt(X_field**2 + Y_field**2)
            gaussian = np.exp(-r_field**2 / (self.rx_aperture_size/4)**2)
            mode_pattern = phase_l * gaussian
            reconstructed_field += mixed_coefficients[i] * mode_pattern

        return reconstructed_field

    def angular_spectrum_propagation(self, field):
        """MIGRADO DE 2.py: Propagación angular spectrum EXACTA"""
        k = 2 * np.pi / self.sys_config.wavelength
        N = field.shape[0]

        # Espaciado espacial consistente con malla linspace
        dx = self.sys_config.tx_aperture_size / (N - 1)
        fx = np.fft.fftfreq(N, dx)
        fy = np.fft.fftfreq(N, dx)
        FX, FY = np.meshgrid(fx, fy)

        field_spectrum = np.fft.fft2(field)
        kz = np.sqrt(k**2 - (2*np.pi*FX)**2 - (2*np.pi*FY)**2 + 0j)
        H = np.exp(1j * kz * self.sys_config.propagation_distance)
        # Máscara evanescente corregida: |f| <= 1/λ
        evanescent_mask = (FX**2 + FY**2) <= (1/self.sys_config.wavelength)**2
        H = H * evanescent_mask

        propagated_spectrum = field_spectrum * H
        propagated_field = np.fft.ifft2(propagated_spectrum)

        # Solo remuestrear si hay diferencia significativa
        # Para evitar problemas de shapes incompatibles en correlación
        size_diff_ratio = abs(self.rx_aperture_size - self.sys_config.tx_aperture_size) / self.sys_config.tx_aperture_size
        if size_diff_ratio > 0.1:  # Solo si hay >10% de diferencia
            propagated_field = self.resample_field(propagated_field)

        return propagated_field

    def resample_field(self, field):
        """MIGRADO DE 2.py: B1 Optimized - Evita interpolación cuando TX/RX coinciden"""
        # Si las aperturas coinciden, no es necesario remuestrear
        if hasattr(self, '_aperture_matched') and self._aperture_matched:
            return field.copy()

        # Método rápido para casos simples con factor entero
        tx_size = self.sys_config.tx_aperture_size
        rx_size = self.rx_aperture_size
        size_ratio = rx_size / tx_size

        # Si el ratio es muy cercano a 1, usar campo directo
        if abs(size_ratio - 1.0) < 0.05:  # 5% de tolerancia
            return field.copy()

        # Para otros casos, usar método optimizado con FFT cuando sea posible
        if abs(size_ratio - round(size_ratio)) < 0.1:  # Factor casi entero
            return self._fast_resample_integer(field, round(size_ratio))

        # Método alternativo: RegularGridInterpolator
        return self._slow_resample_interpolation(field)

    def _fast_resample_integer(self, field, factor):
        """MIGRADO DE 2.py: Remuestreo rápido con factor entero"""
        if factor == 1:
            return field.copy()
        elif factor > 1:
            # Upsampling usando zero-padding en frecuencia
            F = np.fft.fftshift(np.fft.fft2(field))
            N_old = field.shape[0]
            N_new = N_old * factor
            F_new = np.zeros((N_new, N_new), dtype=complex)
            start = (N_new - N_old) // 2
            end = start + N_old
            F_new[start:end, start:end] = F
            upsampled = np.fft.ifft2(np.fft.ifftshift(F_new)) * (factor**2)
            return upsampled
        else:
            # Downsampling usando crop en frecuencia
            F = np.fft.fftshift(np.fft.fft2(field))
            N_old = field.shape[0]
            N_new = N_old // int(1/factor)
            start = (N_old - N_new) // 2
            end = start + N_new
            F_cropped = F[start:end, start:end]
            downsampled = np.fft.ifft2(np.fft.ifftshift(F_cropped)) / (factor**2)
            return downsampled

    def _slow_resample_interpolation(self, field):
        """MIGRADO DE 2.py: Remuestreo lento con interpolación"""
        # Para simplicidad, usar el método de interpolación ya implementado en decoder
        # Este método está en oam_decoder._resample_field_to_template_size
        # Por ahora, devolver el campo sin cambios (implementación pendiente)
        return field.copy()

    def add_noise(self, field):
        """Simula ruido AWGN y mide SNR efectivo (pre/post ruido)"""
        def _measure_snr(clean, noisy):
            # ROI: evita zonas casi nulas
            I = np.abs(clean)**2
            thr = 1e-12 * (I.max() if I.size else 1.0)
            mask = I > thr

            if clean.ndim == 2:
                s_pow = np.mean(I[mask]) if np.any(mask) else np.mean(I)
                n_pow = np.mean(np.abs((noisy - clean))[mask]**2) if np.any(mask) else np.mean(np.abs(noisy - clean)**2)
                snr_lin = np.inf if n_pow == 0 else s_pow / n_pow
                return 10*np.log10(snr_lin)
            else:
                # 3D: por frame
                snrs = []
                for t in range(clean.shape[0]):
                    It = np.abs(clean[t])**2
                    thr_t = 1e-12 * (It.max() if It.size else 1.0)
                    m = It > thr_t
                    s_pow = np.mean(It[m]) if np.any(m) else np.mean(It)
                    n_pow = np.mean(np.abs((noisy[t] - clean[t]))[m]**2) if np.any(m) else np.mean(np.abs(noisy[t] - clean[t])**2)
                    snr_lin = np.inf if n_pow == 0 else s_pow / n_pow
                    snrs.append(10*np.log10(snr_lin))
                return snrs

        # Calcular ruido del receptor basado en potencia de referencia del transmisor
        # El ruido del receptor es constante (no depende de la señal recibida)
        signal_power = np.mean(np.abs(field)**2)

        # Establecer potencia de referencia en el primer símbolo (antes de atenuación)
        if self.reference_signal_power is None:
            self.reference_signal_power = signal_power
            log_debug('channel', f"Potencia de referencia establecida: {self.reference_signal_power:.3e}")

        # Ruido constante basado en SNR objetivo y potencia de referencia
        snr_linear_cfg = np.inf if not np.isfinite(self.ch_config.snr_db) else 10**(self.ch_config.snr_db/10)
        noise_power = 0.0 if np.isinf(snr_linear_cfg) else self.reference_signal_power / snr_linear_cfg
        noise = (np.random.randn(*field.shape) + 1j*np.random.randn(*field.shape)) * np.sqrt(noise_power/2)

        noisy = field + noise

        # Medición SNR efectivo
        snr_measured_db = _measure_snr(field, noisy)
        self.snr_log.append(snr_measured_db)

        # LOG SNR EN PIPELINE
        from pipeline import pipeline
        if isinstance(snr_measured_db, list):
            for snr_val in snr_measured_db:
                pipeline.log_snr(snr_val)
        else:
            pipeline.log_snr(snr_measured_db)

        return noisy

    def get_snr_report(self):
        """Devuelve SNR medido por frame y promedio en dB"""
        # Aplana lista de llamadas; cada entrada puede ser float o lista de floats
        vals = []
        for v in self.snr_log:
            if isinstance(v, (list, tuple, np.ndarray)):
                vals.extend([float(x) for x in v])
            else:
                vals.append(float(v))
        if not vals:
            return {"frames": [], "snr_avg_db": float('nan')}
        return {"frames": vals, "snr_avg_db": float(np.mean(vals))}

    def propagate(self, field):
        """Propaga con turbulencia split-step (Ns pantallas + Fresnel entre pantallas)"""
        # GUARDAR CAMPO DE ENTRADA EN PIPELINE BUFFER
        from pipeline import pipeline

        # Implementación: EMPUJAR field_tx ORIGINAL DEL ENCODER SI EXISTE, SINO field PROCESADO
        if hasattr(pipeline, 'field_tx') and pipeline.field_tx is not None:
            symbol_idx = getattr(pipeline, 'current_symbol_idx', 0)
            if symbol_idx < len(pipeline.field_tx):
                # Usar el campo original del encoder (donut puro)
                pipeline.push_field("before_channel", pipeline.field_tx[symbol_idx])
            else:
                pipeline.push_field("before_channel", field)
        else:
            pipeline.push_field("before_channel", field)

        # GUARDAR VERSIÓN WIDE SI EXISTE PARA DIAGNÓSTICO
        if hasattr(pipeline, 'field_wide') and pipeline.field_wide is not None:
            # Obtener el índice del símbolo actual del pipeline de símbolos
            symbol_idx = getattr(pipeline, 'current_symbol_idx', 0)
            if symbol_idx < len(pipeline.field_wide):
                pipeline.push_field("before_channel_wide", pipeline.field_wide[symbol_idx])
        # Número de pantallas de fase en split-step propagation
        # Para propagación de espacio libre se usa un solo paso
        Ns = 1

        # Beam-wander según turbulencia
        wander_active = self.ch_config.cn2 >= 1e-14

        # REGISTRAR AMBIENTE EN PIPELINE
        from pipeline import pipeline
        pipeline.env.update({
            'cn2': self.ch_config.cn2,
            'snr_db': self.ch_config.snr_db,
            'distance': self.sys_config.propagation_distance,
            'Ns': Ns,
            'wander': wander_active,
            'oam_channels': self.sys_config.oam_channels
        })

        self._log_channel_cfg_if_changed(Ns)
        dz = self.sys_config.propagation_distance / Ns
        out = field.copy()

        # Grillas para tilt (beam-wander)
        N = out.shape[0]
        x = np.linspace(-self.rx_aperture_size/2, self.rx_aperture_size/2, N)
        y = x.copy()
        X_rx, Y_rx = np.meshgrid(x, y)

        for s in range(Ns):
            # 1) Propagación Fresnel corta
            out = self.angular_spectrum_step(out, dz)

            # 2) Turbulencia por tramo: fase + scintilación
            if self.ch_config.enable_turbulence:
                ps = self.generate_phase_screens(1, out.shape[0])[0]           # pantalla de fase
                ss = self.generate_scintillation_screens(1, out.shape[0])[0]   # pantalla de scintilación (amplitud)
                out *= np.exp(1j * ps)
                out *= np.sqrt(ss)

                # Desplazamiento del haz (beam-wander) proporcional a turbulencia
                # Varianza del tilt angular: σ²_tilt ∝ Cn² × L^(5/3)
                # Modelo simplificado para turbulencia débil a fuerte
                k = 2 * np.pi / self.sys_config.wavelength
                sigma_tilt = 0.43 * np.sqrt(self.ch_config.cn2 * self.sys_config.propagation_distance**(5/3) / self.rx_aperture_size**(1/3))

                # Aplicar tilt aleatorio si hay turbulencia
                if sigma_tilt > 1e-9:  # Umbral mínimo para evitar cálculos innecesarios
                    ax = np.random.normal(0, sigma_tilt)
                    ay = np.random.normal(0, sigma_tilt)
                    phase_tilt = np.exp(1j * k * (ax * X_rx + ay * Y_rx))
                    out *= phase_tilt
                    log_debug('channel', f"Beam-wander: σ_tilt={sigma_tilt:.3e} rad, ax={ax:.3e}, ay={ay:.3e}")

        # 3) (Opcional) Crosstalk empírico SOLO si NO hay turbulencia física
        if self.ch_config.enable_crosstalk and not self.ch_config.enable_turbulence:
            cx = self.calculate_crosstalk_matrix()
            out = self.apply_crosstalk(out, cx)

        # Aplicar atenuación de espacio libre y turbulencia atmosférica
        # Incluye pérdidas geométricas por divergencia del haz y pérdidas por turbulencia
        loss_linear = 10**(-self.total_loss_db / 10)
        out *= np.sqrt(loss_linear)
        log_debug('channel', f"Atenuación total: {self.total_loss_db:.2f} dB (geométrica={self.geometric_loss_db:.2f} dB + turbulencia={self.turbulence_loss_db:.2f} dB)")

        # 4) Ruido
        if self.ch_config.enable_noise:
            out = self.add_noise(out)

        # GUARDAR CAMPO PROPAGADO EN PIPELINE BUFFER
        from pipeline import pipeline
        pipeline.push_field("after_channel", out)

        # CALCULAR MÉTRICAS DE POTENCIA ANTES VS DESPUÉS
        if len(pipeline.frame_buffer.get("before_channel", [])) > 0:
            # Obtener el campo antes del canal (recién guardado)
            field_before = pipeline.frame_buffer["before_channel"][-1]
            field_after = out

            # Calcular potencias
            power_before_w = float(np.sum(np.abs(field_before)**2))
            power_after_w = float(np.sum(np.abs(field_after)**2))

            # Calcular pérdida en dB (evitar división por cero)
            if power_before_w > 0 and power_after_w > 0:
                power_loss_db = float(10 * np.log10(power_before_w / power_after_w))
            else:
                power_loss_db = 0.0

            # Guardar métricas en el pipeline
            power_metrics = {
                "power_before_w": power_before_w,
                "power_after_w": power_after_w,
                "power_loss_db": power_loss_db
            }
            pipeline.power_log.append(power_metrics)

        return out

    def angular_spectrum_step(self, field, dz):
        """Un paso de Fresnel por espectro angular en distancia dz"""
        k  = 2*np.pi/self.sys_config.wavelength
        N  = field.shape[0]
        dx = self.sys_config.tx_aperture_size / (N - 1)
        fx = np.fft.fftfreq(N, dx); fy = np.fft.fftfreq(N, dx)
        FX, FY = np.meshgrid(fx, fy)
        kz = np.sqrt(k**2 - (2*np.pi*FX)**2 - (2*np.pi*FY)**2 + 0j)
        H  = np.exp(1j * kz * dz)
        ev_mask = (FX**2 + FY**2) <= (1/self.sys_config.wavelength)**2
        F = np.fft.fft2(field); F *= (H * ev_mask)
        return np.fft.ifft2(F)

    def _log_channel_cfg_if_changed(self, Ns):
        """Log channel config only if something changed"""
        current_cfg = (self.ch_config.cn2, self.ch_config.snr_db, Ns, self.ch_config.enable_turbulence)
        if current_cfg != self._last_channel_cfg:
            log_info('channel', f"CHANNEL_CONFIG: Cn²={self.ch_config.cn2:.1e}, SNR={self.ch_config.snr_db}dB, Ns={Ns}, turbulence={self.ch_config.enable_turbulence}")
            self._last_channel_cfg = current_cfg

    def work(self, input_items, output_items):
        """Pipeline: Process encoder symbols through atmospheric channel"""
        input_data = input_items[0]
        output = output_items[0]

        if len(input_data) == 0:
            return 0

        # Process symbols from global pipeline
        if pipeline.processing_stage == "encoding_complete" and not hasattr(self, 'processing_complete'):
            log_info('channel', f"PIPELINE: Tomando {len(pipeline.encoder_symbols)} símbolos del encoder")

            # Process all symbols through atmospheric channel
            propagated_symbols = []
            for i, field_2d in enumerate(pipeline.encoder_symbols):
                # Track current symbol index for wide field access
                pipeline.current_symbol_idx = i
                # Apply atmospheric propagation
                propagated_field = self.propagate(field_2d)
                propagated_symbols.append(propagated_field)

                if i < 3:  # Log first few symbols
                    power_in = np.mean(np.abs(field_2d)**2)
                    power_out = np.mean(np.abs(propagated_field)**2)
                    log_debug('channel', f"Símbolo {i}: Pin={power_in:.6f}, Pout={power_out:.6f}")

            # Store propagated symbols in pipeline
            pipeline.channel_symbols = propagated_symbols
            pipeline.processing_stage = "channel_complete"
            self.processing_complete = True

            # REPORTAR SNR MEDIDO
            snr_report = self.get_snr_report()
            log_info('channel', f"SNR: avg_dB: {snr_report['snr_avg_db']:.2f}")
            log_info('channel', f"PIPELINE: Enviados {len(propagated_symbols)} símbolos propagados al decodificador")

        # Send trigger signal to next block
        if len(output) > 0:
            output[0] = 1
            return 1
        return 0

    def propagate_symbols(self, input_symbols):
        """Método directo de generación: Propaga símbolos por canal atmosférico"""
        log_info('channel', f"PYTHON DIRECTO: Propagando {len(input_symbols)} símbolos...")

        # Usar lógica de propagate() existente
        propagated_symbols = self.propagate(input_symbols)
        log_info('channel', f"Propagados {len(propagated_symbols)} símbolos con SNR={self.ch_config.snr_db}dB")
        return propagated_symbols