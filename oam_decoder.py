# Modulo: oam_decoder.py
# Proposito: Decodificacion de simbolos OAM a mensajes ASCII mediante flujo directo de bits
# Dependencias clave: numpy, gnuradio, scipy, pipeline
# Notas: Soporta auto-deteccion de M=1,2,3,4 modos por simbolo

#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

# Configuracion de mapeo de modos
# ACTIVE_MAGNITUDES se calcula dinámicamente desde configuración central
CHANNELS_PER_SYMBOL   = 4              # K (modos simultaneos)
SYMBOL_ENDIAN         = "MSB"          # Orden de bits en simbolo
FRAME_RANGE           = None           # Rango de frames de datos

# Funciones de metrica BER
def _bytes_to_bits(b: bytes) -> np.ndarray:
    """
    Convierte bytes a bits con orden MSB primero.

    Parametros:
        b: Datos binarios.

    Retorna:
        Array de bits como enteros 0 o 1.
    """
    if len(b) == 0:
        return np.array([], dtype=np.uint8)
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8), bitorder='big')

def compute_ber(tx_bytes: bytes, rx_bytes: bytes):
    """
    Calcula BER comparando bytes transmitidos vs recibidos

    Args:
        tx_bytes: Bytes originales transmitidos (sin STX/ETX)
        rx_bytes: Bytes recuperados del decoder (sin STX/ETX)

    Returns:
        dict: Métricas BER completas
    """
    # Convertir a bits (MSB-first)
    tx_bits = _bytes_to_bits(tx_bytes)
    rx_bits = _bytes_to_bits(rx_bytes)

    # Alinear longitudes
    n = min(len(tx_bits), len(rx_bits))
    if n == 0:
        return {"bits_total": 0, "bit_errors": 0, "BER": float('nan'),
                "bytes_total": 0, "byte_errors": 0, "note": "Sin datos para comparar"}

    bit_errors = int(np.sum(tx_bits[:n] != rx_bits[:n]))
    BER = bit_errors / n

    # Métrica opcional por bytes (útil para diagnóstico)
    m = min(len(tx_bytes), len(rx_bytes))
    byte_errors = sum(tb != rb for tb, rb in zip(tx_bytes[:m], rx_bytes[:m]))

    note = ""
    if len(tx_bits) != len(rx_bits):
        note = f"Longitudes distintas: TX={len(tx_bits)} bits, RX={len(rx_bits)} bits. Comparado hasta {n}."

    return {
        "bits_total": n,
        "bit_errors": bit_errors,
        "BER": BER,
        "BER_dB": -10 * np.log10(BER) if BER > 0 else float('inf'),
        "bytes_total": m,
        "byte_errors": byte_errors,
        "byte_error_rate": byte_errors / m if m > 0 else 0,
        "note": note
    }

# SystemConfig - COPIADO EXACTO de 2.py
@dataclass
class SystemConfig:
    """Parámetros del sistema"""
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
        self._validate_physical_parameters()

    def _validate_physical_parameters(self):
        """Validación simplificada para decoder"""
        if not isinstance(self.grid_size, int) or self.grid_size <= 0:
            raise ValueError(f"grid_size debe ser un entero positivo, obtenido: {self.grid_size}")
        if self.grid_size < 64 or self.grid_size > 2048:
            raise ValueError(f"grid_size debe estar entre 64 y 2048, obtenido: {self.grid_size}")

# ChannelConfig - COPIADO EXACTO de 2.py
@dataclass
class ChannelConfig:
    """Configuración del canal"""
    cn2: float = None
    wind_speed: float = 2.0
    snr_db: float = None
    enable_turbulence: bool = True
    enable_noise: bool = True

    enable_crosstalk: bool = True

    def __post_init__(self):
        # Obtener configuración centralizada para valores None
        config = get_system_config()

        if self.cn2 is None:
            self.cn2 = config['cn2']
        if self.snr_db is None:
            self.snr_db = config['snr_target']

        if not self.enable_noise:
            self.snr_db = float('inf')
        self._validate_channel_parameters()

    def _validate_channel_parameters(self):
        """Validación simplificada para decoder"""
        if not isinstance(self.cn2, (int, float)) or self.cn2 < 0:
            raise ValueError(f"cn2 debe ser no negativo, obtenido: {self.cn2}")


class oam_decoder(gr.sync_block):
    """
    OAM Decoder Block - COMPLETAMENTE SINCRONIZADO con 2.py
    Usa método experimental de 3 orificios exactamente igual que 2.py
    """
    def __init__(self, grid_size=None, wavelength=None, tx_aperture_size=None, tx_beam_waist=None,
                 oam_channels=None, propagation_distance=None, input_factor=1, aperture_mode="none"):

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
        if propagation_distance is None:
            propagation_distance = config['propagation_distance']

        # Configurar entrada como vector
        if isinstance(oam_channels, str):
            oam_channels_list = eval(oam_channels)
        else:
            oam_channels_list = oam_channels

        magnitudes = sorted(list(set([abs(l) for l in oam_channels_list if l != 0])))
        n_modes = len(magnitudes) if len(magnitudes) > 0 else 4

        # Configuración de vector length simplificado
        # max_message_size = 50
        # max_total_chars = 1 + 1 + max_message_size + 1  # STX + len + message + ETX
        # max_symbols = (max_total_chars * 8 + n_modes - 1) // n_modes  # bits to symbols
        # vlen = int(grid_size * grid_size * max_symbols)  # Asegurar entero

        gr.sync_block.__init__(self,
            name="oam_decoder",
            in_sig=[np.uint8],  # GNU Radio connection (ignored)
            out_sig=[np.uint8])  # GNU Radio connection (output signal)

        try:
            # Convertir string a lista si es necesario
            if isinstance(oam_channels, str):
                oam_channels = eval(oam_channels)

            # Crear SystemConfig EXACTO como en 2.py
            self.config = SystemConfig(
                grid_size=int(grid_size),
                wavelength=float(wavelength),
                propagation_distance=float(propagation_distance),
                tx_aperture_size=float(tx_aperture_size),
                tx_beam_waist=float(tx_beam_waist),
                oam_channels=list(oam_channels),
                modulation='OAMPSK'
            )

            # Implementación: Guardar aperture_mode para control de máscara
            self.aperture_mode = str(aperture_mode)
            valid_modes = ["none", "circular_hann", "rect_slit", "tri_slit"]
            if self.aperture_mode not in valid_modes:
                raise ValueError(f"aperture_mode debe ser uno de {valid_modes}, obtenido: {aperture_mode}")
            log_info('decoder', f"Aperture mode: {self.aperture_mode}")

            # Configuración del canal (mock para decoder)
            # Calcular beam expansion real usando difracción Gaussiana
            z_R = np.pi * float(tx_beam_waist)**2 / float(wavelength)  # Rayleigh distance
            z = float(propagation_distance)
            rx_beam_waist_calculated = float(tx_beam_waist) * np.sqrt(1 + (z/z_R)**2)

            self.channel = type('MockChannel', (), {
                'rx_aperture_size': float(tx_aperture_size) * 3.5,  # Apertura RX (70mm si TX=20mm)
                'rx_beam_size': rx_beam_waist_calculated,           # Expansión física real
                'wavelength': float(wavelength),
                'config': ChannelConfig()
            })()

            # Inicialización EXACTA como OAMDecoder de 2.py
            self.setup_grids()
            self.generate_modes()

            log_info('decoder', "Inicializado con método experimental 3 orificios")
            log_info('decoder', f"Config: grid={grid_size}, modos={oam_channels}")

            # CACHE DE PLANTILLAS: Generar una sola vez
            log_info('decoder', "Generando cache de plantillas")
            self._init_template_cache()
            log_info('decoder', "Cache de plantillas listo")

            # SECUENCIAL: Activar monitor para procesar pipeline automáticamente
            log_info('decoder', "Modo secuencial activado con monitor automático")
            self._start_pipeline_monitor()

        except Exception as e:
            log_error('decoder', f"Error de inicialización: {e}")
            raise

    def setup_grids(self):
        """Setup grids EXACTO como en 2.py"""
        self.L = self.config.tx_aperture_size
        self.x = np.linspace(-self.L/2, self.L/2, self.config.grid_size)
        self.y = np.linspace(-self.L/2, self.L/2, self.config.grid_size)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Para decoder: grilla expandida para recepción
        self.rx_aperture_size = self.channel.rx_aperture_size
        self.rx_beam_size = self.channel.rx_beam_size

    def _apply_aperture_mask(self, field):
        """Aplica máscara de apertura según aperture_mode seleccionado"""
        if self.aperture_mode == "none":
            Fw = field  # Sin modificación
        else:
            r = np.sqrt(self.X**2 + self.Y**2)

            if self.aperture_mode == "circular_hann":
                # Máscara circular suave tipo Hann
                r_max = self.config.tx_aperture_size / 2
                mask = np.where(r <= r_max,
                               0.5 * (1 + np.cos(np.pi * r / r_max)),
                               0.0)
                Fw = field * mask

            elif self.aperture_mode == "rect_slit":
                # Máscara rectangular (slit)
                aperture_half = self.config.tx_aperture_size / 2
                mask_x = (np.abs(self.X) <= aperture_half)
                mask_y = (np.abs(self.Y) <= aperture_half)
                mask = mask_x & mask_y
                Fw = field * mask.astype(float)

            elif self.aperture_mode == "tri_slit":
                # Máscara triangular (3 orificios para detección modal)
                aperture_radius = self.config.tx_aperture_size / 4  # Radio más pequeño para orificios
                # Posiciones de los 3 orificios en triángulo
                hole_distance = self.config.tx_aperture_size / 2
                positions = [
                    (0, hole_distance * 2/3),           # Arriba
                    (-hole_distance * np.sqrt(3)/2, -hole_distance/3),  # Abajo izquierda
                    (hole_distance * np.sqrt(3)/2, -hole_distance/3)    # Abajo derecha
                ]

                mask = np.zeros_like(self.X, dtype=float)
                for x0, y0 in positions:
                    r_hole = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
                    mask += (r_hole <= aperture_radius).astype(float)

                mask = np.clip(mask, 0, 1)  # Asegurar que no exceda 1
                Fw = field * mask

            else:
                # Por defecto, sin máscara
                Fw = field

        return Fw

    def generate_modes(self):
        """Generar modos EXACTO como en 2.py"""
        self.modes = {}
        self.rx_modes = {}

        for l in self.config.oam_channels:
            self.modes[l] = self.generate_laguerre_gaussian(l, 0)
            self.rx_modes[l] = self.generate_laguerre_gaussian(l, 0)  # Simplificado
            log_debug('decoder', f"Modo OAM l={l} generado")

    def generate_laguerre_gaussian(self, l, p, z=0):
        """Generar haz Laguerre-Gaussian EXACTO como en 2.py"""
        r = np.sqrt(self.X**2 + self.Y**2)
        theta = np.arctan2(self.Y, self.X)

        w0 = self.config.tx_beam_waist
        w_z = w0 * np.sqrt(1 + (z / (np.pi * w0**2 / self.config.wavelength))**2) if z != 0 else w0

        rho = np.sqrt(2) * r / w_z
        L_pl = special.eval_genlaguerre(p, abs(l), rho**2)

        radial = (np.sqrt(2) * r / w_z)**abs(l) * np.exp(-rho**2 / 2) * L_pl
        angular = np.exp(1j * l * theta)
        norm = np.sqrt(2 / np.pi) / w_z

        return norm * radial * angular

    def _detect_sign_standard_correlation(self, field, magnitude_l):
        """
        Correlacion directa con modos OAM de referencia.

        Parametros:
            field: Campo optico recibido.
            oam_modes: Diccionario de modos de referencia.

        Retorna:
            Diccionario con correlaciones por modo.

        Notas:
            - Usa correlacion modal estandar sin mascara triangular.
        """
        # Generar patrones de referencia para +l y -l
        pos_pattern = self._generate_laguerre_gaussian_rx(field.shape[0], +magnitude_l)
        neg_pattern = self._generate_laguerre_gaussian_rx(field.shape[0], -magnitude_l)

        # Calcular correlaciones
        pos_correlation = np.abs(np.sum(field * np.conj(pos_pattern)))
        neg_correlation = np.abs(np.sum(field * np.conj(neg_pattern)))

        # Detectar signo basado en mayor correlación
        if pos_correlation > neg_correlation:
            sign = +1
            confidence = pos_correlation / (pos_correlation + neg_correlation)
        else:
            sign = -1
            confidence = neg_correlation / (pos_correlation + neg_correlation)

        return sign, confidence, {
            'method': 'standard_correlation',
            'pos_correlation': pos_correlation,
            'neg_correlation': neg_correlation,
            'magnitude_l': magnitude_l
        }

    def _create_three_hole_mask(self, N, center_x, center_y, radius, rotation_deg):
        """
        Deteccion de signo usando mascara con orificios espaciados.

        Parametros:
            field: Campo optico recibido.
            magnitude_l: Magnitud del modo OAM.

        Retorna:
            Diccionario con correlaciones positiva y negativa.

        Notas:
            - Implementa mascara con orificios espaciados cada 120 grados.
        """
        log_warning('decoder', "_create_three_hole_mask no debería ejecutarse - usar solo búsqueda exhaustiva")
        mask = np.zeros((N, N))

        # Ángulos de los 3 orificios (0°, 120°, 240°) + rotación
        hole_angles = np.array([0, 120, 240]) + rotation_deg
        hole_angles_rad = np.deg2rad(hole_angles)

        # Tamaño de cada orificio (proporcional al radio y magnitud)
        hole_size = max(2, int(radius * 0.15))  # 15% del radio

        # Crear coordenadas
        y_coords, x_coords = np.indices((N, N))

        # Posición de cada orificio
        for angle_rad in hole_angles_rad:
            hole_x = center_x + radius * np.cos(angle_rad)
            hole_y = center_y + radius * np.sin(angle_rad)

            # Distancia de cada píxel al centro del orificio
            dist_to_hole = np.sqrt((x_coords - hole_x)**2 + (y_coords - hole_y)**2)

            # Orificio circular con suavizado gaussiano en bordes
            hole_mask = np.exp(-0.5 * (dist_to_hole / (hole_size * 0.5))**2)
            hole_mask[dist_to_hole > hole_size] = 0

            mask += hole_mask

        # Normalizar
        if mask.max() > 0:
            mask = mask / mask.max()

        return mask

    def _apply_photometry_normalization(self, field):
        """
        MEJORA 1: Normalización fotométrica avanzada
        Normaliza la amplitud del campo complejo preservando la fase
        """
        # Preservar el campo complejo, NO convertir a intensidad
        if not np.iscomplexobj(field):
            log_warning('decoder', "Campo no complejo en normalización - omitiendo normalización")
            return field

        # Normalizar la amplitud del campo complejo preservando la fase
        intensity = np.abs(field)**2

        # Restar mediana de intensidad para eliminar offset DC
        median_val = np.median(intensity)
        intensity_corrected = np.maximum(intensity - median_val, 0)  # Evitar negativos

        # Calcular factor de corrección de amplitud
        amplitude_corrected = np.sqrt(intensity_corrected)

        # Aplicar corrección de amplitud preservando fase
        phase = np.angle(field)
        field_corrected = amplitude_corrected * np.exp(1j * phase)

        # Normalizar por norma total
        norm = np.sqrt(np.sum(np.abs(field_corrected)**2))
        if norm > 0:
            field_normalized = field_corrected / norm
        else:
            field_normalized = field_corrected

        return field_normalized

    def _create_circular_mask(self, grid_size, inner_radius=0.2, outer_radius=0.8):
        """
        MEJORA 2: Máscara circular y apodización
        Enfoca en región útil del haz LG con ventana suave
        """
        center = grid_size // 2
        y, x = np.ogrid[:grid_size, :grid_size]
        r = np.sqrt((x - center)**2 + (y - center)**2) / (grid_size//2)

        # Máscara circular con transición suave (ventana Hann)
        mask = np.zeros((grid_size, grid_size))

        # Región interna (100%)
        inner_region = r <= inner_radius
        mask[inner_region] = 1.0

        # Región de transición (ventana Hann)
        transition_region = (r > inner_radius) & (r <= outer_radius)
        transition_values = 0.5 * (1 + np.cos(np.pi * (r[transition_region] - inner_radius) / (outer_radius - inner_radius)))
        mask[transition_region] = transition_values

        # Región externa (0%)
        # Ya está en ceros por inicialización

        return mask

    def _detect_mode_count_enhanced(self, field):
        """
        Método de detección mejorado: Detección con normalización fotométrica y enmascarado
        Incorpora mejoras 1, 2 y umbrales básicos
        """
        import time
        corr_start = time.time()
        magnitudes = sorted(list(set([abs(l) for l in self.config.oam_channels if l != 0])))

        log_info('decoder', f"Detección mejorada: Detección mejorada para magnitudes {magnitudes}")

        # Precalcular plantillas incrementales si no existen
        if not hasattr(self, '_enhanced_templates_cache'):
            self._precalculate_enhanced_templates(magnitudes)

        # MEJORA 1: Normalización fotométrica del campo recibido
        field_normalized = self._apply_photometry_normalization(field)

        # MEJORA 2: Aplicar máscara circular
        mask = self._create_circular_mask(self.config.grid_size,
                                          inner_radius=0.1,  # Núcleo central
                                          outer_radius=0.7)  # Región útil LG
        field_masked = field_normalized * mask

        # Recalcular norma después del enmascarado
        norm_field = np.sqrt(np.sum(field_masked**2))

        best_correlation = -1
        best_mode_count = 1
        correlations = []

        # Probar cada plantilla incremental mejorada
        for count in range(1, len(magnitudes) + 1):
            template_data = self._enhanced_templates_cache[count]
            reference_field = template_data['field_masked']
            norm_ref = template_data['norm_masked']

            # Correlación normalizada mejorada
            correlation = np.abs(np.sum(reference_field * field_masked))

            if norm_ref > 0 and norm_field > 0:
                correlation_normalized = correlation / (norm_ref * norm_field)
                correlations.append((count, correlation_normalized))

                if correlation_normalized > best_correlation:
                    best_correlation = correlation_normalized
                    best_mode_count = count

        # MEJORA 3: Umbrales básicos ROC
        threshold_1_mode = 0.25  # Umbral para 1 modo seguro
        threshold_2_mode = 0.15  # Umbral para 2 modos seguro
        margin_threshold = 0.05  # Margen mínimo entre top1 y top2

        # Calcular margen de confianza
        correlations_sorted = sorted(correlations, key=lambda x: x[1], reverse=True)
        margin = correlations_sorted[0][1] - correlations_sorted[1][1] if len(correlations_sorted) > 1 else 1.0

        # Evaluación con umbrales
        confidence_level = "HIGH"
        if best_correlation < threshold_1_mode or margin < margin_threshold:
            confidence_level = "LOW"

        log_info('decoder', f"Correlaciones mejoradas por cantidad de modos:")
        for count, corr in correlations:
            marker = " ←← MEJOR" if count == best_mode_count else ""
            log_debug('decoder', f"{count} modos: {corr:.3f}{marker}")

        log_info('decoder', f"Confianza: {confidence_level} (ρ={best_correlation:.3f}, Δ={margin:.3f})")

        corr_end = time.time()
        log_info('decoder', f"Detección mejorada: {corr_end-corr_start:.3f}s")

        # Convertir a bits de compatibilidad (4 bits siempre por estándar)
        dummy_bits = [1 if i < best_mode_count else 0 for i in range(4)]
        dummy_combination = [magnitudes[i] if i < best_mode_count else -magnitudes[i]
                           for i in range(len(magnitudes))]

        # ETAPA 2: Si se detecta exactamente 1 modo, determinar signo por difracción triangular
        final_combination = list(dummy_combination)
        if best_mode_count == 1:
            log_info('decoder', f"ETAPA 2: Detectando signo para modo |l|={magnitudes[0]}")
            log_debug('decoder', f"Llamando NCC para |l|={magnitudes[0]}")
            detected_sign, sign_confidence = self._detect_sign_correlation(field, magnitudes[0])

            # Aplicar signo detectado
            final_combination[0] = magnitudes[0] * detected_sign
            log_info('decoder', f"RESULTADO FINAL: l={final_combination[0]} (confianza signo: {sign_confidence:.3f})")
        else:
            log_info('decoder', f"ETAPA 2: OMITIDA (cantidad={best_mode_count} ≠ 1)")

        return final_combination, dummy_bits, best_correlation

    def _detect_specific_modes_and_signs(self, field):
        """
        Método completo de detección: Detecta modos específicos presentes y sus signos
        Implementa mapeo correcto: +l=1, -l=0, posición según carga topológica

        Mapeo de bits:
        bit[0] = l=1 (LSB del simbolo)
        bit[1] = l=2
        bit[2] = l=3
        bit[3] = l=4 (MSB del simbolo)
        """
        log_info('decoder', f"COMPLETE: Detección completa de modos específicos y signos")

        magnitudes = sorted(list(set([abs(l) for l in self.config.oam_channels if l != 0])))

        # Inicializar bits como ausentes
        detected_bits = [0, 0, 0, 0]  # [l=1, l=2, l=3, l=4]
        detected_modes = []
        mode_confidences = []

        # Para cada magnitud, verificar si está presente y determinar signo
        for i, magnitude in enumerate(magnitudes):
            log_debug('decoder', f"Analizando modo |l|={magnitude}:")

            # PASO 1: Verificar si este modo específico está presente
            mode_present, mode_confidence = self._is_specific_mode_present(field, magnitude)

            if mode_present:
                log_debug('decoder', f"Modo |l|={magnitude} PRESENTE (confianza: {mode_confidence:.3f})")

                # PASO 2: Determinar signo usando correlación normalizada
                log_debug('decoder', f"Llamando NCC para |l|={magnitude}")
                detected_sign, sign_confidence = self._detect_sign_correlation(field, magnitude)

                # PASO 3: Mapear a bit según reglas
                if detected_sign > 0:
                    bit_value = 1  # +l → bit = 1
                else:
                    bit_value = 0  # -l → bit = 0

                detected_bits[i] = bit_value
                detected_modes.append(magnitude * detected_sign)
                mode_confidences.append(mode_confidence * sign_confidence)

                log_debug('decoder', f"Signo detectado: {detected_sign:+d}")
                log_debug('decoder', f"Mapeo: l={detected_sign:+d}{magnitude} → bit[{i}] = {bit_value}")

            else:
                log_error('decoder', f"Modo |l|={magnitude} AUSENTE")

        # Resultado final
        log_info('decoder', f"RESULTADO COMPLETO:")
        log_debug('decoder', f"Modos detectados: {detected_modes}")
        log_debug('decoder', f"Bits generados: {detected_bits}")
        log_debug('decoder', f"Confianzas: {[f'{c:.3f}' for c in mode_confidences]}")

        return detected_modes, detected_bits, np.mean(mode_confidences) if mode_confidences else 0.0

    def _is_specific_mode_present(self, field, target_magnitude):
        """
        Presencia de |l|=target_magnitude por matched-filter COMPLEJO.
        Decide 'presente' si max(|<F, +l>|, |<F, -l>|) supera umbral.
        """
        N = field.shape[0]

        # Ventana/apodización suave (ver función más abajo o usa _create_circular_mask)
        def _hann_apod(N, inner=0.08, outer=0.85):
            center = N // 2
            y, x = np.ogrid[:N, :N]
            r = np.sqrt((x - center)**2 + (y - center)**2) / (N//2)
            w = np.zeros((N, N), dtype=float)
            core = r <= inner
            trans = (r > inner) & (r <= outer)
            w[core] = 1.0
            w[trans] = 0.5 * (1 + np.cos(np.pi * (r[trans] - inner) / (outer - inner)))
            return w

        w = _hann_apod(N, inner=0.08, outer=0.85)

        # Implementación: Aplicar máscara seleccionable por aperture_mode
        if self.aperture_mode == "circular_hann":
            # Campo COMPLEJO apodizado
            Fw = field * w
        else:
            # Para "none" y "rect_slit", usar método de apertura unificado
            Fw = self._apply_aperture_mask(field)
            w = np.ones_like(field.real)  # Para templates, usar máscara unitaria

        # Plantillas RX del caché (+l y -l), apodizadas
        if self.aperture_mode == "circular_hann":
            tpl_pos = self.template_cache[target_magnitude]['pos'] * w
            tpl_neg = self.template_cache[target_magnitude]['neg'] * w
        else:
            # Para otros modos, aplicar la misma máscara que al campo
            tpl_pos = self._apply_aperture_mask(self.template_cache[target_magnitude]['pos'])
            tpl_neg = self._apply_aperture_mask(self.template_cache[target_magnitude]['neg'])

        # Normalizaciones de energía
        nF = np.sqrt(np.sum(np.abs(Fw)**2)) or 1.0
        nP = np.sqrt(np.sum(np.abs(tpl_pos)**2)) or 1.0
        nN = np.sqrt(np.sum(np.abs(tpl_neg)**2)) or 1.0

        # Correlaciones COMPLEJAS con conjugado (matched filter)
        cpos = np.vdot(tpl_pos/nP, Fw/nF)  # vdot = conj(tpl) * Fw sum
        cneg = np.vdot(tpl_neg/nN, Fw/nF)

        mpos = np.abs(cpos)
        mneg = np.abs(cneg)
        best = max(mpos, mneg)

        # Umbral más sensato para presencia (ajústalo si hace falta)
        threshold = 0.15
        is_present = best > threshold

        log_debug('decoder', f"|<F,+{target_magnitude}>|={mpos:.3f}  |<F,-{target_magnitude}>|={mneg:.3f}  → present={is_present}")

        return is_present, best

    # === NEW API: Inverse bit flow functions ===
    
    def symbol_bits_from_modes(self, modes):
        """Convert OAM modes back to symbol bits (inverse of modes_from_symbol_bits)"""
        bits = []
        magnitudes = sorted(list(set([abs(l) for l in self.config.oam_channels if l != 0])))
        
        for i, magnitude in enumerate(magnitudes):
            if i < len(modes):
                mode = modes[i]
                # Extract sign: positive mode -> bit=1, negative mode -> bit=0
                if mode > 0:
                    bits.append(1)
                else:
                    bits.append(0)
            else:
                bits.append(0)  # Default for unused positions
        
        return bits
    
    def bits_to_bitstream(self, symbol_bits_list):
        """Convert list of symbol bit arrays back to continuous bitstream"""
        bitstream = []
        for sym_bits in symbol_bits_list:
            bitstream.extend(sym_bits)
        return bitstream
    
    def bitstream_to_bytes(self, bitstream):
        """Convert bitstream back to bytes (inverse of bitstream_from_bytes)

        LSB-first: bits[0] = bit0 (LSB), bits[7] = bit7 (MSB)

        Ejemplo:
            bits = [1,0,1,1,0,0,1,0] (LSB-first)
                    0 1 2 3 4 5 6 7 (posiciones)
            byte = bit0<<0 | bit1<<1 | ... | bit7<<7
                 = 1<<0 | 0<<1 | 1<<2 | 1<<3 | 0<<4 | 0<<5 | 1<<6 | 0<<7
                 = 0x4D
        """
        bytes_list = []
        # Process 8 bits at a time (LSB first)
        for i in range(0, len(bitstream), 8):
            byte_bits = bitstream[i:i+8]
            # Pad with zeros if incomplete byte
            while len(byte_bits) < 8:
                byte_bits.append(0)

            # Convert bits to byte (LSB first)
            byte_val = 0
            for j, bit in enumerate(byte_bits):
                byte_val |= (bit << j)  # LSB first: bit position j goes to bit j

            bytes_list.append(byte_val)

        return bytes_list
    
    def ascii_from_bytes(self, bytes_list):
        """Convert bytes back to ASCII string (inverse of bytes_from_ascii)"""
        try:
            return bytes(bytes_list).decode('ascii', errors='replace')
        except:
            return ''.join(chr(b) if 32 <= b <= 126 else '?' for b in bytes_list)

    def _detect_bits_matched_filter(self, field):
        """
        MAPEO FIJO POR SÍMBOLO (simbolo ↔ modos):
        - Orden de evaluación: |1|, |2|, |3|, |4| (magnitud creciente)
        - Construcción del simbolo (MSB→LSB): bit3=|4|, bit2=|3|, bit1=|2|, bit0=|1|
        - Regla: bit=1 si signo(l)='+', bit=0 si signo(l)='-'
        - Respeta: posición → |l|, valor → signo
        """
        N = field.shape[0]
        # Ventana de apodización
        def _hann_apod(N, inner=0.08, outer=0.85):
            center = N // 2
            y, x = np.ogrid[:N, :N]
            r = np.sqrt((x - center)**2 + (y - center)**2) / (N//2)
            w = np.zeros((N, N), dtype=float)
            core = r <= inner
            trans = (r > inner) & (r <= outer)
            w[core] = 1.0
            w[trans] = 0.5 * (1 + np.cos(np.pi * (r[trans] - inner) / (outer - inner)))
            return w

        # Implementación: Aplicar máscara seleccionable por aperture_mode
        if self.aperture_mode == "circular_hann":
            w = _hann_apod(N, inner=0.08, outer=0.85)
            Fw = field * w
        else:
            # Para "none" y "rect_slit", usar método de apertura unificado
            Fw = self._apply_aperture_mask(field)

        # GUARDAR CAMPO CON APERTURA/SLIT EN PIPELINE BUFFER
        from pipeline import pipeline
        pipeline.push_field("after_slit", Fw)

        # ORDEN FIJO: |1|, |2|, |3|, |4| (usando constantes globales)
        magnitudes = sorted(list(set([abs(l) for l in self.config.oam_channels if l != 0])))
        bits = []

        nF = np.sqrt(np.sum(np.abs(Fw)**2)) or 1.0

        # GUARDAR CAMPO NORMALIZADO EN DECODER con verificación de seguridad
        try:
            normalized_field = Fw/nF if nF > 0 else Fw
            if np.isfinite(normalized_field).all():
                pipeline.push_field("at_decoder", normalized_field)
            else:
                log_warning('decoder', "Campo normalizado contiene valores inválidos, guardando sin normalizar")
                pipeline.push_field("at_decoder", Fw)
        except Exception as e:
            log_warning('decoder', f"Error al normalizar campo: {e}, guardando campo original")
            pipeline.push_field("at_decoder", Fw)

        for i, mag in enumerate(magnitudes):
            if self.aperture_mode == "circular_hann":
                pos = self.template_cache[mag]['pos'] * w
                neg = self.template_cache[mag]['neg'] * w
            else:
                # Para otros modos, aplicar la misma máscara que al campo
                pos = self._apply_aperture_mask(self.template_cache[mag]['pos'])
                neg = self._apply_aperture_mask(self.template_cache[mag]['neg'])
            nP = np.sqrt(np.sum(np.abs(pos)**2)) or 1.0
            nN = np.sqrt(np.sum(np.abs(neg)**2)) or 1.0

            cpos = np.vdot(pos/nP, Fw/nF)
            cneg = np.vdot(neg/nN, Fw/nF)

            # bit=1 si +l gana, bit=0 si -l gana
            mag_pos = np.abs(cpos)
            mag_neg = np.abs(cneg)
            bit = 1 if mag_pos >= mag_neg else 0
            bits.append(bit)

            # LOG NCC EN PIPELINE
            from pipeline import pipeline
            pipeline.log_ncc(mag, mag_pos, mag_neg)

            # Registro de mapeo de correlación a bit
            winner = '+' if bit == 1 else '-'
            log_debug('decoder', f"|{mag}|: +{mag}={mag_pos:.3f}, -{mag}={mag_neg:.3f} → {winner}{mag} → bit{i}={bit}")

        # bits = [b0, b1, b2, ...] donde bi corresponde a |i+1|
        # Construir simbolo dinámicamente basado en el número de bits disponibles
        symbol_value = 0
        for i in range(len(bits)):
            symbol_value |= (bits[i] << i)
        log_debug('decoder', f"Símbolo reconstruido: bits=[{','.join(map(str,bits))}] → 0x{symbol_value:X}")

        return bits

    def _detect_sign_correlation(self, field, magnitude_l):
        """
        Método de detección corregido: Detección de signo por correlación normalizada (NCC)
        Usa la MISMA física que el encoder (Laguerre, curvatura, Gouy)
        con re-escalado por expansión del haz según la distancia de propagación
        """
        # DIAGNÓSTICO COMPLETO
        import inspect, sys, traceback

        log_debug('decoder', f"file: {__file__}")
        log_debug('decoder', f"_detect_sign_correlation: {inspect.getfile(self._detect_sign_correlation)}")

        log_debug('decoder', f"ENTRANDO A _detect_sign_correlation |l|={magnitude_l}")

        try:
            # Verificar que field es complejo (no intensidad)
            assert np.iscomplexobj(field), "ERROR: field NO es complejo (llegó intensidad)"

            # Generar plantillas con TX physics
            template_pos = self._generate_template_tx_physics(magnitude_l, +1)
            template_neg = self._generate_template_tx_physics(magnitude_l, -1)

            # Verificar tamaños exactos
            assert template_pos.shape == field.shape == template_neg.shape, "shape mismatch"

            # Aplicar máscara consistente
            mask = self._create_circular_mask(self.config.grid_size, inner_radius=0.1, outer_radius=0.8)
            F = field * mask
            Tpos = template_pos * mask
            Tneg = template_neg * mask

            # Calcular normas
            nf = np.sqrt(np.sum(np.abs(F)**2)) + 1e-12
            np1 = np.sqrt(np.sum(np.abs(Tpos)**2)) + 1e-12
            nn1 = np.sqrt(np.sum(np.abs(Tneg)**2)) + 1e-12

            # Productos coherentes
            zpos = np.sum(F * np.conj(Tpos))
            zneg = np.sum(F * np.conj(Tneg))

            # Dos métricas: parte real y magnitud
            ncc_pos = np.real(zpos) / (nf*np1)
            ncc_neg = np.real(zneg) / (nf*nn1)
            mag_pos = np.abs(zpos) / (nf*np1)
            mag_neg = np.abs(zneg) / (nf*nn1)

            # Decisión robusta a fase global: usa magnitud
            if mag_pos > mag_neg:
                detected_sign, confidence = +1, mag_pos
            else:
                detected_sign, confidence = -1, mag_neg

            log_debug('decoder', f"|l|={magnitude_l}: Re{{·}} pos={ncc_pos:.3f} neg={ncc_neg:.3f} |·| pos={mag_pos:.3f} neg={mag_neg:.3f} → sign={detected_sign:+d} conf={confidence:.3f}")

            # NCC ya se registra en el caller, no duplicar aquí
            return detected_sign, float(confidence)

        except Exception:
            log_error('decoder', f"EXCEPTION: {traceback.format_exc()}")
            raise

    def _generate_template_tx_physics(self, l, sign):
        """
        Genera plantilla usando la MISMA física que el encoder
        Incluye: Laguerre polynomials, curvatura R(z), fase Gouy, w(z)
        Re-escalado por propagation_distance
        """
        # Usar MISMOS parámetros que el encoder
        wavelength = self.config.wavelength
        tx_beam_waist = self.config.tx_beam_waist
        tx_aperture_size = self.config.tx_aperture_size
        grid_size = self.config.grid_size

        # Distancia de propagación (desde oam_complete_system.py: 20m)
        z = getattr(self.config, 'propagation_distance', 20.0)

        # Parámetros físicos con expansión del haz
        k = 2 * np.pi / wavelength
        zR = np.pi * tx_beam_waist**2 / wavelength

        # Expansión del haz a distancia z
        w_z = tx_beam_waist * np.sqrt(1 + (z/zR)**2)
        R_z = z * (1 + (zR/z)**2) if z != 0 else float('inf')
        gouy_phase = (abs(l) + 1) * np.arctan(z/zR)  # p=0 para modos fundamentales

        # Grilla espacial (usar apertura TX original)
        x = np.linspace(-tx_aperture_size/2, tx_aperture_size/2, grid_size)
        y = np.linspace(-tx_aperture_size/2, tx_aperture_size/2, grid_size)
        X, Y = np.meshgrid(x, y)

        # Coordenadas cilíndricas
        r = np.sqrt(X**2 + Y**2)
        phi = np.arctan2(Y, X)

        # Fórmula EXACTA del encoder (con signo aplicado)
        signed_l = sign * l
        p = 0  # modos fundamentales

        # Normalización
        norm = np.sqrt(2 * special.factorial(p) / (np.pi * special.factorial(p + abs(signed_l))))

        # Componente radial con Laguerre
        rho = np.sqrt(2) * r / w_z
        radial = rho**abs(signed_l) * np.exp(-rho**2/2)
        laguerre = special.eval_genlaguerre(p, abs(signed_l), rho**2)

        # Curvatura del frente de onda
        if np.isfinite(R_z):
            curvature = np.exp(1j * k * r**2 / (2 * R_z))
        else:
            curvature = 1

        # Componente azimutal
        azimuthal = np.exp(1j * signed_l * phi)

        # Fase de Gouy
        gouy = np.exp(-1j * gouy_phase)

        # Campo completo (IGUAL que en encoder)
        template = norm * radial * laguerre * curvature * azimuthal * gouy / w_z

        return template

    def _compute_ncc(self, field, template):
        """
        Calcula correlación cruzada normalizada (NCC)
        NCC(F, R) = Re{sum(F * R*)} / (||F||2 * ||R||2)
        """
        # Producto interno complejo
        cross_corr = np.sum(field * np.conj(template))

        # Normas
        norm_field = np.sqrt(np.sum(np.abs(field)**2))
        norm_template = np.sqrt(np.sum(np.abs(template)**2))

        # NCC (parte real del producto normalizado)
        if norm_field > 0 and norm_template > 0:
            ncc = np.real(cross_corr) / (norm_field * norm_template)
        else:
            ncc = 0.0

        return ncc

    def _create_fixed_triangular_mask(self, N):
        """
        Crear máscara triangular fija para simular hardware real
        Triángulo equilátero centrado, orientación estándar
        """
        center = N // 2
        radius = N // 4  # Radio característico del triángulo

        # Crear triángulo equilátero con vértices a 0°, 120°, 240°
        angles = np.array([0, 120, 240]) * np.pi / 180
        vertices = []

        for angle in angles:
            x = center + radius * np.cos(angle)
            y = center + radius * np.sin(angle)
            vertices.append([x, y])

        # Crear máscara usando Path (polígono)
        from matplotlib.path import Path
        y, x = np.mgrid[:N, :N]
        points = np.column_stack([x.ravel(), y.ravel()])
        path = Path(vertices)
        mask = path.contains_points(points).reshape(N, N).astype(float)

        return mask

    def _get_tri_mask(self, N):
        if not hasattr(self, "_tri_mask_cache"):
            self._tri_mask_cache = {}
        if N not in self._tri_mask_cache:
            self._tri_mask_cache[N] = self._create_fixed_triangular_mask(N)
        return self._tri_mask_cache[N]

    def _get_tripod_ref_orientation(self, magnitude_l, sign, N):
        if not hasattr(self, "_tripod_ref_cache"):
            self._tripod_ref_cache = {}
        key = (magnitude_l, sign, N)
        if key not in self._tripod_ref_cache:
            patt = self._generate_reference_tripod_pattern(magnitude_l, sign, N)
            self._tripod_ref_cache[key] = self._extract_tripod_orientation(patt)
        return self._tripod_ref_cache[key]

    def _generate_reference_tripod_pattern(self, magnitude_l, sign, N):
        """
        Generar patrón trípode de referencia para l*sign
        Usa generador RX y máscara triangular consistente
        """
        # Campo LG con parámetros RX
        ref_field = self._generate_laguerre_gaussian_rx(N, sign * magnitude_l)
        # Misma máscara triangular fija
        tri_mask = self._get_tri_mask(N)
        far = np.fft.fftshift(np.fft.fft2(ref_field * tri_mask))
        return (np.abs(far) ** 2)

    def _extract_tripod_orientation(self, intensity_pattern):
        """
        Extraer orientación del patrón trípode usando análisis polar
        Encuentra la orientación dominante de la estructura triangular
        """
        N = intensity_pattern.shape[0]
        center = N // 2

        # Convertir a coordenadas polares
        y, x = np.mgrid[:N, :N]
        x = x - center
        y = y - center
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        # Análisis en anillo (región de interés)
        r_min, r_max = N//8, N//3
        ring_mask = (r >= r_min) & (r <= r_max)

        # Extraer intensidades en el anillo
        theta_ring = theta[ring_mask]
        intensity_ring = intensity_pattern[ring_mask]

        # Histograma angular ponderado por intensidad
        angles = np.linspace(-np.pi, np.pi, 360)
        weighted_histogram = np.zeros_like(angles)

        for i, angle in enumerate(angles):
            # Ventana angular pequeña
            angle_window = 0.1  # ~6 grados
            mask = np.abs(theta_ring - angle) < angle_window
            if np.sum(mask) > 0:
                weighted_histogram[i] = np.sum(intensity_ring[mask])

        # Encontrar orientación dominante
        max_idx = np.argmax(weighted_histogram)
        orientation = angles[max_idx] * 180 / np.pi

        # Normalizar a 0-360°
        if orientation < 0:
            orientation += 360

        return orientation

    def _angular_difference(self, angle1, angle2):
        """
        Calcular diferencia angular mínima considerando periodicidad
        """
        diff = abs(angle1 - angle2)
        if diff > 180:
            diff = 360 - diff
        return diff

    def _calculate_angular_stability(self, intensity_pattern, base_orientation):
        """
        Calcular confianza basada en derivada angular (estabilidad del trípode)
        Mide pequeñas rotaciones ±Δθ y promedia
        """
        delta_theta = 2.0  # ±2 grados

        # Rotar pattern ligeramente
        from scipy.ndimage import rotate
        pattern_plus = rotate(intensity_pattern, delta_theta, reshape=False)
        pattern_minus = rotate(intensity_pattern, -delta_theta, reshape=False)

        # Extraer orientaciones
        orient_plus = self._extract_tripod_orientation(pattern_plus)
        orient_minus = self._extract_tripod_orientation(pattern_minus)

        # Calcular estabilidad (menor variación = mayor confianza)
        variation = abs(orient_plus - orient_minus)
        if variation > 180:
            variation = 360 - variation

        # Convertir a confianza (0-1)
        stability = max(0, 1.0 - variation / 45.0)  # Normalizar por 45°

        return stability

    def _precalculate_enhanced_templates(self, magnitudes):
        """
        Precalcula plantillas incrementales con mejoras aplicadas
        """
        self._enhanced_templates_cache = {}
        log_debug('decoder', f"Precalculando plantillas MEJORADAS para magnitudes: {magnitudes}")

        # Crear máscara una vez para todas las plantillas
        mask = self._create_circular_mask(self.config.grid_size,
                                          inner_radius=0.1,
                                          outer_radius=0.7)

        for count in range(1, len(magnitudes) + 1):
            # Usar los primeros 'count' modos con signo positivo para la plantilla
            modes_subset = [+magnitudes[i] for i in range(count)]
            log_debug('decoder', f"Enhanced_Template_{count}: modos {modes_subset}")

            # Generar campo combinado para esta cantidad de modos
            combined_field = np.zeros((self.config.grid_size, self.config.grid_size), dtype=complex)

            for mode in modes_subset:
                # Generar campo LG para este modo - USAR RX
                lg_field = self._generate_laguerre_gaussian_rx(
                    self.config.grid_size, mode  # N, l=mode
                )
                # Sumar al campo combinado
                combined_field += lg_field

            # Aplicar normalización fotométrica a la plantilla
            field_normalized = self._apply_photometry_normalization(combined_field)

            # Aplicar máscara circular
            field_masked = field_normalized * mask

            # Calcular norma final
            norm_masked = np.sqrt(np.sum(field_masked**2))

            # Guardar en cache mejorado
            self._enhanced_templates_cache[count] = {
                'field_raw': combined_field,
                'field_normalized': field_normalized,
                'field_masked': field_masked,
                'norm_masked': norm_masked,
                'modes': modes_subset
            }

        log_info('decoder', f"Cache de plantillas MEJORADAS listo: {len(self._enhanced_templates_cache)} templates")

    def _generate_laguerre_gaussian_rx(self, N, l):
        """
        Genera patrón Laguerre-Gaussian EN EL PLANO TX
        Para matching con campos que han sido propagados via Angular Spectrum
        (que ya incluye automáticamente curvatura y fase)
        """
        # Usar parámetros TX originales
        tx_beam_waist = self.config.tx_beam_waist  # Cintura en TX
        tx_aperture_size = self.config.tx_aperture_size  # Apertura TX

        # Grilla TX (igual que encoder)
        x = np.linspace(-tx_aperture_size/2, tx_aperture_size/2, N)
        y = np.linspace(-tx_aperture_size/2, tx_aperture_size/2, N)
        X, Y = np.meshgrid(x, y)

        # Coordenadas cilíndricas
        r = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)

        # LG en plano TX (z=0, sin curvatura, sin Gouy)
        rho = np.sqrt(2) * r / tx_beam_waist
        radial = (rho**abs(l)) * np.exp(-rho**2/2)
        angular = np.exp(1j * l * theta)
        norm = np.sqrt(2 / np.pi) / tx_beam_waist

        return norm * radial * angular

    def decode(self, field, modulation=None, modes_per_symbol=None):
        """
        Decodifica simbolos OAM a mensaje ASCII usando flujo directo de bits.

        Parametros:
            field: Array de simbolos OAM recibidos.
            modulation: Tipo de modulacion (opcional).
            modes_per_symbol: Numero de modos por simbolo (auto-detectado si es None).

        Retorna:
            Mensaje decodificado como bytes.

        Notas:
            - Soporta auto-deteccion de M=1,2,3,4 modos por simbolo.
            - Usa metodo enhanced con fotometria y mascarado circular.
        """

        if len(field) == 0:
            return b''

        # Obtener todas las magnitudes disponibles para mapeo de bits
        magnitudes = sorted(list(set([abs(l) for l in self.config.oam_channels if l != 0])))

        # Determinar modes_per_symbol
        from pipeline import pipeline
        detected_modes_per_symbol = modes_per_symbol

        # Auto-detectar M desde metadatos del pipeline
        if detected_modes_per_symbol is None and hasattr(pipeline, 'symbol_metadata') and pipeline.symbol_metadata:
            for meta in pipeline.symbol_metadata:
                if 'bits' in meta and isinstance(meta['bits'], list):
                    detected_modes_per_symbol = len(meta['bits'])
                    break

        # Fallback: usar todas las magnitudes disponibles
        if detected_modes_per_symbol is None:
            detected_modes_per_symbol = len(magnitudes)
            log_warning('decoder', f"Auto-detectando M={detected_modes_per_symbol} desde magnitudes disponibles")

        log_info('decoder', f"DIRECT BIT FLOW: M={detected_modes_per_symbol} modos por símbolo")
        return self._decode_direct_bit_flow(field, detected_modes_per_symbol)
    
    def _decode_direct_bit_flow(self, field, modes_per_symbol):
        """Decode using direct bit flow: bits->bytes->ASCII"""
        log_info('decoder', f"Decodificando {len(field)} símbolos con M={modes_per_symbol}")

        # Access pipeline metadata
        from pipeline import pipeline
        if not hasattr(pipeline, 'symbol_metadata') or not pipeline.symbol_metadata:
            log_error('decoder', "No symbol metadata - system requires complete pipeline")
            return b''

        symbol_metadata = pipeline.symbol_metadata
        log_info('decoder', f"Available metadata: {len(symbol_metadata)} symbols")

        # STEP 1: Detect bits from all symbols
        all_detected_bits = []
        for s in range(min(len(field), len(symbol_metadata))):
            # Detectar bits usando filtro acoplado limitado a modes_per_symbol
            full_bits = self._detect_bits_matched_filter(field[s])
            # Usar solo los primeros modes_per_symbol bits
            symbol_bits = full_bits[:modes_per_symbol]
            all_detected_bits.append(symbol_bits)

        # STEP 2: Filter only data symbols
        data_symbol_bits = []
        for s, meta in enumerate(symbol_metadata):
            if meta['kind'] == 'datos' and s < len(all_detected_bits):
                data_symbol_bits.append(all_detected_bits[s])

        log_info('decoder', f"Data symbols detected: {len(data_symbol_bits)}")

        # STEP 3: Reconstruct bitstream from symbols
        bitstream = []
        for symbol_bits in data_symbol_bits:
            bitstream.extend(symbol_bits)

        log_info('decoder', f"Bitstream reconstructed: {len(bitstream)} bits")

        # STEP 4: Convert bitstream to bytes (8 bits per byte, LSB first)
        bytes_list = []
        for i in range(0, len(bitstream), 8):
            if i + 8 <= len(bitstream):
                byte_bits = bitstream[i:i+8]
                byte_val = 0
                for j, bit in enumerate(byte_bits):
                    byte_val |= bit << j  # LSB first: bit position j goes to bit j
                bytes_list.append(byte_val)

        log_info('decoder', f"Bytes reconstructed: {len(bytes_list)} bytes")

        # STEP 5: Extract payload between STX/ETX
        data_bytes = bytes(bytes_list)
        log_info('decoder', f"Complete data: {data_bytes.hex().upper()}")

        try:
            stx_pos = data_bytes.index(0x02)  # STX
            etx_pos = data_bytes.index(0x03, stx_pos+1)  # ETX after STX
            payload = data_bytes[stx_pos+1:etx_pos]  # Extract content between markers
            log_info('decoder', f"Payload extracted: {payload.hex().upper()}")
        except ValueError:
            log_warning('decoder', "STX/ETX markers not found - using complete data")
            payload = data_bytes

        log_info('decoder', f"MENSAJE FINAL: {payload}")
        return payload
    
    def _decode_symbol_flow(self, field):
        """
        Decodifica usando flujo de simbolos de 4 bits por simbolo.

        Parametros:
            field: Array de simbolos OAM recibidos.

        Retorna:
            Mensaje decodificado como bytes.

        Notas:
            - Modo de compatibilidad con implementaciones anteriores.
        """
        # DIAGNÓSTICO DE ARCHIVO (Paso 1 del checklist)
        import inspect, sys
        log_debug('decoder', f"file: {__file__}")
        log_debug('decoder', f"_detect_sign_correlation: {inspect.getfile(self._detect_sign_correlation)}")
        log_debug('decoder', f"Lista de módulos: {[m for m in sys.modules if m.endswith('oam_decoder')]}")

        log_info('decoder', "Decodificando con Método de detección mejorado (fotometría + mascarado circular)")

        # Obtener todas las magnitudes disponibles para mapeo de bits
        magnitudes = sorted(list(set([abs(l) for l in self.config.oam_channels if l != 0])))

        # === PROCESAMIENTO POR METADATOS ÚNICAMENTE ===

        # --- PROCESAMIENTO REALISTA POR ETIQUETAS ---
        log_info('decoder', "PROCESAMIENTO REALISTA")
        log_info('decoder', f"Total símbolos recibidos: {len(field)}")
        
        # Acceder a metadatos del pipeline
        from pipeline import pipeline
        if not hasattr(pipeline, 'symbol_metadata') or not pipeline.symbol_metadata:
            log_error('decoder', "No hay metadatos de símbolos - sistema requiere pipeline completo")
            return b''
        
        symbol_metadata = pipeline.symbol_metadata
        log_info('decoder', f"Metadatos disponibles: {len(symbol_metadata)} símbolos")
        
        # === Paso 1: DETECTAR BITS RAW DE TODOS LOS SÍMBOLOS ===
        raw_detections = []
        for s in range(min(len(field), len(symbol_metadata))):
            detected_bits = self._detect_bits_matched_filter(field[s])
            raw_detections.append(detected_bits)
        
        # === Paso 2: LOGS MÍNIMOS OBLIGATORIOS ===
        log_info('decoder', f"[TAG] DETECCIONES RAW POR SÍMBOLO:")
        for s in range(len(symbol_metadata)):
            sym_meta = symbol_metadata[s]
            kind = sym_meta['kind']
            detected_bits = raw_detections[s] if s < len(raw_detections) else [0,0,0,0]
            
            # Clasificar tipo de símbolo
            is_preamble = (kind == "piloto_cero")
            is_start = (kind == "piloto_signo")
            is_data = (kind == "datos")
            is_stop = False  # No tenemos stop frames por ahora
            
            # Signos raw antes de calibrar
            signs_raw = ['+' if b==1 else '-' for b in detected_bits]
            
            log_debug('decoder', f"[TAG] sym={s} pre={is_preamble} start={is_start} data={is_data} stop={is_stop} bits={detected_bits} signs={signs_raw}")

        # === Paso 3: CALIBRACIÓN DE SIGNO EXPLÍCITA CON PILOTOS ===
        sign_map = np.array([1, 1, 1, 1])  # Inicializar como sin inversión
        pilot_symbols = [s for s, meta in enumerate(symbol_metadata) if meta['kind'] == "piloto_signo"]
        
        if pilot_symbols:
            log_info('decoder', f"[CALIBRATION] Calibrando con pilotos en símbolos: {pilot_symbols}")

            # DERIVAR PILOTO DESDE PIPELINE (no hardcodear)
            try:
                from pipeline import pipeline
                if hasattr(pipeline, 'symbol_metadata') and len(pipeline.symbol_metadata) > 1:
                    pilot_meta = pipeline.symbol_metadata[1]  # Símbolo 1 = pilot_sign
                    expected_pilot = pilot_meta.get('bits', [1, 1, 1, 1])
                    log_info('decoder', f"[CALIBRATION] Piloto desde pipeline: {expected_pilot}")
                else:
                    expected_pilot = [1, 1, 1, 1]  # Fallback
                    log_info('decoder', f"[CALIBRATION] Piloto fallback (sin pipeline): {expected_pilot}")
            except:
                expected_pilot = [1, 1, 1, 1]  # Fallback
                log_info('decoder', f"[CALIBRATION] Piloto fallback (error pipeline): {expected_pilot}")
            
            # Usar primer piloto para calibración
            pilot_idx = pilot_symbols[0]
            if pilot_idx < len(raw_detections):
                measured_pilot = raw_detections[pilot_idx]
                
                # Construir sign_map: +1 si coincide, -1 si hay que invertir
                for j in range(4):
                    if measured_pilot[j] != expected_pilot[j]:
                        sign_map[j] = -1  # Hay que invertir este modo
                        log_info('decoder', f"[CALIBRATION] Modo |{j+1}|: INVERTIR (det={measured_pilot[j]} vs exp={expected_pilot[j]})")
                    else:
                        sign_map[j] = +1   # Este modo está bien
                        log_info('decoder', f"[CALIBRATION] Modo |{j+1}|: OK (det={measured_pilot[j]} vs exp={expected_pilot[j]})")
                
                log_info('decoder', f"[CALIBRATION] sign_map final: {sign_map}")
        
        # === Paso 4: FILTRADO DE TRAMA SOLO CON ETIQUETAS ===
        data_symbols = [(s, meta) for s, meta in enumerate(symbol_metadata) if meta['kind'] == "datos"]
        log_info('decoder', f"[FILTER] Símbolos de datos encontrados: {len(data_symbols)} de {len(symbol_metadata)} total")

        # Verificaciones de sanidad
        assert all(not (meta['kind'] == "piloto_cero") for _, meta in data_symbols), "Los datos no deben incluir preámbulo"
        assert all(not (meta['kind'] == "piloto_signo") for _, meta in data_symbols), "Los datos no deben incluir pilotos"
        assert len(data_symbols) * 4 % 8 == 0, f"Bits de datos ({len(data_symbols)*4}) debe ser múltiplo de 8"
        
        # === Paso 5: ORDEN DE BITS CONSISTENTE (LSB-FIRST) ===
        data_symbols = []
        for s, meta in data_symbols:
            if s < len(raw_detections):
                detected_bits = raw_detections[s]
                
                # Aplicar calibración: bits_corr = (sign_map * signos_medidos > 0)
                signs_measured = np.array([1 if b==1 else -1 for b in detected_bits])  # Convertir a ±1
                bits_corr = (sign_map * signs_measured > 0).astype(int)  # 1 si coincide, 0 si invierte
                
                # Mapeo fijo: |1|→b0, |2|→b1, |3|→b2, |4|→b3
                b0, b1, b2, b3 = bits_corr[0], bits_corr[1], bits_corr[2], bits_corr[3]
                
                # Simbolo (LSB-first): symbol_val = (b0 << 0) | (b1 << 1) | (b2 << 2) | (b3 << 3)
                symbol_val = (b0 << 0) | (b1 << 1) | (b2 << 2) | (b3 << 3)
                data_symbols.append(symbol_val)
                
                # Log de calibración aplicada
                signs_corr = ['+' if b==1 else '-' for b in bits_corr]
                log_debug('decoder', f"[CALIBRATION] sym={s} raw={detected_bits} → corr={list(bits_corr)} signs={signs_corr} → symbol=0x{symbol_val:X}")
                
                # Verificar contra ground truth si disponible
                if meta.get('symbol_value') is not None:
                    expected_symbol = meta['symbol_value']
                    match = "" if symbol_val == expected_symbol else "ERROR"
                    log_info('decoder', f"[VERIFY] Ground truth: 0x{expected_symbol:X} {match}")
        
        log_info('decoder', "SÍMBOLOS DE DATOS EXTRAÍDOS")
        log_info('decoder', f"Simbolos: {[f'0x{n:X}' for n in data_symbols]}")
        
        # === ENSAMBLADO CORRECTO: SIMBOLOS → BYTES (igual que encoder) ===
        bytes_data = []
        for i in range(0, len(data_symbols)-1, 2):  # -1 para evitar índice fuera de rango
            hi = data_symbols[i]      # Primer simbolo = simbolo alto
            lo = data_symbols[i+1]    # Segundo simbolo = simbolo bajo
            
            # Emparejar según SYMBOL_ENDIAN (igual que encoder: hi primero)
            if SYMBOL_ENDIAN == "MSB":
                byte_val = (hi << 4) | lo
            else:
                byte_val = (lo << 4) | hi
            
            bytes_data.append(byte_val)
            
            char_repr = chr(byte_val) if 32 <= byte_val <= 126 else '?'
            log_debug('decoder', f"Byte {len(bytes_data)-1}: simbolos[{i},{i+1}] = 0x{hi:X},0x{lo:X} → 0x{byte_val:02X} ('{char_repr}')")
        
        # === EXTRACCIÓN POR STX/ETX (no por índices de frames) ===
        data_bytes = bytes(bytes_data)
        log_info('decoder', "EXTRACCIÓN STX/ETX")
        log_info('decoder', f"Datos completos: {data_bytes.hex().upper()}")
        
        try:
            s = data_bytes.index(0x02)  # STX
            e = data_bytes.index(0x03, s+1)  # ETX después de STX
            payload = data_bytes[s+1:e]  # Extraer solo el contenido entre marcadores
            log_info('decoder', f"STX encontrado en posición: {s}")
            log_info('decoder', f"ETX encontrado en posición: {e}")
            log_info('decoder', f"Payload extraído: {payload.hex().upper()}")
        except ValueError:
            log_warning('decoder', "No se encontraron marcadores STX/ETX - usando datos completos")
            payload = data_bytes
        
        # Resultado final
        message = payload
        log_info('decoder', f"MENSAJE FINAL: {message}")
        
        try:
            decoded_text = message.decode('utf-8', errors='replace')
            log_info('decoder', f"TEXTO DECODIFICADO: '{decoded_text}'")
        except:
            log_warning('decoder', "No se puede decodificar como texto UTF-8")

        # Almacenar simbolos para BER posterior
        self.last_decoded_symbols = data_symbols

        return message
    
    # LEGACY CODE REMOVED - Solo procesamiento por metadatos

    def _start_pipeline_monitor(self):
        """Inicia monitor del pipeline en hilo separado - VERSIÓN SEGURA"""
        def monitor():
            try:
                import time
                processing_started = False
                start_time = time.time()
                timeout = 30  # 30 segundos máximo

                log_info('decoder', "Monitor iniciado - buscando pipeline")

                while (not hasattr(self, 'message_decoded') and
                       not processing_started and
                       (time.time() - start_time) < timeout):
                    try:
                        # Verificar si el pipeline global existe y está listo
                        from pipeline import pipeline

                        log_debug('decoder', f"stage={getattr(pipeline, 'processing_stage', 'None')}, symbols={hasattr(pipeline, 'channel_symbols')}")

                        if (hasattr(pipeline, 'processing_stage') and
                            pipeline.processing_stage == "channel_complete" and
                            hasattr(pipeline, 'channel_symbols') and
                            pipeline.channel_symbols is not None):

                            log_info('decoder', "Pipeline detectado - iniciando procesamiento automático")
                            processing_started = True
                            self._process_pipeline_data()
                            break

                    except ImportError:
                        # Pipeline no disponible aún
                        log_debug('decoder', "Pipeline no disponible")
                        pass
                    except Exception as e:
                        log_warning('decoder', f"Error en monitor: {e}")
                        break

                    time.sleep(0.2)  # Check every 200ms (menos frecuente)

                if (time.time() - start_time) >= timeout:
                    log_warning('decoder', "TIMEOUT: Monitor timeout - pipeline no detectado")

                log_info('decoder', "DETENIDO: Monitor terminado")

            except Exception as e:
                log_error('decoder', f"Error crítico en monitor: {e}")
                import traceback
                traceback.print_exc()

        import threading
        monitor_thread = threading.Thread(target=monitor, daemon=True, name="OAM_Monitor")
        monitor_thread.start()
        log_info('decoder', "Monitor thread iniciado")

    def work(self, input_items, output_items):
        """GNU Radio work - MINIMAL: Solo pass-through para GUI"""
        ninput_items = len(input_items[0]) if len(input_items[0]) > 0 else 0
        if ninput_items > 0:
            output_items[0][:ninput_items] = input_items[0][:ninput_items]
        return ninput_items

    def decode_message(self, received_symbols, modes_per_symbol=None):
        """
        Decodifica simbolos directamente sin depender de GNU Radio.

        Parametros:
            received_symbols: Array de simbolos OAM recibidos.
            modes_per_symbol: Numero de modos por simbolo (opcional).

        Retorna:
            Mensaje decodificado como bytes.
        """
        log_info('decoder', f"Decodificación directa: {len(received_symbols)} símbolos")
        return self.decode(received_symbols, modes_per_symbol=modes_per_symbol)

    def _init_template_cache(self):
        """Genera cache de plantillas una sola vez para reutilizar"""
        import time
        cache_start = time.time()

        self.template_cache = {}
        magnitudes = sorted(list(set([abs(l) for l in self.config.oam_channels if l != 0])))

        for mag in magnitudes:
            # Generar plantillas +l y -l con parámetros RX
            pos_template = self._generate_laguerre_gaussian_rx(self.config.grid_size, +mag)
            neg_template = self._generate_laguerre_gaussian_rx(self.config.grid_size, -mag)

            self.template_cache[mag] = {
                'pos': pos_template,
                'neg': neg_template
            }

        cache_end = time.time()
        log_info('decoder', f"Cache generado en {cache_end-cache_start:.2f}s para {len(magnitudes)} magnitudes")

    def _process_pipeline_data(self):
        """PROCESAMIENTO AUTOMÁTICO: Decodifica datos del pipeline directamente"""
        import time
        total_start = time.time()
        log_info('decoder', f"INICIO procesamiento pipeline: {time.strftime('%H:%M:%S')}")
        try:
            if not hasattr(pipeline, 'channel_symbols') or pipeline.channel_symbols is None:
                log_warning('decoder', "No hay datos en el pipeline")
                return

            channel_symbols = pipeline.channel_symbols
            if len(channel_symbols) == 0:
                log_warning('decoder', "Pipeline vacío")
                return

            log_info('decoder', f"PIPELINE: Procesando {len(channel_symbols)} símbolos automáticamente")

            # Procesar todos los símbolos
            decode_start = time.time()
            log_debug('decoder', f"Preparando datos: {time.strftime('%H:%M:%S')}")
            field_3d = np.array(channel_symbols)  # Todos los símbolos

            log_debug('decoder', f"Iniciando decode: {time.strftime('%H:%M:%S')}")
            message = self.decode(field_3d)
            decode_end = time.time()
            log_info('decoder', f"Decode completado en {decode_end-decode_start:.2f}s: {time.strftime('%H:%M:%S')}")

            if len(message) > 0:
                symbols_for_ber = getattr(self, 'last_decoded_symbols', [])
                self._measure_performance(message, symbols_for_ber)
                log_info('decoder', f"MENSAJE DECODIFICADO: {len(message)} bytes")
                log_info('decoder', f"Contenido: {message}")

                # Guardar resultado
                with open('datos_decodificados.bin', 'wb') as f:
                    f.write(message)

                self.message_decoded = True
            else:
                log_error('decoder', "No se pudo decodificar el mensaje")

        except Exception as e:
            log_error('decoder', f"Error en procesamiento automático: {e}")
            import traceback
            traceback.print_exc()

    def _measure_performance(self, message, data_symbols):
        """Medición de rendimiento (BER, SNR, diafonía)"""
        # === BER: Medición exacta usando pipeline ===
        try:
            from pipeline import pipeline
            tx_symbols = getattr(pipeline, "original_symbols", None)
            rx_symbols = data_symbols  # Los símbolos decodificados (solo datos, sin preámbulo/piloto)

            if tx_symbols and rx_symbols:
                # Extraer solo simbolos de datos (eliminar dummy + pilot)
                data_start_idx = 2  # Saltar dummy[0] + pilot[1]
                tx_data_symbols = tx_symbols  # Original ya son puros datos con STX/ETX
                rx_data_symbols = rx_symbols  # Ya filtrados en decode()

                # Alinear longitudes
                min_len = min(len(tx_data_symbols), len(rx_data_symbols))
                if min_len > 0:
                    bit_err = 0
                    for i in range(min_len):
                        t_symbol = tx_data_symbols[i]
                        r_symbol = rx_data_symbols[i]
                        # Contar bits diferentes usando XOR + popcount
                        bit_err += bin(t_symbol ^ r_symbol).count("1")

                    total_bits = 4 * min_len  # 4 bits por simbolo
                    ber = bit_err / total_bits if total_bits > 0 else float('nan')

                    log_info('decoder', f"[BER] MEDICIÓN EXACTA (pipeline):")
                    log_info('decoder', f"[BER] simbolos_comparados: {min_len}")
                    log_info('decoder', f"[BER] bit_errors: {bit_err}")
                    log_info('decoder', f"[BER] total_bits: {total_bits}")
                    log_info('decoder', f"[BER] BER_exacto: {ber:.4f} ({bit_err} errores sobre {total_bits} bits)")
                    if bit_err > 0:
                        log_info('decoder', f"[BER] BER_dB: {-10 * np.log10(ber):.1f} dB")
                else:
                    log_warning('decoder', "[BER] Sin datos válidos para comparar (longitud cero)")
            else:
                log_warning('decoder', "[BER] No hay ground truth en pipeline o datos vacíos")
                if not tx_symbols:
                    log_warning('decoder', "[BER] - tx_symbols no encontrados en pipeline")
                if not rx_symbols:
                    log_warning('decoder', "[BER] - rx_symbols vacíos")
        except Exception as e:
            log_error('decoder', f"[BER] Error en medición pipeline: {e}")
            import traceback
            traceback.print_exc()

        # === SNR: Medición automática ===
        # === RESUMEN FINAL CON PIPELINE ===
        try:
            from pipeline import pipeline

            # Calcular NCC promedio por magnitud
            ncc_avg_by_mag = {}
            for mag_key, vals in pipeline.ncc_log.items():
                if vals:
                    # Manejar ambos formatos: tuplas (ncc_pos, ncc_neg) o valores simples
                    if isinstance(vals[0], tuple):
                        # Formato nuevo: promediar el máximo de cada tupla
                        max_vals = [max(ncc_pos, ncc_neg) for ncc_pos, ncc_neg in vals]
                        ncc_avg_by_mag[mag_key] = sum(max_vals) / len(max_vals)
                    else:
                        # Formato antiguo: promediar directamente
                        ncc_avg_by_mag[mag_key] = sum(vals) / len(vals)
                else:
                    ncc_avg_by_mag[mag_key] = 0.0

            # Finalizar pipeline
            pipeline.finalize(message)

            # RESUMEN COMPLETO
            log_info('decoder', "\n" + "="*60)
            log_info('decoder', " RESUMEN FINAL - MÉTRICAS DE LABORATORIO")
            log_info('decoder', "="*60)

            log_info('decoder', f"[CHANNEL_CONFIG] Cn²={pipeline.env.get('cn2', 0):.1e}, SNR={pipeline.env.get('snr_db', 0)}dB, Ns={pipeline.env.get('Ns', 1)}, wander={pipeline.env.get('wander', False)}")

            # Obtener último BER calculado
            try:
                last_ber_from_previous = 0.0  # Se definirá arriba
                # Buscar en variables locales si existe
                import inspect
                frame = inspect.currentframe()
                if 'ber' in frame.f_locals:
                    last_ber_from_previous = frame.f_locals['ber']
            except:
                last_ber_from_previous = 0.0

            snr_avg = sum(pipeline.snr_log) / len(pipeline.snr_log) if pipeline.snr_log else float('nan')
            log_info('decoder', f"SNR medido promedio: {snr_avg:.2f}dB (samples={len(pipeline.snr_log)})")

            log_info('decoder', "NCC promedio por magnitud:")
            for mag in [1, 2, 3, 4]:
                key = f"|{mag}|"
                avg_val = ncc_avg_by_mag.get(key, 0)
                log_info('decoder', f"{key}: {avg_val:.3f}")

            log_info('decoder', "="*60)

        except Exception as e:
            log_error('decoder', f"[RESUMEN] Error: {e}")
            import traceback
            traceback.print_exc()