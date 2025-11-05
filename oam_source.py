# Modulo: oam_source.py
# Proposito: Fuente de mensajes para sistema OAM con control de paquetes
# Dependencias clave: gnuradio, numpy
# Notas: Genera mensajes de prueba y controla la transmision de paquetes

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from gnuradio import gr
import time

# Sistema de logging unificado
from oam_logging import log_info, log_warning, log_error, log_debug

# Configuración centralizada del sistema
from oam_system_config import get_system_config, OAMConfig

class oam_source(gr.sync_block):
    """
    OAM Source Block - Generador de datos para transmisión OAM
    Genera patrones de datos, texto y secuencias binarias para pruebas
    """
    def __init__(self, message=None, packet_size=32):
        gr.sync_block.__init__(self,
            name="oam_source",
            in_sig=None,  # Source block
            out_sig=[np.uint8])  # Output bytes

        try:
            # Obtener configuración centralizada
            config = get_system_config()

            # Usar mensaje de configuración si no se especifica
            if message is None:
                message = config['message_text']

            # Validar parámetros de entrada
            if not isinstance(message, str) or len(message) == 0:
                raise ValueError("message debe ser una cadena no vacía")
            if not isinstance(packet_size, int) or packet_size <= 0:
                raise ValueError("packet_size debe ser un entero positivo")

            self.message = message
            self.packet_size = packet_size
            self.rate = 1e9  # Tasa por defecto del generador de datos

            # Patrones de generación disponibles
            self.patterns = {
                'random': self._generate_random,
                'text': self._generate_text,
                'pattern': self._generate_pattern,
                'file': self._generate_from_file,
            }

            # Generar paquete de datos usando método de generación
            self.data_packet = self.generate(type='text', message=self.message)
            self.packet_index = 0
            self.bytes_sent = 0
            self.finished = False

            # Logging de inicialización
            log_info('source', "Bloque inicializado correctamente")
            log_info('source', f"Mensaje: '{self.message}' ({len(self.data_packet)} bytes)")
            log_info('source', f"Tamaño de paquete: {self.packet_size}")
            log_info('source', f"Tasa de datos: {self.rate}")
            log_info('source', "Configuración completa")

        except Exception as e:
            log_error('source', f"Error en inicialización: {e}")
            raise RuntimeError(f"No se pudo inicializar OAM Source: {e}")

    def generate(self, type='text', **kwargs):
        """Genera datos según el tipo especificado"""
        if type in self.patterns:
            return self.patterns[type](**kwargs)
        else:
            raise ValueError(f"Tipo de datos no soportado: {type}")

    def _generate_random(self, num_bytes=100):
        """Genera datos aleatorios"""
        return np.random.randint(0, 256, num_bytes, dtype=np.uint8).tobytes()

    def _generate_text(self, message=None):
        """Convierte texto a bytes UTF-8 con marcadores STX/ETX para delimitar mensaje"""
        # Usar mensaje de configuración si no se especifica
        if message is None:
            config = get_system_config()
            message = config['message_text']

        # RESTORED: Agregar STX/ETX para detectar inicio y fin de mensaje
        STX = 0x02  # Start of Text
        ETX = 0x03  # End of Text
        message_bytes = message.encode('utf-8')
        return bytes([STX]) + message_bytes + bytes([ETX])

    def _generate_pattern(self, pattern=[0xFF, 0x00], repeat=10):
        """Genera patrón repetitivo"""
        full_pattern = pattern * repeat
        return bytes(full_pattern)

    def _generate_from_file(self, filename):
        """Lee datos desde archivo"""
        with open(filename, 'rb') as f:
            return f.read()

    def work(self, input_items, output_items):
        """Generate data using batch mode - send complete message at once"""
        if self.finished:
            return -1  # Signal end of data

        output = output_items[0]

        # FASE 1: Enviar el mensaje actual
        if self.bytes_sent == 0:
            # Modo lote: Enviar todos los datos en una sola llamada work()
            message_size = len(self.data_packet)

            # Asegurar que tenemos buffer de salida suficiente para mensaje completo
            if len(output) < message_size:
                # GNU Radio nos llamará de nuevo con buffer más grande
                log_debug('source', f"BATCH: Necesita {message_size} buffer, obtuvo {len(output)}, esperando...")
                return 0  # Esperar tamaño de buffer adecuado

            # Enviar mensaje completo de una vez
            for i in range(message_size):
                output[i] = self.data_packet[i]
                self.bytes_sent += 1
                # log_debug('source', f"BATCH[{self.bytes_sent}]: {output[i]} ({'STX' if output[i]==2 else 'ETX' if output[i]==3 else chr(output[i]) if 32<=output[i]<=126 else 'CTRL'})")

            log_info('source', f"Lote completo: Enviado mensaje completo ({message_size} bytes) de una vez")
            log_info('source', "CONTINUA: Mantendrá ejecución hasta que encoder termine todos los símbolos")

            # NUEVA: Crear campo simple de referencia en source (Gaussian básico)
            self._create_source_field()

            log_debug('source', f"Mensaje completo hex: {self.data_packet.hex()}")
            return message_size  # Retornar todos los bytes de una vez

        # FASE 2: Enviar datos de relleno para mantener flujo activo
        else:
            # TERMINACIÓN: Detener envío de datos después de tiempo razonable
            if self.bytes_sent > 0:
                # Contar cuántas llamadas work() hemos hecho después del mensaje inicial
                if not hasattr(self, 'padding_calls'):
                    self.padding_calls = 0
                self.padding_calls += 1

                # Detener después de tiempo suficiente para que el decodificador procese
                if self.padding_calls > 500:  # Permitir tiempo suficiente para procesamiento completo
                    log_info('source', "TERMINANDO: Mensaje enviado y procesado, deteniendo fuente")
                    return -1  # Señalar fin de datos

            # Enviar datos nulos para mantener el pipeline funcionando
            padding_size = min(len(output), 1)  # Enviar relleno mínimo
            for i in range(padding_size):
                output[i] = 0  # Datos de relleno nulos
            log_debug('source', f"PADDING: Enviados {padding_size} bytes nulos para mantener pipeline activo")
            return padding_size

    def generate_message(self):
        """Método directo de generación: Genera mensaje sin depender de GNU Radio"""
        log_info('source', f"PYTHON DIRECTO: Generando mensaje '{self.message}'")
        return self.message

    def _create_source_field(self):
        """Crear campo de referencia simple en el source (Gaussian básico)"""
        try:
            from pipeline import pipeline
            import numpy as np

            # Obtener parámetros de configuración centralizada
            config = get_system_config()
            N = config['grid_size']  # Grid size desde configuración
            L = config['tx_aperture_size']  # Aperture size desde configuración
            wo = OAMConfig.get_tx_beam_waist()  # Beam waist calculado

            # Crear coordenadas
            x = np.linspace(-L/2, L/2, N)
            y = np.linspace(-L/2, L/2, N)
            X, Y = np.meshgrid(x, y)
            R = np.sqrt(X**2 + Y**2)

            # Gaussian básico (sin fase OAM, solo intensidad)
            w = wo  # beam waist
            gaussian = np.exp(-(R/w)**2)

            # Crear campo complejo simple (solo real, sin fase especial)
            field_src = gaussian.astype(complex)

            # Guardar en pipeline
            pipeline.push_field("at_source", field_src)
            log_info('source', f"PIPELINE: Guardado campo Gaussian básico en 'at_source' ({N}x{N})")

        except Exception as e:
            log_warning('source', f"Error creando campo source: {e}")
            # No es crítico, continuar sin campo source