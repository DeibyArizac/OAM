# Modulo: pipeline.py
# Proposito: Sistema de pipeline global para compartir datos entre bloques GNU Radio
# Dependencias clave: numpy, pickle, csv
# Notas: Punto unico de verdad para metricas de laboratorio y flujo de datos

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sistema de pipeline para intercambio global de datos entre bloques GNU Radio.
Permite que los bloques GNU Radio actuen como interfaces frontend mientras
Python procesa los datos vectorialmente en el backend para maxima velocidad.
"""

import time
import csv
import os
import pickle
import hashlib
import json
import numpy as np
import threading
from typing import List, Dict, Any, Optional

# Sistema de logging unificado
from oam_logging import log_info, log_warning, log_error, log_debug

class OAMPipeline:
    """
    Pipeline global para intercambio de datos entre bloques GNU Radio.

    Atributos:
        encoder_symbols: Simbolos del encoder.
        original_bytes: Bytes originales con STX/ETX.
        channel_symbols: Simbolos del canal.
        decoder_message: Mensaje decodificado final.
        processing_stage: Etapa de procesamiento actual.

    Notas:
        - Punto unico de verdad para metricas de laboratorio.
        - Permite procesamiento vectorial eficiente en backend Python.
    """
    def __init__(self):
        # === DATOS ORIGINALES (ENCODER) ===
        self.encoder_symbols = []      # Symbols from encoder
        self.original_bytes = []       # Ground truth bytes con STX/ETX
        self.symbol_metadata = []      # Lista de dicts: {kind, bits, byte_index, is_pilot, idx}

        # === DATOS CANAL ===
        self.channel_symbols = []      # Symbols from channel
        self.env = {}                  # Environment: {cn2, snr_db, distance, Ns, wander}
        self.snr_log = []             # SNR medido por símbolo
        self.power_log = []           # Power antes vs después por símbolo

        # === DATOS DECODER ===
        self.decoder_message = ""      # Final decoded message
        self.ncc_log = {}             # NCC por magnitud: {"|1|": [vals], "|2|": [vals], ...}
        self.power_data_log = []      # Power data por símbolo: [{magnitude: {'P_pos', 'P_neg', 'z_pos', 'z_neg'}}, ...]
        self.modal_metrics = {}       # Métricas modales: eta_des, CT_dB, heatmap

        # === CONTROL Y MÉTRICAS ===
        self.processing_stage = "waiting"  # waiting, encoding_complete, channel_complete, complete
        self.run_id = None            # Timestamp del run
        self.metrics_csv_path = None  # Path opcional para CSV
        self.data_saved_event = threading.Event()  # Evento para sincronizar guardado de datos

        # === BUFFER DE FRAMES (secuencia completa) ===
        self.frame_buffer = {
            "before_channel": [],
            "after_encoder": [],
            "after_channel": [],
            "after_slit": [],
            "at_decoder": []
        }

        # === METADATOS DE VELOCIDAD ===
        self.run_meta = {}            # Metadatos adicionales: K, symbol_rate_hz, etc
        self.metrics = {}             # Métricas calculadas finales

    def reset(self, env: Optional[Dict] = None):
        """Reset pipeline for new message with optional environment"""
        self.encoder_symbols = []
        self.channel_symbols = []
        self.decoder_message = ""
        self.processing_stage = "waiting"

        # Reset evento de sincronización
        self.data_saved_event.clear()

        # Reset métricas
        self.original_bytes = []
        # Uso de flujo directo de bits sin etapa intermedia
        self.symbol_metadata = []
        self.snr_log = []
        self.ncc_log = {"|1|": [], "|2|": [], "|3|": [], "|4|": []}
        self.power_data_log = []
        self.modal_metrics = {}

        # Reset frame buffer como listas
        self.frame_buffer = {
            "before_channel": [],
            "after_encoder": [],
            "after_channel": [],
            "after_slit": [],
            "at_decoder": []
        }

        # Reset metadatos y métricas
        self.run_meta = {}
        self.metrics = {}

        # Nuevo run
        self.run_id = time.strftime('%Y%m%d_%H%M%S')
        if env:
            self.env = env.copy()
        else:
            self.env = {}

    def log_snr(self, snr_val: float):
        """Log SNR value per symbol"""
        self.snr_log.append(snr_val)

    def log_ncc(self, mag: int, ncc_pos: float, ncc_neg: float):
        """Log NCC values for magnitude detection"""
        key = f"|{mag}|"
        if key not in self.ncc_log:
            self.ncc_log[key] = []

        # Guardar AMBOS valores como tupla (ncc_pos, ncc_neg) para Dashboard D
        self.ncc_log[key].append((ncc_pos, ncc_neg))

    def log_power_data(self, magnitude: int, power_data: dict):
        """
        Log power data por magnitud para métricas modales físicas

        Args:
            magnitude: Magnitud del modo (sin signo)
            power_data: {'P_pos': float, 'P_neg': float, 'z_pos': complex, 'z_neg': complex}
        """
        # Inicializar si es el primer símbolo
        if not self.power_data_log or len(self.power_data_log) <= 0:
            self.power_data_log.append({})

        # Si es un nuevo símbolo, crear nuevo diccionario
        # Asumimos que se llama en orden de magnitudes para cada símbolo
        current_symbol_data = self.power_data_log[-1]

        current_symbol_data[magnitude] = power_data

    def push_field(self, stage: str, field: np.ndarray):
        """Apila campo por símbolo en la secuencia completa"""
        if stage not in self.frame_buffer:
            self.frame_buffer[stage] = []

        # Verificación de seguridad para evitar datos corruptos
        try:
            if not isinstance(field, np.ndarray):
                print(f"WARNING: {stage} - field no es numpy array, convirtiendo")
                field = np.array(field)

            if not np.isfinite(field).all():
                print(f"WARNING: {stage} - field contiene valores inválidos (inf/nan)")
                field = np.nan_to_num(field, nan=0.0, posinf=1.0, neginf=-1.0)

            # APILAR cada símbolo (no sobrescribir)
            self.frame_buffer[stage].append(np.array(field, copy=True))
        except Exception as e:
            print(f"ERROR: No se pudo guardar field en {stage}: {e}")
            # Guardar un array de ceros como fallback
            fallback = np.zeros_like(field) if hasattr(field, 'shape') else np.zeros((512, 512), dtype=complex)
            self.frame_buffer[stage].append(fallback)

    def get_field(self, stage: str) -> Optional[np.ndarray]:
        """Devuelve último campo o None"""
        return self.frame_buffer.get(stage, None)

    def set_run_meta(self, meta_dict: Dict[str, Any]):
        """Agregar metadatos de velocidad teórica"""
        self.run_meta.update(meta_dict)

    def get_run_meta(self) -> Dict[str, Any]:
        """Obtener metadatos de velocidad"""
        return self.run_meta.copy()

    def finalize(self, message: bytes):
        """Finalize processing with decoded message"""
        self.decoder_message = message
        self.processing_stage = "complete"

        # Calcular métricas de velocidad teórica
        meta = self.get_run_meta()
        K  = float(meta.get("K_bits_per_symbol", 0))
        Rs = float(meta.get("symbol_rate_hz", 0.0))

        bitrate_gross_bps = Rs * K if (Rs > 0 and K > 0) else None

        # Factor de símbolos de datos (neta teórica, sin canal)
        Ntot = float(meta.get("symbols_total", 0))
        Ndat = float(meta.get("symbols_data", 0))
        frac = (Ndat / Ntot) if (Ntot > 0) else 0.0
        bitrate_net_bps = (bitrate_gross_bps * frac) if bitrate_gross_bps else None

        self.metrics.setdefault("rates", {})
        self.metrics["rates"].update({
            "symbol_rate_hz":    Rs,
            "K_bits_per_symbol": K,
            "bitrate_gross_bps": bitrate_gross_bps,
            "bitrate_net_bps":   bitrate_net_bps,
        })

        if bitrate_gross_bps:
            def fmt_rate(x):
                if x >= 1e9: return f"{x/1e9:.3f} Gb/s"
                elif x >= 1e6: return f"{x/1e6:.3f} Mb/s"
                elif x >= 1e3: return f"{x/1e3:.3f} kb/s"
                else: return f"{x:.1f} bps"
            log_info('pipeline', f"Velocidad teórica: {fmt_rate(bitrate_gross_bps)} (bruta), {fmt_rate(bitrate_net_bps)} (neta)")

        # GUARDAR DATOS PARA DASHBOARD
        log_info('pipeline', f"FINALIZE: Verificando datos para guardar...")
        log_info('pipeline', f"  - ncc_log: {len(self.ncc_log) if self.ncc_log else 0} entradas")
        log_info('pipeline', f"  - symbol_metadata: {len(self.symbol_metadata) if self.symbol_metadata else 0} simbolos")
        log_info('pipeline', f"  - frame_buffer stages: {list(self.frame_buffer.keys())}")

        # Guardar siempre si hay symbol_metadata o campos, aunque no haya NCC
        should_save = (
            (self.ncc_log and any(len(v) > 0 for v in self.ncc_log.values())) or
            self.symbol_metadata or
            any(self.frame_buffer.values())
        )

        if should_save:
            log_info('pipeline', "EJECUTANDO _save_pipeline_data()...")
            self._save_pipeline_data()
            log_info('pipeline', "_save_pipeline_data() COMPLETADO")
        else:
            log_warning('pipeline', "No hay datos para guardar - ejecución incompleta")

    def export_metrics_row(self, ber: float, ncc_avg_by_mag: Dict[str, float]) -> Dict[str, Any]:
        """Export metrics as dict ready for CSV"""
        snr_avg = sum(self.snr_log) / len(self.snr_log) if self.snr_log else float('nan')

        return {
            'timestamp': self.run_id,
            'distance_m': self.env.get('distance', 0),
            'cn2': self.env.get('cn2', 0),
            'snr_config_db': self.env.get('snr_db', 0),
            'snr_measured_avg_db': snr_avg,
            'BER': ber,
            'NCC_mag1': ncc_avg_by_mag.get('|1|', 0),
            'NCC_mag2': ncc_avg_by_mag.get('|2|', 0),
            'NCC_mag3': ncc_avg_by_mag.get('|3|', 0),
            'NCC_mag4': ncc_avg_by_mag.get('|4|', 0),
            'Ns': self.env.get('Ns', 1),
            'wander': self.env.get('wander', False)
        }

    def save_to_csv(self, ber: float, ncc_avg_by_mag: Dict[str, float]):
        """Save metrics to CSV file"""
        if not self.metrics_csv_path:
            return

        row = self.export_metrics_row(ber, ncc_avg_by_mag)

        # Create directory if needed
        os.makedirs(os.path.dirname(self.metrics_csv_path), exist_ok=True)

        # Check if file exists to write header
        file_exists = os.path.exists(self.metrics_csv_path)

        with open(self.metrics_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _save_pipeline_data(self):
        """Save current pipeline data for dashboard access - OFFLINE MODE"""
        # COMPLETAR env con claves que usa el dashboard
        self.env.setdefault("distance_m", self.env.get("distance", 0.0))
        self.env.setdefault("oam_channels", [-4,-3,-2,-1,1,2,3,4])
        self.env.setdefault("symbol_rate_hz", 32000.0)

        # Crear estructura current_run/ (carpeta fija que se sobrescribe)
        current_run_dir = os.path.join(os.path.dirname(__file__), "current_run")

        # Limpiar carpeta existente si existe
        if os.path.exists(current_run_dir):
            import shutil
            shutil.rmtree(current_run_dir)

        # Crear carpeta limpia
        os.makedirs(current_run_dir, exist_ok=True)
        run_dir = current_run_dir

        # 1. Guardar metadatos (meta.json)
        # Convert bytes to base64 for JSON serialization
        decoder_message_serializable = self.decoder_message
        if isinstance(self.decoder_message, bytes):
            import base64
            decoder_message_serializable = {
                "type": "bytes",
                "data": base64.b64encode(self.decoder_message).decode('ascii'),
                "length": len(self.decoder_message)
            }

        original_bytes_serializable = []
        if self.original_bytes:
            import base64
            for item in self.original_bytes:
                if isinstance(item, bytes):
                    original_bytes_serializable.append({
                        "type": "bytes",
                        "data": base64.b64encode(item).decode('ascii'),
                        "length": len(item)
                    })
                else:
                    original_bytes_serializable.append(item)

        # Ensure all data is JSON serializable
        def make_json_serializable(obj):
            if isinstance(obj, bytes):
                import base64
                return {
                    "type": "bytes",
                    "data": base64.b64encode(obj).decode('ascii'),
                    "length": len(obj)
                }
            elif hasattr(obj, 'dtype') and hasattr(obj, 'tolist'):  # numpy array
                return {
                    "type": "numpy_array",
                    "data": obj.tolist(),
                    "dtype": str(obj.dtype),
                    "shape": list(obj.shape)
                }
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            else:
                # Convert other types to string representation
                return str(obj)

        meta_data = {
            "run_id": self.run_id,
            "timestamp": self.run_id,
            "processing_stage": self.processing_stage,
            "env": make_json_serializable(self.env),
            "run_meta": make_json_serializable(self.run_meta),
            "decoder_message": decoder_message_serializable,
            "original_bytes": original_bytes_serializable,
            "modal_metrics": make_json_serializable(self.modal_metrics) if self.modal_metrics else {}
        }

        meta_path = os.path.join(run_dir, "meta.json")
        import json
        try:
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f, indent=2)
        except TypeError as e:
            log_error('pipeline', f"JSON serialization error: {e}")
            log_error('pipeline', f"Problematic data types in meta_data:")
            for key, value in meta_data.items():
                log_error('pipeline', f"  {key}: {type(value)} = {str(value)[:100]}...")
            raise

        # 2. Guardar métricas por símbolo (metrics.jsonl)
        metrics_path = os.path.join(run_dir, "metrics.jsonl")
        with open(metrics_path, 'w') as f:
            # SNR por símbolo
            for i, snr_val in enumerate(self.snr_log):
                entry = {"symbol_idx": i, "metric": "snr_db", "value": snr_val}
                f.write(json.dumps(entry) + '\n')

            # NCC por magnitud y símbolo
            for mag_key, ncc_vals in self.ncc_log.items():
                for i, ncc_val in enumerate(ncc_vals):
                    entry = {"symbol_idx": i, "metric": f"ncc_{mag_key}", "value": ncc_val}
                    f.write(json.dumps(entry) + '\n')

            # Power metrics por símbolo
            for i, power_entry in enumerate(self.power_log):
                for metric_name, value in power_entry.items():
                    entry = {"symbol_idx": i, "metric": metric_name, "value": value}
                    f.write(json.dumps(entry) + '\n')

            # CORRECCIÓN C: Calcular fluctuación de potencia en escala lineal (rigor estadístico)
            if len(self.power_log) > 0:
                # Extraer potencias en escala lineal
                powers_linear = [p["power_after_w"] for p in self.power_log if "power_after_w" in p]
                if len(powers_linear) > 1:
                    P = np.array(powers_linear)
                    mu = np.mean(P)
                    sigma_lin = np.std(P)

                    # Fluctuación en dB (método físicamente correcto)
                    if mu > 0:
                        fluct_db = float(10 * np.log10((mu + sigma_lin) / mu))
                    else:
                        fluct_db = 0.0

                    # Guardar métricas agregadas de fluctuación
                    f.write(json.dumps({"metric": "power_fluctuation_db", "value": fluct_db}) + '\n')
                    f.write(json.dumps({"metric": "power_mean_w", "value": float(mu)}) + '\n')
                    f.write(json.dumps({"metric": "power_std_w", "value": float(sigma_lin)}) + '\n')
                    f.write(json.dumps({"metric": "power_cv_percent", "value": float(100 * sigma_lin / mu) if mu > 0 else 0.0}) + '\n')

        # 2.5. Guardar power_data_log (datos modales detallados por símbolo)
        if self.power_data_log:
            power_data_path = os.path.join(run_dir, "power_data.jsonl")
            with open(power_data_path, 'w') as f:
                for i, power_data_dict in enumerate(self.power_data_log):
                    entry = {
                        "symbol_idx": i,
                        "power_data": make_json_serializable(power_data_dict)
                    }
                    f.write(json.dumps(entry) + '\n')
            log_debug('pipeline', f"power_data.jsonl guardado: {len(self.power_data_log)} símbolos")

        # 3. Guardar campos por etapa (NPZ files)
        for stage, field_list in self.frame_buffer.items():
            if field_list:  # Solo si hay datos
                stage_path = os.path.join(run_dir, f"fields_{stage}.npz")
                try:
                    # Verificar que todos los campos tengan la misma forma
                    shapes = [np.array(f).shape for f in field_list]
                    if len(set(shapes)) == 1:
                        # Todas las formas son iguales - convertir a array 3D
                        field_array = np.array(field_list)
                        np.savez_compressed(stage_path, fields=field_array)
                    else:
                        # Formas diferentes - guardar como array de objetos
                        log_warning('pipeline', f"Stage {stage}: campos con formas diferentes {set(shapes)}")
                        field_dict = {f"field_{i}": np.array(f) for i, f in enumerate(field_list)}
                        field_dict['metadata'] = np.array({
                            'num_fields': len(field_list),
                            'shapes': shapes,
                            'dtypes': [str(np.array(f).dtype) for f in field_list]
                        }, dtype=object)
                        np.savez_compressed(stage_path, **field_dict)
                except Exception as e:
                    log_error('pipeline', f"Error guardando campos stage {stage}: {e}")
                    # Fallback: guardar individualmente
                    field_dict = {f"field_{i}": np.array(f) for i, f in enumerate(field_list)}
                    field_dict['metadata'] = {
                        'num_fields': len(field_list),
                        'error': str(e)
                    }
                    np.savez_compressed(stage_path, **field_dict)

        # 4. Guardar metadatos de símbolos (symbol_metadata.json)
        log_info('pipeline', f"GUARDANDO symbol_metadata: existe={self.symbol_metadata is not None}, len={len(self.symbol_metadata) if self.symbol_metadata else 0}")
        if self.symbol_metadata:
            symbol_metadata_path = os.path.join(run_dir, "symbol_metadata.json")
            log_info('pipeline', f"Guardando symbol_metadata.json en: {symbol_metadata_path}")
            with open(symbol_metadata_path, 'w') as f:
                json.dump(self.symbol_metadata, f, indent=2)
            log_info('pipeline', f"symbol_metadata.json guardado exitosamente con {len(self.symbol_metadata)} simbolos")
        else:
            log_warning('pipeline', "NO se guarda symbol_metadata.json porque self.symbol_metadata esta vacio o None")

        # 5. Guardar métricas calculadas (rates.json)
        if self.metrics:
            rates_path = os.path.join(run_dir, "rates.json")
            with open(rates_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)

        # MANTENER compatibilidad temporal con archivo antiguo
        import tempfile
        temp_root = os.path.join(tempfile.gettempdir(), "oam")
        os.makedirs(temp_root, exist_ok=True)
        temp_path = os.path.join(temp_root, "pipeline_result.pkl")

        pipeline_data = {
            'frame_buffer': self.frame_buffer,
            'snr_log': self.snr_log.copy(),
            'ncc_log': {k: v.copy() for k, v in self.ncc_log.items()},
            'power_log': self.power_log.copy(),
            'env': self.env.copy(),
            'run_id': self.run_id,
            'processing_stage': self.processing_stage,
            'decoder_message': self.decoder_message,
            'original_bytes': self.original_bytes.copy(),
            'metrics': self.metrics.copy(),
            'run_meta': self.run_meta.copy(),
            'symbol_metadata': self.symbol_metadata.copy() if self.symbol_metadata else []
        }

        # Escritura atómica del pickle para evitar lecturas truncadas
        temp_write_path = temp_path + ".tmp"
        try:
            # Verificar que el directorio existe antes de escribir
            if not os.path.exists(temp_root):
                log_warning('pipeline', f"Recreando directorio temporal: {temp_root}")
                os.makedirs(temp_root, exist_ok=True)

            with open(temp_write_path, 'wb') as f:
                pickle.dump(pipeline_data, f)
                f.flush()  # Asegurar que se escriba al disco
                os.fsync(f.fileno())  # Forzar escritura física

            # Verificar que el archivo temporal existe antes de renombrar
            if not os.path.exists(temp_write_path):
                raise FileNotFoundError(f"Archivo temporal no creado: {temp_write_path}")

            # Mover archivo temporal al final para escritura atómica
            os.rename(temp_write_path, temp_path)
            log_info('pipeline', f"Pickle escrito atómicamente: {temp_path}")

        except Exception as e:
            # Limpiar archivo temporal si hay error
            if os.path.exists(temp_write_path):
                os.remove(temp_write_path)
            log_error('pipeline', f"Error escribiendo pickle: {e}")
            # No re-lanzar el error - solo registrarlo para no bloquear el sistema
            log_warning('pipeline', "Continuando sin guardar pickle temporal (no afecta datos en current_run/)")

        # 5. Guardar hash de configuración para cache inteligente
        current_config = self.get_current_config()
        self.save_config_hash(current_config)

        log_info('pipeline', f"Datos guardados offline en: {run_dir}")

        # 6. Señalizar que los datos fueron guardados exitosamente
        self.data_saved_event.set()
        log_info('pipeline', "Evento data_saved_event establecido - datos listos para dashboards")
        log_debug('pipeline', f"Compatibilidad temporal: {temp_path}")

    def load_run(self, run_id: str):
        """Load pipeline data from offline runs/<run_id>/ or current_run/ structure"""
        if run_id == "current":
            # Cargar desde carpeta current_run
            run_dir = os.path.join(os.path.dirname(__file__), "current_run")
        else:
            # Cargar desde runs/<run_id> (compatibilidad con runs antiguos)
            run_dir = os.path.join(os.path.dirname(__file__), "runs", run_id)

        if not os.path.exists(run_dir):
            log_warning('pipeline', f"Run directory not found: {run_dir}")
            return False

        try:
            import json

            # 1. Cargar metadatos (meta.json)
            meta_path = os.path.join(run_dir, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta_data = json.load(f)

                # Deserialize bytes objects recursively
                def deserialize_json_data(obj):
                    if isinstance(obj, dict):
                        if obj.get('type') == 'bytes':
                            import base64
                            return base64.b64decode(obj['data'])
                        elif obj.get('type') == 'numpy_array':
                            data = obj['data']
                            dtype = obj['dtype']
                            shape = tuple(obj['shape'])
                            return np.array(data, dtype=dtype).reshape(shape)
                        else:
                            return {k: deserialize_json_data(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [deserialize_json_data(item) for item in obj]
                    else:
                        return obj

                self.run_id = meta_data.get('run_id', run_id)
                self.processing_stage = meta_data.get('processing_stage', 'complete')
                self.env = deserialize_json_data(meta_data.get('env', {}))
                self.run_meta = deserialize_json_data(meta_data.get('run_meta', {}))
                self.decoder_message = deserialize_json_data(meta_data.get('decoder_message', ""))
                self.original_bytes = deserialize_json_data(meta_data.get('original_bytes', []))
                self.modal_metrics = deserialize_json_data(meta_data.get('modal_metrics', {}))

            # 2. Cargar métricas por símbolo (metrics.jsonl)
            self.snr_log = []
            self.ncc_log = {"|1|": [], "|2|": [], "|3|": [], "|4|": []}
            self.power_log = []

            metrics_path = os.path.join(run_dir, "metrics.jsonl")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        metric = entry.get('metric', '')
                        value = entry.get('value', 0)

                        if metric == 'snr_db':
                            self.snr_log.append(value)
                        elif metric.startswith('ncc_'):
                            mag_key = metric.replace('ncc_', '')
                            if mag_key in self.ncc_log:
                                # Convertir lista [ncc_pos, ncc_neg] a tupla para consistencia
                                if isinstance(value, list) and len(value) == 2:
                                    self.ncc_log[mag_key].append(tuple(value))
                                else:
                                    self.ncc_log[mag_key].append(value)
                        elif metric.startswith('power_'):
                            symbol_idx = entry.get('symbol_idx', 0)
                            if len(self.power_log) <= symbol_idx:
                                self.power_log.extend([{}] * (symbol_idx + 1 - len(self.power_log)))
                            self.power_log[symbol_idx][metric] = value

            # 2.5. Cargar power_data_log (power_data.jsonl)
            self.power_data_log = []
            power_data_path = os.path.join(run_dir, "power_data.jsonl")
            if os.path.exists(power_data_path):
                try:
                    with open(power_data_path, 'r') as f:
                        for line in f:
                            entry = json.loads(line.strip())
                            symbol_idx = entry.get('symbol_idx', 0)
                            power_data = deserialize_json_data(entry.get('power_data', {}))

                            # Extender lista si es necesario
                            if len(self.power_data_log) <= symbol_idx:
                                self.power_data_log.extend([{}] * (symbol_idx + 1 - len(self.power_data_log)))

                            self.power_data_log[symbol_idx] = power_data

                    log_debug('pipeline', f"power_data_log cargado: {len(self.power_data_log)} símbolos")
                except Exception as e:
                    log_error('pipeline', f"Error cargando power_data.jsonl: {e}")
                    self.power_data_log = []
            else:
                log_debug('pipeline', f"power_data.jsonl no existe en: {power_data_path}")

            # 2.6. Cargar metadatos de símbolos (symbol_metadata.json)
            self.symbol_metadata = []
            symbol_metadata_path = os.path.join(run_dir, "symbol_metadata.json")

            if os.path.exists(symbol_metadata_path):
                try:
                    with open(symbol_metadata_path, 'r') as f:
                        self.symbol_metadata = json.load(f)

                    # Verificar que tienen oam_modes
                    if self.symbol_metadata and len(self.symbol_metadata) > 0:
                        first_symbol = self.symbol_metadata[0]
                        if 'oam_modes' not in first_symbol:
                            log_warning('pipeline', f"Metadatos sin 'oam_modes': {first_symbol.keys()}")

                except Exception as e:
                    log_error('pipeline', f"Error cargando symbol_metadata: {e}")
                    self.symbol_metadata = []
            else:
                log_error('pipeline', f" Archivo symbol_metadata.json NO EXISTE en: {symbol_metadata_path}")

            # 3. Cargar campos por etapa (NPZ files)
            self.frame_buffer = {
                "before_channel": [],
                "after_encoder": [],
                "after_channel": [],
                "after_slit": [],
                "at_decoder": []
            }

            for stage in self.frame_buffer.keys():
                stage_path = os.path.join(run_dir, f"fields_{stage}.npz")
                if os.path.exists(stage_path):
                    log_debug('pipeline', f"Cargando {stage_path} (tamaño: {os.path.getsize(stage_path)} bytes)")
                    try:
                        # Intentar cargar directamente con numpy (más robusto que zipfile check)
                        npz_data = np.load(stage_path)
                        if 'fields' in npz_data:
                            # Formato uniforme (array 3D)
                            field_array = npz_data['fields']
                            self.frame_buffer[stage] = [field_array[i] for i in range(field_array.shape[0])]
                        elif 'metadata' in npz_data:
                            # Formato individual (campos separados)
                            metadata = npz_data['metadata'].item()
                            num_fields = metadata.get('num_fields', 0)
                            field_list = []
                            for i in range(num_fields):
                                field_key = f"field_{i}"
                                if field_key in npz_data:
                                    field_list.append(npz_data[field_key])
                            self.frame_buffer[stage] = field_list
                            if 'error' in metadata:
                                log_warning('pipeline', f"Stage {stage} cargado con error anterior: {metadata['error']}")
                        else:
                            log_warning('pipeline', f"Formato desconocido en {stage_path}")
                    except Exception as e:
                        log_error('pipeline', f"Error cargando campos stage {stage}: {e}")
                        self.frame_buffer[stage] = []
                else:
                    log_debug('pipeline', f"Stage {stage} no disponible (archivo no existe)")
                    self.frame_buffer[stage] = []

            # 4. Cargar métricas calculadas (rates.json)
            self.metrics = {}
            rates_path = os.path.join(run_dir, "rates.json")
            if os.path.exists(rates_path):
                with open(rates_path, 'r') as f:
                    self.metrics = json.load(f)

            log_info('pipeline', f"Run {run_id} cargado desde: {run_dir}")
            return True

        except Exception as e:
            log_error('pipeline', f"Error cargando run {run_id}: {e}")
            return False

    def load_saved_data(self):
        """Load pipeline data from temporary file - LEGACY COMPATIBILITY"""
        import tempfile
        root = os.path.join(tempfile.gettempdir(), "oam")
        temp_path = os.path.join(root, "pipeline_result.pkl")

        if not os.path.exists(temp_path):
            log_warning('pipeline', f"No se encontraron datos de pipeline en: {temp_path}")
            return False

        try:
            with open(temp_path, 'rb') as f:
                data = pickle.load(f)

            # Restaurar datos
            self.snr_log = data.get('snr_log', [])
            self.ncc_log = data.get('ncc_log', {})
            self.env = data.get('env', {})
            self.run_id = data.get('run_id', None)
            self.processing_stage = data.get('processing_stage', 'waiting')
            self.decoder_message = data.get('decoder_message', "")
            self.original_bytes = data.get('original_bytes', [])
            self.frame_buffer = data.get('frame_buffer', {})
            self.metrics = data.get('metrics', {})
            self.run_meta = data.get('run_meta', {})

            log_info('pipeline', f"Datos cargados desde: {temp_path}")
            return True

        except Exception as e:
            log_error('pipeline', f"Error cargando datos de pipeline: {e}")
            return False

    def list_runs(self):
        """List available offline runs"""
        runs_dir = os.path.join(os.path.dirname(__file__), "runs")
        if not os.path.exists(runs_dir):
            return []

        runs = []
        for item in os.listdir(runs_dir):
            run_dir = os.path.join(runs_dir, item)
            if os.path.isdir(run_dir):
                meta_path = os.path.join(run_dir, "meta.json")
                if os.path.exists(meta_path):
                    runs.append(item)

        return sorted(runs, reverse=True)  # Most recent first

    def status(self):
        """Get current pipeline status"""
        return {
            'stage': self.processing_stage,
            'encoder_symbols': len(self.encoder_symbols),
            'channel_symbols': len(self.channel_symbols),
            'decoded_message': self.decoder_message,
            'run_id': self.run_id,
            'env': self.env,
            'snr_samples': len(self.snr_log),
            'ncc_samples': {k: len(v) for k, v in self.ncc_log.items()}
        }

    def calculate_config_hash(self, config_data: dict) -> str:
        """
        Calcular hash SHA256 de los parámetros de configuración.

        Args:
            config_data: Diccionario con parámetros de simulación

        Returns:
            String hexadecimal del hash
        """
        # Crear diccionario ordenado para hash consistente
        sorted_config = {}

        # Parámetros críticos que afectan los resultados
        critical_params = [
            'grid_size', 'wavelength', 'tx_aperture_size', 'tx_beam_waist',
            'num_oam_modes', 'oam_channels', 'propagation_distance', 'cn2', 'snr_db',
            'message_text', 'modulation'
        ]

        for key in sorted(critical_params):
            if key in config_data:
                value = config_data[key]
                # Convertir listas a tuplas para ser hasheable
                if isinstance(value, list):
                    value = tuple(value)
                # Redondear valores float para evitar diferencias mínimas
                elif isinstance(value, float):
                    value = round(value, 10)
                sorted_config[key] = value

        # Convertir a JSON string determinístico
        config_str = json.dumps(sorted_config, sort_keys=True, separators=(',', ':'))

        # Calcular hash SHA256
        return hashlib.sha256(config_str.encode('utf-8')).hexdigest()

    def is_cache_valid(self, config_data: dict) -> bool:
        """
        Verificar si el cache actual es válido para la configuración dada.

        Args:
            config_data: Configuración actual de simulación

        Returns:
            True si el cache es válido, False si necesita regenerar
        """
        current_hash = self.calculate_config_hash(config_data)
        cache_file = os.path.join(os.path.dirname(__file__), "current_run", "config_hash.txt")

        if not os.path.exists(cache_file):
            log_debug('pipeline', "No hay archivo de hash de cache - regenerar")
            return False

        try:
            with open(cache_file, 'r') as f:
                cached_hash = f.read().strip()

            if cached_hash == current_hash:
                log_info('pipeline', "Cache valido - parametros sin cambios")
                return True
            else:
                log_info('pipeline', "Cache invalido - parametros cambiaron")
                log_debug('pipeline', f"Hash anterior: {cached_hash[:12]}...")
                log_debug('pipeline', f"Hash actual:   {current_hash[:12]}...")
                return False

        except Exception as e:
            log_error('pipeline', f"Error leyendo cache hash: {e}")
            return False

    def save_config_hash(self, config_data: dict):
        """
        Guardar hash de configuración en el cache.

        Args:
            config_data: Configuración de simulación a cachear
        """
        try:
            current_hash = self.calculate_config_hash(config_data)
            cache_dir = os.path.join(os.path.dirname(__file__), "current_run")
            os.makedirs(cache_dir, exist_ok=True)

            cache_file = os.path.join(cache_dir, "config_hash.txt")
            with open(cache_file, 'w') as f:
                f.write(current_hash)

            log_debug('pipeline', f"Hash de configuración guardado: {current_hash[:12]}...")

        except Exception as e:
            log_error('pipeline', f"Error guardando config hash: {e}")

    def get_current_config(self) -> dict:
        """
        Extraer configuración actual del pipeline para cache.

        Returns:
            Diccionario con parámetros de configuración
        """
        config = {}

        # Parámetros del entorno
        config.update(self.env)

        # Parámetros del mensaje (si hay)
        if hasattr(self, 'original_bytes') and self.original_bytes:
            # Convertir bytes a string para hash consistente
            try:
                message_bytes = bytes(self.original_bytes)
                config['message_text'] = message_bytes.decode('utf-8', errors='ignore')
            except:
                config['message_bytes_hash'] = hashlib.md5(bytes(self.original_bytes)).hexdigest()

        # Parámetros de modulación
        if hasattr(self, 'modulation'):
            config['modulation'] = self.modulation

        return config

    def push_symbol_metadata(self, symbols_data: list):
        """
        Guardar metadatos de símbolos en el pipeline.

        Args:
            symbols_data: Lista de diccionarios con metadatos de símbolos
        """
        self.symbol_metadata = symbols_data.copy()
        log_info('pipeline', f"Symbol metadata guardados: {len(self.symbol_metadata)} símbolos")

        # Verificar que todos tienen oam_modes
        for i, symbol in enumerate(self.symbol_metadata):
            if 'oam_modes' not in symbol:
                log_warning('pipeline', f"Símbolo {i} sin 'oam_modes': {symbol.keys()}")
            else:
                log_debug('pipeline', f"Símbolo {i} oam_modes: {symbol['oam_modes']}")

    def get_symbol_oam_modes(self, symbol_index: int) -> list:
        """
        Obtener modos OAM para un símbolo específico.

        Args:
            symbol_index: Índice del símbolo (0-based)

        Returns:
            Lista de modos OAM [l1, l2, l3, l4] o [] si no hay datos
        """
        if not self.symbol_metadata or symbol_index >= len(self.symbol_metadata):
            return []

        symbol_meta = self.symbol_metadata[symbol_index]

        # USAR ÚNICAMENTE MODOS OAM DIRECTOS - NO FALLBACK
        if 'oam_modes' not in symbol_meta:
            raise ValueError(f"Pipeline: ERROR CRÍTICO - Símbolo {symbol_index} sin 'oam_modes' guardados. Sistema debe fallar para identificar problema.")

        return symbol_meta['oam_modes'].copy()

    def get_symbol_info_string(self, symbol_index: int) -> str:
        """
        Obtener información completa de un símbolo como string.

        Args:
            symbol_index: Índice del símbolo (0-based)

        Returns:
            String con información del símbolo
        """
        if not self.symbol_metadata or symbol_index >= len(self.symbol_metadata):
            return f"Símbolo {symbol_index+1}: Sin metadatos"

        symbol_meta = self.symbol_metadata[symbol_index]
        kind = symbol_meta.get('kind', 'unknown')
        bits = symbol_meta.get('bits', [])
        modes = self.get_symbol_oam_modes(symbol_index)

        # Formatear modos con signo
        modes_str = [f"{m:+d}" for m in modes] if modes else ["N/A"]
        bits_str = "".join(str(b) for b in bits) if bits else "N/A"

        result = f"Símbolo {symbol_index+1}: [{bits_str}] → {modes_str} ({kind})"
        return result

# Global pipeline instance
pipeline = OAMPipeline()