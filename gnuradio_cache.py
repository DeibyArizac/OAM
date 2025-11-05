#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sistema de caché JSON para sincronización de bloques GNU Radio

Este módulo proporciona funciones para que los bloques escriban/lean
sus parámetros en un archivo JSON compartido, permitiendo que el Source
reúna toda la configuración antes de lanzar el pipeline headless.
"""

import json
import os
import time
import threading
from datetime import datetime
from oam_logging import log_info, log_warning, log_error, log_debug

# Ruta del archivo de caché
CACHE_FILE = "current_run/gnuradio_cache.json"

# Lista de bloques esperados
EXPECTED_BLOCKS = ["oam_source", "oam_encoder", "oam_channel", "oam_decoder", "oam_visualizer"]

# Lock global para escrituras atómicas
_cache_lock = threading.Lock()


def init_cache(run_id):
    """
    Inicializar archivo de caché con run_meta

    Args:
        run_id: ID único para esta ejecución (timestamp)
    """
    try:
        with _cache_lock:
            os.makedirs("current_run", exist_ok=True)

            cache_data = {
                "run_meta": {
                    "run_id": run_id,
                    "expected": EXPECTED_BLOCKS,
                    "timestamp": datetime.now().isoformat()
                }
            }

            with open(CACHE_FILE, 'w') as f:
                json.dump(cache_data, f, indent=2)

            log_debug('cache', f"Caché inicializado con run_id: {run_id}")
            return True

    except Exception as e:
        log_error('cache', f"Error inicializando caché: {e}")
        return False


def write_block_params(block_name, params, run_id):
    """
    Escribir parámetros de un bloque al caché (thread-safe)

    Args:
        block_name: Nombre del bloque (ej: "oam_encoder")
        params: Diccionario con parámetros del bloque
        run_id: ID de la ejecución actual
    """
    try:
        with _cache_lock:
            # Leer caché actual
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, 'r') as f:
                    cache_data = json.load(f)
            else:
                cache_data = {}

            # Agregar/actualizar sección del bloque
            cache_data[block_name] = {
                "run_id": run_id,
                "params": params,
                "timestamp": datetime.now().isoformat()
            }

            # Escribir de vuelta atómicamente
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache_data, f, indent=2)

            log_debug('cache', f"Bloque '{block_name}' escribió parámetros (run_id: {run_id})")
            return True

    except Exception as e:
        log_error('cache', f"Error escribiendo bloque '{block_name}': {e}")
        return False


def wait_for_all_blocks(run_id, timeout=5.0):
    """
    Esperar hasta que todos los bloques esperados escriban sus parámetros

    Args:
        run_id: ID de ejecución esperado
        timeout: Tiempo máximo de espera en segundos

    Returns:
        bool: True si todos los bloques están listos, False si timeout
    """
    log_debug('cache', f"Esperando a que todos los bloques escriban (run_id: {run_id})...")

    start_time = time.time()
    check_interval = 0.1

    while (time.time() - start_time) < timeout:
        try:
            if not os.path.exists(CACHE_FILE):
                time.sleep(check_interval)
                continue

            with _cache_lock:
                with open(CACHE_FILE, 'r') as f:
                    cache_data = json.load(f)

            # Verificar que todos los bloques esperados estén presentes
            missing_blocks = []
            for block_name in EXPECTED_BLOCKS:
                if block_name not in cache_data:
                    missing_blocks.append(block_name)
                elif cache_data[block_name].get('run_id') != run_id:
                    missing_blocks.append(f"{block_name}(wrong_id)")

            if not missing_blocks:
                elapsed = time.time() - start_time
                log_debug('cache', f"Todos los bloques listos en {elapsed:.2f}s")
                return True

            log_debug('cache', f"Faltan bloques: {missing_blocks}")
            time.sleep(check_interval)

        except Exception as e:
            log_debug('cache', f"Error leyendo caché (reintentando): {e}")
            time.sleep(check_interval)

    # Timeout
    elapsed = time.time() - start_time
    log_warning('cache', f"Timeout después de {elapsed:.2f}s esperando bloques")
    return False


def merge_all_params():
    """
    Fusionar parámetros de todos los bloques en una configuración unificada

    Returns:
        dict: Configuración completa con todos los parámetros
    """
    try:
        if not os.path.exists(CACHE_FILE):
            log_error('cache', "Archivo de caché no existe")
            return {}

        with _cache_lock:
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)

        # Extraer parámetros de cada bloque y fusionar
        merged_config = {}

        for block_name in EXPECTED_BLOCKS:
            if block_name in cache_data:
                block_params = cache_data[block_name].get('params', {})
                merged_config.update(block_params)
                log_debug('cache', f"Fusionados parámetros de '{block_name}': {list(block_params.keys())}")

        log_debug('cache', f"Configuración fusionada con {len(merged_config)} parámetros")
        return merged_config

    except Exception as e:
        log_error('cache', f"Error fusionando parámetros: {e}")
        return {}


def save_merged_config(config, output_file="current_run/config_from_grc.json"):
    """
    Guardar configuración fusionada a archivo JSON

    Args:
        config: Diccionario con configuración completa
        output_file: Ruta donde guardar el JSON
    """
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)

        log_debug('cache', f"Configuración guardada en: {output_file}")
        return True

    except Exception as e:
        log_error('cache', f"Error guardando configuración: {e}")
        return False


def get_run_id_from_cache():
    """Obtener run_id del caché actual"""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
            return cache_data.get('run_meta', {}).get('run_id')
    except:
        pass
    return None