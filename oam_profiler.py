#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OAM PROFILER - Sistema de medición de tiempos de ejecución
===========================================================

Módulo para medir y reportar el tiempo de ejecución de cada etapa
del procesamiento OAM. Crítico para análisis de rendimiento y optimización.

Autor: Deiby Fernando Ariza Cadena
Propósito: Responder a requerimiento del director de tesis sobre tiempos de procesamiento
"""

import time
import json
import os
from collections import defaultdict
from datetime import datetime
from oam_logging import log_info, log_debug, log_warning

class OAMProfiler:
    """
    Sistema de profiling para medición de tiempos de ejecución

    Características:
    - Mide tiempo de cada etapa del pipeline
    - Almacena tiempos por símbolo procesado
    - Genera reportes estadísticos
    - Integración con pipeline para almacenamiento persistente
    """

    def __init__(self):
        """Inicializar profiler"""
        self.timings = defaultdict(list)  # {stage: [time1, time2, ...]}
        self.stage_starts = {}  # {stage: start_time}
        self.total_start = None
        self.total_end = None
        self.symbol_count = 0

        log_debug('profiler', "OAM Profiler inicializado")

    def start_total(self):
        """Marcar inicio del procesamiento total"""
        self.total_start = time.time()
        log_debug('profiler', "=== INICIO PROFILING TOTAL ===")

    def end_total(self):
        """Marcar fin del procesamiento total"""
        self.total_end = time.time()
        total_time = self.total_end - self.total_start
        log_info('profiler', f"=== FIN PROFILING TOTAL: {total_time:.3f}s ===")

    def start_stage(self, stage_name):
        """
        Marcar inicio de una etapa

        Args:
            stage_name: Nombre de la etapa (source, encoder, channel, decoder)
        """
        self.stage_starts[stage_name] = time.time()
        log_debug('profiler', f"[{stage_name.upper()}] Inicio")

    def end_stage(self, stage_name, symbol_index=None):
        """
        Marcar fin de una etapa y registrar tiempo

        Args:
            stage_name: Nombre de la etapa
            symbol_index: Índice del símbolo procesado (opcional)
        """
        if stage_name not in self.stage_starts:
            log_warning('profiler', f"[{stage_name}] No hay start_stage previo - ignorando")
            return

        end_time = time.time()
        start_time = self.stage_starts[stage_name]
        elapsed = end_time - start_time

        # Registrar tiempo
        self.timings[stage_name].append({
            'elapsed': elapsed,
            'symbol': symbol_index,
            'timestamp': end_time
        })

        # Logging condicional: solo cada 10 símbolos para no saturar logs
        if symbol_index is None or symbol_index % 10 == 0:
            log_info('profiler', f"[{stage_name.upper()}] Fin - {elapsed:.3f}s" +
                    (f" (símbolo {symbol_index})" if symbol_index is not None else ""))

        # Limpiar start
        del self.stage_starts[stage_name]

    def increment_symbols(self, count=1):
        """Incrementar contador de símbolos procesados"""
        self.symbol_count += count

    def get_stage_stats(self, stage_name):
        """
        Obtener estadísticas de una etapa

        Args:
            stage_name: Nombre de la etapa

        Returns:
            dict: Estadísticas (total, avg, min, max, count)
        """
        if stage_name not in self.timings or not self.timings[stage_name]:
            return {
                'total': 0.0,
                'avg': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0
            }

        times = [t['elapsed'] for t in self.timings[stage_name]]

        return {
            'total': sum(times),
            'avg': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'count': len(times),
            'std': self._std_dev(times) if len(times) > 1 else 0.0
        }

    def _std_dev(self, values):
        """Calcular desviación estándar"""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def get_full_report(self):
        """
        Generar reporte completo de profiling

        Returns:
            dict: Reporte con todas las métricas
        """
        total_time = (self.total_end - self.total_start) if self.total_end else 0.0

        stages = ['source', 'encoder', 'channel', 'decoder']
        stage_stats = {stage: self.get_stage_stats(stage) for stage in stages}

        # Calcular tiempo total de stages (puede ser diferente del total medido)
        stages_total = sum(stats['total'] for stats in stage_stats.values())

        # Calcular porcentajes
        stage_percentages = {}
        if stages_total > 0:
            for stage, stats in stage_stats.items():
                stage_percentages[stage] = (stats['total'] / stages_total) * 100

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_time': total_time,
            'symbol_count': self.symbol_count,
            'throughput': self.symbol_count / total_time if total_time > 0 else 0.0,
            'stages': stage_stats,
            'percentages': stage_percentages,
            'stages_total': stages_total,
            'overhead': total_time - stages_total if total_time > 0 else 0.0
        }

        return report

    def print_summary(self):
        """Imprimir resumen de profiling en consola"""
        report = self.get_full_report()

        print("\n" + "=" * 80)
        print(" REPORTE DE TIEMPOS DE EJECUCIÓN - SISTEMA OAM")
        print("=" * 80)

        print(f"\n📊 RESUMEN GENERAL:")
        print(f"  • Tiempo total:        {report['total_time']:.3f} s")
        print(f"  • Símbolos procesados: {report['symbol_count']}")
        print(f"  • Throughput:          {report['throughput']:.2f} símbolos/s")
        print(f"  • Overhead sistema:    {report['overhead']:.3f} s ({(report['overhead']/report['total_time']*100):.1f}%)")

        print(f"\n⏱️  TIEMPOS POR ETAPA:")
        print(f"  {'Etapa':<15} {'Total (s)':<12} {'Promedio (s)':<15} {'Min (s)':<10} {'Max (s)':<10} {'%':<8}")
        print(f"  {'-'*15} {'-'*12} {'-'*15} {'-'*10} {'-'*10} {'-'*8}")

        for stage in ['source', 'encoder', 'channel', 'decoder']:
            stats = report['stages'][stage]
            pct = report['percentages'].get(stage, 0.0)
            print(f"  {stage.upper():<15} {stats['total']:>10.3f}   {stats['avg']:>13.6f}   "
                  f"{stats['min']:>8.6f}   {stats['max']:>8.6f}   {pct:>6.1f}%")

        print(f"\n  {'TOTAL STAGES':<15} {report['stages_total']:>10.3f}                                        100.0%")

        print("\n" + "=" * 80)

        # Logging también
        log_info('profiler', f"PROFILING: Total={report['total_time']:.3f}s, Símbolos={report['symbol_count']}, Throughput={report['throughput']:.2f} sym/s")
        for stage in ['source', 'encoder', 'channel', 'decoder']:
            stats = report['stages'][stage]
            pct = report['percentages'].get(stage, 0.0)
            log_info('profiler', f"  {stage.upper()}: {stats['total']:.3f}s ({pct:.1f}%), avg={stats['avg']:.6f}s/símbolo")

    def save_to_json(self, filepath):
        """
        Guardar reporte completo en JSON

        Args:
            filepath: Ruta del archivo JSON de salida
        """
        report = self.get_full_report()

        try:
            # Asegurar que el directorio existe
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)

            log_info('profiler', f"Reporte de profiling guardado: {filepath}")

        except Exception as e:
            log_warning('profiler', f"Error guardando reporte de profiling: {e}")

    def reset(self):
        """Resetear profiler para nueva ejecución"""
        self.timings.clear()
        self.stage_starts.clear()
        self.total_start = None
        self.total_end = None
        self.symbol_count = 0
        log_debug('profiler', "Profiler reset completado")


# Instancia global del profiler
profiler = OAMProfiler()


# Funciones de conveniencia para uso en otros módulos
def start_profiling():
    """Iniciar profiling total"""
    profiler.start_total()

def end_profiling():
    """Finalizar profiling total y mostrar reporte"""
    profiler.end_total()
    profiler.print_summary()

def start_stage(stage_name):
    """Iniciar medición de etapa"""
    profiler.start_stage(stage_name)

def end_stage(stage_name, symbol_index=None):
    """Finalizar medición de etapa"""
    profiler.end_stage(stage_name, symbol_index)

def increment_symbols(count=1):
    """Incrementar contador de símbolos"""
    profiler.increment_symbols(count)

def save_profiling_report(filepath="current_run/profiling_report.json"):
    """Guardar reporte de profiling"""
    profiler.save_to_json(filepath)

def get_profiling_report():
    """Obtener reporte de profiling"""
    return profiler.get_full_report()

def reset_profiling():
    """Resetear profiler"""
    profiler.reset()


if __name__ == "__main__":
    # Test del profiler
    print("=== TEST DEL PROFILER ===\n")

    profiler.start_total()

    # Simular procesamiento
    import time

    for i in range(5):
        profiler.start_stage('source')
        time.sleep(0.01)
        profiler.end_stage('source', i)

        profiler.start_stage('encoder')
        time.sleep(0.05)
        profiler.end_stage('encoder', i)

        profiler.start_stage('channel')
        time.sleep(0.03)
        profiler.end_stage('channel', i)

        profiler.start_stage('decoder')
        time.sleep(0.02)
        profiler.end_stage('decoder', i)

        profiler.increment_symbols()

    profiler.end_total()
    profiler.print_summary()
    profiler.save_to_json("/tmp/test_profiling.json")

    print("\nTest completado - reporte guardado en /tmp/test_profiling.json")
