"""
Модуль для сбора метрик в реальном времени.

Отслеживает:
    - Latency операций (inference, routing, memory access)
    - Throughput (tokens/sec, requests/sec)
    - Memory usage по модулям
    - Routing statistics (confidence, распределение)
    - Data flow между модулями
"""

import time
import psutil
from collections import deque, defaultdict
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MetricSnapshot:
    """Снимок метрики в конкретный момент времени."""
    timestamp: float
    value: float


class MetricsCollector:
    """
    Сборщик метрик для real-time мониторинга системы MoE.

    Использует circular buffers (deque) для хранения истории метрик
    с автоматической очисткой старых данных.

    Примеры использования:
        >>> collector = MetricsCollector(history_size=1000)
        >>>
        >>> # Трекинг операции
        >>> with collector.track_operation('inference'):
        >>>     result = model.generate(input_ids)
        >>>
        >>> # Запись data flow
        >>> collector.record_data_transfer('Router', 'Expert_Python', data_tensor)
        >>>
        >>> # Получение метрик
        >>> metrics = collector.get_latest_metrics(window_sec=60)
    """

    def __init__(self, history_size: int = 1000):
        """
        Инициализация сборщика метрик.

        Args:
            history_size: Максимальный размер истории для каждой метрики
        """
        self.history_size = history_size

        # Circular buffers для истории метрик
        self.metrics_history = {
            'timestamps': deque(maxlen=history_size),
            'latency_inference': deque(maxlen=history_size),
            'latency_routing': deque(maxlen=history_size),
            'latency_memory': deque(maxlen=history_size),
            'throughput_tokens': deque(maxlen=history_size),
            'throughput_requests': deque(maxlen=history_size),
            'memory_usage_mb': deque(maxlen=history_size),
            'routing_confidence': deque(maxlen=history_size),
        }

        # Data flow tracking (для визуализации)
        # Формат: {('module_from', 'module_to'): [(bytes_count, timestamp), ...]}
        self.data_flow = defaultdict(lambda: deque(maxlen=100))

        # Routing statistics
        self.routing_stats = defaultdict(int)  # {expert_id: count}

        # Module-level метрики
        self.module_metrics = {}  # {'Router': {...}, 'Memory': {...}, ...}

        # Process для измерения системных метрик
        self.process = psutil.Process()

        # Счётчики для throughput
        self._tokens_generated = 0
        self._requests_processed = 0
        self._last_throughput_calc = time.time()

    @contextmanager
    def track_operation(self, operation_name: str):
        """
        Context manager для трекинга времени выполнения операций.

        Автоматически записывает latency операции в историю.

        Args:
            operation_name: Название операции ('inference', 'routing', 'memory')

        Usage:
            >>> with collector.track_operation('inference'):
            >>>     result = model.generate(...)
        """
        start_time = time.time()
        try:
            yield
        finally:
            latency_ms = (time.time() - start_time) * 1000
            self._record_latency(operation_name, latency_ms)

    def _record_latency(self, operation: str, latency_ms: float):
        """Записывает latency операции в историю."""
        timestamp = time.time()

        self.metrics_history['timestamps'].append(timestamp)

        if operation == 'inference':
            self.metrics_history['latency_inference'].append(latency_ms)
        elif operation == 'routing':
            self.metrics_history['latency_routing'].append(latency_ms)
        elif operation == 'memory':
            self.metrics_history['latency_memory'].append(latency_ms)

    def record_data_transfer(self, from_module: str, to_module: str, bytes_count: int):
        """
        Записывает передачу данных между модулями.

        Args:
            from_module: Модуль-отправитель
            to_module: Модуль-получатель
            bytes_count: Количество переданных байт
        """
        key = (from_module, to_module)
        timestamp = time.time()

        self.data_flow[key].append((bytes_count, timestamp))

        # Очищаем старые записи (>60 секунд)
        self._cleanup_old_transfers(key, timestamp - 60)

    def _cleanup_old_transfers(self, key: Tuple[str, str], cutoff_timestamp: float):
        """Удаляет старые записи data flow."""
        if key not in self.data_flow:
            return

        # Фильтруем записи новее cutoff_timestamp
        transfers = self.data_flow[key]
        while transfers and transfers[0][1] < cutoff_timestamp:
            transfers.popleft()

    def get_transfer_speed(self, from_module: str, to_module: str) -> float:
        """
        Возвращает скорость передачи данных (bytes/sec).

        Вычисляется как сумма bytes за последние 60 секунд / 60.

        Args:
            from_module: Модуль-отправитель
            to_module: Модуль-получатель

        Returns:
            Скорость в bytes/sec
        """
        key = (from_module, to_module)

        if key not in self.data_flow or not self.data_flow[key]:
            return 0.0

        transfers = self.data_flow[key]

        # Считаем только за последние 60 секунд
        current_time = time.time()
        cutoff_time = current_time - 60

        total_bytes = sum(
            bytes_count
            for bytes_count, timestamp in transfers
            if timestamp >= cutoff_time
        )

        # Определяем временной интервал
        if transfers:
            oldest_timestamp = max(transfers[0][1], cutoff_time)
            time_window = current_time - oldest_timestamp

            if time_window > 0:
                return total_bytes / time_window

        return 0.0

    def record_routing(self, expert_id: str, confidence: float):
        """
        Записывает результат routing.

        Args:
            expert_id: ID выбранного эксперта
            confidence: Уверенность router'а (0.0-1.0)
        """
        timestamp = time.time()

        self.metrics_history['timestamps'].append(timestamp)
        self.metrics_history['routing_confidence'].append(confidence)

        self.routing_stats[expert_id] += 1

    def record_tokens_generated(self, tokens_count: int):
        """Записывает количество сгенерированных токенов."""
        self._tokens_generated += tokens_count

    def record_request_processed(self):
        """Записывает обработанный запрос."""
        self._requests_processed += 1

    def update_system_metrics(self):
        """Обновляет системные метрики (CPU, RAM)."""
        timestamp = time.time()

        # Memory usage
        memory_mb = self.process.memory_info().rss / (1024 * 1024)

        self.metrics_history['timestamps'].append(timestamp)
        self.metrics_history['memory_usage_mb'].append(memory_mb)

        # Вычисляем throughput
        time_elapsed = timestamp - self._last_throughput_calc

        if time_elapsed >= 1.0:  # Обновляем каждую секунду
            tokens_per_sec = self._tokens_generated / time_elapsed
            requests_per_sec = self._requests_processed / time_elapsed

            self.metrics_history['throughput_tokens'].append(tokens_per_sec)
            self.metrics_history['throughput_requests'].append(requests_per_sec)

            # Сбрасываем счётчики
            self._tokens_generated = 0
            self._requests_processed = 0
            self._last_throughput_calc = timestamp

    def get_latest_metrics(self, window_sec: int = 60) -> Dict[str, Any]:
        """
        Возвращает последние метрики за указанное окно времени.

        Args:
            window_sec: Размер окна в секундах

        Returns:
            Словарь с метриками:
                - latency_avg_ms: средний latency
                - throughput_tokens_avg: средний throughput (tokens/sec)
                - memory_usage_mb: текущее использование памяти
                - routing_confidence_avg: средняя confidence
                - data_flow: скорости передачи между модулями
        """
        current_time = time.time()
        cutoff_time = current_time - window_sec

        # Фильтруем метрики по временному окну
        indices = [
            i for i, ts in enumerate(self.metrics_history['timestamps'])
            if ts >= cutoff_time
        ]

        if not indices:
            return self._get_empty_metrics()

        # Вычисляем средние значения
        def avg(metric_key):
            values = [
                self.metrics_history[metric_key][i]
                for i in indices
                if i < len(self.metrics_history[metric_key])
            ]
            return sum(values) / len(values) if values else 0.0

        # Собираем data flow stats
        data_flow_stats = {}
        for key in self.data_flow.keys():
            speed = self.get_transfer_speed(key[0], key[1])
            if speed > 0:
                data_flow_stats[key] = speed

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)

        return {
            'latency_inference_avg_ms': avg('latency_inference'),
            'latency_routing_avg_ms': avg('latency_routing'),
            'latency_memory_avg_ms': avg('latency_memory'),
            'throughput_tokens_avg': avg('throughput_tokens'),
            'throughput_requests_avg': avg('throughput_requests'),
            'memory_usage_mb': self.metrics_history['memory_usage_mb'][-1] if self.metrics_history['memory_usage_mb'] else 0.0,
            'routing_confidence_avg': avg('routing_confidence'),
            'cpu_usage_percent': cpu_percent,
            'data_flow': data_flow_stats,
            'timestamp': current_time
        }

    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Возвращает пустые метрики (когда нет данных)."""
        return {
            'latency_inference_avg_ms': 0.0,
            'latency_routing_avg_ms': 0.0,
            'latency_memory_avg_ms': 0.0,
            'throughput_tokens_avg': 0.0,
            'throughput_requests_avg': 0.0,
            'memory_usage_mb': 0.0,
            'routing_confidence_avg': 0.0,
            'cpu_usage_percent': 0.0,
            'data_flow': {},
            'timestamp': time.time()
        }

    def get_routing_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику по routing.

        Returns:
            {
                'distribution': {expert_id: count},
                'total_requests': int,
                'most_used_expert': str
            }
        """
        total = sum(self.routing_stats.values())

        most_used = max(
            self.routing_stats.items(),
            key=lambda x: x[1],
            default=('none', 0)
        )[0]

        return {
            'distribution': dict(self.routing_stats),
            'total_requests': total,
            'most_used_expert': most_used
        }

    def get_history_for_plot(self, metric_name: str, max_points: int = 100) -> List[Dict[str, float]]:
        """
        Возвращает историю метрики в формате для построения графиков.

        Args:
            metric_name: Название метрики ('latency_inference', 'throughput_tokens', etc.)
            max_points: Максимальное количество точек

        Returns:
            Список словарей {'timestamp': float, 'value': float}
        """
        if metric_name not in self.metrics_history:
            return []

        timestamps = list(self.metrics_history['timestamps'])
        values = list(self.metrics_history[metric_name])

        # Берём последние max_points
        if len(timestamps) > max_points:
            timestamps = timestamps[-max_points:]
            values = values[-max_points:]

        # Форматируем для Plotly/Gradio
        return [
            {'timestamp': ts, 'value': val}
            for ts, val in zip(timestamps, values)
            if val is not None
        ]

    def reset(self):
        """Сбрасывает все метрики."""
        for key in self.metrics_history:
            self.metrics_history[key].clear()

        self.data_flow.clear()
        self.routing_stats.clear()
        self.module_metrics.clear()

        self._tokens_generated = 0
        self._requests_processed = 0
        self._last_throughput_calc = time.time()
