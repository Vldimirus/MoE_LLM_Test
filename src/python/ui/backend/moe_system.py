"""
Унифицированный интерфейс для Domain-Specific MoE системы.

Объединяет:
    - ExpertModel management (загрузка/выгрузка экспертов)
    - Router (маршрутизация запросов)
    - ThreeLevelMemory (управление памятью)
    - Trainer (обучение моделей)
    - MetricsCollector (real-time метрики)
"""

import sys
import time
import yaml
import torch
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple, Any
from collections import OrderedDict

# Импорты из проекта
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.expert import ExpertModel
from routing.router import SimpleRouter, RoutingResult
from memory.three_level_memory import ThreeLevelMemory
from training.dataset import SimpleTokenizer
from ui.backend.metrics_collector import MetricsCollector


class MoESystem:
    """
    Унифицированный интерфейс для Domain-Specific MoE системы.

    Предоставляет единую точку входа для всех операций:
        - Chat через полный pipeline (Router → Expert → Generation)
        - Управление экспертами (lazy loading, LRU cache)
        - Мониторинг системы (метрики, data flow)
        - Обучение моделей
        - Визуализация архитектуры

    Примеры использования:
        >>> system = MoESystem(config_path="configs/ui_config.yaml")
        >>>
        >>> # Chat
        >>> response = system.chat("Напиши функцию на Python")
        >>> print(response['response'])
        >>>
        >>> # Мониторинг
        >>> metrics = system.get_system_metrics()
        >>> print(f"Latency: {metrics['latency_inference_avg_ms']:.2f}ms")
    """

    def __init__(self, config_path: str = "configs/ui_config.yaml"):
        """
        Инициализация MoE системы.

        Args:
            config_path: Путь к файлу конфигурации
        """
        # Загрузка конфигурации
        self.config = self._load_config(config_path)

        # Инициализация компонентов
        self.device = self.config.get('system', {}).get('device', 'cpu')
        self.max_loaded_experts = self.config.get('system', {}).get('max_loaded_experts', 3)

        # Router
        router_config = self.config.get('router', {})
        self.router = SimpleRouter(
            default_expert_id=router_config.get('default_expert_id', 'general')
        )

        # Регистрируем экспертов из конфигурации
        self._register_experts_from_config()

        # Memory
        memory_config = self.config.get('memory', {})
        max_tokens = memory_config.get('max_tokens_per_level', 250000)
        self.memory = ThreeLevelMemory(max_tokens_per_level=max_tokens)

        # Tokenizer (будет инициализирован при необходимости)
        self.tokenizer = None

        # Кэш загруженных экспертов (LRU)
        self.experts: OrderedDict[str, ExpertModel] = OrderedDict()

        # Metrics collector
        self.metrics_collector = MetricsCollector(
            history_size=self.config.get('system', {}).get('metrics_history_size', 1000)
        )

        # Trainer (будет инициализирован при запуске обучения)
        self.trainer = None
        self.training_active = False

        # Статистика модулей для визуализации
        self.module_stats = {
            'Router': {'type': 'router', 'layers': 0, 'params': 0, 'memory_mb': 0},
            'Memory': {'type': 'memory', 'layers': 0, 'params': 0, 'memory_mb': 0},
        }

    def _load_config(self, config_path: str) -> Dict:
        """Загружает конфигурацию из YAML файла."""
        config_file = Path(config_path)

        if not config_file.exists():
            # Возвращаем конфигурацию по умолчанию
            return self._get_default_config()

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Возвращает конфигурацию по умолчанию."""
        return {
            'system': {
                'device': 'cpu',
                'max_loaded_experts': 3,
                'metrics_history_size': 1000
            },
            'router': {
                'default_expert_id': 'general'
            },
            'memory': {
                'max_tokens_per_level': 250000
            },
            'experts': {
                'models_dir': 'models/experts',
                'available_experts': [
                    {
                        'id': 'general',
                        'name': 'General Assistant',
                        'priority': 3,
                        'keywords': []
                    }
                ]
            },
            'monitoring': {
                'enable_system_metrics': True,
                'enable_dataflow_tracking': True
            }
        }

    def _register_experts_from_config(self):
        """Регистрирует экспертов в Router из конфигурации."""
        experts_config = self.config.get('experts', {}).get('available_experts', [])

        for expert_cfg in experts_config:
            self.router.add_expert(
                expert_id=expert_cfg['id'],
                name=expert_cfg.get('name', expert_cfg['id']),
                keywords=expert_cfg.get('keywords', []),
                priority=expert_cfg.get('priority', 5)
            )

    # === CHAT INTERFACE ===

    def chat(
        self,
        user_message: str,
        expert_id: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Обработка пользовательского сообщения через полный pipeline.

        Args:
            user_message: Сообщение пользователя
            expert_id: ID эксперта (если None - используется Router)
            max_tokens: Максимальное количество токенов для генерации
            temperature: Температура sampling

        Returns:
            {
                'response': str,
                'expert_used': str,
                'routing_confidence': float,
                'tokens_generated': int,
                'latency_ms': float,
                'memory_stats': dict
            }
        """
        start_time = time.time()

        try:
            # 1. Routing (если expert_id не указан)
            if expert_id is None:
                with self.metrics_collector.track_operation('routing'):
                    routing_results = self.router.route(user_message, top_k=1)

                if routing_results:
                    expert_id = routing_results[0].expert_id
                    confidence = routing_results[0].confidence
                else:
                    expert_id = self.router.default_expert_id
                    confidence = 0.5

                self.metrics_collector.record_routing(expert_id, confidence)
            else:
                confidence = 1.0  # Manual selection

            # 2. Подготовка контекста из памяти
            with self.metrics_collector.track_operation('memory'):
                # Добавляем сообщение в память
                token_count = len(user_message.split())  # Упрощённый подсчёт
                self.memory.add_message(
                    content=user_message,
                    token_count=token_count,
                    importance=0.5
                )

                memory_stats = self.memory.get_stats()

            # 3. Inference (генерация ответа)
            with self.metrics_collector.track_operation('inference'):
                # Для прототипа: возвращаем заглушку
                # В полной реализации здесь будет вызов expert.generate()
                response_text = self._generate_mock_response(user_message, expert_id)
                tokens_generated = len(response_text.split())

            # 4. Записываем метрики
            self.metrics_collector.record_tokens_generated(tokens_generated)
            self.metrics_collector.record_request_processed()

            # 5. Data flow tracking
            if self.config.get('monitoring', {}).get('enable_dataflow_tracking', True):
                # Router → Expert
                self.metrics_collector.record_data_transfer('Router', expert_id, token_count * 4)
                # Expert → Output
                self.metrics_collector.record_data_transfer(expert_id, 'Output', tokens_generated * 4)

            latency_ms = (time.time() - start_time) * 1000

            return {
                'response': response_text,
                'expert_used': expert_id,
                'routing_confidence': confidence,
                'tokens_generated': tokens_generated,
                'latency_ms': latency_ms,
                'memory_stats': memory_stats
            }

        except Exception as e:
            return {
                'response': f"Error: {str(e)}",
                'expert_used': expert_id or 'unknown',
                'routing_confidence': 0.0,
                'tokens_generated': 0,
                'latency_ms': (time.time() - start_time) * 1000,
                'memory_stats': {}
            }

    def _generate_mock_response(self, user_message: str, expert_id: str) -> str:
        """
        Генерирует mock ответ (заглушка для прототипа).

        В полной реализации будет вызов model.generate().
        """
        responses = {
            'general': f"Я общий ассистент. Вы сказали: '{user_message}'. Чем могу помочь?",
            'python_expert': f"Я эксперт по Python. Ваш запрос: '{user_message}'. Предлагаю создать функцию...",
            'math_expert': f"Я математический эксперт. Задача: '{user_message}'. Решение: ..."
        }

        return responses.get(expert_id, f"Ответ от {expert_id}: получен запрос '{user_message}'")

    # === EXPERT MANAGEMENT ===

    def load_expert(self, expert_id: str) -> bool:
        """
        Загружает эксперта в память (lazy loading).

        Args:
            expert_id: ID эксперта

        Returns:
            True если загрузка успешна
        """
        # Проверяем, не загружен ли уже
        if expert_id in self.experts:
            # Перемещаем в конец (LRU)
            self.experts.move_to_end(expert_id)
            return True

        # Проверяем лимит загруженных экспертов
        if len(self.experts) >= self.max_loaded_experts:
            # Выгружаем самого старого (first item)
            oldest_id = next(iter(self.experts))
            self.unload_expert(oldest_id)

        try:
            # Загружаем модель (прототип - создаём заглушку)
            # В полной реализации:
            # model_path = self.config['experts']['models_dir'] + f'/{expert_id}'
            # expert = ExpertModel.load(model_path)

            expert = None  # Заглушка для прототипа

            self.experts[expert_id] = expert

            # Обновляем статистику
            self.module_stats[expert_id] = {
                'type': 'expert',
                'layers': 8,  # Заглушка
                'params': 0,  # Будет вычислено при загрузке реальной модели
                'memory_mb': 0.0
            }

            return True

        except Exception as e:
            print(f"Error loading expert {expert_id}: {e}")
            return False

    def unload_expert(self, expert_id: str) -> bool:
        """
        Выгружает эксперта из памяти.

        Args:
            expert_id: ID эксперта

        Returns:
            True если выгрузка успешна
        """
        if expert_id in self.experts:
            del self.experts[expert_id]

            if expert_id in self.module_stats:
                del self.module_stats[expert_id]

            return True

        return False

    def list_available_experts(self) -> List[Dict[str, Any]]:
        """
        Возвращает список всех доступных экспертов с метаданными.

        Returns:
            [
                {'id': 'general', 'name': 'General Assistant', 'loaded': True, ...},
                ...
            ]
        """
        experts_config = self.config.get('experts', {}).get('available_experts', [])

        experts_list = []
        for expert_cfg in experts_config:
            expert_id = expert_cfg['id']

            experts_list.append({
                'id': expert_id,
                'name': expert_cfg.get('name', expert_id),
                'priority': expert_cfg.get('priority', 5),
                'keywords': expert_cfg.get('keywords', []),
                'loaded': expert_id in self.experts,
                'status': 'loaded' if expert_id in self.experts else 'unloaded'
            })

        return experts_list

    # === MEMORY MANAGEMENT ===

    def get_memory_stats(self) -> Dict[str, Any]:
        """Возвращает статистику по 3-уровневой памяти."""
        return self.memory.get_stats()

    # === SYSTEM METRICS ===

    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Возвращает текущие системные метрики.

        Returns:
            Словарь с метриками (latency, throughput, memory, CPU)
        """
        # Обновляем системные метрики
        self.metrics_collector.update_system_metrics()

        # Получаем latest metrics
        metrics = self.metrics_collector.get_latest_metrics(window_sec=60)

        # Добавляем информацию о загруженных экспертах
        metrics['loaded_experts_count'] = len(self.experts)
        metrics['loaded_experts'] = list(self.experts.keys())

        return metrics

    def get_data_flow_stats(self) -> Dict[Tuple[str, str], float]:
        """
        Возвращает скорости передачи данных между модулями.

        Returns:
            {
                ('Router', 'Expert_Python'): 1234.5,  # bytes/sec
                ('Memory', 'Router'): 567.8,
                ...
            }
        """
        data_flow = {}

        # Получаем все зарегистрированные пары модулей
        for key in self.metrics_collector.data_flow.keys():
            speed = self.metrics_collector.get_transfer_speed(key[0], key[1])
            if speed > 0:
                data_flow[key] = speed

        return data_flow

    # === ARCHITECTURE VISUALIZATION ===

    def get_architecture_graph(self) -> Dict[str, Any]:
        """
        Возвращает структуру архитектуры для визуализации.

        Returns:
            {
                'nodes': [
                    {'id': 'Router', 'type': 'router', 'layers': 0, 'params': 0, 'memory_mb': 0},
                    {'id': 'Memory', 'type': 'memory', ...},
                    {'id': 'Expert_Python', 'type': 'expert', 'layers': 8, ...},
                    ...
                ],
                'edges': [
                    {'from': 'Router', 'to': 'Expert_Python', 'data_flow': 1234.5, 'activity_level': 0.8},
                    ...
                ]
            }
        """
        nodes = []
        edges = []

        # Добавляем основные модули
        nodes.append({
            'id': 'Router',
            'type': 'router',
            'layers': 0,
            'params_count': 0,
            'memory_mb': 0.1
        })

        nodes.append({
            'id': 'Memory',
            'type': 'memory',
            'layers': 0,
            'params_count': 0,
            'memory_mb': self.memory.get_stats().get('total_chunks', 0) * 0.001  # Примерная оценка
        })

        # Добавляем загруженных экспертов
        for expert_id in self.experts.keys():
            stats = self.module_stats.get(expert_id, {})
            nodes.append({
                'id': expert_id,
                'type': 'expert',
                'layers': stats.get('layers', 8),
                'params_count': stats.get('params', 1_000_000),  # Заглушка
                'memory_mb': stats.get('memory_mb', 100.0)  # Заглушка
            })

        # Формируем рёбра на основе data flow
        data_flow = self.get_data_flow_stats()

        for (from_module, to_module), speed in data_flow.items():
            # Вычисляем activity level (0.0-1.0)
            # Нормализуем относительно максимальной скорости
            max_speed = max(data_flow.values()) if data_flow.values() else 1.0
            activity_level = min(speed / max_speed, 1.0) if max_speed > 0 else 0.0

            edges.append({
                'from': from_module,
                'to': to_module,
                'data_flow': speed,  # bytes/sec
                'activity_level': activity_level
            })

        # Добавляем стандартные соединения (если нет data flow)
        standard_edges = [
            ('Router', 'Memory'),
        ]

        for from_mod, to_mod in standard_edges:
            # Проверяем, нет ли уже такого ребра
            if not any(e['from'] == from_mod and e['to'] == to_mod for e in edges):
                edges.append({
                    'from': from_mod,
                    'to': to_mod,
                    'data_flow': 0.0,
                    'activity_level': 0.1
                })

        # Добавляем соединения Router → Expert для всех загруженных экспертов
        for expert_id in self.experts.keys():
            if not any(e['from'] == 'Router' and e['to'] == expert_id for e in edges):
                edges.append({
                    'from': 'Router',
                    'to': expert_id,
                    'data_flow': 0.0,
                    'activity_level': 0.1
                })

        return {
            'nodes': nodes,
            'edges': edges
        }

    # === TRAINING (ЗАГЛУШКА) ===

    def start_training(
        self,
        expert_id: str,
        train_config: Dict[str, Any],
        callback: Optional[Callable] = None
    ):
        """
        Запуск обучения эксперта (заглушка для прототипа).

        Args:
            expert_id: ID эксперта для обучения
            train_config: Конфигурация обучения
            callback: Callback для обновления UI
        """
        # В полной реализации здесь будет запуск Trainer
        self.training_active = True
        print(f"Training started for {expert_id} with config: {train_config}")

        # Mock callback
        if callback:
            for epoch in range(1, train_config.get('num_epochs', 10) + 1):
                metrics = {
                    'loss': 3.0 - epoch * 0.1,
                    'perplexity': 20.0 - epoch * 0.5
                }
                callback(epoch, metrics)

        self.training_active = False

    def get_training_progress(self) -> Dict[str, Any]:
        """Возвращает прогресс обучения (заглушка)."""
        return {
            'active': self.training_active,
            'current_epoch': 0,
            'total_epochs': 0,
            'current_loss': 0.0,
            'current_perplexity': 0.0
        }
