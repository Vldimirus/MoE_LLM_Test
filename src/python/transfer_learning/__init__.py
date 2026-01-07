"""
Transfer Learning модуль для импорта знаний из GGUF моделей.

Основные компоненты:
    - GGUFParser: парсинг GGUF файлов с lazy loading
    - WeightMapper: автоматический mapping весов GGUF → ExpertModel
    - TokenizerAligner: alignment vocabulary между GGUF и BPE
    - MemoryManager: управление памятью при загрузке больших моделей
    - TransferPipeline: high-level API для всего процесса

Пример использования:
    >>> from python.transfer_learning import TransferLearningPipeline
    >>>
    >>> pipeline = TransferLearningPipeline(
    ...     gguf_path="models/gguf/phi-3-mini-q8.gguf",
    ...     target_model_config={
    ...         'vocab_size': 8000,
    ...         'd_model': 512,
    ...         'n_layers': 6,
    ...         'n_heads': 8
    ...     }
    ... )
    >>>
    >>> # Проверка совместимости
    >>> compat = pipeline.validate_compatibility()
    >>>
    >>> # Инициализация модели
    >>> model = pipeline.initialize_model_from_gguf()
"""

from .gguf_parser import GGUFParser
from .weight_mapper import WeightMapper
from .tokenizer_aligner import TokenizerAligner
from .memory_manager import MemoryManager
from .transfer_pipeline import TransferLearningPipeline

__all__ = [
    'GGUFParser',
    'WeightMapper',
    'TokenizerAligner',
    'MemoryManager',
    'TransferLearningPipeline'
]
