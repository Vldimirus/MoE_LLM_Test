"""
Transfer Learning Pipeline для инициализации ExpertModel из GGUF моделей.

Этот модуль предоставляет high-level API для всего процесса transfer learning:
    - Парсинг GGUF файла
    - Проверка совместимости
    - Alignment vocabulary
    - Перенос весов
    - Инициализация ExpertModel

Пример использования:
    >>> from python.transfer_learning import TransferLearningPipeline
    >>>
    >>> pipeline = TransferLearningPipeline(
    ...     gguf_path="models/gguf/phi-3-mini-q8.gguf",
    ...     target_model_config={
    ...         'vocab_size': 8000,
    ...         'd_model': 512,
    ...         'n_layers': 6,
    ...         'n_heads': 8,
    ...         'd_ff': 2048
    ...     },
    ...     bpe_tokenizer_path="models/tokenizers/bpe_multilang.model"
    ... )
    >>>
    >>> # Проверка совместимости
    >>> compat = pipeline.validate_compatibility()
    >>> print(f"Vocab overlap: {compat['vocab_overlap']:.1%}")
    >>>
    >>> # Инициализация модели
    >>> model = pipeline.initialize_model_from_gguf(
    ...     layers_to_transfer=[0, 1, 2, 3, 4, 5],
    ...     freeze_transferred_layers=True
    ... )
    >>>
    >>> # Модель готова к fine-tuning!
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import torch

from .gguf_parser import GGUFParser
from .weight_mapper import WeightMapper
from .tokenizer_aligner import TokenizerAligner
from .memory_manager import MemoryManager

# Import ExpertModel
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.expert import ExpertModel


class TransferLearningPipeline:
    """
    Главный API для transfer learning из GGUF моделей.

    Оркестрирует все компоненты (GGUFParser, WeightMapper, TokenizerAligner, MemoryManager)
    для безопасного и эффективного переноса знаний из GGUF модели в ExpertModel.

    Attributes:
        gguf_path: Путь к GGUF файлу
        target_model_config: Конфигурация целевой модели (vocab_size, d_model, etc.)
        bpe_tokenizer: BPE токенайзер (опционально)
        max_ram_gb: Максимальное использование RAM

    Example:
        >>> pipeline = TransferLearningPipeline(
        ...     gguf_path="phi-3-mini-q8.gguf",
        ...     target_model_config={'vocab_size': 8000, 'd_model': 512, 'n_layers': 6}
        ... )
        >>>
        >>> model = pipeline.initialize_model_from_gguf()
    """

    def __init__(
        self,
        gguf_path: str,
        target_model_config: Dict[str, Any],
        bpe_tokenizer_path: Optional[str] = None,
        max_ram_gb: float = 12.0
    ):
        """
        Инициализирует Transfer Learning Pipeline.

        Args:
            gguf_path: Путь к GGUF файлу
            target_model_config: Конфигурация целевой модели
                Обязательные ключи: vocab_size, d_model, n_layers, n_heads
                Опциональные: d_ff (default 4*d_model), max_seq_len (default 2048), dropout (default 0.1)
            bpe_tokenizer_path: Путь к BPE токенайзеру (опционально)
            max_ram_gb: Максимальное использование RAM в GB

        Raises:
            FileNotFoundError: Если GGUF файл не найден
            ValueError: Если конфигурация некорректна
        """
        self.gguf_path = Path(gguf_path)
        self.target_model_config = target_model_config
        self.bpe_tokenizer_path = bpe_tokenizer_path
        self.max_ram_gb = max_ram_gb

        # Валидация конфигурации
        self._validate_config()

        # Инициализация компонентов
        print(f"\n{'='*60}")
        print("🚀 Инициализация Transfer Learning Pipeline")
        print(f"{'='*60}")

        # 1. GGUF Parser
        self.gguf_parser = GGUFParser(str(self.gguf_path))
        self.gguf_metadata = self.gguf_parser.get_metadata()

        # 2. Memory Manager
        self.memory_manager = MemoryManager(
            max_ram_gb=max_ram_gb,
            safety_margin_gb=2.0
        )

        # 3. BPE Tokenizer (если указан)
        self.bpe_tokenizer = None
        if bpe_tokenizer_path:
            try:
                # Пытаемся загрузить BPE токенайзер
                # TODO: Интеграция с реальным BPE токенайзером
                print(f"⚠️ BPE токенайзер пока не загружен (будет в следующей версии)")
                self.bpe_tokenizer = None
            except Exception as e:
                print(f"⚠️ Не удалось загрузить BPE токенайзер: {e}")
                self.bpe_tokenizer = None

        # 4. Tokenizer Aligner
        self.tokenizer_aligner = TokenizerAligner(
            gguf_parser=self.gguf_parser,
            bpe_tokenizer=self.bpe_tokenizer
        )

        print(f"{'='*60}\n")

    def _validate_config(self):
        """
        Валидирует target_model_config.

        Raises:
            ValueError: Если конфигурация некорректна
        """
        required_keys = ['vocab_size', 'd_model', 'n_layers', 'n_heads']

        for key in required_keys:
            if key not in self.target_model_config:
                raise ValueError(f"Отсутствует обязательный ключ в config: '{key}'")

        # Проверяем типы и значения
        if self.target_model_config['vocab_size'] <= 0:
            raise ValueError("vocab_size должен быть положительным")

        if self.target_model_config['d_model'] <= 0:
            raise ValueError("d_model должен быть положительным")

        if self.target_model_config['n_layers'] <= 0:
            raise ValueError("n_layers должен быть положительным")

        if self.target_model_config['n_heads'] <= 0:
            raise ValueError("n_heads должен быть положительным")

        # Устанавливаем значения по умолчанию
        if 'd_ff' not in self.target_model_config:
            self.target_model_config['d_ff'] = 4 * self.target_model_config['d_model']

        if 'max_seq_len' not in self.target_model_config:
            self.target_model_config['max_seq_len'] = 2048

        if 'dropout' not in self.target_model_config:
            self.target_model_config['dropout'] = 0.1

    def validate_compatibility(self) -> Dict[str, Any]:
        """
        Проверяет совместимость GGUF модели с target config.

        Выполняет проверки:
            - Размерности модели (d_model, n_layers)
            - Vocab overlap между GGUF и BPE
            - Доступность памяти для transfer

        Returns:
            Словарь с результатами проверки:
                - compatible: bool - общая совместимость
                - warnings: List[str] - список предупреждений
                - vocab_overlap: float - процент overlap vocabulary (0.0-1.0)
                - transferable_layers: int - количество слоёв для переноса
                - memory_estimate: Dict - оценка памяти

        Example:
            >>> compat = pipeline.validate_compatibility()
            >>> if compat['compatible']:
            ...     print(f"✅ Совместимо! Vocab overlap: {compat['vocab_overlap']:.1%}")
            ... else:
            ...     print("⚠️ Предупреждения:", compat['warnings'])
        """
        print(f"\n{'='*60}")
        print("📊 Проверка совместимости GGUF модели с target config")
        print(f"{'='*60}\n")

        warnings = []
        compatible = True

        # 1. Проверка размерности d_model
        gguf_d_model = self.gguf_metadata.get('d_model', 0)
        target_d_model = self.target_model_config['d_model']

        print(f"d_model: GGUF={gguf_d_model}, Target={target_d_model}")

        if gguf_d_model != target_d_model:
            warnings.append(
                f"d_model не совпадает: GGUF={gguf_d_model}, Target={target_d_model}. "
                f"Потребуется interpolation или truncation."
            )
            print(f"⚠️ d_model mismatch: будет применена адаптация")
        else:
            print(f"✅ d_model совпадает")

        # 2. Проверка количества слоёв
        gguf_n_layers = self.gguf_metadata.get('n_layers', 0)
        target_n_layers = self.target_model_config['n_layers']

        print(f"\nn_layers: GGUF={gguf_n_layers}, Target={target_n_layers}")

        transferable_layers = min(gguf_n_layers, target_n_layers)

        if gguf_n_layers < target_n_layers:
            warnings.append(
                f"GGUF модель имеет меньше слоёв ({gguf_n_layers}) чем target ({target_n_layers}). "
                f"Будут перенесены только первые {transferable_layers} слоёв, остальные - random init."
            )
            print(f"⚠️ GGUF слоёв меньше: будет перенесено только {transferable_layers}")
        elif gguf_n_layers > target_n_layers:
            warnings.append(
                f"GGUF модель имеет больше слоёв ({gguf_n_layers}) чем target ({target_n_layers}). "
                f"Будут использованы только первые {transferable_layers} слоёв."
            )
            print(f"⚠️ GGUF слоёв больше: будут использованы первые {transferable_layers}")
        else:
            print(f"✅ n_layers совпадает")

        # 3. Vocab overlap
        vocab_overlap = self.tokenizer_aligner.estimate_vocab_overlap()

        print(f"\nVocab overlap: {vocab_overlap:.1%}")

        if vocab_overlap < 0.3:
            warnings.append(
                f"Низкий vocab overlap ({vocab_overlap:.1%}). "
                f"Большинство токенов будут random initialized."
            )
            print(f"⚠️ Низкий vocab overlap: качество может быть низким")
        elif vocab_overlap < 0.5:
            warnings.append(
                f"Умеренный vocab overlap ({vocab_overlap:.1%}). "
                f"Рекомендуется дополнительное обучение на вашем домене."
            )
            print(f"⚠️ Умеренный vocab overlap")
        else:
            print(f"✅ Хороший vocab overlap")

        # 4. Оценка памяти
        memory_estimate = self.memory_manager.estimate_model_memory(
            n_layers=target_n_layers,
            d_model=target_d_model,
            vocab_size=self.target_model_config['vocab_size'],
            d_ff=self.target_model_config.get('d_ff')
        )

        print(f"\n📊 Оценка памяти:")
        print(f"   Модель (inference): {memory_estimate['inference_gb']:.2f} GB")
        print(f"   Модель (training):  {memory_estimate['training_gb']:.2f} GB")
        print(f"   Доступно RAM:       {memory_estimate['available_gb']:.2f} GB")

        if not memory_estimate['can_fit_inference']:
            warnings.append(
                f"Недостаточно RAM для inference: требуется {memory_estimate['inference_gb']:.2f} GB, "
                f"доступно {memory_estimate['available_gb']:.2f} GB."
            )
            compatible = False
            print(f"❌ Недостаточно памяти для inference!")

        if not memory_estimate['can_fit_training']:
            warnings.append(
                f"Недостаточно RAM для training: требуется {memory_estimate['training_gb']:.2f} GB, "
                f"доступно {memory_estimate['available_gb']:.2f} GB. "
                f"Рекомендуется использовать gradient checkpointing или LoRA."
            )
            print(f"⚠️ Недостаточно памяти для training (но inference возможен)")

        # 5. Проверка архитектуры
        gguf_arch = self.gguf_metadata.get('architecture', 'unknown')
        print(f"\nАрхитектура GGUF: {gguf_arch}")

        if gguf_arch not in ['llama', 'mistral', 'phi', 'qwen']:
            warnings.append(
                f"Неизвестная архитектура: {gguf_arch}. "
                f"Transfer learning может работать некорректно. "
                f"Поддерживаются: llama, mistral, phi, qwen."
            )
            print(f"⚠️ Неизвестная архитектура, возможны проблемы")
        else:
            print(f"✅ Архитектура поддерживается")

        # Итоговый результат
        print(f"\n{'='*60}")
        if compatible:
            print("✅ Модель совместима! Transfer learning возможен.")
        else:
            print("❌ Модель несовместима! Требуется изменение конфигурации.")

        if warnings:
            print(f"\n⚠️ Предупреждения ({len(warnings)}):")
            for i, warning in enumerate(warnings, 1):
                print(f"   {i}. {warning}")

        print(f"{'='*60}\n")

        return {
            'compatible': compatible,
            'warnings': warnings,
            'vocab_overlap': vocab_overlap,
            'transferable_layers': transferable_layers,
            'memory_estimate': memory_estimate,
            'gguf_metadata': {
                'architecture': gguf_arch,
                'd_model': gguf_d_model,
                'n_layers': gguf_n_layers,
                'vocab_size': self.gguf_metadata.get('vocab_size', 0)
            }
        }

    def initialize_model_from_gguf(
        self,
        layers_to_transfer: Optional[List[int]] = None,
        freeze_transferred_layers: bool = True,
        align_embeddings: bool = True
    ) -> ExpertModel:
        """
        Инициализирует ExpertModel с весами из GGUF модели.

        Полный pipeline:
            1. Создаём ExpertModel с target_model_config
            2. Создаём weight mapping между GGUF и ExpertModel
            3. Align embeddings (если align_embeddings=True)
            4. Переносим веса layer-by-layer (memory efficient)
            5. Замораживаем перенесённые слои (если freeze_transferred_layers=True)

        Args:
            layers_to_transfer: Список индексов слоёв для переноса
                Если None, переносятся все доступные слои
            freeze_transferred_layers: Если True, замораживает перенесённые веса
            align_embeddings: Если True, адаптирует embeddings под BPE vocab

        Returns:
            ExpertModel с перенесёнными весами из GGUF

        Raises:
            RuntimeError: Если transfer weights не удался

        Example:
            >>> model = pipeline.initialize_model_from_gguf(
            ...     layers_to_transfer=[0, 1, 2, 3],
            ...     freeze_transferred_layers=True
            ... )
            >>> # Модель готова к fine-tuning!
        """
        print(f"\n{'='*60}")
        print("🔄 Инициализация ExpertModel из GGUF модели")
        print(f"{'='*60}\n")

        # 1. Создаём ExpertModel
        print("1️⃣ Создание ExpertModel с target конфигурацией...")

        model = ExpertModel(
            vocab_size=self.target_model_config['vocab_size'],
            d_model=self.target_model_config['d_model'],
            n_layers=self.target_model_config['n_layers'],
            n_heads=self.target_model_config['n_heads'],
            d_ff=self.target_model_config.get('d_ff', 4 * self.target_model_config['d_model']),
            max_seq_len=self.target_model_config.get('max_seq_len', 2048),
            dropout=self.target_model_config.get('dropout', 0.1)
        )

        print(f"✅ ExpertModel создана:")
        print(f"   Параметры: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        print(f"   Конфигурация: {self.target_model_config}")

        # 2. Определяем слои для переноса
        if layers_to_transfer is None:
            gguf_n_layers = self.gguf_metadata.get('n_layers', 0)
            target_n_layers = self.target_model_config['n_layers']
            max_layers = min(gguf_n_layers, target_n_layers)
            layers_to_transfer = list(range(max_layers))

        print(f"\n2️⃣ Слои для переноса: {layers_to_transfer}")

        # 3. Создаём WeightMapper
        print(f"\n3️⃣ Создание weight mapping GGUF → ExpertModel...")

        weight_mapper = WeightMapper(
            gguf_parser=self.gguf_parser,
            expert_model=model,
            memory_manager=self.memory_manager
        )

        mapping = weight_mapper.create_mapping()
        print(f"✅ Mapping создан: {len(mapping)} тензоров")

        # 4. Align embeddings (если нужно)
        if align_embeddings:
            print(f"\n4️⃣ Alignment embeddings для BPE vocabulary...")
            vocab_mapping = self.tokenizer_aligner.create_vocab_mapping()
            print(f"✅ Vocab mapping создан: {len(vocab_mapping)} совпадающих токенов")

        # 5. Перенос весов
        print(f"\n5️⃣ Перенос весов из GGUF в ExpertModel...")

        transfer_report = weight_mapper.transfer_weights(
            layers_to_transfer=layers_to_transfer,
            freeze_layers=freeze_transferred_layers,
            resize_embeddings=align_embeddings
        )

        print(f"\n✅ Transfer weights завершён!")
        print(f"   Перенесено: {len(transfer_report['transferred'])} тензоров")
        print(f"   Пропущено: {len(transfer_report['skipped'])} тензоров")
        print(f"   Resized: {len(transfer_report['resized'])} тензоров")

        if freeze_transferred_layers:
            print(f"   Заморожено: {len(transfer_report['frozen'])} параметров")

        # 6. Финальная статистика
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        print(f"\n📊 Финальная статистика модели:")
        print(f"   Всего параметров: {total_params / 1e6:.2f}M")
        print(f"   Trainable: {trainable_params / 1e6:.2f}M ({100 * trainable_params / total_params:.1f}%)")
        print(f"   Frozen: {(total_params - trainable_params) / 1e6:.2f}M ({100 * (total_params - trainable_params) / total_params:.1f}%)")

        print(f"\n{'='*60}")
        print("🎉 ExpertModel успешно инициализирована из GGUF!")
        print("   Модель готова к fine-tuning на ваших данных.")
        print(f"{'='*60}\n")

        return model

    def get_memory_stats(self) -> Dict[str, float]:
        """
        Возвращает текущую статистику по памяти.

        Returns:
            Словарь с информацией о памяти (total_gb, available_gb, used_gb, etc.)

        Example:
            >>> stats = pipeline.get_memory_stats()
            >>> print(f"Доступно RAM: {stats['available_gb']:.2f} GB")
        """
        return self.memory_manager.get_memory_stats()

    def print_memory_report(self):
        """
        Выводит детальный отчёт о состоянии памяти.

        Example:
            >>> pipeline.print_memory_report()
        """
        self.memory_manager.print_memory_report()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TransferLearningPipeline(\n"
            f"  gguf_path='{self.gguf_path.name}',\n"
            f"  target_config={self.target_model_config},\n"
            f"  max_ram_gb={self.max_ram_gb}\n"
            f")"
        )
