"""
Memory Manager для управления RAM при загрузке больших GGUF моделей.

Основные возможности:
    - Мониторинг доступной RAM через psutil
    - Проверка возможности загрузки тензора
    - Безопасная загрузка с контролем памяти
    - Предотвращение OOM (Out of Memory) ошибок
"""

from typing import Optional, Dict
import psutil
import torch


class MemoryManager:
    """
    Управляет памятью при загрузке GGUF моделей.

    Отслеживает доступную RAM и предотвращает OOM ошибки
    при загрузке больших тензоров.

    Пример:
        >>> mem_mgr = MemoryManager(max_ram_gb=12.0)
        >>> print(f"Доступно RAM: {mem_mgr.get_available_ram():.1f} GB")
        >>>
        >>> if mem_mgr.can_load_tensor_size(1024 * 1024 * 1024):  # 1GB
        >>>     # Безопасно загружать
        >>>     tensor = load_large_tensor()
    """

    def __init__(self, max_ram_gb: float = 12.0, safety_margin_gb: float = 2.0):
        """
        Инициализирует Memory Manager.

        Args:
            max_ram_gb: Максимальное количество RAM для использования (GB)
            safety_margin_gb: Резерв RAM, который всегда оставляем свободным (GB)

        Example:
            >>> # Ограничиваем использование 12GB, оставляем 2GB для системы
            >>> mem_mgr = MemoryManager(max_ram_gb=12.0, safety_margin_gb=2.0)
        """
        self.max_ram_gb = max_ram_gb
        self.safety_margin_gb = safety_margin_gb

        # Проверяем что psutil доступен
        self.psutil_available = True
        try:
            psutil.virtual_memory()
        except Exception:
            self.psutil_available = False
            print("⚠️ psutil недоступен, мониторинг памяти отключён")

        print(f"✅ MemoryManager инициализирован:")
        print(f"   Макс. RAM: {max_ram_gb:.1f} GB")
        print(f"   Резерв: {safety_margin_gb:.1f} GB")
        print(f"   Доступно сейчас: {self.get_available_ram():.1f} GB")

    def get_available_ram(self) -> float:
        """
        Возвращает доступную RAM в GB.

        Returns:
            Количество доступной RAM в GB

        Example:
            >>> available = mem_mgr.get_available_ram()
            >>> print(f"Доступно: {available:.2f} GB")
        """
        if not self.psutil_available:
            # Если psutil недоступен, возвращаем оптимистичную оценку
            return self.max_ram_gb

        try:
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024 ** 3)

            # Учитываем safety margin
            usable_gb = min(available_gb - self.safety_margin_gb, self.max_ram_gb)

            return max(0.0, usable_gb)
        except Exception as e:
            print(f"⚠️ Ошибка при проверке RAM: {e}")
            return self.max_ram_gb / 2  # Консервативная оценка

    def get_memory_stats(self) -> Dict[str, float]:
        """
        Возвращает детальную статистику по памяти.

        Returns:
            Словарь со статистикой:
                - total_gb: Всего RAM в системе
                - available_gb: Доступно RAM
                - used_gb: Используется RAM
                - percent: Процент использования
                - usable_gb: Доступно с учётом лимитов

        Example:
            >>> stats = mem_mgr.get_memory_stats()
            >>> print(f"Использовано: {stats['percent']:.1f}%")
        """
        if not self.psutil_available:
            return {
                'total_gb': self.max_ram_gb,
                'available_gb': self.max_ram_gb,
                'used_gb': 0.0,
                'percent': 0.0,
                'usable_gb': self.max_ram_gb
            }

        try:
            mem = psutil.virtual_memory()
            return {
                'total_gb': mem.total / (1024 ** 3),
                'available_gb': mem.available / (1024 ** 3),
                'used_gb': mem.used / (1024 ** 3),
                'percent': mem.percent,
                'usable_gb': self.get_available_ram()
            }
        except Exception as e:
            print(f"⚠️ Ошибка при получении статистики: {e}")
            return {'total_gb': 0, 'available_gb': 0, 'used_gb': 0, 'percent': 0, 'usable_gb': 0}

    def can_load_tensor_size(self, size_bytes: int) -> bool:
        """
        Проверяет, можно ли безопасно загрузить тензор заданного размера.

        Args:
            size_bytes: Размер тензора в байтах

        Returns:
            True если можно безопасно загрузить, False иначе

        Example:
            >>> tensor_size = 4 * 1024 * 1024 * 1024  # 4GB
            >>> if mem_mgr.can_load_tensor_size(tensor_size):
            >>>     print("Можно загружать")
            >>> else:
            >>>     print("Недостаточно памяти")
        """
        size_gb = size_bytes / (1024 ** 3)
        available = self.get_available_ram()

        can_load = size_gb <= available

        if not can_load:
            print(f"⚠️ Недостаточно памяти:")
            print(f"   Требуется: {size_gb:.2f} GB")
            print(f"   Доступно: {available:.2f} GB")

        return can_load

    def can_load_tensor(self, tensor_shape: tuple, dtype: torch.dtype = torch.float32) -> bool:
        """
        Проверяет, можно ли безопасно загрузить PyTorch тензор.

        Args:
            tensor_shape: Размерности тензора (tuple)
            dtype: Тип данных PyTorch (torch.float32, torch.float16, etc.)

        Returns:
            True если можно безопасно загрузить, False иначе

        Example:
            >>> # Проверяем embedding матрицу [32000, 4096] в FP32
            >>> if mem_mgr.can_load_tensor((32000, 4096), torch.float32):
            >>>     embeddings = torch.randn(32000, 4096)
        """
        # Вычисляем размер тензора в байтах
        n_elements = 1
        for dim in tensor_shape:
            n_elements *= dim

        # Размер одного элемента в байтах
        if dtype == torch.float32:
            bytes_per_element = 4
        elif dtype == torch.float16 or dtype == torch.bfloat16:
            bytes_per_element = 2
        elif dtype == torch.int8:
            bytes_per_element = 1
        elif dtype == torch.int64 or dtype == torch.float64:
            bytes_per_element = 8
        else:
            bytes_per_element = 4  # Default

        total_bytes = n_elements * bytes_per_element

        return self.can_load_tensor_size(total_bytes)

    def estimate_model_memory(
        self,
        n_layers: int,
        d_model: int,
        vocab_size: int,
        d_ff: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Оценивает требования к памяти для модели.

        Args:
            n_layers: Количество transformer слоёв
            d_model: Размерность embeddings
            vocab_size: Размер vocabulary
            d_ff: Размерность feed-forward (по умолчанию 4 * d_model)

        Returns:
            Словарь с оценками:
                - embeddings_gb: Память для embeddings
                - transformer_gb: Память для transformer слоёв
                - total_gb: Общая память модели
                - can_fit: Помещается ли в доступную RAM

        Example:
            >>> # Оценка для модели 7.26M параметров
            >>> estimate = mem_mgr.estimate_model_memory(
            ...     n_layers=4, d_model=256, vocab_size=8000
            ... )
            >>> print(f"Требуется: {estimate['total_gb']:.2f} GB")
        """
        if d_ff is None:
            d_ff = 4 * d_model

        # Embeddings: vocab_size * d_model * 4 bytes (FP32)
        embeddings_params = vocab_size * d_model
        embeddings_gb = (embeddings_params * 4) / (1024 ** 3)

        # Transformer block:
        # - Attention: Q, K, V, O = 4 * (d_model * d_model)
        # - FFN: 2 * (d_model * d_ff)
        # - LayerNorm: 2 * d_model (negligible)
        params_per_layer = (
            4 * d_model * d_model +  # Attention
            2 * d_model * d_ff        # FFN
        )
        transformer_params = n_layers * params_per_layer
        transformer_gb = (transformer_params * 4) / (1024 ** 3)

        # Output projection (LM head): d_model * vocab_size
        lm_head_params = d_model * vocab_size
        lm_head_gb = (lm_head_params * 4) / (1024 ** 3)

        total_gb = embeddings_gb + transformer_gb + lm_head_gb

        # Добавляем overhead для градиентов и активаций (~2x во время training)
        training_gb = total_gb * 2.5

        available = self.get_available_ram()

        return {
            'embeddings_gb': embeddings_gb,
            'transformer_gb': transformer_gb,
            'lm_head_gb': lm_head_gb,
            'total_params_m': (embeddings_params + transformer_params + lm_head_params) / 1_000_000,
            'inference_gb': total_gb,
            'training_gb': training_gb,
            'available_gb': available,
            'can_fit_inference': total_gb <= available,
            'can_fit_training': training_gb <= available
        }

    def print_memory_report(self):
        """
        Выводит детальный отчёт о состоянии памяти.

        Example:
            >>> mem_mgr.print_memory_report()
        """
        stats = self.get_memory_stats()

        print("\n" + "="*60)
        print("📊 Memory Manager - Отчёт о памяти")
        print("="*60)
        print(f"Всего RAM:        {stats['total_gb']:.2f} GB")
        print(f"Используется:     {stats['used_gb']:.2f} GB ({stats['percent']:.1f}%)")
        print(f"Доступно:         {stats['available_gb']:.2f} GB")
        print(f"Резерв:           {self.safety_margin_gb:.2f} GB")
        print(f"Можно использ.:   {stats['usable_gb']:.2f} GB")
        print("="*60 + "\n")
