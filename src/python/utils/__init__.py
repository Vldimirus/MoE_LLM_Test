"""
Утилиты для работы с моделями и данными.

Компоненты:
    - config_loader: загрузка YAML конфигураций
    - tensor_utils: утилиты для работы с тензорами
"""

from .config_loader import load_config

__all__ = ['load_config']
