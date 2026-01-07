"""
Модуль с архитектурами нейросетей.

Содержит:
- TransformerBlock: базовый блок трансформера
- MultiHeadAttention: механизм внимания с несколькими головами
- FeedForward: position-wise feed-forward network
- PositionalEncoding: позиционное кодирование
- ExpertModel: полная модель эксперта
- RouterNetwork: сеть маршрутизации (в разработке)
"""

from typing import Optional
import sys
import os

# Добавляем путь для импортов
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../..')))

# Импорты компонентов
try:
    from python.models.transformer import (
        TransformerBlock,
        MultiHeadAttention,
        FeedForward
    )
    from python.models.expert import (
        ExpertModel,
        PositionalEncoding
    )

    __all__ = [
        'TransformerBlock',
        'MultiHeadAttention',
        'FeedForward',
        'PositionalEncoding',
        'ExpertModel',
    ]
except ImportError as e:
    print(f"Warning: Could not import models: {e}")
    __all__ = []
