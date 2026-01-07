"""
Модуль маршрутизации запросов к экспертам.

Содержит:
- SimpleRouter: rule-based роутер на основе ключевых слов
- ExpertInfo: информация об эксперте
- RoutingResult: результат маршрутизации
"""

from typing import Optional
import sys
import os

# Добавляем путь для импортов
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../..')))

# Импорты компонентов
try:
    from python.routing.router import (
        SimpleRouter,
        ExpertInfo,
        RoutingResult
    )

    __all__ = [
        'SimpleRouter',
        'ExpertInfo',
        'RoutingResult',
    ]
except ImportError as e:
    print(f"Warning: Could not import routing components: {e}")
    __all__ = []
