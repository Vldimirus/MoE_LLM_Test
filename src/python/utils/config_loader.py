"""
Утилита для загрузки YAML конфигураций.
"""

from pathlib import Path
from typing import Dict, Any
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Загружает YAML конфигурацию из файла.

    Args:
        config_path: Путь к YAML файлу

    Returns:
        Словарь с конфигурацией

    Raises:
        FileNotFoundError: Если файл не найден
        yaml.YAMLError: Если ошибка парсинга YAML

    Example:
        >>> config = load_config('configs/transfer_learning_config.yaml')
        >>> print(config['target_model']['d_model'])
        512
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Ошибка парсинга YAML файла {config_path}: {str(e)}")
