#!/bin/bash

# Скрипт установки зависимостей для проекта Domain-Specific MoE System

echo "========================================="
echo "Установка Domain-Specific MoE System"
echo "========================================="

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 не найден. Установите Python 3.10+"
    exit 1
fi

echo "✅ Python $(python3 --version) найден"

# Создание виртуального окружения
echo ""
echo "Создание виртуального окружения..."
python3 -m venv venv

# Активация окружения
echo "Активация окружения..."
source venv/bin/activate

# Установка зависимостей
echo ""
echo "Установка зависимостей..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "========================================="
echo "✅ Установка завершена!"
echo "========================================="
echo ""
echo "Для активации окружения выполните:"
echo "  source venv/bin/activate"
echo ""
echo "Для запуска тестов:"
echo "  python src/python/memory/three_level_memory.py"
echo "  python src/python/models/transformer.py  # Требует PyTorch"
echo ""
