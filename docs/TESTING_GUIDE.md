# Testing and Coverage Guide

Руководство по запуску тестов и работе с coverage отчётами для Domain-Specific MoE System.

## 📋 Содержание

- [Быстрый старт](#быстрый-старт)
- [Структура тестов](#структура-тестов)
- [Запуск тестов](#запуск-тестов)
- [Coverage отчёты](#coverage-отчёты)
- [Написание тестов](#написание-тестов)
- [CI/CD Integration](#cicd-integration)

## 🚀 Быстрый старт

### Установка зависимостей

```bash
# Активировать виртуальное окружение
source venv/bin/activate

# Установить зависимости для тестирования
pip install pytest pytest-cov
```

### Запуск всех тестов

```bash
# Простой запуск
pytest

# С подробным выводом
pytest -v

# С coverage
pytest --cov=src/python --cov-report=term-missing
```

### Быстрый coverage отчёт с HTML

```bash
# Запустить тесты с HTML отчётом и открыть в браузере
./scripts/run_tests_with_coverage.sh --html --open
```

## 📁 Структура тестов

```
tests/
├── __init__.py
├── conftest.py                 # Фикстуры и конфигурация pytest
├── test_transformer.py         # Тесты базовой архитектуры (77 тестов)
├── test_expert.py              # Тесты ExpertModel (46 тестов)
├── test_router.py              # Тесты Router системы (55 тестов)
├── test_memory.py              # Тесты ThreeLevelMemory (11 тестов)
├── test_dataset.py             # Тесты Dataset (10 тестов)
└── test_trainer.py             # Тесты Trainer (12 тестов)
```

**Всего: 211 unit тестов**

## 🧪 Запуск тестов

### Основные команды

```bash
# Все тесты
pytest

# Конкретный файл
pytest tests/test_transformer.py

# Конкретный класс
pytest tests/test_transformer.py::TestMultiHeadAttention

# Конкретный тест
pytest tests/test_transformer.py::TestMultiHeadAttention::test_forward_shape

# С выводом print statements
pytest -s

# Остановиться на первой ошибке
pytest -x

# Запустить последние провалившиеся тесты
pytest --lf

# Показать топ-10 медленных тестов
pytest --durations=10
```

### Параллельный запуск (pytest-xdist)

```bash
# Установить
pip install pytest-xdist

# Запустить в 4 процессах
pytest -n 4
```

### Фильтрация тестов

```bash
# По маркерам
pytest -m slow          # Только медленные тесты
pytest -m "not slow"    # Исключить медленные

# По именам
pytest -k "transformer"      # Только тесты с "transformer" в имени
pytest -k "not integration"  # Исключить integration тесты
```

## 📊 Coverage отчёты

### Конфигурация Coverage

Конфигурация находится в `.coveragerc`:

```ini
[run]
source = src/python
branch = True

[report]
fail_under = 80
show_missing = True
```

### Использование скрипта run_tests_with_coverage.sh

#### Базовое использование

```bash
# Запуск с дефолтными настройками (console report)
./scripts/run_tests_with_coverage.sh

# Генерация HTML отчёта
./scripts/run_tests_with_coverage.sh --html

# Открыть HTML отчёт в браузере
./scripts/run_tests_with_coverage.sh --html --open
```

#### Расширенные опции

```bash
# Генерация XML для CI/CD
./scripts/run_tests_with_coverage.sh --xml

# Генерация JSON отчёта
./scripts/run_tests_with_coverage.sh --json

# Установить минимальный процент coverage
./scripts/run_tests_with_coverage.sh --min 85

# Комбинация опций
./scripts/run_tests_with_coverage.sh --html --xml --json --open
```

#### Параметры скрипта

| Опция | Описание |
|-------|----------|
| `--html` | Генерировать HTML отчёт в `htmlcov/` |
| `--xml` | Генерировать XML отчёт в `coverage.xml` |
| `--json` | Генерировать JSON отчёт в `coverage.json` |
| `--open` | Открыть HTML отчёт в браузере (автоматически включает --html) |
| `--report` | Показать подробный console отчёт |
| `--min N` | Минимальный процент coverage (default: 80) |

### Ручные команды coverage

```bash
# Запустить тесты с coverage
pytest --cov=src/python

# С детализацией пропущенных строк
pytest --cov=src/python --cov-report=term-missing

# Генерация HTML отчёта
pytest --cov=src/python --cov-report=html

# Генерация XML (для CI/CD)
pytest --cov=src/python --cov-report=xml

# Несколько форматов одновременно
pytest --cov=src/python --cov-report=html --cov-report=xml --cov-report=term
```

### Coverage по конкретным модулям

```bash
# Только transformer модуль
pytest tests/test_transformer.py --cov=src/python/models/transformer

# Только routing модуль
pytest tests/test_router.py --cov=src/python/routing

# Несколько модулей
pytest tests/ --cov=src/python/models --cov=src/python/routing
```

### Просмотр HTML отчётов

```bash
# Генерация
pytest --cov=src/python --cov-report=html

# Открыть в браузере
xdg-open htmlcov/index.html  # Linux
open htmlcov/index.html      # macOS

# Или использовать скрипт
./scripts/run_tests_with_coverage.sh --html --open
```

HTML отчёт показывает:
- ✅ Общий процент покрытия
- ✅ Coverage по каждому файлу
- ✅ Подсветка покрытых/непокрытых строк
- ✅ Branch coverage (if/else ветки)
- ✅ Детализация по функциям и классам

## 📈 Текущее состояние Coverage

### Общая статистика

```
Total Coverage: 41.30%
Total Tests: 211
All Tests: PASSED ✓
```

### Coverage по модулям

| Модуль | Coverage | Статус |
|--------|----------|--------|
| `training/dataset.py` | 96.63% | 🟢 Отлично |
| `training/trainer.py` | 89.66% | 🟢 Отлично |
| `memory/three_level_memory.py` | 79.17% | 🟡 Хорошо |
| `models/transformer.py` | 74.71% | 🟡 Хорошо |
| `routing/router.py` | 67.33% | 🟡 Приемлемо |
| `models/expert.py` | 57.50% | 🟠 Требует улучшения |
| **UI компоненты** | 0.00% | ⚪ Не тестируются unit-тестами |

### Почему UI компоненты имеют 0% coverage?

UI компоненты (`src/python/ui/`) **намеренно исключены из unit-тестов**, так как:

1. **Gradio специфика**: Требуют запущенного web-сервера
2. **Integration тесты**: Нужны E2E тесты (Selenium, Playwright)
3. **Визуальное тестирование**: Проверка через UI Dashboard
4. **Приоритет**: Core логика (models, training, routing) более критична

## ✍️ Написание тестов

### Структура теста

```python
import pytest
import torch
from src.python.models.transformer import MultiHeadAttention

class TestMultiHeadAttention:
    """Тесты для MultiHeadAttention модуля."""
    
    def test_forward_shape(self):
        """Проверка размерности выходного тензора."""
        # Arrange
        d_model, n_heads = 512, 8
        batch_size, seq_len = 4, 10
        
        attention = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Act
        output = attention(x, x, x)
        
        # Assert
        assert output.shape == (batch_size, seq_len, d_model)
```

### Использование фикстур

Фикстуры определены в `tests/conftest.py`:

```python
def test_with_fixture(simple_model):
    """Использование фикстуры из conftest.py"""
    output = simple_model(torch.randn(2, 10, 512))
    assert output.shape == (2, 10, 512)
```

### Параметризация тестов

```python
@pytest.mark.parametrize("d_model,n_heads", [
    (512, 8),
    (768, 12),
    (1024, 16),
])
def test_different_configs(d_model, n_heads):
    attention = MultiHeadAttention(d_model, n_heads)
    assert attention.d_model == d_model
    assert attention.n_heads == n_heads
```

### Проверка исключений

```python
def test_invalid_input():
    """Проверка обработки некорректного ввода."""
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=512, n_heads=7)  # 512 не делится на 7
```

### Маркировка тестов

```python
@pytest.mark.slow
def test_large_model_training():
    """Медленный тест для больших моделей."""
    # ...

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_acceleration():
    """Тест требует GPU."""
    # ...
```

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests and Coverage

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests with coverage
      run: |
        pytest --cov=src/python --cov-report=xml --cov-report=term
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

### Pre-commit Hook

Создайте `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Запустить тесты перед коммитом

echo "Running tests..."
pytest tests/ -q

if [ $? -ne 0 ]; then
    echo "❌ Tests failed! Commit aborted."
    exit 1
fi

echo "✓ All tests passed!"
exit 0
```

```bash
chmod +x .git/hooks/pre-commit
```

## 📚 Полезные ссылки

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Pytest-cov Plugin](https://pytest-cov.readthedocs.io/)

## 💡 Советы и Best Practices

### 1. Запускайте тесты часто

```bash
# После изменения кода
pytest tests/test_transformer.py -v

# Перед коммитом
pytest
```

### 2. Используйте coverage для поиска пробелов

```bash
# Найти непокрытые строки
./scripts/run_tests_with_coverage.sh --html --open
```

### 3. Пишите тесты для новых функций

Для каждой новой функции/класса:
- ✅ Тест основного функционала
- ✅ Тест edge cases
- ✅ Тест обработки ошибок

### 4. Поддерживайте высокий coverage

Цель для core модулей: **>80%**

```bash
# Проверить минимальный coverage
./scripts/run_tests_with_coverage.sh --min 80
```

### 5. Используйте pytest fixtures

Избегайте дублирования кода в тестах - выносите общую логику в фикстуры.

## 🐛 Troubleshooting

### Тесты не находятся

```bash
# Убедитесь что находитесь в корне проекта
cd /path/to/NM_LLM_Test_2

# Убедитесь что pytest установлен
pip install pytest
```

### Coverage не работает

```bash
# Установите pytest-cov
pip install pytest-cov

# Проверьте .coveragerc
cat .coveragerc
```

### Медленные тесты

```bash
# Найти медленные тесты
pytest --durations=10

# Запустить параллельно
pip install pytest-xdist
pytest -n 4
```

### HTML отчёт не открывается

```bash
# Проверьте что отчёт создан
ls -la htmlcov/

# Откройте вручную
xdg-open htmlcov/index.html  # Linux
open htmlcov/index.html      # macOS
```

---

**Версия:** 1.0  
**Дата:** 2026-01-07  
**Автор:** Domain-Specific MoE System Team
