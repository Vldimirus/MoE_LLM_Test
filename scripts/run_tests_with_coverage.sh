#!/bin/bash
#
# Скрипт запуска тестов с coverage отчётами
#
# Usage:
#   ./scripts/run_tests_with_coverage.sh [options]
#
# Options:
#   --html      Генерировать HTML отчёт
#   --xml       Генерировать XML отчёт (для CI/CD)
#   --json      Генерировать JSON отчёт
#   --open      Открыть HTML отчёт в браузере
#   --report    Показать подробный отчёт в консоли
#   --min N     Минимальный процент coverage (default: 80)
#

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Конфигурация
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$PROJECT_ROOT/venv"
HTML_REPORT=false
XML_REPORT=false
JSON_REPORT=false
OPEN_BROWSER=false
SHOW_REPORT=true
MIN_COVERAGE=80

# Парсинг аргументов
while [[ $# -gt 0 ]]; do
    case $1 in
        --html)
            HTML_REPORT=true
            shift
            ;;
        --xml)
            XML_REPORT=true
            shift
            ;;
        --json)
            JSON_REPORT=true
            shift
            ;;
        --open)
            OPEN_BROWSER=true
            HTML_REPORT=true  # Автоматически включаем HTML
            shift
            ;;
        --report)
            SHOW_REPORT=true
            shift
            ;;
        --min)
            MIN_COVERAGE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  Running Tests with Coverage${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# Активируем виртуальное окружение
if [ -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}→${NC} Активация виртуального окружения..."
    source "$VENV_PATH/bin/activate"
else
    echo -e "${RED}✗${NC} Виртуальное окружение не найдено: $VENV_PATH"
    echo -e "${YELLOW}→${NC} Запустите: ./setup.sh"
    exit 1
fi

# Проверяем наличие pytest и coverage
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}✗${NC} pytest не установлен"
    echo -e "${YELLOW}→${NC} Установите: pip install pytest pytest-cov"
    exit 1
fi

# Очистка старых отчётов
echo -e "${YELLOW}→${NC} Очистка старых coverage данных..."
rm -f .coverage coverage.xml coverage.json
rm -rf htmlcov/

# Запуск тестов с coverage
echo -e "${YELLOW}→${NC} Запуск pytest с coverage..."
echo ""

pytest tests/ \
    --cov=src/python \
    --cov-report=term-missing \
    --cov-fail-under=$MIN_COVERAGE \
    -v

PYTEST_EXIT_CODE=$?

echo ""
echo -e "${BLUE}=========================================${NC}"

# Генерация дополнительных отчётов
if [ "$HTML_REPORT" = true ]; then
    echo -e "${YELLOW}→${NC} Генерация HTML отчёта..."
    coverage html
    echo -e "${GREEN}✓${NC} HTML отчёт создан: htmlcov/index.html"
fi

if [ "$XML_REPORT" = true ]; then
    echo -e "${YELLOW}→${NC} Генерация XML отчёта..."
    coverage xml
    echo -e "${GREEN}✓${NC} XML отчёт создан: coverage.xml"
fi

if [ "$JSON_REPORT" = true ]; then
    echo -e "${YELLOW}→${NC} Генерация JSON отчёта..."
    coverage json
    echo -e "${GREEN}✓${NC} JSON отчёт создан: coverage.json"
fi

# Показать детальный отчёт
if [ "$SHOW_REPORT" = true ]; then
    echo ""
    echo -e "${BLUE}📊 Coverage Summary:${NC}"
    coverage report --sort=Cover
fi

# Открыть HTML отчёт в браузере
if [ "$OPEN_BROWSER" = true ]; then
    echo ""
    echo -e "${YELLOW}→${NC} Открытие HTML отчёта в браузере..."
    
    if command -v xdg-open &> /dev/null; then
        xdg-open htmlcov/index.html &
    elif command -v open &> /dev/null; then
        open htmlcov/index.html &
    else
        echo -e "${YELLOW}⚠${NC}  Не удалось открыть браузер автоматически"
        echo -e "${YELLOW}→${NC} Откройте вручную: htmlcov/index.html"
    fi
fi

echo ""
echo -e "${BLUE}=========================================${NC}"

# Финальный статус
if [ $PYTEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Все тесты пройдены успешно!${NC}"
    echo -e "${GREEN}✓ Coverage >= ${MIN_COVERAGE}%${NC}"
else
    echo -e "${RED}✗ Некоторые тесты провалились или coverage < ${MIN_COVERAGE}%${NC}"
fi

echo -e "${BLUE}=========================================${NC}"
echo ""

exit $PYTEST_EXIT_CODE
