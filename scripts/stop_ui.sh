#!/bin/bash
#
# Скрипт остановки Gradio Web UI Dashboard
#
# Usage:
#   ./scripts/stop_ui.sh
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
PID_FILE="$PROJECT_ROOT/.ui_dashboard.pid"

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  MoE System Dashboard - Stop Script${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# Проверяем наличие PID файла
if [ ! -f "$PID_FILE" ]; then
    echo -e "${YELLOW}⚠${NC}  UI процесс не запущен (PID файл не найден)"
    exit 0
fi

# Читаем PID
PID=$(cat "$PID_FILE")

# Проверяем что процесс действительно запущен
if ! ps -p "$PID" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠${NC}  Процесс с PID $PID не найден"
    echo -e "${YELLOW}→${NC} Удаляем устаревший PID файл..."
    rm -f "$PID_FILE"
    exit 0
fi

echo -e "${BLUE}→${NC} Обнаружен процесс UI (PID: $PID)"

# Пытаемся корректно завершить (SIGTERM)
echo -e "${YELLOW}→${NC} Отправка сигнала SIGTERM..."
kill -TERM "$PID" 2>/dev/null || true

# Ждём до 5 секунд
for i in {1..10}; do
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Процесс успешно остановлен"
        rm -f "$PID_FILE"
        echo ""
        exit 0
    fi
    sleep 0.5
    echo -n "."
done

echo ""
echo -e "${YELLOW}⚠${NC}  Процесс не остановился, принудительное завершение..."

# Принудительно убиваем (SIGKILL)
kill -KILL "$PID" 2>/dev/null || true
sleep 1

# Проверяем результат
if ! ps -p "$PID" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Процесс принудительно завершён"
    rm -f "$PID_FILE"
    echo ""
    exit 0
else
    echo -e "${RED}✗${NC} Не удалось остановить процесс!"
    echo -e "${YELLOW}→${NC} Попробуйте вручную: kill -9 $PID"
    echo ""
    exit 1
fi
