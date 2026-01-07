#!/bin/bash
#
# Скрипт проверки статуса Gradio Web UI Dashboard
#
# Usage:
#   ./scripts/status_ui.sh
#

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Конфигурация
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/.ui_dashboard.pid"
LOG_FILE="$PROJECT_ROOT/ui_dashboard.log"

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  MoE System Dashboard - Status${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# Проверяем наличие PID файла
if [ ! -f "$PID_FILE" ]; then
    echo -e "${RED}✗ Статус:${NC} Не запущен"
    echo ""
    echo -e "${YELLOW}💡 Для запуска:${NC}"
    echo -e "   ./scripts/restart_ui.sh"
    echo ""
    exit 0
fi

# Читаем PID
PID=$(cat "$PID_FILE")

# Проверяем что процесс действительно запущен
if ! ps -p "$PID" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠ Статус:${NC} PID файл существует, но процесс не запущен"
    echo -e "${YELLOW}→${NC} Возможно процесс был некорректно завершён"
    echo ""
    echo -e "${YELLOW}💡 Для очистки и перезапуска:${NC}"
    echo -e "   ./scripts/restart_ui.sh"
    echo ""
    exit 0
fi

# Процесс запущен - собираем информацию
echo -e "${GREEN}✓ Статус:${NC} Запущен"
echo ""

# PID
echo -e "${BLUE}📝 PID:${NC} $PID"

# Время работы
UPTIME=$(ps -o etime= -p "$PID" | tr -d ' ')
echo -e "${BLUE}⏰ Uptime:${NC} $UPTIME"

# Использование памяти
MEM_KB=$(ps -o rss= -p "$PID" | tr -d ' ')
MEM_MB=$((MEM_KB / 1024))
echo -e "${BLUE}💾 Memory:${NC} ${MEM_MB} MB"

# CPU usage
CPU=$(ps -o %cpu= -p "$PID" | tr -d ' ')
echo -e "${BLUE}⚙️  CPU:${NC} ${CPU}%"

# Порт (попытка найти)
PORT=$(lsof -Pan -p $PID -i 2>/dev/null | grep LISTEN | awk '{print $9}' | cut -d':' -f2 | head -n1)
if [ ! -z "$PORT" ]; then
    echo -e "${BLUE}🌐 Port:${NC} $PORT"
    echo -e "${BLUE}🔗 URL:${NC} http://localhost:$PORT"
fi

# Log file
if [ -f "$LOG_FILE" ]; then
    LOG_SIZE=$(du -h "$LOG_FILE" | cut -f1)
    echo -e "${BLUE}📂 Log file:${NC} $LOG_FILE (${LOG_SIZE})"

    # Последние строки лога
    echo ""
    echo -e "${BLUE}📋 Последние строки лога:${NC}"
    echo -e "${YELLOW}─────────────────────────────────────────${NC}"
    tail -n 5 "$LOG_FILE" 2>/dev/null || echo "Не удалось прочитать лог"
    echo -e "${YELLOW}─────────────────────────────────────────${NC}"
fi

echo ""
echo -e "${YELLOW}💡 Команды управления:${NC}"
echo -e "   ./scripts/restart_ui.sh  - Перезапустить"
echo -e "   ./scripts/stop_ui.sh     - Остановить"
echo -e "   tail -f $LOG_FILE  - Просмотр логов"
echo ""
