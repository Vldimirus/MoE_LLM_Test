#!/bin/bash
#
# Скрипт автоматического перезапуска Gradio Web UI Dashboard
#
# Функциональность:
#   - Проверяет, запущен ли UI процесс
#   - Если запущен - корректно завершает его
#   - Запускает новый экземпляр UI
#   - Сохраняет PID для последующих перезапусков
#
# Usage:
#   ./scripts/restart_ui.sh [--port 7860] [--share]
#

set -e  # Прекратить выполнение при ошибке

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
PORT=${1:-7860}  # Порт по умолчанию 7860
SHARE_FLAG=""

# Обработка аргументов
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --share)
            SHARE_FLAG="--share"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}  MoE System Dashboard - Restart Script${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""

# Функция для проверки запущенного процесса
check_running() {
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")

        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            echo -e "${YELLOW}✓${NC} Обнаружен запущенный процесс UI (PID: $OLD_PID)"
            return 0
        else
            echo -e "${YELLOW}⚠${NC}  PID файл существует, но процесс не запущен"
            rm -f "$PID_FILE"
            return 1
        fi
    else
        echo -e "${GREEN}✓${NC} UI процесс не запущен"
        return 1
    fi
}

# Функция для остановки процесса
stop_ui() {
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")

        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            echo -e "${YELLOW}→${NC} Останавливаем старый процесс (PID: $OLD_PID)..."

            # Пытаемся корректно завершить (SIGTERM)
            kill -TERM "$OLD_PID" 2>/dev/null || true

            # Ждём до 5 секунд
            for i in {1..10}; do
                if ! ps -p "$OLD_PID" > /dev/null 2>&1; then
                    echo -e "${GREEN}✓${NC} Процесс успешно остановлен"
                    rm -f "$PID_FILE"
                    return 0
                fi
                sleep 0.5
            done

            # Если не остановился - принудительно убиваем (SIGKILL)
            echo -e "${YELLOW}⚠${NC}  Процесс не остановился, принудительное завершение..."
            kill -KILL "$OLD_PID" 2>/dev/null || true
            sleep 1

            if ! ps -p "$OLD_PID" > /dev/null 2>&1; then
                echo -e "${GREEN}✓${NC} Процесс принудительно завершён"
                rm -f "$PID_FILE"
                return 0
            else
                echo -e "${RED}✗${NC} Не удалось остановить процесс!"
                return 1
            fi
        else
            rm -f "$PID_FILE"
        fi
    fi

    return 0
}

# Функция для проверки порта
check_port() {
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${RED}✗${NC} Порт $PORT уже занят другим процессом!"
        echo -e "${YELLOW}→${NC} Попытка найти процесс на порту $PORT..."

        PROCESS_ON_PORT=$(lsof -Pi :$PORT -sTCP:LISTEN -t)
        if [ ! -z "$PROCESS_ON_PORT" ]; then
            echo -e "${YELLOW}→${NC} Найден процесс PID: $PROCESS_ON_PORT"

            # Проверяем, это наш процесс?
            if [ -f "$PID_FILE" ] && [ "$(cat $PID_FILE)" == "$PROCESS_ON_PORT" ]; then
                echo -e "${YELLOW}→${NC} Это наш UI процесс, останавливаем..."
                stop_ui
            else
                echo -e "${RED}✗${NC} Порт занят другим процессом!"
                echo -e "${YELLOW}→${NC} Используйте другой порт: ./scripts/restart_ui.sh --port 8080"
                return 1
            fi
        fi
    fi

    return 0
}

# Функция для запуска UI
start_ui() {
    echo ""
    echo -e "${BLUE}→${NC} Запуск нового экземпляра UI Dashboard..."
    echo -e "${BLUE}→${NC} Порт: $PORT"
    echo -e "${BLUE}→${NC} Share: ${SHARE_FLAG:-No}"
    echo ""

    # Переходим в корень проекта
    cd "$PROJECT_ROOT"

    # Активируем виртуальное окружение (если существует)
    if [ -f "venv/bin/activate" ]; then
        echo -e "${BLUE}→${NC} Активация виртуального окружения..."
        source venv/bin/activate
    fi

    # Запускаем UI в фоне
    nohup python scripts/run_ui.py --port $PORT $SHARE_FLAG > "$LOG_FILE" 2>&1 &
    NEW_PID=$!

    # Сохраняем PID
    echo $NEW_PID > "$PID_FILE"

    # Ждём запуска
    echo -e "${YELLOW}→${NC} Ожидание запуска сервера..."
    sleep 3

    # Проверяем что процесс запустился
    if ps -p "$NEW_PID" > /dev/null 2>&1; then
        echo ""
        echo -e "${GREEN}==========================================${NC}"
        echo -e "${GREEN}✓ UI Dashboard успешно запущен!${NC}"
        echo -e "${GREEN}==========================================${NC}"
        echo ""
        echo -e "${BLUE}📊 Dashboard доступен:${NC}"
        echo -e "   ${GREEN}http://localhost:$PORT${NC}"
        echo ""
        echo -e "${BLUE}📝 PID:${NC} $NEW_PID"
        echo -e "${BLUE}📂 Log file:${NC} $LOG_FILE"
        echo ""
        echo -e "${YELLOW}💡 Для просмотра логов:${NC}"
        echo -e "   tail -f $LOG_FILE"
        echo ""
        echo -e "${YELLOW}💡 Для остановки:${NC}"
        echo -e "   kill $NEW_PID"
        echo -e "   или ./scripts/stop_ui.sh"
        echo ""

        return 0
    else
        echo -e "${RED}✗ Ошибка при запуске UI!${NC}"
        echo -e "${YELLOW}→${NC} Проверьте логи: tail -f $LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

# === MAIN EXECUTION ===

# 1. Проверяем запущенный процесс
if check_running; then
    echo ""
    stop_ui || {
        echo -e "${RED}✗ Не удалось остановить старый процесс${NC}"
        exit 1
    }
    echo ""
fi

# 2. Проверяем порт
check_port || exit 1

# 3. Запускаем UI
start_ui || exit 1

echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}  Готово!${NC}"
echo -e "${GREEN}==========================================${NC}"
