# INSTALLATION.md
# Руководство по установке Domain-Specific MoE System

---

## Содержание

1. [Системные требования](#системные-требования)
2. [Быстрая установка](#быстрая-установка)
3. [Установка для разработки](#установка-для-разработки)
4. [Установка на Edge устройства](#установка-на-edge-устройства)
5. [Docker установка](#docker-установка)
6. [Настройка базы данных](#настройка-базы-данных)
7. [Проверка установки](#проверка-установки)
8. [Решение проблем](#решение-проблем)

---

## Системные требования

### Минимальные требования

- **OS**: Linux (Ubuntu 20.04+), Windows 10+, macOS 11+
- **CPU**: 4 cores (Ryzen 5 4500U или эквивалент)
- **RAM**: 8 GB
- **Storage**: 50 GB SSD
- **Python**: 3.10 или выше
- **GPU** (опционально): NVIDIA с 4GB+ VRAM

### Рекомендуемые требования

- **CPU**: 8+ cores
- **RAM**: 16+ GB
- **Storage**: 200 GB NVMe SSD
- **GPU**: NVIDIA RTX 3060 или лучше (12GB VRAM)

### Для роботизированного тела

- **Edge Device**: NVIDIA Jetson Xavier NX или лучше
- **Servos**: Dynamixel или совместимые
- **Cameras**: USB 3.0 или CSI камеры
- **Microphone**: USB array microphone
- **Power**: 12V 10A battery

---

## Быстрая установка

### Шаг 1: Клонирование репозитория

```bash
git clone https://github.com/your-org/domain-moe-system.git
cd domain-moe-system
```

### Шаг 2: Создание виртуального окружения

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

### Шаг 3: Установка зависимостей

```bash
pip install -r requirements.txt
```

### Шаг 4: Настройка конфигурации

```bash
cp config.example.yaml config.yaml
# Отредактируйте config.yaml под ваши нужды
```

### Шаг 5: Инициализация базы данных

```bash
python scripts/init_database.py
```

### Шаг 6: Запуск системы

```bash
# Запуск backend API
uvicorn main:app --host 0.0.0.0 --port 8000

# В другом терминале: запуск мониторинга
python monitoring.py
```

### Шаг 7: Проверка

```bash
# Откройте в браузере
http://localhost:8000/docs

# Или через CLI
python cli.py list-experts
```

---

## Установка для разработки

### Дополнительные зависимости

```bash
pip install -r requirements-dev.txt
```

Включает:
- pytest (тестирование)
- black (форматирование)
- flake8 (линтинг)
- mypy (проверка типов)
- jupyter (ноутбуки)

### Pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

### Запуск тестов

```bash
# Все тесты
pytest tests/ -v

# Конкретный тест
pytest tests/test_expert.py::test_expert_creation -v

# С coverage
pytest --cov=. tests/
```

### Форматирование кода

```bash
# Автоформатирование
black .

# Проверка
flake8 .
```

---

## Установка на Edge устройства

### NVIDIA Jetson Xavier NX

#### Шаг 1: JetPack SDK

```bash
# Установите JetPack SDK через NVIDIA SDK Manager
# Или используйте готовый образ

# Проверка
jtop
```

#### Шаг 2: PyTorch для Jetson

```bash
# Установите PyTorch wheel для Jetson
wget https://nvidia.box.com/shared/static/[link].whl
pip install torch-*.whl
```

#### Шаг 3: TensorRT

```bash
# TensorRT уже включен в JetPack
python -c "import tensorrt; print(tensorrt.__version__)"
```

#### Шаг 4: Оптимизация моделей

```bash
# Конвертация моделей в TensorRT
python scripts/convert_to_tensorrt.py \
  --model models/experts/python_expert/model.onnx \
  --output models/experts/python_expert/model.trt \
  --fp16  # Используем FP16 на Xavier
```

### Raspberry Pi 4

#### Шаг 1: Подготовка системы

```bash
# Обновление
sudo apt update && sudo apt upgrade -y

# Установка системных зависимостей
sudo apt install -y \
  python3-dev \
  python3-pip \
  build-essential \
  cmake \
  git
```

#### Шаг 2: PyTorch для ARM

```bash
# Установка PyTorch (скомпилированная для ARM)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Шаг 3: Оптимизация для ARM

```bash
# Квантизация моделей в INT8
python scripts/quantize_model.py \
  --model models/experts/python_expert/model.pt \
  --quantization int8 \
  --backend qnnpack  # Оптимизировано для ARM
```

#### Шаг 4: Настройка swap (для 4GB моделей)

```bash
# Увеличить swap до 4GB
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# CONF_SWAPSIZE=4096
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

---

## Docker установка

### Быстрый старт с Docker

```bash
# Сборка образа
docker build -t embodied-ai:latest .

# Запуск
docker run -d \
  --name embodied-ai \
  -p 8000:8000 \
  -p 9090:9090 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  embodied-ai:latest
```

### Docker Compose

```bash
# Запуск всех сервисов
docker-compose up -d

# Проверка логов
docker-compose logs -f backend

# Остановка
docker-compose down
```

### Docker Compose включает:

- **backend** - Основное API
- **db** - PostgreSQL база данных
- **redis** - Кэш и очереди
- **prometheus** - Мониторинг метрик
- **grafana** - Визуализация

---

## Настройка базы данных

### PostgreSQL

#### Установка

```bash
# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib

# macOS
brew install postgresql

# Запуск
sudo systemctl start postgresql
```

#### Создание базы данных

```bash
# Подключение
sudo -u postgres psql

# Создание пользователя и БД
CREATE USER moe_user WITH PASSWORD 'your_password';
CREATE DATABASE moe_db OWNER moe_user;
GRANT ALL PRIVILEGES ON DATABASE moe_db TO moe_user;

# Выход
\q
```

#### Инициализация схемы

```bash
# Применить миграции
python scripts/init_database.py --config config.yaml
```

### Redis

#### Установка

```bash
# Ubuntu/Debian
sudo apt install redis-server

# macOS
brew install redis

# Запуск
redis-server
```

#### Проверка

```bash
redis-cli ping
# Должно вернуть: PONG
```

---

## Настройка конфигурации

### config.yaml

```yaml
# Основные настройки
system:
  name: "Domain MoE System"
  version: "1.0.0"
  log_level: "INFO"

# База данных
database:
  host: "localhost"
  port: 5432
  name: "moe_db"
  user: "moe_user"
  password: "your_password"

# Redis
redis:
  host: "localhost"
  port: 6379
  db: 0

# Эксперты
experts:
  model_dir: "models/experts"
  cache_size: 3
  default_quantization: "Q8"

# Router
router:
  strategy: "learned"  # learned, rule_based, hierarchical
  confidence_threshold: 0.5

# Мультимодальность
multimodal:
  vision:
    enabled: true
    device: "cpu"  # cpu, cuda
  audio:
    enabled: true
    sample_rate: 16000
  motor:
    enabled: false
    config_file: "configs/robot_config.yaml"

# API
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4

# Мониторинг
monitoring:
  prometheus:
    enabled: true
    port: 9090
  grafana:
    enabled: true
    port: 3001
```

---

## Проверка установки

### Системная проверка

```bash
python scripts/check_installation.py
```

Вывод:
```
✓ Python version: 3.10.12
✓ PyTorch version: 2.1.0
✓ CUDA available: True (11.8)
✓ Database connection: OK
✓ Redis connection: OK
✓ Model directory: OK
✓ Permissions: OK

Installation check: PASSED
```

### Тест производительности

```bash
python scripts/benchmark.py
```

Вывод:
```
Running benchmarks...

Text Inference: 15.2ms (65 tok/s)
Vision Inference: 46.8ms
Audio Inference: 12.3ms
Router Latency: 3.1ms

Memory Usage: 2.8GB / 16GB
CPU Usage: 45%
GPU Usage: 78%

Benchmark: PASSED
```

### Функциональный тест

```bash
# Создать тестового эксперта
python cli.py create-expert \
  --name test_expert \
  --domain test \
  --size tiny

# Протестировать inference
python cli.py test "Hello world"
```

---

## Решение проблем

### Проблема: CUDA not available

**Симптомы:**
```
RuntimeError: CUDA not available
```

**Решение:**
```bash
# Проверить драйверы NVIDIA
nvidia-smi

# Переустановить CUDA toolkit
# Для Ubuntu:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install cuda

# Переустановить PyTorch
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Проблема: Out of memory

**Симптомы:**
```
RuntimeError: CUDA out of memory
```

**Решение:**
```bash
# 1. Уменьшить batch size
# В config.yaml:
# batch_size: 2  # вместо 4

# 2. Использовать квантизацию
python scripts/quantize_model.py --quantization Q4

# 3. Уменьшить размер модели
# Использовать меньших экспертов

# 4. Включить gradient checkpointing
# В config.yaml:
# training:
#   gradient_checkpointing: true
```

### Проблема: Database connection failed

**Симптомы:**
```
psycopg2.OperationalError: could not connect to server
```

**Решение:**
```bash
# Проверить статус PostgreSQL
sudo systemctl status postgresql

# Запустить если не запущен
sudo systemctl start postgresql

# Проверить соединение
psql -h localhost -U moe_user -d moe_db

# Проверить права
sudo -u postgres psql
\du  # список пользователей
\l   # список баз данных
```

### Проблема: Port already in use

**Симптомы:**
```
Error: Address already in use
```

**Решение:**
```bash
# Найти процесс использующий порт
sudo lsof -i :8000

# Убить процесс
kill -9 <PID>

# Или изменить порт в config.yaml
api:
  port: 8001
```

### Проблема: Slow inference on CPU

**Симптомы:**
- Inference speed < 10 tok/s

**Решение:**
```bash
# 1. Проверить квантизацию
python cli.py info <expert_id>
# Должно быть Q8 или Q4

# 2. Включить оптимизации
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

# 3. Использовать llama.cpp
python scripts/convert_to_gguf.py \
  --model models/experts/expert.pt \
  --output models/experts/expert.gguf \
  --quantization Q4_K_M

# 4. Уменьшить context length
# В config.yaml:
# max_seq_len: 1024  # вместо 2048
```

### Проблема: ImportError

**Симптомы:**
```
ImportError: No module named 'xxx'
```

**Решение:**
```bash
# Проверить виртуальное окружение
which python
# Должно показать путь к venv

# Активировать если не активно
source venv/bin/activate

# Переустановить зависимости
pip install -r requirements.txt --force-reinstall

# Или установить конкретный пакет
pip install <package_name>
```

---

## Следующие шаги

После успешной установки:

1. **Создайте первого эксперта**
   ```bash
   python cli.py create-expert --name python_expert --domain python --size medium
   ```

2. **Загрузите данные для обучения**
   ```bash
   python scripts/download_dataset.py --domain python
   ```

3. **Обучите эксперта**
   ```bash
   python cli.py train 1 data/python_dataset.jsonl --epochs 10
   ```

4. **Протестируйте систему**
   ```bash
   python cli.py test "Write Python code to sort a list"
   ```

5. **Настройте мониторинг**
   - Откройте Grafana: http://localhost:3001
   - Логин: admin / admin
   - Импортируйте dashboard из `grafana-dashboards/`

6. **Изучите документацию**
   - [ARCHITECTURE.md](ARCHITECTURE.md) - Техническая архитектура
   - [API_REFERENCE.md](API_REFERENCE.md) - API документация
   - [HARDWARE_GUIDE.md](HARDWARE_GUIDE.md) - Подключение hardware

---

## Поддержка

Если возникли проблемы:

1. Проверьте [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Создайте issue на GitHub
3. Спросите в Discord сообществе
4. Email: support@example.com

---

**Удачной установки!** 🚀
