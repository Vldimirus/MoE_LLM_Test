# TROUBLESHOOTING.md
# Решение проблем Domain-Specific MoE System

---

## Содержание

1. [Общие проблемы](#общие-проблемы)
2. [Проблемы установки](#проблемы-установки)
3. [Проблемы с экспертами](#проблемы-с-экспертами)
4. [Проблемы обучения](#проблемы-обучения)
5. [Проблемы производительности](#проблемы-производительности)
6. [Проблемы с hardware](#проблемы-с-hardware)
7. [Проблемы с базой данных](#проблемы-с-базой-данных)
8. [Проблемы с сетью](#проблемы-с-сетью)
9. [Отладка](#отладка)
10. [FAQ](#faq)

---

## Общие проблемы

### Проблема: Система не запускается

**Симптомы:**
```
Error: Failed to start application
```

**Диагностика:**
```bash
# Проверить логи
tail -f logs/app.log

# Проверить порты
sudo lsof -i :8000
sudo lsof -i :9090

# Проверить процессы
ps aux | grep python
```

**Решения:**

1. **Порт занят:**
   ```bash
   # Убить процесс
   kill -9 $(lsof -t -i:8000)
   
   # Или изменить порт
   # В config.yaml:
   api:
     port: 8001
   ```

2. **Отсутствуют зависимости:**
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

3. **Проблемы с правами:**
   ```bash
   chmod +x main.py
   sudo chown -R $USER:$USER .
   ```

### Проблема: Import Error

**Симптомы:**
```
ModuleNotFoundError: No module named 'xxx'
```

**Решения:**

1. **Проверить виртуальное окружение:**
   ```bash
   which python
   # Должно показать путь к venv
   
   # Если нет, активировать:
   source venv/bin/activate
   ```

2. **Установить отсутствующий модуль:**
   ```bash
   pip install <module_name>
   ```

3. **Проверить PYTHONPATH:**
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

---

## Проблемы установки

### CUDA not available

**Симптомы:**
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Диагностика:**
```bash
# Проверить NVIDIA драйвер
nvidia-smi

# Проверить CUDA
nvcc --version

# Проверить PyTorch версию
python -c "import torch; print(torch.__version__)"
```

**Решения:**

1. **Установить NVIDIA драйвер:**
   ```bash
   # Ubuntu
   sudo apt install nvidia-driver-525
   sudo reboot
   ```

2. **Установить CUDA Toolkit:**
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
   sudo sh cuda_11.8.0_520.61.05_linux.run
   
   # Добавить в PATH
   echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Переустановить PyTorch с CUDA:**
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Database Connection Failed

**Симптомы:**
```
psycopg2.OperationalError: could not connect to server
```

**Диагностика:**
```bash
# Проверить статус PostgreSQL
sudo systemctl status postgresql

# Проверить порт
sudo ss -tulpn | grep 5432

# Попробовать подключиться
psql -h localhost -U moe_user -d moe_db
```

**Решения:**

1. **Запустить PostgreSQL:**
   ```bash
   sudo systemctl start postgresql
   sudo systemctl enable postgresql
   ```

2. **Проверить конфигурацию:**
   ```bash
   # /etc/postgresql/*/main/postgresql.conf
   listen_addresses = 'localhost'
   port = 5432
   
   # Перезапустить
   sudo systemctl restart postgresql
   ```

3. **Проверить пароль:**
   ```bash
   sudo -u postgres psql
   \password moe_user
   # Введите новый пароль
   ```

4. **Проверить права доступа:**
   ```bash
   # /etc/postgresql/*/main/pg_hba.conf
   # Добавить:
   local   all   moe_user   md5
   host    all   moe_user   127.0.0.1/32   md5
   
   sudo systemctl restart postgresql
   ```

### Out of Memory (OOM)

**Симптомы:**
```
RuntimeError: CUDA out of memory
Killed (процесс убит kernel)
```

**Диагностика:**
```bash
# Проверить память
free -h

# Проверить GPU память
nvidia-smi

# Мониторинг в реальном времени
watch -n 1 nvidia-smi
```

**Решения:**

1. **Уменьшить batch size:**
   ```yaml
   # config.yaml
   training:
     batch_size: 2  # вместо 4
   ```

2. **Включить gradient checkpointing:**
   ```yaml
   training:
     gradient_checkpointing: true
   ```

3. **Использовать меньшую модель:**
   ```bash
   python cli.py create-expert --size small  # вместо medium
   ```

4. **Увеличить swap:**
   ```bash
   sudo swapoff -a
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

5. **Очистить кэш GPU:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

---

## Проблемы с экспертами

### Expert Not Found

**Симптомы:**
```json
{
  "error": "Expert with id=999 not found"
}
```

**Диагностика:**
```bash
# Список экспертов
python cli.py list-experts

# Проверить в БД
psql -U moe_user -d moe_db
SELECT id, name, status FROM experts;
```

**Решения:**

1. **Использовать правильный ID:**
   ```bash
   # Найти ID
   python cli.py list-experts | grep python_expert
   ```

2. **Создать эксперта если не существует:**
   ```bash
   python cli.py create-expert --name python_expert --domain python
   ```

### Expert Inference Too Slow

**Симптомы:**
- Inference speed < 10 tok/s на CPU
- Долгое ожидание ответа

**Диагностика:**
```bash
# Benchmark
python scripts/benchmark_expert.py --expert_id 5

# Проверить квантизацию
python cli.py info 5 | grep quantization
```

**Решения:**

1. **Квантизация:**
   ```bash
   # Конвертировать в Q4
   python scripts/quantize_model.py \
     --expert_id 5 \
     --quantization Q4
   ```

2. **Использовать llama.cpp:**
   ```bash
   # Конвертировать в GGUF
   python scripts/convert_to_gguf.py \
     --expert_id 5 \
     --quantization Q4_K_M
   
   # Запустить
   ./llama-cli --model expert_5.gguf --threads 6
   ```

3. **Уменьшить context length:**
   ```yaml
   experts:
     max_seq_len: 1024  # вместо 2048
   ```

4. **Включить KV-cache:**
   ```yaml
   inference:
     use_kv_cache: true
   ```

5. **Оптимизации компилятора:**
   ```bash
   export OMP_NUM_THREADS=6
   export MKL_NUM_THREADS=6
   ```

### Expert Gives Poor Quality Results

**Симптомы:**
- Nonsensical output
- Hallucinations
- Low confidence

**Диагностика:**
```bash
# Проверить метрики
python cli.py info 5

# Проверить training history
psql -U moe_user -d moe_db
SELECT * FROM training_jobs WHERE expert_id = 5 ORDER BY started_at DESC LIMIT 5;
```

**Решения:**

1. **Переобучить с большим датасетом:**
   ```bash
   python cli.py train 5 data/large_dataset.jsonl --epochs 20
   ```

2. **Увеличить размер модели:**
   ```bash
   python cli.py resize 5 --size medium
   ```

3. **Проверить качество датасета:**
   ```python
   # Анализ датасета
   python scripts/analyze_dataset.py data/dataset.jsonl
   ```

4. **Настроить sampling parameters:**
   ```python
   result = client.inference.generate(
       prompt="...",
       temperature=0.3,  # меньше = более детерминированно
       top_p=0.9,
       top_k=40
   )
   ```

---

## Проблемы обучения

### Training Job Stuck

**Симптомы:**
- Status остаётся "running"
- Loss не изменяется
- Процесс завис

**Диагностика:**
```bash
# Проверить процесс
ps aux | grep train

# Проверить логи
tail -f logs/training_456.log

# Проверить GPU utilization
nvidia-smi
```

**Решения:**

1. **Убить зависший процесс:**
   ```bash
   # Найти PID
   ps aux | grep train_expert
   
   # Убить
   kill -9 <PID>
   
   # Обновить статус в БД
   psql -U moe_user -d moe_db
   UPDATE training_jobs SET status = 'failed' WHERE id = 456;
   ```

2. **Перезапустить с checkpoint:**
   ```bash
   python cli.py train 5 dataset.jsonl --resume --checkpoint epoch_3.pt
   ```

### Loss Exploding/NaN

**Симптомы:**
```
Epoch 1, Step 50: Loss=0.456
Epoch 1, Step 51: Loss=NaN
```

**Диагностика:**
```bash
# Проверить данные
python scripts/validate_dataset.py data/dataset.jsonl

# Проверить learning rate
python cli.py training-status 456 | grep learning_rate
```

**Решения:**

1. **Уменьшить learning rate:**
   ```yaml
   training:
     learning_rate: 1e-5  # вместо 1e-4
   ```

2. **Gradient clipping:**
   ```yaml
   training:
     max_grad_norm: 1.0
   ```

3. **Проверить данные на NaN:**
   ```python
   import json
   
   with open('dataset.jsonl') as f:
       for i, line in enumerate(f):
           data = json.loads(line)
           if 'nan' in str(data).lower():
               print(f"NaN found at line {i}")
   ```

4. **Использовать mixed precision:**
   ```yaml
   training:
     use_fp16: true
   ```

### Training Very Slow

**Симптомы:**
- < 1 it/s
- ETA > 100 hours

**Диагностика:**
```bash
# CPU/GPU utilization
htop
nvidia-smi dmon

# I/O wait
iostat -x 1
```

**Решения:**

1. **Увеличить batch size:**
   ```yaml
   training:
     batch_size: 8  # если память позволяет
   ```

2. **Gradient accumulation:**
   ```yaml
   training:
     batch_size: 2
     gradient_accumulation: 4  # эффективный batch = 8
   ```

3. **Больше workers для dataloader:**
   ```yaml
   training:
     num_workers: 4
   ```

4. **Использовать SSD вместо HDD:**
   ```bash
   # Переместить данные
   mv data/ /path/to/ssd/data/
   ln -s /path/to/ssd/data/ data
   ```

5. **LoRA вместо full fine-tuning:**
   ```yaml
   training:
     use_lora: true
     lora_rank: 8
   ```

---

## Проблемы производительности

### High Latency

**Симптомы:**
- API requests > 1 second
- Slow response time

**Диагностика:**
```bash
# Профилирование
python -m cProfile -o profile.stats main.py

# Анализ
python -m pstats profile.stats
stats> sort cumulative
stats> stats 20

# Мониторинг
curl http://localhost:9090/metrics | grep latency
```

**Решения:**

1. **Кэширование:**
   ```yaml
   inference:
     cache_enabled: true
     cache_size: 1000
   ```

2. **Батчинг запросов:**
   ```yaml
   inference:
     dynamic_batching: true
     max_batch_size: 32
     timeout_ms: 10
   ```

3. **Асинхронная обработка:**
   ```python
   # Используй async/await
   async def process_request(request):
       result = await inference_async(request)
       return result
   ```

4. **Load balancing:**
   ```yaml
   # Несколько worker процессов
   api:
     workers: 4
   ```

### Memory Leak

**Симптомы:**
- Memory usage постоянно растёт
- Система становится медленнее
- Eventually OOM

**Диагностика:**
```bash
# Мониторинг памяти
watch -n 1 'free -h'

# Профилирование памяти
python -m memory_profiler main.py

# Поиск утечек
python -m tracemalloc main.py
```

**Решения:**

1. **Очистка кэшей:**
   ```python
   import gc
   import torch
   
   # После каждого inference
   torch.cuda.empty_cache()
   gc.collect()
   ```

2. **Ограничить размер кэша:**
   ```yaml
   experts:
     cache_size: 3  # максимум 3 эксперта в памяти
   ```

3. **Периодический рестарт:**
   ```bash
   # Systemd service с рестартом
   [Service]
   Restart=always
   RuntimeMaxSec=86400  # рестарт каждые 24 часа
   ```

4. **Использовать context manager:**
   ```python
   with ExpertContext(expert_id) as expert:
       result = expert.inference(prompt)
   # expert автоматически выгружается
   ```

### High CPU Usage

**Симптомы:**
- CPU 100% постоянно
- Система тормозит

**Диагностика:**
```bash
# Top процессы
top -o %CPU

# Профилирование CPU
py-spy record -o profile.svg -- python main.py
```

**Решения:**

1. **Ограничить потоки:**
   ```bash
   export OMP_NUM_THREADS=4
   export MKL_NUM_THREADS=4
   ```

2. **Nice priority:**
   ```bash
   nice -n 10 python main.py
   ```

3. **CPU affinity:**
   ```bash
   taskset -c 0-3 python main.py  # только cores 0-3
   ```

---

## Проблемы с hardware

### Servo Not Responding

**Симптомы:**
- Servo не двигается
- Timeout errors

**Диагностика:**
```bash
# Scan for servos
python scripts/scan_servos.py

# Check power
# Multimeter: должно быть 12V на servo bus

# Check connection
ls /dev/ttyUSB*
```

**Решения:**

1. **Проверить питание:**
   ```bash
   # Servo bus должен иметь 12V
   # Если нет, проверить PSU и проводку
   ```

2. **Проверить baudrate:**
   ```python
   # Попробовать разные baudrates
   for baudrate in [57600, 115200, 1000000]:
       try:
           scan_servos(baudrate)
       except:
           continue
   ```

3. **Сбросить servo ID:**
   ```python
   # Если ID неизвестен
   from dynamixel_sdk import *
   
   # Попробовать broadcast
   packet_handler.broadcastPing(port_handler)
   ```

4. **Проверить кабель:**
   ```bash
   # Swap cable с working servo
   # Если проблема переместилась - кабель плохой
   ```

### Camera Not Detected

**Симптомы:**
```
cv2.error: !_src.empty()
/dev/video0: No such file or directory
```

**Диагностика:**
```bash
# List USB devices
lsusb

# List video devices
ls -l /dev/video*

# Check dmesg
dmesg | grep video
```

**Решения:**

1. **Проверить USB подключение:**
   ```bash
   # Reconnect camera
   # Check dmesg
   dmesg | tail
   ```

2. **Права доступа:**
   ```bash
   sudo chmod 666 /dev/video0
   sudo usermod -a -G video $USER
   
   # Logout and login
   ```

3. **Установить драйверы:**
   ```bash
   sudo apt install v4l-utils
   v4l2-ctl --list-devices
   ```

4. **Попробовать другой порт USB:**
   ```bash
   # USB 3.0 ports (blue)
   # Некоторые камеры не работают на USB 2.0
   ```

### IMU Drift

**Симптомы:**
- Orientation неточная
- Значения дрейфуют
- Incorrect readings

**Диагностика:**
```python
# Тест IMU
python tests/hardware/test_imu.py

# Проверить на flat surface
# Accel Z должен быть ~9.8 m/s²
# Gyro должен быть ~0 °/s
```

**Решения:**

1. **Калибровка:**
   ```python
   python scripts/calibrate_imu.py --samples 2000
   ```

2. **Проверить крепление:**
   ```bash
   # IMU должен быть жёстко закреплён
   # Vibration вызывает дрейф
   ```

3. **Temperature compensation:**
   ```python
   # Подождать пока IMU прогреется
   time.sleep(60)
   # Потом калибровать
   ```

4. **Complementary filter:**
   ```python
   # Объединить accel и gyro
   angle = 0.98 * (angle + gyro * dt) + 0.02 * accel_angle
   ```

---

## Проблемы с базой данных

### Database Lock

**Симптомы:**
```
database is locked
```

**Решения:**

1. **Закрыть все соединения:**
   ```bash
   # PostgreSQL
   psql -U postgres
   SELECT pg_terminate_backend(pid) 
   FROM pg_stat_activity 
   WHERE datname = 'moe_db';
   ```

2. **Увеличить timeout:**
   ```yaml
   database:
     timeout: 30  # seconds
   ```

### Slow Queries

**Симптомы:**
- API requests медленные
- Database CPU 100%

**Диагностика:**
```sql
-- Найти медленные запросы
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;
```

**Решения:**

1. **Добавить индексы:**
   ```sql
   CREATE INDEX idx_experts_status ON experts(status);
   CREATE INDEX idx_training_jobs_expert ON training_jobs(expert_id);
   ```

2. **Vacuum:**
   ```bash
   psql -U moe_user -d moe_db
   VACUUM ANALYZE;
   ```

3. **Оптимизация запросов:**
   ```sql
   -- Плохо
   SELECT * FROM experts;
   
   -- Хорошо
   SELECT id, name, status FROM experts WHERE status = 'ready';
   ```

---

## Проблемы с сетью

### Connection Refused

**Симптомы:**
```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Диагностика:**
```bash
# Проверить порт
sudo ss -tulpn | grep 8000

# Проверить firewall
sudo ufw status

# Попробовать curl
curl http://localhost:8000/health
```

**Решения:**

1. **Проверить что сервис запущен:**
   ```bash
   ps aux | grep uvicorn
   ```

2. **Открыть порт в firewall:**
   ```bash
   sudo ufw allow 8000/tcp
   sudo ufw reload
   ```

3. **Bind на правильный интерфейс:**
   ```yaml
   api:
     host: "0.0.0.0"  # не 127.0.0.1
     port: 8000
   ```

### Timeout Errors

**Симптомы:**
```
TimeoutError: Request timed out
```

**Решения:**

1. **Увеличить timeout:**
   ```python
   client = Client(
       api_key="...",
       timeout=300  # 5 minutes
   )
   ```

2. **Использовать async:**
   ```python
   result = await client.inference.generate_async(...)
   ```

3. **Streaming response:**
   ```python
   for chunk in client.inference.generate_stream(...):
       print(chunk)
   ```

---

## Отладка

### Включить debug logging

```yaml
# config.yaml
system:
  log_level: "DEBUG"
```

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Трассировка запросов

```python
# Добавить request ID
import uuid

request_id = str(uuid.uuid4())
logger.info(f"[{request_id}] Starting inference")
```

### Профилирование

```bash
# CPU profiling
python -m cProfile -o profile.stats main.py

# Memory profiling
python -m memory_profiler main.py

# Line profiling
kernprof -l -v main.py
```

### Remote debugging

```python
# PyCharm remote debug
import pydevd_pycharm
pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True)

# VSCode remote debug
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()
```

---

## FAQ

### Q: Как увеличить скорость inference?

**A:** 
1. Используй квантизацию (Q4/Q8)
2. Уменьши context length
3. Используй KV-cache
4. Batch requests
5. Используй llama.cpp вместо PyTorch

### Q: Как уменьшить использование памяти?

**A:**
1. Квантизация моделей
2. Уменьши cache size
3. Gradient checkpointing при обучении
4. Используй mmap для экспертов
5. Периодически чисти кэш

### Q: Можно ли использовать без GPU?

**A:** Да! Система оптимизирована для CPU:
- Q8 квантизация
- llama.cpp backend
- Оптимизации BLAS (MKL, OpenBLAS)
- Ожидается 100-300 tok/s на хорошем CPU

### Q: Как добавить новый домен эксперта?

**A:**
```bash
# 1. Создать эксперта
python cli.py create-expert --name rust_expert --domain rust

# 2. Подготовить датасет
python scripts/prepare_dataset.py --domain rust

# 3. Обучить
python cli.py train <expert_id> data/rust_dataset.jsonl

# 4. Переобучить router
python scripts/retrain_router.py
```

### Q: Как обновить эксперта без downtime?

**A:** Blue-green deployment:
```bash
# 1. Создать новую версию
python cli.py create-expert --name python_expert_v2 --base python_expert

# 2. Обучить
python cli.py train <new_id> dataset.jsonl

# 3. A/B тест
python cli.py ab-test --control <old_id> --treatment <new_id>

# 4. Если успешно, переключить
python cli.py promote <new_id>
```

---

## Получить помощь

Если проблема не решена:

1. **Проверь логи:**
   ```bash
   tail -f logs/*.log
   ```

2. **Создай issue на GitHub** с:
   - Версией системы
   - Описанием проблемы
   - Шагами для воспроизведения
   - Логами

3. **Discord сообщество:** [ссылка]

4. **Email support:** support@example.com

---

**Last Updated**: 2026-01-06  
**Version**: 1.0.0
