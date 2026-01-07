# API_REFERENCE.md
# API Reference - Domain-Specific MoE System

---

## Содержание

1. [Обзор API](#обзор-api)
2. [Аутентификация](#аутентификация)
3. [Expert Management API](#expert-management-api)
4. [Training API](#training-api)
5. [Inference API](#inference-api)
6. [Router API](#router-api)
7. [Multimodal API](#multimodal-api)
8. [Monitoring API](#monitoring-api)
9. [Hardware Control API](#hardware-control-api)
10. [WebSocket API](#websocket-api)
11. [Error Codes](#error-codes)
12. [Rate Limiting](#rate-limiting)

---

## Обзор API

### Base URL

```
Production: https://api.example.com/v1
Development: http://localhost:8000/v1
```

### Response Format

Все ответы в JSON формате:

```json
{
  "success": true,
  "data": {...},
  "error": null,
  "timestamp": "2026-01-06T12:00:00Z"
}
```

### HTTP Status Codes

- `200` - OK
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `429` - Too Many Requests
- `500` - Internal Server Error

---

## Аутентификация

### API Key

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://api.example.com/v1/experts/list
```

### OAuth 2.0

```bash
# Получить токен
curl -X POST https://api.example.com/oauth/token \
  -d "grant_type=client_credentials" \
  -d "client_id=YOUR_CLIENT_ID" \
  -d "client_secret=YOUR_CLIENT_SECRET"

# Использовать токен
curl -H "Authorization: Bearer ACCESS_TOKEN" \
  https://api.example.com/v1/experts/list
```

---

## Expert Management API

### Список экспертов

**GET** `/experts/list`

Получить список всех экспертов.

**Query Parameters:**

- `category` (optional) - Фильтр по категории
- `status` (optional) - Фильтр по статусу (`ready`, `training`, `disabled`)
- `sort_by` (optional) - Сортировка (`name`, `accuracy`, `speed`)
- `limit` (optional) - Количество результатов (default: 50)
- `offset` (optional) - Offset для пагинации

**Example:**

```bash
curl "https://api.example.com/v1/experts/list?category=programming&status=ready"
```

**Response:**

```json
{
  "success": true,
  "data": {
    "experts": [
      {
        "id": 1,
        "name": "python_expert",
        "domain": "Python Programming",
        "category": "programming",
        "status": "ready",
        "total_params": 1024000000,
        "quantization": "Q8",
        "accuracy": 0.95,
        "inference_speed": 180.5,
        "total_calls": 15000,
        "created_at": "2026-01-01T00:00:00Z"
      }
    ],
    "total": 1,
    "limit": 50,
    "offset": 0
  }
}
```

### Получить эксперта

**GET** `/experts/{expert_id}`

Получить детальную информацию об эксперте.

**Path Parameters:**

- `expert_id` - ID эксперта

**Example:**

```bash
curl "https://api.example.com/v1/experts/5"
```

**Response:**

```json
{
  "success": true,
  "data": {
    "id": 5,
    "name": "python_expert",
    "domain": "Python Programming",
    "category": "programming",
    "config": {
      "d_model": 2048,
      "n_layers": 8,
      "n_heads": 16,
      "d_ff": 8192
    },
    "metrics": {
      "accuracy": 0.95,
      "perplexity": 12.5,
      "validation_loss": 0.234
    },
    "performance": {
      "inference_speed_tokens_per_sec": 180.5,
      "memory_usage_mb": 1024,
      "latency_p50_ms": 15,
      "latency_p95_ms": 45
    },
    "usage_stats": {
      "total_calls": 15000,
      "last_used": "2026-01-06T08:15:30Z"
    }
  }
}
```

### Создать эксперта

**POST** `/experts/create`

Создать нового эксперта.

**Request Body:**

```json
{
  "name": "rust_expert",
  "domain": "Rust Programming",
  "category": "programming",
  "d_model": 2048,
  "n_layers": 8,
  "n_heads": 16,
  "quantization": "Q8",
  "base_model": null
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "expert_id": 42,
    "status": "initializing",
    "estimated_size_gb": 1.02,
    "message": "Expert creation started"
  }
}
```

### Обновить эксперта

**PUT** `/experts/{expert_id}`

Обновить метаданные эксперта.

**Request Body:**

```json
{
  "domain": "Rust Programming Advanced",
  "category": "programming"
}
```

### Удалить эксперта

**DELETE** `/experts/{expert_id}`

Удалить эксперта.

**Query Parameters:**

- `hard_delete` (optional) - Permanent deletion (default: false)

**Example:**

```bash
curl -X DELETE "https://api.example.com/v1/experts/5?hard_delete=true"
```

### Изменить размер эксперта

**POST** `/experts/{expert_id}/resize`

Создать новую версию эксперта с другим размером.

**Request Body:**

```json
{
  "d_model": 1024,
  "n_layers": 6,
  "n_heads": 8
}
```

---

## Training API

### Загрузить датасет

**POST** `/datasets/upload`

Загрузить датасет для обучения.

**Form Data:**

- `file` - Файл датасета (JSONL, TXT, CSV)
- `expert_id` - ID эксперта
- `name` - Название датасета

**Example:**

```bash
curl -X POST https://api.example.com/v1/datasets/upload \
  -F "file=@python_dataset.jsonl" \
  -F "expert_id=5" \
  -F "name=Python Training Data"
```

**Response:**

```json
{
  "success": true,
  "data": {
    "dataset_id": 123,
    "num_samples": 100000,
    "size_mb": 245.3,
    "status": "ready"
  }
}
```

### Запустить обучение

**POST** `/training/start`

Запустить обучение эксперта.

**Request Body:**

```json
{
  "expert_id": 5,
  "dataset_id": 123,
  "learning_rate": 0.0001,
  "batch_size": 4,
  "num_epochs": 10,
  "warmup_steps": 100,
  "gradient_accumulation": 4,
  "use_lora": true,
  "lora_rank": 8
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "job_id": 456,
    "status": "queued",
    "estimated_time_hours": 8.5
  }
}
```

### Статус обучения

**GET** `/training/{job_id}/status`

Получить статус задачи обучения.

**Response:**

```json
{
  "success": true,
  "data": {
    "job_id": 456,
    "status": "running",
    "current_epoch": 3,
    "total_epochs": 10,
    "current_loss": 0.456,
    "best_loss": 0.423,
    "train_loss": [0.8, 0.6, 0.5, 0.456],
    "val_loss": [0.75, 0.55, 0.48, 0.45],
    "eta_hours": 5.2,
    "recent_logs": [
      "Epoch 3, Step 100: Loss=0.456",
      "Epoch 3, Step 110: Loss=0.449"
    ]
  }
}
```

### Остановить обучение

**POST** `/training/{job_id}/stop`

Остановить задачу обучения.

**Response:**

```json
{
  "success": true,
  "data": {
    "status": "stopped",
    "checkpoint_saved": true
  }
}
```

---

## Inference API

### Генерация текста

**POST** `/inference/generate`

Генерация текста с помощью эксперта.

**Request Body:**

```json
{
  "prompt": "Write a Python function to sort a list",
  "expert_id": 5,
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "stop_sequences": ["\n\n", "```"]
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "text": "Here's a Python function to sort a list:\n\n```python\ndef sort_list(lst):\n    return sorted(lst)\n```",
    "tokens_generated": 23,
    "inference_time_ms": 145.2,
    "expert_used": "python_expert"
  }
}
```

### Streaming генерация

**POST** `/inference/generate/stream`

Server-Sent Events (SSE) для потоковой генерации.

**Example:**

```javascript
const eventSource = new EventSource('/v1/inference/generate/stream', {
  method: 'POST',
  body: JSON.stringify({
    prompt: "Write Python code",
    expert_id: 5
  })
});

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.token);
};
```

### Batch inference

**POST** `/inference/batch`

Обработка множественных запросов.

**Request Body:**

```json
{
  "requests": [
    {
      "id": "req1",
      "prompt": "Write Python code",
      "expert_id": 5
    },
    {
      "id": "req2",
      "prompt": "Solve integral",
      "expert_id": 7
    }
  ],
  "max_tokens": 100
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "results": [
      {
        "id": "req1",
        "text": "...",
        "inference_time_ms": 145
      },
      {
        "id": "req2",
        "text": "...",
        "inference_time_ms": 152
      }
    ],
    "total_time_ms": 298
  }
}
```

---

## Router API

### Статистика router

**GET** `/router/stats`

Получить статистику работы router.

**Response:**

```json
{
  "success": true,
  "data": {
    "stats": [
      {
        "expert_id": 5,
        "expert_name": "python_expert",
        "selections": 1500,
        "avg_confidence": 0.85,
        "accuracy": 0.92
      }
    ],
    "issues": [
      {
        "type": "unused_expert",
        "expert": "old_expert",
        "recommendation": "Consider removing or retraining"
      }
    ]
  }
}
```

### Переобучить router

**POST** `/router/retrain`

Запустить переобучение router.

**Request Body:**

```json
{
  "training_data_path": "data/router_training.jsonl",
  "num_epochs": 10
}
```

---

## Multimodal API

### Обработка мультимодального входа

**POST** `/multimodal/process`

Обработка текста, изображения и аудио.

**Form Data:**

- `text` (optional) - Текстовый запрос
- `image` (optional) - Изображение (PNG, JPG)
- `audio` (optional) - Аудио (WAV, MP3)

**Example:**

```bash
curl -X POST https://api.example.com/v1/multimodal/process \
  -F "text=What do you see?" \
  -F "image=@photo.jpg"
```

**Response:**

```json
{
  "success": true,
  "data": {
    "understanding": {
      "vision": {
        "objects": ["person", "car", "tree"],
        "scene": "outdoor street",
        "confidence": 0.89
      },
      "text": {
        "intent": "query",
        "entities": []
      }
    },
    "response": "I see a person standing near a car on a street with trees in the background."
  }
}
```

### Детекция объектов

**POST** `/vision/detect`

Детекция объектов на изображении.

**Form Data:**

- `image` - Изображение

**Response:**

```json
{
  "success": true,
  "data": {
    "detections": [
      {
        "class": "person",
        "confidence": 0.95,
        "bbox": [100, 150, 200, 400],
        "center": [150, 275]
      }
    ],
    "num_objects": 1,
    "inference_time_ms": 45
  }
}
```

### Распознавание речи

**POST** `/audio/transcribe`

Распознавание речи из аудио.

**Form Data:**

- `audio` - Аудио файл

**Response:**

```json
{
  "success": true,
  "data": {
    "text": "Hello, how are you?",
    "language": "en",
    "confidence": 0.92,
    "duration_seconds": 2.5
  }
}
```

### Синтез речи

**POST** `/audio/synthesize`

Text-to-Speech синтез.

**Request Body:**

```json
{
  "text": "Hello world",
  "voice": "default",
  "speed": 1.0
}
```

**Response:** Binary audio data (WAV)

---

## Monitoring API

### Метрики системы

**GET** `/monitoring/metrics`

Получить системные метрики.

**Response:**

```json
{
  "success": true,
  "data": {
    "system": {
      "cpu_usage": 45.2,
      "ram_usage_gb": 8.3,
      "ram_total_gb": 16.0,
      "gpu_usage": 78.5,
      "gpu_memory_gb": 6.2
    },
    "experts": {
      "total": 64,
      "active": 42,
      "training": 2
    },
    "inference": {
      "requests_per_minute": 150,
      "avg_latency_ms": 156.7,
      "cache_hit_rate": 0.65
    }
  }
}
```

### Health check

**GET** `/health`

Проверка здоровья системы.

**Response:**

```json
{
  "status": "healthy",
  "components": {
    "database": "ok",
    "redis": "ok",
    "experts": "ok",
    "gpu": "ok"
  },
  "uptime_seconds": 86400
}
```

---

## Hardware Control API

### Управление сервоприводами

**POST** `/hardware/servo/move`

Переместить сервопривод.

**Request Body:**

```json
{
  "joint_name": "shoulder_pitch_r",
  "angle": 45.0,
  "duration_ms": 1000
}
```

### Захват объекта

**POST** `/hardware/grasp`

Захватить объект.

**Request Body:**

```json
{
  "object_id": 5,
  "force": 5.0
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "grasped": true,
    "force_applied": 4.8,
    "time_ms": 1250
  }
}
```

### Статус сенсоров

**GET** `/hardware/sensors`

Получить показания всех сенсоров.

**Response:**

```json
{
  "success": true,
  "data": {
    "joint_positions": {
      "shoulder_pitch_r": 45.2,
      "elbow_pitch_r": 90.1
    },
    "forces": {
      "gripper_r": 3.5
    },
    "temperatures": {
      "motor_base": 42.3
    },
    "cameras": {
      "head_camera": "active",
      "wrist_camera": "active"
    }
  }
}
```

---

## WebSocket API

### Подключение

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  console.log('Connected');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
};
```

### Подписка на события

**Message:**

```json
{
  "action": "subscribe",
  "channels": ["training", "inference", "hardware"]
}
```

### Получение событий

```json
{
  "channel": "training",
  "event": "epoch_completed",
  "data": {
    "job_id": 456,
    "epoch": 3,
    "loss": 0.456
  }
}
```

---

## Error Codes

| Code | Значение | Описание |
|------|----------|----------|
| 1000 | EXPERT_NOT_FOUND | Эксперт не найден |
| 1001 | EXPERT_NOT_READY | Эксперт не готов |
| 1002 | EXPERT_CREATION_FAILED | Ошибка создания |
| 2000 | TRAINING_FAILED | Ошибка обучения |
| 2001 | DATASET_INVALID | Неверный датасет |
| 3000 | INFERENCE_FAILED | Ошибка inference |
| 3001 | TIMEOUT | Превышен timeout |
| 4000 | HARDWARE_ERROR | Ошибка hardware |
| 4001 | SAFETY_VIOLATION | Нарушение безопасности |

**Example Error Response:**

```json
{
  "success": false,
  "error": {
    "code": 1000,
    "message": "Expert not found",
    "details": "Expert with id=999 does not exist"
  }
}
```

---

## Rate Limiting

### Лимиты

- **Free tier**: 100 запросов/час
- **Pro tier**: 1000 запросов/час
- **Enterprise**: Без лимитов

### Headers

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 987
X-RateLimit-Reset: 1641484800
```

### Превышение лимита

```json
{
  "success": false,
  "error": {
    "code": 429,
    "message": "Rate limit exceeded",
    "retry_after_seconds": 3600
  }
}
```

---

## SDK Examples

### Python

```python
from embodied_ai import Client

client = Client(api_key="YOUR_API_KEY")

# Список экспертов
experts = client.experts.list(category="programming")

# Создание эксперта
expert = client.experts.create(
    name="rust_expert",
    domain="Rust Programming",
    size="medium"
)

# Inference
result = client.inference.generate(
    prompt="Write Rust code",
    expert_id=expert.id,
    max_tokens=100
)

print(result.text)
```

### JavaScript

```javascript
const EmbodiedAI = require('embodied-ai');

const client = new EmbodiedAI({
  apiKey: 'YOUR_API_KEY'
});

// Список экспертов
const experts = await client.experts.list({
  category: 'programming'
});

// Inference
const result = await client.inference.generate({
  prompt: 'Write JavaScript code',
  expertId: 5,
  maxTokens: 100
});

console.log(result.text);
```

---

## Versioning

API версионируется через URL:

- `v1` - Current stable version
- `v2` - Beta version (preview)

---

**API Version**: 1.0.0  
**Last Updated**: 2026-01-06
