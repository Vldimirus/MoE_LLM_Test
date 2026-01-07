"""
Примеры использования ExpertModel.

Демонстрирует основные возможности модели.
"""

import torch
import sys
import os

# Добавляем путь к src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from python.models.expert import ExpertModel


def example_1_basic_usage():
    """Пример 1: Базовое использование."""
    print("=" * 70)
    print("Пример 1: Базовое использование ExpertModel")
    print("=" * 70)

    # Создаём небольшую модель
    model = ExpertModel(
        vocab_size=5000,
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=1024,
        max_seq_len=128
    )

    print(f"\nСоздана модель с {model.get_num_params():,} параметрами")

    # Тестовый вход
    input_ids = torch.randint(0, 5000, (1, 10))
    print(f"Входные токены: {input_ids[0].tolist()}")

    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)

    print(f"Выходные логиты: {logits.shape}")
    print(f"Предсказанные следующие токены: {torch.argmax(logits[0], dim=-1).tolist()}")


def example_2_text_generation():
    """Пример 2: Генерация текста с разными параметрами."""
    print("\n" + "=" * 70)
    print("Пример 2: Генерация текста")
    print("=" * 70)

    model = ExpertModel(
        vocab_size=1000,
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=1024
    )

    # Начальная последовательность
    start_tokens = torch.tensor([[1, 2, 3, 4, 5]])

    print(f"\nНачальные токены: {start_tokens[0].tolist()}")

    # Генерация с разными параметрами
    strategies = [
        {"name": "Default", "params": {}},
        {"name": "Low temp", "params": {"temperature": 0.5}},
        {"name": "High temp", "params": {"temperature": 1.5}},
        {"name": "Top-k", "params": {"temperature": 0.8, "top_k": 20}},
        {"name": "Top-p", "params": {"temperature": 0.8, "top_p": 0.9}},
        {"name": "Combined", "params": {"temperature": 0.7, "top_k": 50, "top_p": 0.9}},
    ]

    for strategy in strategies:
        generated = model.generate(
            start_tokens,
            max_new_tokens=10,
            **strategy["params"]
        )
        tokens = generated[0].tolist()
        print(f"{strategy['name']:12} → {tokens}")


def example_3_model_sizes():
    """Пример 3: Разные размеры моделей."""
    print("\n" + "=" * 70)
    print("Пример 3: Сравнение размеров моделей")
    print("=" * 70)

    configs = [
        {
            "name": "Tiny",
            "vocab_size": 5000,
            "d_model": 128,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": 512
        },
        {
            "name": "Small",
            "vocab_size": 10000,
            "d_model": 256,
            "n_layers": 4,
            "n_heads": 4,
            "d_ff": 1024
        },
        {
            "name": "Medium",
            "vocab_size": 32000,
            "d_model": 512,
            "n_layers": 6,
            "n_heads": 8,
            "d_ff": 2048
        },
        {
            "name": "Large",
            "vocab_size": 50000,
            "d_model": 768,
            "n_layers": 8,
            "n_heads": 12,
            "d_ff": 3072
        },
    ]

    print(f"\n{'Модель':<10} {'Параметры':>15} {'FP32':>10} {'FP16':>10} {'INT8':>10}")
    print("-" * 70)

    for config in configs:
        name = config.pop("name")
        model = ExpertModel(**config)

        params = model.get_num_params()
        fp32_mb = params * 4 / 1024**2
        fp16_mb = params * 2 / 1024**2
        int8_mb = params / 1024**2

        print(f"{name:<10} {params:>15,} {fp32_mb:>9.1f}M {fp16_mb:>9.1f}M {int8_mb:>9.1f}M")


def example_4_inference_speed():
    """Пример 4: Тест скорости inference."""
    print("\n" + "=" * 70)
    print("Пример 4: Скорость inference")
    print("=" * 70)

    import time

    model = ExpertModel(
        vocab_size=10000,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048
    )

    # Тестовые данные
    batch_sizes = [1, 2, 4, 8]
    seq_len = 50

    print(f"\n{'Batch':>6} {'Время (ms)':>12} {'Tok/s':>10}")
    print("-" * 35)

    for batch_size in batch_sizes:
        input_ids = torch.randint(0, 10000, (batch_size, seq_len))

        # Warmup
        with torch.no_grad():
            _ = model(input_ids)

        # Benchmark
        start = time.time()
        num_runs = 10

        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_ids)

        elapsed = (time.time() - start) / num_runs * 1000  # ms
        tokens_per_sec = (batch_size * seq_len) / (elapsed / 1000)

        print(f"{batch_size:>6} {elapsed:>12.2f} {tokens_per_sec:>10.1f}")


def example_5_save_load():
    """Пример 5: Сохранение и загрузка модели."""
    print("\n" + "=" * 70)
    print("Пример 5: Сохранение и загрузка модели")
    print("=" * 70)

    # Создаём модель
    model = ExpertModel(
        vocab_size=1000,
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=1024
    )

    print(f"\nОригинальная модель: {model.get_num_params():,} параметров")

    # Тестовый вход
    test_input = torch.randint(0, 1000, (1, 10))

    model.eval()  # Важно! Отключаем dropout для детерминированного вывода
    with torch.no_grad():
        original_output = model(test_input)

    # Сохраняем
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': model.get_model_config()
    }
    torch.save(checkpoint, '/tmp/test_expert_model.pt')
    print("✅ Модель сохранена в /tmp/test_expert_model.pt")

    # Загружаем
    checkpoint = torch.load('/tmp/test_expert_model.pt')
    config = checkpoint['config']

    new_model = ExpertModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len']
    )

    new_model.load_state_dict(checkpoint['model_state_dict'])
    new_model.eval()  # Важно! Отключаем dropout
    print("✅ Модель загружена из checkpoint")

    # Проверяем что выходы одинаковые
    with torch.no_grad():
        new_output = new_model(test_input)

    difference = torch.abs(original_output - new_output).max().item()
    print(f"\nМаксимальная разница в выходах: {difference:.10f}")
    print("✅ Модель загружена корректно!" if difference < 1e-6 else "❌ Проблема с загрузкой")


if __name__ == "__main__":
    example_1_basic_usage()
    example_2_text_generation()
    example_3_model_sizes()
    example_4_inference_speed()
    example_5_save_load()

    print("\n" + "=" * 70)
    print("✅ Все примеры выполнены успешно!")
    print("=" * 70)
