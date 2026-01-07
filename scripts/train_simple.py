#!/usr/bin/env python3
"""
Простой скрипт для обучения первой модели general.

Использует:
    - Датасет: data/training/simple_dialogues.jsonl
    - Модель: models/experts/general (перезаписывает тестовую версию)
    - Epochs: 10 (небольшое обучение для первого теста)
"""

import sys
from pathlib import Path

# Добавляем src/python в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src" / "python"))

import torch
from models.expert import ExpertModel
from training.dataset import SimpleTokenizer, create_dataloaders
from training.trainer import Trainer


def main():
    print("=" * 80)
    print("Обучение модели general")
    print("=" * 80)

    # === 1. Параметры обучения ===
    expert_id = "general"
    train_file = project_root / "data" / "training" / "simple_dialogues.jsonl"
    models_dir = project_root / "models" / "experts"
    expert_dir = models_dir / expert_id
    checkpoints_dir = project_root / "checkpoints" / expert_id

    # Гиперпараметры
    vocab_size = 5000  # Увеличим словарь для лучшего покрытия
    d_model = 256
    n_layers = 2
    n_heads = 4
    d_ff = 1024
    max_seq_len = 128  # Уменьшим для экономии памяти

    num_epochs = 10
    batch_size = 8
    learning_rate = 5e-4
    device = "cpu"  # Используем CPU

    print(f"\nКонфигурация:")
    print(f"  Датасет: {train_file}")
    print(f"  Модель: {expert_id}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Model size: d_model={d_model}, n_layers={n_layers}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")

    # === 2. Создание токенайзера и dataloader ===
    print("\n" + "=" * 80)
    print("Загрузка датасета и создание токенайзера")
    print("=" * 80)

    train_loader, val_loader, tokenizer = create_dataloaders(
        train_file=str(train_file),
        val_file=None,  # Пока без валидации
        tokenizer=None,  # Создаётся автоматически
        batch_size=batch_size,
        max_length=max_seq_len,
        num_workers=0
    )

    actual_vocab_size = len(tokenizer)
    print(f"\n✅ Датасет загружен!")
    print(f"   Реальный размер словаря: {actual_vocab_size}")
    print(f"   Training batches: {len(train_loader)}")

    # === 3. Создание модели ===
    print("\n" + "=" * 80)
    print("Создание модели ExpertModel")
    print("=" * 80)

    model = ExpertModel(
        vocab_size=actual_vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=0.1
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Модель создана!")
    print(f"   Параметры: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)")

    # === 4. Создание Trainer ===
    print("\n" + "=" * 80)
    print("Инициализация Trainer")
    print("=" * 80)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        device=device,
        gradient_accumulation_steps=1,
        checkpoint_dir=str(checkpoints_dir),
        log_interval=5
    )

    print("\n✅ Trainer готов к обучению!")

    # === 5. Обучение ===
    print("\n" + "=" * 80)
    print("Начало обучения")
    print("=" * 80)

    try:
        history = trainer.train(
            num_epochs=num_epochs,
            save_every=2,  # Сохраняем каждые 2 эпохи
            early_stopping_patience=None  # Без early stopping
        )

        print("\n" + "=" * 80)
        print("✅ Обучение завершено!")
        print("=" * 80)

        # Выводим финальные метрики
        print("\nФинальные метрики:")
        print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final Train Perplexity: {history['train_perplexity'][-1]:.2f}")

    except KeyboardInterrupt:
        print("\n\n⚠️ Обучение прервано пользователем!")
    except Exception as e:
        print(f"\n\n❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # === 6. Сохранение финальной модели ===
    print("\n" + "=" * 80)
    print("Сохранение обученной модели")
    print("=" * 80)

    # Создаём директорию если не существует
    expert_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем модель
    model_path = expert_dir / "model.pt"

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': actual_vocab_size,
            'd_model': d_model,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'd_ff': d_ff,
            'max_seq_len': max_seq_len,
            'dropout': 0.1
        },
        'expert_id': expert_id,
        'training_history': history
    }

    torch.save(checkpoint, model_path)
    print(f"\n✅ Модель сохранена: {model_path}")

    # Сохраняем токенайзер
    tokenizer_path = expert_dir / "tokenizer.json"
    import json

    tokenizer_data = {
        'vocab_size': actual_vocab_size,
        'word2idx': tokenizer.word2idx,
        'idx2word': tokenizer.idx2word,
        'pad_token_id': tokenizer.pad_token_id,
        'unk_token_id': tokenizer.unk_token_id,
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id
    }

    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Токенайзер сохранён: {tokenizer_path}")

    # Обновляем metadata.json
    metadata_path = expert_dir / "metadata.json"

    metadata = {
        "expert_id": expert_id,
        "name": "General Expert",
        "version": "0.2.0-trained",
        "type": "trained_model",
        "architecture": {
            "vocab_size": actual_vocab_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "max_seq_len": max_seq_len
        },
        "parameters": {
            "total": total_params,
            "trainable": total_params
        },
        "metrics": {
            "final_train_loss": history['train_loss'][-1],
            "final_train_perplexity": history['train_perplexity'][-1],
            "epochs_trained": num_epochs
        },
        "description": f"General Expert trained on {len(train_loader) * batch_size * num_epochs} examples",
        "created_by": "scripts/train_simple.py"
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ Metadata обновлён: {metadata_path}")

    print("\n" + "=" * 80)
    print("🎉 Всё готово! Модель обучена и сохранена.")
    print("=" * 80)
    print(f"\nМожно протестировать через UI: http://localhost:7860")
    print(f"Выберите эксперта 'general' в Manual режиме.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
