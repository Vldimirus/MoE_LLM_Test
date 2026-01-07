#!/usr/bin/env python3
"""
Обучение модели general с новым BPE токенайзером.

Использует:
    - BPE токенайзер: models/tokenizers/bpe_multilang.model
    - Датасет: data/training/corpus_for_bpe.txt
    - Vocab size: 8000 (BPE)
    - Модель: ExpertModel с обновлёнными параметрами
"""

import sys
from pathlib import Path

# Добавляем src/python в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src" / "python"))

import torch
from torch.utils.data import Dataset, DataLoader
from models.expert import ExpertModel
from training.bpe_tokenizer import BPETokenizer
from training.trainer import Trainer
from typing import List, Dict


class BPETextDataset(Dataset):
    """
    Dataset для обучения с BPE токенайзером.

    Аналог TextDataset, но использует BPETokenizer вместо SimpleTokenizer.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: BPETokenizer,
        max_length: int = 128,
        stride: int = 64
    ):
        """
        Инициализация dataset.

        Args:
            file_path: Путь к текстовому файлу (.txt)
            tokenizer: BPE токенайзер
            max_length: Максимальная длина последовательности
            stride: Шаг для sliding window
        """
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.samples: List[List[int]] = []

        # Загружаем данные
        self._load_data()

    def _load_data(self) -> None:
        """Загрузка и preprocessing данных."""
        print(f"Загрузка данных из {self.file_path}...")

        # Читаем текстовый файл
        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        print(f"Загружено {len(lines)} строк текста")

        # Tokenization и создание samples
        for text in lines:
            # Кодируем текст с BPE
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=None,
                padding=False,
                truncation=False
            )

            # Если текст длиннее max_length, разбиваем на chunks с overlap
            if len(tokens) > self.max_length:
                for i in range(0, len(tokens) - self.max_length + 1, self.stride):
                    chunk = tokens[i:i + self.max_length]
                    self.samples.append(chunk)
            else:
                self.samples.append(tokens)

        print(f"Создано {len(self.samples)} samples для обучения")

    def __len__(self) -> int:
        """Количество samples в dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Получение одного sample.

        Returns:
            Словарь с input_ids и labels
        """
        tokens = self.samples[idx]

        # Для языкового моделирования: input = tokens[:-1], target = tokens[1:]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': labels
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Collate function для DataLoader с BPE токенами.

    Args:
        batch: Список samples
        pad_token_id: ID токена для padding

    Returns:
        Batch данных с padding
    """
    # Находим максимальную длину в batch
    max_len = max(len(sample['input_ids']) for sample in batch)

    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []

    for sample in batch:
        input_ids = sample['input_ids']
        labels = sample['labels']

        # Padding
        padding_len = max_len - len(input_ids)

        # Добавляем padding справа
        padded_input = torch.cat([
            input_ids,
            torch.full((padding_len,), pad_token_id, dtype=torch.long)
        ])

        padded_labels = torch.cat([
            labels,
            torch.full((padding_len,), -100, dtype=torch.long)  # -100 игнорируется в CrossEntropyLoss
        ])

        # Attention mask
        attention_mask = torch.cat([
            torch.ones(len(input_ids), dtype=torch.long),
            torch.zeros(padding_len, dtype=torch.long)
        ])

        batch_input_ids.append(padded_input)
        batch_labels.append(padded_labels)
        batch_attention_mask.append(attention_mask)

    return {
        'input_ids': torch.stack(batch_input_ids),
        'labels': torch.stack(batch_labels),
        'attention_mask': torch.stack(batch_attention_mask)
    }


def main():
    print("=" * 80)
    print("Обучение модели general с BPE токенайзером")
    print("=" * 80)

    # === 1. Параметры обучения ===
    expert_id = "general"
    tokenizer_path = project_root / "models" / "tokenizers" / "bpe_multilang.model"
    train_file = project_root / "data" / "training" / "corpus_for_bpe.txt"
    models_dir = project_root / "models" / "experts"
    expert_dir = models_dir / expert_id
    checkpoints_dir = project_root / "checkpoints" / f"{expert_id}_bpe"

    # Гиперпараметры модели
    vocab_size = 8000  # BPE vocab size
    d_model = 256
    n_layers = 4  # Увеличиваем с 2 до 4 для лучшего качества
    n_heads = 8    # Увеличиваем с 4 до 8
    d_ff = 1024
    max_seq_len = 128

    # Гиперпараметры обучения
    num_epochs = 15  # Увеличиваем до 15 эпох
    batch_size = 8
    learning_rate = 3e-4  # Немного уменьшаем для стабильности
    device = "cpu"

    print(f"\nКонфигурация:")
    print(f"  Токенайзер: {tokenizer_path}")
    print(f"  Датасет: {train_file}")
    print(f"  Модель: {expert_id}")
    print(f"  Vocab size: {vocab_size} (BPE)")
    print(f"  Model size: d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")

    # === 2. Загрузка BPE токенайзера ===
    print("\n" + "=" * 80)
    print("Загрузка BPE токенайзера")
    print("=" * 80)

    tokenizer = BPETokenizer(str(tokenizer_path))

    print(f"\n✅ BPE токенайзер загружен!")
    print(f"   Vocabulary size: {len(tokenizer)}")

    # === 3. Создание dataset и dataloader ===
    print("\n" + "=" * 80)
    print("Создание dataset")
    print("=" * 80)

    train_dataset = BPETextDataset(
        file_path=str(train_file),
        tokenizer=tokenizer,
        max_length=max_seq_len,
        stride=max_seq_len // 2
    )

    # Collate function с правильным pad_token_id
    collate_with_pad = lambda batch: collate_fn(batch, pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_with_pad,
        num_workers=0
    )

    print(f"\n✅ Dataset создан!")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Training batches: {len(train_loader)}")

    # === 4. Создание модели ===
    print("\n" + "=" * 80)
    print("Создание модели ExpertModel")
    print("=" * 80)

    model = ExpertModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=0.1
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n✅ Модель создана!")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB")

    # === 5. Создание Trainer ===
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
        val_dataloader=None,  # Без validation для упрощения
        optimizer=optimizer,
        device=device,
        gradient_accumulation_steps=1,
        checkpoint_dir=str(checkpoints_dir),
        log_interval=5
    )

    print("\n✅ Trainer готов к обучению!")

    # === 6. Обучение ===
    print("\n" + "=" * 80)
    print("Начало обучения")
    print("=" * 80)

    try:
        history = trainer.train(
            num_epochs=num_epochs,
            save_every=3,  # Сохраняем каждые 3 эпохи
            early_stopping_patience=None
        )

        print("\n" + "=" * 80)
        print("✅ Обучение завершено!")
        print("=" * 80)

        # Выводим финальные метрики
        print("\nФинальные метрики:")
        print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final Train Perplexity: {history['train_perplexity'][-1]:.2f}")

        # Проверяем улучшение
        if len(history['train_loss']) > 1:
            initial_loss = history['train_loss'][0]
            final_loss = history['train_loss'][-1]
            improvement = ((initial_loss - final_loss) / initial_loss) * 100
            print(f"  Improvement: {improvement:.1f}% (loss снизился с {initial_loss:.4f} до {final_loss:.4f})")

    except KeyboardInterrupt:
        print("\n\n⚠️  Обучение прервано пользователем!")
    except Exception as e:
        print(f"\n\n❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # === 7. Сохранение финальной модели ===
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
            'vocab_size': vocab_size,
            'd_model': d_model,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'd_ff': d_ff,
            'max_seq_len': max_seq_len,
            'dropout': 0.1
        },
        'expert_id': expert_id,
        'training_history': history,
        'tokenizer_type': 'bpe',
        'tokenizer_path': str(tokenizer_path)
    }

    torch.save(checkpoint, model_path)
    print(f"\n✅ Модель сохранена: {model_path}")

    # Сохраняем конфигурацию токенайзера (ссылка на BPE модель)
    import json

    tokenizer_config_path = expert_dir / "tokenizer_config.json"

    tokenizer_config = {
        'tokenizer_type': 'bpe',
        'model_path': str(tokenizer_path.relative_to(project_root)),
        'vocab_size': vocab_size,
        'pad_token_id': tokenizer.pad_token_id,
        'unk_token_id': tokenizer.unk_token_id,
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id
    }

    with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

    print(f"✅ Конфигурация токенайзера сохранена: {tokenizer_config_path}")

    # Обновляем metadata.json
    metadata_path = expert_dir / "metadata.json"

    metadata = {
        "expert_id": expert_id,
        "name": "General Expert (BPE)",
        "version": "0.3.0-bpe",
        "type": "trained_model",
        "tokenizer": {
            "type": "bpe",
            "vocab_size": vocab_size,
            "supports_typos": True,
            "supports_multilingual": True
        },
        "architecture": {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "max_seq_len": max_seq_len
        },
        "parameters": {
            "total": total_params,
            "trainable": trainable_params
        },
        "metrics": {
            "final_train_loss": history['train_loss'][-1],
            "final_train_perplexity": history['train_perplexity'][-1],
            "epochs_trained": num_epochs
        },
        "training": {
            "dataset": str(train_file.relative_to(project_root)),
            "samples": len(train_dataset),
            "batch_size": batch_size,
            "learning_rate": learning_rate
        },
        "description": f"General Expert trained with BPE tokenizer on {len(train_dataset)} samples. Supports typos and multilingual text.",
        "created_by": "scripts/train_bpe.py"
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ Metadata обновлён: {metadata_path}")

    # Сохраняем историю обучения
    history_path = expert_dir / "training_history.json"
    trainer.save_history(history_path)

    print("\n" + "=" * 80)
    print("🎉 Всё готово! Модель обучена и сохранена с BPE токенайзером.")
    print("=" * 80)
    print(f"\nОсобенности новой модели:")
    print(f"  ✅ BPE токенайзер (8000 токенов)")
    print(f"  ✅ Поддержка опечаток")
    print(f"  ✅ Мультиязычность (русский + английский)")
    print(f"  ✅ Byte-level fallback для любых символов")
    print(f"\nМожно протестировать через UI: http://localhost:7860")
    print(f"Выберите эксперта 'general' в Manual режиме.")

    # === 8. Быстрый тест генерации ===
    print("\n" + "=" * 80)
    print("Тест генерации текста")
    print("=" * 80)

    model.eval()

    test_prompts = [
        "Привет, как дела?",
        "Hello, how are you?",
        "првиет кк дела",  # С опечатками
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")

        # Кодируем prompt
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        # Генерируем 10 токенов
        with torch.no_grad():
            for _ in range(10):
                logits = model(input_tensor)  # [1, seq_len, vocab_size]
                next_token_logits = logits[0, -1, :]  # [vocab_size]
                next_token = torch.argmax(next_token_logits).item()

                # Добавляем токен к последовательности
                input_tensor = torch.cat([
                    input_tensor,
                    torch.tensor([[next_token]], dtype=torch.long).to(device)
                ], dim=1)

                # Останавливаемся на EOS
                if next_token == tokenizer.eos_token_id:
                    break

        # Декодируем результат
        generated_ids = input_tensor[0].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(f"Generated: '{generated_text}'")

    print("\n" + "=" * 80)
    print("✅ Тесты завершены!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
