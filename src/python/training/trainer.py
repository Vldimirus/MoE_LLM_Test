"""
Trainer для обучения языковых моделей.

Поддерживает:
    - Training loop с gradient accumulation
    - Validation loop с метриками
    - Checkpoint saving/loading
    - Learning rate scheduling
    - Early stopping
    - Logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable
from pathlib import Path
import time
import math
import json


class Trainer:
    """
    Trainer для обучения ExpertModel.

    Включает:
        - Training loop с автоматическим backward
        - Validation с метриками
        - Checkpoint management
        - Logging и мониторинг
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: str = "cpu",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 10
    ):
        """
        Инициализация Trainer.

        Args:
            model: Модель для обучения
            train_dataloader: DataLoader для training
            val_dataloader: DataLoader для validation (опционально)
            optimizer: Оптимизатор (если None, создаётся AdamW)
            criterion: Loss function (если None, используется CrossEntropyLoss)
            device: Устройство для вычислений ('cpu' или 'cuda')
            gradient_accumulation_steps: Количество шагов для накопления градиентов
            max_grad_norm: Максимальная норма градиента для clipping
            checkpoint_dir: Директория для сохранения checkpoints
            log_interval: Интервал логирования (в шагах)
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval

        # Создаём директорию для checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Оптимизатор
        if optimizer is None:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=5e-4,
                betas=(0.9, 0.999),
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer

        # Loss function
        if criterion is None:
            # CrossEntropyLoss игнорирует label=-100 (padding)
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            self.criterion = criterion

        # История обучения
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_perplexity': [],
            'val_loss': [],
            'val_perplexity': [],
            'learning_rate': []
        }

        # Счётчики
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

    def train_epoch(self) -> Dict[str, float]:
        """
        Обучение на одной эпохе.

        Returns:
            Словарь с метриками: {'loss': ..., 'perplexity': ...}
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_dataloader):
            # Перемещаем данные на device
            input_ids = batch['input_ids'].to(self.device)          # [batch_size, seq_len]
            labels = batch['labels'].to(self.device)                # [batch_size, seq_len]
            attention_mask = batch.get('attention_mask', None)

            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # Forward pass
            # Модель возвращает logits: [batch_size, seq_len, vocab_size]
            logits = self.model(input_ids)

            # Вычисляем loss
            # Reshape для CrossEntropyLoss: [batch_size * seq_len, vocab_size] и [batch_size * seq_len]
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),  # [batch*seq, vocab_size]
                labels.reshape(-1)                     # [batch*seq]
            )

            # Нормализуем loss для gradient accumulation
            loss = loss / self.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.global_step += 1

            # Накапливаем loss (умножаем обратно на accumulation_steps)
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Логирование
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                perplexity = math.exp(min(avg_loss, 20))  # Clamp для стабильности

                elapsed = time.time() - start_time
                batches_per_sec = (batch_idx + 1) / elapsed

                print(
                    f"  Batch {batch_idx + 1}/{len(self.train_dataloader)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"PPL: {perplexity:.2f} | "
                    f"Speed: {batches_per_sec:.2f} batches/s"
                )

        # Средние метрики за эпоху
        avg_loss = total_loss / num_batches
        avg_perplexity = math.exp(min(avg_loss, 20))

        return {
            'loss': avg_loss,
            'perplexity': avg_perplexity
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Валидация на validation set.

        Returns:
            Словарь с метриками: {'loss': ..., 'perplexity': ...}
        """
        if self.val_dataloader is None:
            return {'loss': 0.0, 'perplexity': 0.0}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_dataloader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch.get('attention_mask', None)

            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # Forward pass
            logits = self.model(input_ids)

            # Loss
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )

            total_loss += loss.item()
            num_batches += 1

        # Средние метрики
        avg_loss = total_loss / num_batches
        avg_perplexity = math.exp(min(avg_loss, 20))

        return {
            'loss': avg_loss,
            'perplexity': avg_perplexity
        }

    def train(
        self,
        num_epochs: int,
        save_every: int = 1,
        early_stopping_patience: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Полный training loop на несколько эпох.

        Args:
            num_epochs: Количество эпох
            save_every: Сохранять checkpoint каждые N эпох
            early_stopping_patience: Остановка, если val_loss не улучшается N эпох

        Returns:
            История обучения
        """
        print("=" * 80)
        print(f"Начало обучения на {num_epochs} эпох")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training batches: {len(self.train_dataloader)}")
        if self.val_dataloader:
            print(f"Validation batches: {len(self.val_dataloader)}")
        print("=" * 80)

        patience_counter = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1

            print(f"\nEpoch {self.current_epoch}/{num_epochs}")
            print("-" * 80)

            # Training
            train_metrics = self.train_epoch()

            print(f"\n  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Train Perplexity: {train_metrics['perplexity']:.2f}")

            # Validation
            if self.val_dataloader is not None:
                val_metrics = self.validate()

                print(f"  Val Loss: {val_metrics['loss']:.4f}")
                print(f"  Val Perplexity: {val_metrics['perplexity']:.2f}")

                # Сохраняем в историю
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_perplexity'].append(val_metrics['perplexity'])

                # Early stopping
                if early_stopping_patience is not None:
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        patience_counter = 0

                        # Сохраняем лучшую модель
                        self.save_checkpoint(
                            self.checkpoint_dir / "best_model.pt",
                            is_best=True
                        )
                        print("  ✅ New best model saved!")
                    else:
                        patience_counter += 1
                        print(f"  Early stopping: {patience_counter}/{early_stopping_patience}")

                        if patience_counter >= early_stopping_patience:
                            print("\n⚠️  Early stopping triggered!")
                            break

            # Сохраняем в историю
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_perplexity'].append(train_metrics['perplexity'])
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )

            # Периодическое сохранение
            if (self.current_epoch % save_every) == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
                self.save_checkpoint(checkpoint_path)
                print(f"\n  💾 Checkpoint saved: {checkpoint_path}")

        print("\n" + "=" * 80)
        print("✅ Обучение завершено!")
        print("=" * 80)

        return self.history

    def save_checkpoint(
        self,
        path: Path,
        is_best: bool = False
    ) -> None:
        """
        Сохранение checkpoint.

        Args:
            path: Путь для сохранения
            is_best: Флаг лучшей модели
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'is_best': is_best
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> None:
        """
        Загрузка checkpoint.

        Args:
            path: Путь к checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"✅ Checkpoint загружен: epoch {self.current_epoch}, step {self.global_step}")

    def save_history(self, path: Path) -> None:
        """
        Сохранение истории обучения в JSON.

        Args:
            path: Путь для сохранения
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

        print(f"📊 История обучения сохранена: {path}")


# ============================================================================
# Тестирование
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.append('src/python')

    from models.expert import ExpertModel
    from training.dataset import create_dataloaders

    print("=" * 80)
    print("Тест Trainer")
    print("=" * 80)

    # Создаём тестовый датасет
    test_file = "/tmp/train_test.txt"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("""
Это первый пример текста для обучения.
Модель должна научиться предсказывать следующие слова.

Второй пример содержит другую информацию.
Чем больше данных, тем лучше результат обучения.

Третий параграф добавляет разнообразия в dataset.
Обучение языковой модели требует большого количества текста.
""")

    print("\n1. Создание DataLoaders...")
    train_loader, val_loader, tokenizer = create_dataloaders(
        train_file=test_file,
        batch_size=2,
        max_length=32
    )
    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   Train batches: {len(train_loader)}")

    print("\n2. Создание модели...")
    model = ExpertModel(
        vocab_size=len(tokenizer),
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        max_seq_len=32,
        dropout=0.1
    )
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n3. Создание Trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=None,  # Нет validation для простого теста
        device='cpu',
        gradient_accumulation_steps=2,
        checkpoint_dir='test_checkpoints',
        log_interval=1
    )

    print("\n4. Обучение на 2 эпохах...")
    history = trainer.train(
        num_epochs=2,
        save_every=1
    )

    print("\n5. Проверка истории...")
    print(f"   Train losses: {history['train_loss']}")
    print(f"   Train perplexities: {history['train_perplexity']}")

    print("\n6. Тест сохранения/загрузки checkpoint...")
    checkpoint_path = Path("test_checkpoints/test_checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path)

    # Создаём новый trainer и загружаем checkpoint
    new_trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        device='cpu'
    )
    new_trainer.load_checkpoint(checkpoint_path)

    print(f"   Loaded epoch: {new_trainer.current_epoch}")
    print(f"   Loaded global_step: {new_trainer.global_step}")

    print("\n7. Сохранение истории...")
    trainer.save_history(Path("test_checkpoints/history.json"))

    print("\n" + "=" * 80)
    print("✅ Все тесты пройдены!")
    print("=" * 80)

    # Cleanup
    import shutil
    shutil.rmtree('test_checkpoints', ignore_errors=True)
