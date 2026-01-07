"""
Примеры использования Training Pipeline для обучения ExpertModel.

Демонстрирует:
    1. Подготовка данных и создание DataLoaders
    2. Создание и конфигурация модели
    3. Базовое обучение
    4. Обучение с validation
    5. Обучение с early stopping
    6. Сохранение и загрузка checkpoints
    7. Продолжение обучения с checkpoint
"""

import sys
sys.path.append('src/python')

from models.expert import ExpertModel
from training.dataset import create_dataloaders, SimpleTokenizer
from training.trainer import Trainer
import torch
from pathlib import Path


# ============================================================================
# Пример 1: Базовое обучение модели
# ============================================================================

def example_1_basic_training():
    """Базовое обучение ExpertModel на текстовых данных."""
    print("\n" + "=" * 80)
    print("Пример 1: Базовое обучение модели")
    print("=" * 80)

    # Создаём тестовый dataset
    train_file = "/tmp/example_train.txt"
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write("""
Искусственный интеллект меняет мир вокруг нас.
Машинное обучение позволяет компьютерам учиться на данных.

Нейронные сети моделируют работу человеческого мозга.
Глубокое обучение использует многослойные сети.

Трансформеры revolutionized обработку естественного языка.
Attention механизм позволяет модели фокусироваться на важном.
""")

    print("\n1. Создание DataLoaders...")
    train_loader, _, tokenizer = create_dataloaders(
        train_file=train_file,
        batch_size=2,
        max_length=64
    )

    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   Train samples: {len(train_loader.dataset)}")

    print("\n2. Создание модели (Small config)...")
    model = ExpertModel(
        vocab_size=len(tokenizer),
        d_model=256,
        n_layers=4,
        n_heads=8,
        d_ff=1024,
        max_seq_len=64,
        dropout=0.1
    )
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n3. Создание Trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        device='cpu',
        checkpoint_dir='checkpoints/example1',
        log_interval=2
    )

    print("\n4. Обучение на 3 эпохах...")
    history = trainer.train(num_epochs=3, save_every=1)

    print("\n5. Результаты:")
    print(f"   Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"   Final perplexity: {history['train_perplexity'][-1]:.2f}")


# ============================================================================
# Пример 2: Обучение с Validation
# ============================================================================

def example_2_training_with_validation():
    """Обучение с использованием validation set."""
    print("\n" + "=" * 80)
    print("Пример 2: Обучение с Validation")
    print("=" * 80)

    # Создаём train и validation datasets
    train_file = "/tmp/example_train2.txt"
    val_file = "/tmp/example_val2.txt"

    with open(train_file, 'w', encoding='utf-8') as f:
        f.write("""
Python отличный язык для машинного обучения.
PyTorch популярный фреймворк для deep learning.
Transformers библиотека от Hugging Face очень полезна.
""")

    with open(val_file, 'w', encoding='utf-8') as f:
        f.write("""
NumPy используется для численных вычислений.
TensorFlow альтернатива PyTorch.
""")

    print("\n1. Создание DataLoaders с validation...")
    train_loader, val_loader, tokenizer = create_dataloaders(
        train_file=train_file,
        val_file=val_file,
        batch_size=1,
        max_length=32
    )

    print(f"   Train samples: {len(train_loader.dataset)}")
    print(f"   Val samples: {len(val_loader.dataset)}")

    print("\n2. Создание модели (Tiny config)...")
    model = ExpertModel(
        vocab_size=len(tokenizer),
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        max_seq_len=32
    )

    print("\n3. Trainer с validation...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,  # Добавляем validation
        device='cpu',
        checkpoint_dir='checkpoints/example2'
    )

    print("\n4. Обучение...")
    history = trainer.train(num_epochs=5)

    print("\n5. Сравнение train vs validation:")
    for i, (train_loss, val_loss) in enumerate(zip(
        history['train_loss'],
        history['val_loss']
    ), 1):
        print(f"   Epoch {i}: Train={train_loss:.4f}, Val={val_loss:.4f}")


# ============================================================================
# Пример 3: Early Stopping
# ============================================================================

def example_3_early_stopping():
    """Обучение с early stopping для предотвращения overfitting."""
    print("\n" + "=" * 80)
    print("Пример 3: Early Stopping")
    print("=" * 80)

    # Используем те же данные
    train_file = "/tmp/example_train3.txt"
    val_file = "/tmp/example_val3.txt"

    with open(train_file, 'w', encoding='utf-8') as f:
        f.write("Текст для обучения. " * 20)

    with open(val_file, 'w', encoding='utf-8') as f:
        f.write("Текст для валидации. " * 10)

    train_loader, val_loader, tokenizer = create_dataloaders(
        train_file=train_file,
        val_file=val_file,
        batch_size=2,
        max_length=32
    )

    model = ExpertModel(
        vocab_size=len(tokenizer),
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        max_seq_len=32
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device='cpu',
        checkpoint_dir='checkpoints/example3'
    )

    print("\n1. Обучение с early stopping (patience=2)...")
    history = trainer.train(
        num_epochs=20,  # Много эпох
        early_stopping_patience=2  # Остановимся, если нет улучшения 2 эпохи
    )

    print(f"\n2. Остановились на эпохе: {trainer.current_epoch}")
    print(f"   Best val loss: {trainer.best_val_loss:.4f}")


# ============================================================================
# Пример 4: Сохранение и загрузка checkpoints
# ============================================================================

def example_4_checkpoint_management():
    """Работа с checkpoints: сохранение и загрузка."""
    print("\n" + "=" * 80)
    print("Пример 4: Checkpoint Management")
    print("=" * 80)

    train_file = "/tmp/example_train4.txt"
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write("Пример текста для обучения модели. " * 30)

    train_loader, _, tokenizer = create_dataloaders(
        train_file=train_file,
        batch_size=2,
        max_length=32
    )

    model = ExpertModel(
        vocab_size=len(tokenizer),
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        max_seq_len=32
    )

    print("\n1. Первое обучение (2 эпохи)...")
    trainer1 = Trainer(
        model=model,
        train_dataloader=train_loader,
        device='cpu',
        checkpoint_dir='checkpoints/example4'
    )
    history1 = trainer1.train(num_epochs=2)

    print(f"\n   Loss после 2 эпох: {history1['train_loss'][-1]:.4f}")

    print("\n2. Сохранение checkpoint...")
    checkpoint_path = Path('checkpoints/example4/manual_checkpoint.pt')
    trainer1.save_checkpoint(checkpoint_path)

    print("\n3. Создание новой модели и загрузка checkpoint...")
    new_model = ExpertModel(
        vocab_size=len(tokenizer),
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        max_seq_len=32
    )

    trainer2 = Trainer(
        model=new_model,
        train_dataloader=train_loader,
        device='cpu',
        checkpoint_dir='checkpoints/example4'
    )

    trainer2.load_checkpoint(checkpoint_path)

    print(f"   Loaded epoch: {trainer2.current_epoch}")
    print(f"   Loaded history: {trainer2.history['train_loss']}")

    print("\n4. Продолжение обучения ещё 2 эпохи...")
    history2 = trainer2.train(num_epochs=2)  # Продолжит с эпохи 3

    print(f"\n   Final epoch: {trainer2.current_epoch}")
    print(f"   Final loss: {history2['train_loss'][-1]:.4f}")


# ============================================================================
# Пример 5: Кастомизация Trainer
# ============================================================================

def example_5_custom_configuration():
    """Настройка Trainer с кастомными параметрами."""
    print("\n" + "=" * 80)
    print("Пример 5: Кастомная конфигурация Trainer")
    print("=" * 80)

    train_file = "/tmp/example_train5.txt"
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write("Обучение с кастомными параметрами. " * 40)

    train_loader, _, tokenizer = create_dataloaders(
        train_file=train_file,
        batch_size=4,
        max_length=64
    )

    model = ExpertModel(
        vocab_size=len(tokenizer),
        d_model=256,
        n_layers=4,
        n_heads=8,
        d_ff=1024,
        max_seq_len=64
    )

    print("\n1. Создание кастомного оптимизатора...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,  # Более высокий learning rate
        betas=(0.9, 0.98),
        weight_decay=0.01
    )

    print("\n2. Trainer с кастомными параметрами...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        optimizer=optimizer,  # Кастомный оптимизатор
        device='cpu',
        gradient_accumulation_steps=4,  # Gradient accumulation
        max_grad_norm=0.5,  # Gradient clipping
        checkpoint_dir='checkpoints/example5',
        log_interval=1
    )

    print("\n3. Обучение...")
    history = trainer.train(num_epochs=3)

    print(f"\n4. Результаты:")
    print(f"   Final loss: {history['train_loss'][-1]:.4f}")


# ============================================================================
# Пример 6: Сохранение истории обучения
# ============================================================================

def example_6_save_training_history():
    """Сохранение истории обучения для анализа."""
    print("\n" + "=" * 80)
    print("Пример 6: Сохранение истории обучения")
    print("=" * 80)

    train_file = "/tmp/example_train6.txt"
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write("История обучения сохраняется автоматически. " * 25)

    train_loader, _, tokenizer = create_dataloaders(
        train_file=train_file,
        batch_size=2,
        max_length=32
    )

    model = ExpertModel(
        vocab_size=len(tokenizer),
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        max_seq_len=32
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        device='cpu',
        checkpoint_dir='checkpoints/example6'
    )

    print("\n1. Обучение...")
    history = trainer.train(num_epochs=3)

    print("\n2. Сохранение истории в JSON...")
    history_path = Path('checkpoints/example6/training_history.json')
    trainer.save_history(history_path)

    print(f"\n3. История сохранена в: {history_path}")
    print("   Можно использовать для визуализации или анализа")


# ============================================================================
# Запуск всех примеров
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ TRAINING PIPELINE")
    print("=" * 80)

    # Запускаем все примеры
    example_1_basic_training()
    example_2_training_with_validation()
    example_3_early_stopping()
    example_4_checkpoint_management()
    example_5_custom_configuration()
    example_6_save_training_history()

    print("\n" + "=" * 80)
    print("✅ ВСЕ ПРИМЕРЫ ВЫПОЛНЕНЫ!")
    print("=" * 80)
    print("\nCheckpoints сохранены в: checkpoints/example*/")
    print("Используйте их для inference или продолжения обучения!")
