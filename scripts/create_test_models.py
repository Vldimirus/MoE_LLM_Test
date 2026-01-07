#!/usr/bin/env python3
"""
Скрипт для создания тестовых моделей (с random weights).

Создаёт маленькие ExpertModel для быстрого прототипирования:
- general (общий ассистент)
- python_expert (эксперт по Python)
- math_expert (математический эксперт)

Usage:
    python scripts/create_test_models.py
"""

import sys
import json
import torch
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))

from models.expert import ExpertModel
from training.dataset import SimpleTokenizer


def create_test_model(
    expert_id: str,
    name: str,
    vocab_size: int = 1000,
    d_model: int = 256,
    n_layers: int = 2,
    n_heads: int = 4,
    max_seq_len: int = 512
):
    """
    Создаёт тестовую модель с random weights.
    
    Args:
        expert_id: ID эксперта (например, "general")
        name: Имя эксперта (например, "General Assistant")
        vocab_size: Размер словаря
        d_model: Размерность модели
        n_layers: Количество слоёв
        n_heads: Количество attention heads
        max_seq_len: Максимальная длина последовательности
    """
    print(f"\n{'='*60}")
    print(f"Создание модели: {name}")
    print(f"{'='*60}")
    
    # Директория для модели
    model_dir = Path("models") / "experts" / expert_id
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Создаём модель
    print(f"→ Инициализация ExpertModel...")
    print(f"  d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}")
    
    model = ExpertModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_model * 4,
        max_seq_len=max_seq_len,
        dropout=0.1
    )
    
    # Подсчёт параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Модель создана")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # 2. Сохраняем модель
    model_path = model_dir / "model.pt"
    print(f"\n→ Сохранение модели: {model_path}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'd_ff': d_model * 4,
            'max_seq_len': max_seq_len,
            'dropout': 0.1
        },
        'expert_id': expert_id,
        'expert_name': name
    }, model_path)
    
    print(f"✓ Модель сохранена: {model_path}")
    
    # 3. Создаём metadata
    metadata = {
        "expert_id": expert_id,
        "name": name,
        "version": "0.1.0-test",
        "type": "test_model",
        "architecture": {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_ff": d_model * 4,
            "max_seq_len": max_seq_len
        },
        "parameters": {
            "total": total_params,
            "trainable": trainable_params
        },
        "metrics": {
            "perplexity": None,
            "accuracy": None,
            "note": "Random weights - not trained"
        },
        "description": f"Test model for {name}. Random initialized weights.",
        "created_by": "scripts/create_test_models.py"
    }
    
    metadata_path = model_dir / "metadata.json"
    print(f"\n→ Сохранение metadata: {metadata_path}")
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Metadata сохранён")
    
    # 4. Создаём README
    readme_content = f"""# {name}

**Expert ID:** `{expert_id}`
**Version:** 0.1.0-test
**Type:** Test Model (Random Weights)

## Architecture

- **Model Type:** Transformer-based Expert
- **Parameters:** {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)
- **Layers:** {n_layers}
- **Hidden Size:** {d_model}
- **Attention Heads:** {n_heads}
- **Vocabulary Size:** {vocab_size}
- **Max Sequence Length:** {max_seq_len}

## Status

⚠️ **This is a TEST model with random weights.**

It has NOT been trained and will generate random/nonsensical output.
This model is for testing infrastructure and integration only.

## Usage

```python
from models.expert import ExpertModel

# Load model
checkpoint = torch.load("models/experts/{expert_id}/model.pt")
model = ExpertModel(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])

# Generate (will be random!)
output = model.generate(input_ids, max_length=50)
```

## Training

To train this model, use:

```bash
python scripts/train_expert.py --expert-id {expert_id} --data path/to/data.jsonl
```
"""
    
    readme_path = model_dir / "README.md"
    print(f"\n→ Создание README: {readme_path}")
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✓ README создан")
    print(f"\n{'='*60}")
    print(f"✓ Модель {expert_id} успешно создана!")
    print(f"{'='*60}")


def main():
    """Создаёт 3 тестовые модели."""
    print("\n" + "="*60)
    print("Создание тестовых моделей для MoE System")
    print("="*60)
    
    # Конфигурации для разных экспертов
    experts = [
        {
            'expert_id': 'general',
            'name': 'General Assistant',
            'vocab_size': 1000,
            'd_model': 256,
            'n_layers': 2,
            'n_heads': 4,
            'max_seq_len': 512
        },
        {
            'expert_id': 'python_expert',
            'name': 'Python Programming Expert',
            'vocab_size': 1000,
            'd_model': 256,
            'n_layers': 3,
            'n_heads': 4,
            'max_seq_len': 512
        },
        {
            'expert_id': 'math_expert',
            'name': 'Mathematics Expert',
            'vocab_size': 1000,
            'd_model': 256,
            'n_layers': 2,
            'n_heads': 4,
            'max_seq_len': 512
        }
    ]
    
    # Создаём модели
    for expert_config in experts:
        create_test_model(**expert_config)
    
    print("\n" + "="*60)
    print("✓ Все тестовые модели созданы успешно!")
    print("="*60)
    print("\nМодели сохранены в:")
    print("  models/experts/general/")
    print("  models/experts/python_expert/")
    print("  models/experts/math_expert/")
    print("\nСледующий шаг: интегрировать загрузку в MoESystem")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
