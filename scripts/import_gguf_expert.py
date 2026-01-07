#!/usr/bin/env python3
"""
Скрипт для создания expert модели из GGUF файла.

Использует Transfer Learning Pipeline для переноса знаний из GGUF модели
(Phi-3, Llama, Mistral, и т.д.) в ExpertModel проекта NM_LLM_Test_2.

Примеры использования:

1. Создание нового эксперта из конфига:
    python scripts/import_gguf_expert.py \
        --gguf models/gguf/phi-3-mini-q8.gguf \
        --config configs/transfer_learning_config.yaml \
        --output models/experts/phi3_expert \
        --expert_id phi3_expert

2. Создание эксперта с кастомными параметрами:
    python scripts/import_gguf_expert.py \
        --gguf models/gguf/phi-3-mini-q8.gguf \
        --expert_id my_expert \
        --output models/experts/my_expert \
        --d_model 512 \
        --n_layers 6 \
        --n_heads 8

3. Только проверка совместимости (без создания модели):
    python scripts/import_gguf_expert.py \
        --gguf models/gguf/phi-3-mini-q8.gguf \
        --config configs/transfer_learning_config.yaml \
        --check_only

4. Создание эксперта с незамороженными весами:
    python scripts/import_gguf_expert.py \
        --gguf models/gguf/mistral-7b-q8.gguf \
        --expert_id mistral_expert \
        --output models/experts/mistral_expert \
        --no_freeze
"""

import argparse
import sys
import json
import torch
from pathlib import Path
from typing import Dict, Any

# Добавляем путь к src/python
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))

from transfer_learning import TransferLearningPipeline
from utils.config_loader import load_config


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Импорт эксперта из GGUF модели",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Обязательные аргументы
    parser.add_argument(
        "--gguf",
        type=str,
        required=True,
        help="Путь к GGUF файлу (например: models/gguf/phi-3-mini-q8.gguf)"
    )

    parser.add_argument(
        "--expert_id",
        type=str,
        required=True,
        help="Уникальный идентификатор эксперта (например: python_expert)"
    )

    # Опциональные аргументы
    parser.add_argument(
        "--config",
        type=str,
        default="configs/transfer_learning_config.yaml",
        help="Путь к YAML конфигурации (default: configs/transfer_learning_config.yaml)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Путь для сохранения эксперта (default: models/experts/{expert_id})"
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Человеко-читаемое имя эксперта (default: expert_id)"
    )

    # Переопределение параметров модели
    parser.add_argument(
        "--d_model",
        type=int,
        default=None,
        help="Размерность embeddings (переопределяет config)"
    )

    parser.add_argument(
        "--n_layers",
        type=int,
        default=None,
        help="Количество transformer слоёв (переопределяет config)"
    )

    parser.add_argument(
        "--n_heads",
        type=int,
        default=None,
        help="Количество attention heads (переопределяет config)"
    )

    parser.add_argument(
        "--d_ff",
        type=int,
        default=None,
        help="Размерность feed-forward слоя (переопределяет config)"
    )

    parser.add_argument(
        "--vocab_size",
        type=int,
        default=None,
        help="Размер vocabulary (переопределяет config)"
    )

    # Transfer learning параметры
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Индексы слоёв для переноса (например: '0,1,2,3')"
    )

    parser.add_argument(
        "--no_freeze",
        action="store_true",
        help="Не замораживать перенесённые веса (по умолчанию: замораживаются)"
    )

    parser.add_argument(
        "--no_align_embeddings",
        action="store_true",
        help="Не адаптировать embeddings под BPE vocab"
    )

    # Утилиты
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Только проверка совместимости, без создания модели"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Детальный вывод (DEBUG logging)"
    )

    parser.add_argument(
        "--max_ram_gb",
        type=float,
        default=12.0,
        help="Максимальное использование RAM в GB (default: 12.0)"
    )

    return parser.parse_args()


def load_or_create_config(args) -> Dict[str, Any]:
    """
    Загружает конфиг из файла и применяет переопределения из CLI аргументов.

    Args:
        args: Аргументы командной строки

    Returns:
        Словарь с конфигурацией
    """
    # Загружаем базовый конфиг
    try:
        config = load_config(args.config)
        print(f"✅ Конфиг загружен: {args.config}\n")
    except FileNotFoundError:
        print(f"⚠️ Конфиг не найден: {args.config}")
        print(f"   Используем значения по умолчанию\n")
        config = {
            'target_model': {
                'vocab_size': 8000,
                'd_model': 2048,  # Увеличено для совместимости с Llama-3.2
                'n_layers': 8,
                'n_heads': 16,
                'd_ff': 8192,
                'max_seq_len': 512,
                'dropout': 0.1
            },
            'transfer': {
                'layers_to_transfer': None,
                'freeze_transferred': True,
                'align_embeddings': True
            },
            'tokenizer': {
                'path': 'models/tokenizers/bpe_multilang.model'
            }
        }

    # Применяем переопределения из CLI
    if args.d_model is not None:
        config['target_model']['d_model'] = args.d_model
        print(f"   Переопределено: d_model = {args.d_model}")

    if args.n_layers is not None:
        config['target_model']['n_layers'] = args.n_layers
        print(f"   Переопределено: n_layers = {args.n_layers}")

    if args.n_heads is not None:
        config['target_model']['n_heads'] = args.n_heads
        print(f"   Переопределено: n_heads = {args.n_heads}")

    if args.d_ff is not None:
        config['target_model']['d_ff'] = args.d_ff
        print(f"   Переопределено: d_ff = {args.d_ff}")

    if args.vocab_size is not None:
        config['target_model']['vocab_size'] = args.vocab_size
        print(f"   Переопределено: vocab_size = {args.vocab_size}")

    if args.layers is not None:
        layers = [int(x.strip()) for x in args.layers.split(',')]
        config['transfer']['layers_to_transfer'] = layers
        print(f"   Переопределено: layers_to_transfer = {layers}")

    if args.no_freeze:
        config['transfer']['freeze_transferred'] = False
        print(f"   Переопределено: freeze_transferred = False")

    if args.no_align_embeddings:
        config['transfer']['align_embeddings'] = False
        print(f"   Переопределено: align_embeddings = False")

    return config


def main():
    """Главная функция скрипта."""
    args = parse_args()

    print("\n" + "="*70)
    print("🚀 Import GGUF Expert - Transfer Learning Pipeline")
    print("="*70 + "\n")

    # 1. Проверка существования GGUF файла
    gguf_path = Path(args.gguf)
    if not gguf_path.exists():
        print(f"❌ Ошибка: GGUF файл не найден: {args.gguf}")
        print(f"\nПроверьте путь и попробуйте снова.")
        sys.exit(1)

    print(f"📦 GGUF модель: {gguf_path.name}")
    print(f"   Размер: {gguf_path.stat().st_size / (1024**3):.2f} GB")
    print(f"   Путь: {gguf_path}\n")

    # 2. Загрузка конфигурации
    config = load_or_create_config(args)

    # 3. Определяем output директорию
    if args.output is None:
        output_dir = Path(f"models/experts/{args.expert_id}")
    else:
        output_dir = Path(args.output)

    expert_name = args.name if args.name else args.expert_id

    print(f"🎯 Целевой эксперт:")
    print(f"   ID: {args.expert_id}")
    print(f"   Название: {expert_name}")
    print(f"   Директория: {output_dir}\n")

    # 4. Инициализация Transfer Learning Pipeline
    try:
        pipeline = TransferLearningPipeline(
            gguf_path=str(gguf_path),
            target_model_config=config['target_model'],
            bpe_tokenizer_path=config.get('tokenizer', {}).get('path'),
            max_ram_gb=args.max_ram_gb
        )
    except Exception as e:
        print(f"❌ Ошибка при инициализации pipeline:")
        print(f"   {str(e)}\n")
        sys.exit(1)

    # 5. Проверка совместимости
    print("\n" + "="*70)
    print("📊 Проверка совместимости")
    print("="*70 + "\n")

    compat = pipeline.validate_compatibility()

    if not compat['compatible']:
        print("\n❌ Модель несовместима!")
        print("\nПредупреждения:")
        for i, warning in enumerate(compat['warnings'], 1):
            print(f"  {i}. {warning}")

        print("\n❓ Продолжить anyway? (y/n): ", end='')
        response = input().lower()

        if response != 'y':
            print("\n⚠️ Импорт отменён пользователем.")
            sys.exit(0)

    # Если только проверка - завершаем
    if args.check_only:
        print("\n✅ Проверка завершена (--check_only)")
        print(f"   Совместимость: {'✅ Да' if compat['compatible'] else '❌ Нет'}")
        print(f"   Vocab overlap: {compat['vocab_overlap']:.1%}")
        print(f"   Transferable layers: {compat['transferable_layers']}")
        sys.exit(0)

    # 6. Инициализация модели из GGUF
    print("\n" + "="*70)
    print("🔄 Перенос знаний из GGUF в ExpertModel")
    print("="*70 + "\n")

    try:
        model = pipeline.initialize_model_from_gguf(
            layers_to_transfer=config['transfer'].get('layers_to_transfer'),
            freeze_transferred_layers=config['transfer'].get('freeze_transferred', True),
            align_embeddings=config['transfer'].get('align_embeddings', True)
        )
    except Exception as e:
        print(f"\n❌ Ошибка при переносе весов:")
        print(f"   {str(e)}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 7. Сохранение модели
    print("\n" + "="*70)
    print("💾 Сохранение эксперта")
    print("="*70 + "\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем model checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config['target_model'],
        'expert_id': args.expert_id,
        'expert_name': expert_name,
        'source_gguf': str(gguf_path),
        'transfer_config': config['transfer']
    }

    checkpoint_path = output_dir / "model.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"✅ Checkpoint сохранён: {checkpoint_path}")

    # Сохраняем metadata
    metadata = {
        "expert_id": args.expert_id,
        "name": expert_name,
        "type": "transferred_model",
        "version": "1.0.0-gguf",
        "source_gguf": str(gguf_path),
        "gguf_metadata": compat['gguf_metadata'],
        "vocab_overlap": compat['vocab_overlap'],
        "transferred_layers": config['transfer'].get('layers_to_transfer'),
        "freeze_transferred": config['transfer'].get('freeze_transferred', True),
        "architecture": config['target_model'],
        "trainable_params_M": sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
        "total_params_M": sum(p.numel() for p in model.parameters()) / 1e6
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✅ Metadata сохранена: {metadata_path}")

    # Сохраняем tokenizer config
    tokenizer_config = {
        "tokenizer_type": "bpe",
        "model_path": config.get('tokenizer', {}).get('path', 'models/tokenizers/bpe_multilang.model'),
        "vocab_size": config['target_model']['vocab_size'],
        "pad_token_id": 0,
        "unk_token_id": 1,
        "bos_token_id": 2,
        "eos_token_id": 3
    }

    tokenizer_config_path = output_dir / "tokenizer_config.json"
    with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    print(f"✅ Tokenizer config сохранён: {tokenizer_config_path}")

    # 8. Итоговый отчёт
    print("\n" + "="*70)
    print("🎉 Импорт завершён успешно!")
    print("="*70 + "\n")

    print(f"📊 Статистика:")
    print(f"   Expert ID: {args.expert_id}")
    print(f"   Параметры: {metadata['total_params_M']:.2f}M")
    print(f"   Trainable: {metadata['trainable_params_M']:.2f}M ({100*metadata['trainable_params_M']/metadata['total_params_M']:.1f}%)")
    print(f"   Vocab overlap: {metadata['vocab_overlap']:.1%}")
    print(f"   Директория: {output_dir}")

    print(f"\n📝 Следующие шаги:\n")
    print(f"1. Fine-tuning эксперта на своих данных:")
    print(f"   python scripts/train_simple.py --expert {args.expert_id} --data your_data.jsonl\n")
    print(f"2. Тестирование в UI:")
    print(f"   python src/python/ui/app.py")
    print(f"   Перейдите во вкладку '💬 Chat' и выберите эксперта '{args.expert_id}'\n")
    print(f"3. Inference через API:")
    print(f"   from python.models.expert import ExpertModel")
    print(f"   model = ExpertModel.from_pretrained('{output_dir}')\n")

    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Процесс прерван пользователем (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Критическая ошибка:")
        print(f"   {str(e)}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
