#!/usr/bin/env python3
"""
Скрипт для обучения BPE токенайзера с помощью SentencePiece.

Создаёт мультиязычный токенайзер (русский + английский) с поддержкой опечаток.
"""

import sentencepiece as spm
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent

    # Параметры токенайзера
    corpus_file = project_root / "data" / "training" / "corpus_for_bpe.txt"
    model_prefix = project_root / "models" / "tokenizers" / "bpe_multilang"
    vocab_size = 8000  # Хороший баланс между размером и качеством

    # Создаём директорию для токенайзера
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Обучение BPE токенайзера с помощью SentencePiece")
    print("=" * 80)
    print(f"\nПараметры:")
    print(f"  Корпус: {corpus_file}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model prefix: {model_prefix}")
    print(f"  Алгоритм: BPE (Byte-Pair Encoding)")

    # Параметры обучения
    train_params = {
        'input': str(corpus_file),
        'model_prefix': str(model_prefix),
        'vocab_size': vocab_size,
        'model_type': 'bpe',  # BPE алгоритм
        'character_coverage': 0.9995,  # Покрываем 99.95% символов
        'pad_id': 0,  # <pad> = 0
        'unk_id': 1,  # <unk> = 1
        'bos_id': 2,  # <s> = 2 (begin of sequence)
        'eos_id': 3,  # </s> = 3 (end of sequence)
        'pad_piece': '<pad>',
        'unk_piece': '<unk>',
        'bos_piece': '<s>',
        'eos_piece': '</s>',
        'user_defined_symbols': [],  # Можно добавить спец. символы
        'byte_fallback': True,  # Fallback на байты для неизвестных символов
        'split_digits': True,  # Разделяем цифры
        'split_by_unicode_script': True,  # Разделяем по скриптам (кириллица/латиница)
        'split_by_whitespace': True,  # Разделяем по пробелам
        'split_by_number': True,  # Разделяем числа
        'max_sentence_length': 16384,  # Макс. длина предложения
        'num_threads': 4,  # Используем 4 потока
        'train_extremely_large_corpus': False,
    }

    print("\n" + "=" * 80)
    print("Начало обучения...")
    print("=" * 80)

    try:
        # Обучаем токенайзер
        spm.SentencePieceTrainer.train(**train_params)

        print("\n" + "=" * 80)
        print("✅ Токенайзер успешно обучен!")
        print("=" * 80)

        # Тестируем токенайзер
        print("\nТестирование токенайзера...")
        sp = spm.SentencePieceProcessor(model_file=str(model_prefix) + ".model")

        test_cases = [
            "Привет, как дела?",
            "Hello, how are you?",
            "првиет кк дела",  # С опечатками
            "программирование на Python",
            "machine learning algorithms",
            "нейронная сеть",
            "пайтон програмирование",  # С опечатками
        ]

        print("\nПримеры токенизации:")
        print("-" * 80)
        for text in test_cases:
            tokens = sp.encode(text, out_type=str)
            ids = sp.encode(text, out_type=int)
            decoded = sp.decode(ids)

            print(f"\nОригинал:  {text}")
            print(f"Токены:    {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
            print(f"IDs:       {ids[:10]}{'...' if len(ids) > 10 else ''}")
            print(f"Decoded:   {decoded}")

        print("\n" + "=" * 80)
        print(f"Модель сохранена: {model_prefix}.model")
        print(f"Vocab сохранён:   {model_prefix}.vocab")
        print("=" * 80)

        # Выводим статистику
        print("\nСтатистика токенайзера:")
        print(f"  Vocabulary size: {sp.vocab_size()}")
        print(f"  PAD token: {sp.id_to_piece(sp.pad_id())} (id={sp.pad_id()})")
        print(f"  UNK token: {sp.id_to_piece(sp.unk_id())} (id={sp.unk_id()})")
        print(f"  BOS token: {sp.id_to_piece(sp.bos_id())} (id={sp.bos_id()})")
        print(f"  EOS token: {sp.id_to_piece(sp.eos_id())} (id={sp.eos_id()})")

        return 0

    except Exception as e:
        print(f"\n❌ Ошибка при обучении токенайзера: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
