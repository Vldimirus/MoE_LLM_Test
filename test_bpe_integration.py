#!/usr/bin/env python3
"""
Быстрый тест интеграции BPE токенайзера в MoESystem.

Проверяет:
    - Загрузку эксперта с BPE токенайзером
    - Работу chat() с опечатками
    - Мультиязычность
"""

import sys
from pathlib import Path

# Добавляем src/python в путь
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src" / "python"))

from ui.backend.moe_system import MoESystem


def main():
    print("=" * 80)
    print("Тест интеграции BPE токенайзера в MoESystem")
    print("=" * 80)

    # Инициализация MoESystem
    print("\n1. Инициализация MoESystem...")
    moe = MoESystem(config_path="configs/ui_config.yaml")
    print("   ✅ MoESystem инициализирован")

    # Загрузка эксперта general
    print("\n2. Загрузка эксперта 'general' с BPE токенайзером...")
    success = moe.load_expert('general')

    if not success:
        print("   ❌ Ошибка загрузки эксперта!")
        return 1

    print("   ✅ Эксперт загружен успешно")

    # Проверяем тип токенайзера
    tokenizer = moe.tokenizers.get('general')
    if tokenizer is None:
        print("   ❌ Токенайзер не найден!")
        return 1

    tokenizer_type = type(tokenizer).__name__
    print(f"   Тип токенайзера: {tokenizer_type}")
    print(f"   Vocab size: {len(tokenizer)}")

    # Тесты с разными типами входов
    print("\n" + "=" * 80)
    print("3. Тестирование chat() с различными входами")
    print("=" * 80)

    test_cases = [
        {
            'name': 'Корректный русский',
            'message': 'Привет, как дела?'
        },
        {
            'name': 'Корректный английский',
            'message': 'Hello, how are you?'
        },
        {
            'name': 'Русский с опечатками',
            'message': 'првиет кк дела как настроение'
        },
        {
            'name': 'Английский с опечатками',
            'message': 'helo hw r u tody'
        },
        {
            'name': 'Смешанный текст',
            'message': 'Привет! Hello! How дела?'
        },
        {
            'name': 'Программирование',
            'message': 'напиши функцию на Python'
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Тест {i}: {test_case['name']} ---")
        print(f"User: {test_case['message']}")

        # Отправляем сообщение
        response = moe.chat(
            user_message=test_case['message'],
            expert_id='general'
        )

        # Проверяем результат
        if 'error' in response or 'Error' in response.get('response', ''):
            print(f"❌ Ошибка: {response.get('response', 'Unknown error')}")
        else:
            print(f"Assistant: {response['response'][:100]}{'...' if len(response['response']) > 100 else ''}")
            print(f"Токенов сгенерировано: {response['tokens_generated']}")
            print(f"Latency: {response['latency_ms']:.1f}ms")
            print(f"✅ Тест пройден")

    # Проверка памяти
    print("\n" + "=" * 80)
    print("4. Проверка системы памяти")
    print("=" * 80)

    memory_stats = moe.get_memory_stats()

    print(f"\nСтатистика памяти:")
    print(f"  Total chunks: {memory_stats.get('total_chunks', 0)}")
    print(f"  Total tokens: {memory_stats.get('total_tokens', 0)}")

    if memory_stats.get('total_chunks', 0) > 0:
        print("  ✅ Память работает корректно")
    else:
        print("  ⚠️  Память пустая (нормально для первого запуска)")

    # Проверка токенизации с опечатками
    print("\n" + "=" * 80)
    print("5. Детальная проверка токенизации опечаток")
    print("=" * 80)

    typo_tests = [
        ("привет", "првиет"),
        ("программирование", "програмирование"),
        ("нейронная сеть", "нейроная сеть"),
        ("hello world", "helo wrld"),
    ]

    for correct, typo in typo_tests:
        tokens_correct = tokenizer.encode(correct, add_special_tokens=False)
        tokens_typo = tokenizer.encode(typo, add_special_tokens=False)

        unk_count_correct = sum(1 for t in tokens_correct if t == tokenizer.unk_token_id)
        unk_count_typo = sum(1 for t in tokens_typo if t == tokenizer.unk_token_id)

        print(f"\nКорректно: '{correct}'")
        print(f"  Токенов: {len(tokens_correct)}, UNK: {unk_count_correct}")
        print(f"С опечаткой: '{typo}'")
        print(f"  Токенов: {len(tokens_typo)}, UNK: {unk_count_typo}")

        if unk_count_typo == 0:
            print("  ✅ Опечатки обрабатываются без UNK")
        else:
            print(f"  ⚠️  {unk_count_typo} UNK токенов в опечатке")

    print("\n" + "=" * 80)
    print("✅ Все тесты завершены!")
    print("=" * 80)

    print("\nВыводы:")
    print(f"  • Тип токенайзера: {tokenizer_type}")
    print(f"  • Vocabulary size: {len(tokenizer)}")
    print(f"  • BPE интеграция: {'✅ Работает' if tokenizer_type == 'BPETokenizer' else '⚠️ SimpleTokenizer'}")
    print(f"  • Обработка опечаток: {'✅ Поддерживается' if tokenizer_type == 'BPETokenizer' else '❌ Ограничена'}")
    print(f"  • Система памяти: {'✅ Активна' if memory_stats.get('total_chunks', 0) > 0 else '⚠️ Пустая'}")

    if tokenizer_type == 'BPETokenizer':
        print("\n🎉 BPE токенайзер успешно интегрирован и работает!")
    else:
        print("\n⚠️  BPE токенайзер НЕ загрузился, используется SimpleTokenizer")
        print("   Проверьте наличие tokenizer_config.json в models/experts/general/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
