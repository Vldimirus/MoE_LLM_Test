#!/usr/bin/env python3
"""
BPE Tokenizer - обёртка для SentencePiece токенайзера.

Интегрирует обученный BPE токенайзер в систему обучения и inference.
Поддерживает:
    - Мультиязычность (русский, английский, и другие)
    - Обработку опечаток и грамматических ошибок
    - Byte-level fallback для неизвестных символов
"""

import sentencepiece as spm
from pathlib import Path
from typing import List, Union, Optional
import torch


class BPETokenizer:
    """
    BPE токенайзер на базе SentencePiece.

    Предоставляет единый API для токенизации текста с поддержкой:
        - Encode/decode операций
        - Special tokens (PAD, UNK, BOS, EOS)
        - Batch обработки
        - Padding и truncation

    Args:
        model_path: Путь к .model файлу SentencePiece

    Example:
        >>> tokenizer = BPETokenizer("models/tokenizers/bpe_multilang.model")
        >>> ids = tokenizer.encode("Привет, как дела?")
        >>> text = tokenizer.decode(ids)
    """

    def __init__(self, model_path: str):
        """
        Инициализация токенайзера.

        Args:
            model_path: Путь к обученной модели SentencePiece
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Модель токенайзера не найдена: {model_path}")

        # Загружаем SentencePiece модель
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))

        # Сохраняем важные параметры
        self.vocab_size = self.sp.vocab_size()
        self.pad_token_id = self.sp.pad_id()
        self.unk_token_id = self.sp.unk_id()
        self.bos_token_id = self.sp.bos_id()
        self.eos_token_id = self.sp.eos_id()

        # Строковые представления special tokens
        self.pad_token = self.sp.id_to_piece(self.pad_token_id)
        self.unk_token = self.sp.id_to_piece(self.unk_token_id)
        self.bos_token = self.sp.id_to_piece(self.bos_token_id)
        self.eos_token = self.sp.id_to_piece(self.eos_token_id)

        print(f"BPE Tokenizer загружен:")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  PAD: {self.pad_token} (id={self.pad_token_id})")
        print(f"  UNK: {self.unk_token} (id={self.unk_token_id})")
        print(f"  BOS: {self.bos_token} (id={self.bos_token_id})")
        print(f"  EOS: {self.eos_token} (id={self.eos_token_id})")

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = True
    ) -> List[int]:
        """
        Кодирует текст в список token IDs.

        Args:
            text: Входной текст для токенизации
            add_special_tokens: Добавлять ли BOS/EOS токены
            max_length: Максимальная длина последовательности
            padding: Паддинг до max_length
            truncation: Обрезать если превышает max_length

        Returns:
            Список token IDs

        Example:
            >>> tokenizer.encode("Привет, мир!")
            [2, 156, 7, 234, 89, 3]  # [BOS, tokens..., EOS]
        """
        # Кодируем текст в IDs
        ids = self.sp.encode(text, out_type=int)

        # Добавляем special tokens
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

        # Truncation
        if max_length and truncation and len(ids) > max_length:
            if add_special_tokens:
                # Сохраняем BOS и EOS
                ids = [ids[0]] + ids[1:max_length-1] + [ids[-1]]
            else:
                ids = ids[:max_length]

        # Padding
        if padding and max_length:
            if len(ids) < max_length:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))

        return ids

    def decode(
        self,
        ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Декодирует token IDs обратно в текст.

        Args:
            ids: Список или тензор token IDs
            skip_special_tokens: Пропускать ли PAD/BOS/EOS токены

        Returns:
            Декодированный текст

        Example:
            >>> tokenizer.decode([2, 156, 7, 234, 89, 3])
            "Привет, мир!"
        """
        # Конвертируем torch.Tensor в list
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        # Фильтруем special tokens если нужно
        if skip_special_tokens:
            special_ids = {
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id
            }
            ids = [i for i in ids if i not in special_ids]

        # Декодируем через SentencePiece
        text = self.sp.decode(ids)
        return text

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True
    ) -> List[List[int]]:
        """
        Кодирует батч текстов.

        Args:
            texts: Список текстов для токенизации
            add_special_tokens: Добавлять ли BOS/EOS
            max_length: Максимальная длина (авто если None)
            padding: Паддинг до max_length
            truncation: Обрезать если превышает

        Returns:
            Список списков token IDs

        Example:
            >>> tokenizer.encode_batch(["Привет!", "Hello!"])
            [[2, 156, 89, 3], [2, 234, 89, 3]]
        """
        encoded_batch = []

        # Определяем max_length автоматически если не задан
        if max_length is None and padding:
            max_length = max(
                len(self.sp.encode(text, out_type=int)) + (2 if add_special_tokens else 0)
                for text in texts
            )

        # Кодируем каждый текст
        for text in texts:
            ids = self.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=padding,
                truncation=truncation
            )
            encoded_batch.append(ids)

        return encoded_batch

    def decode_batch(
        self,
        ids_batch: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Декодирует батч token IDs.

        Args:
            ids_batch: Список списков IDs или тензор [batch_size, seq_len]
            skip_special_tokens: Пропускать ли special tokens

        Returns:
            Список декодированных текстов
        """
        # Конвертируем тензор в list of lists
        if isinstance(ids_batch, torch.Tensor):
            ids_batch = ids_batch.tolist()

        # Декодируем каждую последовательность
        texts = [
            self.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in ids_batch
        ]
        return texts

    def tokenize(self, text: str) -> List[str]:
        """
        Токенизирует текст в список строковых токенов (для отладки).

        Args:
            text: Входной текст

        Returns:
            Список токенов (строки)

        Example:
            >>> tokenizer.tokenize("Привет, мир!")
            ['▁Привет', ',', '▁мир', '!']
        """
        return self.sp.encode(text, out_type=str)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Конвертирует строковые токены в IDs.

        Args:
            tokens: Список строковых токенов

        Returns:
            Список token IDs
        """
        return [self.sp.piece_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        Конвертирует IDs в строковые токены.

        Args:
            ids: Список token IDs

        Returns:
            Список строковых токенов
        """
        return [self.sp.id_to_piece(id) for id in ids]

    def __len__(self) -> int:
        """Возвращает размер словаря."""
        return self.vocab_size

    def __call__(self, text: str, **kwargs) -> List[int]:
        """
        Позволяет использовать токенайзер как функцию.

        Example:
            >>> tokenizer("Привет!")
            [2, 156, 89, 3]
        """
        return self.encode(text, **kwargs)

    def get_vocab(self) -> dict:
        """
        Возвращает словарь всех токенов.

        Returns:
            Dict[str, int]: Словарь {токен: ID}
        """
        vocab = {}
        for i in range(self.vocab_size):
            piece = self.sp.id_to_piece(i)
            vocab[piece] = i
        return vocab


def test_tokenizer():
    """
    Тестирование BPE токенайзера.

    Проверяет:
        - Базовую токенизацию
        - Обработку опечаток
        - Мультиязычность
        - Batch операции
    """
    print("=" * 80)
    print("Тестирование BPE Tokenizer")
    print("=" * 80)

    # Инициализация
    model_path = Path(__file__).parent.parent.parent.parent / "models" / "tokenizers" / "bpe_multilang.model"
    tokenizer = BPETokenizer(str(model_path))

    print("\n" + "=" * 80)
    print("Тест 1: Базовая токенизация")
    print("=" * 80)

    test_cases = [
        "Привет, как дела?",
        "Hello, how are you?",
        "првиет кк дела",  # С опечатками
        "программирование на Python",
        "machine learning algorithms",
    ]

    for text in test_cases:
        # Encode
        ids = tokenizer.encode(text, add_special_tokens=True)
        tokens = tokenizer.tokenize(text)

        # Decode
        decoded = tokenizer.decode(ids, skip_special_tokens=True)

        print(f"\nОригинал:  {text}")
        print(f"Токены:    {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        print(f"IDs:       {ids[:12]}{'...' if len(ids) > 12 else ''}")
        print(f"Decoded:   {decoded}")
        print(f"Match:     {'✅' if decoded.strip() == text.strip() else '❌'}")

    print("\n" + "=" * 80)
    print("Тест 2: Batch операции")
    print("=" * 80)

    batch_texts = [
        "Короткий текст",
        "Это более длинный текст для тестирования",
        "Short"
    ]

    # Encode batch с padding
    encoded_batch = tokenizer.encode_batch(
        batch_texts,
        add_special_tokens=True,
        padding=True,
        truncation=True
    )

    print(f"\nБатч из {len(batch_texts)} текстов:")
    for i, (text, ids) in enumerate(zip(batch_texts, encoded_batch)):
        print(f"  [{i}] '{text}' → {len(ids)} токенов")

    # Decode batch
    decoded_batch = tokenizer.decode_batch(encoded_batch, skip_special_tokens=True)

    print("\nДекодированные тексты:")
    for orig, decoded in zip(batch_texts, decoded_batch):
        match = '✅' if orig.strip() == decoded.strip() else '❌'
        print(f"  {match} '{orig}' → '{decoded}'")

    print("\n" + "=" * 80)
    print("Тест 3: Special tokens")
    print("=" * 80)

    text = "Тест"
    ids_with_special = tokenizer.encode(text, add_special_tokens=True)
    ids_without_special = tokenizer.encode(text, add_special_tokens=False)

    print(f"\nТекст: '{text}'")
    print(f"С special tokens:    {ids_with_special}")
    print(f"Без special tokens:  {ids_without_special}")
    print(f"\nBOS (начало): {tokenizer.bos_token} = {tokenizer.bos_token_id}")
    print(f"EOS (конец):  {tokenizer.eos_token} = {tokenizer.eos_token_id}")
    print(f"PAD (паддинг): {tokenizer.pad_token} = {tokenizer.pad_token_id}")

    print("\n" + "=" * 80)
    print("Тест 4: Обработка опечаток")
    print("=" * 80)

    typo_pairs = [
        ("привет как дела", "првиет кк дела"),
        ("программирование", "програмирование"),
        ("нейронная сеть", "нейроная сеть"),
        ("machine learning", "machne lerning"),
    ]

    for correct, typo in typo_pairs:
        ids_correct = tokenizer.encode(correct, add_special_tokens=False)
        ids_typo = tokenizer.encode(typo, add_special_tokens=False)

        tokens_correct = tokenizer.tokenize(correct)
        tokens_typo = tokenizer.tokenize(typo)

        print(f"\nКорректно: '{correct}'")
        print(f"  Токены: {tokens_correct}")
        print(f"С опечаткой: '{typo}'")
        print(f"  Токены: {tokens_typo}")
        print(f"  UNK токенов: {sum(1 for id in ids_typo if id == tokenizer.unk_token_id)}")

    print("\n" + "=" * 80)
    print("✅ Все тесты завершены!")
    print("=" * 80)


if __name__ == "__main__":
    test_tokenizer()
