"""
Tokenizer Aligner для alignment vocabulary между GGUF и BPE.

Основные возможности:
    - Создание mapping между GGUF vocab и BPE vocab
    - Alignment embeddings при несовпадении vocab_size
    - Копирование весов для совпадающих токенов
    - Random initialization для unmapped токенов
"""

from typing import Dict, List, Tuple, Optional
import torch
import numpy as np


class TokenizerAligner:
    """
    Выполняет alignment между vocabulary GGUF модели и BPE токенайзера.

    GGUF модели обычно используют vocab 32k-128k токенов,
    в то время как BPE может иметь 8k токенов.
    Этот класс создаёт mapping и адаптирует embeddings.

    Пример:
        >>> parser = GGUFParser("phi-3-mini-q8.gguf")
        >>> bpe = BPETokenizer.from_file("bpe_multilang.model")
        >>>
        >>> aligner = TokenizerAligner(parser, bpe)
        >>> mapping = aligner.create_vocab_mapping()
        >>> print(f"Vocab overlap: {len(mapping)} / {len(bpe)}")
        >>>
        >>> # Адаптируем embeddings из GGUF
        >>> gguf_embeddings = parser.load_tensor("token_embd.weight")
        >>> aligned_emb = aligner.align_embeddings(gguf_embeddings, 8000)
    """

    def __init__(self, gguf_parser, bpe_tokenizer=None):
        """
        Инициализирует Tokenizer Aligner.

        Args:
            gguf_parser: GGUFParser для извлечения vocabulary
            bpe_tokenizer: BPE токенайзер (опционально)
        """
        self.gguf_parser = gguf_parser
        self.bpe_tokenizer = bpe_tokenizer

        # Извлекаем vocabulary из GGUF
        self.gguf_vocab = gguf_parser.get_vocab()
        self.gguf_vocab_size = len(self.gguf_vocab) if self.gguf_vocab else gguf_parser.get_metadata().get('vocab_size', 32000)

        # BPE vocabulary
        if bpe_tokenizer:
            self.bpe_vocab = self._get_bpe_vocab(bpe_tokenizer)
            self.bpe_vocab_size = len(self.bpe_vocab)
        else:
            self.bpe_vocab = []
            self.bpe_vocab_size = 8000  # Default

        print(f"✅ TokenizerAligner инициализирован:")
        print(f"   GGUF vocab size: {self.gguf_vocab_size}")
        print(f"   BPE vocab size: {self.bpe_vocab_size}")

    def _get_bpe_vocab(self, bpe_tokenizer) -> List[str]:
        """
        Извлекает vocabulary из BPE токенайзера.

        Args:
            bpe_tokenizer: BPE токенайзер

        Returns:
            Список токенов
        """
        try:
            # Пробуем разные способы извлечения vocab
            if hasattr(bpe_tokenizer, 'get_vocab'):
                return list(bpe_tokenizer.get_vocab().keys())
            elif hasattr(bpe_tokenizer, 'vocab'):
                return list(bpe_tokenizer.vocab)
            elif hasattr(bpe_tokenizer, 'id_to_piece'):
                # SentencePiece style
                return [bpe_tokenizer.id_to_piece(i) for i in range(len(bpe_tokenizer))]
            else:
                print("⚠️ Не удалось извлечь BPE vocabulary")
                return []
        except Exception as e:
            print(f"⚠️ Ошибка при извлечении BPE vocab: {e}")
            return []

    def create_vocab_mapping(self) -> Dict[int, int]:
        """
        Создаёт mapping между GGUF token ID и BPE token ID.

        Стратегия:
            - Сравниваем токены по строковому представлению
            - Для совпадающих создаём mapping
            - Для несовпадающих оставляем None (random init)

        Returns:
            Словарь: {gguf_token_id: bpe_token_id}

        Example:
            >>> mapping = aligner.create_vocab_mapping()
            >>> print(f"Mapped {len(mapping)} tokens")
            >>> print(f"Overlap: {len(mapping) / aligner.bpe_vocab_size:.1%}")
        """
        if not self.gguf_vocab or not self.bpe_vocab:
            print("⚠️ Vocabulary не доступен, пропускаем mapping")
            return {}

        mapping = {}

        # Создаём обратный индекс для BPE vocab для быстрого поиска
        bpe_token_to_id = {token: idx for idx, token in enumerate(self.bpe_vocab)}

        matched_count = 0

        for gguf_id, gguf_token in enumerate(self.gguf_vocab):
            # Нормализуем токен для сравнения
            normalized_token = self._normalize_token(gguf_token)

            # Ищем соответствие в BPE
            if normalized_token in bpe_token_to_id:
                bpe_id = bpe_token_to_id[normalized_token]
                mapping[gguf_id] = bpe_id
                matched_count += 1

        overlap_percent = (matched_count / self.bpe_vocab_size) * 100 if self.bpe_vocab_size > 0 else 0

        print(f"✅ Vocab mapping создан:")
        print(f"   Совпало токенов: {matched_count}")
        print(f"   Overlap: {overlap_percent:.1f}%")

        return mapping

    def _normalize_token(self, token: str) -> str:
        """
        Нормализует токен для сравнения.

        Args:
            token: Исходный токен

        Returns:
            Нормализованный токен
        """
        # Убираем специальные префиксы/суффиксы
        token = token.strip()

        # Убираем BPE markers (▁, Ġ, etc.)
        token = token.replace('▁', ' ')  # SentencePiece
        token = token.replace('Ġ', ' ')   # GPT-2 style

        return token

    def align_embeddings(
        self,
        gguf_embeddings: torch.Tensor,
        target_vocab_size: int,
        vocab_mapping: Optional[Dict[int, int]] = None
    ) -> torch.Tensor:
        """
        Адаптирует GGUF embeddings под целевой vocabulary size.

        Стратегия:
            1. Создаём новую матрицу [target_vocab_size, d_model]
            2. Для mapped токенов копируем веса из GGUF
            3. Для unmapped токенов используем random initialization

        Args:
            gguf_embeddings: Embeddings из GGUF [gguf_vocab_size, d_model]
            target_vocab_size: Целевой размер vocabulary (обычно BPE size)
            vocab_mapping: Опциональный mapping (если None, создаётся автоматически)

        Returns:
            Aligned embeddings [target_vocab_size, d_model]

        Example:
            >>> gguf_emb = parser.load_tensor("token_embd.weight")  # [32000, 4096]
            >>> aligned_emb = aligner.align_embeddings(gguf_emb, 8000)  # [8000, 4096]
            >>> print(f"Aligned embeddings: {aligned_emb.shape}")
        """
        gguf_vocab_size, d_model = gguf_embeddings.shape

        print(f"\n🔧 Alignment embeddings:")
        print(f"   Source: {gguf_embeddings.shape}")
        print(f"   Target: ({target_vocab_size}, {d_model})")

        # Если размеры совпадают, просто возвращаем исходные
        if gguf_vocab_size == target_vocab_size:
            print(f"   ✅ Размеры совпадают, alignment не нужен")
            return gguf_embeddings

        # Создаём новую матрицу embeddings
        aligned_embeddings = torch.randn(target_vocab_size, d_model) * 0.02

        # Создаём или используем существующий mapping
        if vocab_mapping is None:
            vocab_mapping = self.create_vocab_mapping()

        # Копируем веса для mapped токенов
        mapped_count = 0
        for gguf_id, bpe_id in vocab_mapping.items():
            if gguf_id < gguf_vocab_size and bpe_id < target_vocab_size:
                aligned_embeddings[bpe_id] = gguf_embeddings[gguf_id]
                mapped_count += 1

        unmapped_count = target_vocab_size - mapped_count

        print(f"   ✅ Скопировано: {mapped_count} токенов")
        print(f"   🔀 Random init: {unmapped_count} токенов")
        print(f"   📊 Overlap: {(mapped_count / target_vocab_size) * 100:.1f}%")

        return aligned_embeddings

    def estimate_vocab_overlap(self) -> float:
        """
        Оценивает overlap между GGUF и BPE vocabulary.

        Returns:
            Процент overlap (0.0 - 1.0)

        Example:
            >>> overlap = aligner.estimate_vocab_overlap()
            >>> print(f"Vocab overlap: {overlap:.1%}")
        """
        if not self.gguf_vocab or not self.bpe_vocab:
            # Если vocab недоступен, даём оптимистичную оценку
            return 0.6  # ~60% overlap типичен для разных tokenizers

        mapping = self.create_vocab_mapping()
        overlap = len(mapping) / self.bpe_vocab_size if self.bpe_vocab_size > 0 else 0.0

        return overlap

    def get_special_tokens_mapping(self) -> Dict[str, Tuple[int, int]]:
        """
        Создаёт mapping для специальных токенов (PAD, UNK, BOS, EOS).

        Returns:
            Словарь: {token_name: (gguf_id, bpe_id)}

        Example:
            >>> special = aligner.get_special_tokens_mapping()
            >>> print(f"PAD token: GGUF={special['PAD'][0]}, BPE={special['PAD'][1]}")
        """
        special_mapping = {}

        # Типичные имена специальных токенов
        special_names = {
            'PAD': ['<pad>', '<|pad|>', '[PAD]'],
            'UNK': ['<unk>', '<|unk|>', '[UNK]'],
            'BOS': ['<s>', '<|startoftext|>', '[BOS]', '<bos>'],
            'EOS': ['</s>', '<|endoftext|>', '[EOS]', '<eos>'],
        }

        for token_type, candidates in special_names.items():
            # Ищем в GGUF vocab
            gguf_id = self._find_token_id(candidates, self.gguf_vocab)

            # Ищем в BPE vocab
            bpe_id = self._find_token_id(candidates, self.bpe_vocab)

            if gguf_id is not None and bpe_id is not None:
                special_mapping[token_type] = (gguf_id, bpe_id)

        return special_mapping

    def _find_token_id(self, candidates: List[str], vocab: List[str]) -> Optional[int]:
        """
        Ищет токен в vocabulary по списку кандидатов.

        Args:
            candidates: Список возможных имён токена
            vocab: Vocabulary для поиска

        Returns:
            ID токена или None если не найден
        """
        for candidate in candidates:
            if candidate in vocab:
                return vocab.index(candidate)

        return None
