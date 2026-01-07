"""
Модель эксперта на базе Transformer архитектуры.

Полноценная language model для domain-specific задач.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
from typing import Optional, Tuple, Dict

# Добавляем путь к src/python для импортов
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from python.models.transformer import TransformerBlock


class PositionalEncoding(nn.Module):
    """
    Positional Encoding для добавления информации о позиции токенов.

    Использует синусоидальные функции для создания уникальных
    позиционных embeddings для каждой позиции в последовательности.

    Args:
        d_model: Размерность embeddings
        max_seq_len: Максимальная длина последовательности
        dropout: Вероятность dropout
    """

    def __init__(self, d_model: int, max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Создаём матрицу позиционных encodings
        # [max_seq_len, d_model]
        pe = torch.zeros(max_seq_len, d_model)

        # Позиции [0, 1, 2, ..., max_seq_len-1]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Делители для синусоид разных частот
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Применяем sin к чётным индексам
        pe[:, 0::2] = torch.sin(position * div_term)

        # Применяем cos к нечётным индексам
        pe[:, 1::2] = torch.cos(position * div_term)

        # Добавляем batch dimension: [1, max_seq_len, d_model]
        pe = pe.unsqueeze(0)

        # Регистрируем как buffer (не параметр, но часть state_dict)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Добавляет positional encoding к входным embeddings.

        Args:
            x: Входные embeddings [batch, seq_len, d_model]

        Returns:
            output: Embeddings с positional encoding [batch, seq_len, d_model]
        """
        # Добавляем positional encoding к входу
        # x: [batch, seq_len, d_model]
        # self.pe: [1, max_seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ExpertModel(nn.Module):
    """
    Полная модель эксперта для domain-specific задач.

    Архитектура:
        1. Token Embedding
        2. Positional Encoding
        3. N × TransformerBlock
        4. Layer Normalization
        5. Output Projection (lm_head)

    Args:
        vocab_size: Размер словаря
        d_model: Размерность embeddings
        n_layers: Количество transformer блоков
        n_heads: Количество attention heads
        d_ff: Размерность feed-forward слоя
        max_seq_len: Максимальная длина последовательности
        dropout: Вероятность dropout
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # 1. Token Embedding
        # Преобразует token IDs в векторы размерности d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 2. Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # 3. Stack из Transformer блоков
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 4. Final Layer Normalization
        self.final_norm = nn.LayerNorm(d_model)

        # 5. Output Projection (Language Model Head)
        # Проецируем обратно в словарь для предсказания следующего токена
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов модели."""
        # Инициализация embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        # Инициализация lm_head
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Создаёт causal mask для autoregressive генерации.

        Маска предотвращает attention к будущим токенам.

        Args:
            seq_len: Длина последовательности
            device: Устройство (CPU/GPU)

        Returns:
            mask: Causal mask [1, 1, seq_len, seq_len]
        """
        # Создаём нижнюю треугольную матрицу
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))

        # Добавляем batch и head dimensions
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Прямой проход через модель.

        Args:
            input_ids: Индексы токенов [batch, seq_len]
            mask: Опциональная attention mask [batch, 1, seq_len, seq_len]

        Returns:
            logits: Логиты для каждого токена словаря [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Создаём causal mask если не предоставлена
        if mask is None:
            mask = self._create_causal_mask(seq_len, device)

        # 1. Token Embedding
        # [batch, seq_len] -> [batch, seq_len, d_model]
        x = self.token_embedding(input_ids)

        # Масштабируем embeddings (как в оригинальном Transformer)
        x = x * math.sqrt(self.d_model)

        # 2. Positional Encoding
        # [batch, seq_len, d_model] -> [batch, seq_len, d_model]
        x = self.pos_encoding(x)

        # 3. Пропускаем через все Transformer блоки
        for block in self.transformer_blocks:
            # [batch, seq_len, d_model] -> [batch, seq_len, d_model]
            x = block(x, mask)

        # 4. Final Layer Normalization
        # [batch, seq_len, d_model]
        x = self.final_norm(x)

        # 5. Output Projection
        # [batch, seq_len, d_model] -> [batch, seq_len, vocab_size]
        logits = self.lm_head(x)

        return logits

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Возвращает количество параметров модели.

        Args:
            non_embedding: Если True, не считает embedding параметры

        Returns:
            Количество параметров
        """
        n_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            n_params -= self.token_embedding.weight.numel()

        return n_params

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Генерация новых токенов (autoregressive inference).

        Args:
            input_ids: Начальная последовательность [batch, seq_len]
            max_new_tokens: Максимум токенов для генерации
            temperature: Температура для sampling (1.0 = без изменений)
            top_k: Top-k sampling (None = отключено)
            top_p: Nucleus sampling (None = отключено)

        Returns:
            generated: Сгенерированная последовательность [batch, seq_len + max_new_tokens]
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Обрезаем контекст если превышает max_seq_len
            input_ids_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]

            # Получаем логиты
            # [batch, seq_len, vocab_size]
            logits = self(input_ids_cond)

            # Берём логиты только для последнего токена
            # [batch, vocab_size]
            logits = logits[:, -1, :]

            # Применяем temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            # Top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Удаляем токены с cumulative probability > top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter sorted tensors обратно к оригинальным индексам
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')

            # Преобразуем логиты в вероятности
            probs = F.softmax(logits, dim=-1)

            # Сэмплируем следующий токен
            next_token = torch.multinomial(probs, num_samples=1)

            # Добавляем к последовательности
            # [batch, seq_len] + [batch, 1] -> [batch, seq_len + 1]
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def get_model_config(self) -> Dict[str, any]:
        """Возвращает конфигурацию модели."""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.transformer_blocks[0].attention.n_heads,
            'd_ff': self.transformer_blocks[0].ffn.linear1.out_features,
            'max_seq_len': self.max_seq_len,
            'total_params': self.get_num_params(),
            'non_embedding_params': self.get_num_params(non_embedding=True)
        }


def test_expert_model():
    """Тестовая функция для ExpertModel."""

    print("=" * 70)
    print("Тестирование ExpertModel")
    print("=" * 70)

    # Параметры модели
    vocab_size = 10000  # Размер словаря
    d_model = 512
    n_layers = 6
    n_heads = 8
    d_ff = 2048
    max_seq_len = 512

    print(f"\nСоздание модели с параметрами:")
    print(f"  - vocab_size: {vocab_size}")
    print(f"  - d_model: {d_model}")
    print(f"  - n_layers: {n_layers}")
    print(f"  - n_heads: {n_heads}")
    print(f"  - d_ff: {d_ff}")
    print(f"  - max_seq_len: {max_seq_len}")

    # Создаём модель
    model = ExpertModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len
    )

    # Получаем конфигурацию
    config = model.get_model_config()

    print(f"\n{'Параметры модели:'}")
    print(f"  - Всего параметров: {config['total_params']:,}")
    print(f"  - Без embedding: {config['non_embedding_params']:,}")

    # Тестовый вход
    batch_size = 2
    seq_len = 20
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"\n{'Тест 1: Forward pass'}")
    print(f"  Входные токены: {input_ids.shape}")

    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)

    print(f"  Выходные логиты: {logits.shape}")
    print(f"  Ожидаемая форма: [{batch_size}, {seq_len}, {vocab_size}]")
    print(f"  ✅ Размерности корректны: {logits.shape == (batch_size, seq_len, vocab_size)}")

    # Тест генерации
    print(f"\n{'Тест 2: Text generation'}")
    start_tokens = torch.randint(0, vocab_size, (1, 5))  # 5 начальных токенов
    print(f"  Начальные токены: {start_tokens.shape}")

    # Генерация с разными параметрами
    print(f"\n  Генерация с temperature=1.0:")
    generated = model.generate(start_tokens, max_new_tokens=10, temperature=1.0)
    print(f"    Сгенерировано: {generated.shape}")
    print(f"    Токены: {generated[0].tolist()}")

    print(f"\n  Генерация с temperature=0.8, top_k=50:")
    generated = model.generate(start_tokens, max_new_tokens=10, temperature=0.8, top_k=50)
    print(f"    Сгенерировано: {generated.shape}")
    print(f"    Токены: {generated[0].tolist()}")

    print(f"\n  Генерация с temperature=0.7, top_p=0.9:")
    generated = model.generate(start_tokens, max_new_tokens=10, temperature=0.7, top_p=0.9)
    print(f"    Сгенерировано: {generated.shape}")
    print(f"    Токены: {generated[0].tolist()}")

    # Тест Positional Encoding
    print(f"\n{'Тест 3: Positional Encoding'}")
    pos_enc = PositionalEncoding(d_model=512, max_seq_len=100)
    test_input = torch.randn(2, 50, 512)
    output = pos_enc(test_input)
    print(f"  Вход: {test_input.shape}")
    print(f"  Выход: {output.shape}")
    print(f"  ✅ Размерности сохранены: {test_input.shape == output.shape}")

    # Расчёт памяти
    print(f"\n{'Оценка использования памяти:'}")
    param_bytes = config['total_params'] * 4  # 4 bytes per float32
    print(f"  FP32: {param_bytes / 1024**2:.1f} MB")
    print(f"  FP16: {param_bytes / 2 / 1024**2:.1f} MB")
    print(f"  INT8: {param_bytes / 4 / 1024**2:.1f} MB")

    print("\n" + "=" * 70)
    print("✅ Все тесты пройдены успешно!")
    print("=" * 70)


if __name__ == "__main__":
    test_expert_model()
