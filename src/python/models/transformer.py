"""
Базовая реализация Transformer архитектуры.

Этот модуль содержит основные блоки трансформера для использования в экспертах.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention механизм.

    Реализует механизм внимания с несколькими головами для параллельной
    обработки различных представлений входных данных.

    Args:
        d_model: Размерность входных embeddings
        n_heads: Количество attention heads
        dropout: Вероятность dropout (по умолчанию 0.1)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, "d_model должен делиться на n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Линейные проекции для Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Выходная проекция
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Прямой проход через multi-head attention.

        Args:
            query: Query тензор [batch, seq_len, d_model]
            key: Key тензор [batch, seq_len, d_model]
            value: Value тензор [batch, seq_len, d_model]
            mask: Опциональная маска [batch, seq_len, seq_len]

        Returns:
            output: Выходной тензор [batch, seq_len, d_model]
        """
        batch_size = query.size(0)

        # Линейные проекции
        # [batch, seq_len, d_model] -> [batch, seq_len, d_model]
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Разделяем на heads
        # [batch, seq_len, d_model] -> [batch, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Вычисляем attention scores
        # [batch, n_heads, seq_len, d_k] @ [batch, n_heads, d_k, seq_len]
        # -> [batch, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Применяем маску если есть
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Применяем softmax
        # [batch, n_heads, seq_len, seq_len]
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Применяем attention к values
        # [batch, n_heads, seq_len, seq_len] @ [batch, n_heads, seq_len, d_k]
        # -> [batch, n_heads, seq_len, d_k]
        output = torch.matmul(attention_weights, V)

        # Объединяем heads
        # [batch, n_heads, seq_len, d_k] -> [batch, seq_len, d_model]
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model)

        # Финальная проекция
        output = self.W_o(output)

        return output


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Применяет две линейные трансформации с активацией GELU между ними.

    Args:
        d_model: Размерность входа/выхода
        d_ff: Размерность скрытого слоя (обычно 4 * d_model)
        dropout: Вероятность dropout
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через FFN.

        Args:
            x: Входной тензор [batch, seq_len, d_model]

        Returns:
            output: Выходной тензор [batch, seq_len, d_model]
        """
        # [batch, seq_len, d_model] -> [batch, seq_len, d_ff]
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        # [batch, seq_len, d_ff] -> [batch, seq_len, d_model]
        x = self.linear2(x)

        return x


class TransformerBlock(nn.Module):
    """
    Один блок Transformer архитектуры.

    Состоит из multi-head attention и feed-forward network
    с residual connections и layer normalization.

    Args:
        d_model: Размерность embeddings
        n_heads: Количество attention heads
        d_ff: Размерность feed-forward слоя
        dropout: Вероятность dropout
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Прямой проход через transformer block.

        Args:
            x: Входной тензор [batch, seq_len, d_model]
            mask: Опциональная attention mask [batch, seq_len, seq_len]

        Returns:
            output: Выходной тензор [batch, seq_len, d_model]
        """
        # Self-attention с residual connection и layer norm
        # [batch, seq_len, d_model]
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward с residual connection и layer norm
        # [batch, seq_len, d_model]
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x


def test_transformer_block():
    """Тестовая функция для проверки TransformerBlock."""

    print("=" * 60)
    print("Тестирование TransformerBlock")
    print("=" * 60)

    # Параметры
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    d_ff = 2048

    # Создаём блок
    block = TransformerBlock(d_model, n_heads, d_ff)

    # Тестовый вход
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\nВходной тензор: {x.shape}")
    print(f"Параметры блока:")
    print(f"  - d_model: {d_model}")
    print(f"  - n_heads: {n_heads}")
    print(f"  - d_ff: {d_ff}")

    # Прямой проход
    output = block(x)

    print(f"\nВыходной тензор: {output.shape}")
    print(f"Размерности совпадают: {output.shape == x.shape}")

    # Количество параметров
    total_params = sum(p.numel() for p in block.parameters())
    print(f"\nВсего параметров в блоке: {total_params:,}")

    print("\n✅ Тест пройден успешно!")
    print("=" * 60)


if __name__ == "__main__":
    test_transformer_block()
