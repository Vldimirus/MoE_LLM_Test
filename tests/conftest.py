"""
Pytest fixtures и конфигурация для тестов.

Общие fixtures доступные всем тестам.
"""

import pytest
import torch
import sys
from pathlib import Path

# Добавляем src/python в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))


# ============================================================================
# Fixtures для моделей
# ============================================================================

@pytest.fixture
def device():
    """Устройство для вычислений (CPU)."""
    return "cpu"


@pytest.fixture
def vocab_size():
    """Стандартный размер словаря для тестов."""
    return 1000


@pytest.fixture
def d_model():
    """Размерность модели для тестов."""
    return 128


@pytest.fixture
def n_heads():
    """Количество attention heads для тестов."""
    return 4


@pytest.fixture
def n_layers():
    """Количество transformer layers для тестов."""
    return 2


@pytest.fixture
def d_ff():
    """Размерность feed-forward для тестов."""
    return 512


@pytest.fixture
def max_seq_len():
    """Максимальная длина последовательности для тестов."""
    return 64


@pytest.fixture
def dropout():
    """Dropout rate для тестов."""
    return 0.1


@pytest.fixture
def batch_size():
    """Размер batch для тестов."""
    return 2


@pytest.fixture
def seq_len():
    """Длина последовательности для тестов."""
    return 16


# ============================================================================
# Fixtures для данных
# ============================================================================

@pytest.fixture
def sample_text():
    """Пример текста для тестирования."""
    return "это пример текста для тестирования модели"


@pytest.fixture
def sample_texts():
    """Список примеров текстов."""
    return [
        "первый пример текста для обучения",
        "второй пример с другими словами",
        "третий текст для проверки работы",
        "четвёртый пример для тестирования"
    ]


@pytest.fixture
def sample_tokens(batch_size, seq_len, vocab_size):
    """Тензор с примерными токенами."""
    return torch.randint(0, vocab_size, (batch_size, seq_len))


@pytest.fixture
def sample_embeddings(batch_size, seq_len, d_model):
    """Тензор с примерными embeddings."""
    return torch.randn(batch_size, seq_len, d_model)


# ============================================================================
# Fixtures для файлов
# ============================================================================

@pytest.fixture
def temp_text_file(tmp_path):
    """Временный текстовый файл для тестов."""
    file_path = tmp_path / "test_data.txt"
    content = """
Это первый параграф текста.
Он содержит несколько строк.

Это второй параграф.
Тоже с несколькими строками текста.

Третий параграф для разнообразия.
"""
    file_path.write_text(content, encoding='utf-8')
    return file_path


@pytest.fixture
def temp_jsonl_file(tmp_path):
    """Временный JSONL файл для тестов."""
    import json
    file_path = tmp_path / "test_data.jsonl"
    data = [
        {"text": "Первая строка в JSONL формате"},
        {"text": "Вторая строка с другим текстом"},
        {"text": "Третья строка для тестирования"}
    ]
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    return file_path


# ============================================================================
# Fixtures для моделей (создание экземпляров)
# ============================================================================

@pytest.fixture
def transformer_block(d_model, n_heads, d_ff, dropout):
    """Экземпляр TransformerBlock для тестов."""
    from models.transformer import TransformerBlock
    return TransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout
    )


@pytest.fixture
def expert_model(vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len, dropout):
    """Экземпляр ExpertModel для тестов."""
    from models.expert import ExpertModel
    return ExpertModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout
    )


@pytest.fixture
def simple_router():
    """Экземпляр SimpleRouter для тестов."""
    from routing.router import SimpleRouter
    router = SimpleRouter()

    # Добавляем тестовых экспертов
    router.add_expert(
        expert_id="python",
        domain="Python Programming",
        keywords=["python", "код", "функция", "класс"],
        priority=8
    )
    router.add_expert(
        expert_id="math",
        domain="Mathematics",
        keywords=["математика", "уравнение", "интеграл"],
        priority=7
    )
    router.add_expert(
        expert_id="general",
        domain="General Knowledge",
        keywords=["общий", "вопрос", "информация"],
        priority=5
    )

    return router


@pytest.fixture
def three_level_memory():
    """Экземпляр ThreeLevelMemory для тестов."""
    from memory.three_level_memory import ThreeLevelMemory
    return ThreeLevelMemory(
        current_limit=100,  # Маленький лимит для тестов
        obsolete_limit=100,
        longterm_limit=100
    )


@pytest.fixture
def simple_tokenizer(sample_texts):
    """Экземпляр SimpleTokenizer для тестов."""
    from training.dataset import SimpleTokenizer
    tokenizer = SimpleTokenizer(vocab_size=100)
    tokenizer.build_vocab(sample_texts)
    return tokenizer


# ============================================================================
# Helper functions
# ============================================================================

@pytest.fixture
def assert_tensor_shape():
    """Helper для проверки формы тензора."""
    def _assert_shape(tensor, expected_shape):
        assert tensor.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {tensor.shape}"
    return _assert_shape


@pytest.fixture
def assert_tensor_dtype():
    """Helper для проверки типа тензора."""
    def _assert_dtype(tensor, expected_dtype):
        assert tensor.dtype == expected_dtype, \
            f"Expected dtype {expected_dtype}, got {tensor.dtype}"
    return _assert_dtype


@pytest.fixture
def count_parameters():
    """Helper для подсчёта параметров модели."""
    def _count(model):
        return sum(p.numel() for p in model.parameters())
    return _count
