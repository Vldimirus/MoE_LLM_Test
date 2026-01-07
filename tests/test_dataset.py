"""
Тесты для Dataset и Tokenizer для обучения моделей.

Тестирует:
    - SimpleTokenizer
    - TextDataset
    - collate_fn
    - create_dataloaders
"""

import pytest
import torch
import json
from pathlib import Path
from training.dataset import SimpleTokenizer, TextDataset, collate_fn, create_dataloaders


# ============================================================================
# Тесты для SimpleTokenizer
# ============================================================================

@pytest.mark.unit
@pytest.mark.training
@pytest.mark.fast
class TestSimpleTokenizer:
    """Тесты для SimpleTokenizer."""

    def test_initialization(self):
        """Тест инициализации токенайзера."""
        tokenizer = SimpleTokenizer(vocab_size=100)

        assert tokenizer.vocab_size == 100
        assert len(tokenizer.word2idx) == 4  # Только special tokens
        assert tokenizer.pad_token_id == 0
        assert tokenizer.unk_token_id == 1
        assert tokenizer.bos_token_id == 2
        assert tokenizer.eos_token_id == 3

    def test_special_tokens(self):
        """Тест специальных токенов."""
        tokenizer = SimpleTokenizer()

        assert tokenizer.word2idx["<PAD>"] == 0
        assert tokenizer.word2idx["<UNK>"] == 1
        assert tokenizer.word2idx["<BOS>"] == 2
        assert tokenizer.word2idx["<EOS>"] == 3

    def test_build_vocab_basic(self):
        """Тест построения словаря."""
        tokenizer = SimpleTokenizer(vocab_size=100)

        texts = [
            "hello world",
            "hello python",
            "python programming"
        ]

        tokenizer.build_vocab(texts)

        # Должно быть 4 special + новые слова
        assert len(tokenizer) > 4
        assert "hello" in tokenizer.word2idx
        assert "python" in tokenizer.word2idx
        assert "world" in tokenizer.word2idx

    def test_build_vocab_frequency(self):
        """Тест что слова сортируются по частоте."""
        tokenizer = SimpleTokenizer(vocab_size=100)

        texts = ["test " * 10, "rare"]

        tokenizer.build_vocab(texts)

        # "test" встречается чаще, должен иметь меньший индекс
        assert "test" in tokenizer.word2idx
        test_idx = tokenizer.word2idx["test"]
        rare_idx = tokenizer.word2idx.get("rare", 999)

        assert test_idx < rare_idx

    def test_build_vocab_size_limit(self):
        """Тест ограничения размера словаря."""
        tokenizer = SimpleTokenizer(vocab_size=10)

        texts = [f"word{i} " for i in range(100)]

        tokenizer.build_vocab(texts)

        # Не должно превышать vocab_size
        assert len(tokenizer) <= 10

    def test_encode_basic(self):
        """Тест базового кодирования."""
        tokenizer = SimpleTokenizer(vocab_size=100)
        tokenizer.build_vocab(["hello world"])

        encoded = tokenizer.encode("hello world", add_special_tokens=True)

        # Должно быть BOS + 2 слова + EOS
        assert len(encoded) == 4
        assert encoded[0] == tokenizer.bos_token_id
        assert encoded[-1] == tokenizer.eos_token_id

    def test_encode_without_special_tokens(self):
        """Тест кодирования без специальных токенов."""
        tokenizer = SimpleTokenizer(vocab_size=100)
        tokenizer.build_vocab(["hello world"])

        encoded = tokenizer.encode("hello world", add_special_tokens=False)

        # Только 2 слова, без BOS/EOS
        assert len(encoded) == 2
        assert encoded[0] != tokenizer.bos_token_id
        assert encoded[-1] != tokenizer.eos_token_id

    def test_encode_unknown_words(self):
        """Тест кодирования неизвестных слов."""
        tokenizer = SimpleTokenizer(vocab_size=100)
        tokenizer.build_vocab(["hello"])

        encoded = tokenizer.encode("hello unknown_word", add_special_tokens=False)

        # "unknown_word" должно быть закодировано как UNK
        assert encoded[1] == tokenizer.unk_token_id

    def test_encode_case_insensitive(self):
        """Тест что кодирование case-insensitive."""
        tokenizer = SimpleTokenizer(vocab_size=100)
        tokenizer.build_vocab(["hello"])

        encoded1 = tokenizer.encode("hello", add_special_tokens=False)
        encoded2 = tokenizer.encode("HELLO", add_special_tokens=False)

        assert encoded1 == encoded2

    def test_decode_basic(self):
        """Тест базового декодирования."""
        tokenizer = SimpleTokenizer(vocab_size=100)
        tokenizer.build_vocab(["hello world"])

        encoded = tokenizer.encode("hello world", add_special_tokens=True)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)

        assert decoded == "hello world"

    def test_decode_with_special_tokens(self):
        """Тест декодирования со специальными токенами."""
        tokenizer = SimpleTokenizer(vocab_size=100)
        tokenizer.build_vocab(["hello"])

        encoded = tokenizer.encode("hello", add_special_tokens=True)
        decoded = tokenizer.decode(encoded, skip_special_tokens=False)

        # Должны быть включены BOS и EOS
        assert "<BOS>" in decoded
        assert "<EOS>" in decoded

    def test_decode_unknown_token(self):
        """Тест декодирования с UNK токенами."""
        tokenizer = SimpleTokenizer(vocab_size=100)
        tokenizer.build_vocab(["hello"])

        encoded = [tokenizer.bos_token_id, tokenizer.unk_token_id, tokenizer.eos_token_id]
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)

        assert "<UNK>" in decoded

    def test_len(self):
        """Тест метода __len__."""
        tokenizer = SimpleTokenizer(vocab_size=100)
        tokenizer.build_vocab(["hello world python"])

        # 4 special + 3 слова
        assert len(tokenizer) == 7


# ============================================================================
# Тесты для TextDataset
# ============================================================================

@pytest.mark.unit
@pytest.mark.training
@pytest.mark.fast
class TestTextDataset:
    """Тесты для TextDataset."""

    @pytest.fixture
    def tokenizer(self):
        """Токенайзер для тестов."""
        tok = SimpleTokenizer(vocab_size=100)
        tok.build_vocab([
            "это первый текст",
            "это второй текст",
            "пример для обучения модели"
        ])
        return tok

    def test_load_txt_file(self, temp_text_file, tokenizer):
        """Тест загрузки .txt файла."""
        dataset = TextDataset(
            file_path=str(temp_text_file),
            tokenizer=tokenizer,
            max_length=20
        )

        assert len(dataset) > 0

    def test_load_jsonl_file(self, temp_jsonl_file, tokenizer):
        """Тест загрузки .jsonl файла."""
        dataset = TextDataset(
            file_path=str(temp_jsonl_file),
            tokenizer=tokenizer,
            max_length=20
        )

        assert len(dataset) > 0

    def test_unsupported_file_format(self, tmp_path, tokenizer):
        """Тест с неподдерживаемым форматом файла."""
        file_path = tmp_path / "test.pdf"
        file_path.write_text("test")

        with pytest.raises(ValueError, match="Неподдерживаемый формат"):
            TextDataset(
                file_path=str(file_path),
                tokenizer=tokenizer,
                max_length=20
            )

    def test_getitem_returns_correct_format(self, temp_text_file, tokenizer):
        """Тест что __getitem__ возвращает правильный формат."""
        dataset = TextDataset(
            file_path=str(temp_text_file),
            tokenizer=tokenizer,
            max_length=20
        )

        sample = dataset[0]

        assert 'input_ids' in sample
        assert 'labels' in sample
        assert isinstance(sample['input_ids'], torch.Tensor)
        assert isinstance(sample['labels'], torch.Tensor)

    def test_getitem_shifts_labels(self, temp_text_file, tokenizer):
        """Тест что labels сдвинуты на 1 относительно input_ids."""
        dataset = TextDataset(
            file_path=str(temp_text_file),
            tokenizer=tokenizer,
            max_length=20
        )

        sample = dataset[0]

        # labels должны быть tokens[1:], input должен быть tokens[:-1]
        assert len(sample['labels']) == len(sample['input_ids'])

    def test_sliding_window_for_long_texts(self, tmp_path, tokenizer):
        """Тест sliding window для длинных текстов."""
        # Создаём файл с длинным текстом
        file_path = tmp_path / "long_text.txt"
        long_text = " ".join(["слово"] * 100)
        file_path.write_text(long_text, encoding='utf-8')

        dataset = TextDataset(
            file_path=str(file_path),
            tokenizer=tokenizer,
            max_length=20,
            stride=10
        )

        # Должно быть создано несколько chunks
        assert len(dataset) > 1

    def test_max_length_respected(self, temp_text_file, tokenizer):
        """Тест что max_length соблюдается."""
        dataset = TextDataset(
            file_path=str(temp_text_file),
            tokenizer=tokenizer,
            max_length=10
        )

        for i in range(len(dataset)):
            sample = dataset[i]
            assert len(sample['input_ids']) <= 10
            assert len(sample['labels']) <= 10

    def test_len(self, temp_text_file, tokenizer):
        """Тест метода __len__."""
        dataset = TextDataset(
            file_path=str(temp_text_file),
            tokenizer=tokenizer,
            max_length=20
        )

        length = len(dataset)
        assert length > 0
        assert isinstance(length, int)


# ============================================================================
# Тесты для collate_fn
# ============================================================================

@pytest.mark.unit
@pytest.mark.training
@pytest.mark.fast
class TestCollateFn:
    """Тесты для collate function."""

    def test_basic_collate(self):
        """Тест базового collate."""
        batch = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'labels': torch.tensor([2, 3, 4])
            },
            {
                'input_ids': torch.tensor([5, 6]),
                'labels': torch.tensor([6, 7])
            }
        ]

        result = collate_fn(batch, pad_token_id=0)

        assert 'input_ids' in result
        assert 'labels' in result
        assert 'attention_mask' in result

    def test_padding_applied(self):
        """Тест что padding применяется."""
        batch = [
            {
                'input_ids': torch.tensor([1, 2, 3, 4]),
                'labels': torch.tensor([2, 3, 4, 5])
            },
            {
                'input_ids': torch.tensor([6, 7]),
                'labels': torch.tensor([7, 8])
            }
        ]

        result = collate_fn(batch, pad_token_id=0)

        # Все sequences должны иметь одинаковую длину
        assert result['input_ids'].shape[1] == 4  # max_len
        assert result['labels'].shape[1] == 4

    def test_attention_mask_correct(self):
        """Тест что attention mask корректная."""
        batch = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'labels': torch.tensor([2, 3, 4])
            },
            {
                'input_ids': torch.tensor([5, 6]),
                'labels': torch.tensor([6, 7])
            }
        ]

        result = collate_fn(batch, pad_token_id=0)

        # Первый sample: 1,1,1 (все реальные)
        # Второй sample: 1,1,0 (два реальных, один padding)
        assert result['attention_mask'][0].sum() == 3
        assert result['attention_mask'][1].sum() == 2

    def test_labels_padding_is_minus_100(self):
        """Тест что padding в labels равен -100."""
        batch = [
            {
                'input_ids': torch.tensor([1, 2, 3, 4]),
                'labels': torch.tensor([2, 3, 4, 5])
            },
            {
                'input_ids': torch.tensor([6, 7]),
                'labels': torch.tensor([7, 8])
            }
        ]

        result = collate_fn(batch, pad_token_id=0)

        # Во втором sample должно быть 2 padding токена с value -100
        labels_second = result['labels'][1]
        assert (labels_second[2:] == -100).all()

    def test_batch_shape(self):
        """Тест размерности batch."""
        batch = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'labels': torch.tensor([2, 3, 4])
            },
            {
                'input_ids': torch.tensor([5, 6]),
                'labels': torch.tensor([6, 7])
            }
        ]

        result = collate_fn(batch, pad_token_id=0)

        # Batch size = 2, max_len = 3
        assert result['input_ids'].shape == (2, 3)
        assert result['labels'].shape == (2, 3)
        assert result['attention_mask'].shape == (2, 3)

    def test_empty_batch(self):
        """Тест с пустым batch."""
        batch = []

        with pytest.raises((ValueError, RuntimeError)):
            collate_fn(batch, pad_token_id=0)

    def test_single_sample_batch(self):
        """Тест batch с одним sample."""
        batch = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'labels': torch.tensor([2, 3, 4])
            }
        ]

        result = collate_fn(batch, pad_token_id=0)

        assert result['input_ids'].shape == (1, 3)
        assert result['labels'].shape == (1, 3)
        assert result['attention_mask'].shape == (1, 3)


# ============================================================================
# Тесты для create_dataloaders
# ============================================================================

@pytest.mark.unit
@pytest.mark.training
@pytest.mark.fast
class TestCreateDataloaders:
    """Тесты для create_dataloaders function."""

    def test_create_train_only(self, temp_text_file):
        """Тест создания только train dataloader."""
        train_loader, val_loader, tokenizer = create_dataloaders(
            train_file=str(temp_text_file),
            val_file=None,
            batch_size=2,
            max_length=20
        )

        assert train_loader is not None
        assert val_loader is None
        assert tokenizer is not None
        assert len(tokenizer) > 4  # special tokens + words

    def test_create_train_and_val(self, temp_text_file, temp_jsonl_file):
        """Тест создания train и val dataloaders."""
        train_loader, val_loader, tokenizer = create_dataloaders(
            train_file=str(temp_text_file),
            val_file=str(temp_jsonl_file),
            batch_size=2,
            max_length=20
        )

        assert train_loader is not None
        assert val_loader is not None
        assert tokenizer is not None

    def test_with_existing_tokenizer(self, temp_text_file):
        """Тест с готовым токенайзером."""
        # Создаём свой токенайзер
        tokenizer = SimpleTokenizer(vocab_size=50)
        tokenizer.build_vocab(["тестовый текст для проверки"])

        train_loader, val_loader, returned_tokenizer = create_dataloaders(
            train_file=str(temp_text_file),
            tokenizer=tokenizer,
            batch_size=2,
            max_length=20
        )

        # Должен вернуть тот же токенайзер
        assert returned_tokenizer is tokenizer

    def test_batch_iteration(self, temp_text_file):
        """Тест итерации по batches."""
        train_loader, _, _ = create_dataloaders(
            train_file=str(temp_text_file),
            batch_size=2,
            max_length=20
        )

        # Должны мочь итерировать
        for batch in train_loader:
            assert 'input_ids' in batch
            assert 'labels' in batch
            assert 'attention_mask' in batch
            break  # Проверяем только первый batch

    def test_batch_size_respected(self, temp_text_file):
        """Тест что batch_size соблюдается."""
        train_loader, _, _ = create_dataloaders(
            train_file=str(temp_text_file),
            batch_size=2,
            max_length=20
        )

        batch = next(iter(train_loader))

        # Размер batch должен быть <= 2 (может быть меньше в последнем batch)
        assert batch['input_ids'].shape[0] <= 2

    def test_shuffle_train(self, temp_text_file):
        """Тест что train loader shuffle."""
        train_loader, _, _ = create_dataloaders(
            train_file=str(temp_text_file),
            batch_size=1,
            max_length=20
        )

        # Train loader должен иметь shuffle=True
        assert train_loader.dataset is not None


# ============================================================================
# Integration тесты
# ============================================================================

@pytest.mark.integration
@pytest.mark.training
class TestDatasetIntegration:
    """Integration тесты для dataset pipeline."""

    def test_full_pipeline_txt(self, tmp_path):
        """Тест полного pipeline с .txt файлом."""
        # Создаём файл
        file_path = tmp_path / "train.txt"
        file_path.write_text("""
Это первый параграф для обучения модели.
Он содержит несколько предложений.

Второй параграф с другой информацией.
Тоже для тренировки.

Третий параграф добавляет разнообразия.
""", encoding='utf-8')

        # Создаём dataloader
        train_loader, _, tokenizer = create_dataloaders(
            train_file=str(file_path),
            batch_size=2,
            max_length=30
        )

        # Проверяем что всё работает
        assert len(tokenizer) > 4
        assert len(train_loader) > 0

        # Проходим по одному batch
        batch = next(iter(train_loader))
        assert batch['input_ids'].shape[0] <= 2
        assert batch['input_ids'].dtype == torch.long
        assert batch['labels'].dtype == torch.long

    def test_full_pipeline_jsonl(self, tmp_path):
        """Тест полного pipeline с .jsonl файлом."""
        # Создаём JSONL файл
        file_path = tmp_path / "train.jsonl"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({"text": "Первый пример текста"}, ensure_ascii=False) + '\n')
            f.write(json.dumps({"text": "Второй пример текста"}, ensure_ascii=False) + '\n')
            f.write(json.dumps({"text": "Третий пример для обучения"}, ensure_ascii=False) + '\n')

        train_loader, _, tokenizer = create_dataloaders(
            train_file=str(file_path),
            batch_size=2,
            max_length=20
        )

        assert len(tokenizer) > 4
        assert len(train_loader) > 0

    def test_encode_decode_roundtrip(self):
        """Тест roundtrip encode/decode."""
        tokenizer = SimpleTokenizer(vocab_size=100)
        texts = ["hello world python programming"]
        tokenizer.build_vocab(texts)

        original = "hello world"
        encoded = tokenizer.encode(original, add_special_tokens=True)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)

        assert decoded == original

    def test_dataset_with_various_lengths(self, tmp_path):
        """Тест dataset с текстами разной длины."""
        file_path = tmp_path / "var_len.txt"
        file_path.write_text("""
Короткий.

Средний текст с несколькими словами.

Очень длинный текст который содержит много слов и будет разбит на несколько chunks с помощью sliding window механизма.
""", encoding='utf-8')

        tokenizer = SimpleTokenizer(vocab_size=100)
        tokenizer.build_vocab(["короткий средний длинный текст слова много"])

        dataset = TextDataset(
            file_path=str(file_path),
            tokenizer=tokenizer,
            max_length=10,
            stride=5
        )

        # Должно быть несколько samples
        assert len(dataset) > 0

        # Все samples должны быть валидны
        for i in range(len(dataset)):
            sample = dataset[i]
            assert len(sample['input_ids']) > 0
            assert len(sample['input_ids']) <= 10

    def test_dataloader_multiple_epochs(self, temp_text_file):
        """Тест использования dataloader для нескольких эпох."""
        train_loader, _, _ = create_dataloaders(
            train_file=str(temp_text_file),
            batch_size=2,
            max_length=20
        )

        # Проходим 2 эпохи
        for epoch in range(2):
            batch_count = 0
            for batch in train_loader:
                assert batch['input_ids'].shape[0] > 0
                batch_count += 1

            assert batch_count > 0
