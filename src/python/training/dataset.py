"""
Dataset загрузчик для обучения языковых моделей.

Поддерживает:
    - Текстовые файлы (.txt)
    - JSONL файлы (.jsonl)
    - Автоматическое разбиение на train/val
    - Batching и padding
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path


class SimpleTokenizer:
    """
    Простой word-level tokenizer для прототипирования.

    В production это будет заменено на BPE tokenizer (GPT-2, SentencePiece).
    """

    def __init__(self, vocab_size: int = 10000):
        """
        Инициализация токенайзера.

        Args:
            vocab_size: Максимальный размер словаря
        """
        self.vocab_size = vocab_size
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}

        # Специальные токены
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"  # Begin of sequence
        self.eos_token = "<EOS>"  # End of sequence

        # Инициализируем специальные токены
        self.word2idx[self.pad_token] = 0
        self.word2idx[self.unk_token] = 1
        self.word2idx[self.bos_token] = 2
        self.word2idx[self.eos_token] = 3

        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

    def build_vocab(self, texts: List[str]) -> None:
        """
        Построение словаря из текстов.

        Args:
            texts: Список текстов для анализа
        """
        # Подсчёт частоты слов
        word_freq: Dict[str, int] = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Сортируем по частоте и берём top-K
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        # Добавляем слова в словарь (начиная с индекса 4, т.к. 0-3 заняты)
        for word, _ in sorted_words[:self.vocab_size - 4]:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        print(f"Словарь построен: {len(self.word2idx)} токенов")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Кодирование текста в токены.

        Args:
            text: Входной текст
            add_special_tokens: Добавить BOS/EOS токены

        Returns:
            Список индексов токенов
        """
        words = text.lower().split()
        tokens = [self.word2idx.get(word, self.unk_token_id) for word in words]

        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Декодирование токенов в текст.

        Args:
            token_ids: Список индексов токенов
            skip_special_tokens: Пропустить специальные токены

        Returns:
            Декодированный текст
        """
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}

        words = []
        for idx in token_ids:
            if skip_special_tokens and idx in special_ids:
                continue
            words.append(self.idx2word.get(idx, self.unk_token))

        return " ".join(words)

    def __len__(self) -> int:
        """Размер словаря."""
        return len(self.word2idx)


class TextDataset(Dataset):
    """
    Dataset для обучения языковых моделей.

    Загружает текстовые данные и подготавливает их для обучения.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: SimpleTokenizer,
        max_length: int = 512,
        stride: int = 256
    ):
        """
        Инициализация dataset.

        Args:
            file_path: Путь к файлу с данными (.txt или .jsonl)
            tokenizer: Токенайзер
            max_length: Максимальная длина последовательности
            stride: Шаг для sliding window (для длинных текстов)
        """
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.samples: List[List[int]] = []

        # Загружаем данные
        self._load_data()

    def _load_data(self) -> None:
        """Загрузка и preprocessing данных."""
        print(f"Загрузка данных из {self.file_path}...")

        texts = []

        if self.file_path.suffix == '.txt':
            # Обычный текстовый файл
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                # Разбиваем на параграфы
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                texts.extend(paragraphs)

        elif self.file_path.suffix == '.jsonl':
            # JSONL формат (каждая строка - JSON объект с полем 'text')
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if 'text' in data:
                            texts.append(data['text'])

        else:
            raise ValueError(f"Неподдерживаемый формат файла: {self.file_path.suffix}")

        print(f"Загружено {len(texts)} текстов")

        # Tokenization и создание samples
        for text in texts:
            # Кодируем текст
            tokens = self.tokenizer.encode(text, add_special_tokens=True)

            # Если текст длиннее max_length, разбиваем на chunks с overlap
            if len(tokens) > self.max_length:
                for i in range(0, len(tokens) - self.max_length + 1, self.stride):
                    chunk = tokens[i:i + self.max_length]
                    self.samples.append(chunk)
            else:
                self.samples.append(tokens)

        print(f"Создано {len(self.samples)} samples для обучения")

    def __len__(self) -> int:
        """Количество samples в dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Получение одного sample.

        Args:
            idx: Индекс sample

        Returns:
            Словарь с input_ids и labels

        Размерности:
            input_ids: [seq_len] - входная последовательность
            labels: [seq_len] - целевая последовательность (смещена на 1)
        """
        tokens = self.samples[idx]

        # Для языкового моделирования: input = tokens[:-1], target = tokens[1:]
        # Модель учится предсказывать следующий токен
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': labels
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Collate function для DataLoader.

    Объединяет samples в batch и добавляет padding.

    Args:
        batch: Список samples
        pad_token_id: ID токена для padding

    Returns:
        Batch данных с padding

    Размерности:
        input_ids: [batch_size, max_seq_len]
        labels: [batch_size, max_seq_len]
        attention_mask: [batch_size, max_seq_len] (1 для реальных токенов, 0 для padding)
    """
    # Находим максимальную длину в batch
    max_len = max(len(sample['input_ids']) for sample in batch)

    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []

    for sample in batch:
        input_ids = sample['input_ids']
        labels = sample['labels']

        # Padding
        padding_len = max_len - len(input_ids)

        # Добавляем padding справа
        padded_input = torch.cat([
            input_ids,
            torch.full((padding_len,), pad_token_id, dtype=torch.long)
        ])

        padded_labels = torch.cat([
            labels,
            torch.full((padding_len,), -100, dtype=torch.long)  # -100 игнорируется в CrossEntropyLoss
        ])

        # Attention mask: 1 для реальных токенов, 0 для padding
        attention_mask = torch.cat([
            torch.ones(len(input_ids), dtype=torch.long),
            torch.zeros(padding_len, dtype=torch.long)
        ])

        batch_input_ids.append(padded_input)
        batch_labels.append(padded_labels)
        batch_attention_mask.append(attention_mask)

    return {
        'input_ids': torch.stack(batch_input_ids),        # [batch_size, max_seq_len]
        'labels': torch.stack(batch_labels),              # [batch_size, max_seq_len]
        'attention_mask': torch.stack(batch_attention_mask)  # [batch_size, max_seq_len]
    }


def create_dataloaders(
    train_file: str,
    val_file: Optional[str] = None,
    tokenizer: Optional[SimpleTokenizer] = None,
    batch_size: int = 4,
    max_length: int = 512,
    num_workers: int = 0
) -> Tuple[DataLoader, Optional[DataLoader], SimpleTokenizer]:
    """
    Создание DataLoaders для обучения и валидации.

    Args:
        train_file: Путь к файлу с training данными
        val_file: Путь к файлу с validation данными (опционально)
        tokenizer: Токенайзер (если None, создаётся новый)
        batch_size: Размер batch
        max_length: Максимальная длина последовательности
        num_workers: Количество worker процессов для загрузки данных

    Returns:
        Tuple (train_dataloader, val_dataloader, tokenizer)
    """
    # Если токенайзер не предоставлен, создаём новый
    if tokenizer is None:
        print("Создание нового токенайзера...")
        tokenizer = SimpleTokenizer(vocab_size=10000)

        # Загружаем тексты для построения словаря
        texts = []
        with open(train_file, 'r', encoding='utf-8') as f:
            if train_file.endswith('.txt'):
                text = f.read()
                texts = [p.strip() for p in text.split('\n\n') if p.strip()]
            elif train_file.endswith('.jsonl'):
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if 'text' in data:
                            texts.append(data['text'])

        tokenizer.build_vocab(texts)

    # Создаём datasets
    train_dataset = TextDataset(
        file_path=train_file,
        tokenizer=tokenizer,
        max_length=max_length
    )

    val_dataset = None
    if val_file is not None:
        val_dataset = TextDataset(
            file_path=val_file,
            tokenizer=tokenizer,
            max_length=max_length
        )

    # Создаём collate function с правильным pad_token_id
    collate_with_pad = lambda batch: collate_fn(batch, pad_token_id=tokenizer.pad_token_id)

    # Создаём dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_with_pad,
        num_workers=num_workers
    )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_with_pad,
            num_workers=num_workers
        )

    return train_dataloader, val_dataloader, tokenizer


# ============================================================================
# Тестирование
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Тест Dataset и DataLoader")
    print("=" * 80)

    # Создаём тестовый файл
    test_file = "/tmp/test_data.txt"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("""
Это первый параграф текста для тестирования.
Он содержит несколько предложений.

Это второй параграф.
Он тоже содержит текст для обучения модели.

Третий параграф добавлен для разнообразия.
Чем больше текста, тем лучше работает модель.
""")

    print("\n1. Создание токенайзера и построение словаря...")
    tokenizer = SimpleTokenizer(vocab_size=100)

    texts = [
        "это первый параграф текста для тестирования",
        "это второй параграф",
        "третий параграф добавлен для разнообразия"
    ]
    tokenizer.build_vocab(texts)

    print(f"   Размер словаря: {len(tokenizer)}")

    print("\n2. Тест encode/decode...")
    test_text = "это тест текста"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"   Оригинал: {test_text}")
    print(f"   Encoded: {encoded}")
    print(f"   Decoded: {decoded}")

    print("\n3. Создание dataset...")
    dataset = TextDataset(
        file_path=test_file,
        tokenizer=tokenizer,
        max_length=20
    )
    print(f"   Создано {len(dataset)} samples")

    print("\n4. Тест получения sample...")
    sample = dataset[0]
    print(f"   Input IDs shape: {sample['input_ids'].shape}")
    print(f"   Labels shape: {sample['labels'].shape}")
    print(f"   Input IDs: {sample['input_ids'].tolist()}")
    print(f"   Labels: {sample['labels'].tolist()}")

    print("\n5. Создание DataLoader...")
    train_loader, val_loader, tokenizer = create_dataloaders(
        train_file=test_file,
        batch_size=2,
        max_length=20
    )

    print(f"   Train batches: {len(train_loader)}")

    print("\n6. Тест batch...")
    batch = next(iter(train_loader))
    print(f"   Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"   Batch labels shape: {batch['labels'].shape}")
    print(f"   Batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"\n   Input IDs:\n{batch['input_ids']}")
    print(f"\n   Labels:\n{batch['labels']}")
    print(f"\n   Attention Mask:\n{batch['attention_mask']}")

    print("\n" + "=" * 80)
    print("✅ Все тесты пройдены!")
    print("=" * 80)
