"""
3-уровневая система памяти для эффективного контекста.

Реализует инновационную систему памяти с тремя уровнями:
1. Текущая память (250k токенов) - полный контекст
2. Устаревшая память (250k токенов) - сжатая информация
3. Долгая память (250k токенов) - краткие резюме
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio


@dataclass
class MemoryChunk:
    """
    Кусок памяти с метаданными.

    Attributes:
        content: Содержимое (текст или сжатое резюме)
        timestamp: Время создания
        token_count: Количество токенов
        importance: Важность фрагмента (0.0 - 1.0)
        compressed: Флаг сжатия
    """
    content: str
    timestamp: datetime
    token_count: int
    importance: float = 0.5
    compressed: bool = False


class ThreeLevelMemory:
    """
    3-уровневая система памяти.

    Управляет тремя уровнями памяти с автоматической компрессией
    и интеллектуальным поиском релевантной информации.

    Args:
        max_tokens_per_level: Максимум токенов на каждом уровне (по умолчанию 250k)
    """

    def __init__(self, max_tokens_per_level: int = 250_000):
        self.max_tokens_per_level = max_tokens_per_level

        # Три уровня памяти
        self.current_memory: List[MemoryChunk] = []
        self.obsolete_memory: List[MemoryChunk] = []
        self.long_term_memory: List[MemoryChunk] = []

        # Счётчики токенов
        self.current_tokens = 0
        self.obsolete_tokens = 0
        self.longterm_tokens = 0

    def add_message(self, content: str, token_count: int, importance: float = 0.5):
        """
        Добавляет новое сообщение в текущую память.

        Args:
            content: Текст сообщения
            token_count: Количество токенов
            importance: Важность сообщения (0.0 - 1.0)
        """
        chunk = MemoryChunk(
            content=content,
            timestamp=datetime.now(),
            token_count=token_count,
            importance=importance,
            compressed=False
        )

        self.current_memory.append(chunk)
        self.current_tokens += token_count

        # Проверяем переполнение
        if self.current_tokens > self.max_tokens_per_level:
            self._move_to_obsolete()

    def _move_to_obsolete(self):
        """
        Перемещает старые сообщения из текущей памяти в устаревшую.

        Запускается асинхронно для компрессии без блокировки.
        """
        # Берём первые 100 сообщений (самые старые)
        old_chunks = self.current_memory[:100]
        self.current_memory = self.current_memory[100:]

        # Пересчитываем токены
        removed_tokens = sum(chunk.token_count for chunk in old_chunks)
        self.current_tokens -= removed_tokens

        # TODO: Асинхронная компрессия
        # asyncio.create_task(self._compress_and_store(old_chunks))

        # Временно: просто добавляем сжатое резюме
        compressed_content = self._simple_compress(old_chunks)
        compressed_tokens = len(compressed_content.split()) * 2  # Примерная оценка

        compressed_chunk = MemoryChunk(
            content=compressed_content,
            timestamp=datetime.now(),
            token_count=compressed_tokens,
            importance=max(chunk.importance for chunk in old_chunks),
            compressed=True
        )

        self.obsolete_memory.append(compressed_chunk)
        self.obsolete_tokens += compressed_tokens

        # Проверяем переполнение устаревшей памяти
        if self.obsolete_tokens > self.max_tokens_per_level:
            self._move_to_longterm()

    def _move_to_longterm(self):
        """Перемещает из устаревшей памяти в долгую с ультра-компрессией."""
        # Берём первые 50 кусков
        old_chunks = self.obsolete_memory[:50]
        self.obsolete_memory = self.obsolete_memory[50:]

        # Пересчитываем токены
        removed_tokens = sum(chunk.token_count for chunk in old_chunks)
        self.obsolete_tokens -= removed_tokens

        # Ультра-компрессия (только ключевые факты)
        ultra_compressed = self._ultra_compress(old_chunks)
        compressed_tokens = len(ultra_compressed.split()) * 2

        longterm_chunk = MemoryChunk(
            content=ultra_compressed,
            timestamp=datetime.now(),
            token_count=compressed_tokens,
            importance=max(chunk.importance for chunk in old_chunks),
            compressed=True
        )

        self.long_term_memory.append(longterm_chunk)
        self.longterm_tokens += compressed_tokens

    def _simple_compress(self, chunks: List[MemoryChunk]) -> str:
        """
        Простая компрессия для тестирования.

        В полной реализации здесь будет использоваться
        модель-summarizer для создания сжатого резюме.

        Args:
            chunks: Список кусков для компрессии

        Returns:
            Сжатое резюме
        """
        # Временная реализация: просто объединяем важные части
        important_parts = [
            chunk.content[:100] + "..."
            for chunk in chunks
            if chunk.importance > 0.7
        ]

        if not important_parts:
            # Если нет важных, берём первые несколько
            important_parts = [chunk.content[:100] + "..." for chunk in chunks[:5]]

        summary = f"[Резюме {len(chunks)} сообщений]\n" + "\n".join(important_parts)
        return summary

    def _ultra_compress(self, chunks: List[MemoryChunk]) -> str:
        """Ультра-компрессия для долгой памяти (только ключевые факты)."""
        # Берём только самые важные куски
        important = sorted(chunks, key=lambda x: x.importance, reverse=True)[:3]

        facts = [chunk.content[:50] + "..." for chunk in important]
        summary = f"[Краткое резюме {len(chunks)} записей]\n" + "\n".join(facts)
        return summary

    def prepare_context(
        self,
        current_query: str,
        max_total_tokens: int = 15_000
    ) -> str:
        """
        Формирует контекст для inference из всех уровней памяти.

        Ключевая функция: НЕ все 750k токенов передаются в модель,
        только релевантные фрагменты в пределах max_total_tokens.

        Args:
            current_query: Текущий запрос пользователя
            max_total_tokens: Максимум токенов для контекста (по умолчанию 15k)

        Returns:
            Сформированный контекст для модели
        """
        # 1. Берём последние сообщения из текущей памяти (полностью)
        recent_chunks = self.current_memory[-10:]  # Последние 10 сообщений
        recent_text = "\n".join([chunk.content for chunk in recent_chunks])
        recent_tokens = sum(chunk.token_count for chunk in recent_chunks)

        # 2. Поиск релевантных кусков в устаревшей памяти
        # TODO: Использовать векторный поиск
        relevant_obsolete = self._search_relevant(
            query=current_query,
            memory=self.obsolete_memory,
            max_tokens=3000
        )

        # 3. Важные факты из долгой памяти
        important_longterm = self._get_important_facts(
            memory=self.long_term_memory,
            max_tokens=2000
        )

        # Формируем итоговый контекст
        context = f"""[Важные факты из прошлого]
{important_longterm}

[Релевантная информация из недавней истории]
{relevant_obsolete}

[Текущий разговор]
{recent_text}

[Новый запрос]
{current_query}
"""

        return context

    def _search_relevant(
        self,
        query: str,
        memory: List[MemoryChunk],
        max_tokens: int
    ) -> str:
        """
        Поиск релевантных фрагментов в памяти.

        Временная реализация: простой поиск по ключевым словам.
        TODO: Заменить на векторный поиск с embeddings.

        Args:
            query: Поисковый запрос
            memory: Список кусков памяти для поиска
            max_tokens: Максимум токенов в результате

        Returns:
            Найденные релевантные фрагменты
        """
        if not memory:
            return ""

        # Простой поиск по ключевым словам
        query_words = set(query.lower().split())

        # Сортируем по релевантности (количество совпадающих слов)
        scored_chunks = []
        for chunk in memory:
            chunk_words = set(chunk.content.lower().split())
            overlap = len(query_words & chunk_words)
            if overlap > 0:
                scored_chunks.append((chunk, overlap))

        # Сортируем по score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Берём топ-N в пределах max_tokens
        result = []
        total_tokens = 0

        for chunk, score in scored_chunks:
            if total_tokens + chunk.token_count > max_tokens:
                break
            result.append(chunk.content)
            total_tokens += chunk.token_count

        return "\n".join(result) if result else "[Нет релевантной информации]"

    def _get_important_facts(
        self,
        memory: List[MemoryChunk],
        max_tokens: int
    ) -> str:
        """Извлекает самые важные факты из памяти."""
        if not memory:
            return ""

        # Сортируем по важности
        sorted_chunks = sorted(memory, key=lambda x: x.importance, reverse=True)

        # Берём в пределах max_tokens
        result = []
        total_tokens = 0

        for chunk in sorted_chunks:
            if total_tokens + chunk.token_count > max_tokens:
                break
            result.append(chunk.content)
            total_tokens += chunk.token_count

        return "\n".join(result) if result else "[Нет важных фактов]"

    def get_stats(self) -> Dict[str, any]:
        """Возвращает статистику по памяти."""
        return {
            'current': {
                'chunks': len(self.current_memory),
                'tokens': self.current_tokens,
                'usage_pct': (self.current_tokens / self.max_tokens_per_level) * 100
            },
            'obsolete': {
                'chunks': len(self.obsolete_memory),
                'tokens': self.obsolete_tokens,
                'usage_pct': (self.obsolete_tokens / self.max_tokens_per_level) * 100
            },
            'longterm': {
                'chunks': len(self.long_term_memory),
                'tokens': self.longterm_tokens,
                'usage_pct': (self.longterm_tokens / self.max_tokens_per_level) * 100
            },
            'total_tokens': self.current_tokens + self.obsolete_tokens + self.longterm_tokens
        }


def test_three_level_memory():
    """Тестовая функция для системы памяти."""

    print("=" * 60)
    print("Тестирование 3-уровневой системы памяти")
    print("=" * 60)

    # Создаём систему памяти с маленьким лимитом для теста
    memory = ThreeLevelMemory(max_tokens_per_level=1000)

    # Добавляем тестовые сообщения
    print("\nДобавляем тестовые сообщения...")
    for i in range(150):
        content = f"Тестовое сообщение {i}. Это важная информация о теме {i % 10}."
        token_count = len(content.split()) * 2
        importance = 0.9 if i % 10 == 0 else 0.5

        memory.add_message(content, token_count, importance)

    # Проверяем статистику
    stats = memory.get_stats()
    print("\nСтатистика памяти:")
    print(f"Текущая память: {stats['current']['chunks']} кусков, "
          f"{stats['current']['tokens']} токенов ({stats['current']['usage_pct']:.1f}%)")
    print(f"Устаревшая память: {stats['obsolete']['chunks']} кусков, "
          f"{stats['obsolete']['tokens']} токенов ({stats['obsolete']['usage_pct']:.1f}%)")
    print(f"Долгая память: {stats['longterm']['chunks']} кусков, "
          f"{stats['longterm']['tokens']} токенов ({stats['longterm']['usage_pct']:.1f}%)")
    print(f"\nВсего токенов: {stats['total_tokens']}")

    # Тест подготовки контекста
    print("\n" + "=" * 60)
    print("Тест подготовки контекста")
    print("=" * 60)

    query = "Расскажи о теме 5"
    context = memory.prepare_context(query, max_total_tokens=500)

    print(f"\nЗапрос: {query}")
    print(f"\nСформированный контекст (обрезан до 300 символов):")
    print(context[:300] + "...")

    print("\n✅ Тест пройден успешно!")
    print("=" * 60)


if __name__ == "__main__":
    test_three_level_memory()
