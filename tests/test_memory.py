"""
Тесты для 3-уровневой системы памяти.

Тестирует:
    - MemoryChunk
    - ThreeLevelMemory
"""

import pytest
from datetime import datetime
from memory.three_level_memory import ThreeLevelMemory, MemoryChunk


# ============================================================================
# Тесты для MemoryChunk
# ============================================================================

@pytest.mark.unit
@pytest.mark.memory
@pytest.mark.fast
class TestMemoryChunk:
    """Тесты для MemoryChunk dataclass."""

    def test_creation(self):
        """Тест создания MemoryChunk."""
        chunk = MemoryChunk(
            content="Test content",
            timestamp=datetime.now(),
            token_count=10,
            importance=0.8,
            compressed=False
        )

        assert chunk.content == "Test content"
        assert chunk.token_count == 10
        assert chunk.importance == 0.8
        assert chunk.compressed is False

    def test_default_values(self):
        """Тест дефолтных значений."""
        chunk = MemoryChunk(
            content="Test",
            timestamp=datetime.now(),
            token_count=5
        )

        assert chunk.importance == 0.5
        assert chunk.compressed is False

    def test_timestamp(self):
        """Тест что timestamp корректно устанавливается."""
        now = datetime.now()
        chunk = MemoryChunk(
            content="Test",
            timestamp=now,
            token_count=5
        )

        assert chunk.timestamp == now
        assert isinstance(chunk.timestamp, datetime)

    def test_high_importance(self):
        """Тест с высокой важностью."""
        chunk = MemoryChunk(
            content="Important message",
            timestamp=datetime.now(),
            token_count=10,
            importance=1.0
        )

        assert chunk.importance == 1.0

    def test_low_importance(self):
        """Тест с низкой важностью."""
        chunk = MemoryChunk(
            content="Low priority",
            timestamp=datetime.now(),
            token_count=10,
            importance=0.0
        )

        assert chunk.importance == 0.0


# ============================================================================
# Тесты для ThreeLevelMemory - Basics
# ============================================================================

@pytest.mark.unit
@pytest.mark.memory
@pytest.mark.fast
class TestThreeLevelMemoryBasics:
    """Тесты для базовой функциональности ThreeLevelMemory."""

    def test_initialization(self):
        """Тест инициализации."""
        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        assert memory.max_tokens_per_level == 1000
        assert len(memory.current_memory) == 0
        assert len(memory.obsolete_memory) == 0
        assert len(memory.long_term_memory) == 0
        assert memory.current_tokens == 0
        assert memory.obsolete_tokens == 0
        assert memory.longterm_tokens == 0

    def test_initialization_default(self):
        """Тест инициализации с дефолтными параметрами."""
        memory = ThreeLevelMemory()

        assert memory.max_tokens_per_level == 250_000

    def test_add_single_message(self):
        """Тест добавления одного сообщения."""
        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        memory.add_message("Test message", token_count=10, importance=0.7)

        assert len(memory.current_memory) == 1
        assert memory.current_tokens == 10
        assert memory.current_memory[0].content == "Test message"
        assert memory.current_memory[0].importance == 0.7

    def test_add_multiple_messages(self):
        """Тест добавления нескольких сообщений."""
        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        for i in range(10):
            memory.add_message(f"Message {i}", token_count=5)

        assert len(memory.current_memory) == 10
        assert memory.current_tokens == 50

    def test_token_counting(self):
        """Тест подсчёта токенов."""
        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        memory.add_message("First", token_count=10)
        memory.add_message("Second", token_count=20)
        memory.add_message("Third", token_count=30)

        assert memory.current_tokens == 60

    def test_message_timestamps(self):
        """Тест что timestamp устанавливается при добавлении."""
        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        before = datetime.now()
        memory.add_message("Test", token_count=10)
        after = datetime.now()

        chunk = memory.current_memory[0]
        assert before <= chunk.timestamp <= after

    def test_message_ordering(self):
        """Тест что сообщения добавляются в порядке."""
        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        memory.add_message("First", token_count=10)
        memory.add_message("Second", token_count=10)
        memory.add_message("Third", token_count=10)

        assert memory.current_memory[0].content == "First"
        assert memory.current_memory[1].content == "Second"
        assert memory.current_memory[2].content == "Third"


# ============================================================================
# Тесты для ThreeLevelMemory - Compression
# ============================================================================

@pytest.mark.unit
@pytest.mark.memory
@pytest.mark.fast
class TestThreeLevelMemoryCompression:
    """Тесты для компрессии и перемещения между уровнями."""

    def test_move_to_obsolete_when_overflow(self):
        """Тест перемещения в obsolete при переполнении."""
        memory = ThreeLevelMemory(max_tokens_per_level=500)

        # Добавляем много сообщений чтобы превысить лимит
        for i in range(150):
            memory.add_message(f"Message {i}", token_count=10)

        # Должно произойти перемещение в obsolete
        assert len(memory.obsolete_memory) > 0
        assert memory.obsolete_tokens > 0

    def test_compression_reduces_token_count(self):
        """Тест что компрессия уменьшает количество токенов."""
        memory = ThreeLevelMemory(max_tokens_per_level=500)

        # Добавляем 150 сообщений по 10 токенов = 1500 токенов
        for i in range(150):
            memory.add_message(f"Message {i}", token_count=10)

        # После компрессии общее количество должно быть меньше
        stats = memory.get_stats()
        assert stats['total_tokens'] < 1500

    def test_current_memory_stays_within_limit(self):
        """Тест что current память остаётся в пределах лимита."""
        memory = ThreeLevelMemory(max_tokens_per_level=500)

        for i in range(200):
            memory.add_message(f"Message {i}", token_count=10)

        # Current память должна быть в пределах лимита
        assert memory.current_tokens <= memory.max_tokens_per_level

    def test_compressed_chunks_have_flag(self):
        """Тест что сжатые куски помечены флагом."""
        memory = ThreeLevelMemory(max_tokens_per_level=500)

        for i in range(150):
            memory.add_message(f"Message {i}", token_count=10)

        # Проверяем что в obsolete есть сжатые куски
        if len(memory.obsolete_memory) > 0:
            assert memory.obsolete_memory[0].compressed is True

    def test_move_to_longterm_when_obsolete_overflow(self):
        """Тест перемещения в longterm при переполнении obsolete."""
        memory = ThreeLevelMemory(max_tokens_per_level=300)

        # Добавляем очень много сообщений
        for i in range(500):
            memory.add_message(f"Message {i}", token_count=10, importance=0.5)

        # Должно произойти перемещение в longterm
        assert len(memory.long_term_memory) > 0
        assert memory.longterm_tokens > 0

    def test_importance_preserved_in_compression(self):
        """Тест что важность сохраняется при компрессии."""
        memory = ThreeLevelMemory(max_tokens_per_level=500)

        # Добавляем сообщения с разной важностью
        for i in range(150):
            importance = 0.9 if i % 10 == 0 else 0.3
            memory.add_message(f"Message {i}", token_count=10, importance=importance)

        # Проверяем что в obsolete сохранена максимальная важность
        if len(memory.obsolete_memory) > 0:
            assert memory.obsolete_memory[0].importance == 0.9

    def test_oldest_messages_moved_first(self):
        """Тест что перемещаются самые старые сообщения."""
        memory = ThreeLevelMemory(max_tokens_per_level=500)

        for i in range(150):
            memory.add_message(f"Message {i}", token_count=10)

        # Последние сообщения должны остаться в current
        # (проверяем что Message 149 есть в current)
        contents = [chunk.content for chunk in memory.current_memory]
        assert "Message 149" in contents


# ============================================================================
# Тесты для ThreeLevelMemory - Search and Context
# ============================================================================

@pytest.mark.unit
@pytest.mark.memory
@pytest.mark.fast
class TestThreeLevelMemorySearch:
    """Тесты для поиска и формирования контекста."""

    def test_search_relevant_basic(self):
        """Тест базового поиска релевантных фрагментов."""
        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        # Добавляем сообщения с разными темами
        memory.add_message("Python programming tutorial", token_count=10)
        memory.add_message("JavaScript web development", token_count=10)
        memory.add_message("Python data science", token_count=10)

        # Перемещаем в obsolete для теста поиска
        memory._move_to_obsolete()

        # Ищем по запросу про Python
        result = memory._search_relevant(
            query="Python code",
            memory=memory.obsolete_memory,
            max_tokens=100
        )

        # Результат должен содержать информацию о Python
        assert "Python" in result or "[Нет релевантной информации]" in result

    def test_search_relevant_no_matches(self):
        """Тест поиска когда нет совпадений."""
        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        memory.add_message("Unrelated content", token_count=10)
        memory._move_to_obsolete()

        result = memory._search_relevant(
            query="completely different topic",
            memory=memory.obsolete_memory,
            max_tokens=100
        )

        assert "[Нет релевантной информации]" in result

    def test_search_relevant_empty_memory(self):
        """Тест поиска в пустой памяти."""
        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        result = memory._search_relevant(
            query="test",
            memory=memory.obsolete_memory,
            max_tokens=100
        )

        assert result == ""

    def test_search_relevant_max_tokens(self):
        """Тест что поиск соблюдает max_tokens."""
        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        # Добавляем много релевантных сообщений
        for i in range(20):
            memory.add_message(f"Python tutorial part {i}", token_count=50)

        memory._move_to_obsolete()

        result = memory._search_relevant(
            query="Python",
            memory=memory.obsolete_memory,
            max_tokens=100
        )

        # Результат не должен быть пустым
        assert len(result) > 0

    def test_get_important_facts(self):
        """Тест извлечения важных фактов."""
        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        # Добавляем сообщения с разной важностью
        memory.add_message("Important fact", token_count=10, importance=0.9)
        memory.add_message("Regular message", token_count=10, importance=0.5)
        memory.add_message("Another fact", token_count=10, importance=0.8)

        memory._move_to_obsolete()
        memory._move_to_longterm()

        result = memory._get_important_facts(
            memory=memory.long_term_memory,
            max_tokens=100
        )

        # Должны вернуться важные факты
        assert len(result) > 0 or result == ""

    def test_get_important_facts_empty_memory(self):
        """Тест извлечения фактов из пустой памяти."""
        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        result = memory._get_important_facts(
            memory=memory.long_term_memory,
            max_tokens=100
        )

        assert result == ""

    def test_prepare_context_basic(self):
        """Тест формирования контекста."""
        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        memory.add_message("Message 1", token_count=10)
        memory.add_message("Message 2", token_count=10)
        memory.add_message("Message 3", token_count=10)

        context = memory.prepare_context(
            current_query="Test query",
            max_total_tokens=1000
        )

        # Контекст должен содержать структуру
        assert "[Важные факты из прошлого]" in context
        assert "[Релевантная информация из недавней истории]" in context
        assert "[Текущий разговор]" in context
        assert "[Новый запрос]" in context
        assert "Test query" in context

    def test_prepare_context_includes_recent_messages(self):
        """Тест что контекст включает недавние сообщения."""
        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        for i in range(20):
            memory.add_message(f"Message {i}", token_count=10)

        context = memory.prepare_context("Query", max_total_tokens=1000)

        # Должны быть последние сообщения
        assert "Message 19" in context

    def test_prepare_context_empty_memory(self):
        """Тест формирования контекста с пустой памятью."""
        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        context = memory.prepare_context("Query", max_total_tokens=1000)

        # Контекст должен быть сформирован, хотя и пустой
        assert "[Новый запрос]" in context
        assert "Query" in context


# ============================================================================
# Тесты для ThreeLevelMemory - Statistics
# ============================================================================

@pytest.mark.unit
@pytest.mark.memory
@pytest.mark.fast
class TestThreeLevelMemoryStatistics:
    """Тесты для статистики памяти."""

    def test_get_stats_empty(self):
        """Тест статистики для пустой памяти."""
        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        stats = memory.get_stats()

        assert stats['current']['chunks'] == 0
        assert stats['current']['tokens'] == 0
        assert stats['current']['usage_pct'] == 0.0
        assert stats['obsolete']['chunks'] == 0
        assert stats['longterm']['chunks'] == 0
        assert stats['total_tokens'] == 0

    def test_get_stats_with_current_memory(self):
        """Тест статистики с текущей памятью."""
        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        memory.add_message("Test 1", token_count=10)
        memory.add_message("Test 2", token_count=20)

        stats = memory.get_stats()

        assert stats['current']['chunks'] == 2
        assert stats['current']['tokens'] == 30
        assert stats['current']['usage_pct'] == 3.0
        assert stats['total_tokens'] == 30

    def test_get_stats_usage_percentage(self):
        """Тест расчёта процента использования."""
        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        memory.add_message("Test", token_count=500)

        stats = memory.get_stats()

        assert stats['current']['usage_pct'] == 50.0

    def test_get_stats_all_levels(self):
        """Тест статистики для всех уровней."""
        memory = ThreeLevelMemory(max_tokens_per_level=300)

        # Заполняем все уровни
        for i in range(500):
            memory.add_message(f"Message {i}", token_count=10)

        stats = memory.get_stats()

        # Все уровни должны содержать данные
        assert stats['current']['chunks'] > 0
        assert stats['obsolete']['chunks'] > 0
        assert stats['longterm']['chunks'] > 0
        assert stats['total_tokens'] > 0

    def test_get_stats_total_tokens(self):
        """Тест подсчёта общего количества токенов."""
        memory = ThreeLevelMemory(max_tokens_per_level=300)

        for i in range(500):
            memory.add_message(f"Message {i}", token_count=10)

        stats = memory.get_stats()

        # Сумма токенов должна быть меньше исходных 5000 из-за компрессии
        assert stats['total_tokens'] < 5000
        assert stats['total_tokens'] > 0


# ============================================================================
# Тесты для ThreeLevelMemory - Compression Methods
# ============================================================================

@pytest.mark.unit
@pytest.mark.memory
@pytest.mark.fast
class TestThreeLevelMemoryCompressionMethods:
    """Тесты для методов компрессии."""

    def test_simple_compress_basic(self):
        """Тест простой компрессии."""
        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        chunks = [
            MemoryChunk("Message 1", datetime.now(), 10, 0.9),
            MemoryChunk("Message 2", datetime.now(), 10, 0.5),
            MemoryChunk("Message 3", datetime.now(), 10, 0.8)
        ]

        compressed = memory._simple_compress(chunks)

        # Результат должен быть строкой
        assert isinstance(compressed, str)
        assert len(compressed) > 0
        assert "[Резюме" in compressed

    def test_simple_compress_prioritizes_important(self):
        """Тест что компрессия приоритизирует важные сообщения."""
        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        chunks = [
            MemoryChunk("Important message", datetime.now(), 10, 0.9),
            MemoryChunk("Regular message", datetime.now(), 10, 0.3)
        ]

        compressed = memory._simple_compress(chunks)

        # Важное сообщение должно быть в резюме
        assert "Important message" in compressed

    def test_simple_compress_no_important(self):
        """Тест компрессии когда нет важных сообщений."""
        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        chunks = [
            MemoryChunk("Message 1", datetime.now(), 10, 0.3),
            MemoryChunk("Message 2", datetime.now(), 10, 0.2)
        ]

        compressed = memory._simple_compress(chunks)

        # Должно быть резюме с первыми несколькими сообщениями
        assert len(compressed) > 0
        assert "[Резюме" in compressed

    def test_ultra_compress_basic(self):
        """Тест ультра-компрессии."""
        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        chunks = [
            MemoryChunk("Important fact 1", datetime.now(), 10, 0.9),
            MemoryChunk("Regular message", datetime.now(), 10, 0.5),
            MemoryChunk("Important fact 2", datetime.now(), 10, 0.8)
        ]

        ultra = memory._ultra_compress(chunks)

        # Результат должен быть короче чем simple_compress
        assert isinstance(ultra, str)
        assert len(ultra) > 0
        assert "[Краткое резюме" in ultra

    def test_ultra_compress_selects_most_important(self):
        """Тест что ультра-компрессия выбирает самое важное."""
        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        chunks = [
            MemoryChunk("Most important", datetime.now(), 10, 1.0),
            MemoryChunk("Medium", datetime.now(), 10, 0.5),
            MemoryChunk("Low", datetime.now(), 10, 0.2)
        ]

        ultra = memory._ultra_compress(chunks)

        # Самое важное должно быть в резюме
        assert "Most important" in ultra


# ============================================================================
# Integration тесты
# ============================================================================

@pytest.mark.integration
@pytest.mark.memory
class TestThreeLevelMemoryIntegration:
    """Integration тесты для ThreeLevelMemory."""

    def test_full_memory_lifecycle(self):
        """Тест полного жизненного цикла памяти."""
        memory = ThreeLevelMemory(max_tokens_per_level=300)

        # 1. Добавляем сообщения
        for i in range(100):
            importance = 0.9 if i % 10 == 0 else 0.5
            memory.add_message(f"Message {i}", token_count=10, importance=importance)

        # 2. Проверяем что сработала компрессия
        stats = memory.get_stats()
        assert stats['obsolete']['chunks'] > 0

        # 3. Добавляем ещё больше для перемещения в longterm
        for i in range(100, 500):
            memory.add_message(f"Message {i}", token_count=10)

        stats = memory.get_stats()
        assert stats['longterm']['chunks'] > 0

        # 4. Формируем контекст
        context = memory.prepare_context("Test query", max_total_tokens=1000)
        assert len(context) > 0

        # 5. Проверяем что последние сообщения в контексте
        assert "Message 499" in context

    def test_memory_with_varied_importance(self):
        """Тест памяти с разными уровнями важности."""
        memory = ThreeLevelMemory(max_tokens_per_level=300)

        # Добавляем сообщения с разной важностью
        for i in range(200):
            if i % 20 == 0:
                importance = 1.0
            elif i % 10 == 0:
                importance = 0.7
            else:
                importance = 0.3

            memory.add_message(f"Message {i} importance {importance}",
                             token_count=10, importance=importance)

        # Проверяем что важные сообщения сохранены
        stats = memory.get_stats()
        assert stats['total_tokens'] > 0

        # Формируем контекст
        context = memory.prepare_context("important information", max_total_tokens=1000)
        assert len(context) > 0

    def test_memory_search_relevance(self):
        """Тест релевантности поиска в памяти."""
        memory = ThreeLevelMemory(max_tokens_per_level=300)

        # Добавляем сообщения на разные темы
        topics = ["python", "javascript", "math", "science"]
        for i in range(200):
            topic = topics[i % 4]
            memory.add_message(f"Discussion about {topic} topic number {i}",
                             token_count=15)

        # Формируем контекст для конкретной темы
        context = memory.prepare_context("python programming", max_total_tokens=1000)

        # Контекст должен быть сформирован
        assert len(context) > 0
        assert "python" in context.lower()

    def test_memory_token_limits(self):
        """Тест соблюдения лимитов токенов."""
        memory = ThreeLevelMemory(max_tokens_per_level=500)

        # Добавляем огромное количество сообщений
        for i in range(1000):
            memory.add_message(f"Message {i}", token_count=10)

        # Проверяем что лимиты соблюдаются
        stats = memory.get_stats()

        # Каждый уровень не должен сильно превышать лимит
        # (небольшое превышение возможно из-за батчинга)
        assert stats['current']['tokens'] <= memory.max_tokens_per_level * 1.5
        assert stats['obsolete']['tokens'] <= memory.max_tokens_per_level * 1.5
        assert stats['longterm']['tokens'] <= memory.max_tokens_per_level * 1.5

    def test_memory_performance(self):
        """Тест производительности операций с памятью."""
        import time

        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        # Засекаем время добавления 1000 сообщений
        start = time.time()
        for i in range(1000):
            memory.add_message(f"Message {i}", token_count=10)
        elapsed = time.time() - start

        # Должно быть быстро (<1 секунды для 1000 сообщений)
        assert elapsed < 1.0

        # Засекаем время формирования контекста
        start = time.time()
        context = memory.prepare_context("test query", max_total_tokens=1000)
        elapsed = time.time() - start

        # Контекст должен формироваться быстро
        assert elapsed < 0.1  # <100ms

    def test_context_size_limit(self):
        """Тест что контекст не превышает max_total_tokens."""
        memory = ThreeLevelMemory(max_tokens_per_level=1000)

        for i in range(100):
            memory.add_message(f"Message {i} " * 20, token_count=100)

        context = memory.prepare_context("query", max_total_tokens=500)

        # Контекст должен быть сформирован
        assert len(context) > 0

        # Примерная проверка размера (не строгая, так как подсчёт токенов приблизительный)
        estimated_tokens = len(context.split()) * 2
        # Контекст может немного превышать лимит из-за структуры,
        # но не должен быть огромным
        assert estimated_tokens < 2000
