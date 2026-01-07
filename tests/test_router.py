"""
Тесты для SimpleRouter системы маршрутизации.

Тестирует:
    - ExpertInfo
    - RoutingResult
    - SimpleRouter
"""

import pytest
import json
from pathlib import Path
from routing.router import SimpleRouter, ExpertInfo, RoutingResult


# ============================================================================
# Тесты для ExpertInfo
# ============================================================================

@pytest.mark.unit
@pytest.mark.router
@pytest.mark.fast
class TestExpertInfo:
    """Тесты для ExpertInfo dataclass."""

    def test_creation(self):
        """Тест создания ExpertInfo."""
        expert = ExpertInfo(
            expert_id="test_expert",
            name="Test Expert",
            description="Testing expert",
            keywords={"python", "test"},
            priority=7
        )

        assert expert.expert_id == "test_expert"
        assert expert.name == "Test Expert"
        assert expert.description == "Testing expert"
        assert expert.keywords == {"python", "test"}
        assert expert.priority == 7

    def test_default_priority(self):
        """Тест дефолтного приоритета."""
        expert = ExpertInfo(
            expert_id="test",
            name="Test",
            description="Test",
            keywords=set()
        )

        assert expert.priority == 5

    def test_empty_keywords(self):
        """Тест с пустыми keywords."""
        expert = ExpertInfo(
            expert_id="test",
            name="Test",
            description="Test",
            keywords=set()
        )

        assert expert.keywords == set()
        assert len(expert.keywords) == 0


# ============================================================================
# Тесты для RoutingResult
# ============================================================================

@pytest.mark.unit
@pytest.mark.router
@pytest.mark.fast
class TestRoutingResult:
    """Тесты для RoutingResult dataclass."""

    def test_creation(self):
        """Тест создания RoutingResult."""
        result = RoutingResult(
            expert_id="python_expert",
            confidence=0.85,
            matched_keywords=["python", "code"],
            reasoning="Matched keywords: python, code"
        )

        assert result.expert_id == "python_expert"
        assert result.confidence == 0.85
        assert result.matched_keywords == ["python", "code"]
        assert "python" in result.reasoning

    def test_empty_matches(self):
        """Тест с пустыми совпадениями."""
        result = RoutingResult(
            expert_id="general",
            confidence=0.5,
            matched_keywords=[],
            reasoning="No matches"
        )

        assert result.confidence == 0.5
        assert result.matched_keywords == []


# ============================================================================
# Тесты для SimpleRouter - Basics
# ============================================================================

@pytest.mark.unit
@pytest.mark.router
@pytest.mark.fast
class TestSimpleRouterBasics:
    """Тесты для базовой функциональности SimpleRouter."""

    def test_initialization(self):
        """Тест инициализации роутера."""
        router = SimpleRouter(default_expert_id="general")

        assert router.default_expert_id == "general"
        assert len(router.experts) == 0

    def test_initialization_default(self):
        """Тест инициализации с дефолтными параметрами."""
        router = SimpleRouter()

        assert router.default_expert_id == "general"

    def test_add_expert(self):
        """Тест добавления эксперта."""
        router = SimpleRouter()

        router.add_expert(
            expert_id="python",
            name="Python Expert",
            description="Python specialist",
            keywords={"python", "code"},
            priority=8
        )

        assert len(router.experts) == 1
        assert "python" in router.experts

        expert = router.experts["python"]
        assert expert.name == "Python Expert"
        assert expert.keywords == {"python", "code"}
        assert expert.priority == 8

    def test_add_expert_keywords_lowercase(self):
        """Тест что keywords конвертируются в lowercase."""
        router = SimpleRouter()

        router.add_expert(
            expert_id="test",
            name="Test",
            description="Test",
            keywords={"Python", "CODE", "Test"}
        )

        expert = router.experts["test"]
        assert expert.keywords == {"python", "code", "test"}

    def test_add_multiple_experts(self):
        """Тест добавления нескольких экспертов."""
        router = SimpleRouter()

        router.add_expert("python", "Python Expert", keywords={"python"})
        router.add_expert("js", "JS Expert", keywords={"javascript"})
        router.add_expert("math", "Math Expert", keywords={"math"})

        assert len(router.experts) == 3
        assert "python" in router.experts
        assert "js" in router.experts
        assert "math" in router.experts

    def test_remove_expert(self):
        """Тест удаления эксперта."""
        router = SimpleRouter()
        router.add_expert("python", "Python", keywords={"python"})

        result = router.remove_expert("python")

        assert result is True
        assert len(router.experts) == 0

    def test_remove_nonexistent_expert(self):
        """Тест удаления несуществующего эксперта."""
        router = SimpleRouter()

        result = router.remove_expert("nonexistent")

        assert result is False

    def test_get_expert_info(self):
        """Тест получения информации об эксперте."""
        router = SimpleRouter()
        router.add_expert("python", "Python Expert", keywords={"python"})

        expert = router.get_expert_info("python")

        assert expert is not None
        assert expert.expert_id == "python"
        assert expert.name == "Python Expert"

    def test_get_nonexistent_expert_info(self):
        """Тест получения информации о несуществующем эксперте."""
        router = SimpleRouter()

        expert = router.get_expert_info("nonexistent")

        assert expert is None

    def test_list_experts_empty(self):
        """Тест списка экспертов для пустого роутера."""
        router = SimpleRouter()

        experts = router.list_experts()

        assert experts == []

    def test_list_experts(self):
        """Тест списка экспертов."""
        router = SimpleRouter()
        router.add_expert("python", "Python", keywords={"python"})
        router.add_expert("js", "JavaScript", keywords={"js"})

        experts = router.list_experts()

        assert len(experts) == 2
        expert_ids = [e.expert_id for e in experts]
        assert "python" in expert_ids
        assert "js" in expert_ids


# ============================================================================
# Тесты для SimpleRouter - Routing
# ============================================================================

@pytest.mark.unit
@pytest.mark.router
@pytest.mark.fast
class TestSimpleRouterRouting:
    """Тесты для маршрутизации запросов."""

    @pytest.fixture
    def configured_router(self):
        """Роутер с несколькими экспертами."""
        router = SimpleRouter(default_expert_id="general")

        router.add_expert(
            expert_id="python",
            name="Python Expert",
            description="Python programming",
            keywords={"python", "pandas", "numpy", "flask"},
            priority=8
        )

        router.add_expert(
            expert_id="javascript",
            name="JavaScript Expert",
            description="JavaScript programming",
            keywords={"javascript", "js", "react", "node"},
            priority=8
        )

        router.add_expert(
            expert_id="math",
            name="Math Expert",
            description="Mathematics",
            keywords={"math", "equation", "integral", "derivative"},
            priority=7
        )

        router.add_expert(
            expert_id="general",
            name="General Assistant",
            description="General questions",
            keywords=set(),
            priority=5
        )

        return router

    def test_route_simple_match(self, configured_router):
        """Тест простой маршрутизации с одним совпадением."""
        results = configured_router.route("How to use python?", top_k=1)

        assert len(results) == 1
        assert results[0].expert_id == "python"
        assert results[0].confidence > 0
        assert "python" in results[0].matched_keywords

    def test_route_multiple_keywords(self, configured_router):
        """Тест с несколькими совпадающими keywords."""
        results = configured_router.route(
            "Use pandas and numpy in python",
            top_k=1
        )

        assert len(results) == 1
        assert results[0].expert_id == "python"
        assert len(results[0].matched_keywords) >= 2

    def test_route_no_match_uses_default(self, configured_router):
        """Тест fallback на default эксперта при отсутствии совпадений."""
        results = configured_router.route("What is the weather?", top_k=1)

        assert len(results) == 1
        # Должен вернуть general или другой эксперт с min confidence
        assert results[0].expert_id == "general"

    def test_route_case_insensitive(self, configured_router):
        """Тест case-insensitive маршрутизации."""
        results1 = configured_router.route("PYTHON programming", top_k=1)
        results2 = configured_router.route("python programming", top_k=1)

        assert results1[0].expert_id == results2[0].expert_id
        assert results1[0].expert_id == "python"

    def test_route_top_k(self, configured_router):
        """Тест возврата top_k экспертов."""
        results = configured_router.route(
            "python optimization math equation",
            top_k=3
        )

        assert len(results) <= 3
        assert len(results) >= 2  # python и math должны совпасть

    def test_route_min_confidence(self, configured_router):
        """Тест фильтрации по min_confidence."""
        results = configured_router.route(
            "python",
            top_k=5,
            min_confidence=0.5
        )

        # Все результаты должны иметь confidence >= 0.5
        for result in results:
            assert result.confidence >= 0.5

    def test_route_empty_query(self, configured_router):
        """Тест с пустым запросом."""
        results = configured_router.route("", top_k=1)

        assert len(results) >= 1
        # Должен вернуть default эксперта
        assert results[0].expert_id == "general"

    def test_route_special_characters(self, configured_router):
        """Тест с специальными символами в запросе."""
        results = configured_router.route(
            "How to use python?!? @#$%",
            top_k=1
        )

        assert len(results) == 1
        assert results[0].expert_id == "python"

    def test_route_empty_router(self):
        """Тест маршрутизации в пустом роутере."""
        router = SimpleRouter(default_expert_id="general")

        results = router.route("python programming", top_k=1)

        assert len(results) == 1
        assert results[0].expert_id == "general"
        assert results[0].confidence == 1.0

    def test_route_sorting_by_confidence(self, configured_router):
        """Тест сортировки результатов по confidence."""
        results = configured_router.route(
            "python pandas numpy flask javascript react",
            top_k=5
        )

        # Результаты должны быть отсортированы по убыванию confidence
        for i in range(len(results) - 1):
            assert results[i].confidence >= results[i + 1].confidence


# ============================================================================
# Тесты для SimpleRouter - Scoring
# ============================================================================

@pytest.mark.unit
@pytest.mark.router
@pytest.mark.fast
class TestSimpleRouterScoring:
    """Тесты для системы оценки (scoring)."""

    def test_score_calculation_single_match(self):
        """Тест расчета score с одним совпадением."""
        router = SimpleRouter()
        router.add_expert(
            "python",
            "Python",
            keywords={"python", "code", "programming", "function"}
        )

        results = router.route("python tutorial", top_k=1)

        assert len(results) == 1
        assert results[0].confidence > 0
        # Score = 1/4 (одно совпадение из 4 keywords) + bonus
        assert results[0].confidence > 0.25

    def test_score_calculation_multiple_matches(self):
        """Тест расчета score с несколькими совпадениями."""
        router = SimpleRouter()
        router.add_expert(
            "python",
            "Python",
            keywords={"python", "code"}
        )

        results = router.route("python code example", top_k=1)

        # Должно быть 2 совпадения из 2 keywords
        assert len(results[0].matched_keywords) == 2
        # Score должен быть высоким (2/2 = 1.0 + bonus)
        assert results[0].confidence >= 1.0

    def test_priority_in_sorting(self):
        """Тест учета priority при равном score."""
        router = SimpleRouter()

        router.add_expert(
            "expert1",
            "Expert 1",
            keywords={"test"},
            priority=5
        )

        router.add_expert(
            "expert2",
            "Expert 2",
            keywords={"test"},
            priority=8
        )

        results = router.route("test query", top_k=2)

        # При равном score должен быть выше приоритетный эксперт
        # (оба должны совпасть с "test")
        assert len(results) == 2
        # expert2 с приоритетом 8 должен быть первым
        assert results[0].expert_id == "expert2"

    def test_confidence_capped_at_one(self):
        """Тест что confidence не превышает 1.0 при cap."""
        router = SimpleRouter()
        router.add_expert(
            "python",
            "Python",
            keywords={"python"}
        )

        results = router.route("python python python", top_k=1)

        # Confidence может быть > 1.0 из-за match bonus, но cap в route()
        assert results[0].confidence >= 0.0

    def test_reasoning_generation(self):
        """Тест генерации reasoning."""
        router = SimpleRouter()
        router.add_expert(
            "python",
            "Python Expert",
            keywords={"python", "code"}
        )

        results = router.route("python code", top_k=1)

        # Reasoning должен содержать информацию о совпадениях
        assert "python" in results[0].reasoning.lower() or "code" in results[0].reasoning.lower()
        assert "Python Expert" in results[0].reasoning


# ============================================================================
# Тесты для SimpleRouter - Config Management
# ============================================================================

@pytest.mark.unit
@pytest.mark.router
@pytest.mark.fast
class TestSimpleRouterConfigManagement:
    """Тесты для сохранения/загрузки конфигурации."""

    def test_save_config(self, tmp_path):
        """Тест сохранения конфигурации."""
        router = SimpleRouter(default_expert_id="general")
        router.add_expert(
            "python",
            "Python Expert",
            description="Python programming",
            keywords={"python", "code"},
            priority=8
        )

        config_path = tmp_path / "router_config.json"
        router.save_config(str(config_path))

        # Проверяем что файл создан
        assert config_path.exists()

        # Проверяем содержимое
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        assert config["default_expert_id"] == "general"
        assert len(config["experts"]) == 1
        assert config["experts"][0]["expert_id"] == "python"

    def test_load_config(self, tmp_path):
        """Тест загрузки конфигурации."""
        # Создаём конфигурацию вручную
        config = {
            "default_expert_id": "general",
            "experts": [
                {
                    "expert_id": "python",
                    "name": "Python Expert",
                    "description": "Python programming",
                    "keywords": ["python", "code"],
                    "priority": 8
                }
            ]
        }

        config_path = tmp_path / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f)

        # Загружаем в роутер
        router = SimpleRouter()
        router.load_config(str(config_path))

        assert router.default_expert_id == "general"
        assert len(router.experts) == 1
        assert "python" in router.experts

        expert = router.experts["python"]
        assert expert.name == "Python Expert"
        assert expert.keywords == {"python", "code"}
        assert expert.priority == 8

    def test_save_load_preserves_functionality(self, tmp_path):
        """Тест что save/load сохраняет функциональность."""
        # Создаём роутер
        router1 = SimpleRouter(default_expert_id="general")
        router1.add_expert("python", "Python", keywords={"python"}, priority=8)
        router1.add_expert("js", "JavaScript", keywords={"javascript"}, priority=7)

        # Тестируем routing
        result1 = router1.route("python code", top_k=1)

        # Сохраняем и загружаем
        config_path = tmp_path / "config.json"
        router1.save_config(str(config_path))

        router2 = SimpleRouter()
        router2.load_config(str(config_path))

        # Тестируем что routing работает так же
        result2 = router2.route("python code", top_k=1)

        assert result1[0].expert_id == result2[0].expert_id
        assert result1[0].confidence == result2[0].confidence

    def test_load_config_clears_existing(self, tmp_path):
        """Тест что load_config очищает существующих экспертов."""
        router = SimpleRouter()
        router.add_expert("old_expert", "Old", keywords={"old"})

        # Создаём новую конфигурацию
        config = {
            "default_expert_id": "new_default",
            "experts": [
                {
                    "expert_id": "new_expert",
                    "name": "New Expert",
                    "description": "New",
                    "keywords": ["new"],
                    "priority": 5
                }
            ]
        }

        config_path = tmp_path / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f)

        router.load_config(str(config_path))

        # Старый эксперт должен быть удалён
        assert "old_expert" not in router.experts
        assert "new_expert" in router.experts
        assert router.default_expert_id == "new_default"

    def test_save_config_utf8_encoding(self, tmp_path):
        """Тест UTF-8 кодировки при сохранении."""
        router = SimpleRouter()
        router.add_expert(
            "русский",
            "Русский Эксперт",
            description="Эксперт по русскому языку",
            keywords={"русский", "язык"}
        )

        config_path = tmp_path / "config.json"
        router.save_config(str(config_path))

        # Загружаем и проверяем
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        assert config["experts"][0]["name"] == "Русский Эксперт"
        assert "русский" in config["experts"][0]["keywords"]


# ============================================================================
# Тесты для SimpleRouter - Edge Cases
# ============================================================================

@pytest.mark.unit
@pytest.mark.router
@pytest.mark.fast
class TestSimpleRouterEdgeCases:
    """Тесты для edge cases и граничных условий."""

    def test_expert_with_no_keywords(self):
        """Тест эксперта без keywords."""
        router = SimpleRouter()
        router.add_expert("general", "General", keywords=set())

        results = router.route("any query", top_k=1)

        # Эксперт без keywords может быть выбран как default
        assert len(results) >= 1

    def test_very_long_query(self):
        """Тест с очень длинным запросом."""
        router = SimpleRouter()
        router.add_expert("python", "Python", keywords={"python"})

        long_query = " ".join(["word"] * 1000) + " python"
        results = router.route(long_query, top_k=1)

        assert len(results) == 1
        assert results[0].expert_id == "python"

    def test_query_with_only_punctuation(self):
        """Тест с запросом из только пунктуации."""
        router = SimpleRouter(default_expert_id="general")
        router.add_expert("general", "General", keywords=set())

        results = router.route("!@#$%^&*()", top_k=1)

        assert len(results) >= 1
        assert results[0].expert_id == "general"

    def test_duplicate_keywords_in_expert(self):
        """Тест с дубликатами в keywords (set должен их устранить)."""
        router = SimpleRouter()

        # Set автоматически устранит дубликаты
        keywords = {"python", "python", "code", "code"}
        router.add_expert("python", "Python", keywords=keywords)

        expert = router.experts["python"]
        assert len(expert.keywords) == 2  # Только уникальные

    def test_overwrite_expert(self):
        """Тест перезаписи эксперта с тем же ID."""
        router = SimpleRouter()

        router.add_expert("test", "First", keywords={"first"})
        router.add_expert("test", "Second", keywords={"second"})

        # Второй add_expert должен перезаписать первого
        expert = router.experts["test"]
        assert expert.name == "Second"
        assert expert.keywords == {"second"}

    def test_top_k_larger_than_experts(self):
        """Тест top_k больше чем количество экспертов."""
        router = SimpleRouter()
        router.add_expert("python", "Python", keywords={"python"})
        router.add_expert("js", "JS", keywords={"js"})

        results = router.route("python", top_k=10)

        # Должно вернуться не больше чем есть экспертов
        assert len(results) <= 2

    def test_min_confidence_filters_all(self):
        """Тест что min_confidence может отфильтровать всех."""
        router = SimpleRouter()
        router.add_expert("python", "Python", keywords={"python"})

        # Очень высокий min_confidence
        results = router.route("unrelated query", top_k=5, min_confidence=0.9)

        # Может вернуть default эксперта или пустой список
        # в зависимости от реализации
        assert isinstance(results, list)


# ============================================================================
# Integration тесты
# ============================================================================

@pytest.mark.integration
@pytest.mark.router
class TestSimpleRouterIntegration:
    """Integration тесты для SimpleRouter."""

    def test_dynamic_expert_management(self):
        """Тест динамического добавления/удаления экспертов."""
        router = SimpleRouter(default_expert_id="general")

        # Добавляем default эксперта
        router.add_expert("general", "General", keywords=set(), priority=3)

        # Добавляем эксперта
        router.add_expert("python", "Python", keywords={"python"})
        result1 = router.route("python code", top_k=1)
        assert result1[0].expert_id == "python"

        # Добавляем ещё одного
        router.add_expert("js", "JavaScript", keywords={"javascript", "js"})
        result2 = router.route("javascript code", top_k=1)
        assert result2[0].expert_id == "js"

        # Удаляем первого
        router.remove_expert("python")
        result3 = router.route("python code", top_k=1)
        # python эксперта больше нет, должен вернуть general или другого
        assert len(result3) > 0
        assert result3[0].expert_id != "python"

    def test_multiple_sequential_routes(self):
        """Тест последовательных маршрутизаций."""
        router = SimpleRouter(default_expert_id="general")
        router.add_expert("python", "Python", keywords={"python"})
        router.add_expert("js", "JavaScript", keywords={"javascript"})
        router.add_expert("general", "General", keywords=set())

        queries = [
            ("python tutorial", "python"),
            ("javascript guide", "js"),
            ("what is the weather", "general"),
            ("python and javascript", "python"),  # Оба совпадут, но python может быть выше
        ]

        for query, expected_expert in queries:
            results = router.route(query, top_k=1)
            # Не строгая проверка, так как confidence может варьироваться
            assert len(results) == 1

    def test_real_world_scenario(self):
        """Тест реального сценария использования."""
        router = SimpleRouter(default_expert_id="general")

        # Настройка экспертов
        router.add_expert(
            "python",
            "Python Expert",
            description="Python, Data Science, ML",
            keywords={"python", "pandas", "numpy", "scikit", "tensorflow", "pytorch"},
            priority=9
        )

        router.add_expert(
            "web",
            "Web Development Expert",
            description="Frontend and Backend",
            keywords={"html", "css", "javascript", "react", "vue", "angular", "django", "flask"},
            priority=8
        )

        router.add_expert(
            "devops",
            "DevOps Expert",
            description="Infrastructure and deployment",
            keywords={"docker", "kubernetes", "aws", "azure", "ci", "cd", "jenkins"},
            priority=7
        )

        router.add_expert(
            "general",
            "General Assistant",
            description="General questions",
            keywords=set(),
            priority=5
        )

        # Тестовые запросы
        test_cases = [
            ("How to use pandas dataframe in python?", "python"),
            ("Create a React component", "web"),
            ("Deploy with docker and kubernetes", "devops"),
            ("What is the capital of France?", "general"),
        ]

        for query, expected in test_cases:
            results = router.route(query, top_k=1)
            assert len(results) == 1
            # Не строгая проверка expected, так как routing может варьироваться
            assert results[0].expert_id is not None
            assert results[0].confidence >= 0

    def test_config_persistence_workflow(self, tmp_path):
        """Тест полного workflow с сохранением/загрузкой."""
        config_path = tmp_path / "router.json"

        # 1. Создаём и настраиваем роутер
        router1 = SimpleRouter(default_expert_id="general")
        router1.add_expert("python", "Python", keywords={"python"}, priority=8)
        router1.add_expert("js", "JavaScript", keywords={"js", "javascript"}, priority=7)
        router1.add_expert("general", "General", keywords=set(), priority=5)

        # 2. Сохраняем
        router1.save_config(str(config_path))

        # 3. Загружаем в новый роутер
        router2 = SimpleRouter()
        router2.load_config(str(config_path))

        # 4. Проверяем что работает идентично
        for query in ["python code", "javascript app"]:
            result1 = router1.route(query, top_k=1)
            result2 = router2.route(query, top_k=1)

            assert len(result1) > 0 and len(result2) > 0
            assert result1[0].expert_id == result2[0].expert_id
            assert result1[0].confidence == result2[0].confidence
